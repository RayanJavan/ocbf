"""Gaussian message-passing factor banks.

Banks share a cavity-to-site interface but implement different local likelihoods:
Gaussian evidence, pairwise correlation, transformed observations, interval censoring,
precedence and conditional-Gaussian mean coupling. Non-Gaussian sites use tilted moments;
local moment matching does not preserve a general multimodal joint distribution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ocbf.model.copula.marginals import MarginalTransform, truncated_normal_moments
from ocbf.model.gaussian import (
    MIN_PRECISION,
    POTENTIAL,
    PRECISION,
    gauss_hermite,
    site_from_moments,
    to_moments,
    to_natural,
)

DEFAULT_QUADRATURE = 64
"""Gauss-Hermite nodes used by the quadrature banks.

Quadrature error and [`EPConfig.tol`][ocbf.inference.gabp_ep.EPConfig.tol] must be assessed
together. Heavy-tailed observation likelihoods can require more nodes than smooth Gaussian
sites. Compare tilted moments with adaptive quadrature for the intended cavity and tail
regime; the default node count is not a universal integration-error guarantee.
"""

LogLikelihood = Callable[[np.ndarray], np.ndarray]
"""Observed-unit log-likelihood callable. The bank supplies values through its marginal transform and integrates the returned log density against a cavity. Callers own the report's probability semantics."""


def _quadrature_moments(
    cavity: np.ndarray, log_weight: Callable[[np.ndarray], np.ndarray], nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tilted mean, variance and validity from a cavity and a log-potential.

    The shared core of every quadrature bank: place Gauss-Hermite nodes under the cavity,
    reweight them by the factor, and read off the first two moments. Returns a validity mask
    so callers can leave a site untouched where the tilted distribution underflowed -- which
    happens when a claim lands far out in the cavity's tail, and where inventing moments
    would be worse than adding no evidence at all this sweep.
    """
    mean, var = to_moments(cavity)
    grid = mean[:, None] + np.sqrt(var)[:, None] * nodes[None, :]

    log_w = np.log(weights)[None, :] + log_weight(grid)
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    total = w.sum(axis=1)

    valid = np.isfinite(total) & (total > 1e-300)
    safe = np.where(valid, total, 1.0)[:, None]
    p = w / safe

    tilted_mean = (p * grid).sum(axis=1)
    tilted_var = (p * np.square(grid)).sum(axis=1) - np.square(tilted_mean)
    valid &= np.isfinite(tilted_mean) & np.isfinite(tilted_var) & (tilted_var > 0.0)
    return tilted_mean, tilted_var, valid


def _project(
    cavity: np.ndarray, tilted_mean: np.ndarray, tilted_var: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Moment-match where the tilted distribution is usable, contribute nothing where not."""
    site = site_from_moments(
        np.where(valid, tilted_mean, 0.0), np.where(valid, tilted_var, 1.0), cavity
    )
    return np.where(valid[:, None], site, 0.0)


class GaussianEvidenceBank:
    """Exactly Gaussian evidence on single latent coordinates.

    The continuous analogue of [`UnaryBank`][ocbf.model.banks.UnaryBank], and it carries
    attribution for the same reason: its message does not depend on anything incoming, so
    each edge is one named, additive contribution to its variable's posterior.

    Used for channels whose likelihood is already Gaussian in the latent coordinate, and for
    any caller supplying a prior directly in natural parameters.
    """

    __slots__ = ("name", "_vars", "_site", "labels")

    def __init__(
        self,
        var_ids: Sequence[int] | np.ndarray,
        precision: Sequence[float] | np.ndarray,
        potential: Sequence[float] | np.ndarray,
        *,
        name: str = "gaussian_evidence",
        labels: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        self._vars = np.asarray(var_ids, dtype=np.int64)
        lam = np.asarray(precision, dtype=np.float64)
        eta = np.asarray(potential, dtype=np.float64)
        if not (len(self._vars) == len(lam) == len(eta)):
            raise ValueError("var_ids, precision and potential must have the same length")
        if np.any(lam < 0.0):
            raise ValueError(f"bank {name!r} was given a negative precision")
        self._site = np.stack([lam, eta], axis=-1)
        self.labels = tuple(labels) if labels is not None else ()

    @classmethod
    def from_moments(
        cls,
        var_ids: Sequence[int] | np.ndarray,
        mean: Sequence[float] | np.ndarray,
        variance: Sequence[float] | np.ndarray,
        *,
        name: str = "gaussian_evidence",
        labels: Sequence[str] | None = None,
    ) -> GaussianEvidenceBank:
        """Build from ``(mean, variance)`` instead of natural parameters."""
        natural = to_natural(np.asarray(mean, dtype=np.float64), np.asarray(variance, dtype=np.float64))
        return cls(var_ids, natural[..., PRECISION], natural[..., POTENTIAL], name=name, labels=labels)

    def edge_vars(self) -> np.ndarray:
        return self._vars

    def factor_to_var(self, cavity: np.ndarray) -> np.ndarray:
        return self._site

    def n_factors(self) -> int:
        return len(self._vars)


class CorrelationBank:
    """The copula's latent correlations, as exact pairwise Gaussian couplings.

    Carries only what a correlated pair *adds* to two independent standard normals -- see
    [`coupling_precision`][ocbf.model.copula.structure.coupling_precision] -- so the
    standard-normal priors the copula transform guarantees are counted once, by the graph.

    Edge layout is ``[all A-side edges, all B-side edges]``, matching
    [`PairwiseBank`][ocbf.model.banks.PairwiseBank], so both directions are one batched
    expression rather than a loop over factors.
    """

    __slots__ = ("name", "_a", "_b", "_diag", "_off")

    def __init__(
        self,
        a_ids: Sequence[int] | np.ndarray,
        b_ids: Sequence[int] | np.ndarray,
        coupling: np.ndarray,
        *,
        name: str = "copula_correlation",
    ) -> None:
        self.name = name
        self._a = np.asarray(a_ids, dtype=np.int64)
        self._b = np.asarray(b_ids, dtype=np.int64)
        blocks = np.asarray(coupling, dtype=np.float64)
        if blocks.shape != (len(self._a), 2, 2):
            raise ValueError(f"coupling must be {(len(self._a), 2, 2)}, got {blocks.shape}")
        if len(self._a) != len(self._b):
            raise ValueError("a_ids and b_ids must have the same length")
        self._diag = blocks[:, 0, 0], blocks[:, 1, 1]
        self._off = blocks[:, 0, 1]

    def edge_vars(self) -> np.ndarray:
        return np.concatenate([self._a, self._b])

    def factor_to_var(self, cavity: np.ndarray) -> np.ndarray:
        n = len(self._a)
        d_a, d_b = self._diag
        to_a = self._marginalise(cavity[n:], d_a, d_b)
        to_b = self._marginalise(cavity[:n], d_b, d_a)
        return np.concatenate([to_a, to_b], axis=0)

    def _marginalise(self, other: np.ndarray, own_diag: np.ndarray, other_diag: np.ndarray) -> np.ndarray:
        """Integrate the far variable out of the pair potential times its cavity.

        Standard Gaussian belief propagation: the pair contributes ``own_diag`` to this
        variable's precision, less the Schur complement of the far variable, which is the
        term that carries the correlation.
        """
        denom = other_diag + other[:, PRECISION]
        usable = denom > MIN_PRECISION
        safe = np.where(usable, denom, 1.0)
        precision = own_diag - np.square(self._off) / safe
        potential = -self._off * other[:, POTENTIAL] / safe
        site = np.stack([precision, potential], axis=-1)
        return np.where(usable[:, None], site, 0.0)

    def n_factors(self) -> int:
        return len(self._a)


class ObservationBank:
    """Observed-unit likelihood evaluated through a monotone copula transform.

    The likelihood receives values on the observed scale; quadrature integrates its product with the latent Gaussian cavity and moment-matches the resulting site. Quadrature resolution and moment matching are numerical approximations."""

    __slots__ = ("name", "_vars", "_loglik", "_marginal", "_nodes", "_weights", "labels")

    def __init__(
        self,
        var_ids: Sequence[int] | np.ndarray,
        log_likelihood: LogLikelihood,
        marginal: MarginalTransform,
        *,
        name: str = "observation",
        labels: Sequence[str] | None = None,
        n_quadrature: int = DEFAULT_QUADRATURE,
    ) -> None:
        self.name = name
        self._vars = np.asarray(var_ids, dtype=np.int64)
        self._loglik = log_likelihood
        self._marginal = marginal
        self._nodes, self._weights = gauss_hermite(n_quadrature)
        self.labels = tuple(labels) if labels is not None else ()

    def edge_vars(self) -> np.ndarray:
        return self._vars

    def factor_to_var(self, cavity: np.ndarray) -> np.ndarray:
        def log_weight(latent: np.ndarray) -> np.ndarray:
            return self._loglik(self._marginal.to_value(latent))

        mean, var, valid = _quadrature_moments(cavity, log_weight, self._nodes, self._weights)
        return _project(cavity, mean, var, valid)

    def n_factors(self) -> int:
        return len(self._vars)


class IntervalBank:
    """Interval-censored evidence restricting a latent coordinate to ``[lo, hi]``. Tilted moments use the closed-form truncated-normal calculation before the EP site update."""

    __slots__ = ("name", "_vars", "_lo", "_hi", "labels")

    def __init__(
        self,
        var_ids: Sequence[int] | np.ndarray,
        lo: Sequence[float] | np.ndarray,
        hi: Sequence[float] | np.ndarray,
        *,
        name: str = "interval",
        labels: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        self._vars = np.asarray(var_ids, dtype=np.int64)
        self._lo = np.asarray(lo, dtype=np.float64)
        self._hi = np.asarray(hi, dtype=np.float64)
        if not (len(self._vars) == len(self._lo) == len(self._hi)):
            raise ValueError("var_ids, lo and hi must have the same length")
        if np.any(self._lo > self._hi):
            raise ValueError(f"bank {name!r} was given an empty interval")
        self.labels = tuple(labels) if labels is not None else ()

    def edge_vars(self) -> np.ndarray:
        return self._vars

    def factor_to_var(self, cavity: np.ndarray) -> np.ndarray:
        mean, var = to_moments(cavity)
        tilted_mean, tilted_var = truncated_normal_moments(self._lo, self._hi, mean, var)
        valid = np.isfinite(tilted_mean) & (tilted_var > 0.0)
        return _project(cavity, tilted_mean, tilted_var, valid)

    def n_factors(self) -> int:
        return len(self._vars)


class PrecedenceBank:
    """Pairwise temporal-order evidence in latent coordinates.

    The soft form uses a sigmoid preference with supplied scale and weight. The hard form uses truncated moments. Within EP, these local sites do not imply exact inference for a globally truncated joint distribution."""

    __slots__ = ("name", "_before", "_after", "_slack", "_weight", "_hard", "_nodes", "_weights")

    def __init__(
        self,
        before_ids: Sequence[int] | np.ndarray,
        after_ids: Sequence[int] | np.ndarray,
        slack: Sequence[float] | np.ndarray | float,
        weight: Sequence[float] | np.ndarray | float,
        *,
        hard: bool = False,
        name: str = "precedence",
        n_quadrature: int = DEFAULT_QUADRATURE,
    ) -> None:
        self.name = name
        self._before = np.asarray(before_ids, dtype=np.int64)
        self._after = np.asarray(after_ids, dtype=np.int64)
        if len(self._before) != len(self._after):
            raise ValueError("before_ids and after_ids must have the same length")
        self._slack = np.broadcast_to(np.asarray(slack, dtype=np.float64), self._before.shape)
        self._weight = np.broadcast_to(np.asarray(weight, dtype=np.float64), self._before.shape)
        if np.any(self._slack <= 0.0):
            raise ValueError(f"bank {name!r} needs a positive slack")
        if np.any(self._weight < 0.0):
            raise ValueError(f"bank {name!r} needs a non-negative weight")
        self._hard = bool(hard)
        self._nodes, self._weights = gauss_hermite(n_quadrature)

    def edge_vars(self) -> np.ndarray:
        return np.concatenate([self._before, self._after])

    def factor_to_var(self, cavity: np.ndarray) -> np.ndarray:
        n = len(self._before)
        m_before, v_before = to_moments(cavity[:n])
        m_after, v_after = to_moments(cavity[n:])

        difference = to_natural(m_after - m_before, v_before + v_after)
        if self._hard:
            d_mean, d_var = truncated_normal_moments(
                np.zeros(n), np.full(n, np.inf), m_after - m_before, v_before + v_after
            )
            valid = np.isfinite(d_mean) & (d_var > 0.0)
        else:
            d_mean, d_var, valid = _quadrature_moments(
                difference, self._log_potential, self._nodes, self._weights
            )

        site = _project(difference, d_mean, d_var, valid)
        # A site with non-positive precision would make the difference less certain than its
        # cavity. Expectation propagation permits that in general, but here it cannot be
        # turned into a proper message to either endpoint, so the update is skipped -- the
        # standard response, and one that costs only a sweep.
        usable = site[:, PRECISION] > MIN_PRECISION
        safe = np.where(usable, site[:, PRECISION], 1.0)
        shift = site[:, POTENTIAL] / safe
        spread = 1.0 / safe

        to_after = to_natural(m_before + shift, v_before + spread)
        to_before = to_natural(m_after - shift, v_after + spread)
        out = np.concatenate([to_before, to_after], axis=0)
        return np.where(np.concatenate([usable, usable])[:, None], out, 0.0)

    def _log_potential(self, difference: np.ndarray) -> np.ndarray:
        """``weight * log sigmoid(d / slack)``, evaluated stably for large negative ``d``."""
        scaled = difference / self._slack[:, None]
        return -self._weight[:, None] * np.logaddexp(0.0, -scaled)

    def n_factors(self) -> int:
        return len(self._before)


class CGMeanBank:
    """Conditional-Gaussian mean coupling to a finite variable.

    Components share covariance and differ by mean. The message to the continuous side is moment-matched, while optional uncollapsed components expose the local approximation. The discrete message retains its analytic log-partition contribution."""

    __slots__ = ("name", "_vars", "_means", "_log_weights", "_var", "labels")

    def __init__(
        self,
        var_ids: Sequence[int] | np.ndarray,
        state_means: np.ndarray,
        state_probabilities: np.ndarray,
        conditional_var: Sequence[float] | np.ndarray | float,
        *,
        name: str = "cg_mean",
        labels: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        self._vars = np.asarray(var_ids, dtype=np.int64)
        self._means = np.asarray(state_means, dtype=np.float64)
        probs = np.asarray(state_probabilities, dtype=np.float64)
        if self._means.shape != probs.shape:
            raise ValueError("state_means and state_probabilities must have the same shape")
        if self._means.shape[0] != len(self._vars):
            raise ValueError("state arrays must have one row per edge")
        total = probs.sum(axis=1, keepdims=True)
        # A padded state -- present only because the array is rectangular over ragged type
        # supports -- carries zero probability, and its log must be finite for the
        # normalisation that follows. Clamping to a denormal keeps it decisively excluded
        # without introducing a -inf that would poison the row.
        share = np.where(total > 0.0, probs / np.where(total > 0.0, total, 1.0), 0.0)
        self._log_weights = np.log(np.maximum(share, 1e-300))
        self._var = np.broadcast_to(
            np.asarray(conditional_var, dtype=np.float64), self._vars.shape
        )
        if np.any(self._var <= 0.0):
            raise ValueError(f"bank {name!r} needs a positive conditional variance")
        self.labels = tuple(labels) if labels is not None else ()

    def edge_vars(self) -> np.ndarray:
        return self._vars

    def factor_to_var(self, cavity: np.ndarray) -> np.ndarray:
        weights, means, variances = self.mixture(cavity)
        mixture_mean = (weights * means).sum(axis=1)
        mixture_var = (weights * (variances + np.square(means))).sum(axis=1) - np.square(mixture_mean)
        valid = np.isfinite(mixture_mean) & (mixture_var > 0.0)
        return _project(cavity, mixture_mean, mixture_var, valid)

    def mixture(self, cavity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return retained uncollapsed ``(weights, means, variances)`` for a local conditional-Gaussian message. This is available only when the numerical calculation retained those components; it is not a general joint posterior."""
        cavity_precision = np.maximum(cavity[:, PRECISION], MIN_PRECISION)
        cavity_mean = cavity[:, POTENTIAL] / cavity_precision
        cavity_var = 1.0 / cavity_precision

        evidence = self._log_evidence(cavity_mean[:, None], cavity_var[:, None])
        log_w = self._log_weights + evidence
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        weights = w / w.sum(axis=1, keepdims=True)

        precision = 1.0 / self._var[:, None] + cavity_precision[:, None]
        variances = np.broadcast_to(1.0 / precision, self._means.shape)
        means = (self._means / self._var[:, None] + cavity[:, POTENTIAL][:, None]) / precision
        return weights, means, variances

    def discrete_potentials(self, cavity: np.ndarray) -> np.ndarray:
        """``log p(continuous evidence | discrete = d)`` per edge and state.

        Exact and closed form: convolving a state's conditional Gaussian with the cavity
        gives ``N(m_cavity; mu_d, sigma^2 + v_cavity)``. This is the message that lets a
        timestamp inform an event's type -- the direction that makes the coupling more than
        a way of sharpening the continuous side.

        Rows are shifted so their maximum is zero, because only differences across states
        reach the discrete graph and an unshifted log-density carries an arbitrary offset
        that would otherwise ride along into every log-potential.
        """
        cavity_precision = np.maximum(cavity[:, PRECISION], MIN_PRECISION)
        mean = cavity[:, POTENTIAL] / cavity_precision
        evidence = self._log_evidence(mean[:, None], (1.0 / cavity_precision)[:, None])
        return evidence - evidence.max(axis=1, keepdims=True)

    def _log_evidence(self, cavity_mean: np.ndarray, cavity_var: np.ndarray) -> np.ndarray:
        total = self._var[:, None] + cavity_var
        return -0.5 * (np.log(2.0 * np.pi * total) + np.square(cavity_mean - self._means) / total)

    def n_factors(self) -> int:
        return len(self._vars)


def gaussian_loglik(
    value: np.ndarray, scale: np.ndarray, bias: np.ndarray | float = 0.0
) -> LogLikelihood:
    """Return the observed-unit Gaussian log likelihood for ``y = v + bias + noise`` with supplied scales. This is also a numerical comparison channel for robust error distributions."""
    y = np.asarray(value, dtype=np.float64)[:, None]
    s = np.asarray(scale, dtype=np.float64)[:, None]
    b = np.reshape(np.asarray(bias, dtype=np.float64), (-1, 1)) if np.ndim(bias) else float(bias)

    def log_likelihood(values: np.ndarray) -> np.ndarray:
        return -0.5 * np.square((y - b - values) / s)

    return log_likelihood


def student_t_loglik(
    value: np.ndarray, scale: np.ndarray, df: np.ndarray | float, bias: np.ndarray | float = 0.0
) -> LogLikelihood:
    """Return the observed-unit Student-t log likelihood with supplied bias, positive scale and degrees of freedom. Heavy tails reduce the influence of large residuals under this error assumption."""
    y = np.asarray(value, dtype=np.float64)[:, None]
    s = np.asarray(scale, dtype=np.float64)[:, None]
    nu = np.asarray(df, dtype=np.float64)[:, None] if np.ndim(df) else float(df)
    b = np.reshape(np.asarray(bias, dtype=np.float64), (-1, 1)) if np.ndim(bias) else float(bias)
    power = -0.5 * (nu + 1.0)

    def log_likelihood(values: np.ndarray) -> np.ndarray:
        return power * np.log1p(np.square((y - b - values) / s) / nu)

    return log_likelihood


def tempered_loglik(base: LogLikelihood, temperature: np.ndarray | float) -> LogLikelihood:
    """Multiply an explicitly supplied log likelihood by a temperature.

    The caller must establish its probability semantics; an arbitrary upstream posterior or confidence vector is not automatically a likelihood."""
    lam = np.asarray(temperature, dtype=np.float64)
    lam = lam[:, None] if lam.ndim else lam

    def log_likelihood(values: np.ndarray) -> np.ndarray:
        return lam * base(values)

    return log_likelihood
