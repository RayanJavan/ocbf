"""Posterior beliefs over the latent world."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtri

from ocbf.assertions import AssertionRef, Family, VariableRegistry, VarKind

if TYPE_CHECKING:  # a contract type must not depend on the model that fills it
    from ocbf.model.copula import MarginalTransform


class Verdict(str, Enum):
    """Three-valued static-evidence decision.

    ``UNDETERMINED`` records insufficient evidence under the declared binary-channel threshold, even when structural messages produce a strong marginal. It is not a calibrated correctness probability."""

    TRUE = "true"
    FALSE = "false"
    UNDETERMINED = "undetermined"


class BeliefStateBuilder:
    """Accumulates per-variable beliefs, then seals them into a [`BeliefState`][ocbf.belief.state.BeliefState]."""

    def __init__(self, registry: VariableRegistry) -> None:
        if not registry.is_frozen:
            raise ValueError("registry must be frozen before building beliefs")
        self.registry = registry
        n = len(registry)
        k = max(registry.max_cardinality, 1)

        self.probs = np.zeros((n, k), dtype=np.float64)
        self.mean = np.full(n, np.nan, dtype=np.float64)
        self.var = np.full(n, np.nan, dtype=np.float64)
        self.evidence = np.zeros(n, dtype=np.float64)
        self._attribution: dict[int, dict[str, float]] = {}
        self._marginals: dict[int, MarginalTransform] = {}
        self._mixtures: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._prior_only = np.ones(n, dtype=bool)

        # Uninformative defaults, so an untouched variable answers with its prior rather
        # than with a crash or an arbitrary 0.5.
        card = registry.cardinalities
        for i in range(n):
            c = int(card[i])
            if c > 0:
                self.probs[i, :c] = 1.0 / c

    def set_discrete(self, idx: int, probs: np.ndarray, *, prior_only: bool = False) -> None:
        c = int(self.registry.cardinalities[idx])
        if probs.shape[0] != c:
            raise ValueError(f"variable {idx} has cardinality {c}, got {probs.shape[0]} probs")
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"variable {idx} received unnormalisable belief")
        self.probs[idx, :c] = probs / total
        self._prior_only[idx] = prior_only

    def set_continuous(
        self,
        idx: int,
        mean: float,
        var: float,
        *,
        marginal: MarginalTransform | None = None,
        mixture: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        prior_only: bool = False,
    ) -> None:
        """Record latent Gaussian moments and an optional observed-unit marginal transform.

        An optional uncollapsed mixture supports comparison with local moment matching. The caller supplies all values; no dependence is reconstructed from separate marginals."""
        if var <= 0:
            raise ValueError(f"variable {idx} needs positive variance, got {var}")
        self.mean[idx] = mean
        self.var[idx] = var
        if marginal is not None:
            self._marginals[idx] = marginal
        if mixture is not None:
            self._mixtures[idx] = mixture
        self._prior_only[idx] = prior_only

    def set_evidence(self, idx: int, nats: float) -> None:
        """Total Chernoff information available on this assertion (section 7.2)."""
        self.evidence[idx] = nats

    def set_attribution(self, idx: int, contributions: Mapping[str, float]) -> None:
        self._attribution[idx] = dict(contributions)

    def build(
        self,
        *,
        diagnostics: Mapping[str, object] | None = None,
        evidence_target: float = np.log(1.0 / 0.1),
    ) -> BeliefState:
        return BeliefState(
            registry=self.registry,
            probs=self.probs,
            mean=self.mean,
            var=self.var,
            evidence=self.evidence,
            prior_only=self._prior_only,
            attribution=self._attribution,
            diagnostics=dict(diagnostics or {}),
            evidence_target=evidence_target,
            marginals=self._marginals,
            mixtures=self._mixtures,
        )


class BeliefState:
    """Immutable assertion-marginal view for numerical utilities.

    Discrete beliefs use padded arrays. Continuous values retain latent Gaussian moments, optional observed-unit transforms and optional local mixture components. This representation does not imply a coherent joint-history distribution. Message decompositions explain numerical contributions, not source-removal effects."""

    __slots__ = (
        "registry",
        "_probs",
        "_mean",
        "_var",
        "_evidence",
        "_prior_only",
        "_attribution",
        "_marginals",
        "_mixtures",
        "diagnostics",
        "evidence_target",
    )

    def __init__(
        self,
        registry: VariableRegistry,
        probs: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        evidence: np.ndarray,
        prior_only: np.ndarray,
        attribution: Mapping[int, Mapping[str, float]],
        diagnostics: Mapping[str, object],
        evidence_target: float,
        marginals: Mapping[int, MarginalTransform] | None = None,
        mixtures: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.registry = registry
        self._probs = probs
        self._mean = mean
        self._var = var
        self._evidence = evidence
        self._prior_only = prior_only
        self._attribution = {k: dict(v) for k, v in attribution.items()}
        self._marginals = dict(marginals or {})
        self._mixtures = dict(mixtures or {})
        self.diagnostics = dict(diagnostics)
        self.evidence_target = evidence_target

    # -- marginals ------------------------------------------------------------------

    def marginal(self, ref: AssertionRef) -> np.ndarray:
        """Discrete posterior over the variable's domain."""
        idx = self.registry.index(ref)
        c = int(self.registry.cardinalities[idx])
        if c == 0:
            raise TypeError(f"{ref} is continuous; use gaussian()")
        return self._probs[idx, :c].copy()

    def gaussian(self, ref: AssertionRef) -> tuple[float, float]:
        """Return continuous ``(mean, variance)`` in latent copula coordinates. Use the attached transform for observed-unit summaries. The local Gaussian approximation need not preserve multimodality."""
        idx = self.registry.index(ref)
        if int(self.registry.cardinalities[idx]) != 0:
            raise TypeError(f"{ref} is discrete; use marginal()")
        return float(self._mean[idx]), float(self._var[idx])

    def value(self, ref: AssertionRef) -> float:
        """Posterior **median** of a continuous assertion, in observed units.

        The median rather than the mean, because the copula transform is monotone but not
        linear: the median passes through it exactly, while the mean does not. Under an
        affine marginal -- what timestamps use -- the two coincide anyway.

        Raises:
            KeyError: if no marginal transform was attached. A latent coordinate has no
                observed value on its own, and inventing units would be worse than saying so.
        """
        mean, _ = self.gaussian(ref)
        return float(self.transform(ref).to_value(mean))

    def value_interval(self, ref: AssertionRef, mass: float = 0.9) -> tuple[float, float]:
        """Central credible interval of a continuous assertion, in observed units.

        Computed by transforming the latent interval endpoints, which is exact for any
        monotone marginal -- a quantile of the observed value *is* the image of the
        corresponding latent quantile. For a censored marginal the endpoints land on
        declared levels, which is the honest answer for a quantity that only takes those.
        """
        if not 0.0 < mass < 1.0:
            raise ValueError(f"mass must be in (0, 1), got {mass}")
        mean, var = self.gaussian(ref)
        half = ndtri(0.5 * (1.0 + mass)) * np.sqrt(var)
        transform = self.transform(ref)
        return float(transform.to_value(mean - half)), float(transform.to_value(mean + half))

    def mixture(self, ref: AssertionRef) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return retained uncollapsed ``(weights, means, variances)`` for a local conditional-Gaussian message. This is available only when the numerical calculation retained those components; it is not a general joint posterior."""
        idx = self.registry.get(ref)
        return None if idx is None else self._mixtures.get(idx)

    def transform(self, ref: AssertionRef) -> MarginalTransform:
        """The marginal transform attached to a continuous assertion.

        Public because scoring needs to map a *known* value into latent units, which is the
        inverse of what `value` does, and re-deriving the transform outside the belief state
        would risk answering with a different one than the posterior was computed under.

        Raises:
            KeyError: if none is attached, so the caller cannot silently assume identity.
        """
        idx = self.registry.index(ref)
        try:
            return self._marginals[idx]
        except KeyError:
            raise KeyError(
                f"{ref} has no marginal transform attached, so it has no observed value; "
                "use gaussian() for the latent posterior"
            ) from None

    def prob_true(self, ref: AssertionRef, default: float = 0.5) -> float:
        """Return a binary assertion marginal.

        For a reference outside the active registry, return the explicitly supplied ``default``. That fallback is a caller convention, not an inferred probability for an absent candidate."""
        idx = self.registry.get(ref)
        if idx is None:
            return default
        if int(self.registry.cardinalities[idx]) != 2:
            raise TypeError(f"{ref} is not binary")
        return float(self._probs[idx, 1])

    def argmax(self, ref: AssertionRef) -> int:
        """Index of the most probable state in the variable's domain."""
        idx = self.registry.index(ref)
        c = int(self.registry.cardinalities[idx])
        if c == 0:
            raise TypeError(f"{ref} is continuous")
        return int(np.argmax(self._probs[idx, :c]))

    # -- decidability ---------------------------------------------------------------

    def evidence_nats(self, ref: AssertionRef) -> float:
        """Total Chernoff information available on this assertion, in nats."""
        idx = self.registry.get(ref)
        return 0.0 if idx is None else float(self._evidence[idx])

    def is_decidable(self, ref: AssertionRef) -> bool:
        """Whether source evidence clears the `log(1/eps)` target.

        A lower bound for the structured model, which also draws on structural
        evidence the bound does not count.
        """
        return self.evidence_nats(ref) >= self.evidence_target

    def verdict(self, ref: AssertionRef, threshold: float = 0.5) -> Verdict:
        """Three-valued answer, gated on decidability."""
        if not self.is_decidable(ref):
            return Verdict.UNDETERMINED
        return Verdict.TRUE if self.prob_true(ref) >= threshold else Verdict.FALSE

    def is_prior_only(self, ref: AssertionRef) -> bool:
        """Whether this assertion's belief comes from the prior alone.

        True for refs outside the active graph. Check it before treating a number as
        evidence-driven.
        """
        idx = self.registry.get(ref)
        return True if idx is None else bool(self._prior_only[idx])

    def with_evidence(
        self, evidence: Mapping[AssertionRef, float], target_nats: float | None = None
    ) -> BeliefState:
        """Return a copy carrying per-assertion evidence, hence decidability verdicts.

        Kept separate from inference so that decidability can be recomputed under a
        different ``eps`` target, or attached to a baseline that does not compute evidence
        itself, without re-running anything.
        """
        ev = self._evidence.copy()
        for ref, nats in evidence.items():
            idx = self.registry.get(ref)
            if idx is not None:
                ev[idx] = nats
        return BeliefState(
            registry=self.registry,
            probs=self._probs,
            mean=self._mean,
            var=self._var,
            evidence=ev,
            prior_only=self._prior_only,
            attribution=self._attribution,
            diagnostics=self.diagnostics,
            evidence_target=self.evidence_target if target_nats is None else target_nats,
            marginals=self._marginals,
            mixtures=self._mixtures,
        )

    # -- attribution ----------------------------------------------------------------

    def attribution(self, ref: AssertionRef) -> dict[str, float]:
        """Return additive incoming-message contributions to binary posterior log odds.

        These contributions decompose a numerical belief calculation. Removing a report family requires a separately constructed model and inference run."""
        idx = self.registry.get(ref)
        return dict(self._attribution.get(idx, {})) if idx is not None else {}

    def top_contributors(self, ref: AssertionRef, k: int = 5) -> list[tuple[str, float]]:
        """The `k` largest contributions to the posterior log-odds, by absolute size."""
        items = self.attribution(ref).items()
        return sorted(items, key=lambda kv: -abs(kv[1]))[:k]

    # -- bulk views -----------------------------------------------------------------

    def probs_for_family(self, family: Family) -> tuple[np.ndarray, list[AssertionRef]]:
        s = self.registry.family_slice(family)
        refs = [self.registry.ref(i) for i in range(s.start, s.stop)]
        return self._probs[s], refs

    def binary_scores(self, refs: Sequence[AssertionRef]) -> np.ndarray:
        """`P(true)` for each ref, as an array aligned with `refs`."""
        return np.fromiter(
            (self.prob_true(r) for r in refs), dtype=np.float64, count=len(refs)
        )

    def decidable_mask(self, refs: Sequence[AssertionRef]) -> np.ndarray:
        """Boolean array marking which of `refs` clear the evidence target."""
        return np.fromiter(
            (self.is_decidable(r) for r in refs), dtype=bool, count=len(refs)
        )

    def summary(self) -> dict[str, object]:
        """Counts of decidable, undetermined and prior-only variables, plus diagnostics."""
        n = len(self.registry)
        decidable = int((self._evidence >= self.evidence_target).sum())
        return {
            "variables": n,
            "decidable": decidable,
            "undetermined": n - decidable,
            "prior_only": int(self._prior_only.sum()),
            "evidence_target_nats": round(float(self.evidence_target), 4),
            **{f"diag_{k}": v for k, v in self.diagnostics.items()},
        }

    def __repr__(self) -> str:
        return f"BeliefState(vars={len(self.registry)}, diagnostics={len(self.diagnostics)})"
