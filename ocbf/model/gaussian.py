"""Gaussian factor graph container and bank protocol -- the continuous layer's `graph`.

The deliberate counterpart of [`ocbf.model.graph`][ocbf.model.graph]. Everything about the
shape is the same, so the two engines read as one system: a graph owns variables and banks,
a bank owns edges, and the engine hands a bank the incoming messages for its own edges and
gets the outgoing ones back. Only the *contents* of a message differ.

**Messages are natural parameters, not log-potential rows.** A Gaussian message carries
``(precision, potential) = (1/v, m/v)``, so an edge message is two numbers rather than
``max_card``. Natural parameters rather than ``(mean, variance)`` for the same reason the
discrete engine works in log space: the combination rule is addition. A variable's belief is
the sum of its incoming messages and its prior, and a cavity is that sum minus one term --
both exact, both one subtraction, and neither needing a division or a special case.

**The prior is the standard normal, by construction.** The copula transform of design doc
section 4.1 defines the latent coordinate as ``z = Phi^-1(F(v))``, whose marginal *is*
``N(0, 1)``. So the prior is not a modelling choice here; it is what makes the coordinate
mean what it says. Everything informative enters as a bank.

That standardisation has a practical consequence the engine leans on: a residual, a
tolerance or a variance expressed in latent units means the same thing for a timestamp in
hours and a price in euros. One convergence tolerance is meaningful across the whole block.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ocbf.assertions import AssertionRef, VariableRegistry

PRECISION = 0
"""Column of a natural-parameter array holding the precision ``1 / variance``."""

POTENTIAL = 1
"""Column of a natural-parameter array holding the potential ``mean / variance``."""

MIN_PRECISION = 1.0e-9
"""Floor on any belief or cavity precision.

A cavity is a subtraction, and expectation propagation sites are free to carry *negative*
precision -- a site that says "less certain than the cavity" is legitimate and common. The
sum can therefore approach or cross zero, which is an improper Gaussian with infinite
variance. Flooring keeps every belief proper.

The value matters in the same way [`NEG_INF`][ocbf.model.graph.NEG_INF] does in the discrete
engine, and for the mirror-image reason: it must be small enough that a genuinely
uninformative variable stays uninformative (``1e-9`` is a standard deviation of ~31 623
latent units, which is ignorance by any measure), and large enough that dividing by it does
not overflow. Sites that would push a cavity below it are refused rather than clipped, in
[`ocbf.inference.gabp_ep`][ocbf.inference.gabp_ep] -- clipping there would silently invent
evidence.
"""

MAX_PRECISION = 1.0e12
"""Ceiling on a belief precision, i.e. a floor of ``1e-12`` on its variance.

Reached only when several near-deterministic sites agree, and capping keeps the moment
conversion finite instead of producing a zero-variance belief that no later message can move.
"""


@runtime_checkable
class GaussianBank(Protocol):
    """All groundings of one continuous factor template, stored columnar.

    A bank owns ``n_edges`` edges. Edge ``e`` attaches to the block position
    ``edge_vars()[e]``. The engine hands over the cavity -- the belief with this edge's own
    message removed -- and receives the site approximation back.

    Sites are expectation-propagation approximations for the non-Gaussian banks and exact
    messages for the Gaussian ones, and the protocol deliberately does not distinguish them:
    design doc section 4.3's whole point is that a copula warp, an interval censoring, a
    truncation and a discrete mixture are one operation seen four ways.
    """

    name: str

    def edge_vars(self) -> np.ndarray:
        """``int64[n_edges]`` -- the block position each edge attaches to."""
        ...

    def factor_to_var(self, cavity: np.ndarray) -> np.ndarray:
        """Site messages given the cavity.

        ``cavity`` and the return value are both ``float64[n_edges, 2]`` in natural
        parameters, with columns [`PRECISION`][ocbf.model.gaussian.PRECISION] and
        [`POTENTIAL`][ocbf.model.gaussian.POTENTIAL].
        """
        ...

    def n_factors(self) -> int: ...


class GaussianGraph:
    """The continuous block: a subset of the registry's variables, plus banks over them.

    Only continuous variables participate, and they are re-indexed into a dense **block
    position** ``0 .. n-1``. The discrete graph can index straight into the registry because
    nearly every variable is discrete; here the continuous variables are a minority, and
    dense positions keep every array in the engine proportional to the block rather than to
    the whole world.
    """

    __slots__ = ("registry", "var_ids", "prior", "banks", "_position", "_edge_var", "_offsets")

    def __init__(
        self,
        registry: VariableRegistry,
        var_ids: Sequence[int] | np.ndarray,
        banks: Sequence[GaussianBank],
        prior: np.ndarray | None = None,
    ) -> None:
        if not registry.is_frozen:
            raise ValueError("registry must be frozen")
        ids = np.asarray(var_ids, dtype=np.int64)
        card = registry.cardinalities
        discrete = ids[card[ids] != 0] if ids.size else ids
        if discrete.size:
            raise ValueError(
                f"the Gaussian block accepts continuous variables only; "
                f"{registry.ref(int(discrete[0]))} is discrete"
            )

        self.registry = registry
        self.var_ids = ids
        self.banks = list(banks)
        self._position = {int(v): i for i, v in enumerate(ids)}

        if prior is None:
            # N(0, 1): the copula transform's defining property, not a default to tune.
            prior = np.tile(np.array([1.0, 0.0]), (len(ids), 1))
        if prior.shape != (len(ids), 2):
            raise ValueError(f"prior must be {(len(ids), 2)}, got {prior.shape}")
        self.prior = np.asarray(prior, dtype=np.float64)

        # One flat edge space across all banks, so the engine does a single scatter-add --
        # exactly as the discrete graph does.
        offsets: list[int] = []
        pieces: list[np.ndarray] = []
        total = 0
        for bank in self.banks:
            ev = np.asarray(bank.edge_vars(), dtype=np.int64)
            if ev.size and (ev.min() < 0 or ev.max() >= len(ids)):
                raise ValueError(f"bank {bank.name!r} names a block position outside the block")
            offsets.append(total)
            pieces.append(ev)
            total += len(ev)
        self._offsets = offsets
        self._edge_var = (
            np.concatenate(pieces).astype(np.int64) if pieces else np.zeros(0, dtype=np.int64)
        )

    @property
    def n_vars(self) -> int:
        return len(self.var_ids)

    @property
    def n_edges(self) -> int:
        return len(self._edge_var)

    @property
    def edge_var(self) -> np.ndarray:
        return self._edge_var

    def position(self, ref_or_id: AssertionRef | int) -> int | None:
        """Block position of a registry id or ref; ``None`` if it is not in the block."""
        if isinstance(ref_or_id, AssertionRef):
            idx = self.registry.get(ref_or_id)
            if idx is None:
                return None
            ref_or_id = idx
        return self._position.get(int(ref_or_id))

    def refs(self) -> list[AssertionRef]:
        """The block's assertions, in position order."""
        return [self.registry.ref(int(v)) for v in self.var_ids]

    def bank_slice(self, i: int) -> slice:
        start = self._offsets[i]
        return slice(start, start + len(self.banks[i].edge_vars()))

    def summary(self) -> dict[str, object]:
        return {
            "variables": self.n_vars,
            "edges": self.n_edges,
            "banks": {b.name: b.n_factors() for b in self.banks},
            "factors": sum(b.n_factors() for b in self.banks),
        }

    def __repr__(self) -> str:
        return (
            f"GaussianGraph(vars={self.n_vars}, banks={len(self.banks)}, "
            f"factors={sum(b.n_factors() for b in self.banks)}, edges={self.n_edges})"
        )


# -- natural parameters <-> moments ----------------------------------------------------


def to_moments(natural: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, variance)`` from a ``(..., 2)`` natural-parameter array.

    The precision is clamped to
    [`MIN_PRECISION`][ocbf.model.gaussian.MIN_PRECISION]..[`MAX_PRECISION`][ocbf.model.gaussian.MAX_PRECISION]
    first, so this never divides by zero and never returns a zero-variance belief. Both
    ends are reachable in ordinary running -- the low end from a variable no site has
    spoken about, the high end from several confident sites agreeing.
    """
    precision = np.clip(natural[..., PRECISION], MIN_PRECISION, MAX_PRECISION)
    variance = 1.0 / precision
    return natural[..., POTENTIAL] * variance, variance


def to_natural(mean: np.ndarray | float, variance: np.ndarray | float) -> np.ndarray:
    """A ``(..., 2)`` natural-parameter array from moments."""
    m = np.asarray(mean, dtype=np.float64)
    v = np.clip(np.asarray(variance, dtype=np.float64), 1.0 / MAX_PRECISION, 1.0 / MIN_PRECISION)
    return np.stack([1.0 / v, m / v], axis=-1)


def site_from_moments(
    tilted_mean: np.ndarray,
    tilted_var: np.ndarray,
    cavity: np.ndarray,
) -> np.ndarray:
    """The expectation-propagation site update: matched moments minus the cavity.

    This is the single line every non-Gaussian bank in the layer ends on, and it is what
    design doc section 4.3's "compute the exact tilted moments, project back to a Gaussian,
    propagate" means operationally. A site's job is to carry only what the factor adds to
    the cavity, so the cavity is subtracted back off in natural parameters.

    The result may carry a negative precision. That is not an error -- a factor can make a
    variable *less* certain than the cavity -- and it is left to
    [`ocbf.inference.gabp_ep`][ocbf.inference.gabp_ep] to decide whether the resulting belief
    stays proper, because only the engine can see the other sites.
    """
    tilted = to_natural(tilted_mean, tilted_var)
    return tilted - cavity


def gauss_hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilists' Gauss-Hermite nodes and weights: ``E[f(X)]`` for ``X ~ N(0, 1)``.

    Weights sum to 1 and nodes are scaled for the probabilists' convention, so a tilted
    expectation under a cavity ``N(m, v)`` is ``sum_i w_i f(m + sqrt(v) * x_i)`` with no
    further constants -- which is how every quadrature-based bank in the layer reads.

    Quadrature rather than sampling because the integrands are smooth and one-dimensional:
    32 nodes reach machine precision where Monte Carlo would still be noisy, and noise in a
    message is far worse than bias, since expectation propagation iterates on it.
    """
    if n < 2:
        raise ValueError(f"Gauss-Hermite needs at least 2 nodes, got {n}")
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return nodes, weights / weights.sum()


def scatter_add(edge_var: np.ndarray, values: np.ndarray, n_vars: int) -> np.ndarray:
    """Sum edge messages into their variables.

    ``np.bincount`` per column rather than ``np.add.at``, matching
    [`ocbf.inference.loopy_bp`][ocbf.inference.loopy_bp] and for the same measured reason:
    the unbuffered ufunc is roughly two orders of magnitude slower, and this runs once per
    iteration.
    """
    out = np.zeros((n_vars, values.shape[1]), dtype=np.float64)
    for k in range(values.shape[1]):
        out[:, k] = np.bincount(edge_var, weights=values[:, k], minlength=n_vars)
    return out
