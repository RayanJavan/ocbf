"""Monotone transforms between observed ordered values and standardized Gaussian coordinates. Continuous values use point transforms; discrete ordered values use latent censoring intervals. Unordered categories are outside this representation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from scipy.special import ndtr, ndtri

from ocbf.schema import AttributeKind, AttributeSpec

_TAIL = 1.0e-9
"""Probability clamp keeping ``ndtri`` away from an infinite image.

An empirical CDF hits exactly 0 and 1 at the extremes of its sample, and the latent image
of those is infinite. Every transform clamps instead, because an infinite latent coordinate
is not a very confident observation -- it is a broken one.
"""

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


@runtime_checkable
class MarginalTransform(Protocol):
    """The monotone map between one observed quantity and its latent Gaussian.

    Implementations are frozen and vectorised: every method accepts a scalar or an array
    and returns the matching shape, because grounding transforms thousands of claims at
    once and reporting transforms every variable in the block.
    """

    kind: AttributeKind

    @property
    def is_censored(self) -> bool:
        """Whether an observation pins ``z`` to an interval rather than to a point."""
        ...

    @property
    def scale(self) -> float:
        """Observed-value change spanned by one latent unit at the median.

        The bridge between a tolerance stated in observed units -- a precedence slack in
        hours, a measurement error in euros -- and the latent units the engine works in.
        """
        ...

    def to_latent(self, value: np.ndarray | float) -> np.ndarray:
        """Latent image of an observed value; the interval midpoint for censored kinds."""
        ...

    def to_latent_interval(self, value: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """Latent interval ``[lo, hi]`` consistent with an observed value.

        Degenerate (``lo == hi``) for continuous kinds, where the observation pins ``z``.
        """
        ...

    def to_value(self, z: np.ndarray | float) -> np.ndarray:
        """Observed value at a latent coordinate -- the inverse of `to_latent`."""
        ...


@dataclass(frozen=True, slots=True)
class GaussianMarginal:
    """Affine marginal transform ``v = mean + sd*z``. Linear transforms also allow differences to be expressed exactly in latent coordinates when the remaining model assumptions permit it."""

    mean: float = 0.0
    sd: float = 1.0
    kind: ClassVar[AttributeKind] = AttributeKind.CONTINUOUS

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("GaussianMarginal needs a finite mean")
        if not (self.sd > 0.0 and np.isfinite(self.sd)):
            raise ValueError(f"GaussianMarginal needs a positive finite sd, got {self.sd}")

    @property
    def is_censored(self) -> bool:
        return False

    @property
    def scale(self) -> float:
        return float(self.sd)

    def to_latent(self, value: np.ndarray | float) -> np.ndarray:
        return (_asarray(value) - self.mean) / self.sd

    def to_latent_interval(self, value: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        z = self.to_latent(value)
        return z, z

    def to_value(self, z: np.ndarray | float) -> np.ndarray:
        return self.mean + self.sd * _asarray(z)

    @classmethod
    def fit(cls, samples: Sequence[float] | np.ndarray) -> GaussianMarginal:
        """Method of moments. A degenerate sample gets unit width rather than zero."""
        x = _finite(samples)
        if x.size == 0:
            return cls(0.0, 1.0)
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        return cls(float(x.mean()), sd if sd > 0.0 else 1.0)


@dataclass(frozen=True, slots=True)
class EmpiricalMarginal:
    """A nonparametric transform: the sample quantile function, interpolated.

    The honest default when the marginal shape is unknown, which in this system it usually
    is -- the values are latent and all anyone ever sees are noisy reports of them.
    Plotting positions are ``(i + 0.5) / n``, which keeps both extremes at a finite latent
    image without the asymmetry ``i / n`` would introduce.

    Outside the sample range the map continues linearly at the slope of the end segment.
    Extrapolating is a stated choice rather than an accident: it invents no value the data
    support, and it keeps the transform a bijection on the whole real line, which the
    engine requires.
    """

    values: np.ndarray
    latent: np.ndarray
    kind: ClassVar[AttributeKind] = AttributeKind.CONTINUOUS

    @property
    def is_censored(self) -> bool:
        return False

    @property
    def scale(self) -> float:
        return float(self.to_value(0.5) - self.to_value(-0.5))

    def to_latent(self, value: np.ndarray | float) -> np.ndarray:
        return _interp_extrapolate(_asarray(value), self.values, self.latent)

    def to_latent_interval(self, value: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        z = self.to_latent(value)
        return z, z

    def to_value(self, z: np.ndarray | float) -> np.ndarray:
        return _interp_extrapolate(_asarray(z), self.latent, self.values)

    @classmethod
    def fit(cls, samples: Sequence[float] | np.ndarray) -> EmpiricalMarginal:
        x = np.sort(_finite(samples))
        if x.size < 2 or x[0] == x[-1]:
            # Too little spread to describe a shape. Fall back to the moment fit, which
            # degrades to unit width rather than to a non-invertible constant map.
            fitted = GaussianMarginal.fit(x)
            grid = np.linspace(-3.0, 3.0, 7)
            return cls(fitted.to_value(grid), grid)
        p = (np.arange(x.size) + 0.5) / x.size
        return cls(x, ndtri(p))


@dataclass(frozen=True, slots=True)
class OrdinalMarginal:
    """An interval-censoring transform for count, ordinal and binary values.

    Observing level ``k`` does not pin ``z``; it says ``z`` fell between the cutpoints
    ``Phi^-1(F(k-1))`` and ``Phi^-1(F(k))``. Treating that as a point observation -- which
    is what happens when a count is handed to a continuous transform -- fabricates
    precision the data do not contain, and does so most severely exactly where the levels
    are coarsest.

    ``levels`` are the ordered distinct values; ``cumulative`` is the CDF at each, ending
    at 1.
    """

    levels: np.ndarray
    cumulative: np.ndarray
    kind: AttributeKind = AttributeKind.ORDINAL

    def __post_init__(self) -> None:
        if self.levels.shape != self.cumulative.shape:
            raise ValueError("levels and cumulative must have the same shape")
        if self.levels.size == 0:
            raise ValueError("an ordinal marginal needs at least one level")
        if np.any(np.diff(self.levels) <= 0):
            raise ValueError("levels must be strictly increasing")
        if np.any(np.diff(self.cumulative) < 0):
            raise ValueError("cumulative probabilities must be non-decreasing")

    @property
    def is_censored(self) -> bool:
        return True

    @property
    def cutpoints(self) -> np.ndarray:
        """``K`` latent thresholds; entry ``k`` is the upper edge of level ``k``."""
        return _clamped_ndtri(self.cumulative)

    @property
    def scale(self) -> float:
        return float(self.levels[-1] - self.levels[0]) / max(len(self.levels) - 1, 1)

    def to_latent_interval(self, value: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        cuts = self.cutpoints
        last = len(self.levels) - 1
        k = self._level_index(_asarray(value))
        lo = np.where(k > 0, cuts[np.maximum(k - 1, 0)], -np.inf)
        hi = np.where(k < last, cuts[np.minimum(k, last)], np.inf)
        return lo, hi

    def to_latent(self, value: np.ndarray | float) -> np.ndarray:
        """Mean of the standard normal truncated to the observed level's interval."""
        lo, hi = self.to_latent_interval(value)
        mean, _ = truncated_normal_moments(lo, hi)
        return mean

    def to_value(self, z: np.ndarray | float) -> np.ndarray:
        idx = np.searchsorted(self.cutpoints, _asarray(z), side="left")
        return self.levels[np.clip(idx, 0, len(self.levels) - 1)]

    def _level_index(self, value: np.ndarray) -> np.ndarray:
        """Index of the declared level nearest each value.

        Nearest rather than exact: a source reporting ``2.0`` for an integer count means
        level 2 and should be read that way, not discarded. Values outside the declared
        range saturate at the end levels, which is the reading the cutpoints already give
        them.
        """
        if self.levels.size == 1:
            return np.zeros(np.shape(value), dtype=np.int64)
        pos = np.clip(np.searchsorted(self.levels, value), 1, len(self.levels) - 1)
        left = self.levels[pos - 1]
        right = self.levels[pos]
        return np.where(np.abs(value - left) <= np.abs(right - value), pos - 1, pos)

    @classmethod
    def fit(
        cls,
        samples: Sequence[float] | np.ndarray,
        *,
        kind: AttributeKind = AttributeKind.ORDINAL,
        levels: Sequence[float] | None = None,
    ) -> OrdinalMarginal:
        x = _finite(samples)
        declared = None if levels is None else np.asarray(levels, dtype=np.float64)
        observed = np.unique(x) if x.size else np.array([0.0, 1.0])
        grid = declared if declared is not None else observed
        # Laplace smoothing. A declared level that happens not to appear in the sample must
        # still carry positive mass, or its latent interval collapses and no amount of
        # evidence could ever place a value there.
        counts = np.array([float(np.count_nonzero(x == v)) for v in grid]) + 1.0
        return cls(grid, np.cumsum(counts) / counts.sum(), kind)


@dataclass(frozen=True, slots=True)
class TruncatedMarginal:
    """Marginal transform with a point mass at a floor and a continuous tail above it. Values at the floor represent interval censoring in latent coordinates."""

    point: float
    point_mass: float
    tail: MarginalTransform
    kind: ClassVar[AttributeKind] = AttributeKind.TRUNCATED

    def __post_init__(self) -> None:
        if not 0.0 < self.point_mass < 1.0:
            raise ValueError(f"point_mass must be in (0, 1), got {self.point_mass}")

    @property
    def is_censored(self) -> bool:
        return True

    @property
    def cutpoint(self) -> float:
        """Latent threshold below which the value sits at the floor."""
        return float(_clamped_ndtri(np.asarray(self.point_mass)))

    @property
    def scale(self) -> float:
        return self.tail.scale

    def to_latent_interval(self, value: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        v = _asarray(value)
        z = self._tail_latent(v)
        at_floor = v <= self.point
        lo = np.where(at_floor, -np.inf, z)
        hi = np.where(at_floor, self.cutpoint, z)
        return lo, hi

    def to_latent(self, value: np.ndarray | float) -> np.ndarray:
        lo, hi = self.to_latent_interval(value)
        mean, _ = truncated_normal_moments(lo, hi)
        return np.where(np.isfinite(lo), hi, mean)

    def to_value(self, z: np.ndarray | float) -> np.ndarray:
        zz = _asarray(z)
        p = (ndtr(zz) - self.point_mass) / (1.0 - self.point_mass)
        tail_value = self.tail.to_value(_clamped_ndtri(np.clip(p, 0.0, 1.0)))
        return np.where(zz <= self.cutpoint, self.point, tail_value)

    def _tail_latent(self, v: np.ndarray) -> np.ndarray:
        p_tail = ndtr(self.tail.to_latent(np.maximum(v, self.point)))
        return _clamped_ndtri(self.point_mass + (1.0 - self.point_mass) * p_tail)

    @classmethod
    def fit(cls, samples: Sequence[float] | np.ndarray, *, point: float = 0.0) -> TruncatedMarginal:
        x = _finite(samples)
        above = x[x > point]
        mass = float(np.clip((x.size - above.size + 0.5) / (x.size + 1.0), 1e-3, 1 - 1e-3))
        tail = EmpiricalMarginal.fit(above if above.size else np.array([point, point + 1.0]))
        return cls(point, mass, tail)


def fit_marginal(
    samples: Sequence[float] | np.ndarray,
    *,
    kind: AttributeKind = AttributeKind.CONTINUOUS,
    spec: AttributeSpec | None = None,
) -> MarginalTransform:
    """Fit a standalone marginal transform for the schema-declared attribute kind. Unordered categorical attributes are rejected because they lack a monotone Gaussian transform."""
    if spec is not None:
        kind = spec.kind
    if not kind.in_copula:
        raise ValueError(
            f"{kind.value} has no monotone transform to a latent Gaussian; "
            "use a finite categorical variable for unordered values"
        )
    match kind:
        case AttributeKind.CONTINUOUS:
            return EmpiricalMarginal.fit(samples)
        case AttributeKind.TRUNCATED:
            lower = None if spec is None else spec.bounds[0]
            return TruncatedMarginal.fit(samples, point=0.0 if lower is None else float(lower))
        case AttributeKind.BINARY:
            return OrdinalMarginal.fit(samples, kind=kind, levels=(0.0, 1.0))
        case _:
            levels = None
            if spec is not None and spec.levels:
                levels = tuple(float(i) for i in range(len(spec.levels)))
            return OrdinalMarginal.fit(samples, kind=kind, levels=levels)


def truncated_normal_moments(
    lo: np.ndarray | float,
    hi: np.ndarray | float,
    mean: np.ndarray | float = 0.0,
    var: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of ``N(mean, var)`` restricted to ``[lo, hi]``.

    Closed form, and the workhorse behind every censored observation in the layer: an
    ordinal level, a truncated floor and a hard precedence constraint are all "this
    Gaussian, restricted to an interval".

    Two numerical points are load-bearing, because expectation propagation *will* put a
    cavity far away from an interval while it is still settling, and a wrong answer there
    does not stay local -- it becomes a site with enormous precision that then dominates
    every other factor on the variable.

    **The mass is computed in whichever tail does not cancel.** ``Phi(b) - Phi(a)`` for two
    large positive bounds subtracts two numbers near 1 and loses every significant digit;
    ``Phi(-a) - Phi(-b)`` computes the same quantity from two small numbers and keeps them.

    **A mass that underflows anyway falls back to an asymptotic, not to a point.** Far out in
    a tail the truncated Gaussian is not degenerate: a two-sided interval tends to the
    uniform on that interval, and a one-sided one to the Mills-ratio limit ``mean ~ a + 1/a``
    with ``variance ~ 1/a^2``. Returning a near-zero variance there -- the obvious guard --
    manufactures near-certainty out of an arithmetic underflow, which is exactly the failure
    the guard was meant to prevent.
    """
    m = _asarray(mean)
    v = np.maximum(_asarray(var), _TAIL)
    sd = np.sqrt(v)
    a = (_asarray(lo) - m) / sd
    b = (_asarray(hi) - m) / sd

    upper_tail = a > 0.0
    mass = np.where(upper_tail, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    degenerate = mass <= 1e-300
    safe = np.where(degenerate, 1.0, mass)

    phi_a = _phi(a)
    phi_b = _phi(b)
    # The density is already zero at an infinite bound, so the bound itself is zeroed before
    # the product: ``inf * 0.0`` is ``nan``, and both branches of a ``np.where`` are
    # evaluated.
    ratio = (phi_a - phi_b) / safe
    a_term = np.where(np.isfinite(a), a, 0.0) * phi_a
    b_term = np.where(np.isfinite(b), b, 0.0) * phi_b
    z_var = 1.0 + (a_term - b_term) / safe - np.square(ratio)

    fallback_mean, fallback_var = _tail_moments(a, b)
    z_mean = np.where(degenerate, fallback_mean, ratio)
    z_var = np.where(degenerate | ~np.isfinite(z_var) | (z_var <= 0.0), fallback_var, z_var)
    return m + sd * z_mean, v * np.maximum(z_var, _TAIL)


def _tail_moments(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardised moments of a normal truncated to ``[a, b]``, in the far-tail limit.

    Bounded intervals tend to the uniform; one-sided ones to the Mills-ratio asymptotics.
    Both are continuous with the exact formulas at the edge of where those are computable,
    so the fallback does not introduce a jump.
    """
    bounded = np.isfinite(a) & np.isfinite(b)
    edge = np.where(np.isfinite(a), a, np.where(np.isfinite(b), b, 0.0))
    safe_edge = np.where(np.abs(edge) > 1.0, edge, np.sign(edge) + (edge == 0.0))

    one_sided_mean = safe_edge + 1.0 / safe_edge
    one_sided_var = 1.0 / np.square(safe_edge)

    width = np.where(bounded, b - a, 0.0)
    uniform_mean = np.where(bounded, 0.5 * (a + b), 0.0)
    uniform_var = np.square(width) / 12.0

    mean = np.where(bounded, uniform_mean, one_sided_mean)
    var = np.where(bounded, uniform_var, one_sided_var)
    return mean, np.maximum(var, _TAIL)


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal density, zero at an infinite argument."""
    density = np.exp(-0.5 * np.square(np.clip(x, -40.0, 40.0))) / _SQRT_2PI
    return np.where(np.isfinite(x), density, 0.0)


def _asarray(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _finite(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _clamped_ndtri(p: np.ndarray) -> np.ndarray:
    return ndtri(np.clip(p, _TAIL, 1.0 - _TAIL))


def _interp_extrapolate(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation that continues at the end slopes instead of flattening.

    ``np.interp`` clamps outside its range, which would make the transform non-invertible
    in both tails -- every extreme value would share one latent coordinate.
    """
    out = np.interp(x, xp, fp)
    if xp.size >= 2:
        left = (fp[1] - fp[0]) / (xp[1] - xp[0])
        right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out = np.where(x < xp[0], fp[0] + left * (x - xp[0]), out)
        out = np.where(x > xp[-1], fp[-1] + right * (x - xp[-1]), out)
    return out
