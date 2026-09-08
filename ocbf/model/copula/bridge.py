"""Latent correlations from Kendall's tau, via bridge functions.

Design doc section 4.1 chooses a **rank-based** estimator for the latent correlation matrix,
and the choice is a regime decision rather than a statistical preference. A parametric
mixed MRF needs enough observations per edge to fit a likelihood; rank correlations are
estimable from far fewer, which is what the sparse regime leaves us with
(research notes section 7.2).

The device is a **bridge function**. For two variables that are monotone images of a
bivariate normal with latent correlation ``rho``, the population Kendall's tau is a fixed,
strictly increasing function of ``rho``:

```text
tau  =  bridge(rho)        =>        rho = bridge^-1( tau_hat )
```

For two continuous marginals the bridge is the classical closed form
``tau = (2 / pi) * arcsin(rho)``. Discreteness -- ordinal levels, a binary threshold, a
truncated floor -- introduces ties, which shrink tau toward zero, and the bridge for each
such pair is a different (and considerably longer) expression.

Rather than transcribe one closed form per pair of kinds, this module evaluates the bridge
**once per pair of marginals** by seeded simulation on a grid of ``rho`` and inverts it by
monotone interpolation. Three reasons, in order of weight:

* **It is uniform.** Every combination of the five copula kinds -- including the truncated
  one, whose closed form is the most awkward -- goes through the same code path, so there
  is one thing to get right instead of ten.
* **It is checkable.** The simulated bridge must reproduce the closed form on the
  continuous-continuous pair, and the test suite checks exactly that. A wall of
  special-case formulas offers no such internal cross-check.
* **It is not on any hot path.** Bridges are evaluated when the copula is fitted, not per
  message; the simulation cost is invisible next to inference.

The price is Monte Carlo error in the estimated correlation, which
[`bridge_correlation`][ocbf.model.copula.bridge.bridge_correlation] controls with a fixed
sample size and a fixed seed -- so the result is *deterministic*, and the same inputs give
the same correlation on every run.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats

from ocbf.model.copula.marginals import MarginalTransform

BRIDGE_SAMPLES = 20_000
"""Simulation size for one bridge evaluation.

Large enough that the induced error in ``rho`` is well under the sampling error of the
``tau`` being inverted (which comes from a handful of co-claimed observations), and small
enough that fitting a whole copula stays instant.
"""

BRIDGE_GRID = np.linspace(-0.98, 0.98, 33)
"""Correlation grid the bridge is evaluated on before inversion.

Stops short of +/-1 because the bridge flattens there: a correlation the data cannot
distinguish from 1 should be reported as strong, not as degenerate.
"""

MAX_ABS_CORRELATION = 0.95
"""Ceiling on any fitted latent correlation.

A pairwise Gaussian model built from independently fitted bivariate correlations is not
guaranteed positive definite, and a near-unit correlation is where that fails first.
Clamping keeps the precision blocks diagonally dominant, which is the condition under which
Gaussian belief propagation is known to converge -- so this bound buys convergence, not
just tidiness.
"""


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b between two samples, with the tie correction.

    Tau-b rather than tau-a because ties are the norm here, not an edge case: every
    ordinal, count, binary and truncated marginal produces them by construction, and
    tau-a would read that discreteness as disagreement.

    Delegates the rank statistic to `scipy.stats.kendalltau`, whose merge-sort
    implementation is ``O(n log n)`` -- the naive pairwise form is ``O(n^2)`` and the
    bridge simulation evaluates this on tens of thousands of points.

    Returns 0.0 when fewer than two comparable pairs remain, or when either sample is
    constant. That is the honest reading of "no evidence about this correlation", and it
    inverts to a latent correlation of zero.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"kendall_tau needs matched samples, got {a.shape} and {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2:
        return 0.0
    tau = stats.kendalltau(a, b, variant="b").statistic
    return 0.0 if not np.isfinite(tau) else float(tau)


def bridge_curve(
    left: MarginalTransform, right: MarginalTransform, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """The bridge ``rho -> tau`` for one pair of marginals, on `BRIDGE_GRID`.

    Evaluated by pushing a common seeded bivariate-normal sample through both transforms,
    so the discreteness each marginal introduces enters exactly as it will in the data.
    The returned tau values are monotone by construction of the underlying map, and are
    made strictly monotone before inversion so the interpolation is well posed.
    """
    z = _reference_sample(seed)
    taus = np.empty_like(BRIDGE_GRID)
    for i, rho in enumerate(BRIDGE_GRID):
        z2 = rho * z[0] + np.sqrt(max(1.0 - rho * rho, 0.0)) * z[1]
        taus[i] = kendall_tau(left.to_value(z[0]), right.to_value(z2))
    return BRIDGE_GRID, np.maximum.accumulate(taus)


def bridge_correlation(
    tau: float, left: MarginalTransform, right: MarginalTransform, *, seed: int = 0
) -> float:
    """Invert the bridge: the latent correlation implying an observed Kendall's tau.

    The continuous-continuous pair takes the closed form ``rho = sin(pi * tau / 2)``,
    which is exact and free. Every other combination inverts the simulated
    [`bridge_curve`][ocbf.model.copula.bridge.bridge_curve] by monotone interpolation.

    A tau beyond the bridge's attainable range -- which discreteness makes routine, since
    ties cap the achievable tau well below 1 -- saturates at
    [`MAX_ABS_CORRELATION`][ocbf.model.copula.bridge.MAX_ABS_CORRELATION] rather than
    failing. Saturation is the right reading: the data say "as dependent as this pair of
    marginals can express".
    """
    t = float(np.clip(tau, -1.0, 1.0))
    if not (left.is_censored or right.is_censored):
        return _clamp(np.sin(0.5 * np.pi * t))

    grid, taus = _cached_bridge(left, right, seed)
    if taus[-1] <= taus[0]:
        # A degenerate marginal (one level carrying all the mass) makes tau constant, so
        # the bridge carries no information and no correlation is identified.
        return 0.0
    return _clamp(np.interp(t, taus, grid))


def _clamp(rho: float | np.floating) -> float:
    return float(np.clip(rho, -MAX_ABS_CORRELATION, MAX_ABS_CORRELATION))


@lru_cache(maxsize=256)
def _reference_sample(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two independent standard normal samples, shared across a whole bridge curve.

    Shared so that the tau values along the grid differ only because ``rho`` differs, which
    is what makes the curve monotone rather than jittery -- and therefore invertible.
    """
    rng = np.random.default_rng(seed + 20250908)
    draw = rng.standard_normal((2, BRIDGE_SAMPLES))
    return draw[0], draw[1]


_BRIDGE_CACHE: dict[
    tuple[int, int, int], tuple[MarginalTransform, MarginalTransform, tuple[np.ndarray, np.ndarray]]
] = {}
"""Memoised bridge curves, keyed by transform identity.

A copula over one attribute family shares its marginals across every pair it appears in, so
without this the same curve is simulated once per pair. Keyed by ``id`` rather than by value
because a transform holding numpy arrays is not hashable; the transforms are retained in the
value so their ids cannot be recycled while an entry is live.
"""


def _cached_bridge(
    left: MarginalTransform, right: MarginalTransform, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    key = (id(left), id(right), seed)
    entry = _BRIDGE_CACHE.get(key)
    if entry is None:
        entry = (left, right, bridge_curve(left, right, seed=seed))
        _BRIDGE_CACHE[key] = entry
    return entry[2]
