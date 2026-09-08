"""Point aggregation of continuous claims -- the bar the continuous layer must clear.

Design doc section 8.3 makes a baseline mandatory and always-reported, and the argument is
no weaker on the continuous side than on the binary one. If fusing timestamps through a
copula, a conditional-Gaussian coupling and a truncation factor does not beat taking the
median of what the sources said, the machinery is not earning its place.

Both estimators here return a full [`BeliefState`][ocbf.belief.state.BeliefState] rather than
a bare number, so the comparison is on the same footing as the engine's: a method that
cannot state its uncertainty cannot be scored on calibration, and calibration is where
Stage 1 section 3.2 says the value actually is. The uncertainty they state is the honest one
available to an aggregator -- the spread of the claims themselves, falling back to the pooled
channel scale where a single witness leaves no spread to measure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ocbf.assertions import AssertionRef, VariableRegistry
from ocbf.belief import BeliefState, BeliefStateBuilder
from ocbf.model.copula import CopulaSpec, MarginalTransform
from ocbf.reliability.params import ContinuousChannelTable
from ocbf.sources import ClaimSet


def _continuous_claims(claim_set: ClaimSet, ref: AssertionRef) -> list[tuple[str, float]]:
    return [
        (c.source_id, float(c.value))
        for c in claim_set.for_ref(ref)
        if isinstance(c.value, (int, float)) and not isinstance(c.value, bool)
    ]


def claim_median(
    registry: VariableRegistry,
    claim_set: ClaimSet,
    copula: CopulaSpec,
    *,
    channels: ContinuousChannelTable | None = None,
) -> BeliefState:
    """Median of the claims, with a spread read off their disagreement.

    The floor beneath the floor, and the continuous counterpart of
    [`majority_vote`][ocbf.baselines.vote.majority_vote]. The median rather than the mean
    because the simulated regime -- and the real one it stands for -- produces gross
    outliers, and a baseline that a single wild claim can drag is a straw man rather than a
    bar.

    Spread is the claims' own scaled median absolute deviation divided by the square root of
    their count, which is the standard-error shape an aggregator can justify from what it can
    see. A lone witness has no spread to measure and falls back to the pooled channel scale.
    """
    return _aggregate(registry, claim_set, copula, channels, _median_estimate)


def weighted_mean(
    registry: VariableRegistry,
    claim_set: ClaimSet,
    copula: CopulaSpec,
    *,
    channels: ContinuousChannelTable | None = None,
) -> BeliefState:
    """Inverse-variance-weighted mean, using each source's fitted channel scale.

    The continuous analogue of [`weighted_vote`][ocbf.baselines.vote.weighted_vote], and the
    genuine bar: it already uses the per-source reliability that the fitted channels supply,
    so beating it means the *structure* is doing work -- the copula's correlations, the time
    bracket, the type coupling, the precedence orderings -- and not merely that the engine
    knows which sources to trust.

    Inverse-variance weighting is the maximum-likelihood combination for independent Gaussian
    observations, so it is also the strongest thing a per-assertion aggregator can be, which
    is what makes it the right bar rather than a convenient one.
    """
    return _aggregate(registry, claim_set, copula, channels, _weighted_estimate)


def _aggregate(
    registry: VariableRegistry,
    claim_set: ClaimSet,
    copula: CopulaSpec,
    channels: ContinuousChannelTable | None,
    estimate: Callable[
        [Sequence[tuple[str, float]], str, ContinuousChannelTable | None], tuple[float, float]
    ],
) -> BeliefState:
    """Shared plumbing: aggregate in observed units, then report in latent ones.

    Aggregating in observed units and transforming afterwards, rather than transforming each
    claim first, is what keeps this a fair baseline: an aggregator that had to know the copula
    to combine two numbers would not be the simple method it is standing in for. The transform
    enters only to state the answer in the same coordinates the engine's posterior uses, so
    both are scored by the same rule.
    """
    from ocbf.model.copula import marginal_key

    builder = BeliefStateBuilder(registry)
    for ref in claim_set.refs:
        idx = registry.get(ref)
        if idx is None or int(registry.cardinalities[idx]) != 0 or not copula.has(ref):
            continue
        values = _continuous_claims(claim_set, ref)
        if not values:
            continue
        marginal: MarginalTransform = copula.marginal(ref)
        centre, spread = estimate(values, marginal_key(ref), channels)

        # A spread in observed units becomes a latent one through the local slope of the
        # transform, which is exact for the affine marginal timestamps use and a first-order
        # reading elsewhere -- the same reading the baseline's own standard error is.
        latent_mean = float(np.asarray(marginal.to_latent(centre)))
        latent_sd = max(spread / max(marginal.scale, 1e-9), 1e-6)
        builder.set_continuous(
            idx, latent_mean, latent_sd**2, marginal=marginal, prior_only=False
        )
    return builder.build(diagnostics={"method": "continuous_baseline"})


def _median_estimate(
    values: Sequence[tuple[str, float]],
    template: str,
    channels: ContinuousChannelTable | None,
) -> tuple[float, float]:
    y = np.array([v for _s, v in values], dtype=np.float64)
    centre = float(np.median(y))
    if y.size > 1:
        mad = 1.4826 * float(np.median(np.abs(y - centre)))
        if mad > 0.0:
            return centre, mad / np.sqrt(y.size)
    fallback = channels[(template, values[0][0])].scale if channels is not None else 1.0
    return centre, float(fallback) / np.sqrt(y.size)


def _weighted_estimate(
    values: Sequence[tuple[str, float]],
    template: str,
    channels: ContinuousChannelTable | None,
) -> tuple[float, float]:
    if channels is None:
        return _median_estimate(values, template, channels)
    biases = np.array([channels[(template, s)].bias for s, _v in values])
    scales = np.array([channels[(template, s)].scale for s, _v in values])
    y = np.array([v for _s, v in values], dtype=np.float64) - biases
    weights = 1.0 / np.square(np.maximum(scales, 1e-9))
    total = weights.sum()
    return float((weights * y).sum() / total), float(1.0 / np.sqrt(total))
