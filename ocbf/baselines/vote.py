"""Majority and weighted voting over binary assertions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ocbf.assertions import AssertionRef, VariableRegistry
from ocbf.belief import BeliefState, BeliefStateBuilder
from ocbf.reliability import TripletEstimate, triplet_accuracies
from ocbf.sources import ClaimSet, CoverageSemantics, Source

_LOGIT_CAP = 12.0
"""Cap on the summed log-odds.

Without it, twenty copied claims from one cluster drive the posterior to 1 - 1e-9, which is
not merely overconfident but numerically indistinguishable from certainty. The cap does not
fix the dependency problem -- only the ESS diagnostic and the cluster random effect do --
but it keeps the failure legible instead of catastrophic.
"""


def _binary_refs(registry: VariableRegistry, claim_set: ClaimSet) -> list[AssertionRef]:
    out = []
    for ref in claim_set.refs:
        idx = registry.get(ref)
        if idx is not None and int(registry.cardinalities[idx]) == 2:
            out.append(ref)
    return out


def majority_vote(
    registry: VariableRegistry,
    claim_set: ClaimSet,
    *,
    prior: float = 0.5,
    pseudocount: float = 1.0,
) -> BeliefState:
    """Unweighted vote with Laplace smoothing.

    The floor beneath the floor. Any method that cannot beat this has no reason to exist.
    """
    builder = BeliefStateBuilder(registry)
    for ref in _binary_refs(registry, claim_set):
        idx = registry.index(ref)
        votes = [c.value for c in claim_set.for_ref(ref) if isinstance(c.value, bool)]
        if not votes:
            continue
        n_true = sum(1 for v in votes if v)
        p = (n_true + pseudocount * prior) / (len(votes) + pseudocount)
        builder.set_discrete(idx, np.array([1.0 - p, p]))
    return builder.build(diagnostics={"method": "majority_vote"})


def weighted_vote(
    registry: VariableRegistry,
    claim_set: ClaimSet,
    *,
    weights: Mapping[str, float] | None = None,
    estimate: TripletEstimate | None = None,
    prior_logit: float = 0.0,
    sources: Mapping[str, Source] | None = None,
) -> BeliefState:
    """Log-odds vote with per-source weights.

    Weights come from, in order of preference: an explicit ``weights`` map; a
    [`TripletEstimate`][ocbf.reliability.TripletEstimate]; or a fresh triplet estimate computed here.
    The triplet route is label-free, which is the point -- a "weighted vote baseline" that
    needed gold labels would not be a baseline for an unsupervised problem.

    Silence is honoured. When ``sources`` is supplied, a source declaring
    ``COMPLETE_OVER_SCOPE`` contributes a negative vote on every in-scope assertion it did
    not claim. Ignoring that would discard the false-negative signal that is the whole
    point of a detector-style source (Stage 1 section 4.5).
    """
    if weights is None:
        estimate = estimate or triplet_accuracies(claim_set)
        weights = {s: estimate.weight(s) for s in claim_set.source_ids}
    default_w = float(np.median(list(weights.values()))) if weights else 0.0

    builder = BeliefStateBuilder(registry)
    contributions: dict[AssertionRef, dict[str, float]] = {}

    for ref in _binary_refs(registry, claim_set):
        acc: dict[str, float] = {"prior": prior_logit}
        for c in claim_set.for_ref(ref):
            if not isinstance(c.value, bool):
                continue
            w = weights.get(c.source_id, default_w)
            acc[c.source_id] = acc.get(c.source_id, 0.0) + (w if c.value else -w)
        contributions[ref] = acc

    if sources:
        for source in sources.values():
            profile = source.profile
            if profile.coverage is not CoverageSemantics.COMPLETE_OVER_SCOPE:
                continue
            w = weights.get(profile.source_id, default_w)
            claimed = {c.ref for c in source.claims()}
            for ref in source.scope():
                if ref in claimed or ref not in contributions:
                    continue
                acc = contributions[ref]
                acc[profile.source_id] = acc.get(profile.source_id, 0.0) - w

    for ref, acc in contributions.items():
        idx = registry.index(ref)
        logit = float(np.clip(sum(acc.values()), -_LOGIT_CAP, _LOGIT_CAP))
        p = 1.0 / (1.0 + np.exp(-logit))
        builder.set_discrete(idx, np.array([1.0 - p, p]))
        builder.set_attribution(idx, acc)

    return builder.build(diagnostics={"method": "weighted_vote"})
