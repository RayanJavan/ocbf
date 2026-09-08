"""Dawid-Skene EM for binary assertions.

The 1979 reference model, and the intermediate baseline between weighted voting and the
full engine. Two-sided quality (sensitivity and specificity separately) rather than a
single coin, because Stage 1 section 5.1 is emphatic that a scalar accuracy cannot express
systematic bias, and because under ``COMPLETE_OVER_SCOPE`` coverage the specificity is
precisely what a source's silence means.

Beta priors are not decoration here. In the sparse regime a source with three claims would
otherwise get an accuracy of 0 or 1, and EM would then treat it as an oracle. The priors
are the minimal form of the pooling that design doc section 5.3 does properly.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ocbf.assertions import AssertionRef, VariableRegistry
from ocbf.belief import BeliefState, BeliefStateBuilder
from ocbf.sources import ClaimSet, CoverageSemantics, Source

_EPS = 1e-9


def dawid_skene(
    registry: VariableRegistry,
    claim_set: ClaimSet,
    *,
    sources: Mapping[str, Source] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    prior_true: float = 0.5,
    alpha_prior: tuple[float, float] = (4.0, 2.0),
    beta_prior: tuple[float, float] = (4.0, 2.0),
    init_sensitivity: float = 0.7,
    init_specificity: float = 0.7,
) -> BeliefState:
    """Two-sided Dawid-Skene by EM.

    ``alpha_prior``/``beta_prior`` are ``(a, b)`` Beta hyper-parameters on sensitivity and
    specificity. Their means sit above 0.5, which is the better-than-chance assumption that
    breaks the "all sources are adversarial" mirror symmetry (Stage 1 section 10 item 1).
    Without it EM is free to converge on the mirror solution and report every truth
    inverted, with an identical likelihood.
    """
    refs = [
        r
        for r in claim_set.refs
        if (i := registry.get(r)) is not None and int(registry.cardinalities[i]) == 2
    ]
    if not refs:
        return BeliefStateBuilder(registry).build(diagnostics={"method": "dawid_skene", "n": 0})

    ref_index = {r: j for j, r in enumerate(refs)}
    source_ids = list(claim_set.source_ids)
    src_index = {s: i for i, s in enumerate(source_ids)}
    n_a, n_s = len(refs), len(source_ids)

    # Observation matrix: +1 claimed true, -1 claimed false, 0 no claim.
    obs = np.zeros((n_s, n_a), dtype=np.int8)
    for c in claim_set:
        j = ref_index.get(c.ref)
        if j is None or not isinstance(c.value, bool):
            continue
        obs[src_index[c.source_id], j] = 1 if c.value else -1

    # Silence as evidence, for sources that declared it as such.
    if sources:
        for source in sources.values():
            profile = source.profile
            if profile.coverage is not CoverageSemantics.COMPLETE_OVER_SCOPE:
                continue
            i = src_index.get(profile.source_id)
            if i is None:
                continue
            for ref in source.scope():
                j = ref_index.get(ref)
                if j is not None and obs[i, j] == 0:
                    obs[i, j] = -1

    said_true = obs == 1
    said_false = obs == -1

    sens = np.full(n_s, init_sensitivity, dtype=np.float64)
    spec = np.full(n_s, init_specificity, dtype=np.float64)
    pi = float(prior_true)
    posterior = np.full(n_a, pi, dtype=np.float64)

    a_a, a_b = alpha_prior
    b_a, b_b = beta_prior

    for _ in range(max_iter):
        # -- E step: log-odds of truth, accumulated over claims and honoured silences --
        log_true = (
            said_true * np.log(sens + _EPS)[:, None] + said_false * np.log(1 - sens + _EPS)[:, None]
        ).sum(axis=0)
        log_false = (
            said_true * np.log(1 - spec + _EPS)[:, None] + said_false * np.log(spec + _EPS)[:, None]
        ).sum(axis=0)

        logit = np.log(pi + _EPS) - np.log(1 - pi + _EPS) + log_true - log_false
        new_posterior = 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))

        delta = float(np.abs(new_posterior - posterior).max())
        posterior = new_posterior

        # -- M step: Beta-posterior means, which is where the shrinkage happens ---------
        w_true = posterior[None, :]
        w_false = 1.0 - posterior[None, :]

        tp = (said_true * w_true).sum(axis=1)
        fn = (said_false * w_true).sum(axis=1)
        tn = (said_false * w_false).sum(axis=1)
        fp = (said_true * w_false).sum(axis=1)

        sens = (tp + a_a) / (tp + fn + a_a + a_b)
        spec = (tn + b_a) / (tn + fp + b_a + b_b)
        pi = float((posterior.sum() + 1.0) / (n_a + 2.0))

        if delta < tol:
            break

    builder = BeliefStateBuilder(registry)
    for ref, j in ref_index.items():
        idx = registry.index(ref)
        p = float(np.clip(posterior[j], _EPS, 1 - _EPS))
        builder.set_discrete(idx, np.array([1.0 - p, p]))

    return builder.build(
        diagnostics={
            "method": "dawid_skene",
            "assertions": n_a,
            "sources": n_s,
            "prior_true": round(pi, 4),
            "sensitivity_mean": round(float(sens.mean()), 4),
            "specificity_mean": round(float(spec.mean()), 4),
        }
    )
