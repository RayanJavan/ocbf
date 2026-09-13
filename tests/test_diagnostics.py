"""Diagnostics are results, so they get tested like results.

Each test here pins a claim the pipeline makes about its own preconditions. If the
identifiability verdict, the effective sample size or the decidability threshold is wrong,
the belief state is still *produced* -- it is just no longer trustworthy in the specific way
it advertises. These tests are what keep that advertisement honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocbf.assertions import AssertionRef
from ocbf.diagnostics import (
    Identifiability,
    chernoff_binary,
    decidability,
    effective_sample_sizes,
    overlap_report,
)
from ocbf.reliability import pairwise_agreement, triplet_accuracies
from ocbf.reliability.moments import BinaryClaimTable
from ocbf.sources import Claim, ClaimSet


def _refs(n: int) -> list[AssertionRef]:
    return [AssertionRef.event_exists(f"e{i}") for i in range(n)]


def _claims(assignments: dict[str, dict[int, bool]], refs: list[AssertionRef]) -> ClaimSet:
    return ClaimSet(
        [Claim(sid, refs[i], v) for sid, votes in assignments.items() for i, v in votes.items()]
    )


# -- Chernoff information -----------------------------------------------------------


def test_chance_channel_carries_no_information():
    """A source at chance must contribute exactly zero, not merely a little."""
    assert chernoff_binary(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)


def test_chernoff_increases_with_accuracy():
    values = [chernoff_binary(a, a) for a in (0.55, 0.7, 0.85, 0.95)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


def test_chernoff_is_symmetric_in_its_two_sides():
    assert chernoff_binary(0.9, 0.6) == pytest.approx(chernoff_binary(0.6, 0.9), rel=1e-6)


# -- decidability -------------------------------------------------------------------


def test_weak_sources_cannot_reach_the_threshold():
    """Weak binary reports do not meet the declared information threshold.

    Three sources at 0.6 accuracy carry ~0.02 nats each. The eps=0.1 target is 2.3 nats.
    No aggregation rule decides this assertion, and the report must say so."""
    refs = _refs(1)
    cs = _claims({f"s{i}": {0: True} for i in range(3)}, refs)
    rep = decidability(cs, {f"s{i}": 0.6 for i in range(3)}, eps=0.1)
    assert not rep.is_decidable(refs[0])
    assert rep.evidence[refs[0]] < 0.1


def test_many_strong_sources_reach_the_threshold():
    refs = _refs(1)
    cs = _claims({f"s{i}": {0: True} for i in range(8)}, refs)
    rep = decidability(cs, {f"s{i}": 0.9 for i in range(8)}, eps=0.1)
    assert rep.is_decidable(refs[0])


def test_ess_correction_only_ever_reduces_evidence():
    refs = _refs(1)
    cs = _claims({f"s{i}": {0: True} for i in range(6)}, refs)
    acc = {f"s{i}": 0.9 for i in range(6)}
    raw = decidability(cs, acc, eps=0.1)
    corrected = decidability(cs, acc, eps=0.1, ess={refs[0]: 2.0})
    assert corrected.evidence[refs[0]] < raw.evidence[refs[0]]


# -- overlap and identifiability -----------------------------------------------------


def test_isolated_sources_are_prior_only():
    """No overlap means reliability is assumed, not measured."""
    refs = _refs(6)
    cs = _claims({"s0": {0: True, 1: True}, "s1": {2: True, 3: True}}, refs)
    rep = overlap_report(cs, min_overlap=2)
    assert all(v is Identifiability.PRIOR_ONLY for v in rep.verdict.values())
    assert not rep.is_globally_identifiable


def test_bipartite_component_is_flagged_sign_ambiguous():
    """A component with no odd cycle admits the mirror solution (arXiv:1706.06660).

    Sources a and b both overlap c, but not each other: the component is a path, which is
    bipartite, so the sign of every skill is unresolvable from data alone.
    """
    refs = _refs(8)
    cs = _claims(
        {
            "a": {0: True, 1: True},
            "c": {0: True, 1: True, 2: True, 3: True},
            "b": {2: True, 3: True},
        },
        refs,
    )
    rep = overlap_report(cs, min_overlap=2)
    assert rep.n_components == 1
    assert set(rep.verdict.values()) == {Identifiability.SIGN_AMBIGUOUS}
    assert not rep.is_globally_identifiable


def test_triangle_is_identified():
    """A triangle is an odd cycle, which is exactly the identifiability condition."""
    refs = _refs(4)
    votes = {i: True for i in range(4)}
    cs = _claims({"a": votes, "b": votes, "c": votes}, refs)
    rep = overlap_report(cs, min_overlap=2)
    assert set(rep.verdict.values()) == {Identifiability.IDENTIFIED}
    assert rep.is_globally_identifiable


def test_explain_mentions_non_identifiability():
    refs = _refs(4)
    cs = _claims({"s0": {0: True}, "s1": {1: True}}, refs)
    assert "Not globally identifiable" in overlap_report(cs).explain()


# -- effective sample size ------------------------------------------------------------


def test_duplicate_cluster_members_deflate_ess():
    """Copies within a declared family must not count as independent votes."""
    refs = _refs(10)
    votes = {i: True for i in range(10)}
    cs = _claims({"s0": votes, "s1": votes, "s2": votes}, refs)
    clusters = dict.fromkeys(["s0", "s1", "s2"], "same_vendor")
    rep = effective_sample_sizes(cs, clusters, {s: 0.8 for s in clusters})
    assert rep.deg[refs[0]] == 3
    assert rep.ess[refs[0]] < 3.0
    assert rep.deflation(refs[0]) < 1.0


def test_distinct_clusters_are_not_deflated():
    refs = _refs(10)
    votes = {i: True for i in range(10)}
    cs = _claims({"s0": votes, "s1": votes, "s2": votes}, refs)
    clusters = {"s0": "a", "s1": "b", "s2": "c"}
    rep = effective_sample_sizes(cs, clusters, {s: 0.8 for s in clusters})
    assert rep.ess[refs[0]] == pytest.approx(3.0)


# -- moment estimators ----------------------------------------------------------------


def test_pairwise_agreement_is_nan_without_overlap():
    """Absent overlap is 'we cannot say', which must not be collapsed into 'uncorrelated'."""
    refs = _refs(4)
    cs = _claims({"s0": {0: True, 1: True}, "s1": {2: True, 3: True}}, refs)
    table = BinaryClaimTable(cs)
    agree = pairwise_agreement(table, min_overlap=2)
    i, j = table.source_index("s0"), table.source_index("s1")
    assert np.isnan(agree[i, j])


def test_triplet_reports_uncovered_sources():
    refs = _refs(4)
    cs = _claims({"s0": {0: True}, "s1": {1: True}, "s2": {2: True}}, refs)
    est = triplet_accuracies(cs, min_overlap=3)
    assert set(est.uncovered) == {"s0", "s1", "s2"}
    assert all(est.accuracy[s] == est.default_accuracy for s in est.uncovered)


def test_triplet_recovers_accuracy_ordering():
    """With enough overlap the closed form should rank a reliable source above a noisy one."""
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.random(n) < 0.5
    accuracies = {"good": 0.95, "mid": 0.8, "poor": 0.62}
    refs = [AssertionRef.event_exists(f"e{i}") for i in range(n)]
    claims = [
        Claim(sid, refs[i], bool(truth[i]) if rng.random() < acc else bool(~truth[i]))
        for sid, acc in accuracies.items()
        for i in range(n)
    ]
    est = triplet_accuracies(ClaimSet(claims), min_overlap=50)
    assert est.accuracy["good"] > est.accuracy["mid"] > est.accuracy["poor"]
    for sid, acc in accuracies.items():
        assert est.accuracy[sid] == pytest.approx(acc, abs=0.08)


def test_triplet_weight_is_clipped():
    """An accuracy estimated from two triplets must not dominate the pool."""
    refs = _refs(2)
    est = triplet_accuracies(_claims({"s0": {0: True}}, refs))
    est.accuracy["s0"] = 0.99999
    assert abs(est.weight("s0", cap=4.0)) <= 4.0
