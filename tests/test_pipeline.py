"""End-to-end behaviour, including the comparisons design doc section 8.3 makes mandatory.

These are slow relative to the unit tests because they run the real pipeline on a real
generated world. That is the point: the claims being tested here -- that the model beats
weighted voting, that the decidability flag means something, that a misspecified soft
constraint degrades gracefully where a hard one would not -- cannot be checked on a toy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ocbf.assertions import AssertionRef, Family
from ocbf.baselines import majority_vote, weighted_vote
from ocbf.eval import evaluate_binary
from ocbf.inference import BPConfig, run_bp, to_belief_state
from ocbf.model import GraphSpec, build_graph
from ocbf.pipeline import FusionConfig, fuse
from ocbf.reliability import triplet_accuracies
from ocbf.reliability.params import ReliabilityTable
from ocbf.schema import ConstraintClass, ConstraintRegister, Schema, Strength
from ocbf.schema.core import ONE
from ocbf.sources import ClaimSet
from ocbf.synth import ProcessConfig, SourceRegime, simulate_process, simulate_sources
from ocbf.synth.process import GroundTruth

pytestmark = pytest.mark.filterwarnings("ignore")

BINARY_FAMILIES = (Family.E2O, Family.O2O, Family.EVENT_EXISTS)


@pytest.fixture(scope="module")
def world():
    gt = simulate_process(ProcessConfig(n_orders=18, seed=5))
    universe = gt.build_universe()
    sim = simulate_sources(universe, gt, SourceRegime(n_sources=250, n_hotspots=8, seed=5))
    sources = {s.profile.source_id: s for s in sim.sources}
    claims = ClaimSet.from_sources(sim.sources)
    refs = [r for r in claims.refs if r.family in BINARY_FAMILIES]
    return gt, universe, sim, sources, claims, refs


def _bp_belief(universe, claims, sources, spec=None, register=None):
    rel = ReliabilityTable.from_triplet(triplet_accuracies(claims))
    graph = build_graph(
        universe, claims, rel, sources=sources, spec=spec, register=register
    )
    return to_belief_state(graph, run_bp(graph, BPConfig(max_iter=120, damping=0.8)))


# -- the generated world is a fair problem ---------------------------------------------


def test_generated_world_is_object_centric(world):
    """Events shared across objects are what make the log non-flattenable."""
    gt = world[0]
    ship_links = [k for k in gt.e2o if gt.events[k[0]].event_type == "Ship"]
    per_event: dict[str, int] = {}
    for event_id, _q, _o in ship_links:
        per_event[event_id] = per_event.get(event_id, 0) + 1
    assert max(per_event.values()) >= 3, "no Ship event links an order and several items"


def test_regime_is_sparse_and_weak(world):
    """Guard the benchmark against silently becoming easy."""
    claims = world[4]
    summary = claims.summary()
    assert summary["density"] < 0.05
    assert summary["deg_a_median"] <= 3


def test_universe_contains_decoys_and_type_uncertainty(world):
    gt, universe = world[0], world[1]
    assert len(universe.events) > len(gt.events), "no decoy events: existence would be known"
    supports = [len(universe.event_type_domain(e)) for e in universe.events]
    assert max(supports) > 1, "no type uncertainty: T_e would be determined"


# -- the mandatory comparison ----------------------------------------------------------


def test_beats_weighted_vote(world):
    """Design doc section 8.3: a fusion system that does not clear this is not working."""
    gt, universe, _sim, sources, claims, refs = world
    truth = gt.truth_map(refs)

    baseline = evaluate_binary(weighted_vote(universe.registry, claims, sources=sources), truth, refs)
    model = evaluate_binary(_bp_belief(universe, claims, sources), truth, refs)

    assert model.auc > baseline.auc, f"AUC {model.auc:.4f} vs baseline {baseline.auc:.4f}"
    assert model.brier < baseline.brier
    assert model.ece < baseline.ece, "calibration is where fusion is supposed to pay"


def test_beats_majority_vote(world):
    gt, universe, _sim, sources, claims, refs = world
    truth = gt.truth_map(refs)
    baseline = evaluate_binary(majority_vote(universe.registry, claims), truth, refs)
    model = evaluate_binary(_bp_belief(universe, claims, sources), truth, refs)
    assert model.auc > baseline.auc


# -- structure is doing the work -------------------------------------------------------


def test_structural_factors_improve_on_channels_alone(world):
    """Strip the structure and the model should degrade toward per-assertion voting.

    This is the ablation behind Stage 1 section 4.2's claim that the structural prior --
    not the reliability model -- is what carries a sparse regime.
    """
    gt, universe, _sim, sources, claims, refs = world
    truth = gt.truth_map(refs)

    bare = GraphSpec(
        include_referential_integrity=False,
        include_type_gate=False,
        include_cardinality=False,
    )
    without = evaluate_binary(_bp_belief(universe, claims, sources, spec=bare), truth, refs)
    with_structure = evaluate_binary(_bp_belief(universe, claims, sources), truth, refs)
    assert with_structure.auc > without.auc


def test_decidability_flag_predicts_accuracy_for_vote_only_methods(world):
    """The flag separates callable assertions from uncallable ones *for voting*.

    The bound it implements is about evidence carried by source votes, so it is against
    vote-only aggregation that it must hold. If accuracy on the decidable subset were not
    materially higher here, the flag would be miscalibrated and worse than useless.
    """
    gt, universe, _sim, sources, claims, refs = world
    result = fuse(
        universe,
        sources,
        FusionConfig(outer_iterations=1, fit_reliability=False, bp=BPConfig(max_iter=60)),
    )
    truth = gt.truth_map(refs)
    voted = weighted_vote(universe.registry, claims, sources=sources).with_evidence(
        result.decidability.evidence
    )
    report = evaluate_binary(voted, truth, refs)
    assert report.n_decidable > 0
    assert report.accuracy_decidable > report.accuracy + 0.03


def test_structure_resolves_assertions_that_votes_cannot(world):
    """The empirical form of Stage 1 section 4.2's central claim.

    Decidability counts only the Chernoff information carried by source votes. The
    structured model additionally propagates belief along referential integrity, the type
    gate and cardinality, so it can resolve assertions that *no* aggregation rule could
    decide from votes alone. This test asserts exactly that gap, and it is the reason the
    structural prior is described as primary rather than as a refinement.

    A corollary worth stating: for the structured model the decidability flag is a *lower
    bound* on what is resolvable, not a prediction of its accuracy.
    """
    gt, universe, _sim, sources, claims, refs = world
    result = fuse(
        universe,
        sources,
        FusionConfig(outer_iterations=1, fit_reliability=False, bp=BPConfig(max_iter=120)),
    )
    truth = gt.truth_map(refs)
    undecidable = [r for r in refs if not result.decidability.is_decidable(r)]
    assert undecidable, "fixture has no undecidable assertions to test"

    structured = evaluate_binary(result.belief, truth, undecidable)
    voted = evaluate_binary(
        weighted_vote(universe.registry, claims, sources=sources), truth, undecidable
    )
    assert structured.auc > voted.auc
    assert structured.accuracy > voted.accuracy


# -- robustness ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dense_world():
    """A world with enough source density that the affected links carry real evidence.

    The shared ``world`` fixture is deliberately sparse, which is right for most tests but
    leaves only a single claimed ``Ship``/``item`` link -- no basis for measuring anything.
    The soft-versus-hard question is specifically about whether *evidence can overrule a
    false constraint*, so it needs assertions that have evidence.
    """
    gt = simulate_process(ProcessConfig(n_orders=40, items_per_order=(3, 5), seed=5))
    universe = gt.build_universe()
    sim = simulate_sources(
        universe,
        gt,
        SourceRegime(n_sources=1500, n_hotspots=6, accuracy_mean=0.75, seed=5),
    )
    sources = {s.profile.source_id: s for s in sim.sources}
    return gt, universe, sources, ClaimSet.from_sources(sim.sources)


def test_soft_beats_hard_when_the_constraint_is_wrong(dense_world):
    """Design doc section 8.3's sensitivity experiment: the justification for decision 6.

    A genuinely *false* schema belief is injected -- ``Ship`` is declared to take exactly one
    item, when in this process it takes several. Note that merely raising the weight on a
    correct constraint is not misspecification; it makes the model better, which is why the
    wrong multiplicity has to be asserted outright.

    The claim under test is precise: a soft constraint can be **overruled by evidence**, a
    hard one cannot. So the measurement is taken on the affected links that actually carry
    claims, and against a reference run with the constraint switched off entirely. Soft
    should land near the reference; hard should collapse well below it.
    """
    gt, _universe, sources, claims = dense_world

    wrong_schema = Schema(
        event_types=list(gt.schema.event_types.values()),
        object_types=list(gt.schema.object_types.values()),
        e2o=[
            replace(q, multiplicity=ONE)
            if (q.event_type, q.qualifier) == ("Ship", "item")
            else q
            for q in gt.schema.e2o_qualifiers
        ],
        o2o=list(gt.schema.o2o_qualifiers),
        lifecycles=[
            gt.schema.lifecycle(t) for t in gt.schema.object_types if gt.schema.lifecycle(t)
        ],
    )
    wrong_universe = GroundTruth(
        wrong_schema, gt.events, gt.object_types, gt.e2o, gt.o2o, gt.config
    ).build_universe()

    claimed = set(claims.refs)
    affected = [
        AssertionRef.e2o(e, q, o)
        for (e, q, o) in gt.e2o
        if q == "item"
        and gt.events[e].event_type == "Ship"
        and AssertionRef.e2o(e, q, o) in claimed
        and wrong_universe.registry.get(AssertionRef.e2o(e, q, o)) is not None
    ]
    assert len(affected) >= 5, f"only {len(affected)} claimed Ship/item links: underpowered"

    def mean_posterior(**kwargs) -> float:
        belief = _bp_belief(wrong_universe, claims, sources, **kwargs)
        return float(np.mean([belief.prob_true(r) for r in affected]))

    reference = mean_posterior(spec=GraphSpec(include_cardinality=False))
    soft = mean_posterior(
        register=ConstraintRegister().override(ConstraintClass.CARDINALITY, weight=1.2)
    )
    hard = mean_posterior(
        register=ConstraintRegister().override(ConstraintClass.CARDINALITY, weight=500.0)
    )

    assert soft > 0.8 * reference, (
        f"soft {soft:.4f} fell far below the constraint-free reference {reference:.4f}; "
        "evidence should still win against a false but soft constraint"
    )
    assert hard < 0.5 * soft, (
        f"hard {hard:.4f} did not collapse relative to soft {soft:.4f}; the experiment "
        "cannot distinguish the two regimes and so proves nothing about decision 6"
    )


def test_prior_only_assertions_answer_with_the_prior(world):
    """Graceful degradation: an unknown ref returns the prior, not an exception."""
    from ocbf.assertions import AssertionRef

    universe, sources, claims = world[1], world[3], world[4]
    belief = _bp_belief(universe, claims, sources)
    unknown = AssertionRef.e2o("nonexistent_event", "q", "nonexistent_object")
    assert belief.prob_true(unknown, default=0.25) == 0.25
    assert belief.is_prior_only(unknown)
    assert belief.attribution(unknown) == {}


def test_attribution_recovers_contributing_sources(world):
    """Attribution should name the sources that actually moved a posterior."""
    universe, sources, claims, refs = world[1], world[3], world[4], world[5]
    belief = _bp_belief(universe, claims, sources)
    covered = [r for r in refs if claims.deg_assertion(r) >= 2]
    assert covered, "fixture produced no multiply-covered assertion"

    ref = covered[0]
    contributors = belief.attribution(ref)
    assert contributors
    claimed_by = claims.sources_covering(ref)
    named = {k.split(":", 1)[1] for k in contributors if k.startswith("channel:")}
    assert named & claimed_by, "attribution named none of the claiming sources"


# -- the full two-block loop ------------------------------------------------------------


def test_fuse_returns_the_full_contract(world):
    universe, sources = world[1], world[3]
    result = fuse(
        universe,
        sources,
        FusionConfig(outer_iterations=1, fit_reliability=False, bp=BPConfig(max_iter=60)),
    )
    assert result.belief is not None
    assert result.overlap.n_sources > 0
    assert result.ess.ess
    assert result.decidability.evidence
    assert len(result.history) == 1
    assert "Not globally identifiable" in result.report() or result.overlap.is_globally_identifiable


@pytest.mark.slow
def test_hierarchical_fit_improves_reliability_recovery(world):
    """Stage 1 decision 3: pooling should beat the label-free initialiser.

    Also guards the variance-collapse pathology -- a MAP fit that drives every hierarchical
    scale to zero reports a clean result while silently disabling the partial pooling.
    """
    _gt, universe, sim, sources, claims, _refs = world
    result = fuse(
        universe,
        sources,
        FusionConfig(outer_iterations=2, bp=BPConfig(max_iter=100, damping=0.8)),
    )

    triplet = triplet_accuracies(claims)
    common = [s for s in sim.truth if s in triplet.accuracy]
    true = np.array([sim.truth[s].sensitivity for s in common])
    init = np.array([triplet.accuracy[s] for s in common])
    fitted = np.array([result.reliability[s].sensitivity for s in common])

    assert np.abs(fitted - true).mean() < np.abs(init - true).mean()

    scales = result.history[-1]["reliability"]
    assert scales["sigma_individual"] > 1e-3, "hierarchical variance collapsed to zero"
    assert scales["sigma_cluster"] > 1e-3
