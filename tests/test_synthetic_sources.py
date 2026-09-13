"""Object-centric synthetic source and candidate validation."""

import pytest
from ocbf.assertions import Family
from ocbf.sources import ClaimSet
from ocbf.synth import ProcessConfig, SourceRegime, simulate_process, simulate_sources

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
