"""The GTSAM exact oracle, and the backend plumbing that makes it loadable.

Three claims are under test, and they are separable:

1. **The translation is faithful.** A `FactorGraph` written out through `dense_factors` and
   rebuilt as a GTSAM `DiscreteFactorGraph` must give the same posterior as enumerating the
   original by hand. Table order, log-to-linear scale, padded cardinalities and hard zeros
   are each a place two conventions could silently disagree, so each is exercised.
2. **The oracle earns its keep.** If it only worked where brute force already worked it
   would be redundant. `test_agrees_with_brute_force_on_a_loopy_grid` checks elimination
   against enumeration on a graph loopy enough for BP to be wrong on;
   `test_scales_past_enumeration` then runs one enumeration cannot reach at all.
3. **It refuses what it cannot do, by measurement.** Elimination has no iteration cap: it
   finishes or it exhausts memory. `test_the_clique_ceiling_catches_what_variable_count_misses`
   is the test that a 36-variable graph can be far more expensive than a 60-variable one,
   and that the guard notices.

The backend tests at the bottom do not need GTSAM and always run -- discovery and
idempotence are exactly the parts that must behave on a machine without it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ocbf.assertions import AssertionRef, VariableRegistry
from ocbf.backends import (
    CUDA_BIN_ENV,
    cuda_library_dirs,
    enable_cuda_library_search,
    gtsam_backend,
    import_gtsam,
    report,
)
from ocbf.inference import (
    BPConfig,
    ExactConfig,
    compare_to_exact,
    elimination_cost,
    exact_marginals,
    run_bp,
)
from ocbf.model.banks import MAX_DENSE_GROUP, CardinalityBank, PairwiseBank, UnaryBank
from ocbf.model.graph import NEG_INF, FactorGraph
from test_exactness import _prior, _registry, brute_force_marginals

requires_gtsam = pytest.mark.skipif(
    not gtsam_backend().available, reason=f"gtsam backend unavailable: {gtsam_backend().reason}"
)


def _counting_graph(n: int, groups: list[list[int]], seed: int) -> FactorGraph:
    """`n` binary links under exactly-one counting constraints over the given groups."""
    reg = VariableRegistry()
    for i in range(n):
        reg.add_binary(AssertionRef.e2o(f"e{i}", "q", f"o{i}"))
    reg.freeze()
    rng = np.random.default_rng(seed)
    return FactorGraph(
        reg,
        _prior(reg, reg.max_cardinality),
        [
            UnaryBank(np.arange(n), rng.normal(size=(n, reg.max_cardinality)), name="channel"),
            CardinalityBank(groups, [1] * len(groups), [1] * len(groups), weight=2.5),
        ],
    )


def _grid_graph(rows: int, cols: int, seed: int) -> FactorGraph:
    """A counting constraint along every row and column of a ``rows x cols`` block.

    Loopy, and badly conditioned for elimination: every variable sits in two groups and the
    fill-in compounds. Twenty variables more than the 4x4 used below costs four orders of
    magnitude more clique, which is what
    [`ExactConfig.max_clique_states`][ocbf.inference.gtsam_exact.ExactConfig] exists for.
    Keep the sizes here small and deliberate.
    """
    groups = [[r * cols + c for c in range(cols)] for r in range(rows)]
    groups += [[r * cols + c for r in range(rows)] for c in range(cols)]
    return _counting_graph(rows * cols, groups, seed)


def _ring_graph(n: int, window: int, seed: int) -> FactorGraph:
    """Overlapping windows of `window` consecutive links, closed into a ring.

    Loopy -- the ring and the overlaps both make cycles -- but of bounded treewidth, so
    elimination stays cheap however long the ring gets. The contrast with `_grid_graph` is
    the whole lesson: neither variable count nor loopiness predicts the cost. Fill-in does.
    """
    groups = [[(i + j) % n for j in range(window)] for i in range(n)]
    return _counting_graph(n, groups, seed)


# --------------------------------------------------------------------------------------
# The translation
# --------------------------------------------------------------------------------------


@requires_gtsam
def test_matches_brute_force_on_unary_evidence():
    reg = _registry(4)
    width = reg.max_cardinality
    rng = np.random.default_rng(0)
    graph = FactorGraph(
        reg, _prior(reg, width), [UnaryBank(np.arange(4), rng.normal(size=(4, width)), name="ch")]
    )
    np.testing.assert_allclose(
        exact_marginals(graph).beliefs, brute_force_marginals(graph), atol=1e-9
    )


@requires_gtsam
def test_matches_brute_force_on_a_pairwise_chain():
    reg = _registry(4)
    width = reg.max_cardinality
    rng = np.random.default_rng(1)
    graph = FactorGraph(
        reg,
        _prior(reg, width),
        [
            UnaryBank(np.arange(4), rng.normal(size=(4, width)), name="prior"),
            PairwiseBank(
                np.array([0, 1, 2]),
                np.array([1, 2, 3]),
                rng.normal(size=(1, 2, 2)),
                np.zeros(3, dtype=np.int64),
            ),
        ],
    )
    np.testing.assert_allclose(
        exact_marginals(graph).beliefs, brute_force_marginals(graph), atol=1e-9
    )


@requires_gtsam
@pytest.mark.parametrize(("k", "lo", "hi"), [(3, 1, 1), (4, 1, 1), (5, 1, None), (6, 2, 3)])
def test_matches_brute_force_on_a_cardinality_group(k: int, lo: int, hi: int | None):
    """The ``2^k`` table `dense_factors` builds must agree with the recursion it replaces."""
    reg = _registry(k)
    width = reg.max_cardinality
    rng = np.random.default_rng(k * 17 + lo)
    graph = FactorGraph(
        reg,
        _prior(reg, width),
        [
            UnaryBank(np.arange(k), rng.normal(size=(k, width)), name="channel"),
            CardinalityBank([list(range(k))], [lo], [hi], weight=1.7),
        ],
    )
    np.testing.assert_allclose(
        exact_marginals(graph).beliefs, brute_force_marginals(graph), atol=1e-9
    )


@requires_gtsam
def test_mixed_cardinalities_and_hard_zeros_survive_the_round_trip():
    """A categorical gated against a binary link: unequal domains *and* forbidden corners.

    Padding and hard zeros are the two ways the translation could quietly go wrong -- the
    first by offering GTSAM a state that does not exist, the second by turning
    [`NEG_INF`][ocbf.model.graph.NEG_INF] into a small positive potential instead of a zero.
    """
    reg = VariableRegistry()
    type_ref = AssertionRef.event_type("e0")
    link_ref = AssertionRef.e2o("e0", "q", "o0")
    reg.add_categorical(type_ref, 3)
    reg.add_binary(link_ref)
    reg.freeze()
    width = reg.max_cardinality
    type_var, link_var = reg.index(type_ref), reg.index(link_ref)

    table = np.full((1, width, 2), NEG_INF)
    table[0, :3, 0] = 0.0
    table[0, :3, 1] = [NEG_INF, NEG_INF, 0.0]  # only type 2 permits the link

    graph = FactorGraph(
        reg,
        _prior(reg, width),
        [
            UnaryBank(np.array([link_var]), np.array([[0.0, 3.0]]), name="channel"),
            PairwiseBank(
                np.array([type_var]), np.array([link_var]), table, np.zeros(1, dtype=np.int64)
            ),
        ],
    )
    result = exact_marginals(graph)
    np.testing.assert_allclose(result.beliefs, brute_force_marginals(graph), atol=1e-9)

    # Evidence for the link reaches the type through the gate -- the forbidden corner is
    # unreachable jointly, not marginally, so it is the concentration that shows it worked.
    assert result.beliefs[type_var, :3].argmax() == 2
    assert result.beliefs[type_var, 2] > 0.8
    assert result.map_assignment is not None
    assert result.map_assignment[type_var] == 2
    assert result.map_assignment[link_var] == 1


@requires_gtsam
def test_padded_states_stay_impossible():
    """A binary in a graph four states wide must keep zero mass on the padding."""
    reg = VariableRegistry()
    type_ref = AssertionRef.event_type("e0")
    exists_ref = AssertionRef.event_exists("e1")
    reg.add_categorical(type_ref, 4)
    reg.add_binary(exists_ref)
    reg.freeze()

    graph = FactorGraph(reg, _prior(reg, reg.max_cardinality), [])
    beliefs = exact_marginals(graph).beliefs
    assert beliefs[reg.index(exists_ref), 2:].max() == pytest.approx(0.0)
    np.testing.assert_allclose(beliefs[reg.index(type_ref)], 0.25, atol=1e-12)


# --------------------------------------------------------------------------------------
# What the oracle is for
# --------------------------------------------------------------------------------------


@requires_gtsam
def test_bp_is_exact_on_a_tree():
    """A sanity floor. If BP and the oracle disagree here, one of them is broken."""
    reg = _registry(5)
    width = reg.max_cardinality
    rng = np.random.default_rng(7)
    graph = FactorGraph(
        reg,
        _prior(reg, width),
        [
            UnaryBank(np.arange(5), rng.normal(size=(5, width)), name="channel"),
            CardinalityBank([[0, 1, 2, 3, 4]], [1], [1], weight=2.0),
        ],
    )
    bp = run_bp(graph, BPConfig(max_iter=400, damping=0.0, tol=1e-13))
    assert bp.converged
    gap = compare_to_exact(graph, bp)
    assert gap.n_compared == 5
    assert gap.max_abs_error < 1e-9, gap.summary()
    assert gap.max_kl < 1e-15, gap.summary()


@requires_gtsam
def test_agrees_with_brute_force_on_a_loopy_grid():
    """Elimination against enumeration where BP is genuinely an approximation.

    A 4x4 block is 65 536 configurations -- about as much as enumeration can be asked for in
    a test, and already well past a tree. The two oracles must agree, and BP must not.
    """
    graph = _grid_graph(4, 4, seed=3)
    exact = exact_marginals(graph)
    np.testing.assert_allclose(exact.beliefs, brute_force_marginals(graph), atol=1e-9)

    gap = compare_to_exact(graph, run_bp(graph, BPConfig(max_iter=500)))
    assert gap.max_abs_error > 1e-6, (
        f"BP was exact on a loopy graph, which is suspicious: {gap.summary()}"
    )
    assert gap.max_abs_error < 0.5, gap.summary()


@requires_gtsam
def test_scales_past_enumeration():
    """The whole point: a graph brute force cannot touch, solved exactly anyway.

    60 binary variables is 2**60 configurations. Elimination pays for fill-in instead, and a
    ring of overlapping windows has a bounded amount of it however long the ring gets -- so
    this returns in a fraction of a second, and the BP gap it measures is a number that was
    simply not obtainable before.
    """
    graph = _ring_graph(60, window=4, seed=11)
    assert 2**graph.n_vars > 1e18  # what enumeration would have had to visit
    assert elimination_cost(graph).largest_clique_states <= 256

    exact = exact_marginals(graph)
    assert exact.log_beliefs.shape == (60, graph.max_card)
    np.testing.assert_allclose(np.exp(exact.log_beliefs[:, :2]).sum(axis=1), 1.0, atol=1e-9)

    gap = compare_to_exact(graph, run_bp(graph, BPConfig(max_iter=500)))
    assert np.isfinite(gap.max_abs_error) and np.isfinite(gap.max_kl)
    assert gap.n_compared == 60
    assert gap.worst_ref is not None


@requires_gtsam
def test_comparison_accepts_a_raw_belief_array():
    graph = _grid_graph(3, 3, seed=5)
    gap = compare_to_exact(graph, exact_marginals(graph).beliefs)
    assert gap.max_abs_error < 1e-12
    assert set(gap.diagnostics()) == {
        "oracle_max_abs_error",
        "oracle_mean_abs_error",
        "oracle_max_kl",
        "oracle_worst_variable",
    }


@requires_gtsam
def test_result_reports_its_backend_and_cost():
    graph = _grid_graph(3, 3, seed=5)
    result = exact_marginals(graph)
    diagnostics = result.diagnostics()
    assert diagnostics["exact_backend"].startswith("gtsam ")
    assert diagnostics["exact_factors"] > 0
    assert diagnostics["exact_seconds"] >= 0.0
    # The cost reported afterwards is the one that was checked beforehand.
    assert diagnostics["exact_largest_clique"] == result.cost.largest_clique_states
    assert result.cost.largest_clique_states == elimination_cost(graph).largest_clique_states


# --------------------------------------------------------------------------------------
# Refusing what it cannot do
# --------------------------------------------------------------------------------------


@requires_gtsam
def test_the_clique_ceiling_catches_what_variable_count_misses():
    """The guard must fire on fill-in, because fill-in is what kills the process.

    A 6x6 counting block is 36 variables -- fewer than the 60-variable ring that solves in a
    fraction of a second -- and needs a clique of ~1.7e7 states, which is gigabytes.
    Refusing it by measurement rather than by variable count is the difference between an
    error message and an out-of-memory kill.
    """
    cheap = elimination_cost(_ring_graph(60, window=4, seed=1))
    expensive = elimination_cost(_grid_graph(6, 6, seed=1))
    assert expensive.largest_clique_states > 1_000 * cheap.largest_clique_states

    with pytest.raises(ValueError, match="max_clique_states"):
        exact_marginals(_grid_graph(6, 6, seed=1))


@requires_gtsam
def test_variable_ceiling_is_reported_not_attempted():
    graph = _grid_graph(4, 4, seed=1)
    with pytest.raises(ValueError, match="max_variables"):
        exact_marginals(graph, ExactConfig(max_variables=4))


@requires_gtsam
def test_table_ceiling_is_reported_not_attempted():
    graph = _grid_graph(4, 4, seed=1)
    with pytest.raises(ValueError, match="max_table_states"):
        exact_marginals(graph, ExactConfig(max_table_states=4))


@requires_gtsam
def test_a_discrete_factor_on_a_continuous_variable_is_an_error():
    reg = VariableRegistry()
    exists_ref = AssertionRef.event_exists("e0")
    amount_ref = AssertionRef.event_attr("e0", "amount")
    reg.add_binary(exists_ref)
    reg.add_continuous(amount_ref)
    reg.freeze()
    width = reg.max_cardinality
    bad = np.array([reg.index(amount_ref)])
    graph = FactorGraph(
        reg, _prior(reg, width), [UnaryBank(bad, np.zeros((1, width)), name="bad")]
    )
    with pytest.raises(ValueError, match="continuous"):
        exact_marginals(graph)


def test_a_bank_that_cannot_enumerate_itself_says_so():
    """`dense_factors` is opt-in, so refusing must name the bank and the way out."""

    class Opaque:
        name = "opaque"

        def edge_vars(self):
            return np.array([0], dtype=np.int64)

        def factor_to_var(self, incoming):
            return incoming

        def n_factors(self):
            return 1

    reg = _registry(1)
    graph = FactorGraph(reg, _prior(reg, reg.max_cardinality), [Opaque()])
    with pytest.raises(TypeError, match="dense_factors"):
        list(graph.dense_factors())


def test_an_oversized_cardinality_group_refuses_to_allocate():
    k = MAX_DENSE_GROUP + 1
    reg = _registry(k)
    graph = FactorGraph(
        reg,
        _prior(reg, reg.max_cardinality),
        [CardinalityBank([list(range(k))], [1], [1], weight=1.0)],
    )
    with pytest.raises(ValueError, match="ceiling"):
        list(graph.dense_factors())


# --------------------------------------------------------------------------------------
# The backend plumbing
# --------------------------------------------------------------------------------------


def test_cuda_search_is_idempotent():
    first = enable_cuda_library_search()
    assert enable_cuda_library_search() == first


def test_the_override_replaces_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv(CUDA_BIN_ENV, f"{tmp_path}{os.pathsep}{tmp_path / 'absent'}")
    assert cuda_library_dirs() == (tmp_path,)


def test_discovery_only_returns_directories_that_exist():
    assert all(path.is_dir() for path in cuda_library_dirs())


def test_report_is_printable_either_way():
    text = report()
    assert "gtsam" in text and "cuda" in text


@requires_gtsam
def test_backend_describes_a_working_install():
    backend = gtsam_backend()
    assert backend.version
    assert backend.reason is None
    assert import_gtsam() is backend.module
    # cuda_built is a property of the build, devices are a property of the machine. Neither
    # is asserted -- a CPU-only GTSAM is a perfectly good oracle -- but they must agree.
    assert backend.cuda_usable == (backend.cuda_built and bool(backend.devices))
