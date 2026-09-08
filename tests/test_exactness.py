"""Verify the BP engine against brute-force enumeration.

The engine is an approximation on loopy graphs, so it needs an oracle it can be checked
against on graphs small enough to enumerate. This is design doc section 6.3's tier-1
argument in test form: a system whose core is approximate needs exact ground truth to
measure the approximation, and the cardinality forward-backward recursion in particular is
subtle enough that "it looked plausible" is not evidence.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ocbf.assertions import AssertionRef, VarKind, VariableRegistry
from ocbf.inference import BPConfig, run_bp
from ocbf.model.banks import CardinalityBank, PairwiseBank, UnaryBank
from ocbf.model.graph import NEG_INF, FactorGraph


def brute_force_marginals(graph: FactorGraph) -> np.ndarray:
    """Exact marginals by enumerating every discrete configuration.

    Reads the factors through
    [`FactorGraph.dense_factors`][ocbf.model.graph.FactorGraph.dense_factors] rather than by
    reaching into each bank, so the same enumeration works for any bank that can write
    itself out -- and so every test below is also a test of that enumeration, which the
    GTSAM oracle in `test_gtsam_oracle.py` depends on being right.
    """
    reg = graph.registry
    card = [int(c) for c in reg.cardinalities]
    domains = [range(c) for c in card]
    factors = list(graph.dense_factors())

    marg = np.zeros((len(reg), graph.max_card))
    log_probs: list[float] = []
    configs: list[tuple[int, ...]] = []

    for assignment in itertools.product(*domains):
        lp = sum(graph.log_prior[v, a] for v, a in enumerate(assignment))
        for var_ids, table in factors:
            lp += float(table[tuple(assignment[int(v)] for v in var_ids)])
        configs.append(assignment)
        log_probs.append(lp)

    lp = np.array(log_probs)
    lp -= lp.max()
    w = np.exp(lp)
    w /= w.sum()
    for assignment, weight in zip(configs, w, strict=True):
        for v, a in enumerate(assignment):
            marg[v, a] += weight
    return marg


def _registry(n_binary: int, categorical: int = 0) -> VariableRegistry:
    reg = VariableRegistry()
    for i in range(n_binary):
        reg.add_binary(AssertionRef.event_exists(f"e{i}"))
    if categorical:
        reg.add_categorical(AssertionRef.event_type("t0"), categorical)
    return reg.freeze()


def _prior(reg: VariableRegistry, width: int) -> np.ndarray:
    out = np.zeros((len(reg), width))
    for i in range(len(reg)):
        c = int(reg.cardinalities[i])
        out[i, c:] = NEG_INF
    return out


def test_unary_only_is_exact():
    reg = _registry(4)
    width = reg.max_cardinality
    rng = np.random.default_rng(0)
    pots = rng.normal(size=(4, width))
    graph = FactorGraph(
        reg, _prior(reg, width), [UnaryBank(np.arange(4), pots, name="channel")]
    )
    result = run_bp(graph, BPConfig(max_iter=50, damping=0.0))
    assert result.converged
    np.testing.assert_allclose(result.beliefs, brute_force_marginals(graph), atol=1e-8)


def test_pairwise_tree_is_exact():
    """BP is exact on trees. If it is not exact here, the message algebra is wrong."""
    reg = _registry(4)
    width = reg.max_cardinality
    rng = np.random.default_rng(1)
    tables = rng.normal(size=(1, 2, 2))
    graph = FactorGraph(
        reg,
        _prior(reg, width),
        [
            UnaryBank(np.arange(4), rng.normal(size=(4, width)), name="prior"),
            # A chain: 0-1, 1-2, 2-3. Acyclic, so sum-product is exact.
            PairwiseBank(
                np.array([0, 1, 2]),
                np.array([1, 2, 3]),
                tables,
                np.zeros(3, dtype=np.int64),
            ),
        ],
    )
    result = run_bp(graph, BPConfig(max_iter=200, damping=0.0, tol=1e-12))
    assert result.converged
    np.testing.assert_allclose(result.beliefs, brute_force_marginals(graph), atol=1e-7)


@pytest.mark.parametrize(
    ("k", "lo", "hi"),
    [(3, 1, 1), (4, 1, 1), (5, 1, None), (4, 0, 2), (6, 2, 3)],
)
def test_cardinality_single_group_is_exact(k: int, lo: int, hi: int | None):
    """A single counting factor makes a tree, so BP must reproduce enumeration exactly.

    This is the test that matters most: the ``O(k*C)`` forward-backward recursion replaces
    a ``2^k`` factor, and an error in it would be invisible in aggregate metrics while
    quietly corrupting every cardinality-constrained group in the model.
    """
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
    result = run_bp(graph, BPConfig(max_iter=300, damping=0.0, tol=1e-12))
    assert result.converged, f"BP did not converge (residual {result.residual})"
    np.testing.assert_allclose(result.beliefs, brute_force_marginals(graph), atol=1e-6)


def test_hard_implication_forbids_configuration():
    """A hard factor must drive the forbidden corner to numerically exact zero."""
    reg = _registry(2)
    width = reg.max_cardinality
    table = np.array([[[0.0, 0.0], [NEG_INF, 0.0]]])
    graph = FactorGraph(
        reg,
        _prior(reg, width),
        [
            # Push hard toward link=1 and endpoint=0, which the factor forbids.
            UnaryBank(
                np.array([0, 1]),
                np.array([[0.0, 4.0], [4.0, 0.0]]),
                name="channel",
            ),
            PairwiseBank(
                np.array([0]), np.array([1]), table, np.zeros(1, dtype=np.int64)
            ),
        ],
    )
    result = run_bp(graph, BPConfig(max_iter=200, damping=0.0, tol=1e-12))
    exact = brute_force_marginals(graph)
    np.testing.assert_allclose(result.beliefs, exact, atol=1e-7)

    # P(link=1, endpoint=0) == 0, so the evidence must be overridden somewhere.
    p_link = result.beliefs[0, 1]
    p_endpoint = result.beliefs[1, 1]
    assert p_link + (1 - p_endpoint) <= 1.0 + 1e-6


def test_categorical_gate_couples_type_and_link():
    """Evidence about a link must move belief about the event type, and vice versa.

    This is the structural propagation the whole design rests on: if the type gate did not
    transmit belief in both directions, the model would be parallel voting with extra steps.
    """
    reg = VariableRegistry()
    reg.add_categorical(AssertionRef.event_type("e0"), 3)
    reg.add_binary(AssertionRef.e2o("e0", "q", "o0"))
    reg.freeze()
    width = reg.max_cardinality

    # Only type 2 permits the link.
    table = np.full((1, width, 2), NEG_INF)
    table[0, :3, 0] = 0.0
    table[0, :3, 1] = [NEG_INF, NEG_INF, 0.0]

    graph = FactorGraph(
        reg,
        _prior(reg, width),
        [
            UnaryBank(np.array([1]), np.array([[0.0, 3.0]]), name="channel"),
            PairwiseBank(
                np.array([0]), np.array([1]), table, np.zeros(1, dtype=np.int64)
            ),
        ],
    )
    result = run_bp(graph, BPConfig(max_iter=200, damping=0.0, tol=1e-12))
    np.testing.assert_allclose(result.beliefs, brute_force_marginals(graph), atol=1e-7)

    # A source claiming the link should concentrate the type posterior on type 2.
    type_post = result.beliefs[0, :3]
    assert type_post[2] > 0.8, f"link evidence did not propagate to type: {type_post}"
