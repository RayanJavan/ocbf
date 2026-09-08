"""Loopy belief propagation over the discrete backbone."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef, Family
from ocbf.belief import BeliefState, BeliefStateBuilder
from ocbf.model.banks import UnaryBank
from ocbf.mathx import logsumexp
from ocbf.model.graph import NEG_INF, FactorGraph


@dataclass(slots=True)
class BPConfig:
    """Engine settings.

    ``damping`` defaults high because the graph is loopy by construction: cardinality
    factors couple every link of an event, and referential integrity couples those links
    back through shared endpoints. Undamped BP oscillates on exactly that structure.
    """

    max_iter: int = 200
    damping: float = 0.8
    tol: float = 1e-5
    oscillation_window: int = 12
    track_attribution: bool = True


@dataclass(slots=True)
class BPResult:
    """Beliefs plus an honest account of how they were obtained."""

    log_beliefs: np.ndarray
    messages: np.ndarray
    converged: bool
    iterations: int
    residual: float
    belief_residual: float
    oscillating: bool
    residual_history: list[float] = field(default_factory=list)
    belief_history: list[float] = field(default_factory=list)

    @property
    def beliefs(self) -> np.ndarray:
        return np.exp(self.log_beliefs)

    def diagnostics(self) -> dict[str, object]:
        """Convergence status, iteration count, and both residuals."""
        return {
            "bp_converged": self.converged,
            "bp_iterations": self.iterations,
            "bp_msg_residual": round(float(self.residual), 8),
            "bp_belief_residual": round(float(self.belief_residual), 8),
            "bp_oscillating": self.oscillating,
        }


def _normalise(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Row-normalise in log space, keeping invalid states at [`NEG_INF`][ocbf.model.graph.NEG_INF]."""
    masked = np.where(valid, x, NEG_INF)
    total = logsumexp(masked, axis=1, keepdims=True)
    total = np.where(np.isfinite(total), total, 0.0)
    return np.where(valid, masked - total, NEG_INF)


def _scatter_add(edge_var: np.ndarray, values: np.ndarray, n_vars: int) -> np.ndarray:
    """Sum edge messages into their variables.

    ``np.bincount`` per state rather than ``np.add.at``: the latter is an unbuffered ufunc
    call and is roughly two orders of magnitude slower at this scale, which matters because
    this runs once per BP iteration.
    """
    out = np.zeros((n_vars, values.shape[1]), dtype=np.float64)
    for k in range(values.shape[1]):
        out[:, k] = np.bincount(edge_var, weights=values[:, k], minlength=n_vars)
    return out


def run_bp(graph: FactorGraph, config: BPConfig | None = None) -> BPResult:
    """Sum-product BP with damping and convergence monitoring."""
    cfg = config or BPConfig()
    n_vars, width = graph.n_vars, graph.max_card
    edge_var = graph.edge_var
    n_edges = graph.n_edges

    card = graph.registry.cardinalities
    valid_var = np.arange(width)[None, :] < card[:, None]
    valid_edge = valid_var[edge_var]

    messages = np.zeros((n_edges, width), dtype=np.float64)
    messages = np.where(valid_edge, messages, NEG_INF)

    log_beliefs = _normalise(graph.log_prior, valid_var)
    residual = np.inf
    belief_residual = np.inf
    history: list[float] = []
    belief_history: list[float] = []
    converged = False
    iterations = 0

    for iterations in range(1, cfg.max_iter + 1):
        acc = _scatter_add(edge_var, np.where(valid_edge, messages, 0.0), n_vars)
        unnormalised = graph.log_prior + acc
        new_beliefs = _normalise(unnormalised, valid_var)

        # Convergence is judged on *beliefs*, not on raw messages. With hard factors the
        # message residual is dominated by edges swinging between finite values and the
        # NEG_INF sentinel -- a swing that says nothing about whether the posterior has
        # settled. The beliefs are what callers consume, so they are what must converge.
        prev = np.exp(log_beliefs)
        belief_residual = float(np.abs(np.exp(new_beliefs) - prev).max())
        belief_history.append(belief_residual)
        log_beliefs = new_beliefs

        # var -> factor: the belief with this edge's own contribution removed. Subtract
        # from the *unnormalised* belief -- the per-variable constant would cancel in the
        # re-normalisation anyway, and skipping it avoids rescaling rows whose states are
        # all near-impossible.
        incoming = _normalise(unnormalised[edge_var] - messages, valid_edge)

        updated = np.empty_like(messages)
        for i, bank in enumerate(graph.banks):
            s = graph.bank_slice(i)
            updated[s] = bank.factor_to_var(incoming[s])
        updated = _normalise(updated, valid_edge)

        finite = valid_edge & np.isfinite(updated) & np.isfinite(messages)
        residual = float(np.abs(updated[finite] - messages[finite]).max()) if finite.any() else 0.0
        history.append(residual)

        messages = np.where(
            valid_edge, cfg.damping * messages + (1.0 - cfg.damping) * updated, NEG_INF
        )

        if iterations > 1 and belief_residual < cfg.tol:
            converged = True
            break

    acc = _scatter_add(edge_var, np.where(valid_edge, messages, 0.0), n_vars)
    log_beliefs = _normalise(graph.log_prior + acc, valid_var)

    return BPResult(
        log_beliefs=log_beliefs,
        messages=messages,
        converged=converged,
        iterations=iterations,
        residual=residual,
        belief_residual=belief_residual,
        oscillating=_is_oscillating(belief_history, cfg.oscillation_window),
        residual_history=history,
        belief_history=belief_history,
    )


def _is_oscillating(history: list[float], window: int) -> bool:
    """Detect a residual that has stopped falling without reaching tolerance.

    A flat or rising tail means BP is cycling rather than converging. Saying so is the
    difference between a reported approximation and a silent one.
    """
    if len(history) < 2 * window:
        return False
    recent = np.array(history[-window:])
    earlier = np.array(history[-2 * window : -window])
    return bool(recent.mean() >= earlier.mean() * 0.98)


def fill_belief_state(
    builder: BeliefStateBuilder, graph: FactorGraph, result: BPResult
) -> BeliefStateBuilder:
    """Write the discrete backbone's posteriors into a belief-state builder.

    Separate from sealing the state because a hybrid run has two engines writing into one
    builder and neither of them owns the result;
    [`ocbf.inference.gabp_ep.fill_belief_state`][ocbf.inference.gabp_ep.fill_belief_state]
    is its mirror image over the Gaussian block.

    Attribution is extracted here and costs nothing extra: for a binary assertion the
    posterior log-odds decomposes additively over incoming messages, so each unary edge's
    log-ratio *is* that source's contribution (design doc section 6.5). It is the retained
    messages, not a second computation.
    """
    reg = graph.registry
    probs = np.exp(result.log_beliefs)

    card = reg.cardinalities
    for i in range(len(reg)):
        c = int(card[i])
        if c == 0:
            continue
        row = probs[i, :c]
        total = row.sum()
        builder.set_discrete(i, row / total if total > 0 else np.full(c, 1.0 / c))

    contributions: dict[int, dict[str, float]] = defaultdict(dict)
    for i, bank in enumerate(graph.banks):
        if not isinstance(bank, UnaryBank):
            continue
        s = graph.bank_slice(i)
        msgs = result.messages[s]
        var_ids = bank.edge_vars()
        labels = bank.labels
        for e in range(len(var_ids)):
            idx = int(var_ids[e])
            if int(card[idx]) != 2:
                continue
            label = labels[e] if e < len(labels) else bank.name
            key = f"{bank.name}:{label}" if labels else bank.name
            delta = float(msgs[e, 1] - msgs[e, 0])
            if np.isfinite(delta):
                contributions[idx][key] = contributions[idx].get(key, 0.0) + delta

    for idx, contrib in contributions.items():
        builder.set_attribution(idx, contrib)
    return builder


def to_belief_state(
    graph: FactorGraph,
    result: BPResult,
    *,
    evidence: Mapping[AssertionRef, float] | None = None,
    extra_diagnostics: Mapping[str, object] | None = None,
) -> BeliefState:
    """Convert a discrete-only BP run into the public inference contract."""
    builder = BeliefStateBuilder(graph.registry)
    fill_belief_state(builder, graph, result)

    if evidence:
        for ref, nats in evidence.items():
            idx = graph.registry.get(ref)
            if idx is not None:
                builder.set_evidence(idx, nats)

    diagnostics = {"method": "loopy_bp", **result.diagnostics(), **dict(extra_diagnostics or {})}
    return builder.build(diagnostics=diagnostics)
