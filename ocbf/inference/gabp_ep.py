"""Gaussian belief propagation with expectation-propagation sites.

Banks supply cavity-to-site updates; non-Gaussian sites use tilted moments. Damping, iteration limits and belief residuals assess numerical convergence. Caller-grounded EP returns marginal moments, not a general joint-history representation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef
from ocbf.belief import BeliefState, BeliefStateBuilder
from ocbf.model.copula import CopulaSpec
from ocbf.model.gaussian import (
    MIN_PRECISION,
    POTENTIAL,
    PRECISION,
    GaussianGraph,
    scatter_add,
    to_moments,
)
from ocbf.model.gaussian_banks import CGMeanBank


@dataclass(slots=True)
class EPConfig:
    """Engine settings.

    ``damping`` matches the discrete engine's default and for the same structural reason:
    the continuous block is loopy wherever the copula correlates a pair or a precedence
    factor couples two timestamps, and expectation propagation on a loopy graph oscillates
    undamped just as belief propagation does.
    """

    max_iter: int = 800
    """Maximum sweeps, interpreted together with ``degree_scaled_damping``.

    Scaling updates by degree can require more sweeps to accumulate precision. Inspect
    belief residuals and convergence status: an iteration cap alone does not establish
    accurate variances. Work and memory depend on the caller-grounded graph.
    """

    damping: float = 0.8
    tol: float = 1e-5
    """Convergence tolerance on the belief moments, in **latent** units.

    The same numeric value as the discrete engine's, and it means the same thing there and
    here for a non-obvious reason: the discrete residual is on probabilities in ``[0, 1]``,
    and the copula transform makes every continuous coordinate a standard normal, so both
    residuals are on quantities of order one. A tolerance in hours would not have that
    property, and would have to be chosen per deployment.
    """

    oscillation_window: int = 12

    degree_scaled_damping: bool = True
    """Divide each edge update by the number of sites on its variable. This can stabilize synchronous updates; it does not guarantee convergence or improve the target's physical accuracy."""

    track_attribution: bool = True


@dataclass(slots=True)
class EPResult:
    """Gaussian beliefs plus an honest account of how they were obtained.

    ``mean`` and ``var`` are in **latent** units -- the standardised copula coordinate, not
    the observed value. Converting to hours or euros is the marginal transform's job, and
    keeping the two apart is what stops a number in the wrong units from looking plausible.
    """

    mean: np.ndarray
    var: np.ndarray
    messages: np.ndarray
    converged: bool
    iterations: int
    residual: float
    belief_residual: float
    oscillating: bool
    skipped_cavities: int
    residual_history: list[float] = field(default_factory=list)
    belief_history: list[float] = field(default_factory=list)

    def diagnostics(self) -> dict[str, object]:
        """Convergence status, iteration count, both residuals, and skipped updates."""
        return {
            "ep_converged": self.converged,
            "ep_iterations": self.iterations,
            "ep_msg_residual": round(float(self.residual), 8),
            "ep_belief_residual": round(float(self.belief_residual), 8),
            "ep_oscillating": self.oscillating,
            "ep_skipped_cavities": self.skipped_cavities,
        }


def run_ep(graph: GaussianGraph, config: EPConfig | None = None) -> EPResult:
    """Expectation propagation over the Gaussian block, with damping and monitoring."""
    cfg = config or EPConfig()
    n_vars, n_edges = graph.n_vars, graph.n_edges
    edge_var = graph.edge_var

    messages = np.zeros((n_edges, 2), dtype=np.float64)
    degree = (
        np.bincount(edge_var, minlength=n_vars)[edge_var].astype(np.float64)
        if cfg.degree_scaled_damping and n_edges
        else np.ones(n_edges)
    )
    step = ((1.0 - cfg.damping) / np.maximum(degree, 1.0))[:, None]

    mean, var = to_moments(graph.prior)
    residual = np.inf
    belief_residual = np.inf
    history: list[float] = []
    belief_history: list[float] = []
    skipped = 0
    converged = False
    iterations = 0

    for iterations in range(1, cfg.max_iter + 1):
        natural = graph.prior + scatter_add(edge_var, messages, n_vars)
        new_mean, new_var = to_moments(natural)

        # Convergence uses belief residuals. Both moments
        # count -- a mean that has settled while the variance is still moving is not a
        # converged posterior. The standard deviation is compared rather than the variance
        # so both terms are in the same units as the tolerance.
        belief_residual = float(
            max(
                np.abs(new_mean - mean).max(initial=0.0),
                np.abs(np.sqrt(new_var) - np.sqrt(var)).max(initial=0.0),
            )
        )
        belief_history.append(belief_residual)
        mean, var = new_mean, new_var

        cavity = natural[edge_var] - messages
        improper = cavity[:, PRECISION] <= MIN_PRECISION
        skipped = int(improper.sum())
        cavity = np.where(
            improper[:, None], natural[edge_var], cavity
        )

        updated = np.empty_like(messages)
        for i, bank in enumerate(graph.banks):
            s = graph.bank_slice(i)
            updated[s] = bank.factor_to_var(cavity[s])
        updated = np.where(improper[:, None], messages, updated)
        updated = np.where(np.isfinite(updated), updated, messages)

        residual = float(np.abs(updated - messages).max(initial=0.0))
        history.append(residual)
        messages = messages + step * (updated - messages)

        if iterations > 1 and belief_residual < cfg.tol:
            converged = True
            break

    natural = graph.prior + scatter_add(edge_var, messages, n_vars)
    mean, var = to_moments(natural)

    return EPResult(
        mean=mean,
        var=var,
        messages=messages,
        converged=converged,
        iterations=iterations,
        residual=residual,
        belief_residual=belief_residual,
        oscillating=_is_oscillating(belief_history, cfg.oscillation_window),
        skipped_cavities=skipped,
        residual_history=history,
        belief_history=belief_history,
    )


def _is_oscillating(history: list[float], window: int) -> bool:
    """Detect a residual that has stopped falling without reaching tolerance.

    Identical in form to the discrete engine's test, and kept identical on purpose: the
    question -- "is this cycling rather than converging?" -- is the same question, and two
    different answers to it would be a trap for anyone reading both reports.
    """
    if len(history) < 2 * window:
        return False
    recent = np.array(history[-window:])
    earlier = np.array(history[-2 * window : -window])
    return bool(recent.mean() >= earlier.mean() * 0.98)


def cavities(graph: GaussianGraph, result: EPResult) -> np.ndarray:
    """Per-edge cavity at the returned solution.

    The cavity, not the belief, is what a factor's outgoing message must be computed
    against, so any consumer of a converged run -- the conditional-Gaussian message to the
    discrete graph, an oracle check, a diagnostic -- needs this rather than the marginals.
    """
    natural = graph.prior + scatter_add(graph.edge_var, result.messages, graph.n_vars)
    return natural[graph.edge_var] - result.messages


def cg_discrete_potentials(
    graph: GaussianGraph, result: EPResult
) -> dict[int, np.ndarray]:
    """Compute the conditional-Gaussian log evidence for each coupled discrete state.

    This local analytic message preserves state-dependent integration constants; coupling it to an approximate Gaussian belief does not establish global exact inference."""
    cavity = cavities(graph, result)
    out: dict[int, np.ndarray] = {}
    for i, bank in enumerate(graph.banks):
        if not isinstance(bank, CGMeanBank):
            continue
        rows = bank.discrete_potentials(cavity[graph.bank_slice(i)])
        for e, position in enumerate(bank.edge_vars()):
            out[int(position)] = rows[e]
    return out


def fill_belief_state(
    builder: BeliefStateBuilder,
    graph: GaussianGraph,
    result: EPResult,
    copula: CopulaSpec | None = None,
    *,
    track_attribution: bool = True,
) -> BeliefStateBuilder:
    """Write the Gaussian block's posteriors into a belief-state builder.

    Separate from sealing the belief state because a hybrid run has two engines writing into
    one builder, and neither of them owns the result. The discrete engine's
    [`fill_belief_state`][ocbf.inference.loopy_bp.fill_belief_state] is its mirror image.

    Attribution is the exact continuous analogue of the discrete decomposition and costs
    nothing extra for the same reason -- it is the retained messages. Where a binary
    assertion's log-odds decompose additively over incoming messages, a Gaussian's *mean*
    decomposes additively over incoming potentials:

    ```text
    mean(z)  =  eta_prior / lambda_total  +  sum over sites f of  eta_f / lambda_total
    ```

    The terms sum to the posterior mean exactly, so an unexpected timestamp can be traced to
    the source, the bracket or the precedence factor that moved it.
    """
    natural = graph.prior + scatter_add(graph.edge_var, result.messages, graph.n_vars)
    total_precision = np.maximum(natural[:, PRECISION], MIN_PRECISION)

    mixtures = _mixtures(graph, result)
    for position, var_id in enumerate(graph.var_ids):
        ref = graph.registry.ref(int(var_id))
        builder.set_continuous(
            int(var_id),
            float(result.mean[position]),
            float(result.var[position]),
            marginal=copula.marginal(ref) if copula is not None and copula.has(ref) else None,
            mixture=mixtures.get(position),
            prior_only=bool(
                natural[position, PRECISION]
                <= graph.prior[position, PRECISION] + MIN_PRECISION
            ),
        )

    if not track_attribution:
        return builder

    contributions: dict[int, dict[str, float]] = defaultdict(dict)
    for position in range(graph.n_vars):
        contributions[position]["prior"] = float(
            graph.prior[position, POTENTIAL] / total_precision[position]
        )
    for i, bank in enumerate(graph.banks):
        s = graph.bank_slice(i)
        msgs = result.messages[s]
        labels = getattr(bank, "labels", ())
        for e, position in enumerate(bank.edge_vars()):
            p = int(position)
            label = labels[e] if e < len(labels) else bank.name
            key = f"{bank.name}:{label}" if labels else bank.name
            delta = float(msgs[e, POTENTIAL] / total_precision[p])
            if np.isfinite(delta):
                contributions[p][key] = contributions[p].get(key, 0.0) + delta

    for position, contrib in contributions.items():
        builder.set_attribution(int(graph.var_ids[position]), contrib)
    return builder


def _mixtures(
    graph: GaussianGraph, result: EPResult
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Uncollapsed conditional-Gaussian components, for the variables that have them."""
    banks = [(i, b) for i, b in enumerate(graph.banks) if isinstance(b, CGMeanBank)]
    if not banks:
        return {}
    cavity = cavities(graph, result)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, bank in banks:
        weights, means, variances = bank.mixture(cavity[graph.bank_slice(i)])
        for e, position in enumerate(bank.edge_vars()):
            out[int(position)] = (weights[e], means[e], variances[e])
    return out


def to_belief_state(
    graph: GaussianGraph,
    result: EPResult,
    copula: CopulaSpec | None = None,
    *,
    evidence: Mapping[AssertionRef, float] | None = None,
    extra_diagnostics: Mapping[str, object] | None = None,
) -> BeliefState:
    """Convert a continuous-only run into the public inference contract.

    The counterpart of [`ocbf.inference.loopy_bp.to_belief_state`][ocbf.inference.loopy_bp.to_belief_state]
    for a graph with no discrete part -- a copula fitted on its own, or a unit test. A hybrid
    run uses `fill_belief_state` from both engines instead, so that one belief state carries
    both halves of the world.
    """
    builder = BeliefStateBuilder(graph.registry)
    fill_belief_state(builder, graph, result, copula)
    if evidence:
        for ref, nats in evidence.items():
            idx = graph.registry.get(ref)
            if idx is not None:
                builder.set_evidence(idx, nats)
    diagnostics = {
        "method": "gabp_ep", **result.diagnostics(), **dict(extra_diagnostics or {})
    }
    return builder.build(diagnostics=diagnostics)
