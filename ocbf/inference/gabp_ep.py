"""Gaussian belief propagation with expectation-propagation sites -- the continuous engine.

Design doc section 4.3 commits to **one** algorithm for the whole continuous layer. A copula
warp, an interval censoring, a precedence truncation and a conditional-Gaussian mixture are
not four inference problems; they are four ways of computing a tilted moment, after which
every one of them projects back to a Gaussian and propagates. This module is that single
algorithm, and the four cases live in
[`ocbf.model.gaussian_banks`][ocbf.model.gaussian_banks] as banks it cannot tell apart.

The loop is deliberately the same shape as
[`ocbf.inference.loopy_bp`][ocbf.inference.loopy_bp] -- scatter-add, belief, cavity by
subtraction, per-bank update, damped write-back -- because it is the same algorithm over a
different message algebra. Log-potential rows become natural parameters; ``logsumexp``
becomes moment matching. Anyone who can read one engine can read the other.

The same hygiene applies, and for the same reasons: damping, oscillation detection, an
iteration cap that reports non-convergence rather than returning the last sweep, and
**convergence judged on beliefs rather than on messages** (design doc section 11.6). One
thing is easier here than in the discrete engine: the latent coordinates are standard normal
by construction of the copula, so a single tolerance is meaningful across a block mixing
timestamps in hours with prices in euros.

Two failure modes are specific to expectation propagation and are handled explicitly rather
than hoped away:

* **An improper cavity.** Removing a site's own contribution can leave non-positive
  precision. The update for that edge is skipped for the sweep rather than clipped, because
  clipping would invent evidence the graph does not contain.
* **A tilted distribution that underflows.** A claim far out in the cavity's tail can leave
  no quadrature mass. The bank contributes no site rather than a fabricated one, and the
  count is reported.
"""

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
    """Sweep cap, well above the discrete engine's -- and coupled to `degree_scaled_damping`.

    A step divided by a variable's degree needs proportionally more sweeps to accumulate its
    precision, and stopping early shows up not as an error but as intervals that are too
    wide, which is the quietest possible failure. The two settings are one decision; design
    record section 11.9 has the measurements. Affordable because the Gaussian block holds
    timestamps and attributes rather than links, so it is far smaller than the discrete graph.
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
    """Divide each edge's step by the number of sites on its variable.

    A synchronous sweep updates every site on a variable against a cavity that all the other
    sites are simultaneously moving, so a variable carrying ``d`` sites takes a step roughly
    ``d`` times too large. Dividing by the degree is the correction.

    What it buys is *not* a better answer -- both settings reach the same posterior. What
    differs is that the scaled residual falls monotonically while the unscaled one oscillates
    around the fixed point, so convergence cannot be told from wobble. Design doc section 11.6
    makes the residual the thing convergence is judged on, so a residual that reports nothing
    is the failure being avoided.
    """

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

        # Beliefs, not messages: the same finding as design doc section 11.6. Both moments
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
    """``log p(continuous evidence | discrete state)`` per coupled continuous variable.

    The exact, closed-form half of the conditional-Gaussian coupling (design doc
    section 4.3). Collected here rather than inside the engine because it is the *hybrid*
    loop's business: the Gaussian engine has no discrete variables, and giving it any would
    blur the split that makes both engines simple.

    Keyed by block position; the hybrid loop maps each to the discrete variable the bank was
    grounded against and attaches the row as a unary log-potential.
    """
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
