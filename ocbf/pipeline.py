"""The two-block alternating scheme -- the top-level entry point.

Design doc section 6.1, the central architectural commitment:

```text
initialise:  triplet accuracies -> weighted vote -> initial beliefs

repeat:
    BELIEF BLOCK      (huge, sparse, ~1e7 vars)
        given theta, run BP over the grounded factor graph -> marginals
    PARAMETER BLOCK   (small, dense, ~1e2-1e3 params)
        given marginals, fit the hierarchical reliability GLM
```

This is variational EM with a structured mean-field split. The justification is Stage 1
section 4.4: the assertion block is huge and sparse, which is exactly where belief
propagation is provably near-optimal; the parameter block is small, which is exactly where
real Bayesian inference is affordable and where the pooling lives. Running MCMC over ten
million assertion variables would be both slower *and* worse.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef
from ocbf.belief import BeliefState
from ocbf.diagnostics import (
    DecidabilityReport,
    ESSReport,
    OverlapReport,
    decidability,
    effective_sample_sizes,
    overlap_report,
)
from ocbf.belief.state import BeliefStateBuilder
from ocbf.inference import BPConfig, EPConfig, run_bp, run_ep, to_belief_state
from ocbf.inference.gabp_ep import cg_discrete_potentials
from ocbf.inference.gabp_ep import fill_belief_state as fill_continuous
from ocbf.inference.loopy_bp import fill_belief_state as fill_discrete
from ocbf.model import GraphSpec, build_graph
from ocbf.model.banks import UnaryBank
from ocbf.model.continuous import (
    ContinuousGrounding,
    ContinuousSpec,
    build_continuous_graph,
    fit_copula_from_claims,
)
from ocbf.model.continuous import continuous_refs as _continuous_refs
from ocbf.model.copula import marginal_key
from ocbf.model.graph import NEG_INF
from ocbf.reliability import pairwise_channels, triplet_accuracies
from ocbf.reliability.params import ReliabilityTable
from ocbf.schema import ConstraintRegister
from ocbf.sources import ClaimSet, Source
from ocbf.universe import Universe


@dataclass(slots=True)
class FusionConfig:
    """Settings for the whole pipeline."""

    outer_iterations: int = 3
    """Belief/parameter alternations. Two or three is usually enough; the first parameter
    fit does most of the work once the triplet initialiser has broken the symmetry."""

    bp: BPConfig = field(default_factory=BPConfig)
    graph: GraphSpec = field(default_factory=GraphSpec)
    constraints: ConstraintRegister = field(default_factory=ConstraintRegister)

    ep: EPConfig = field(default_factory=EPConfig)
    continuous: ContinuousSpec | None = field(default_factory=ContinuousSpec)
    """Settings for the continuous layer, or ``None`` to run the discrete backbone alone.

    Enabled by default and free when unused: a universe with no timestamps or attributes in
    play grounds an empty Gaussian block, the engine is skipped, and the answers are
    identical to a discrete-only run. Turning it off explicitly is for ablations -- design
    doc section 8.3 asks for exactly that comparison.
    """

    fit_reliability: bool = True
    """When ``False``, keep the triplet estimates and skip the parameter block. Useful as an
    ablation -- design doc section 8.3 asks for exactly this comparison."""

    reliability_method: str = "map"
    eps_target: float = 0.1
    min_overlap: int = 2
    seed: int = 0


@dataclass(slots=True)
class FusionResult:
    """Everything the inference contract of Stage 1 decision 8 promises, plus diagnostics."""

    belief: BeliefState
    reliability: ReliabilityTable
    overlap: OverlapReport
    ess: ESSReport
    decidability: DecidabilityReport
    claim_set: ClaimSet
    continuous: ContinuousGrounding | None = None
    history: list[dict[str, object]] = field(default_factory=list)

    def report(self) -> str:
        """Formatted summary: identifiability, ESS, decidability, and iteration history."""
        lines = [
            "=" * 72,
            "OCBF fusion report",
            "=" * 72,
            "",
            self.overlap.explain(),
            "",
            f"Effective sample size: {self.ess.summary()}",
            f"Decidability:          {self.decidability.summary()}",
            f"Reliability:           {self.reliability.summary()}",
        ]
        if self.continuous is not None and not self.continuous.is_empty:
            lines.append(f"Continuous layer:      {self.continuous.summary()}")
        lines += [
            "",
            "Outer iterations:",
        ]
        for h in self.history:
            lines.append(f"  {h}")
        return "\n".join(lines)


def fuse(
    universe: Universe,
    sources: Sequence[Source] | Mapping[str, Source],
    config: FusionConfig | None = None,
) -> FusionResult:
    """Run the two-block scheme to convergence and return the belief state.

    The diagnostics are computed and returned unconditionally, not on request. Stage 1
    decision 10 makes them core outputs: with sources this weak, a posterior without its
    identifiability verdict, its effective sample size and its decidability flags is not a
    result -- it is a number with unstated preconditions.
    """
    cfg = config or FusionConfig()
    source_map = (
        dict(sources) if isinstance(sources, Mapping) else {s.profile.source_id: s for s in sources}
    )
    claim_set = ClaimSet.from_sources(source_map.values())
    clusters = {sid: s.profile.cluster_id for sid, s in source_map.items()}

    # -- initialisation: label-free moments, then weighted vote -----------------------
    # Spectral/moment initialisation keeps the outer EM loop out of the bad local optima
    # that plague truth-discovery objectives (design doc section 6.2).
    triplet = triplet_accuracies(claim_set, min_overlap=cfg.min_overlap + 1)
    reliability = ReliabilityTable.from_triplet(triplet)

    overlap = overlap_report(claim_set, min_overlap=cfg.min_overlap)
    history: list[dict[str, object]] = []
    belief: BeliefState | None = None
    ess: ESSReport | None = None
    dec: DecidabilityReport | None = None

    # The continuous layer is grounded once and re-grounded each outer iteration, but its
    # copula is fitted once: marginals and latent correlations are properties of the world,
    # not of the current belief, so refitting them per iteration would let the model chase
    # its own posterior.
    continuous_spec = cfg.continuous
    in_block = frozenset(_continuous_refs(universe))
    copula = (
        fit_copula_from_claims(universe, claim_set, seed=cfg.seed)
        if continuous_spec is not None and in_block
        else None
    )
    channels = (
        pairwise_channels(
            claim_set, lambda ref: marginal_key(ref) if ref in in_block else None
        )
        if copula is not None
        else None
    )
    grounding: ContinuousGrounding | None = None
    feedback: list[UnaryBank] = []

    for outer in range(1, cfg.outer_iterations + 1):
        t0 = time.perf_counter()

        # -- belief block -------------------------------------------------------------
        graph = build_graph(
            universe,
            claim_set,
            reliability,
            sources=source_map,
            spec=cfg.graph,
            register=cfg.constraints,
            extra_banks=feedback,
        )
        bp = run_bp(graph, cfg.bp)

        ess = effective_sample_sizes(
            claim_set, clusters, reliability.sensitivities(), min_overlap=cfg.min_overlap + 1
        )
        dec = decidability(
            claim_set,
            reliability.sensitivities(),
            reliability.specificities(),
            eps=cfg.eps_target,
            default_accuracy=reliability.default.sensitivity,
            ess=ess.ess,
        )
        belief = to_belief_state(graph, bp, evidence=dec.evidence)

        entry: dict[str, object] = {
            "iteration": outer,
            **bp.diagnostics(),
            "graph_factors": sum(b.n_factors() for b in graph.banks),
            "decidable": dec.summary()["decidable"],
        }

        # -- continuous block ---------------------------------------------------------
        # Grounded against the discrete belief just computed, because the factors that need
        # it -- precedence and the conditional-Gaussian coupling -- ask which links and types
        # the discrete layer believes in. The exact, closed-form message back to the discrete
        # graph is carried into the next iteration as an ordinary unary bank.
        if copula is not None and continuous_spec is not None:
            grounding = build_continuous_graph(
                universe,
                claim_set,
                reliability,
                channels=channels,
                copula=copula,
                discrete=belief,
                spec=continuous_spec,
                register=cfg.constraints,
                seed=cfg.seed,
            )
            if not grounding.is_empty:
                ep = run_ep(grounding.graph, cfg.ep)
                belief = _merge_beliefs(graph, bp, grounding, ep, dec.evidence)
                feedback = _cg_feedback(graph, grounding, ep)
                entry.update(ep.diagnostics())
                entry["continuous_factors"] = sum(b.n_factors() for b in grounding.graph.banks)

        # -- parameter block ----------------------------------------------------------
        if cfg.fit_reliability:
            from ocbf.reliability.hierarchical import fit_hierarchical, soft_counts

            counts = soft_counts(claim_set, belief, source_map)
            fit = fit_hierarchical(counts, method=cfg.reliability_method, seed=cfg.seed + outer)
            reliability = fit.table
            entry["reliability"] = fit.summary()

        entry["seconds"] = round(time.perf_counter() - t0, 2)
        history.append(entry)

    assert belief is not None and ess is not None and dec is not None
    return FusionResult(
        belief=belief,
        reliability=reliability,
        overlap=overlap,
        ess=ess,
        decidability=dec,
        claim_set=claim_set,
        continuous=grounding,
        history=history,
    )


def _merge_beliefs(
    graph, bp, grounding: ContinuousGrounding, ep, evidence: Mapping[AssertionRef, float]
) -> BeliefState:
    """One belief state carrying both halves of the world.

    Two engines write into one builder rather than two belief states being stitched together
    afterwards. Stage 1 decision 8 makes the belief state *the* inference contract, and a
    caller asking about an event's existence and its timestamp should not need to know that
    two different algorithms produced the answers.
    """
    builder = BeliefStateBuilder(graph.registry)
    fill_discrete(builder, graph, bp)
    fill_continuous(builder, grounding.graph, ep, grounding.copula)
    for ref, nats in evidence.items():
        idx = graph.registry.get(ref)
        if idx is not None:
            builder.set_evidence(idx, nats)
    return builder.build(
        diagnostics={"method": "loopy_bp+gabp_ep", **bp.diagnostics(), **ep.diagnostics()}
    )


def _cg_feedback(graph, grounding: ContinuousGrounding, ep) -> list[UnaryBank]:
    """The continuous layer's exact message back to the discrete backbone.

    One unary log-potential per conditional-Gaussian-coupled discrete variable, carrying the
    Gaussian log-partition per state. This is the direction of design doc section 4.3 that is
    *exact*, which is worth stating plainly: the approximation in the coupling is entirely on
    the continuous side, where a mixture is collapsed.
    """
    potentials = cg_discrete_potentials(grounding.graph, ep)
    if not potentials:
        return []
    width = graph.max_card
    var_ids: list[int] = []
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for position, row in potentials.items():
        target = grounding.cg_targets.get(position)
        if target is None:
            continue
        padded = np.full(width, NEG_INF)
        padded[: min(width, len(row))] = row[:width]
        var_ids.append(int(target))
        rows.append(padded)
        labels.append(grounding.graph.registry.ref(int(target)).subject)
    if not var_ids:
        return []
    return [
        UnaryBank(
            np.array(var_ids, dtype=np.int64), np.stack(rows), name="cg_time", labels=labels
        )
    ]
