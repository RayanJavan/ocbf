"""Inference engines.

Design doc section 6.3 tiers these. Tier 2 -- loopy BP over the discrete backbone -- is the
only mandatory one, and it is the *theoretically indicated* engine rather than merely a
convenient one: sparse claim graphs are locally tree-like, and belief propagation exactly
matches the fundamental limit for Dawid-Skene in that regime (arXiv:1602.03619,
arXiv:1110.3564). Sampling belongs in the small parameter block, not over millions of
assertion variables.

BP hygiene is non-negotiable and implemented as such: damping, message-residual monitoring,
oscillation detection, and an iteration cap that **reports non-convergence** rather than
silently returning whatever the last sweep produced.

Tier 2 has two halves, and they are deliberately the same algorithm over different message
algebras: [`loopy_bp`][ocbf.inference.loopy_bp] carries log-potential rows over the discrete
backbone, and [`gabp_ep`][ocbf.inference.gabp_ep] carries natural parameters over the
Gaussian block, projecting back to a Gaussian by moment matching wherever a factor is not
Gaussian. Design doc section 4.3's claim is that the whole continuous layer needs only that
one mechanism, so there is one continuous engine and not four.

Tier 1 -- exact elimination through GTSAM -- is here too, as
[`gtsam_exact`][ocbf.inference.gtsam_exact]. It is optional and never on the pipeline's
path: its only job is to measure how far tier 2's approximation sits from exact on graphs
small enough to solve both ways.
"""

from ocbf.inference.gtsam_exact import (
    EliminationCost,
    ExactConfig,
    ExactResult,
    OracleComparison,
    compare_to_exact,
    elimination_cost,
    exact_marginals,
    to_discrete_factor_graph,
)
from ocbf.inference.gabp_ep import (
    EPConfig,
    EPResult,
    cavities,
    cg_discrete_potentials,
    run_ep,
)
from ocbf.inference.loopy_bp import BPConfig, BPResult, run_bp, to_belief_state

__all__ = [
    "BPConfig",
    "BPResult",
    "EPConfig",
    "EPResult",
    "EliminationCost",
    "ExactConfig",
    "ExactResult",
    "OracleComparison",
    "cavities",
    "cg_discrete_potentials",
    "compare_to_exact",
    "elimination_cost",
    "exact_marginals",
    "run_bp",
    "run_ep",
    "to_belief_state",
    "to_discrete_factor_graph",
]
