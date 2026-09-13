"""Numerical engines and message-passing utilities.

The canonical workflow uses capability-based planning and explicit policies through
``ocbf.api``. Finite elimination, bounded hybrid integration and blocked sampling provide
different posterior capabilities. BP provides restricted approximate finite marginals;
caller-grounded EP provides Gaussian marginal moments. Convergence alone is not accuracy.

This package also exports low-level BP/EP kernels and the discrete graph oracle used to
validate them. The canonical GTSAM engine is implemented in ``inference.adapters.exact``.
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
