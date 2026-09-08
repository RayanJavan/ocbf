"""Diagnostics: results, not logging.

Stage 1 decision 10 makes these core outputs. Each answers a question that the sparse /
unreliable / numerous regime makes unavoidable, and each is a *precondition check* the
model would otherwise silently violate:

* [`overlap`][ocbf.diagnostics.overlap] -- **is source reliability even identifiable?** Skills
  are identifiable only when the source overlap graph is irreducible and contains an odd
  cycle. A bipartite component admits a global sign flip and cannot be resolved by any
  amount of data (Stage 1 section 4.1).
* [`decidability`][ocbf.diagnostics.decidability] -- **is this assertion decidable at all?** From the
  exact error exponent, deciding an assertion to error ``eps`` needs total Chernoff
  information above ``log(1/eps)``. Below that, no aggregation rule succeeds and the honest
  answer is ``UNDETERMINED`` (Stage 1 section 4.2).
* [`ess`][ocbf.diagnostics.ess] -- **how many independent votes is this really?** Twenty
  sources that share an upstream model are not twenty votes. The design effect makes that
  visible instead of fatal (Stage 1 section 4.6).
"""

from ocbf.diagnostics.decidability import DecidabilityReport, chernoff_binary, decidability
from ocbf.diagnostics.ess import ESSReport, effective_sample_sizes
from ocbf.diagnostics.overlap import Identifiability, OverlapReport, overlap_report

__all__ = [
    "DecidabilityReport",
    "ESSReport",
    "Identifiability",
    "OverlapReport",
    "chernoff_binary",
    "decidability",
    "effective_sample_sizes",
    "overlap_report",
]
