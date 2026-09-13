"""Evidence and numerical diagnostics with distinct interpretations.

Source-overlap identifiability, source-evidence effective sample size and binary evidence
thresholds describe declared static-source assumptions. Monte Carlo diagnostics in
``diagnostics.monte_carlo`` assess sampled quantities. Neither kind certifies physical
provenance, posterior calibration or causal responsibility.
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
