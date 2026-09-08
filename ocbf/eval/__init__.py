"""Evaluation against known ground truth.

Calibration metrics are first-class here, not an afterthought. Stage 1 section 3.2 found
that the payoff from fusion is in calibration rather than in sharper point predictions, so
a harness that reported only accuracy would be measuring the wrong thing and would happily
certify a system that is confidently wrong.

The same priority governs both halves. Binary assertions are scored with the Brier score and
expected calibration error; continuous ones with the CRPS and interval coverage, which are
the same two questions -- is the stated distribution proper, and does it mean what it says --
asked of a distribution over the real line.
"""

from ocbf.eval.continuous import (
    ContinuousReport,
    compare_continuous,
    evaluate_continuous,
    gaussian_crps,
    pit_values,
)
from ocbf.eval.metrics import BinaryReport, compare, evaluate_binary, reliability_diagram

__all__ = [
    "BinaryReport",
    "ContinuousReport",
    "compare",
    "compare_continuous",
    "evaluate_binary",
    "evaluate_continuous",
    "gaussian_crps",
    "pit_values",
    "reliability_diagram",
]
