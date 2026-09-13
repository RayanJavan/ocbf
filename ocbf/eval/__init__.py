"""Numerical evaluation against explicitly supplied reference truth.

Binary and continuous scores assess calibration and accuracy on the given reference.
Synthetic ground truth supports numerical experiments; agreement on synthetic fixtures
does not establish physical factory calibration.
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
