"""Monotone marginal transforms and Gaussian dependence utilities.

Transforms map admitted continuous, ordinal or count marginals to standardized latent
coordinates. Unordered categories are excluded. Caller-grounded Gaussian inference needs
these transforms to translate latent marginal summaries back to observed units.
"""

from ocbf.model.copula.bridge import (
    MAX_ABS_CORRELATION,
    bridge_correlation,
    bridge_curve,
    kendall_tau,
)
from ocbf.model.copula.marginals import (
    EmpiricalMarginal,
    GaussianMarginal,
    MarginalTransform,
    OrdinalMarginal,
    TruncatedMarginal,
    fit_marginal,
    truncated_normal_moments,
)
from ocbf.model.copula.structure import (
    EVENT_TIME_KEY,
    CopulaSpec,
    coupling_precision,
    fit_copula,
    marginal_key,
)

__all__ = [
    "CopulaSpec",
    "EVENT_TIME_KEY",
    "EmpiricalMarginal",
    "GaussianMarginal",
    "MAX_ABS_CORRELATION",
    "MarginalTransform",
    "OrdinalMarginal",
    "TruncatedMarginal",
    "bridge_correlation",
    "bridge_curve",
    "coupling_precision",
    "fit_copula",
    "fit_marginal",
    "kendall_tau",
    "marginal_key",
    "truncated_normal_moments",
]
