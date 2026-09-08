"""The latent Gaussian copula: the transform half of the continuous layer.

Design doc section 4.1 models every *ordered* quantity -- timestamps, and continuous, count,
ordinal, binary and truncated attributes -- as a monotone image of a latent Gaussian. The
whole mixed-attribute problem then collapses into a single Gaussian block, and that block is
what [`ocbf.inference.gabp_ep`][ocbf.inference.gabp_ep] runs on.

Three pieces:

* [`marginals`][ocbf.model.copula.marginals] -- the per-quantity transform ``v = F^-1(Phi(z))``,
  with discrete kinds **interval-censored** rather than mapped to a point;
* [`bridge`][ocbf.model.copula.bridge] -- latent correlations from Kendall's tau. Rank-based
  because the sparse regime cannot afford a parametric mixed MRF's data appetite
  (research notes section 7.2);
* [`structure`][ocbf.model.copula.structure] -- which templates are in the block and where
  the latent precision is allowed to be non-zero.

The honest boundary is stated once and enforced by
[`AttributeKind.in_copula`][ocbf.schema.core.AttributeKind.in_copula]: **unordered
categoricals are not in this layer**. They have no monotone image in a Gaussian and stay
discrete (design doc section 4.2). That is a real limitation of the copula approach rather
than an implementation gap.
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
