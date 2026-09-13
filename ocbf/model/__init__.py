"""Canonical model construction and reusable numerical representations.

Use ``model.spec`` and ``model.compile`` to build the scientific target from interpreted
evidence and resolved parameters. Discrete graphs, Gaussian graphs, factor banks and copula
transforms are numerical utilities used by message-passing engines and their validation.
Their local marginal views do not imply a canonical joint-history capability.
"""

from ocbf.model.banks import MAX_DENSE_GROUP, CardinalityBank, PairwiseBank, UnaryBank
from ocbf.model.copula import CopulaSpec, MarginalTransform, fit_copula
from ocbf.model.gaussian import MIN_PRECISION, GaussianBank, GaussianGraph
from ocbf.model.graph import NEG_INF, DenseFactors, FactorBank, FactorGraph

__all__ = [
    "CardinalityBank",
    "CopulaSpec",
    "DenseFactors",
    "FactorBank",
    "FactorGraph",
    "GaussianBank",
    "GaussianGraph",
    "MAX_DENSE_GROUP",
    "MIN_PRECISION",
    "MarginalTransform",
    "NEG_INF",
    "PairwiseBank",
    "UnaryBank",
    "fit_copula",
]
