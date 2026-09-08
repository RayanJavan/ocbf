"""The factor graph: templates, banks, and assembly.

Design doc section 2.1 commits to a **parameterised factor graph**. Factor *templates* are
keyed by schema elements and ground against the universe, which buys two things at once:

1. **Parameter sharing is pooling.** All groundings of a template share parameters, so a
   type gets statistical strength from every one of its instances. This is Stage 1
   section 6.2's observation that lifting and statistical pooling are the same operation --
   and it is why a sparse-evidence regime is survivable at all.
2. **Vectorisation is free.** All groundings of a template have identical message shapes,
   so their messages are one batched array operation. That is what a [`FactorBank`][ocbf.model.graph.FactorBank]
   is: every factor of one template, stored columnar.

The package has two halves that mirror each other deliberately. The discrete backbone lives
in [`graph`][ocbf.model.graph], [`banks`][ocbf.model.banks] and [`build`][ocbf.model.build];
the continuous layer of design doc section 4 lives in
[`gaussian`][ocbf.model.gaussian], [`gaussian_banks`][ocbf.model.gaussian_banks],
[`continuous`][ocbf.model.continuous] and the [`copula`][ocbf.model.copula] subpackage.
Container, banks and assembly play the same roles on both sides; only the message algebra
differs.
"""

from ocbf.model.banks import MAX_DENSE_GROUP, CardinalityBank, PairwiseBank, UnaryBank
from ocbf.model.build import GraphSpec, build_graph
from ocbf.model.continuous import (
    ContinuousGrounding,
    ContinuousSpec,
    build_continuous_graph,
    continuous_refs,
    fit_copula_from_claims,
)
from ocbf.model.copula import CopulaSpec, MarginalTransform, fit_copula
from ocbf.model.gaussian import MIN_PRECISION, GaussianBank, GaussianGraph
from ocbf.model.graph import NEG_INF, DenseFactors, FactorBank, FactorGraph

__all__ = [
    "CardinalityBank",
    "ContinuousGrounding",
    "ContinuousSpec",
    "CopulaSpec",
    "DenseFactors",
    "FactorBank",
    "FactorGraph",
    "GaussianBank",
    "GaussianGraph",
    "GraphSpec",
    "MAX_DENSE_GROUP",
    "MIN_PRECISION",
    "MarginalTransform",
    "NEG_INF",
    "PairwiseBank",
    "UnaryBank",
    "build_continuous_graph",
    "build_graph",
    "continuous_refs",
    "fit_copula",
    "fit_copula_from_claims",
]
