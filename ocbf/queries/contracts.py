"""Process query evaluator interface; no concrete evaluator dependencies."""

from typing import Protocol

from ocbf.belief.estimates import Estimate
from ocbf.belief.posterior import QueryRequirements


class QueryEvaluator(Protocol):
    version: str
    cooperative: bool

    def requirements(self, query) -> QueryRequirements: ...
    def evaluate(self, result, query, *, store=None, control=None) -> Estimate: ...


class JointQueryEvaluator(QueryEvaluator, Protocol):
    """Evaluate aligned state arrays with weights and numerical-assessment semantics."""

    def evaluate_on(self, result, query, data, *, store=None, control=None) -> Estimate: ...
