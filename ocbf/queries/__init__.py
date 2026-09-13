"""Generic execution projections and queries over declared posterior capabilities."""

from .evaluate import evaluate, requirements_for
from .expressions import ConditionInterval, Endpoint, ExecutionProjection, QueryBundle, QuerySpec

__all__ = [
    "ConditionInterval",
    "Endpoint",
    "ExecutionProjection",
    "QueryBundle",
    "QuerySpec",
    "evaluate",
    "requirements_for",
]
