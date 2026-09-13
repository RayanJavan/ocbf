"""Semantic assertion addresses and execution-local variable storage. Fixed assertion families distinguish existence, types, qualified links, times and attributes."""

from ocbf.assertions.refs import AssertionRef, Family
from ocbf.assertions.registry import VariableRegistry, VarKind

__all__ = ["AssertionRef", "Family", "VarKind", "VariableRegistry"]
