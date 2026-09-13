"""OCBF — Object-Centric Belief Fusion.

Build explicit probability models from versioned evidence under a fixed semantic context,
then evaluate process questions through declared posterior capabilities. The application
facade lives in ``ocbf.api``; semantic types are also exported here. Source interpretation,
manual parameters, model compilation, inference and query evaluation have separate owners.
"""

__version__ = "0.1.0"

from ocbf.assertions import AssertionRef, Family, VarKind, VariableRegistry
from ocbf.schema import (
    AttributeKind,
    AttributeSpec,
    E2OQualifier,
    EventType,
    Multiplicity,
    O2OQualifier,
    ObjectType,
    Schema,
    Strength,
)

__all__ = [
    "AssertionRef",
    "AttributeKind",
    "AttributeSpec",
    "E2OQualifier",
    "EventType",
    "Family",
    "Multiplicity",
    "O2OQualifier",
    "ObjectType",
    "Schema",
    "Strength",
    "VarKind",
    "VariableRegistry",
    "__version__",
]
