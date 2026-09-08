"""OCEL 2.0 schema: the strictly-clamped part of the world.

Per Stage 2 decision 2 the schema is *definitional* -- it is never inferred. It supplies
the type system, the legal qualifier signatures, the attribute frames, the multiplicity
declarations and the lifecycle orderings, and it drives both candidate pruning
(``ocbf.universe``) and the hard factors (``ocbf.model.factors.hard_schema``).
"""

from ocbf.schema.constraints import ConstraintClass, ConstraintRegister, ConstraintSpec, Strength
from ocbf.schema.core import (
    AttributeKind,
    AttributeSpec,
    E2OQualifier,
    EventType,
    Lifecycle,
    Multiplicity,
    O2OQualifier,
    ObjectType,
    Schema,
)

__all__ = [
    "AttributeKind",
    "AttributeSpec",
    "ConstraintClass",
    "ConstraintRegister",
    "ConstraintSpec",
    "E2OQualifier",
    "EventType",
    "Lifecycle",
    "Multiplicity",
    "O2OQualifier",
    "ObjectType",
    "Schema",
    "Strength",
]
