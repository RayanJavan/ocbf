"""Fixed object/event types, qualified relations and constraint declarations.

Schema values define semantic meanings. Candidate instances and interpreted evidence are
supplied separately. Classify constraints explicitly as support, descriptive or normative.
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
