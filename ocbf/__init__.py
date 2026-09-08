"""OCBF -- Object-Centric Belief Fusion.

A joint probabilistic belief over a latent OCEL 2.0 log, fused from many sparse,
individually unreliable, structurally-typed sources.

Sources are sparse, weak and numerous, and that regime — not the OCEL format — drives the
design. Most assertions cannot be decided by voting at all, so the schema's structure
carries the inference; and most sources make too few claims to be assessed individually, so
their reliability is pooled.

The documentation is organised as Getting started, How-to guides, Explanation and API
reference. `docs/explanation/research-notes.md` holds the literature and formal grounding;
`docs/explanation/design-record.md` holds the committed model and the findings that revised
it.
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
