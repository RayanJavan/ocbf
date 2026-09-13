"""Snapshot ownership around existing semantic types; no solver or application imports."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ocbf._values import fingerprint, freeze
from ocbf.schema import ConstraintRegister, Schema
from ocbf.schema.constraints import ConstraintClass
from ocbf.universe.core import Universe


@dataclass(frozen=True)
class SemanticContext:
    """Immutable semantic and candidate snapshot made from the existing Universe API."""

    definition: Mapping
    grounding_provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "definition", freeze(self.definition))
        object.__setattr__(self, "grounding_provenance", freeze(self.grounding_provenance))

    @classmethod
    def from_universe(cls, universe: Universe, *, constraints=None, provenance=None):
        schema = universe.schema
        register = constraints or ConstraintRegister()
        return cls(
            {
                "event_types": tuple(sorted(schema.event_types.values(), key=lambda x: x.name)),
                "object_types": tuple(sorted(schema.object_types.values(), key=lambda x: x.name)),
                "e2o_qualifiers": tuple(
                    sorted(
                        schema.e2o_qualifiers,
                        key=lambda x: (x.event_type, x.qualifier, x.object_type),
                    )
                ),
                "o2o_qualifiers": tuple(
                    sorted(
                        schema.o2o_qualifiers,
                        key=lambda x: (x.source_type, x.qualifier, x.target_type),
                    )
                ),
                "lifecycles": tuple(
                    lc
                    for name in sorted(schema.object_type_names)
                    if (lc := schema.lifecycle(name)) is not None
                ),
                "constraints": tuple(register[c] for c in ConstraintClass),
                "events": tuple(sorted(universe.events.values(), key=lambda x: x.id)),
                "objects": tuple(sorted(universe.objects.values(), key=lambda x: x.id)),
                "e2o": tuple(sorted(universe.e2o_candidates)),
                "o2o": tuple(sorted(universe.o2o_candidates)),
            },
            provenance or {},
        )

    @property
    def context_id(self):
        return fingerprint("context", (self.definition, self.grounding_provenance))

    @property
    def schema(self) -> Schema:
        """A fresh compatibility view; mutating it cannot mutate the snapshot."""
        d = self.definition
        return Schema(
            event_types=d["event_types"],
            object_types=d["object_types"],
            e2o=d["e2o_qualifiers"],
            o2o=d["o2o_qualifiers"],
            lifecycles=d["lifecycles"],
        )
