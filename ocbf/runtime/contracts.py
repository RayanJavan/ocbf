"""Small execution ports and neutral values; no facade, engines or evaluators."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ocbf._values import freeze
from ocbf.errors import ValidationError


class ArtifactStore(Protocol):
    """Stores owned immutable values. A miss or refused insertion is always harmless."""

    def get(self, namespace: str, key: str, *, slot: str | None = None): ...
    def put(self, namespace: str, key: str, value, *, slot: str | None = None) -> bool: ...
    @property
    def stats(self) -> Mapping: ...


@dataclass(frozen=True)
class ExecutionAssessment:
    status: str = "complete"
    stage: str = ""
    elapsed_seconds: float = 0.0
    resources: Mapping = field(default_factory=dict)
    reuse: Mapping = field(default_factory=dict)
    details: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("complete", "cancelled", "resource-exhausted", "failed"):
            raise ValidationError("invalid execution status")
        for name in ("resources", "reuse", "details"):
            object.__setattr__(self, name, freeze(getattr(self, name)))


@dataclass(frozen=True)
class DependencyIndex:
    """Operational sidecar: adding dependencies does not alter scientific Model ID."""

    structure_id: str
    support_id: str | None
    numerics: Mapping
    parents: Mapping
    tokens: Mapping
    version: str = "1"

    def __post_init__(self):
        for name in ("numerics", "parents", "tokens"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def changed(self, previous):
        return self.changed_values(previous.numerics)

    def changed_values(self, previous_numerics):
        keys = set(self.numerics) | set(previous_numerics)
        return tuple(sorted(k for k in keys if self.numerics.get(k) != previous_numerics.get(k)))
