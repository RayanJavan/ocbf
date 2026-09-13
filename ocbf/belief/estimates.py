"""Scientific estimates distinguish posterior, computation, sensitivity and coverage."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ocbf._values import freeze


@dataclass(frozen=True)
class Estimate:
    query_id: str
    name: str
    quantity: str
    value: float | None
    unit: str
    status: str
    denominator: float | None
    computation: str
    outcomes: Mapping = field(default_factory=dict)
    qualifications: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping = field(default_factory=dict)
    distribution: Mapping = field(default_factory=dict)
    numerical: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "outcomes", freeze(self.outcomes))
        object.__setattr__(self, "qualifications", tuple(self.qualifications))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "metadata", freeze(self.metadata))
        object.__setattr__(self, "distribution", freeze(self.distribution))
        object.__setattr__(self, "numerical", freeze(self.numerical))


@dataclass(frozen=True)
class QueryResults:
    model_id: str
    run_id: str
    bundle_id: str
    estimates: tuple[Estimate, ...]
    execution: object = field(default=None, metadata={"identity_omit_default": True})

    def __post_init__(self):
        object.__setattr__(self, "estimates", tuple(self.estimates))
