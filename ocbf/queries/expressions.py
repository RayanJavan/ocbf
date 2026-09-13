"""Small portable expressions. Factory event names and fields belong in caller bindings."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from ocbf._values import fingerprint, freeze, utc
from ocbf.errors import ValidationError


@dataclass(frozen=True)
class Endpoint:
    """Occurrence and associations with either fixed reported time or a modeled time key."""

    event_key: str
    time: datetime | None
    association: tuple[tuple[str, object], ...] = ()
    time_key: str | None = field(default=None, metadata={"identity_omit_default": True})

    def __post_init__(self):
        if self.time is not None:
            object.__setattr__(self, "time", utc(self.time))
        if self.time is not None and self.time_key is not None:
            raise ValidationError("endpoint cannot have both fixed and modeled time")
        object.__setattr__(self, "association", freeze(self.association))

    @property
    def scope(self):
        return tuple(
            sorted(
                {
                    self.event_key,
                    *(k for k, _ in self.association),
                    *((self.time_key,) if self.time_key else ()),
                }
            )
        )

    def present(self, state):
        return bool(state[self.event_key]) and all(state[k] == v for k, v in self.association)


@dataclass(frozen=True)
class ExecutionProjection:
    execution_id: str
    population_id: str
    starts: tuple[Endpoint, ...]
    ends: tuple[Endpoint, ...]
    applicable_when: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        for name in ("starts", "ends", "applicable_when"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    @property
    def scope(self):
        return tuple(
            sorted(
                {
                    *(k for k, _ in self.applicable_when),
                    *(k for e in (*self.starts, *self.ends) for k in e.scope),
                }
            )
        )


@dataclass(frozen=True)
class ConditionInterval:
    start: datetime
    end: datetime | None
    present_when: tuple[tuple[str, object], ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "start", utc(self.start))
        if self.end is not None:
            object.__setattr__(self, "end", utc(self.end))
            if self.end < self.start:
                raise ValidationError("condition interval is reversed")
        object.__setattr__(self, "present_when", freeze(self.present_when))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


@dataclass(frozen=True)
class QuerySpec:
    name: str
    kind: str
    executions: tuple[ExecutionProjection, ...]
    window_start: datetime
    horizon: datetime
    reference: Mapping = field(default_factory=dict)
    conditions: tuple[ConditionInterval, ...] = ()
    coverage_verified: bool = False
    version: str = "1"

    def __post_init__(self):
        object.__setattr__(self, "executions", freeze(self.executions))
        object.__setattr__(self, "reference", freeze(self.reference))
        object.__setattr__(self, "conditions", freeze(self.conditions))
        object.__setattr__(self, "window_start", utc(self.window_start))
        object.__setattr__(self, "horizon", utc(self.horizon))
        if self.horizon <= self.window_start:
            raise ValidationError("query horizon must follow window start")
        ids = [e.execution_id for e in self.executions]
        if len(set(ids)) != len(ids):
            raise ValidationError("duplicate execution in query population")

    @property
    def query_id(self):
        return fingerprint("query", self)


@dataclass(frozen=True)
class QueryBundle:
    queries: tuple[QuerySpec, ...]

    def __post_init__(self):
        object.__setattr__(self, "queries", tuple(self.queries))
        if len({q.name for q in self.queries}) != len(self.queries):
            raise ValidationError("query names must be unique")

    @property
    def bundle_id(self):
        return fingerprint("bundle", self)
