"""Interpreted content describes observation meaning, not a backend's message arrays."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ocbf._values import fingerprint, freeze
from ocbf.errors import ValidationError

from .records import AdmissionIssue
from .snapshots import EvidenceSnapshot


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source_id: str
    family: str
    producer_version: str
    channel: str
    scope: tuple[str, ...]
    value: object
    evidence_ids: tuple[str, ...]
    interpretation: str
    information_id: str
    applicability: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not self.scope or len(set(self.scope)) != len(self.scope):
            raise ValidationError(
                "nonempty unique variable scope required", key=self.observation_id
            )
        if not self.evidence_ids or not self.information_id or not self.interpretation:
            raise ValidationError("evidence, information and interpretation identities required")
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "value", freeze(self.value))
        object.__setattr__(self, "applicability", freeze(self.applicability))


@dataclass(frozen=True)
class InterpretedEvidence:
    snapshot: EvidenceSnapshot
    observations: tuple[Observation, ...]
    issues: tuple[AdmissionIssue, ...] = ()
    manifest: Mapping = field(default_factory=dict)

    def __post_init__(self):
        obs = tuple(sorted(self.observations, key=lambda o: o.observation_id))
        if len({o.observation_id for o in obs}) != len(obs):
            raise ValidationError("duplicate observation identity")
        available = {r.revision_id for r in self.snapshot.effective}
        for o in obs:
            if not set(o.evidence_ids) <= available:
                raise ValidationError(
                    "observation references ineffective evidence", key=o.observation_id
                )
        object.__setattr__(self, "observations", obs)
        object.__setattr__(
            self, "issues", tuple(sorted(self.issues, key=lambda i: (i.key, i.status, i.reason)))
        )
        object.__setattr__(self, "manifest", freeze(self.manifest))

    @property
    def interpretation_id(self):
        return fingerprint(
            "interpretation", (self.snapshot.snapshot_id, self.observations, self.manifest)
        )
