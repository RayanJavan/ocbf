"""Immutable retained report envelopes and explicit revision actions."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ocbf._values import freeze, utc
from ocbf.errors import ValidationError


class EvidenceAction(str, Enum):
    ASSERT = "assert"
    REPLACE = "replace"
    RETRACT = "retract"
    CLOSE = "close"


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    revision_id: str
    source_id: str
    producer_version: str
    payload: Mapping
    known_at: datetime | None
    action: EvidenceAction = EvidenceAction.ASSERT
    previous_revision: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    provenance: Mapping = field(default_factory=dict)
    lineage: tuple[str, ...] = ()

    def __post_init__(self):
        if not all((self.record_id, self.revision_id, self.source_id, self.producer_version)):
            raise ValidationError("record, revision, source and producer version are required")
        object.__setattr__(self, "action", EvidenceAction(self.action))
        for name in ("known_at", "valid_from", "valid_to"):
            if (value := getattr(self, name)) is not None:
                object.__setattr__(self, name, utc(value))
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValidationError("valid_to precedes valid_from", key=self.revision_id)
        if (self.action is EvidenceAction.ASSERT) != (self.previous_revision is None):
            raise ValidationError(
                "assert has no predecessor; other actions require one", key=self.revision_id
            )
        object.__setattr__(self, "payload", freeze(self.payload))
        object.__setattr__(self, "provenance", freeze(self.provenance))
        object.__setattr__(self, "lineage", tuple(sorted(set(self.lineage))))


@dataclass(frozen=True)
class AdmissionIssue:
    key: str
    status: str
    reason: str
