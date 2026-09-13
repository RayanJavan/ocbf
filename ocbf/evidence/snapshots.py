"""Deterministic explicit-chain materialization; arrival order is never precedence."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from ocbf._values import canonical_json, fingerprint, utc
from ocbf.errors import EvidenceConflict, ValidationError

from .records import AdmissionIssue, EvidenceAction, EvidenceRecord


@dataclass(frozen=True)
class EvidenceSnapshot:
    records: tuple[EvidenceRecord, ...]
    effective: tuple[EvidenceRecord, ...]
    as_of: datetime | None
    policy: str
    issues: tuple[AdmissionIssue, ...]

    def __post_init__(self):
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "effective", tuple(self.effective))
        object.__setattr__(
            self, "issues", tuple(sorted(self.issues, key=lambda i: (i.key, i.status, i.reason)))
        )
        if self.as_of is not None:
            object.__setattr__(self, "as_of", utc(self.as_of))

    @property
    def snapshot_id(self):
        # Retransmission counts are diagnostic, not a changed scientific evidence state.
        return fingerprint("evidence", (self.records, self.as_of, self.policy))


def materialize(records, *, as_of, policy="explicit-chain-v1", control=None) -> EvidenceSnapshot:
    from ocbf.runtime.control import checkpoint

    checkpoint(control, "evidence.materialize")
    if policy not in ("explicit-chain-v1", "static-v1"):
        raise ValidationError("unsupported revision policy", key=policy)
    cutoff = utc(as_of) if as_of is not None else None
    if policy != "static-v1" and cutoff is None:
        raise ValidationError("a knowledge cutoff is required")
    unique = {}
    issues = []
    for r in records:
        checkpoint(control, "evidence.record", key=r.revision_id)
        if r.known_at is None and policy != "static-v1":
            raise ValidationError(
                "missing knowledge time; use explicit static policy", key=r.revision_id
            )
        if cutoff is not None and r.known_at is not None and r.known_at > cutoff:
            issues.append(AdmissionIssue(r.revision_id, "out_of_scope", "after knowledge cutoff"))
            continue
        if r.revision_id in unique:
            if canonical_json(unique[r.revision_id]) != canonical_json(r):
                raise EvidenceConflict(
                    "revision identity has conflicting contents", key=r.revision_id
                )
            issues.append(AdmissionIssue(r.revision_id, "duplicate", "idempotent retransmission"))
        unique[r.revision_id] = r
    groups = defaultdict(list)
    for r in unique.values():
        groups[(r.source_id, r.record_id)].append(r)
    effective = []
    for (_, record_id), group in sorted(groups.items()):
        roots = [r for r in group if r.previous_revision is None]
        if len(roots) != 1:
            raise EvidenceConflict("exactly one initial assertion is required", key=record_id)
        successors = {}
        for r in group:
            if r.previous_revision is not None:
                if r.previous_revision in successors:
                    raise EvidenceConflict(
                        "branching revisions require a source resolution rule", key=record_id
                    )
                parent = unique.get(r.previous_revision)
                if parent is None or (parent.source_id, parent.record_id) != (
                    r.source_id,
                    r.record_id,
                ):
                    raise EvidenceConflict("missing or foreign predecessor", key=r.revision_id)
                if parent.known_at and r.known_at and parent.known_at > r.known_at:
                    raise EvidenceConflict("revision predates its predecessor", key=r.revision_id)
                successors[r.previous_revision] = r
        current = roots[0]
        seen = {current.revision_id}
        while current.revision_id in successors:
            issues.append(AdmissionIssue(current.revision_id, "superseded", "explicit successor"))
            current = successors[current.revision_id]
            if current.revision_id in seen:
                raise EvidenceConflict("revision cycle", key=record_id)
            seen.add(current.revision_id)
        if len(seen) != len(group):
            raise EvidenceConflict("disconnected revision chain", key=record_id)
        if current.action is EvidenceAction.RETRACT:
            issues.append(AdmissionIssue(current.revision_id, "retracted", "explicit retraction"))
        else:
            effective.append(current)
    return EvidenceSnapshot(
        tuple(unique[k] for k in sorted(unique)),
        tuple(sorted(effective, key=lambda r: r.revision_id)),
        cutoff,
        policy,
        tuple(sorted(issues, key=lambda i: (i.key, i.status, i.reason))),
    )
