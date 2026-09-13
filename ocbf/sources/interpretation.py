"""Local, versioned interpreter registry; producer meaning never lives in the compiler."""

from ocbf.errors import ValidationError
from ocbf.evidence import AdmissionIssue, InterpretedEvidence
from ocbf.runtime.cache import cached, extension_key
from ocbf.runtime.control import checkpoint

from .contracts import EvidenceInterpreter

__all__ = ["EvidenceInterpreter", "interpret_snapshot"]


def interpret_snapshot(snapshot, context, interpreters, *, store=None, control=None):
    """Registry keys are exact (source_id, producer_version) pairs; no discovery."""
    observations, issues = [], list(snapshot.issues)
    used = {}
    for record in snapshot.effective:
        checkpoint(control, "interpret.record", key=record.revision_id)
        key = (record.source_id, record.producer_version)
        interpreter = interpreters.get(key)
        if interpreter is None:
            issues.append(
                AdmissionIssue(record.revision_id, "uninterpreted", "no versioned interpreter")
            )
            continue
        token = extension_key(interpreter)
        obs, report = cached(
            store if token else None,
            "interpret.record",
            (record, context.context_id, token),
            lambda interpreter=interpreter, record=record: interpreter.interpret(record, context),
            slot=record.record_id,
        )
        if any(record.revision_id not in o.evidence_ids for o in obs):
            raise ValidationError("interpreter lost input lineage", key=record.revision_id)
        observations.extend(obs)
        issues.extend(report)
        used["/".join(key)] = f"{interpreter.name}@{interpreter.version}"
    return InterpretedEvidence(
        snapshot,
        tuple(observations),
        tuple(issues),
        {"interpreters": used, "context_id": context.context_id},
    )
