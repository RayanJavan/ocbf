"""Source interpretation interface, independent of registry execution."""

from typing import Protocol

from ocbf.evidence import AdmissionIssue, EvidenceRecord, Observation
from ocbf.universe.context import SemanticContext


class EvidenceInterpreter(Protocol):
    name: str
    version: str

    def interpret(
        self, record: EvidenceRecord, context: SemanticContext
    ) -> tuple[tuple[Observation, ...], tuple[AdmissionIssue, ...]]: ...
