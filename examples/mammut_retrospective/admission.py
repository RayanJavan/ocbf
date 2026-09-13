"""Mammut-specific reconciliation. Facts cannot be inferred from missing join keys."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from ocbf._values import fingerprint, freeze, utc
from ocbf.errors import EvidenceConflict, ValidationError


@dataclass(frozen=True)
class ProductionCommand:
    report_id: str
    action: str
    chassis_id: str
    entry_id: str | None = None
    station: str | None = None
    entered_at: datetime | None = None
    exited_at: datetime | None = None
    product_type: str | None = None

    def __post_init__(self):
        if self.action not in ("open", "update", "close"):
            raise ValidationError("unsupported production action")
        for name in ("entered_at", "exited_at"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, utc(getattr(self, name)))
        if self.action == "open" and (not self.station or self.entered_at is None):
            raise ValidationError("Open requires station and reported entered_at")
        if self.action != "open" and self.entry_id is None:
            raise ValidationError("Update/Close requires its actual persistence entry_id")


@dataclass(frozen=True)
class EntryBinding:
    """Verified sink/database crosswalk, supplied separately from source commands."""

    entry_id: str
    open_report_id: str
    evidence: str


def parse_production(row):
    """Parse one retained silver row, preserving action-specific fields and clocks.

    Transport identity uses topic/partition/offset. Publication time is deliberately not
    converted to an application knowledge or receipt clock.
    """
    names = {0: "open", 1: "update", 2: "close"}
    try:
        action = names[int(row["action"])]
        report_id = f"{row['topic']}:{row['partition']}:{row['offset']}"
        parse_time = lambda name: datetime.fromisoformat(row[name]) if row.get(name) else None
        return ProductionCommand(
            report_id,
            action,
            row["chassis_id"],
            str(row["entry_id"]) if row.get("entry_id") is not None else None,
            row.get("station"),
            parse_time("entered_at"),
            parse_time("exited_at"),
            row.get("product_type"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError("invalid retained production command") from exc


def reconcile_production(commands, bindings=()):
    reports = {}
    for command in commands:
        previous = reports.setdefault(command.report_id, command)
        if previous != command:
            raise EvidenceConflict(
                "conflicting payload for transport identity", key=command.report_id
            )
    crosswalk = {}
    for binding in bindings:
        if not binding.evidence:
            raise ValidationError("entry crosswalk requires supporting provenance")
        if (
            binding.open_report_id not in reports
            or reports[binding.open_report_id].action != "open"
        ):
            raise ValidationError("entry crosswalk must name a retained Open")
        if binding.entry_id in crosswalk and crosswalk[binding.entry_id] != binding:
            raise EvidenceConflict("conflicting entry crosswalk", key=binding.entry_id)
        crosswalk[binding.entry_id] = binding
    episodes, unresolved = defaultdict(list), []
    for command in sorted(reports.values(), key=lambda c: c.report_id):
        if command.action == "open":
            episodes[command.report_id].append(command)
            continue
        binding = crosswalk.get(command.entry_id)
        if binding is None:
            unresolved.append((command.report_id, "missing verified entry_id-to-Open crosswalk"))
            continue
        opened = reports[binding.open_report_id]
        if command.chassis_id != opened.chassis_id:
            raise EvidenceConflict(
                "entry crosswalk changes chassis identity", key=command.report_id
            )
        episodes[binding.open_report_id].append(command)
    return freeze(
        {
            "episodes": dict(episodes),
            "unresolved": unresolved,
            "reconciliation_id": fingerprint("production-reconciliation", (reports, crosswalk)),
        }
    )


def initial_gate(audit):
    """No verified Operation dossier/crosswalk accompanied the retained Kafka archive."""
    return {
        "status": "incomplete",
        "deliverable": "2A",
        "eligible_jobs": [],
        "eligible_operations": [],
        "assessable_process_results": 0,
        "selection": "first ten eligible typed Jobs in stable identifier order; no score-based selection",
        "baseline": ["2026-09-02T00:00:00Z", "2026-09-07T00:00:00Z"],
        "assessment": ["2026-09-07T00:00:00Z", "2026-09-09T00:00:00Z"],
        "knowledge_cutoff": audit["retained_at"],
        "missing_evidence": [
            "Verified persistence entry_id-to-Open crosswalk or an independently verified Operation endpoint binding.",
            "Historically applicable typed Job/Operation/Station/ProductType identity bindings and relevant alternatives.",
            "Historically applicable producer/deployment provenance covering the baseline and assessment records.",
            "Tracking opportunity/availability coverage and separation of simulation input from plant input where applicable.",
        ],
        "reference": {
            "status": "unavailable",
            "groups": [],
            "sample_sizes": {},
            "method": "deduplicated closed baseline Operations; linear empirical p90 by Station/ProductType",
        },
        "exposure": "unavailable: no verified complete tracking-loss interval coverage",
        "likelihood_admission": "none; command vocabulary and retained counts do not verify endpoint associations",
    }
