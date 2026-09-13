"""RTLS representation adapters and conservative dossier-based semantic admission.

No zone observation is automatically an Operation endpoint or a coverage interval.
The external dossier supplies historically applicable candidates and semantic bindings.
"""

from dataclasses import dataclass, field
from datetime import datetime

from ocbf._values import fingerprint, freeze, utc
from ocbf.errors import ValidationError
from ocbf.evidence import AdmissionIssue, Observation


@dataclass(frozen=True)
class ZoneBinding:
    entity_id: str
    target_key: str
    candidates: tuple[str, ...]
    zone_labels: object
    valid_from: datetime
    valid_to: datetime
    provenance: tuple[str, ...]
    producer_version: str
    context_id: str
    verified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "valid_from", utc(self.valid_from))
        object.__setattr__(self, "valid_to", utc(self.valid_to))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "zone_labels", freeze(self.zone_labels))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if (
            self.valid_to <= self.valid_from
            or not self.candidates
            or len(set(self.candidates)) != len(self.candidates)
        ):
            raise ValidationError("invalid RTLS binding window/candidates")
        if not set(self.zone_labels.values()) <= set(self.candidates):
            raise ValidationError("zone binding names an undeclared candidate")


def normalized_zone(payload):
    """Retained telemetry.normalized representation."""
    return {
        "entity": payload["entity_id"],
        "zone": payload["zone"],
        "time": utc(datetime.fromisoformat(payload["timestamp"])),
        "information_id": payload.get("information_id"),
    }


def wrapped_zone(payload):
    """Alternative transport envelope for the same normalized observation."""
    return normalized_zone(payload["observation"])


@dataclass(frozen=True)
class ZoneInterpreter:
    bindings: tuple[ZoneBinding, ...] = ()
    parser: object = field(default=normalized_zone, repr=False, compare=False)
    name: str = "rtls-zone"
    version: str = "1"

    def interpret(self, record, context):
        def reject(reason):
            return (), (AdmissionIssue(record.revision_id, "uninterpreted", reason),)

        try:
            report = self.parser(record.payload)
        except (KeyError, TypeError, ValueError):
            return reject("invalid or unsupported RTLS zone representation")
        matches = [
            b
            for b in self.bindings
            if b.entity_id == report["entity"]
            and b.valid_from <= report["time"] < b.valid_to
            and b.producer_version == record.producer_version
            and b.context_id == context.context_id
        ]
        if len(matches) != 1:
            return reject("missing or ambiguous historical identity/zone binding dossier")
        binding = matches[0]
        if not binding.verified or not binding.provenance:
            return reject("historical binding provenance is unverified")
        if record.provenance.get("physical_origin") not in ("verified_sensor", "synthetic_fixture"):
            return reject("sensor versus simulation origin remains unverified")
        if report["zone"] not in binding.zone_labels:
            return reject("zone outside declared semantic bindings")
        information_id = report["information_id"] or record.provenance.get("information_id")
        if not information_id:
            return reject("shared lineage/information contribution is unspecified")
        return (
            Observation(
                record.record_id,
                record.source_id,
                "zone_association",
                record.producer_version,
                "association_mode",
                (binding.target_key,),
                binding.zone_labels[report["zone"]],
                (record.revision_id,),
                "rtls-zone@1:" + fingerprint("binding", binding),
                information_id,
            ),
        ), ()
