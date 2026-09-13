"""Frozen empirical investigation reference; never imported by the OCBF compiler."""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from ocbf._values import fingerprint, freeze, utc
from ocbf.errors import EvidenceConflict, ValidationError


@dataclass(frozen=True)
class ClosedBaselineOperation:
    operation_id: str
    station: str
    product_type: str
    start: datetime
    end: datetime
    evidence_ids: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "start", utc(self.start))
        object.__setattr__(self, "end", utc(self.end))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        if (
            self.end < self.start
            or not self.evidence_ids
            or not self.station
            or not self.product_type
        ):
            raise ValidationError(
                "baseline requires admitted closed endpoints and reference groups"
            )


def freeze_reference(operations, *, start, end, provenance):
    start, end = utc(start), utc(end)
    if end <= start or not provenance:
        raise ValidationError("explicit baseline window and provenance are required")
    unique = {}
    for operation in operations:
        previous = unique.setdefault(operation.operation_id, operation)
        if previous != operation:
            raise EvidenceConflict("conflicting baseline Operation", key=operation.operation_id)
        if not start <= operation.start < end or operation.end >= end:
            raise ValidationError(
                "baseline Operation must be closed inside the frozen baseline window"
            )
    groups = defaultdict(list)
    for operation in unique.values():
        groups[(operation.station, operation.product_type)].append(
            (operation.end - operation.start).total_seconds() / 60
        )
    entries = []
    for (station, product_type), values in sorted(groups.items()):
        values.sort()
        index = (len(values) - 1) * 0.9
        low, high = math.floor(index), math.ceil(index)
        quantile = values[low] + (index - low) * (values[high] - values[low])
        entries.append(
            {
                "station": station,
                "product_type": product_type,
                "sample_size": len(values),
                "threshold_minutes": quantile,
            }
        )
    content = {
        "format": "empirical-duration-reference-v1",
        "window": (start, end),
        "method": "linear empirical quantile, q=0.9, index=(n-1)q",
        "provenance": provenance,
        "operations": tuple(
            {
                "operation_id": o.operation_id,
                "station": o.station,
                "product_type": o.product_type,
                "start": o.start,
                "end": o.end,
                "evidence_ids": o.evidence_ids,
            }
            for k in sorted(unique)
            for o in (unique[k],)
        ),
        "groups": entries,
        "meaning": "investigation reference, not a production standard",
    }
    return freeze({**content, "reference_id": fingerprint("reference", content)})


def bind_thresholds(reference, operation_groups):
    thresholds = {
        (g["station"], g["product_type"]): g["threshold_minutes"] for g in reference["groups"]
    }
    return freeze(
        {
            "reference_id": reference["reference_id"],
            "threshold_minutes": {
                key: thresholds[group]
                for key, group in operation_groups.items()
                if group in thresholds
            },
        }
    )
