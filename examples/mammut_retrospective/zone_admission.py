"""Inspect bounded retained RTLS admission without constructing a factory posterior.

Run python -m examples.mammut_retrospective.zone_admission --archive PATH.
"""

import argparse
from pathlib import Path

from ocbf.api import prepare_evidence
from ocbf.evidence import EvidenceRecord
from ocbf.io import write_bundle
from ocbf.schema import Schema
from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext
from .archive import _query, file_digest
from .rtls import ZoneInterpreter


def retained_admission(archive, output):
    root = Path(archive).resolve()
    database = root / "catalog/kafka.duckdb"
    rows = _query(
        database,
        """SELECT topic, partition, "offset", entity_id, zone, timestamp
        FROM silver.telemetry_normalized
        WHERE source='rtls' AND source_topic='rtls/zone' AND entity_type='chassis'
        ORDER BY partition, "offset" LIMIT 100""",
    )
    context = SemanticContext.from_universe(UniverseBuilder(Schema([], [])).build())
    records = tuple(
        EvidenceRecord(
            f"{r['topic']}:{r['partition']}:{r['offset']}",
            f"{r['topic']}:{r['partition']}:{r['offset']}:1",
            "rtls",
            "historical-version-unverified",
            {k: r[k] for k in ("entity_id", "zone", "timestamp")},
            None,
            provenance={
                "physical_origin": "unverified",
                "archive": str(database),
                "transport": "Kafka",
            },
        )
        for r in rows
    )
    evidence = prepare_evidence(
        records,
        as_of=None,
        revision_policy="static-v1",
        context=context,
        interpreters={("rtls", "historical-version-unverified"): ZoneInterpreter()},
    )
    result = {
        "status": "real retained reports; business admission incomplete",
        "records": len(records),
        "admitted": len(evidence.observations),
        "issues": evidence.issues,
        "manifest_sha256": file_digest(root / "data/raw/manifest.json"),
        "database_sha256": file_digest(database),
        "sampling_scope": "first 100 normalized RTLS chassis zone reports in partition/offset order",
        "missing": "verified historical identity/zone/producer dossier and sensor-versus-simulation origin",
        "clock": "static retained snapshot; no application receipt times invented",
    }
    write_bundle(output / "retained-inputs.json", records)
    write_bundle(output / "retained-admission.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="artifacts/mammut-zone-admission")
    options = parser.parse_args()
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    report = retained_admission(options.archive, output)
    print({key: report[key] for key in ("status", "records", "admitted")})
