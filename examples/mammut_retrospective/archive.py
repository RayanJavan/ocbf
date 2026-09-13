"""Read-only retained archive inspection. No source access from model or query modules."""

import hashlib
import json
import subprocess
from pathlib import Path

from ocbf.errors import ValidationError


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query(database, sql):
    completed = subprocess.run(
        ["duckdb", "-readonly", str(database), "-json", "-c", "SET TimeZone='UTC'; " + sql],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def inspect_archive(root):
    """Audit action-specific identifiers and clocks without constructing likelihoods."""
    root = Path(root).resolve()
    manifest_path = root / "data/raw/manifest.json"
    database = root / "catalog/kafka.duckdb"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dumped_at") != "2026-09-09T06:50:44Z":
        raise ValidationError("initial study requires the frozen September 9 archive")
    production = _query(
        database,
        """
        SELECT action, count(*) AS records, count(DISTINCT entry_id) AS entry_ids,
          count(*) FILTER (WHERE entry_id IS NULL) AS missing_entry_id,
          count(*) FILTER (WHERE entered_at IS NULL) AS missing_entered_at,
          count(*) FILTER (WHERE exited_at IS NULL) AS missing_exited_at,
          count(*) FILTER (WHERE station IS NULL) AS missing_station,
          min(event_time) AS first_publication_utc, max(event_time) AS last_publication_utc
        FROM silver.events_production GROUP BY action ORDER BY action""",
    )
    occurrence = _query(
        database,
        """
        SELECT action, CAST(COALESCE(TRY_CAST(entered_at AS TIMESTAMPTZ),
                                      TRY_CAST(exited_at AS TIMESTAMPTZ)) AS DATE) AS occurrence_date_utc,
          count(*) AS records,
          count(DISTINCT (chassis_id,station,entered_at)) FILTER (WHERE action=0) AS open_signatures
        FROM silver.events_production GROUP BY ALL ORDER BY action, occurrence_date_utc""",
    )
    tracking = _query(
        database,
        """
        SELECT event_type, count(*) AS records, count(DISTINCT entity_id) AS entities,
          min(TRY_CAST(timestamp AS TIMESTAMPTZ)) AS first_occurrence_utc,
          max(TRY_CAST(timestamp AS TIMESTAMPTZ)) AS last_occurrence_utc
        FROM silver.events_tracking GROUP BY event_type ORDER BY event_type""",
    )
    normalized = _query(
        database,
        """
        SELECT source, source_topic, entity_type, count(*) AS records,
          count(DISTINCT entity_id) AS entities,
          min(TRY_CAST(timestamp AS TIMESTAMPTZ)) AS first_occurrence_utc,
          max(TRY_CAST(timestamp AS TIMESTAMPTZ)) AS last_occurrence_utc
        FROM silver.telemetry_normalized WHERE entity_type='chassis'
        GROUP BY ALL ORDER BY source, source_topic""",
    )
    return {
        "format": "mammut-source-audit-v1",
        "archive_root": str(root),
        "manifest_sha256": file_digest(manifest_path),
        "database_sha256": file_digest(database),
        "retained_at": manifest["dumped_at"],
        "production_actions": production,
        "production_occurrence_dates": occurrence,
        "tracking_families": tracking,
        "normalized_chassis": normalized,
        "publication_clock": "Kafka envelope, distinct from unverified application receipt",
        "interpretation": "retrospective inspection v1; not an as-interpreted-then reconstruction",
    }
