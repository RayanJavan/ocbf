"""Readable exports of neutral results, with no probability or source-reconciliation logic."""

import json
from pathlib import Path



def write_admission_report(output, audit, gate):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output / "admission.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    rows = [
        "| Action | Reports | Missing entry ID | Missing entry time | Missing station |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {"0": "Open", "1": "Update", "2": "Close"}
    for row in audit["production_actions"]:
        rows.append(
            f"| {names[str(row['action'])]} | {row['records']} | {row['missing_entry_id']} | "
            f"{row['missing_entered_at']} | {row['missing_station']} |"
        )
    text = (
        """# Retrospective Operation exception report — admission incomplete

No verified assessable cohort has been established. No exception
probabilities, expected counts, or empirical duration thresholds have been calculated from
these records. This is the source-admission artifact, not a completed business assessment.

The requested baseline is September 2–6 UTC, the assessment September 7–8 UTC, and the
retention cutoff 2026-09-09T06:50:44Z. The intended selection is the first ten eligible typed
Jobs in stable identifier order. Eligibility cannot use exception scores. This inspection
is a retrospective reinterpretation; Kafka publication is not application receipt time.

## Retained command evidence

"""
        + "\n".join(rows)
        + "\n\n"
    )
    text += "Open and Update/Close do not share a universal episode key. A timestamp/chassis join "
    text += "cannot establish the persistence crosswalk. Repeated Opens also cannot be assumed "
    text += "to be independent observations or safely merged into one persisted episode.\n\n"
    text += "## Missing admission evidence\n\n" + "\n".join(
        "- " + s for s in gate["missing_evidence"]
    )
    text += """

## Boundaries and reproducibility

The source audit preserves occurrence-date distributions and publication bounds separately.
Retention inside the requested dates does not establish complete observation coverage.
Missing baseline groups remain unavailable. The intended comparator is a separately frozen
linear empirical 90th percentile by Station and ProductType, with sample sizes published.

Any admitted duration analysis will condition on verified reported endpoint times; missing
endpoints remain unresolved. Occurrence and association uncertainty must remain joint.
Tracking overlap is descriptive, not a causal delay share. No readiness, staffing defaults,
placeholder confidence, soft report values or negative silence have entered likelihoods.

`source-audit.json` records archive hashes and aggregate findings. `admission.json` records
the gate and requested windows. Raw retained data stays outside tracked source files.
The separately labeled synthetic replay example validates numerical mechanics only.
"""
    (output / "report.md").write_text(text, encoding="utf-8")
