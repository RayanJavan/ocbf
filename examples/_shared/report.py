"""Render neutral query results without changing their meaning."""

import json
from ocbf._values import portable


def render_results(results, *, label, source_status):
    lines = [
        f"# {label}",
        "",
        source_status,
        "",
        "Results are conditional on the supplied candidate support, time model and manual trust.",
        "",
        "| Query | Value | Unit | Evaluable denominator | Status |",
        "|---|---:|---|---:|---|",
    ]
    for estimate in results.estimates:
        value = "unavailable" if estimate.value is None else f"{estimate.value:.8g}"
        lines.append(
            f"| {estimate.name} | {value} | {estimate.unit} | "
            f"{estimate.denominator:.8g} | {estimate.status} |"
        )
    for estimate in results.estimates:
        lines.extend(
            [
                "",
                f"## {estimate.name}",
                "",
                estimate.quantity,
                "",
                "Outcomes: " + json.dumps(portable(estimate.outcomes), sort_keys=True),
                "",
                *estimate.qualifications,
                "",
                "Evidence lineage: " + ", ".join(estimate.evidence_ids),
                "",
                "This lineage shows contributing reports and factors. It is not an evidence-removal effect.",
            ]
        )
    return "\n".join(lines) + "\n"
