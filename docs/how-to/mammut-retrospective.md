# Mammut retrospective example

Run read-only admission against a retained Mammut archive and read which prerequisites
are still missing. The integration contains read-only source auditing, action-specific
parsing, historical identity/binding checks, a structural bridge, and separately frozen
reference calculations.

!!! warning "No verified eligible cohort"

    The retained source assessment has not established a verified eligible cohort.
    No completed factory duration/conformance result follows from the synthetic examples.

## Check the prerequisites

The commands require the external archive and the DuckDB CLI. An assessable retained-data
run additionally needs:

- verified Operation endpoints or a persistence crosswalk;
- historical typed Job/Operation/Station/ProductType bindings and alternatives;
- historically applicable producer provenance;
- coverage/provenance for any admitted tracking-loss interval.

The dated source register is retained in the repository research corpus with archive hashes
and inspected producer references. It records observations about retained reports rather
than physical ground truth.

## Run read-only admission

```bash
python -m examples.mammut_retrospective --archive /path/to/data-extract --output artifacts/mammut-retrospective
python -m examples.mammut_retrospective.zone_admission --archive /path/to/data-extract --output artifacts/mammut-zone-admission
```

Use a Windows path for `--archive` in PowerShell. Output is written to the selected artifact
directory; the input archive remains read-only.

The zone command inspects a bounded retained sample separately from numerical inference.
An empty binding dossier yields admission issues, not fabricated candidate associations.

## Interpret the admission output

Admission checks establish what retained sources contain and whether historical bindings
are adequate. They do not supply physical ground truth.

A zone report is not automatically an Operation endpoint, continuous occupancy, readiness,
or a coverage interval. A current identity assignment cannot establish a historical
assignment. Producer defaults and uncertain simulation lineage do not become observations
because a numeric field is populated.

The synthetic [example studies](../getting-started/examples.md) validate complete numerical
composition without fetching this archive. They make no claim that the external structure
bridge or eligible cohort has been validated in every local environment. See
[integrate an application](integrate-application.md) for the general boundary.
