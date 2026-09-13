# Mammut retrospective use case and source register

Deliverable **2A**, recorded 11 September 2026. **Incomplete: no verified eligible cohort
has been established, and no retained-data likelihoods have been constructed.**
This is the admission decision for the first business assessment, not a factory result.

The subsequent numerical validation record adds a read-only admission check of 100 retained RTLS
zone reports. It does not change this incomplete cohort/provenance decision.

## Frozen intended use case

| Item | Contract |
|---|---|
| Business question | Operation-duration exception probability; expected applicable exception count; descriptive overlap with observed tracking-loss intervals where coverage supports it. |
| Baseline | `[2026-09-02T00:00:00Z, 2026-09-07T00:00:00Z)`; admitted deduplicated Operations must close in this baseline. |
| Assessment | `[2026-09-07T00:00:00Z, 2026-09-09T00:00:00Z)`. |
| Retention cutoff | `2026-09-09T06:50:44Z`. Application receipt times remain unverified. |
| Cohort rule | First ten eligible typed Jobs in stable identifier order, with relevant Operations, association alternatives and dependency closure. Eligibility uses provenance/identity/coverage sufficiency, never exception scores. |
| Duration reference | Separately frozen empirical p90 by Station/ProductType; linear interpolation at index `(n-1)*0.9`. Publish sample sizes and input identities. Missing groups unavailable. |
| Temporal model | Condition on verified reported endpoint times. Preserve joint occurrence/association uncertainty; missing endpoints remain unresolved, with pending/censoring at the horizon reported separately. |
| Sensitivity | Named fixed manual-trust settings, holding interpretation, priors, scope and reference fixed. Assumption sensitivity, without mixture weights. |
| Interpretation status | Retrospective reinterpretation using the inspected contract, not an “as interpreted then” reconstruction. |

## Retained archive evidence

The read-only input was the September 9 archive at `elastocel/data-extract`; the application
uses its existing `catalog/kafka.duckdb` silver tables. It does not rebuild the archive.
Raw reports remain outside tracked OCBF source files. The local generated
`artifacts/mammut-retrospective/source-audit.json` retains the aggregate distributions.

| Input | SHA-256 |
|---|---|
| `data/raw/manifest.json` | `50a1dd60a2681da088b4be8b99939703c7fd151e66c78f510be9082d91820e73` |
| `catalog/kafka.duckdb` | `d2b8c5d7ccd09a89777dcb5f0995b77df465fbc2de4dd470d68a47ffc4c98773` |

| Production action | Reports | Distinct entry IDs | Missing entry ID | Missing entered-at | Missing station |
|---|---:|---:|---:|---:|---:|
| Open | 12,619 | 0 | 12,619 | 0 | 0 |
| Update | 65,487 | 902 | 0 | 65,487 | 65,487 |
| Close | 1,354 | 1,354 | 0 | 1,354 | 1,354 |

Production publication spans September 2 at 11:36:20.021 UTC through September 8 at
20:20:47.357 UTC. Open occurrence dates are September 2, 3, 5, 6, 7 and 8. Their distinct
`(chassis, station, entered_at)` counts are respectively 70, 111, 131, 397, 70 and 1,184.
No September 4 Open is retained; this is a gap, not proof that no Operation happened.
Repeated Open signatures are not verified persisted episode identities. All Update rows
lack occurrence endpoints; publication time cannot fill them. Retention overlaps the
requested windows, but does not prove complete observation opportunity or coverage.

## Producer and provenance register

The active repository was consulted at commit
[`de6f301fa1d9bcaead2b7971038b04aaadb4b07e`](https://github.com/amirhosseinniknam/factory-monitor/commit/de6f301fa1d9bcaead2b7971038b04aaadb4b07e),
dated September 3 at 13:06:52 UTC. The corresponding
[Deploy workflow run](https://github.com/amirhosseinniknam/factory-monitor/actions/runs/33759201825)
was observed successful. This is not a historical deployment attestation for every
retained record, especially the September 2–3 baseline. Inspected producer files are cached
locally under `artifacts/producer/<commit>/`; they are not bundled into generic OCBF.

| Family / field | Observed contract | Provenance classification and admission |
|---|---|---|
| Production Open | Emits chassis, station, product type and entered-at; entry ID is assigned later by the persistence sink. Repeated Open may be emitted before registration returns. | Observed retained command; derived state output. Endpoint-to-persisted-Operation binding unresolved. |
| Production Update/Close | Names persistence entry ID. Close carries exited-at; station/entered-at are omitted. | Observed command; requires a verified entry-ID-to-Open crosswalk. No universal episode key is assumed. |
| Tracking station transitions / location change | Multiple outputs arise from the same state change. | Derived, dependent observations. They cannot count as independent corroboration of production derived from that state. |
| Tracking online/offline | Offline carries last-seen metadata and detection time; observed online events do not alone prove complete reacquisition coverage. | Observed/derived transitions; no admitted negative silence or complete loss-interval model. |
| Normalized RTLS | Retained normalized chassis observations include RTLS zone/position streams; some identities fall back to tags. | Observed sensor-derived reports, requiring historical identity bindings. Reused tags and current assignments cannot be projected backward. |
| Simulation ticks | Live state consumes both normalized telemetry and simulation ticks. Its downstream output does not itself identify which input produced it. | Synthetic provenance may be unresolved in downstream records. An empty retained simulation topic is insufficient historical proof. |
| Staffing, workers, placeholder confidence | Not verified here as measured staffing or calibrated report probabilities. | Default/provisional/unresolved; excluded from likelihoods and attribution. |
| Operation identity, catalogue bindings, reference groups | Fixture attribution vocabulary exists but is not a retained production crosswalk. | Derived bindings remain unresolved until independently verified. Fixtures are synthetic vocabulary only. |
| Manual trust | Explicit named parameters supplied by caller, no empirical fitting. | Declared assumptions, never source facts. |

The action/ID behavior follows
[ProductionHistoryEmitter](https://github.com/amirhosseinniknam/factory-monitor/blob/de6f301fa1d9bcaead2b7971038b04aaadb4b07e/src/FactoryMonitor.App/State/ProductionHistoryEmitter.cs)
and [EventSinkWorker](https://github.com/amirhosseinniknam/factory-monitor/blob/de6f301fa1d9bcaead2b7971038b04aaadb4b07e/src/FactoryMonitor.App/Workers/EventSinkWorker.cs).
Shared producer inputs are visible in
[LiveStateWorker](https://github.com/amirhosseinniknam/factory-monitor/blob/de6f301fa1d9bcaead2b7971038b04aaadb4b07e/src/FactoryMonitor.App/Workers/LiveStateWorker.cs).
These source observations constrain admission; they do not calibrate physical factory truth.

## Gate decision and continuation inputs

No eligible cohort or frozen empirical reference has been admitted. A production pairing
based only on chassis/timestamps would hide unresolved identity. A tracking station visit
also needs a verified semantic binding before it can substitute for an Operation.
Thus every reference group remains unavailable, and Stage 2 has no assessable business result.

Required evidence is a verified persistence crosswalk or independently verified Operation
endpoints, historical typed Job/Operation/Station/ProductType bindings and alternatives,
historically applicable producer/deployment provenance, and coverage/provenance sufficient
for any tracking-loss interval admitted. Once supplied, the batch integration reuses
`elastocel.structure.derive_structure`; it does not duplicate catalogue derivation. Its real
catalogue/log invocation remains unvalidated in this environment because `elastocel` is not
installed in OCBF's environment and a verified cohort dossier is absent.

`examples/mammut_retrospective` contains read-only auditing, action-specific parsing and
crosswalk reconciliation, separate reference freezing, the bridge, report rendering, and
contract-based study/reproduction functions. It is a partial integration, not an autonomous
verified-cohort extractor. `examples/fixed_parameters.py` exercises the shared numerical
workflow with synthetic reports only; its correction/retraction are injected demonstrations.
