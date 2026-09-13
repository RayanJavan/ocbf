# Results and interchange

The [query topic](../concepts/queries.md) illustrates result meaning with Operation
and Job questions. This page defines the returned fields and status rules.

## Inference results

`InferenceResult` carries model, plan, and run identities; a posterior representation;
available capabilities; computation meaning; diagnostics; and a manifest. Controlled
execution can attach a separate execution assessment.

| Result dimension | Read as |
|---|---|
| `computation` | Exact-on-model, analytical hybrid, sampling, or a named marginal approximation. |
| `capabilities` | Operations actually supplied by this posterior. |
| `diagnostics` | Numerical assessment specific to the route. |
| `manifest` | Scientific scope, input lineage, resolved choices, and implementation qualifications. |
| `execution` | Completion, cancellation, resource exhaustion, or failure, when assessed. |

A completed run need not be well mixed or supported by sufficient physical evidence.

## Query estimates

`QueryResults` associates a bundle and its estimates with a model/run. An `Estimate`
contains quantity, value, unit, status, denominator, outcomes, qualifications, evidence
identities, metadata, and optional distribution/numerical detail.

| Field | Interpretation |
|---|---|
| `value=None` | No assessed scalar value; consult status and qualifications. Distribution-valued results may use `distribution`. |
| `denominator` | Effective applicable/evaluable mass or population measure defined by the quantity. |
| `outcomes` | Satisfied, violated, pending, unresolved, or inapplicable history mass/counts. |
| `distribution` | Structured output such as per-rank probabilities. |
| `numerical` | Query-specific MCSE, chain/ESS/R-hat assessments, or unavailable precision. |
| `evidence_ids` | Traceable contributing inputs; not causal responsibility. |

For whole-Job status, violations take precedence; without a violation, unresolved
precedes pending. No applicable obligation means inapplicable.

For a duration obligation with one valid start inside the query window:

- No end is pending only while elapsed time at the horizon is within the supplied
  threshold; otherwise it remains unresolved.
- One known end after the horizon is pending. One valid end at or before the horizon
  is closed and can be compared with the threshold.
- Ambiguous endpoints, missing times, reversed intervals, or a missing threshold for a
  closed duration obligation remain unresolved.

An explicitly false applicability condition or a valid start outside the window makes
the execution inapplicable. When start candidates are declared but none is selected,
the execution is also inapplicable; an empty start declaration remains unresolved.
These classifications concern each modeled history, not proof from missing physical evidence.

Priority results use only common histories in which every Job has applicable and evaluable
obligations. Competition ranks preserve ties. Unrankable mass and the ranks of mean
scores remain separate outputs.

Exposure is descriptive overlap with supplied conditions. It does not establish readiness,
causal delay, or additive attribution shares.

## Incomplete execution

Execution status is `complete`, `cancelled`, `resource-exhausted`, or `failed`.
Before usable output, a typed failure carries the assessment. Usable partial chains retain
their numerical qualifications. Partial query bundles identify missing queries rather than
filling them with zero.

## Interchange and retention

Neutral JSON bundles retain supported allowlisted values. Array artifacts pair a typed
JSON manifest with validated `.npy` payloads loaded without pickle. Neither serializes
native solver workspaces or persistent computational caches.

`summaries_only(result)` returns a copy without posterior capabilities. Keep query results
separately. Other references and session entries remain owned by their holders.

Scientific identities and serialized contract versions are preserved on supported replay;
a new execution receives a new run identity. See [export and replay](../how-to/export-replay.md).
