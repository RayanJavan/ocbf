# Use joint inference

Use coherent joint information when a question depends on several occurrences,
associations, timestamps, or shared source modes.

## Run a same-target comparison

```bash
python -m examples.joint_inference --output artifacts/joint-inference
```

The synthetic example compares finite exact answers with four blocked chains. It also
checks a constrained twelve-variable case against an independent symmetric reference,
records numerical diagnostics, and verifies seeded replay.

Inputs, posterior draws, comparisons, and reports are written beneath the selected
directory. Real source admission is a separate [Mammut integration](../integrations/mammut.md).

## Preserve shared uncertainty

Submit related questions as one query bundle. When drawing is needed, evaluation uses a
common draw set across the union of requested scopes. A source mode shared by two Jobs is
sampled once per history.

Whole-Job conformance and priority distributions cannot be reconstructed by multiplying
marginal probabilities. Expected sums can use suitable separate local joint expectations;
that does not make the full population independent.

## Work with uncertain time

The `timestamp_gaussian` channel requires `TimestampParameters` and a supplied
`ContinuousVariableSpec` with a proper prior and declared units. Modeled semantic times
use the existing event-time assertion keys and `UTC seconds`.

An endpoint names either a fixed reported time or a modeled `time_key`. Mixing the two
for one event is rejected. Inactive event coordinates are omitted from decoded times.

Bounded Gaussian integration is admitted only for supported factor structure. Hard temporal
inequalities remain in sampled state; arbitrary truncation or nonlinear integration does
not become exact Gaussian inference. See [channel and factor capabilities](../reference/capabilities.md).

## Assess numerical quality

Inspect each estimate's `numerical` assessment. MCSE describes the reported quantity,
including its numerator/denominator dependence. ESS and R-hat are empirical diagnostics;
constant sampled indicators do not certify exact zero/one probabilities or zero error.

The sampler can integrate admitted variables for transitions and reconstruct them jointly
for nonlinear queries. General query-functional Rao–Blackwellization is not exposed.

Use [inference selection](choose-inference.md), [query evaluation](evaluate-queries.md),
and [diagnostics](read-diagnostics.md) for the corresponding procedures.
