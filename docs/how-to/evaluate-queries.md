# Evaluate process queries

Define a projection, population, time window, and normative reference. Evaluate against
a posterior that declares the required capabilities.
The [query topic](../concepts/queries.md) explains projections, evaluability, and the
meaning of the different process questions.

## Evaluate a bundle

```python
from examples.fixed_parameters import example_inputs
from ocbf.api import compile_model, evaluate, infer, requirements_for
from ocbf.inference.contracts import InferencePolicy

spec, queries, settings, records = example_inputs()
result = infer(
    compile_model(spec),
    requirements=requirements_for(queries),
    policy=InferencePolicy(engine="reference_elimination"),
)
answers = evaluate(result, queries)
for estimate in answers.estimates:
    print(estimate.name, estimate.status, estimate.value, estimate.denominator)
    print(estimate.outcomes, estimate.qualifications)
```

The example defines duration exception, expected count, and descriptive exposure queries.
The [query reference](../reference/capabilities.md#process-queries) lists all built-in kinds.

## Bind the intended execution

`ExecutionProjection` has an execution identity, population identity, alternative
start/end endpoints, and optional applicability conditions. Each `Endpoint` specifies
occurrence, required associations, and either fixed or modeled time.

Fixed endpoint times must agree with model decoding. A changed comparator can be
query-only; changed modeled endpoint time requires a new model.

A `QuerySpec` supplies a window, horizon, reference, and any condition intervals. Exposure
requires declared coverage. Intervals use half-open union semantics so overlapping
conditions do not double-count elapsed time.

## Read availability before value

Keep violated, satisfied, unresolved, pending, and inapplicable outcomes distinct.
A zero evaluable denominator produces an unresolved quantity.

Priority distributions use a fixed Job population and common evaluable histories.
Competition ranks preserve ties. Report unrankable mass and distinguish ranks of mean
scores from posterior rank probabilities.

Evidence and factor lineage explain which inputs contributed. They are not causal shares
or the result of removing a source. See [result meanings](../reference/results.md).
