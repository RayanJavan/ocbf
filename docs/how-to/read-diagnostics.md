# Read diagnostics

Check three distinct layers before interpreting a result.

| Layer | Inspect | What it tells you |
|---|---|---|
| Evidence/model | Admission issues, candidate scope, interpretation/parameter origins, manifest | What was modeled and under which assumptions. |
| Computation | Engine diagnostics and execution assessment | What calculation completed and its numerical limitations. |
| Query | Denominator, outcomes, qualifications, numerical assessment | Which histories support this particular estimate. |

## Inspect a complete finite calculation

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
print(spec.evidence.issues)
print(result.computation, result.diagnostics)
for estimate in evaluate(result, queries).estimates:
    print(estimate.name, estimate.denominator, estimate.qualifications, estimate.numerical)
```

## Interpret the selected numerical route

For exact routes, inspect normalization, admitted scope, and costs. Exactness is conditional
on the supplied model.

For sampling, inspect query-specific MCSE, autocorrelation ESS, chain behavior, and
rank-normalized/folded split R-hat. Unavailable error estimates and poorly explored modes
remain limitations even when execution completes.

For BP/EP, inspect convergence and residuals. A converged approximation need not agree
with the exact target. The caller-grounded EP adapter also rejects improper or clipped
precision regimes.

## Keep diagnostics in their domain

Static source-overlap, source-evidence ESS, and binary decidability utilities describe
their declared source models. They do not estimate MCMC precision, modify likelihoods,
or certify arbitrary channels.

`complete` means execution completed, not that evidence was sufficient or physical truth
was recovered. Consult [sources of uncertainty](../concepts/queries.md#different-sources-of-uncertainty) and
[configuration and errors](../reference/configuration.md) when a result is incomplete.
