# Your first assessment

Assess one synthetic Operation with a reported start and two possible end associations.
The question is whether its duration exceeds a supplied 30-minute reference. Occurrence
and association remain uncertain; endpoint times are fixed inputs.

The complete executable source is `examples/fixed_parameters.py`. The source excerpts
below are included directly from that file.

## 1. Run the complete example

After [installation](installation.md), run:

```bash
python -m examples.fixed_parameters
```

It prints nominal, correction, and retraction estimates and writes bundles and a report
under `artifacts/fixed-parameters`. The command explicitly selects the core
`reference_elimination` engine.

!!! note "Synthetic inputs"

    Reports, identities, times, trust values, and the duration reference are supplied
    demonstration inputs. The calculation validates numerical mechanics, not factory accuracy.
    See [capabilities and limitations](../reference/capabilities.md).

## 2. Supply the semantic context

The fixture declares event/object types and a qualified relation, then instantiates three
candidate events for one Operation. A candidate is a possible occurrence, not an observation.

```python
--8<-- "examples/fixed_parameters.py:context"
```

The snippets inside `example_inputs` share the imports and the fixed UTC constant at the
top of the executable file. They are source excerpts; run the complete module rather than
pasting individual function fragments into a fresh interpreter.

## 3. Interpret versioned reports

Each report has a source, producer version, stable record identity, revision identity,
knowledge time, and explicit synthetic provenance.

```python
--8<-- "examples/fixed_parameters.py:evidence"
```

The supplied `SyntheticInterpreter` maps a positive endpoint report to a conjunction:
the candidate event exists and is linked to this Operation. Inspect its definition to see
exactly which assertions one report concerns.

??? example "The complete fixture interpreter"

    ```python
    --8<-- "examples/fixed_parameters.py:interpreter"
    ```

Interpretation and likelihood construction are separate. In a real integration, inspect
admission issues before accepting observations. See [interpret evidence](../how-to/interpret-evidence.md).

## 4. Resolve manual trust and model assumptions

The example supplies all needed Boolean priors, classifies catalogue multiplicities, and
declares the independence assumption between distinct information groups. It resolves two
named binary-channel settings.

```python
--8<-- "examples/fixed_parameters.py:parameters"
```

These settings are assumptions held fixed during inference; comparing named settings is
[assumption sensitivity](../concepts/parameters.md#assumption-sensitivity), not a credible
interval. The [trust guide](../how-to/configure-trust.md) shows resolution and comparison.

## 5. Define support and process questions

The model permits at most one end association. Its decoding binds the fixed endpoint times.

??? example "Model specification"

    ```python
    --8<-- "examples/fixed_parameters.py:model"
    ```

The query projection supplies start/end alternatives, a population, a time window, and the
separate normative duration reference. It also defines an expected exception count and
descriptive overlap with a supplied condition interval.

??? example "Query definitions"

    ```python
    --8<-- "examples/fixed_parameters.py:queries"
    ```

## 6. Compile, infer, and evaluate

This complete interactive block uses the executable fixture to supply every input:

```python
from examples.fixed_parameters import example_inputs
from ocbf.api import compile_model, evaluate, infer, requirements_for
from ocbf.inference.contracts import InferencePolicy

spec, queries, settings, records = example_inputs()
print("admission issues:", spec.evidence.issues)
print("interpreted observations:", len(spec.evidence.observations))
print("resolved parameters:", spec.parameters.parameter_id)
model = compile_model(spec)
result = infer(
    model,
    requirements=requirements_for(queries),
    policy=InferencePolicy(engine="reference_elimination"),
)
answers = evaluate(result, queries)

for estimate in answers.estimates:
    print(estimate.name, estimate.value, estimate.unit, estimate.denominator)
    print(estimate.status, estimate.computation, estimate.qualifications)
```

Running the block prints the deterministic estimates:

```text
admission issues: ()
interpreted observations: 3
resolved parameters: parameters:e8b04d4e908b047f6e608422e4b7597485e55672c2f652c4260053209ce0030e
duration 0.5 probability 0.6282608695652174
assessed exact-on-finite-model ('Conditional on declared candidate support, time model and evidence coverage.', 'Missing endpoints and unresolved associations are not zero durations.', 'Configured reference is a query input, not an observed production standard.')
count 0.3141304347826087 executions 0.6282608695652174
assessed exact-on-finite-model ('Conditional on declared candidate support, time model and evidence coverage.', 'Missing endpoints and unresolved associations are not zero durations.', 'Configured reference is a query input, not an observed production standard.')
tracking overlap 7.5 minutes 0.6282608695652174
assessed exact-on-finite-model ('Conditional on declared candidate support, time model and evidence coverage.', 'Missing endpoints and unresolved associations are not zero durations.', 'Configured reference is a query input, not an observed production standard.', 'Descriptive overlap is not causal delay or ready-to-work waiting.')
```

Each estimate prints its name, value, unit, and denominator on the first line, then its
status, computation label, and qualifications on the second. The `duration` estimate is a
conditional probability whose denominator is the applicable, evaluable execution mass, and
the qualifications state the conditions under which the number holds.

A missing or non-evaluable history contributes explicit outcome information. Read the
denominator and qualifications alongside the value; see [result meanings](../reference/results.md).

## 7. Replay and revise

The full command exports inputs before inference and checks their replay. Reproduce the
nominal result from the recorded bundle:

```python
from examples._shared.study import reproduce

answers = reproduce("artifacts/fixed-parameters/nominal/inputs.json")
```

The injected correction replaces an end report. The retraction withdraws that revision.
Both create a new effective evidence state and a new calculation. Replaying identical
scientific inputs preserves their identities and deterministic answers; run identity changes.

Continue with [Concepts](../concepts/index.md) for the objects and behavior behind this
assessment, then
[evidence revisions](../how-to/revise-evidence.md),
[inference selection](../how-to/choose-inference.md), or
[session reuse](../how-to/reuse-sessions.md).
