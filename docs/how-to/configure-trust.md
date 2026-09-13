# Configure manual trust

Resolve all required channel parameters before compilation. Inference holds them fixed.
The [parameters and trust topic](../concepts/parameters.md) explains the values and how
they differ from posterior confidence.

## Resolve a binary channel

```python
from examples.fixed_parameters import example_inputs
from ocbf.reliability.config import ChannelValues, TrustRule, resolve_parameters

spec, queries, settings, records = example_inputs()
parameters = resolve_parameters(
    spec.evidence.observations,
    rules=[
        TrustRule(
            ChannelValues(sensitivity=0.8, false_positive=0.2),
            "manually supplied endpoint error assumption",
            family="endpoint",
        )
    ],
    priors=spec.parameters.priors,
    assumptions=spec.parameters.assumptions,
)
```

Channel sensitivity is the probability of a positive report when the assertion is true.
False-positive probability describes the same report when it is false. These are channel
assumptions, not an upstream report's posterior confidence.

Resolution selects an exact source/family/channel/producer-version rule, otherwise one
compatible family rule, otherwise an explicitly supplied default. Ambiguous matches and
missing required values fail. The selected origin travels with the resolved parameters.

## Compare named settings

```python
from ocbf.api import compare_settings
from ocbf.inference.contracts import InferencePolicy

comparison = compare_settings(
    spec,
    queries,
    settings,
    policy=InferencePolicy(engine="reference_elimination"),
)
print(comparison.meaning)
```

Use this block after the preceding setup. All targets are validated before numerical runs.
The comparison holds priors, structural/temporal assumptions, interpretation, and query
reference fixed. Every setting has its own model and parameter identities.

For categorical confusion matrices, shared association modes, and timestamp bias/noise,
use the [channel parameter contracts](../reference/capabilities.md#observation-channels).
Changing a coverage or dependence assumption is a separate model change, not merely
a different confidence slider.
