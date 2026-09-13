# Parameters and trust

Parameters describe the assumptions under which reports affect the model. OCBF resolves
manual values before compilation and holds them fixed during inference.

## Observation channels

For a binary report, sensitivity is the chance of a positive report when the assertion
is true. The false-positive probability is the chance of the same report when it is false.

The synthetic assessment uses these supplied values:

```python
from ocbf.reliability.config import ChannelValues

nominal = ChannelValues(sensitivity=0.85, false_positive=0.15)
cautious = ChannelValues(sensitivity=0.65, false_positive=0.35)
```

Under the nominal setting, a positive report distinguishes true from false more strongly
than under the cautious setting. Neither setting is learned from the fixture, and neither
means “this particular report is correct with that probability.” The posterior also
depends on priors, other reports, and constraints.

Other channels have their own parameter contracts. An association channel can describe
confusion among candidate bindings. A timestamp channel can describe bias and error scale.
These values are not interchangeable with a generic confidence score.

## Resolution and provenance

A `TrustRule` associates values and their stated origin with a source, assertion family,
channel, or producer version. Resolution selects an exact override, a unique matching
family rule, or an explicit default. Missing or ambiguous assignments fail.

The resulting `ParameterSet` retains channel assignments, priors, and assumptions.
This makes a nominal assessment and a cautious assessment distinct, reproducible
calculations.

## Priors, support, and dependence

| Input | Example meaning |
|---|---|
| Prior | A supplied probability contribution for event existence before report evidence; other factors and support also affect the final probability. |
| Channel values | How the source could produce a positive or negative report under each truth. |
| Dependence assumption | Whether distinct information groups have independent report errors given the modeled truth. |
| Hard support | Whether two candidate end associations are allowed to be active together. |

Reliability values do not repair copied evidence, missing candidates, or unknown producer
meaning. A source cluster label alone does not define shared errors.

Hard support belongs to the [model's constraints](model.md#constraints-and-references).
It can change the resulting probabilities even before any report contributes.

## Assumption sensitivity

A nominal and a cautious setting produce separate conditional answers. Their difference
shows how the result responds to those assumptions. Their range is not a credible interval,
and the settings do not form a probability mixture unless a separate model defines one.

The [trust guide](../how-to/configure-trust.md) covers resolution and comparison.
[Channel capabilities](../reference/capabilities.md#observation-channels) owns the supported
parameter combinations.
