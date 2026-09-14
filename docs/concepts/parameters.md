# Parameters and trust

[Observations](evidence.md) say what each report claims. How far a claim should change the
probabilities depends on how reliable the source is: how often it reports an end that really
happened, and how often it reports one that did not. In OCBF you state these values yourself, as
**parameters**. Explicit rules choose the parameters before any probability is computed, and the
parameters do not change during inference. This page adds the trust values of the running example.

## Observation channels

An **[observation channel](../reference/glossary.md)** gives the probability of a report when its
scope is true and when its scope is false. The channel for the endpoint reports has two values:

```python
from ocbf.reliability.config import ChannelValues

nominal_values: ChannelValues = ChannelValues(sensitivity=0.85, false_positive=0.15)  # (1)!
cautious_values: ChannelValues = ChannelValues(sensitivity=0.65, false_positive=0.35)  # (2)!

for label, values in (("nominal", nominal_values), ("cautious", cautious_values)):
    ratio: float = values.sensitivity / values.false_positive
    print(f"{label}: a positive report is {ratio:.2f}x as likely when true as false")
```

1.  **Sensitivity** is the probability that the source sends a positive report when the scope is
    true. **False-positive probability** is the probability that it sends the same report when the
    scope is false.
2.  A more cautious assumption about the same source. Its positive reports tell true and false
    apart less clearly.

Running the block prints:

```text
nominal: a positive report is 5.67x as likely when true as false
cautious: a positive report is 1.86x as likely when true as false
```

For the conjunction channel, "true" means that the whole scope holds: the event happened *and*
belongs to `op`. The ratio shows how strongly one report shifts the probabilities. Under the
nominal values, a positive report is 5.67 times as likely when its scope is true, so it is strong
evidence. Under the cautious values it is only 1.86 times as likely, which is weak evidence.

Neither setting means "this report is correct with probability 0.85". The probability that an
end belongs to `op` also depends on the priors, on the other reports, and on rules between
assertions; inference combines all of them. Other kinds of channel have other parameters, such as
a confusion matrix for categorical reports or bias and error size for timestamps; see
[channel capabilities](../reference/capabilities.md#observation-channels).

## Priors and assumptions

The model also needs a starting probability for every uncertain assertion, and a record of the
assumptions made:

```python
priors: dict[str, tuple[float, float]] = {  # (1)!
    assertion: (0.5, 0.5) for assertion in (*exists.values(), *links.values())
}
assumptions: dict[str, str] = {
    "cardinality_role": "normative",  # (2)!
    "dependence": "distinct reports have conditionally independent errors",  # (3)!
}
```

1.  A **prior** gives each Boolean assertion a starting weight, before any report counts. Here
    every assertion starts with equal weight for false and true. Rules between assertions and the
    reports change the final probability.
2.  How to use the multiplicities declared in the schema, such as `Multiplicity(0, 1)`.
    `"normative"` treats them as rules the process should follow, without changing any
    probability. `"descriptive"` makes histories that break them less likely, and `"support"`
    gives those histories zero probability. There is no default, so compilation fails when this
    choice is missing.
3.  A plain-language statement that report errors are independent of each other. It is stored
    with the parameters for anyone reading the results, and it does not change the calculation. A
    shared cause of errors has to be modeled with a factor.

The assumptions are stored in the resolved parameter set, and each compiled model records the
identifier of that parameter set.

## Resolution and provenance

A **trust rule** assigns channel values to observations, selected by source, family, channel, or
producer version. It also states where the values come from. **Resolution** finds the matching
rule for the channel of every observation:

```python
from ocbf.reliability.config import (
    ParameterSet, ResolvedChannel, TrustRule, resolve_parameters,
)

nominal_rule: TrustRule = TrustRule(
    nominal_values,
    origin="manual synthetic assumption",  # (1)!
    family="endpoint",  # (2)!
)
parameters: ParameterSet = resolve_parameters(
    evidence.observations, rules=[nominal_rule], priors=priors, assumptions=assumptions
)
channel: ResolvedChannel = parameters.channels[0]
print(channel.key)
print(channel.values.sensitivity, channel.values.false_positive, "from", channel.origin)
```

1.  Where the values come from. The origin is stored next to the resolved values.
2.  The rule applies to every observation in the `endpoint` family. A rule can instead name an
    exact source, channel, and producer version.

Running the block prints:

```text
ChannelKey(source_id='fixture', family='endpoint', channel='conjunction', producer_version='1')
0.85 0.15 from manual synthetic assumption
```

All three observations have the same source, family, channel, and producer version. They share one
channel key, so one resolved channel serves all of them. Resolution looks for a rule in this order:

1. a rule that names the exact source, family, channel, and producer version;
2. otherwise, the single rule that names the family;
3. otherwise, an explicitly supplied default.

If no rule matches, or several match equally well, resolution fails instead of guessing:

```python
from ocbf.errors import ValidationError

try:
    resolve_parameters(
        evidence.observations, rules=[], priors=priors, assumptions=assumptions
    )
except ValidationError as error:
    print(type(error).__name__, str(error))
```

Running the block prints:

```text
ValidationError ChannelKey(source_id='fixture', family='endpoint', channel='conjunction', producer_version='1'): missing manual trust rule
```

These are **[manual trust](../reference/glossary.md)** values: OCBF does not estimate them from the
reports. Every value a calculation uses has an origin that you stated.

## Assumption sensitivity

Different trust values give a different parameter set. From it, a separate model and separate
results are computed later:

```python
cautious_rule: TrustRule = TrustRule(
    cautious_values, origin="cautious synthetic assumption", family="endpoint"
)
cautious: ParameterSet = resolve_parameters(
    evidence.observations, rules=[cautious_rule], priors=priors, assumptions=assumptions
)
print("same priors:", cautious.priors == parameters.priors)
print("same assumptions:", cautious.assumptions == parameters.assumptions)
print("separate parameter set:", cautious.parameter_id != parameters.parameter_id)
```

Running the block prints:

```text
same priors: True
same assumptions: True
separate parameter set: True
```

The nominal and cautious settings lead to two answers, and each answer holds only if its trust
values are right. How much the two answers differ shows how strongly the result depends on the
trust assumption. [Queries and result meaning](queries.md#different-sources-of-uncertainty)
compares them. The range between the two answers is not a credible interval. OCBF also does not
average them, because nothing says how likely each setting is.

!!! note "Keep in mind"

    Trust values describe sources, not results. A sensitivity of 0.85 is an assumption about how
    the source reports. The probability that `op` took longer than 30 minutes is only known after
    inference.

## Summary

- Channel values give the probability of a report when its scope is true and when it is false.
  Their ratio shows how strongly one report shifts the probabilities.
- Trust rules are resolved into a `ParameterSet`: the channel values with their stated origins,
  the priors, and the assumptions. If a value is missing, resolution fails.
- Changed trust values give a separate calculation. Compare its results with the original ones; do
  not average them.

Next, [Models and constraints](model.md) combines the context, evidence, and parameters into
possible histories. For the procedure, see [configure manual trust](../how-to/configure-trust.md).
