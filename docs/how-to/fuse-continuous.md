# Fuse timestamps and attributes

The continuous layer runs whenever there is anything continuous to fuse, and costs nothing
when there is not. This guide covers making sure there *is*, reading the result, and checking
it before you trust it.

## Declare the attributes

An attribute reaches the copula only if the schema declares it with a copula-eligible
[`AttributeKind`][ocbf.schema.core.AttributeKind]. Timestamps need no declaration — every
candidate event has one.

```python
from ocbf.schema import AttributeKind, AttributeSpec, ObjectType

ObjectType("Order", (
    AttributeSpec("order_value", AttributeKind.CONTINUOUS, bounds=(0.0, None)),
    AttributeSpec("order_units", AttributeKind.COUNT, bounds=(0.0, None)),
))
```

`CATEGORICAL` is the one kind that does not qualify, and asking for it raises rather than
mis-modelling it silently. Use an `ORDINAL` spec if the levels really are ordered; otherwise
leave the attribute to the discrete layer.

Check what actually grounded:

```python
universe = builder.build()
print(universe.prune_report.summary())
# {..., 'attr_registered': 50, 'attr_ambiguous': 25, 'attr_categorical': 0}
```

!!! tip "A non-zero `attr_ambiguous` is expected, not a misconfiguration"

    **Event** attributes enter the copula only where every type in the event's support
    declares them, so an event whose activity label is uncertain carries none. Raise
    `p_type_known`, or narrow the type supports, if you need more of them.
    [The model](../explanation/the-model.md#the-continuous-layer) explains why the
    restriction is structural. Object attributes are never ambiguous.

## Emit continuous claims

A continuous claim carries a `float`. The channel family follows from the assertion family,
so nothing else about the source changes:

```python
from ocbf.assertions import AssertionRef
from ocbf.sources import Claim

Claim("clock_7", AssertionRef.event_time("ev_00042"), 1714.5)
Claim("erp_3", AssertionRef.object_attr("order_12", "order_value"), 249.90)
```

## Fuse

```python
from ocbf.pipeline import fuse

result = fuse(universe, sources)
print(result.continuous.summary())
```

```text
{'variables': 124, 'edges': 8194, 'banks': {'time_bracket': 74, 'continuous_channel': 395,
 'copula_correlation': 18, 'cg_time': 32, 'time_prior': 42, 'precedence': 407},
 'factors': 7769, 'templates': 7, 'kinds': ['continuous', 'count', 'ordinal', 'truncated'],
 'correlated_pairs': 3, 'max_abs_correlation': 0.8691,
 'continuous_claims': 7196, 'precedence_skipped': 310}
```

`result.continuous.is_empty` is `True` when nothing grounded — usually because no source
emitted a `float`, or the schema declares no copula-eligible attribute. Pass
`FusionConfig(continuous=None)` to switch the layer off for an ablation.

If `precedence` is 0, the discrete layer is not confident enough about the links for any
lifecycle ordering to apply. Lowering `precedence_min_weight` will not fix that — it grounds
thousands of near-vacuous factors instead. The fix is evidence on the links.

## Read a posterior

Continuous beliefs are Gaussian on the **latent** copula coordinate, and are converted to
observed units on request:

```python
ref = AssertionRef.event_time("ev_00042")

result.belief.gaussian(ref)              # (-1.935, 0.002)   latent
result.belief.value(ref)                 # 5.6               hours
result.belief.value_interval(ref, 0.9)   # (4.1, 7.1)
```

[`value`][ocbf.belief.state.BeliefState.value] returns the posterior **median**, not the
mean: the copula transform is monotone but not linear, so the median passes through it
exactly while the mean does not. Under the affine marginal timestamps use, the two coincide.

Attribution reads as it does for binary assertions, except that the terms sum to the
posterior mean rather than to its log-odds:

```python
result.belief.top_contributors(ref, 4)
# [('continuous_channel:src_00410', -0.097), ('continuous_channel:src_00231', -0.097),
#  ('continuous_channel:src_00275', -0.081), ('continuous_channel:src_00318', -0.077)]
```

## Check convergence

`ep_converged` is reported beside `bp_converged`, and both should be looked at.

```python
print({k: v for k, v in result.belief.diagnostics.items() if k.startswith("ep_")})
```

A `False` with a small `ep_belief_residual` is the ordinary case on ordinal coordinates
carrying many claims: the run settles into a limit cycle instead of a fixed point, and the
residual is its amplitude in latent standard deviations. A `False` with a *large* residual
means something is wrong — check that the channel scales were fitted rather than left at
their defaults.

Intervals that look too wide are the other symptom to know. Raise
[`EPConfig.max_iter`][ocbf.inference.gabp_ep.EPConfig.max_iter]: each edge's step is divided
by its variable's site count for stability, so a coordinate with many claims needs more
sweeps to accumulate its precision, and stopping early surfaces as width rather than as an
error.

## Check calibration

Score with a proper scoring rule and the coverage of a stated interval, not with a squared
error — and always against the baselines, as on the discrete side:

```python
from ocbf.baselines import claim_median, weighted_mean
from ocbf.eval import compare_continuous

print(compare_continuous({
    "claim_median": claim_median(universe.registry, claims, result.continuous.copula),
    "ocbf": result.belief,
}, ground_truth, refs))
```

| method | CRPS | coverage | coverage error |
| --- | --- | --- | --- |
| claim median | 0.0663 | 0.769 | −0.132 |
| **OCBF** | **0.0514** | **0.852** | **−0.048** |

Read `coverage_error` the way you would read ECE on the binary side. A baseline that is close
on the point estimate and badly overconfident about it is the common case, and it is what the
layer is for.

For a per-assertion view, [`pit_values`][ocbf.eval.continuous.pit_values] gives the continuous
reliability diagram: uniform when the posterior is honest, piled at both ends when the
intervals are too narrow.

## See also

- **[The model](../explanation/the-model.md#the-continuous-layer)** — what the copula and the
  coupling are.
- **[Inference](../explanation/inference.md#the-continuous-engine)** — why one algorithm
  serves the whole layer.
- **[Benchmark against baselines](compare-baselines.md)** — the same discipline on the
  discrete side.
- **[Limitations](../about/limitations.md#continuous-layer)** — where the approximations are.
