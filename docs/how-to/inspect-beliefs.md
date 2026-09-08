# Inspect a belief state

[`BeliefState`][ocbf.belief.state.BeliefState] is the inference contract. Everything below
works identically whether it came from [`fuse`][ocbf.pipeline.fuse] or from a
[baseline](compare-baselines.md), which is what makes them comparable.

## Get a marginal

```python
from ocbf.assertions import AssertionRef

ref = AssertionRef.e2o("ev_0012", "item", "item_3")

result.belief.prob_true(ref)      # 0.87   — binary assertions
result.belief.marginal(ref)       # array([0.13, 0.87])
```

For a categorical assertion, `marginal` returns the full distribution over its domain:

```python
type_ref = AssertionRef.event_type("ev_0012")
probs = result.belief.marginal(type_ref)
domain = universe.event_type_domain("ev_0012")
dict(zip(domain, probs))          # {'PackItem': 0.05, 'PickItem': 0.11, 'Ship': 0.84}
```

## Get a verdict, not just a number

```python
result.belief.verdict(ref).value   # 'true' | 'false' | 'undetermined'
```

`UNDETERMINED` is **not** "probability near 0.5". It means the evidence on this assertion is
below what any aggregation rule would need to decide it. An assertion can sit at 0.8 and
still be undetermined, if that 0.8 rests on one barely-better-than-chance source.

```python
result.belief.is_decidable(ref)     # False
result.belief.evidence_nats(ref)    # 0.41  — against a target of log(1/0.1) = 2.30
```

Use `verdict` whenever a decision is downstream. Use `prob_true` when you are aggregating
further and want to keep the uncertainty.

## Handle assertions outside the graph

Pruned or never-grounded assertions answer with the prior instead of raising:

```python
unknown = AssertionRef.e2o("never_seen", "item", "nor_this")

result.belief.prob_true(unknown, default=0.05)   # 0.05
result.belief.is_prior_only(unknown)             # True
result.belief.attribution(unknown)               # {}
```

Always check `is_prior_only` before treating a number as evidence-driven.

## Explain a posterior

```python
for name, contribution in result.belief.top_contributors(ref, k=5):
    print(f"{name:<28} {contribution:+.3f}")
```

```text
channel:src_00097            -3.794
silence:src_00108            -3.468
channel:src_00171            -3.415
prior                        -3.178
referential_integrity        +0.002
```

These are additive contributions to the posterior **log-odds** and they sum to the result.
Key prefixes:

| prefix | meaning |
| --- | --- |
| `channel:<source_id>` | that source made a claim |
| `silence:<source_id>` | that source stayed silent, and its coverage makes silence evidence |
| `prior` | the derived structural prior |
| `referential_integrity`, `type_gate`, `cardinality` | structural factors |

Attribution costs nothing extra — it is the retained belief-propagation messages. With
sources this weak, auditability is most of the product.

## Work in bulk

```python
from ocbf.assertions import Family

refs = [r for r in result.claim_set.refs if r.family is Family.E2O]

scores = result.belief.binary_scores(refs)        # np.ndarray, aligned with refs
mask = result.belief.decidable_mask(refs)         # np.ndarray of bool

confident_true = [r for r, s, d in zip(refs, scores, mask) if d and s > 0.9]
```

Per-family slices come back with their refs already aligned:

```python
probs, refs = result.belief.probs_for_family(Family.E2O)
```

## Re-threshold decidability without re-running

Decidability depends on a target error rate you may want to vary. Recompute it and reattach
rather than re-fusing:

```python
from ocbf.diagnostics import decidability

strict = decidability(
    result.claim_set,
    result.reliability.sensitivities(),
    result.reliability.specificities(),
    eps=0.01,                      # demand ~4.6 nats instead of ~2.3
    ess=result.ess.ess,
)
stricter_belief = result.belief.with_evidence(strict.evidence, target_nats=strict.target_nats)
```

## Read a continuous assertion

Timestamps and ordered attributes answer with a Gaussian rather than a marginal, and in
latent units unless asked otherwise:

```python
belief.gaussian(ref)              # (mean, variance), latent copula units
belief.value(ref)                 # posterior median, in observed units
belief.value_interval(ref, 0.9)   # central credible interval, in observed units
```

`marginal` and `gaussian` each raise on the other kind rather than guessing, so a mixed
query cannot silently return the wrong thing. See
[Fuse timestamps and attributes](fuse-continuous.md) for the rest.

## See also

- [Read the diagnostics](read-diagnostics.md) — what the evidence numbers mean.
- [Fuse timestamps and attributes](fuse-continuous.md) — the continuous half of the contract.
- [`ocbf.belief`][ocbf.belief] — the full API.
