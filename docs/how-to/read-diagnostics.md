# Read the diagnostics

The three diagnostics are returned unconditionally by [`fuse`][ocbf.pipeline.fuse], because
with sources this weak a posterior without them is a number with unstated preconditions.
This guide is about what to *do* when one of them looks bad.

```python
print(result.report())      # all three, formatted
```

---

## Identifiability

```python
print(result.overlap.explain())
```

```text
Source overlap graph: 189 sources, 728 edges, 26 components (largest 74).
  identified         174  reliability estimable from co-claims
  sign_ambiguous       0  bipartite component: sign fixed by prior, not data
  prior_only          15  insufficient overlap: reliability is assumed
```

| verdict | meaning | what to do |
| --- | --- | --- |
| `IDENTIFIED` | in a non-bipartite component of ≥ 3 sources; skills jointly estimable | nothing |
| `PRIOR_ONLY` | too little overlap to estimate at all | make sure its `features` and `cluster_id` are accurate — they are now its entire reliability estimate |
| `SIGN_AMBIGUOUS` | bipartite component: estimable only up to a global sign flip | **act** — see below |

### Fixing `SIGN_AMBIGUOUS`

A bipartite overlap component has no odd cycle, so no amount of data distinguishes "these
sources are accurate" from "these sources are systematically inverted". The
better-than-chance prior picks a branch; the data does not.

In order of preference:

1. **Add overlap that closes an odd cycle.** Get any three sources in that component to
   co-claim a shared assertion. A single triangle fixes it.
2. **Add a gold-labelled assertion** inside the component to anchor the sign.
3. **Accept it and check the prior.** The intercept prior in the reliability GLM is what
   resolves the ambiguity, so its `prior_intercept` had better be right.

### Fixing widespread `PRIOR_ONLY`

Try lowering `min_overlap` before concluding the data is too thin — the default of 2
co-claimed assertions is already weak evidence, but 1 is weaker still:

```python
from ocbf.diagnostics import overlap_report
print(overlap_report(result.claim_set, min_overlap=1).counts())
```

If it barely changes, the sources genuinely do not overlap, and reliability is coming from
the hierarchy. That is the design working as intended — but it means your `features` and
`cluster_id` assignments are doing the real work.

---

## Decidability

```python
print(result.decidability.summary())
```

```text
{'assertions': 1281, 'decidable': 32, 'undecidable': 1249,
 'decidable_fraction': 0.025, 'target_nats': 2.3026, 'evidence_median': 0.6987}
```

A low decidable fraction is **normal and expected** in this regime, not a failure. It says
the source votes alone cannot decide most assertions — which is precisely why the structural
prior exists.

!!! important "Decidability is a lower bound for the structured model"

    It counts only the Chernoff information carried by source claims. OCBF also propagates
    belief along referential integrity, the type gate and cardinality, so it resolves
    assertions that *no* aggregation rule could decide from votes. Measured: on the
    undecidable subset, OCBF still beats weighted voting.

    So do not read `decidable_fraction` as "how much of the output is trustworthy". Read it
    as "how much would be trustworthy without the structure".

### If you need more decidable assertions

- **Raise `eps`.** The default target of 0.1 demands 2.30 nats. Accepting 20 % error needs
  1.61.
- **Add redundancy where it counts.** Evidence is additive in Chernoff information, so
  concentrating a few extra sources on the assertions you care about beats spreading them
  thin.
- **Improve source quality.** Chernoff information grows fast: a source at 0.6/0.6 carries
  0.02 nats, one at 0.9/0.9 carries 0.51 — a factor of 25.

```python
from ocbf.diagnostics import chernoff_binary
chernoff_binary(0.6, 0.6)   # 0.0204
chernoff_binary(0.9, 0.9)   # 0.5108
```

---

## Effective sample size

```python
print(result.ess.summary())
```

```text
{'deg_mean': 2.112, 'ess_mean': 1.606, 'deflation_mean': 0.8824,
 'deflation_min': 0.25, 'clusters': 5, 'rho_mean': 0.7227, 'rho_max': 1.0}
```

`deflation = ESS / deg` is the fraction of your evidence that is genuinely independent.

| reading | meaning |
| --- | --- |
| `deflation ≈ 1.0` | sources are independent; every vote counts |
| `deflation ≈ 0.5` | half your apparent evidence is duplicated |
| `rho_max = 1.0` | some declared family is fully correlated — effectively one source |

A `rho` near 1.0 for a cluster means those sources are duplicates. Confirm it is real:

```python
for cluster, rho in sorted(result.ess.cluster_rho.items(), key=lambda kv: -kv[1]):
    print(f"{cluster:<24} rho={rho:.3f}")
```

If a cluster is genuinely redundant, that is fine — the correction handles it. If sources
were grouped into a family that does *not* share errors, split the `cluster_id`; you are
throwing away real evidence.

The opposite error is worse: sources that secretly share an upstream model but declare
different families will not be corrected, and the model will be overconfident. Declared
provenance is trusted, so declare it honestly.

---

## Convergence

```python
for entry in result.history:
    print(entry["iteration"], entry["bp_converged"],
          entry["bp_iterations"], entry["bp_belief_residual"])
```

| symptom | cause | fix |
| --- | --- | --- |
| `bp_converged: False`, residual still falling | not enough iterations | raise `BPConfig.max_iter` |
| `bp_oscillating: True` | loopy graph cycling | raise `BPConfig.damping` toward 0.9 |
| oscillation persists at high damping | a variable with many conflicting hard factors | check whether a constraint should be soft — see [Configure constraints](configure-constraints.md) |
| `ep_converged: False`, small `ep_belief_residual` | a step-warped coordinate in a limit cycle | usually benign; the residual is its amplitude in latent standard deviations |
| `ep_converged: False`, large residual | channel scales left at their defaults | check that the continuous channels were fitted |

Convergence is judged on **beliefs**, not raw messages: with hard factors the message
residual is dominated by edges swinging against the `NEG_INF` sentinel, which says nothing
about whether the posterior settled. The continuous engine reports the same pair of residuals
under `ep_`, and both flags are worth checking — see
[Fuse timestamps and attributes](fuse-continuous.md).

---

## See also

- [Why these three](../explanation/diagnostics.md) — the theory each one implements.
- [`ocbf.diagnostics`][ocbf.diagnostics] — the full API.
