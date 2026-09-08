# Quickstart

Build a synthetic object-centric world, corrupt it into hundreds of weak sources, fuse them
back into a belief, and check the answer against ground truth.

Every output below is from an actual run at `seed=3`, so you can compare yours line by line.

!!! info "Why start with synthetic data"

    Ground truth is the only way to measure **calibration**, and calibration is where fusion
    actually pays. The generator is a first-class part of the library for that reason, not a
    testing convenience.

## 1. Generate a world

```python
from ocbf.synth import ProcessConfig, simulate_process

gt = simulate_process(ProcessConfig(n_orders=25, seed=3))
print(gt.summary())
```

```text
{'events': 216, 'objects': 113, 'e2o': 442, 'o2o': 99, 'n_ConfirmOrder': 23,
 'n_Invoice': 19, 'n_PackItem': 57, 'n_Pay': 13, 'n_PickItem': 57,
 'n_PlaceOrder': 25, 'n_Ship': 22}
```

This is an order-to-cash process where a single `Ship` event links one order and *several*
items. That many-to-many joint is what makes the log genuinely object-centric — no single
case notion covers both the order and the item perspective without duplication.

## 2. Ground a candidate universe

The truth is not handed to the model. Instead we build a *candidate* universe: the real
events plus decoys that never happened, each with a latent type support wider than its
truth, and a loose time bracket.

```python
universe = gt.build_universe()
print(universe.prune_report.summary())
```

```text
{'e2o_naive': 122040, 'e2o_after_signature': 47187, 'e2o_after_temporal': 26917,
 'e2o_reduction': 0.779441, 'o2o_naive': 38307, 'o2o_after_signature': 170}
```

Naive E2O grounding would enumerate 122 040 candidate links. The qualifier-signature and
temporal prunes cut that by 78 %, leaving 26 917 — and 28 010 latent variables in total.

## 3. Corrupt it into sources

```python
from ocbf.synth import SourceRegime, simulate_sources
from ocbf.sources import ClaimSet

sim = simulate_sources(universe, gt, SourceRegime(n_sources=300, n_hotspots=10, seed=3))
claims = ClaimSet.from_sources(sim.sources)
print(claims.summary())
```

```text
{'claims': 2705, 'sources': 232, 'assertions_touched': 1281, 'density': 0.00910186,
 'deg_a_mean': 2.112, 'deg_a_median': 1, 'deg_a_max': 11,
 'deg_s_mean': 11.659, 'deg_s_median': 8, 'deg_s_max': 237}
```

Read those numbers — they *are* the regime:

- `density: 0.009` — sources touch under 1 % of the assertions they could.
- `deg_a_median: 1` — the median assertion has **one** source speaking about it.
- `deg_s_median: 8` — the median source makes eight claims, far too few to estimate its own
  accuracy.

94 of the 300 sources are copiers that duplicate a cluster-mate rather than observing
independently, so their agreement is spurious by construction.

## 4. Fuse

```python
from ocbf.pipeline import fuse, FusionConfig
from ocbf.inference import BPConfig

result = fuse(
    universe,
    sim.sources,
    FusionConfig(outer_iterations=2, bp=BPConfig(max_iter=150)),
)
print(result.report())
```

```text
Source overlap graph: 189 sources, 728 edges, 26 components (largest 74).
  identified         174  reliability estimable from co-claims
  sign_ambiguous       0  bipartite component: sign fixed by prior, not data
  prior_only          15  insufficient overlap: reliability is assumed
  -> Not globally identifiable. Reliabilities are comparable across components
     only through the hierarchical prior.

Effective sample size: {'deg_mean': 2.112, 'ess_mean': 1.606, 'deflation_mean': 0.8824,
                        'clusters': 5, 'rho_mean': 0.7227}
Decidability:          {'decidable': 32, 'undecidable': 1249,
                        'decidable_fraction': 0.025, 'target_nats': 2.3026}
```

Three findings worth pausing on, because they are the diagnostics doing their job:

- **26 overlap components** and 15 prior-only sources — reliability is *not* globally
  identifiable, and the report says so instead of quietly assuming otherwise.
- **ESS 1.61 against degree 2.11** — the copy structure was detected (within-cluster
  residual correlation ≈ 0.72) and the evidence deflated accordingly.
- **2.5 % decidable** — under 40 of 1281 assertions carry enough source evidence for *any*
  aggregation rule to decide them. Everything else rests on the structural prior.

## 5. Query a belief

```python
from ocbf.assertions import Family

ref = next(r for r in result.claim_set.refs
           if r.family is Family.E2O and result.belief.is_decidable(r))

print(ref)                                  # e2o(ev_00022, order, order_16)
print(result.belief.prob_true(ref))         # 0.0
print(result.belief.verdict(ref).value)     # 'false'
print(gt.truth(ref))                        # False  ← ground truth agrees
print(result.belief.top_contributors(ref, 4))
```

```text
[('channel:src_00097', -3.794),
 ('silence:src_00108', -3.468),
 ('channel:src_00171', -3.415),
 ('prior',             -3.178)]
```

Those numbers are additive contributions to the posterior **log-odds**, and they sum to the
result. Note `silence:src_00108`: that source never mentioned this link, and because it
declared `COMPLETE_OVER_SCOPE` coverage, its silence is a genuine negative claim worth −3.47
nats.

Attribution costs nothing extra — it is the belief-propagation messages, retained.

## 6. Check against the baseline

```python
from ocbf.baselines import weighted_vote
from ocbf.eval import compare

sources = {s.profile.source_id: s for s in sim.sources}
refs = [r for r in result.claim_set.refs
        if r.family in (Family.E2O, Family.O2O, Family.EVENT_EXISTS)]

print(compare(
    {
        "weighted_vote": weighted_vote(universe.registry, result.claim_set, sources=sources),
        "ocbf": result.belief,
    },
    gt.truth_map(refs),
    refs,
))
```

| method | accuracy | AUC | Brier | ECE |
| --- | --- | --- | --- | --- |
| weighted vote | 0.8374 | 0.9280 | 0.1110 | 0.1339 |
| **OCBF** | **0.9571** | **0.9817** | **0.0355** | **0.0266** |

The calibration error is 5× lower. That is the headline result: not that the model ranks
assertions better, but that its stated probabilities mean what they say.

## What you just did

You fused 2705 claims from 300 mostly-unreliable sources — where the median assertion had a
single witness and only 2.5 % were decidable by voting — into a calibrated belief that beat
the standard baseline on every metric. The structural prior did most of that work; see
[The regime](../explanation/the-regime.md) for why that is not a lucky accident.

## Next steps

- **[Add your own source](../how-to/add-a-source.md)** — swap the simulator for real evidence.
- **[Read the diagnostics](../how-to/read-diagnostics.md)** — what to do when a run reports
  `sign_ambiguous` or near-zero decidability.
- **[The model](../explanation/the-model.md)** — what those 86 838 factors actually are.
