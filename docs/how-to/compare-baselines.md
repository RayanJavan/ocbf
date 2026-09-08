# Benchmark against baselines

Weighted voting is the bar. The truth-discovery literature repeatedly finds that
sophisticated methods barely beat it, so OCBF treats the comparison as mandatory rather than
optional: a fusion system that does not clear weighted voting is not working.

## Run the comparison

```python
from ocbf.baselines import dawid_skene, majority_vote, weighted_vote
from ocbf.eval import compare
from ocbf.assertions import Family

sources = {s.profile.source_id: s for s in my_sources}
refs = [r for r in result.claim_set.refs
        if r.family in (Family.E2O, Family.O2O, Family.EVENT_EXISTS)]

table = compare(
    {
        "majority":    majority_vote(universe.registry, result.claim_set),
        "weighted":    weighted_vote(universe.registry, result.claim_set, sources=sources),
        "dawid_skene": dawid_skene(universe.registry, result.claim_set, sources=sources),
        "ocbf":        result.belief,
    },
    truth,          # Mapping[AssertionRef, bool]
    refs,
)
```

`compare` intersects the evaluation set across methods, so a method that declines to answer
on hard assertions cannot appear to win by abstaining.

## Read the metrics

| metric | measures | direction |
| --- | --- | --- |
| `accuracy` | thresholded correctness | higher |
| `balanced_accuracy` | correctness averaged over classes | higher |
| `auc` | ranking quality, threshold-free | higher |
| `brier` | squared error of the probabilities | lower |
| `log_loss` | proper scoring rule, punishes confident errors | lower |
| `ece` | **calibration** — do stated probabilities match observed frequencies | lower |
| `accuracy_decidable` | accuracy on the decidable subset only | higher |

Watch **ECE** as closely as AUC. The value of fusion in this regime shows up mainly in
calibration, and a system that ranks well while being systematically overconfident is worse
than useless downstream.

## Inspect calibration directly

```python
from ocbf.eval import reliability_diagram

scores = result.belief.binary_scores(refs)
labels = np.array([truth[r] for r in refs], dtype=int)

for row in reliability_diagram(scores, labels, n_bins=10):
    print(f"{row['bin_lo']:.1f}-{row['bin_hi']:.1f}  "
          f"n={row['count']:<5} said {row['mean_confidence']:.3f}  "
          f"was {row['observed_frequency']:.3f}")
```

A well-calibrated model has `mean_confidence ≈ observed_frequency` in every bin. Systematic
gaps in the high-confidence bins are the dangerous ones.

## Benchmark without real ground truth

If you have no labels, use the synthetic generator to benchmark the *regime* rather than your
data, then check that your real claim set has similar degree statistics:

```python
from ocbf.synth import ProcessConfig, SourceRegime, simulate_process, simulate_sources

gt = simulate_process(ProcessConfig(n_orders=40, seed=0))
universe = gt.build_universe()
sim = simulate_sources(universe, gt, SourceRegime(
    n_sources=600,
    accuracy_mean=0.62,       # match your sources' estimated quality
    n_hotspots=12,            # lower = more concentrated = higher deg(a)
    copy_rate=0.3,            # how much duplication do you actually have?
    seed=0,
))
```

Then compare `ClaimSet.summary()` between the synthetic and real sets. If `density`,
`deg_a_median` and `deg_s_median` are close, the synthetic benchmark is informative about
your case.

## Ablate to see what is helping

```python
from ocbf.model import GraphSpec
from ocbf.pipeline import FusionConfig

variants = {
    "full":            FusionConfig(),
    "no structure":    FusionConfig(graph=GraphSpec(
                           include_referential_integrity=False,
                           include_type_gate=False,
                           include_cardinality=False)),
    "no pooling":      FusionConfig(fit_reliability=False),
}
beliefs = {name: fuse(universe, my_sources, cfg).belief for name, cfg in variants.items()}
print(compare(beliefs, truth, refs))
```

Expect *no structure* to cost the most. In a sparse regime the structural prior carries most
of the inference, so removing it degrades the model toward per-assertion voting.

## Test the model's robustness

The suite contains the experiment that justifies the hard/soft split — deliberately
asserting a false schema constraint and measuring the damage at two strengths:

```bash
.venv/Scripts/python -m pytest tests/test_pipeline.py -q -k "soft_beats_hard"
```

Adapt `tests/test_pipeline.py::test_soft_beats_hard_when_the_constraint_is_wrong` to your own
schema to check that a constraint you are unsure about is registered at a strength you can
live with being wrong about.

## Score the continuous layer

Continuous assertions need a proper scoring rule rather than accuracy, and their own
baselines: [`claim_median`][ocbf.baselines.aggregate.claim_median] is the floor and
[`weighted_mean`][ocbf.baselines.aggregate.weighted_mean] is the bar, scored with
[`compare_continuous`][ocbf.eval.continuous.compare_continuous]. The discipline is identical
— always run them, always report them. [Fuse timestamps and
attributes](fuse-continuous.md#check-calibration) has the recipe.

## See also

- [`ocbf.eval`][ocbf.eval] and [`ocbf.baselines`][ocbf.baselines] — the APIs.
- [The regime](../explanation/the-regime.md) — why weighted vote is such a strong baseline.
