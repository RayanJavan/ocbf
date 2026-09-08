# OCBF

**Object-Centric Belief Fusion** — a joint probabilistic belief over a latent
[OCEL 2.0](https://www.ocel-standard.org/) event log, fused from many **sparse,
individually unreliable, structurally-typed** sources.

```text
sources (sparse, weak, numerous)        OCEL 2.0 schema (clamped)
              │                                   │
              ▼                                   ▼
         ClaimSet ────────▶ par-factor graph ◀──── candidate universe
                                   │
        loopy BP  +  Gaussian EP  ◀─┴─▶  hierarchical reliability GLM
        (discrete)   (continuous)
                                   │
                                   ▼
                BeliefState:  marginals · decidability flags · attribution
                       +  identifiability / effective-sample-size diagnostics
```

## What it is for

You have the *structural skeleton* of a world — an OCEL 2.0 type system, candidate events
and objects, legal qualifier signatures — and many probabilistic sources, each with a known
signal skeleton. The sources are **highly sparse, individually unreliable, and numerous**.
OCBF turns their claims into a calibrated joint belief over the whole log.

```python
from ocbf.pipeline import fuse

result = fuse(universe, sources)
ref = AssertionRef.e2o("ev_00022", "order", "order_16")

result.belief.prob_true(ref)           # 0.041   — a calibrated posterior
result.belief.verdict(ref)             # FALSE   — or UNDETERMINED, if nothing could decide it
result.belief.top_contributors(ref, 2) # [('channel:src_00097', -3.794), ('prior', -3.178)]
result.overlap.explain()               # whether the reliabilities were estimable at all
```

Those last two lines are the point as much as the first. With sources this weak, a
probability without its provenance and its preconditions is a number, not an answer — so
attribution and diagnostics come back from every run rather than on request.

Why the sparse, unreliable, numerous regime forces that: [The
regime](explanation/the-regime.md).

## Results

Synthetic order-to-cash process: 60 orders, 600 sources at mean accuracy ≈ 0.62, median 2
sources per assertion. Ground truth known; all methods scored on identical assertions.

| method | accuracy | AUC | Brier | ECE |
| --- | --- | --- | --- | --- |
| majority vote | 0.7716 | 0.8868 | 0.1521 | 0.1513 |
| **weighted vote** *(the bar)* | 0.8100 | 0.9140 | 0.1333 | 0.1158 |
| Dawid–Skene | 0.6512 | 0.8592 | 0.1909 | 0.1857 |
| **OCBF** | **0.9322** | **0.9824** | **0.0488** | **0.0334** |

!!! note "Why weighted vote is the bar"

    Weighted vote beating Dawid–Skene here reproduces a known and sobering finding from the
    truth-discovery literature inside our own harness. A fusion system that does not clear
    weighted voting is not working, so OCBF runs and reports it on every benchmark.

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting started](getting-started/index.md)**

    Install it and fuse a synthetic log end to end.

-   :material-wrench: **[How-to guides](how-to/index.md)**

    Add your own sources, read a belief state, interpret the diagnostics.

-   :material-book-open-variant: **[Explanation](explanation/index.md)**

    Why the regime forces this design, and the full research and design record.

-   :material-api: **[API reference](reference/ocbf/index.md)**

    Generated from the source, module by module.

</div>

## What it covers

The **discrete backbone** holds belief over existence, event type, and E2O and O2O links,
under hard referential integrity and type gating, soft cardinality, and source channels with
two-sided quality and silence handling.

The **continuous layer** holds belief over timestamps and ordered attributes, through a latent
Gaussian copula coupled back to the backbone. It is additive: a world that declares nothing
continuous answers exactly as a discrete-only run would.

Every run returns the diagnostics that say when not to trust it — identifiability, per-assertion
decidability, and effective sample size — and every posterior carries the attribution that
produced it.

OCBF does not read or write OCEL 2.0 files, sample whole logs, or resolve entity identity.
[Limitations](about/limitations.md) states the full boundary, including the approximations
inside what it does cover.
