# OCBF — Object-Centric Belief Fusion

A joint probabilistic belief over a latent **OCEL 2.0** event log, fused from many **sparse,
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

## Why this is not ordinary truth discovery

The sources are highly sparse, individually unreliable, and numerous — and that regime, not
the OCEL format, drives every design decision:

- **Most assertions cannot be decided by voting at all.** The Dawid–Skene error exponent
  puts a hard floor on how much evidence a decision needs, and weak sources with thin
  redundancy do not reach it. The structural prior is therefore the *primary* inference
  mechanism, not a regulariser.
- **Source reliability is not always identifiable — and that is checkable.** It requires a
  non-bipartite overlap graph. OCBF runs the test and reports a per-source verdict.
- **A free parameter per source is unaffordable**, so reliability is a pooled hierarchical
  GLM over source features, declared families, and per-assertion-family specialisation.
- **Dependency is the dominant failure mode.** Twenty sources sharing an upstream model are
  not twenty votes; effective sample size is reported beside raw degree.

## Results

Synthetic order-to-cash process: 60 orders, 600 sources at mean accuracy ≈ 0.62, median 2
sources per assertion. All methods scored on identical assertions against known truth.

| method | accuracy | AUC | Brier | ECE |
| --- | --- | --- | --- | --- |
| majority vote | 0.7716 | 0.8868 | 0.1521 | 0.1513 |
| **weighted vote** *(the bar)* | 0.8100 | 0.9140 | 0.1333 | 0.1158 |
| Dawid–Skene | 0.6512 | 0.8592 | 0.1909 | 0.1857 |
| **OCBF** | **0.9322** | **0.9824** | **0.0488** | **0.0334** |

Weighted vote beating Dawid–Skene reproduces a known and sobering finding from the
truth-discovery literature inside our own harness — which is why it is the mandatory
baseline on every benchmark.

## Quick start

```bash
git clone https://github.com/RayanJavan/ocbf.git && cd ocbf
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[bayes,dev]"
.venv/Scripts/python -m pytest -q          # 154 tests, ~3 min
```

```python
from ocbf.pipeline import fuse, FusionConfig
from ocbf.synth import ProcessConfig, simulate_process, SourceRegime, simulate_sources

gt       = simulate_process(ProcessConfig(n_orders=60, seed=11))
universe = gt.build_universe()
sim      = simulate_sources(universe, gt, SourceRegime(n_sources=600, seed=11))

result = fuse(universe, sim.sources, FusionConfig(outer_iterations=2))
print(result.report())

ref = result.claim_set.refs[0]
result.belief.prob_true(ref)          # calibrated marginal
result.belief.verdict(ref)            # TRUE | FALSE | UNDETERMINED
result.belief.top_contributors(ref)   # which sources moved it, and by how much
```

## Documentation

```bash
.venv/Scripts/python -m pip install -e ".[docs]"
.venv/Scripts/mkdocs serve
```

| section | for |
| --- | --- |
| [Getting started](docs/getting-started/index.md) | installing it and fusing a log end to end |
| [How-to guides](docs/how-to/index.md) | adding sources, reading beliefs, interpreting diagnostics |
| [Explanation](docs/explanation/index.md) | why the regime forces this design |
| [Research notes](docs/explanation/research-notes.md) | literature and formal grounding, ~200 references |
| [Design record](docs/explanation/design-record.md) | the committed model, and §11 on what implementation revised |
| [Limitations](docs/about/limitations.md) | what is not built, approximated, or assumed |

The API reference is generated from docstrings, so it exists only in the built site: serve it
with the commands above, or download the `ocbf-site` artifact from any
[CI run](https://github.com/RayanJavan/ocbf/actions).

## Status

The **discrete backbone** is complete and tested: existence, event type, E2O/O2O links, hard
referential integrity and type gating, soft cardinality, source channels with two-sided
quality and silence handling, the two-block inference loop, and all three diagnostics.

The **continuous layer** is complete and tested too: latent Gaussian copula over timestamps
and ordered attributes, Gaussian EP, and the conditional-Gaussian coupling back to the
backbone. It is additive — a world that declares nothing continuous answers exactly as a
discrete-only run would.

Not built: OCEL 2.0 file I/O, joint log sampling, entity resolution, and the GPU backend.
Full list, with the approximations inside what *is* built, in
[Limitations](docs/about/limitations.md).

## Repository checks

```bash
.venv/Scripts/python -m pytest -q          # library
.venv/Scripts/mkdocs build --strict        # site
```

Both run on every push and pull request ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

`tests/test_exactness.py` is the load-bearing test: it verifies belief propagation against
brute-force enumeration, including the cardinality forward–backward recursion that replaces
a `2^k` factor.
