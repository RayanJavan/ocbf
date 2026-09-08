# The regime

Sources are **highly sparse, individually unreliable, and numerous**. This is not the regime
most truth-discovery work optimises for, and it changes the answers rather than the
constants.

## Formal setting

Let `A` be the latent assertion variables and `C ⊆ S × A` the claim-incidence graph:
`(s, a) ∈ C` iff source `s` speaks about assertion `a`.

- **Sparse** — `|C| ≪ |S|·|A|`. Both degrees are small: `deg(s)` (assertions per source) is
  small, and `deg(a)` (sources per assertion) is small.
- **Unreliable** — per-source accuracy is barely above chance; per-source Chernoff
  information is small.
- **Numerous** — `|S|` is large.

Four consequences follow, each with a citation and each visible in the code.

---

## 1. Most assertions are undecidable from votes alone

Under Dawid–Skene the exact error exponent for aggregating `m` sources is `m·I(π)`, where
`I(π)` is the pool's average Chernoff information. Reaching misclassification `ε` requires

```text
m  >  log(1/ε) / I(π)
```

In our regime `I(π)` is small *and* `m = deg(a)` is small. So for a large fraction of
assertions, **no aggregation rule whatsoever** can decide them. On the default benchmark
only ~2.5–15 % clear the bar, depending on source density.

This is an information-theoretic statement, not a limitation of any particular estimator.
Two things follow:

- **Abstention must be first-class.** An assertion below threshold gets an explicit
  `UNDETERMINED` verdict rather than a confidently-wrong 0.51. A posterior that cannot
  distinguish "balanced evidence" from "no evidence" hides the difference that matters most
  when sources are weak.
- **The structural prior is the primary inference mechanism.** Belief must flow along the
  OCEL topology from well-covered assertions into thin ones. That is not a refinement; it is
  the difference between decidable and not.

Implemented in [`ocbf.diagnostics.decidability`][ocbf.diagnostics.decidability].

---

## 2. Identifiability is a graph property, and it is checkable

Define the **source overlap graph**: nodes are sources, and an edge joins two that co-claim
an assertion. Two sharp results:

- Single-coin worker skills are asymptotically identifiable **iff** the limiting interaction
  graph is irreducible **and contains an odd cycle** (arXiv:1706.06660).
- Equivalently, viewing skill estimation as rank-one correlation-matrix completion, skills
  are recoverable **iff the sampling pattern has no bipartite connected component**
  (arXiv:1904.11608).

"Contains an odd cycle" and "is not bipartite" are the same statement, so the test is a
two-colouring BFS in `O(V+E)`.

This matters practically because sparse deployments really do produce islanded overlap
graphs, and a bipartite island silently admits the mirror solution where every reliability is
inverted and every truth flipped. Detecting that and saying so beats returning a confident
inversion.

Implemented in [`ocbf.diagnostics.overlap`][ocbf.diagnostics.overlap].

---

## 3. One free parameter per source is unaffordable

A source with three claims has an accuracy estimate with a standard error near 0.29. Four
compatible remedies, all used:

| remedy | mechanism |
| --- | --- |
| **Hierarchical pooling** | thin sources shrink to the population; heavy ones dominate their own estimate |
| **Source clustering** | latent or declared families give strength where individual data cannot |
| **Task specialisation** | reliability is indexed by `(source, assertion family)`, since a barcode scanner is excellent at identity and useless at activity semantics |
| **Feature regression** | accuracy as a function of observable source properties turns `|S|` parameters into `d ≪ |S|`, and generalises to unseen sources |

These compose into one hierarchical GLM. Measured effect: correlation with true source
sensitivity rises from **+0.12** (label-free triplet initialiser) to **+0.61** (pooled fit).

Implemented in [`ocbf.reliability.hierarchical`][ocbf.reliability.hierarchical].

---

## 4. Sparsity is good for the *algorithm*

The one piece of luck. Sparse claim graphs are locally tree-like, and that is exactly where
message passing is asymptotically exact:

- On sparse regular task–worker graphs, iterative message passing is order-optimal and
  analysable by density evolution (arXiv:1110.3564).
- Belief propagation **exactly matches the fundamental limit** under Dawid–Skene
  (arXiv:1602.03619) — the strongest optimality statement in this literature.

So a factor-graph plus BP core is the *theoretically indicated* engine here, not merely a
convenient one. Sampling belongs in the small parameter block, not over millions of assertion
variables.

---

## Two more consequences worth stating

**Sparse coverage makes missingness informative.** A source touching 0.1 % of assertions is
not sampling uniformly — it looks where it looks. Treating silence as "no information"
discards the strongest signal a detector carries: its false-negative rate. Hence the
mandatory coverage declaration on every source.

**Numerous sources make dependency the dominant failure.** `k` near-duplicate sources are not
`k` votes, and with weak sources, over-counting correlated evidence is the fastest route to
confident error. Because learning a dependency graph *also* needs overlap we do not have,
OCBF uses declared source families instead — provenance the operator already knows beats an
under-determined estimate.

---

## The sobering context

Three papers should temper expectations about what fancier reliability modelling buys:

- **"Truth Finding on the Deep Web: Is the Problem Solved?"** (arXiv:1503.00303) — on real
  data, sophisticated methods often barely beat weighted voting; the error mass is in
  copying, extraction noise and semantic mismatch.
- **"Truth Discovery Algorithms: An Experimental Evaluation"** (arXiv:1409.6428) — twelve
  algorithms, no uniform winner.
- **"Calibrated Trust, Not Sharper Prediction"** (arXiv:2608.14617) — a full fusion stack
  sharpened nothing over its raw estimator; the value was in calibration.

The design implication is direct, and it is why this project spends its budget where it
does: **structure, dependency handling and calibration have good returns; ever-fancier
reliability parameterisations do not.** And weighted vote is reported on every benchmark,
because it is a genuinely hard bar.

## Full treatment

[Research notes §4](research-notes.md) develops all of this with the complete citation trail.
