# Glossary

Terms as this project uses them, with pointers into the code.

### Assertion

One latent quantity about the world — "does event `e17` exist", "is `item_3` linked to `e17`
under qualifier `ships`". Addressed by
[`AssertionRef`][ocbf.assertions.refs.AssertionRef], grouped into
[`Family`][ocbf.assertions.refs.Family].

### Attribution

The decomposition of a posterior's log-odds into additive contributions — one per source, per
structural factor, plus the prior. Free, because it *is* the retained belief-propagation
messages. See [`BeliefState.attribution`][ocbf.belief.state.BeliefState.attribution].

### Bridge function

The map from a latent correlation to the Kendall's tau it induces *after* the marginal
transforms. Inverting it recovers the latent correlation from ranks alone, which needs far
fewer observations than a parametric fit — the reason the copula is estimable in this regime at
all. [`bridge_correlation`][ocbf.model.copula.bridge.bridge_correlation].

### Chernoff information

`C = −min_t log Σ_y p₀(y)^t p₁(y)^(1−t)` — how much a single source channel can discriminate.
Zero at chance. Additive across independent sources, which is what makes the decidability test
a sum. See [`chernoff_binary`][ocbf.diagnostics.decidability.chernoff_binary].

### Claim

One source speaking about one assertion. [`Claim`][ocbf.sources.claims.Claim]; the indexed
collection is [`ClaimSet`][ocbf.sources.claims.ClaimSet].

### Conditional Gaussian (CG)

The coupling between the discrete backbone and the Gaussian block: a discrete variable shifts
a Gaussian's *mean*, with the covariance shared across states (homogeneous CG). The message to
the discrete side is exact; the message back is a mixture, collapsed by moment matching.
[`CGMeanBank`][ocbf.model.gaussian_banks.CGMeanBank].

### Coverage semantics

What a source's **silence** means: `OPPORTUNISTIC` (nothing), `COMPLETE_OVER_SCOPE` (a
negative claim), `SELECTIVE` (informative via a truth-dependent propensity). Mandatory on
every source, no default. [`CoverageSemantics`][ocbf.sources.base.CoverageSemantics].

### CRPS

Continuous ranked probability score — a proper scoring rule for a distribution over the real
line, and the continuous counterpart of the Brier score. Rewards sharpness and calibration
together, so it cannot be gamed by reporting a confident point.
[`gaussian_crps`][ocbf.eval.continuous.gaussian_crps].

### Decidability

Whether an assertion carries enough *source* evidence for any aggregation rule to decide it.
`Σ Chernoff ≥ log(1/ε)`. A lower bound for the structured model, which also uses structural
evidence. [`ocbf.diagnostics.decidability`][ocbf.diagnostics.decidability].

### `deg(a)` / `deg(s)`

Distinct sources covering an assertion; distinct assertions touched by a source. Both small in
this regime, and that is what everything turns on. Note `deg(a)` counts *sources*, not claims
— two claims from one source are not two votes.

### Effective sample size (ESS)

How many *independent* votes an assertion really has, after the design-effect correction for
within-family correlation. Reported beside `deg(a)`.
[`ocbf.diagnostics.ess`][ocbf.diagnostics.ess].

### Factor bank

Every grounding of one factor template, stored columnar so its messages are a single batched
array operation. [`FactorBank`][ocbf.model.graph.FactorBank].

### Grounding

Instantiating factor templates against the concrete universe. Includes the three prunes, which
are part of the model rather than an optimisation. [`ocbf.universe`][ocbf.universe].

### Identifiability

Whether source reliability is estimable from the overlap structure at all. Requires a
non-bipartite (odd-cycle-containing) component. [`Identifiability`][ocbf.diagnostics.overlap.Identifiability].

### Latent Gaussian copula

Modelling every *ordered* quantity as a monotone image of a latent Gaussian, `v = F⁻¹(Φ(z))`.
Collapses continuous, count, ordinal, binary and truncated types into one Gaussian block.
Unordered categoricals have no such image and stay discrete.
[`ocbf.model.copula`][ocbf.model.copula].

### Marginal transform

The per-template monotone map between an observed value and its latent coordinate. Discrete
kinds are interval-censored by it rather than pinned to a point.
[`MarginalTransform`][ocbf.model.copula.marginals.MarginalTransform].

### `NEG_INF`

The finite stand-in for `−inf` in log-potentials, fixed at `-1e4`. Large enough to underflow
`exp` to zero, small enough that adding an ordinary log-potential preserves it in float64.
[`NEG_INF`][ocbf.model.graph.NEG_INF].

### OCEL 2.0

The [Object-Centric Event Log standard](https://www.ocel-standard.org/): events and objects,
both typed, joined by *qualified* many-to-many relations, with time-varying object attributes.
The structure OCBF holds a belief over.

### Par-factor graph

A parameterised factor graph: templates keyed by schema elements, grounded against instances.
Parameter sharing across groundings is simultaneously *lifting* (relational learning) and
*pooling* (statistics) — the same operation from two literatures.

### Prior-only

An assertion outside the active graph, answered with the structural prior rather than an
error. [`BeliefState.is_prior_only`][ocbf.belief.state.BeliefState.is_prior_only].

### Source cluster / family

A **declared** group of sources expected to share errors — same vendor, upstream model, or
site. Shares a random effect in the reliability GLM, which *is* the dependency correction.

### Triplet method

Closed-form per-source accuracy from pairwise agreement rates among three conditionally
independent sources, with no labels. `a_i² = (M_ij·M_ik)/M_jk`. Used as the initialiser.
[`triplet_accuracies`][ocbf.reliability.moments.triplet_accuracies].

### Two-sided quality

Sensitivity and specificity kept separate rather than collapsed into an accuracy. Necessary
because a detector's *silence* is governed by its false-negative rate.
[`SourceParams`][ocbf.reliability.params.SourceParams].

### Verdict

The three-valued answer `TRUE` / `FALSE` / `UNDETERMINED`. `UNDETERMINED` is not "probability
near 0.5" — it means the evidence is below what any rule would need.
[`Verdict`][ocbf.belief.state.Verdict].
