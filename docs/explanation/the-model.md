# The model

A parameterised factor graph over a latent OCEL 2.0 log. This page explains what the
variables are and why each factor layer exists.

## The latent world

An OCEL 2.0 log is a tuple of typed sets and partial functions. Lifting each component to a
random variable gives the assertion algebra:

| assertion | variable | status |
| --- | --- | --- |
| event existence | `X_e ∈ {0,1}` | latent |
| event type | `T_e ∈ ET` | latent |
| event time | `τ_e ∈ ℝ` | latent — in the copula |
| object existence | `X_o ∈ {0,1}` | latent |
| object type | `T_o` | **clamped** |
| E2O link | `R_{e,q,o} ∈ {0,1}` | latent — **primary target** |
| O2O link | `S_{o,q,o'} ∈ {0,1}` | latent — **primary target** |
| attributes | mixed | latent — ordered kinds in the copula, categoricals discrete |
| identity | partition | **clamped** (toggle) |

Two asymmetries are deliberate.

**A candidate event is a *slot*, not a typed event.** One categorical `T_e` over the
type support is `K` times cheaper than `K` mutually-exclusive binary candidates, and gives
belief propagation a far better-conditioned graph.

**Object type is clamped while event type is latent.** Clamping identity hands us an object
registry, and an OCEL 2.0 registry entry cannot even name its own attributes without its type
— attribute sets are disjoint per object type, so `objtype` is constitutive of object
identity. Event type carries no such dependency, and activity-label uncertainty is the
single most common uncertainty in the domain literature.

## Grounding and pruning

Naive E2O grounding is `|E| × |Q| × |O|` — around 1e12 and hopeless. Three prunes:

1. **Qualifier signature.** Only declared `(event type, qualifier, object type)` triples are
   legal. Because `T_e` is latent, a candidate survives if *some* type in the support permits
   it. Cuts two to four orders of magnitude.
2. **Temporal windowing.** Objects have lifespans; links to events that cannot overlap are
   dropped.
3. **Claim-anchored blocking.** Candidates with no claim and no structural path to one
   contribute only their prior.

Pruning is **part of the model, not an optimisation** — a pruned candidate is an assertion
that its belief is prior-only. Every prune is counted and reported, because a wrong prune is
a silent `-∞`.

## The factor layers

### Structural priors

Existence priors, plus a **derived** link prior. The link prior is computed per
`(event, qualifier)` group from the declared multiplicity: a group of `k` candidates under
`exactly-one` gets `1/k`.

Deriving rather than fixing is a correctness requirement, not tidiness. A flat prior of 0.08
over 50 candidates expects four links while the cardinality factor insists on one — the two
fight, and the posterior depends on their relative weights instead of on the evidence.
Reading both off the same declaration makes them agree by construction. Fixing this converted
a non-converging run into one that converges in 81 iterations.

### Source channels

One unary factor per claim, carrying `log p(y | truth, θ_s)`.

Binary assertions get **two-sided quality** — sensitivity and specificity separately, never
collapsed into an accuracy. Two-sided is what lets a detector's silence carry its
false-negative rate.

Categorical assertions get a **one-parameter spread model** rather than a `K × K` confusion
matrix, with the off-diagonal profile shared across a source cluster. A per-source confusion
matrix is exactly the parameterisation the sparse regime forbids; the spread model keeps
Dawid–Skene's systematic-confusion insight at one parameter per source, and nests up to a
full matrix for sources with enough claims to earn it.

### Sources and coverage

Every source declares what its **silence** means. The three modes are one family:

```text
π_s(a) = sigmoid( w_s · f(a)  +  γ_s · truth(a) )
```

- `γ_s = 0` → `OPPORTUNISTIC`: the propensity does not depend on the truth, so silence
  factors out entirely.
- `γ_s → ∞` → `COMPLETE_OVER_SCOPE`: silence inside scope is a full negative claim.
- otherwise → `SELECTIVE`: the general case.

Because they are one family, a *mis-declared* mode is detectable by fitting `γ_s` and
comparing. The declaration is mandatory with no default, since a silently-wrong default here
biases everything downstream.

### Hard structural factors

**Referential integrity** — a link implies both endpoints exist. This is the edge that turns
"several sources claim links into `e`" into "`e` probably happened", and conversely lets
confidence that `e` did not happen suppress every link into it.

**Type gate** — a link is impossible under event types whose signature forbids it.
Bidirectional: evidence for a link is evidence about the event's type, and vice versa. That
coupling is much of how event-type belief forms at all when few sources speak about types.

!!! note "Why the type gate is claim-anchored"

    Attaching a gate to every candidate link put ~113 hard factors on a single latent type
    variable. Gate messages from *unclaimed* links are driven purely by the link prior, so a
    type permitting many candidates gained an advantage for having many candidates — not
    evidence about anything. Since those links share the event's existence variable, loopy BP
    double-counted that spurious signal and drove the type posterior into a saturated corner
    it oscillated between. Restricting the gate to claimed links cut factors from 21 663 to
    583 and fixed convergence.

### Cardinality

`Σ_o R_{e,q,o} ∈ [lo, hi]`, implemented as a **counting factor**. A naive factor over `k`
links is `2^k`; the forward–backward recursion over the running count is `O(k·C)` with `C`
the capped count domain — effectively linear for the common `exactly-one` case.

This is a hard requirement, not an optimisation: without it a dense E2O neighbourhood is
intractable and one of the strongest structural signals has to be dropped.

## The continuous layer

Timestamps and every *ordered* attribute — continuous, count, ordinal, binary, truncated —
are modelled as monotone images of a latent Gaussian, `v = F⁻¹(Φ(z))`. That one device
collapses five observed types into a single Gaussian block, which is what lets the whole
layer run on one inference mechanism rather than five.

Discrete kinds are **interval-censored** rather than mapped to a point: observing count `k`
says only that `z` fell between two cutpoints. Treating it as a point fabricates precision
the data do not contain, worst exactly where the levels are coarsest.

Latent correlations come from **bridge functions on Kendall's tau** — rank-based, so
estimable from far fewer observations than a parametric mixed MRF, which is what the sparse
regime demands. Where the correlations are allowed to be non-zero is schema-given: an event
attribute with its own event's timestamp, and two attributes of the same object.

!!! note "Unordered categoricals are not in this layer"

    They have no monotone image in a Gaussian, so they stay discrete. This is a real
    limitation of the copula approach rather than an implementation gap, and the type system
    enforces it: [`AttributeKind.in_copula`][ocbf.schema.core.AttributeKind.in_copula] names
    the boundary, and asking for a categorical transform raises rather than guessing.

### One mechanism, four problems

The design's main technical economy. Four apparently different problems are the same
operation — compute the exact tilted moments, project back to a Gaussian, propagate:

| problem | non-Gaussian element |
| --- | --- |
| copula transform of a non-Gaussian marginal | monotone warp |
| ordinal / count observation | interval censoring |
| lifecycle precedence `τ_e < τ_e'` | truncation |
| conditional-Gaussian coupling to a discrete variable | Gaussian mixture |

So adding a new kind of continuous evidence means writing a tilted-moment computation, not
touching the engine. [Inference](inference.md#the-continuous-engine) covers how that engine
runs; this is what it is asked to do.

### Coupling to the discrete backbone

An event's type modulates its timestamp's mean — `Z_τ | T_e = t ~ N(μ_t, σ²)`: homogeneous
conditional Gaussian, mean modulation with a covariance shared across states. The two
directions are deliberately asymmetric.

**To the discrete variable** the message is *exact* — the Gaussian log-partition per state.
This is how a timestamp informs an event's activity label, and it is the direction that makes
the coupling more than a way of sharpening the continuous side.

**To the continuous variable** the exact message is a mixture with one component per state,
collapsed by moment matching. That approximation is where a multimodal continuous posterior
would be lost, so the uncollapsed components stay available beside the collapse and it can be
measured rather than trusted.

### What the layer is worth

Scored against the mandatory baselines with the CRPS — a proper scoring rule, the continuous
counterpart of the Brier score — and with the coverage of a nominal 90 % interval:

| method | CRPS | coverage error |
| --- | --- | --- |
| claim median | 0.0663 | −0.132 |
| inverse-variance weighted mean | 0.0647 | −0.187 |
| **OCBF** | **0.0514** | **−0.048** |

The same story the discrete layer tells: point estimates are close, and what fusion buys is
that the stated uncertainty means what it says. Both baselines are badly overconfident — the
weighted mean most of all, because it trusts its own fitted scales. The gap is widest on
attributes (CRPS 0.089 against 0.112 and 0.119), where the copula's correlations give the
layer something a per-assertion aggregator cannot see.

## Hard and soft

**Definitional rules are hard; domain beliefs are soft.**

| hard | soft |
| --- | --- |
| referential integrity | cardinality / multiplicity |
| attribute-type domain | lifecycle precedence |
| qualifier legality | attribute monotonicity |
| type disjointness | functional uniqueness |

The distinction is measurable. Under a deliberately *false* multiplicity, links carrying
evidence keep a posterior of **0.72** when the constraint is soft and collapse to **0.14**
when it is effectively hard. A wrong hard constraint assigns probability zero to the truth
and no amount of evidence recovers it; a wrong soft one is overruled by a few good claims.

Soft weights are therefore **denominated in evidence** — dimensionless multipliers on the
Chernoff information of a typical single claim, calibrated from the source pool. A raw nat
count cannot hold that meaning: the same number is a nudge against strong sources and an
unbreakable rule against weak ones.

## Templates and pooling

The internal representation is a **parameterised factor graph**. Factor templates are keyed
by schema elements and grounded against the universe, which buys two things:

- **Parameter sharing is pooling.** All groundings of a template share parameters, so a type
  gains statistical strength from every instance. Lifting and statistical pooling are the same
  operation viewed from two literatures.
- **Vectorisation is free.** All groundings have identical message shapes, so their messages
  are one batched array operation — which is what a
  [`FactorBank`][ocbf.model.graph.FactorBank] is.

## Full treatment

[Design record §§1–5](design-record.md).
