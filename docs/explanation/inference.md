# Inference

## The two-block scheme

```text
initialise:  triplet accuracies  →  weighted vote  →  initial beliefs

repeat:
    BELIEF BLOCK      (huge, sparse, ~1e7 variables)
        given θ, run BP over the discrete backbone      → marginals
        given those marginals, run EP over the Gaussian → continuous posteriors
                                                        → exact message back to the backbone

    PARAMETER BLOCK   (small, dense, ~1e2–1e3 parameters)
        given marginals, fit the hierarchical reliability GLM
```

This is variational EM with a structured mean-field split, and the split is chosen from the
theory rather than for convenience: the assertion block is huge and sparse, which is exactly
where belief propagation is provably near-optimal; the parameter block is small, which is
exactly where real Bayesian inference is affordable and where pooling lives.

Running MCMC over ten million assertion variables would be both slower *and* worse.

The two halves of the belief block are ordered, not interleaved. Belief propagation runs
first because the continuous factors that need discrete context — lifecycle precedence, and
the type–timestamp coupling — ask which links and types the backbone believes in. The
continuous layer's reply is a unary log-potential on the coupled type variables, and it enters
the next sweep as an ordinary bank, attribution included.

## Why belief propagation

On sparse, locally tree-like claim graphs, BP **exactly matches the fundamental limit** under
Dawid–Skene (arXiv:1602.03619), and iterative message passing on sparse regular task–worker
graphs is order-optimal (arXiv:1110.3564). So BP is theoretically indicated here, not merely
tractable.

The graph is nonetheless loopy by construction — cardinality factors couple every link of an
event, and referential integrity couples those links back through shared endpoints — so BP
comes with hygiene rather than guarantees:

- **damping**, defaulting high (0.8) because that loop structure oscillates undamped;
- **belief-residual convergence**, not message-residual;
- **oscillation detection**;
- an iteration cap that **reports** non-convergence rather than silently returning the last
  sweep.

!!! note "Why convergence is judged on beliefs"

    With hard factors the message residual is dominated by edges swinging between finite
    values and the `NEG_INF` sentinel — a swing that says nothing about whether the posterior
    settled. Beliefs are what callers consume, so beliefs are what must converge.

## The sentinel's magnitude is load-bearing

Hard factors are `-1e4`, not `-inf` and not `-1e30`. The value has to satisfy two constraints
at once:

- large enough that `exp(NEG_INF)` underflows to exactly zero;
- **small enough that adding an ordinary log-potential preserves it**.

`-1e30 + 5.0 == -1e30` in float64. That annihilates the finite part — and BP subtracts a
variable's own outgoing message from its belief, so with hard factors both terms sit near the
sentinel and the subtraction is only meaningful if the finite remainder survived the addition.
At `-1e4` it does, with room for a hundred stacked constraints.

## The exact oracle

Belief propagation has no convergence guarantee on this graph, so the approximation has to be
*measured* rather than assumed. That needs an engine whose answer is exact.

Brute-force enumeration is the obvious one, and it is what `tests/test_exactness.py` uses. It
costs the product of the cardinalities, so it runs out at a handful of variables — well short
of the loopy structure the engine actually meets, which is the only place the approximation is
interesting. Exact *elimination* costs exponentially in the **treewidth** instead, so a dozen
cardinality groups coupled through shared endpoints is hopeless to enumerate and unremarkable
to eliminate. [`ocbf.inference.gtsam_exact`][ocbf.inference.gtsam_exact] translates the
discrete backbone into GTSAM's `DiscreteFactorGraph` and does exactly that;
[`compare_to_exact`][ocbf.inference.gtsam_exact.compare_to_exact] reports the gap.

Elimination has no iteration cap — it finishes or it exhausts memory — so its limits have to
be predictive rather than reactive.
[`elimination_cost`][ocbf.inference.gtsam_exact.elimination_cost] bounds the largest
intermediate factor the ordering will build, and a 36-variable graph can be far more expensive
than a 60-variable one.

The oracle is optional and never on the pipeline's path: it is imported lazily through
[`ocbf.backends`][ocbf.backends], and a machine without GTSAM loses a check rather than a
capability. See [design record §11.13](design-record.md#1113-gtsam-took-the-tier-1-oracle-role-pyagrum-was-specified-for)
for why this backend rather than the one originally specified.

## Initialisation

EM on a non-convex truth-discovery objective has bad local optima, and the literature's answer
is a moment-based initialiser. Ours is the **triplet method**: for three conditionally
independent sources, per-source accuracies have a closed form from pairwise agreement rates
alone, with no labels.

```text
E[y_i y_j] = a_i a_j      ⟹      a_i² = (M_ij · M_ik) / M_jk
```

Its limits are reported rather than hidden. It needs overlap — three sources co-claiming
enough assertions — which a thin claim graph often lacks; and it determines `a_i` only up to
**sign**. Resolving that by assuming better-than-chance is the identifiability
symmetry-breaker, not a derived fact, and it is exactly the ambiguity a bipartite overlap
component cannot resolve at all.

Where triplets are unavailable, confidence-aware weighting takes over.

## The continuous engine

The Gaussian block runs the *same loop* over a different message algebra: scatter-add,
belief, cavity by subtraction, per-bank update, damped write-back. Log-potential rows become
natural parameters `(precision, potential)`, and `logsumexp` becomes moment matching. Anyone
who can read one engine can read the other.

That it is one loop rather than four is [the model's
claim](the-model.md#one-mechanism-four-problems), not this engine's: a warp, a censoring, a
truncation and a mixture are four ways of computing a tilted moment, so they are four banks
the engine cannot tell apart.

One thing is easier here than on the discrete side. The copula makes every latent coordinate
a standard normal, so a single tolerance is meaningful across a block mixing timestamps in
hours with prices in euros — nothing to tune per deployment.

The same hygiene applies, and convergence is likewise judged on beliefs. Three failure modes
are specific to expectation propagation, and each is handled explicitly rather than hoped
away:

- **An improper cavity.** Removing a site's own contribution can leave non-positive
  precision. That edge is skipped for the sweep rather than clipped, because clipping would
  invent evidence the graph does not contain.
- **A tilted distribution that underflows.** A claim far out in the cavity's tail leaves no
  quadrature mass. The bank contributes no site rather than a fabricated one, and the count
  is reported.
- **Synchronous overshoot.** A variable carrying `d` sites is updated against a cavity that
  all `d` are simultaneously moving, so its step is about `d` times too large. Each edge's
  step is divided by its variable's degree.

!!! warning "Damping and the iteration cap are one decision"

    Scaling the step by degree buys a residual that falls monotonically rather than wobbling,
    which is what makes convergence distinguishable from oscillation. It costs proportionally
    more sweeps. Change one setting without the other and the engine stops before the
    precision has accumulated, which surfaces as intervals that are too wide rather than as an
    error — the quietest possible failure.

## The parameter block

```text
logit(θ_{s,f}) =  b0                # intercept, prior mass above chance
                + x_s · β           # source features
                + u_{c(s)}          # declared-family random effect
                + m_f               # assertion-family main effect
                + δ_{c(s), f}       # specialisation interaction
                + ε_s               # individual, shrunk
```

The likelihood uses **soft counts** from the current belief state, which makes this the exact
M-step of the EM. Because those counts are fractional, the observation enters as a weighted
log-likelihood potential rather than a Binomial — same objective, without pretending the
counts are integers.

`u_{c(s)}`, shared within a declared family, *is* the dependency correction: sources sharing a
vendor or upstream model share the effect, so their agreement is partly explained rather than
counted as independent confirmation.

!!! warning "MAP collapses hierarchical variances, silently"

    The joint mode of a hierarchical model sits at zero scale with every random effect
    collapsed. It reports a clean fit while switching off exactly the partial pooling the
    sparse regime depends on — a particularly bad failure to have go unnoticed, and one this
    project did hit.

    The fix is Gamma priors whose mode is away from zero, plus non-centred random effects.
    Measured effect: reliability recovery +0.507 → +0.611, accuracy 0.9078 → 0.9434, ECE
    0.0625 → 0.0312.

## Attribution comes free

For a binary assertion the posterior log-odds decomposes additively over incoming messages:

```text
logit p(a)  =  logit prior(a)  +  Σ_f  log( m_{f→a}(1) / m_{f→a}(0) )
```

Each term is one factor's contribution — one source's channel, the cardinality factor, the
prior. **Attribution is not extra computation; it is the retained messages.** It is also why
no black-box inference library could serve as the core engine.

A Gaussian belief decomposes the same way, one level down: its *mean* is additive over
incoming potentials where a binary posterior's log-odds are additive over incoming messages.

```text
mean(z)  =  η_prior / λ_total  +  Σ_f  η_f / λ_total
```

So an unexpected timestamp is traceable to the source, the bracket or the precedence factor
that moved it, on the same terms and at the same cost.

## Performance

At 845k edges the engine holds ~15 MB of live message arrays and peaks near 50 MB, against
470 ms per iteration. It is **CPU-bound by roughly two orders of magnitude**, not
memory-bound: the cost is Python and allocation overhead in the bank loop, not arithmetic.
Replacing `scipy.special.logsumexp` with a direct implementation removed about two thirds of
engine runtime.

The planned torch/GPU backend is the right lever, and graphs could grow ~100× before memory
becomes the constraint.

## Full treatment

[Design record §6](design-record.md), and §11 for the findings that revised it.
