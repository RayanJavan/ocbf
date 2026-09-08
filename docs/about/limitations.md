# Limitations

Stated plainly, because a system whose core is an approximation should say where the
approximations are.

## Not implemented

| area | status |
| --- | --- |
| **Joint OCEL sampling** | The contract promises coherent whole-log samples; only marginals are implemented. Downstream process mining consumes logs, so this matters for that use. |
| **Directly-follows prior** | The object-type-conditioned behavioural prior is specified but not built. |
| **Entity resolution** | Identity is clamped. The latent path is designed but out of scope. |
| **OCEL 2.0 file I/O** | No reading or writing of the standard formats yet. |
| **GPU backend** | The engine is numpy. Templated banks are structured for batched torch, which is where the ~100× headroom is. |
| **Continuous decidability** | The Chernoff bound behind the decidability flag is derived for binary channels. Continuous assertions get a posterior and a credible interval, but no three-valued verdict. |

## Approximations in what *is* built

### Discrete backbone

**Counting-factor saturation.** The cardinality count domain saturates at `hi + cap_slack`.
Violations beyond the cap are penalised as if *at* the cap, so gross over-linking is
under-penalised. The slack keeps the flat region away from the interesting range.

**Claim-anchored type gate.** The type gate attaches only to links carrying a claim. An
unclaimed link therefore keeps its prior mass even under an event type that forbids it. Those
assertions are prior-only and flagged as such, so the relaxation is confined to assertions
where the belief state makes no claim anyway. The exact fix — a single higher-order factor
over `(T_e, all links of e)` — is deferred.

**Conservative cardinality under latent type.** Where an event's type support disagrees about
a qualifier's multiplicity, the most permissive reading is used. Never penalises a
configuration some plausible type allows, at the cost of dropping the constraint where types
disagree.

### Continuous layer

**Unordered categorical attributes are outside the copula.** They have no monotone image in a
Gaussian, so they stay in the discrete layer. Counted in the prune report rather than assumed
away.

**Event attributes need an unambiguous type.** OCEL 2.0 requires attribute sets to be disjoint
across types, so an event attribute is inherently type-specific — and under a latent `T_e`
whose support is wider than one type, whether the attribute applies at all is uncertain. A
latent Gaussian coordinate has no `NA` value to express that, so an event attribute enters the
copula only where *every* type in the support declares it, and is otherwise prior-only and
counted. Object attributes are unaffected, since object type is clamped.

**Conditional-Gaussian messages are moment-matched.** The message from the type-coupled
timestamp factor to the continuous side is exactly a mixture with one component per event
type, and it is collapsed to a Gaussian; multimodal continuous posteriors are lost. The
uncollapsed components are retained beside the collapse, so the gap can be measured on any
variable rather than assumed. The message in the other direction is exact.

**Homogeneous conditional Gaussian only.** Discrete variables shift Gaussian means; they never
change covariances. Heterogeneous CG costs the closed-form message and is deferred.

**Lifecycle precedence is weighted by a mean field.** Whether a declared ordering applies to a
pair of candidate events depends on discrete variables, and the truncation factor carries the
posterior probability that the configuration holds rather than the exact higher-order factor.
That product is small wherever the links are unclaimed, so precedence is claim-anchored in
effect — it acts where the discrete layer is confident and nowhere else.

### Inference

**Neither engine has a convergence guarantee** on this graph, and both report rather than
pretend. Check `bp_converged` and `ep_converged` before trusting a run.

Loopy BP is mitigated by damping, monitoring, [exact
verification](../explanation/inference.md#the-exact-oracle) on small subgraphs and honest
non-convergence reporting — not by hoping. Expectation propagation has one specific
failure worth recognising: on a step-warped coordinate, such as an ordinal attribute carrying
many claims, the tilted distribution is piecewise and the run settles into a small limit cycle
rather than a fixed point. `ep_belief_residual` gives its amplitude in latent standard
deviations, which is typically far below any decision-relevant resolution — but the flag stays
`False`, because it is.

**MAP for the parameter block.** Point estimate by default. `reliability_method="nuts"` gives
the full posterior when the reliability estimates are themselves the object of interest.

## Assumptions

**Declared source families are trusted.** Two sources secretly sharing an upstream model while
declaring different `cluster_id`s will not be dependency-corrected, and the model will be
overconfident. This is a deliberate trade — learning the dependency graph also needs overlap
a sparse regime does not have — but it puts weight on honest provenance.

**Fixed universe.** An event that no source mentions and no candidate slot anticipates cannot
be inferred. Open-universe inference would need reversible-jump MCMC and is an explicit
non-goal.

**Better-than-chance prior.** Both the triplet initialiser and the GLM intercept assume
sources are better than chance. This breaks a real likelihood symmetry the data cannot break
on its own — the mirror solution where every reliability is inverted has identical likelihood.
Where the overlap graph is bipartite, this prior is the *only* thing resolving the sign, and
the diagnostic says so.

**Decidability measures source evidence only.** For the structured model it is a lower bound
on what is resolvable, not a prediction of accuracy. Do not read `decidable_fraction` as "how
much of the output is trustworthy".

## Scale

Tested to ~1.4e5 variables and ~8.5e5 edges. The design targets 1e7, and memory analysis says
there is room, but that has not been demonstrated. At present the engine is CPU-bound by
roughly two orders of magnitude.

## Environment

`gtsam` is optional and nothing in the pipeline calls it. It backs the
[exact oracle](../explanation/inference.md#the-exact-oracle), so a machine without it loses a
check rather than a capability — the suite skips those tests and passes. On Windows a
CUDA-enabled build needs the CUDA runtime on the DLL search path;
[`ocbf.backends`][ocbf.backends] handles that and reports where it looked, and
[Installation](../getting-started/installation.md) has the commands.

`pyagrum` and `problog` are declared in the `oracles` extra and unused — a deviation from
[design record §6.3](../explanation/design-record.md#63-engine-tiers), recorded in
[§11.13](../explanation/design-record.md#1113-gtsam-took-the-tier-1-oracle-role-pyagrum-was-specified-for).
