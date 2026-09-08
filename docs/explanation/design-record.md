# Stage 2 — Model Design & Program Architecture

**Project:** OCBF — Object-Centric Belief Fusion
**Input:** the theory and regime analysis in [research notes](research-notes.md).
**Output:** a committed model, a committed inference architecture, and a module map ready for Stage 3.

---

## 0. Decisions carried in from Stage 1

| # | question | decision |
|---|---|---|
| 1 | universe | **Fixed universe, schema-certain.** Candidates enumerated up front; no transdimensional inference. |
| 2 | clamping | Schema **strictly clamped**. Instance **identity clamped by default** (toggle). Instance **existence and especially links are latent**. |
| 3 | reliability | **Hierarchical GLM** with worker-task-style **specialisation**. Independent per-source MLE explicitly rejected. |
| 4 | coverage | Support all three: `opportunistic`, `complete_over_scope`, `selective`. |
| 5 | behavioural prior | **Object-type-conditioned directly-follows counts** as the baseline. |
| 6 | constraints | **Hard zero factors** for definitional schema rules; **soft high-weight penalties** for domain beliefs. |
| 7 | attribute layer | **Latent Gaussian copula.** |
| 8 | inference contract | calibrated marginals · joint samples · decidability flags · attribution vectors. |
| 9 | evaluation | synthetic OCEL generator · weighted-vote baseline · sensitivity to a misspecified constraint. |
| 10 | diagnostics | identifiability tests and effective-sample-size metrics are **core outputs**, not logging. |

Two consequences follow immediately and shape everything below:

- Decision 2 makes **E2O and O2O link variables the primary inferential target**. The design is optimised around them.
- Decision 2 also switches **entity resolution off by default**, which removes the quadratic candidate blowup and the transitivity constraints. This is the single largest simplification available, and it is why a fixed-universe factor graph is tractable at all.

---

## 1. The latent world

### 1.1 Variables after clamping

Given schema `Sigma` and a fixed candidate universe `U`:

| variable | domain | status | notes |
|---|---|---|---|
| `X_e` | `{0,1}` | **latent** | does candidate event slot `e` correspond to a real occurrence |
| `T_e` | `ET` | **latent** | event type / activity |
| `tau_e` | `R` | **latent** | timestamp |
| `X_o` | `{0,1}` | **latent** | does candidate object `o` really exist |
| `T_o` | `OT` | **clamped** | travels with identity (see §1.2) |
| `R_{e,q,o}` | `{0,1}` | **latent — primary target** | qualified E2O link |
| `S_{o,q,o'}` | `{0,1}` | **latent — primary target** | qualified O2O link |
| `V_{e,a}` | mixed | **latent** | event attribute |
| `V_{o,a}(.)` | mixed path | **latent** | object attribute trajectory |
| identity | partition | **clamped** (toggle) | see §1.2 |

A **candidate event** is a *slot* — a potential occurrence — carrying latent existence, type, and time. This is deliberately different from "a typed candidate per (slot, type) pair with mutual exclusion": one categorical `T_e` is `K` times cheaper than `K` mutually-exclusive binaries and gives BP a much better-conditioned graph.

### 1.2 The identity asymmetry, and why it is principled

Object type `T_o` is clamped while event type `T_e` is latent. This is not an oversight:

- Identity being clamped means we are handed an **object registry**: a list of distinct real-world entities. A registry entry that does not declare its type is not a registry entry — `objtype` is constitutive of object identity in OCEL 2.0 (attribute sets are disjoint per object type, so you cannot even name an object's attributes without its type).
- Event type carries no such dependency. A source reporting "something happened at the loading bay at 14:03" is a genuine observation with an uncertain activity label. This is the single most common uncertainty in the domain literature (arXiv:1910.00089 lists activity-label uncertainty first).

When the identity toggle is switched to latent, `T_o` becomes latent with it, and the ER machinery of [research notes](research-notes.md) §8 activates. That path is designed for but **out of scope for v1**.

### 1.3 Scale targets

| quantity | v1 target | mechanism |
|---|---|---|
| candidate events | 1e4 – 1e6 | given |
| candidate objects | 1e4 – 1e6 | given |
| **E2O candidate links** | 1e5 – 1e7 | after pruning (§2.3) |
| sources | 1e2 – 1e5 | numerous, by assumption |
| claims | 1e5 – 1e8 | sparse incidence |
| reliability parameters | 1e2 – 1e3 | **after pooling** — this is the point of §5 |

The last row is the design's central bet: a world with 1e7 latent variables and 1e5 sources reduced to a *few hundred* free statistical parameters.

---

## 2. Layer 0 — Schema, universe, grounding

### 2.1 The par-factor graph

The internal representation is a **parameterised factor graph** (arXiv:1107.4966). We declare *factor templates* keyed by schema elements and **ground** them against the universe:

```
FactorTemplate(key = (event_type, qualifier, object_type), family = ..., params = theta[key])
    .ground(universe)  ->  [Factor(scope = concrete variables), ...]
```

Two properties fall out of this that we exploit hard:

1. **Parameter sharing = pooling.** All groundings of a template share parameters. This is [research notes](research-notes.md) §6.2's observation that lifting and statistical pooling are the same operation, and it is the reason a sparse-evidence regime is survivable.
2. **Vectorisation = GPU.** All groundings of a template have identical arity and message shape, so their BP messages are one batched tensor op. The GPU is used for *templated message passing*, not for a neural network. This is the right use of the hardware.

### 2.2 Hard schema factors (decision 6, definitional half)

Encoded as `-inf` log-potentials. These carry **no parameters** and are the cheapest information in the system — decisive when evidence is thin.

| constraint | encoding |
|---|---|
| referential integrity | `R_{e,q,o}=1 => X_e=1 and X_o=1`; likewise `S` |
| attribute domain gating | `V_{e,a} = NA` unless `T_e = eatype(a)`; `NA` is an explicit absorbing state in the variable's domain |
| qualifier legality | handled by **pruning at grounding time**, not by a factor (§2.3) |
| type disjointness | structural; enforced by construction |

The `NA` absorbing state matters. Because `T_e` is latent, "does this attribute even apply" is itself uncertain, so it cannot be resolved at grounding time. Giving every attribute variable an explicit `NA` value makes the gate a clean deterministic factor instead of a variable-existence problem.

### 2.3 Grounding and pruning — the scalability lever

Naive E2O grounding is `|E| x |Q| x |O|`, which is 1e12+ and hopeless. Three prunes, applied in order, in the grounding pass:

1. **Qualifier signature.** `Sigma` declares which `(event_type, qualifier, object_type)` triples are legal. Since `T_e` is latent, a link `R_{e,q,o}` survives iff *some* type in `T_e`'s support makes it legal. Typically cuts 2–4 orders of magnitude.
2. **Temporal windowing.** An object has a lifespan; links to events far outside it are pruned. Requires only a coarse bound on `tau_e`.
3. **Claim-anchored blocking.** A candidate link with no source claim *and* no structural path to a claimed variable within `k` hops contributes only its prior — it is materialised lazily, as a prior-only marginal, and never enters the graph.

Prune 3 is the one that makes the sparse regime work *for* us: sparse claims mean the *active* graph is far smaller than the candidate universe. The belief state must still answer queries about pruned links, which it does by returning the structural prior — decision 8's "graceful degradation" requirement, satisfied by construction.

**Pruning is part of the model, not an optimisation.** A pruned link is asserted to have prior-only belief. This must be recorded and reported, because a wrong prune is a silent `-inf`.

---

## 3. Layer 1–2 — The structural and behavioural prior

Per [research notes](research-notes.md) §4.2, the prior is doing most of the work. It gets a correspondingly serious treatment.

### 3.1 Cardinality as counting factors

Schema declares multiplicity `[lo, hi]` for qualifier `q` on event type `t`:

```
phi_card(e, q)  =  soft_penalty( sum_o R_{e,q,o} , [lo,hi] )   conditioned on T_e = t
```

A naive factor over `k` links is `2^k`. The **counting-factor** structure gives exact sum-product messages in `O(k * (hi-lo+2))` by forward-backward over the running sum — the same dynamic program used for cardinality potentials in standard BP. This is what makes a dense E2O neighbourhood tractable, and it is a hard requirement on the implementation, not an optimisation.

Per decision 6, cardinality is a **domain belief** and therefore soft: multiplicity `exactly-1` becomes a steep penalty, not a hard constraint. Real logs violate their own schemas.

### 3.2 Object-type-conditioned directly-follows (decision 5)

For object `o` of type `ot`, the events linked to `o` form its trace. The DF prior wants

```
phi_DF(o) = sum over consecutive (e,e') in trace(o) of  log P_ot( T_e' | T_e )
```

**The difficulty:** "consecutive in `trace(o)`" is a functional of `X`, `R`, and `tau` jointly — a combinatorial object, not a fixed scope. Writing it as a factor is the central modelling problem of this layer.

**Committed solution — decayed pairwise relaxation (`DF-relaxed`, default).** Replace exact consecutiveness with a pairwise factor over ordered pairs on the same object, down-weighted by temporal separation:

```
phi_DF(e, e', o) = w(Delta_tau) * 1[R_{.,.,o}=1] * 1[R_{.,.,o}'=1] * 1[tau_e < tau_e']
                     * log P_ot( T_e' | T_e )
```

with `w(d) = exp(-d / ell_ot)` a decay at object-type-specific length-scale `ell_ot`. As `ell -> 0` this concentrates on genuinely adjacent pairs; as `ell -> inf` it becomes an eventually-follows prior. It is pairwise, its scope is fixed, and it is BP-friendly.

This is an approximation and is documented as one: it double-counts across intervening events. The counter-argument for accepting it: DF counts are themselves only a first-order summary of behaviour, and the decay kernel's error is second-order relative to that. Spending a combinatorial budget to be exact about an already-approximate prior is bad economics.

**Alternative — `DF-exact` (available, off by default).** Introduce successor indicators `N_{e,e',o} in {0,1}` with "at most one successor per (event, object)" and "at most one predecessor" constraints. This is a degree-constrained assignment problem embedded in the factor graph — correct, and solvable by BP (it is the classic bipartite matching BP), but it multiplies the link-layer variable count. Offered as a toggle for small graphs and for validating that `DF-relaxed` does not distort results.

**Where the counts come from.** `P_ot(. | .)` is estimated by EM from the current belief state (soft DF counts under the posterior over `R`, `T`, `tau`), with a Dirichlet prior. It is a *learned* prior, refit each outer iteration — which is exactly the "prior carries the thin assertions" mechanism of [research notes](research-notes.md) §4.2, made concrete.

### 3.3 Lifecycle precedence

Per object type, an automaton over event types induces `tau_e < tau_e'` constraints. These are **domain beliefs** → soft. Implemented as a truncation factor on the difference of two continuous variables, handled by the same moment-matching machinery as everything else in Layer 3 (§4.3). A soft precedence is a smoothed step: `log sigmoid((tau_e' - tau_e)/s)` with a small slack `s`.

### 3.4 The hard/soft register (decision 6, in full)

| constraint class | hard | soft | rationale |
|---|---|---|---|
| referential integrity | **x** | | definitional in the spec |
| attribute-type domain | **x** | | definitional in the spec |
| qualifier legality | **x** (as prune) | | definitional in the schema |
| type disjointness | **x** | | definitional |
| cardinality / multiplicity | | **x** | schemas are aspirational |
| lifecycle precedence | | **x** | processes deviate |
| attribute monotonicity / state machines | | **x** | domain belief |
| functional-dependency uniqueness | | **x** | and only where declared exclusive |

Every constraint carries an explicit `HARD | SOFT` register field, and any soft constraint's weight is a fitted parameter. The register is user-overridable per constraint — decision 6 gives the defaults, not a straitjacket.

---

## 4. Layer 3 — Attributes and time via latent Gaussian copula (decision 7)

### 4.1 The transform

Every continuous, ordinal, count, binary, or truncated variable `v` is the image of a latent Gaussian `Z`:

| observed type | latent relation |
|---|---|
| continuous | `Z = Phi^-1( F(v) )`, `F` the marginal CDF (empirical / parametric) |
| ordinal, count | interval-censored: `Z in [ Phi^-1(F(v-1)), Phi^-1(F(v)) ]` |
| binary | threshold: `Z > c` |
| truncated / zero-inflated | mixture of a point mass and a censored interval |
| **unordered categorical** | **not in this layer** — stays discrete (§4.2) |

Jointly `Z ~ N(0, Sigma)` with sparse precision `Theta = Sigma^-1`. Structure of `Theta` is schema-given where known (attributes of the same object; an event attribute and its event's timestamp) and estimable otherwise. Latent correlations are recovered by **bridge functions on Kendall's tau** (arXiv:2205.06868, arXiv:2211.11700) — rank-based, so estimable from far fewer observations than a parametric mixed MRF, which is what the sparse regime demands.

**Timestamps live in this layer.** `tau_e` is a continuous variable in `Z`, which means durations, precedence slack, and time-correlated attributes are all handled by one mechanism.

### 4.2 The honest boundary

The copula does **not** cover unordered categoricals. Those (event type `T_e`, categorical attributes) remain in the discrete layer with Potts/categorical factors. This is a real limitation of the copula approach and is stated rather than papered over.

The two layers couple through a **conditional Gaussian** construction (Lauritzen–Wermuth): discrete variables modulate the *mean* of the Gaussian block,

```
Z | (discrete = d)  ~  N( mu_d , Sigma )
```

restricted to **homogeneous CG** for v1 — mean-modulation only, shared covariance. This keeps every message a Gaussian with a discrete-indexed offset. Heterogeneous CG (state-dependent covariance) is a later extension; it costs the closed-form message.

### 4.3 One inference mechanism for the whole continuous layer

This is the design's main technical economy. Four apparently different problems all reduce to the same operation:

| problem | non-Gaussian element |
|---|---|
| copula transform of a non-Gaussian marginal | monotone warp |
| ordinal / count observation | interval censoring |
| lifecycle precedence `tau_e < tau_e'` | truncation |
| CG coupling to a discrete variable | Gaussian **mixture** message |

All four are handled by **Gaussian expectation propagation with moment matching**: compute the exact tilted moments, project back to a Gaussian, propagate. The continuous engine is therefore *one* algorithm — GaBP with EP-style moment matching — and not four special cases.

The CG case deserves an explicit caveat: the exact message from a CG factor to a continuous variable is a mixture with one component per discrete state. We **collapse it by moment matching** unless the discrete support is tiny (configurable threshold, default `<= 4` states retained exactly). This is a documented approximation, and multimodality in the continuous posterior is where it will show. The mixture-retaining path exists for validation.

Message from a CG factor *to* a discrete variable is exact and closed-form: the Gaussian log-partition evaluated per state.

---

## 5. Layer 4–5 — Sources and the reliability hierarchy (decisions 3, 4)

### 5.1 The claim and the channel

```
Claim(source_id, assertion_ref, value, [soft_distribution], [observation_features])
```

Channel families, selected by the assertion's type:

**Binary assertions** (existence, E2O link, O2O link) — two-sided quality per LTM (arXiv:1203.0058):

```
p(y=1 | truth=1) = alpha_s     (sensitivity)
p(y=0 | truth=0) = beta_s      (specificity)
```

Two-sided is non-negotiable here: with `complete_over_scope` coverage, `beta` is what silence means.

**Categorical assertions** (event type). A full `K x K` confusion matrix per source is exactly the parameterisation the sparse regime forbids. Default is a **one-parameter spread model**:

```
p(y=k  | T=k)   = rho_s
p(y=k' | T=k)   = (1 - rho_s) * pi_{k'|k}
```

where the off-diagonal profile `pi_{.|.}` is **shared across a source cluster**, not per source. This captures the systematic-confusion phenomenon that motivates Dawid-Skene (an extractor that reliably confuses `Ship` with `Deliver`) while costing one free parameter per source instead of `K^2`. Full per-source confusion matrices are available for sources with enough claims to earn them — the model nests.

**Continuous assertions** (timestamps, numeric attributes):

```
y = v + bias_s + eps ,   eps ~ StudentT(nu_s, sigma_s)
```

Student-t rather than Gaussian, for robustness to the gross outliers that unreliable sources produce. Represented as a scale mixture of normals, which keeps it *conditionally* Gaussian and therefore compatible with §4.3's single mechanism. In copula space this is a Gaussian factor on `Z`.

**Distributional claims** (a source that emits a distribution, not a point — the common case for a learned detector):

```
log p(y | v)  =  lambda_s * log q_s(v)  +  const
```

`lambda_s` is a learned **calibration temperature**. `lambda_s < 1` means the source is overconfident, `> 1` underconfident. This is where decision 8's "calibrated" requirement is actually enforced, and it is the cheapest high-value parameter in the whole model — [research notes](research-notes.md) §3.2 argues that calibration, not accuracy, is where fusion pays.

### 5.2 Coverage semantics (decision 4)

Every source declares one of three, per assertion family:

| mode | `pi_s(a)` | what silence contributes |
|---|---|---|
| `opportunistic` | independent of truth | **nothing** — factorises out |
| `complete_over_scope` | `= 1` on `scope_s` | a full negative claim: `log(1-alpha_s)` or `log(beta_s)` |
| `selective` | `sigma( w_s . f(a) + gamma_s * truth(a) )` | `log(1 - pi_s(a | truth))` — a real factor coupling to truth |

`gamma_s` is the **informativeness parameter**. `gamma_s = 0` recovers `opportunistic`; `gamma_s -> inf` recovers `complete_over_scope`. So the three modes are one family, and `selective` with a fitted `gamma_s` is the general case — which means a source whose declared mode is wrong can be detected by fitting `gamma_s` and comparing.

This directly implements the MNAR treatment of [research notes](research-notes.md) §4.5. Declaring the mode is **mandatory** on every adapter; there is no default, because a silently-wrong default here biases everything downstream.

### 5.3 The hierarchical GLM (decision 3)

Reliability is indexed by `(source, assertion family, type)` — *not* by source alone. That indexing is the worker-task specialisation of arXiv:2004.00101 / arXiv:2111.12550, expressed as a GLM:

```
g( theta_{s,f,t} )  =  beta_0
                     + x_s^T beta_feat          # source features: modality, vendor, model version,
                                                #   placement, sampling rate, latency
                     + u_{c(s)}                 # source-cluster (family) effect
                     + m_{f,t}                  # assertion-family x type main effect
                     + delta_{c(s), f, t}       # SPECIALISATION: cluster x task interaction
                     + eps_s ,  eps_s ~ N(0, sigma_eps^2)     # individual, shrunk
```

`g` is `logit` for `alpha, beta, rho` and `log` for `sigma_s`, `lambda_s`.

Four properties, each answering a specific finding from Stage 1:

- **`beta_feat`** — SLiMFast's feature-regressed reliability (arXiv:1512.06474). Turns `|S|` parameters into `d << |S|` and **generalises to sources never seen before**. In a world with 1e5 sources this is the difference between a fittable model and an unfittable one.
- **`eps_s` with small `sigma_eps`** — partial pooling. A source with 3 claims shrinks to its cluster; a source with 10,000 claims dominates its own estimate. Automatic, no thresholds.
- **`delta_{c,f,t}`** — the barcode scanner is excellent at object identity and useless at activity semantics. A single scalar per source cannot express that; this can.
- **`u_{c(s)}` shared within a declared source family** — this *is* the dependency correction. Sources sharing a vendor, an upstream model, or a physical site share `u_c`, so their agreement is partly explained by the shared effect rather than counted as independent confirmation. Per [research notes](research-notes.md) §4.6, with a thin overlap graph a **declared** family structure beats a learned dependency graph, because learning dependencies also needs overlap.

Cluster assignment `c(s)`: **declared by default**, latent (DP mixture) behind a flag.

Explicit pairwise copy factors remain available for the cases where declared families are insufficient, but they are not the primary mechanism.

---

## 6. Inference architecture

### 6.1 The two-block scheme

The central architectural commitment:

```
initialise:  triplet-method accuracies  ->  weighted vote  ->  initial beliefs

repeat until convergence:
    BELIEF BLOCK   (huge, sparse, ~1e7 vars)
        given theta, run BP on the grounded factor graph  ->  marginals over L*

    PARAMETER BLOCK  (small, dense, ~1e2-1e3 params)
        given assertion marginals, fit the hierarchical GLM
        MAP/Laplace by default; full posterior on demand

    PRIOR BLOCK
        refit object-type DF counts from soft posterior counts (Dirichlet-smoothed)
```

This is variational EM with a structured mean-field split between the assertion block and the parameter block. The justification is precisely [research notes](research-notes.md) §4.4: the assertion block is huge and sparse, where **BP is provably near-optimal**; the parameter block is small, where real Bayesian inference is affordable and where pooling lives. Running MCMC over 1e7 assertion variables would be both slower and *worse*.

### 6.2 Initialisation matters

EM on a non-convex truth-discovery objective has bad local optima, and the literature's answer is a moment-based initialiser (arXiv:1406.3824: spectral init + one EM step is provably optimal). Ours:

1. **Triplet method** — for any three sources with pairwise conditional independence and sufficient overlap, per-source accuracies have a *closed form* from pairwise agreement rates alone, with no labels. Cheap and label-free.
2. Where triplets are unavailable (thin overlap), fall back to **CATD**-style confidence-aware weights — a lower confidence bound on reliability, designed for long-tail sources.
3. Weighted vote under those weights seeds the belief block.

### 6.3 Engine tiers

| tier | engine | scope | backend |
|---|---|---|---|
| 1 | exact variable elimination | small / tree-structured subgraphs; **verification oracle** | pyAgrum |
| 2 | **loopy BP (discrete) + GaBP/EP (continuous) + CG hybrid** | the workhorse, whole graph | custom, torch, GPU |
| 3 | blocked Gibbs / NUTS | parameter block; joint sampling | PyMC |
| 4 | hybrid discrete-continuous MAP | continuous sub-block under precedence constraints | GTSAM (optional) |
| 5 | exact WMC on declarative constraints | constraint feasibility checking | ProbLog (optional) |

Tier 2 is the only mandatory one. Tiers 1, 4, 5 exist because a system whose core is an approximation needs oracles to check itself against — that is their job, and it is a real job.

BP hygiene, non-negotiable: **damping**, message-residual convergence monitoring, oscillation detection, and a hard iteration cap that reports non-convergence rather than silently returning garbage.

### 6.4 Producing joint samples (decision 8)

Marginals are not enough — downstream process mining consumes whole logs (arXiv:2108.08615, arXiv:2203.07507). Two paths:

- **`bp_guided`** (fast): ancestral sampling along a spanning structure using BP marginals as proposals, with rejection against hard factors. Biased, fast, fine for exploratory use.
- **`blocked_gibbs`** (correct, default for reported results): blocked Gibbs seeded from BP marginals, blocks chosen by template (all `R_{e,q,.}` for one `(e,q)` resampled jointly, which respects the cardinality factor exactly).

Every returned sample is a **valid OCEL 2.0 log** — hard factors guarantee it, which is precisely why §3.4 keeps referential integrity and attribute-domain gating hard.

### 6.5 Attribution comes free

For a binary assertion, BP's posterior log-odds decomposes additively over incoming messages:

```
logit p(a)  =  logit prior(a)  +  sum over factors f in N(a) of  log( m_{f->a}(1) / m_{f->a}(0) )
```

Each term is one factor's contribution — one source's channel, the cardinality factor, the DF prior, and so on. **Attribution is not extra computation; it is the retained messages.** This satisfies decision 8's attribution vectors at essentially zero cost, and it is what makes a weak-source system auditable. With sources this unreliable, auditability *is* the product.

---

## 7. Diagnostics as core outputs (decision 10)

These are returned alongside the beliefs, not written to a log file.

### 7.1 Identifiability

Build the **source overlap graph** `G_S` (nodes = sources, edge iff co-claim on some assertion) and report:

- **connected components** — reliabilities are jointly estimable only within a component; across components they are comparable only through the hierarchical prior;
- **bipartiteness per component** — by [research notes](research-notes.md) §4.1 (arXiv:1706.06660, arXiv:1904.11608), a bipartite component admits a global sign flip and its skills are **not identifiable**. Non-bipartite ⟺ contains an odd cycle. Two-colouring BFS, `O(V+E)`;
- a per-source verdict: `IDENTIFIED | PRIOR_ONLY | SIGN_AMBIGUOUS`.

A source flagged `PRIOR_ONLY` or `SIGN_AMBIGUOUS` still contributes — through the GLM's feature and cluster terms — but its individual effect `eps_s` is not data-identified, and the report says so.

### 7.2 Per-assertion decidability

From [research notes](research-notes.md) §4.2, with Chernoff information estimated from the fitted channel parameters of the sources covering `a`:

```
C_s = -min_{t in [0,1]} log sum_y p_0(y)^t p_1(y)^(1-t)

evidence(a) = sum over s covering a of C_s   +   prior contribution
verdict     = DECIDABLE if evidence(a) > log(1/eps_target) else UNDETERMINED
```

`UNDETERMINED` assertions get a posterior *and* a flag. A three-valued output — `true` / `false` / `undetermined` — rather than a bare probability that hides its own emptiness.

### 7.3 Effective sample size

Independent votes are the thing we do not have. With within-family correlation `rho_c` implied by the fitted `u_c` and `sigma_eps`, the design-effect correction per family is

```
ESS_c = m_c / ( 1 + (m_c - 1) * rho_c )
ESS(a) = sum over families of ESS_c
```

Reporting `ESS(a)` next to `deg(a)` makes the "20 sources agree but they are all the same vendor" failure mode visible instead of fatal.

### 7.4 Calibration and convergence

- Reliability diagrams and ECE on any audited subset; the synthetic generator (§9) always provides one.
- BP message residuals, iteration counts, oscillation flags, and a per-component convergence verdict.

---

## 8. Evaluation design (decision 9)

### 8.1 Synthetic OCEL generator

Ground truth is not optional — nothing else measures calibration. The generator produces a *real* object-centric process, not i.i.d. noise:

1. sample a schema (object types, event types, qualifiers with multiplicities, attribute frames);
2. sample an object registry with O2O structure;
3. simulate per-object-type stochastic automata that emit events **shared across objects** — this is what makes it object-centric rather than a set of independent traces;
4. sample attributes through a known copula, and timestamps respecting lifecycle;
5. emit a ground-truth OCEL 2.0 log.

### 8.2 Source simulator with controllable regime

The corruption layer is where the Stage 1 regime is reproduced, with independent knobs on every axis:

| knob | sweep |
|---|---|
| `deg(a)` — sources per assertion | 1 – 20 |
| `deg(s)` — assertions per source | long-tail (power law), controllable exponent |
| source accuracy | barely-above-chance to near-perfect |
| **copy / correlation structure** | independent → families → chains |
| coverage semantics | all three modes, including a **deliberately mis-declared** mode |
| specialisation | sources strong on one assertion family, weak on others |

### 8.3 Mandatory comparisons

- **Weighted vote baseline** — always run, always reported. [research notes](research-notes.md) §3.2 is unambiguous that this is the bar, and a fusion system that does not clear it is not working.
- Dawid-Skene and CRH/CATD as intermediate baselines.
- **Sensitivity to a misspecified constraint** — deliberately assert a false lifecycle precedence or a wrong cardinality and measure the damage. This is the experiment that justifies decision 6's hard/soft split: soft constraints should degrade gracefully where hard ones would zero out the truth. If that does not show up, the split is wrong and should be revisited.
- Ablations: no DF prior; no hierarchy (per-source MLE); no dependency correction; no calibration temperature.

---

## 9. Program architecture

### 9.1 Module map

```
ocbf/
  schema/          OCEL 2.0 schema objects: types, qualifiers, multiplicities,
                   attribute frames, lifecycle automata, constraint register (HARD|SOFT)
  universe/        candidate universe construction, grounding, the three prunes,
                   lazy prior-only materialisation
  assertions/      the assertion algebra: AssertionRef, variable registry,
                   families, domains, NA handling
  sources/         Source protocol, adapters, Claim ingestion,
                   coverage semantics, source feature extraction
  model/
    templates/     FactorTemplate + grounding
    factors/       hard_schema · structural · cardinality (counting) ·
                   behavioural (DF-relaxed, DF-exact) · copula · channel
    copula/        marginal transforms, bridge functions, precision structure
  reliability/     hierarchical GLM, source clustering, triplet init,
                   CATD fallback, calibration temperature
  inference/
    engines/       exact_ve · loopy_bp · gabp_ep · hybrid_cg
    messages/      templated batched message ops (torch, GPU)
    samplers/      bp_guided · blocked_gibbs
    params/        PyMC parameter-block fitting
    schedule/      two-block outer loop, damping, convergence
  belief/          BeliefState: marginals, samples, decidability, attribution
  diagnostics/     overlap graph + bipartiteness, decidability, ESS,
                   calibration, convergence
  baselines/       majority · weighted vote · Dawid-Skene · CRH/CATD
  synth/           OCEL generator + source simulator
  io/              OCEL 2.0 read/write (sqlite, json, xml); pm4py interop
```

### 9.2 The interfaces that matter

Five types carry the whole design. Everything else is implementation.

```python
AssertionRef      # typed, hashable address into the latent world
                  # e.g. E2O(event="e17", qualifier="ships", object="item_3")

Claim             # (source_id, ref, value, soft_dist?, obs_features?)

Source            # protocol:
                  #   scope()              -> iterable[AssertionRef]
                  #   coverage_semantics   -> OPPORTUNISTIC | COMPLETE | SELECTIVE   (mandatory)
                  #   channel_family       -> BINARY | CATEGORICAL | CONTINUOUS | DISTRIBUTIONAL
                  #   features()           -> feature vector for the GLM
                  #   cluster_id           -> declared source family
                  #   claims()             -> iterable[Claim]

FactorTemplate    # key, family, params; .ground(universe) -> Factors
                  # all groundings share params (pooling) and message shape (GPU batching)

BeliefState       # .marginal(ref) · .sample(n) -> [OCEL logs]
                  # .decidability(ref) · .attribution(ref) · .diagnostics()
```

`Source` is the extension point. Adding a new kind of evidence means writing one adapter with a declared scope, coverage semantics, channel family, and feature vector — and nothing in the core changes.

### 9.3 Library assignment, and why each earns its place

| library | role | justification |
|---|---|---|
| **torch** | templated batched BP/EP message passing; GPU | templates give identical message shapes → one batched op per template. The right use of the RTX 5050. |
| **PyMC** | parameter-block posterior (reliability GLM, calibration, DF hyper-params) | ~1e2–1e3 parameters: small enough for real NUTS, and this is exactly where full-Bayes pooling and shrinkage pay off. |
| **pyAgrum** | exact VE oracle on small groundings; BN/MRF export | a system whose core is approximate needs a ground-truth inference oracle. Also gives free model visualisation. |
| **GTSAM** | optional hybrid discrete-continuous MAP on the continuous sub-block | native hybrid factor graphs + elimination (arXiv:2601.00545); a strong cross-check on our EP approximation. |
| **ProbLog** | optional exact WMC for constraint-feasibility checking | verifies that the declared hard-constraint set is satisfiable before we spend an inference run discovering it is not. |
| **pm4py** | OCEL 2.0 I/O and interop | do not reimplement a standard's serialisation. |
| numpy / scipy / networkx | graph algorithms (bipartiteness, components), stats | — |

Custom rather than off-the-shelf for the core BP engine, deliberately: no existing library does templated, GPU-batched, hybrid discrete-continuous message passing with per-message attribution retention. Attribution alone rules out every black-box engine, and it is a hard requirement (decision 8).

### 9.4 Build order for Stage 3

1. `schema` + `assertions` + `universe` — the data model and grounding, with the prunes.
2. `synth` — the generator, **before** the inference engine. Nothing can be validated without ground truth, and building the generator first forces the data model to be honest.
3. `baselines` — weighted vote. The bar to clear, established before there is anything to be optimistic about.
4. `model/factors` — hard schema, structural, cardinality counting factors.
5. `inference/engines/loopy_bp` on the discrete layer only. First end-to-end result: discrete-only fusion beating weighted vote on synthetic data.
6. `reliability` + `inference/params` — the two-block loop with PyMC.
7. `diagnostics` — overlap graph, decidability, ESS.
8. `model/copula` + `gabp_ep` + `hybrid_cg` — the continuous layer.
9. `model/factors/behavioural` — the DF prior.
10. `belief/samplers` — joint sampling; `io` — OCEL round-trip.
11. `engines/exact_ve` (pyAgrum) as verification; optional GTSAM/ProbLog backends.

Steps 1–5 are a complete, testable, useful system on their own. That is the intended v0.1 milestone.

---

## 10. Known limitations, stated up front

1. **`DF-relaxed` double-counts** across intervening events (§3.2). `DF-exact` exists to measure the distortion.
2. **CG messages are moment-matched** (§4.3); multimodal continuous posteriors are collapsed. The mixture-retaining path exists for small discrete supports.
3. **Homogeneous CG only** — discrete variables shift Gaussian means but not covariances.
4. **Unordered categoricals are outside the copula** (§4.2) and are handled discretely.
   Event attributes are additionally restricted to events whose type support agrees they
   apply (§11.12).
5. **Fixed universe** — an event that no source ever mentions and that no candidate slot anticipates cannot be inferred. This is decision 1, taken knowingly.
6. **Identity clamped** by default; ER is designed for but not implemented in v1.
7. **Loopy BP has no convergence guarantee** on our graph. Mitigated by damping, monitoring, exact oracles on subgraphs, and honest non-convergence reporting — not by hoping.
8. **Declared source families are trusted.** If two sources secretly share an upstream model and declare different families, the dependency correction misses them. Learned dependency structure is the fallback, and it needs overlap we may not have.

---

## 11. Stage 3 findings — where implementation revised the design

Recorded because they are corrections to the plan above, not incidental bugs. Each was
found by building the thing and measuring it.

### 11.1 The type gate had to be restricted to claimed links

Section 2.3 treated claim-anchored blocking as a scalability measure. It is also a
*correctness* measure for the type gate specifically.

Attaching a hard gate factor to every candidate link of an event put **113 hard factors on
that event's single latent type variable**. Two failures followed. Gate messages from
*unclaimed* links are driven entirely by the link prior, so a type permitting many
candidates accumulated an advantage for having many candidates — not evidence about
anything. And because those links share the event's existence variable and its cardinality
factors, they are far from independent, so loopy BP double-counted that spurious signal and
drove the type posterior into a saturated corner it then oscillated between.

Restricting the gate to claimed links (`GraphSpec.type_gate_claimed_only`, default on) cut
gate factors from 21663 to 583, **converted a non-converging oscillating run into one that
converges**, and raised AUC from 0.9618 to 0.9735. The bounded cost: an unclaimed link keeps
its prior mass even under a forbidding type. Those assertions are prior-only and already
flagged.

The exact fix is a single higher-order factor over `(T_e, all links of e)` rather than many
pairwise ones. Deferred.

### 11.2 Flat link priors were incoherent with the cardinality factor

The original spec carried a constant `p_e2o_link = 0.08`. Over a group of `k` candidates
under an `exactly-one` qualifier, that prior expects `0.08k` links while the counting factor
insists on one. The two disagree, and the posterior then depends on their relative weights
rather than on the evidence.

Link priors are now **derived** per `(event, qualifier)` group from the same declared
multiplicity the cardinality factor reads. Both fixed a real bug and improved everything
measured:

| | flat prior | derived prior |
|---|---|---|
| BP | 300 iterations, **not converged** | **81 iterations, converged** |
| accuracy | 0.9251 | 0.9322 |
| AUC | 0.9788 | 0.9824 |
| ECE | 0.0373 | 0.0334 |

Generalisation: a prior and a constraint that encode the same schema fact must be read off
that fact, not set independently.

### 11.3 Soft-constraint weights must be denominated in evidence

Section 3.4 specified soft weights in raw nats (cardinality 4.0). Measured against the
source pool, one claim carries a **median 0.178 nats** in the benchmark regime — so 4.0 was
roughly twenty claims' worth, which is not "soft" in any sense that matters.

Weights are now **dimensionless multipliers** on a scale calibrated from the source pool
(`calibrate_constraint_scale`), so `weight=1.0` costs about one confident claim and a
handful of claims can overrule a misstated schema belief. A raw nat count cannot hold that
meaning across pools of differing quality: the same number is a nudge against strong sources
and an unbreakable rule against weak ones.

### 11.4 MAP silently collapses hierarchical variance components

The parameter block fitted `sigma_individual = 0.0` — the joint mode of a hierarchical model
sits at zero scale with every random effect collapsed. It reports a clean fit while
switching off exactly the partial pooling the sparse regime depends on, which makes it a
particularly bad failure to have gone unnoticed.

Fixed with Gamma priors whose mode is away from zero, plus non-centred random effects.
Reliability recovery improved from correlation +0.507 to +0.611, accuracy 0.9078 to 0.9434,
ECE 0.0625 to 0.0312.

### 11.5 Decidability is a lower bound for the structured model

Section 7.2 implicitly treated the decidability flag as predicting model accuracy. It
predicts accuracy for **vote-only** methods, which is what its derivation covers: it counts
only Chernoff information carried by source claims.

The structured model also propagates belief along referential integrity, the type gate and
cardinality, so it resolves assertions that *no* aggregation rule could decide from votes.
Measured: on the undecidable subset the structured model still beats weighted voting. So the
flag is a **lower bound on what is resolvable**, not a prediction of accuracy — and that gap
is the clearest single confirmation of section 4.2's claim that the structural prior is
primary.

### 11.6 Convergence must be judged on beliefs, not messages

With hard factors the message residual is dominated by edges swinging between finite values
and the `NEG_INF` sentinel — a swing that says nothing about whether the posterior settled.
Both are now tracked; convergence is decided on beliefs.

Relatedly, the sentinel's *magnitude* is load-bearing. It must be large enough that
`exp(NEG_INF)` underflows to zero and small enough that adding an ordinary log-potential
preserves it: `-1e30 + 5.0 == -1e30` in float64, which annihilates the finite part that BP's
subtraction step depends on. The value is `-1e4`.

### 11.7 Performance is CPU-bound by two orders of magnitude

At 845k edges the engine holds ~15 MB of live message arrays and peaks near 50 MB — against
470 ms per iteration. Memory is nowhere near the constraint; Python and allocation overhead
in the bank loop is. Replacing `scipy.special.logsumexp` with a direct implementation
removed roughly two thirds of engine runtime. The torch/GPU path of section 9.3 remains the
right lever, and graphs could grow ~100x before memory matters.

### 11.8 Truncated moments must be computed in the tail that does not cancel

Interval censoring is the closed-form case of §4.3, and its obvious implementation is wrong
in exactly the place expectation propagation visits most. `Phi(b) - Phi(a)` for two large
positive bounds subtracts two numbers near 1 and loses every significant digit, and the
natural guard — return a near-zero variance when the mass underflows — manufactures
near-certainty out of an arithmetic failure. The resulting site carried a precision at the
representable ceiling and dominated every other factor on the variable; message residuals
reached `1e11`.

Two changes, both principled rather than defensive. The mass is computed in whichever tail
does not cancel. And where it underflows regardless, the fallback is the *asymptotic* rather
than a point: far out, a two-sided interval tends to the uniform on that interval and a
one-sided one to the Mills-ratio limit. Both are continuous with the exact formulas, so
there is no jump at the switchover.

Generalisation: a numerical guard that returns a *confident* answer is worse than none. The
failing quantity here was a variance, and the safe direction for a variance is up.

### 11.9 The expectation-propagation step must be scaled by variable degree

A synchronous sweep updates every site on a variable against a cavity that all the other
sites are simultaneously moving, so a coordinate carrying `d` sites takes a step about `d`
times too large. On a coordinate with fifty claims the run wobbled around the fixed point
with a residual stuck near `2e-2` that never settled.

Dividing each edge's step by its variable's degree fixes it, and what it buys is worth
stating precisely: **both settings reach the same posterior**. What differs is that the
scaled residual falls monotonically to `1.6e-4` while the unscaled one oscillates. Since
§11.6 makes the residual the thing convergence is judged on, a residual that reports nothing
is the failure being avoided.

The cost is real and was initially missed: a step `d` times smaller needs proportionally
more sweeps, and the continuous engine's iteration cap had been copied from the discrete
one. At 200 sweeps the precision had not finished accumulating, which showed up as 90 %
intervals covering 95 % of the truth — **an over-wide interval, the quietest possible
failure**. The cap is now 800.

| | 200 sweeps | 800 sweeps |
|---|---|---|
| coverage of a nominal 90 % interval | 0.95 | 0.81 |
| mean interval width | 4.00 | 2.82 |

Generalisation: a damping rule and an iteration cap are one decision. Changing either alone
is how a stable engine ends up reporting confident-looking, under-converged answers.

### 11.10 Quadrature accuracy and convergence tolerance are one choice

The Student-t channel is the layer's hardest integrand: Gauss–Hermite assumes
Gaussian-weighted decay and the Student-t tail is polynomial. At 32 nodes the tilted moments
carry a systematic error near `1e-4` — *above* the engine's own `1e-5` belief tolerance, so
the run was chasing a fixed point finer than its integration could resolve. The default is
64 nodes, where the error is nearer `1e-6`.

Generalisation: an approximation's tolerance must sit above every systematic error feeding
it, or the tolerance is measuring noise.

### 11.11 The mean-field precedence weight is attenuated by unclaimed links

Whether a declared lifecycle ordering *applies* to a pair of candidate events depends on
discrete variables — do both events really link to the shared object, and do they carry the
two declared types. Under the structured mean-field split already committed to in §6.1, the
weight on the truncation factor is `E_q[applies]`, the product of four marginals.

That product is severely attenuated in the sparse regime, and the reason is structural
rather than incidental: an **unclaimed** link sits at its derived prior of `1/k` over a group
of `k` candidates (§11.2), so a pair of unclaimed links weighs about `1/k²` however the
threshold is set. On a world where most true links carried no claim, no threshold produced a
factor worth grounding; where the link posteriors were informed, the default threshold
grounded pairs immediately.

So **precedence is claim-anchored in effect rather than by construction** — the same
conclusion §11.1 reached for the type gate, arrived at from the other direction. Lowering the
threshold to make factors appear would have produced thousands of near-vacuous factors and
hidden the finding.

The prune is also now exact rather than heuristic: each of the four terms is a probability,
so the product cannot exceed any one of them, and an event failing on its own link posterior
can be dropped before the quadratic scan. That alone turned a 4.5-million-pair scan into a
few hundred.

### 11.12 Under a latent event type, event attributes are never unambiguously in domain

§2.2 resolves attribute-domain gating with an explicit `NA` state, which works for a discrete
attribute variable. A latent Gaussian coordinate has no "does not apply" value, and there is
no well-posed way to weigh a density against the probability of absence.

That would be a modelling inconvenience except for an OCEL 2.0 requirement that makes it
structural: **attribute sets are disjoint across types**. An event attribute is therefore
inherently type-specific, so whenever `T_e`'s support is wider than one type, whether the
attribute exists at all is uncertain.

The conservative reading is taken, matching what the cardinality factor already does where
types disagree: an event attribute is registered only where *every* type in the support
declares it, and is otherwise prior-only and counted in the prune report. The practical
consequence is that event attributes reach the copula only for events whose activity label is
determined — a common enough real shape, since a label can be certain while the links are
not. Object attributes are unaffected: object type is clamped, so what an object's attributes
are is known even when their values are not.

### 11.13 GTSAM took the tier-1 oracle role pyAgrum was specified for

§6.3 split the optional native backends two ways: pyAgrum for tier 1, the exact
variable-elimination oracle, and GTSAM for tier 4, hybrid discrete-continuous MAP. The
implementation collapsed that into one backend doing the tier-1 job.

The oracle role is what the project actually needs, and it needs elimination rather than
enumeration: brute force costs the product of the cardinalities and runs out well before the
loopy structure BP actually meets, which is the only region where the approximation is worth
measuring. GTSAM's `DiscreteFactorGraph` eliminates, so its cost is exponential in the
treewidth instead. `ocbf.inference.gtsam_exact` is that oracle, on the discrete backbone;
pyAgrum and ProbLog are declared in the `oracles` extra and unused.

The bounded cost is one platform problem, paid once. A CUDA-enabled `gtsam.dll` imports the
CUDA runtime by name and Windows resolves that on the DLL search path rather than on `PATH`,
so a plain import fails with a bare "DLL load failed". `ocbf.backends` wraps the import,
registers the toolkit directories in a stated order, and reports what the build can do and
what the machine has *separately* — a CUDA build without a device and a device without a CUDA
build are different problems, and one boolean would hide which one you have.

Tier 4 is unchanged and unbuilt: hybrid MAP would need the continuous block expressed in
GTSAM's hybrid factor graph, which the EP engine has no reason to produce.

---

## 12. Documentation

The site is MkDocs + Material + mkdocstrings, structured on **Diátaxis**: tutorial, how-to,
reference and explanation are kept strictly separate, because each answers a different
question and blending them serves nobody. A how-to interrupted by theory wastes the reader
in a hurry; an explanation interrupted by commands teaches nothing.

| section | Diátaxis type | answers |
|---|---|---|
| Getting started | tutorial | "teach me by doing" |
| How-to guides | task | "I have a goal, get me there" |
| API reference | information | "what does this function do" |
| Explanation | understanding | "why is it built this way" |

Four conventions worth recording, because each was a decision rather than a default.

**The API reference is generated, never written.** `scripts/gen_ref_pages.py` emits one stub
per module at build time via `mkdocs-gen-files`, and `mkdocs-literate-nav` reads the
generated `SUMMARY.md`. A hand-maintained reference drifts the moment someone adds a module
and forgets the index. Material's `navigation.indexes` binds each package's `index.md` to its
section — the `mkdocs-section-index` plugin does the same job but conflicts with it, so only
one may be used.

**Docstrings are Markdown, not reStructuredText.** mkdocstrings renders docstrings as
Markdown, so Sphinx roles and directives come out as literal text. The codebase originally
carried 33 RST roles and 7 `.. code-block::` directives; all were migrated to mkdocstrings
cross-references (`[Text][dotted.path]`) and fenced blocks. The migration surfaced a stale
reference to `Universe.active_subset`, a method that had been renamed to `active_refs` —
which is an argument for cross-references that a tool can verify over prose that no one
checks.

**Every public member is rendered** (`show_if_no_docstring: true`). In a reference, a member
that exists but is not listed is a hole: the reader cannot distinguish "absent" from
"undocumented". Filling the gap raised a related question — 135 public members had no
docstring — answered by writing them for the user-facing surface and leaving internal
plumbing to its signature. Mechanically generated one-liners were rejected: text like
"Return the events." looks like documentation without being any, which makes a reference
worse rather than better.

**The docs are checked like code.** `mkdocs build --strict` is the gate, and it runs in CI
on every push: an unresolved internal link, an unrecognised mkdocstrings identifier or a page
missing from the nav fails the build rather than printing a warning nobody reads. What it
cannot check is duplication, so that stays a convention — cross-link an idea from several
places, but do not copy the paragraph, because the copies drift.

For agent consumption the site publishes `llms.txt` via `mkdocs-llmstxt`, with sections
ordered as the intended reading order.
