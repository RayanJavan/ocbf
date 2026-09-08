# Stage 1 — Research & Theory
### Belief over an OCEL 2.0 reality, fused from many sparse, unreliable, structurally-typed sources

> Scope: literature, formalism, and theory only. No model commitment (that is Stage 2), no code (Stage 3).
> Every claim about prior work is tied to a retrievable identifier; consolidated list in §12.

---

## 1. Problem statement, made precise

### 1.1 What is given

**(a) A structural skeleton of reality in OCEL 2.0 form.** Per the standard (arXiv:2403.01975), an object-centric event log is

```
L = (E, O, EA, OA, evtype, time, objtype, eatype, oatype, eaval, oaval, E2O, O2O)
```

| symbol | meaning | signature |
|---|---|---|
| `E`, `O` | events, objects | sets |
| `evtype` | event type ("activity") | `E -> U_etype` |
| `time` | event timestamp (events are atomic) | `E -> U_time` |
| `objtype` | object type | `O -> U_otype` |
| `eatype`, `oatype` | which type an attribute belongs to | `EA -> U_etype`, `OA -> U_otype` |
| `eaval` | event attribute value | `(E x EA) -/-> U_val` |
| `oaval` | object attribute value **at a time** | `(O x OA x U_time) -/-> U_val` |
| `E2O` | qualified event->object relation | `subset of E x U_qual x O` |
| `O2O` | qualified object->object relation | `subset of O x U_qual x O` |

Two domain restrictions in the spec are *hard* and become deterministic factors in any probabilistic lift:

- `dom(eaval) subset {(e,ea) : evtype(e) = eatype(ea)}`
- `dom(oaval) subset {(o,oa,t) : objtype(o) = oatype(oa)}`

Attribute sets are disjoint across types. Object attributes are piecewise-constant in time: `oaval^t_oa(o)` is the *latest* value at or before `t`, with timestamp `0` (epoch) meaning "initial / static".

Crucially for us: **O2O relations and object attribute histories exist independently of events**. The only event-object coupling is E2O. That is a real factorization gift — the belief network decomposes into an event layer, an object layer, and a bipartite coupling layer, rather than one monolith.

**(b) Many probabilistic sources with known signal skeletons.** For each source `s` we know a priori:

- *what kind of assertion* it emits (event existence? activity label? timestamp? an E2O link with qualifier `q`? an attribute value?),
- *whose* — which event/object slots in the skeleton it can address (its **scope**),
- *its topology* — the shape of the substructure it observes (e.g. "source `s` sees pairs `(order, item)` and reports whether a `contains` O2O edge exists"; "source `s` reads a scanner and reports an activity label for one event at an approximate time").

### 1.2 The regime (the dominating design constraint)

Sources are **highly sparse, individually unreliable, and numerous**.

Let `A` be the set of latent assertion variables and `C ⊆ S x A` the *claim incidence* bipartite graph: `(s,a) ∈ C` iff source `s` speaks about assertion `a`. Then:

- **Sparse**: `|C| << |S|·|A|`. Both degrees are small: `deg(s)` = number of assertions a source touches is small (long-tail sources), and `deg(a)` = number of sources covering an assertion is small (thin redundancy).
- **Unreliable**: per-source accuracy only slightly above chance; per-source Chernoff information `I(s)` is small.
- **Numerous**: `|S|` is large.

This is *not* the regime most truth-discovery papers optimise for, and it flips several standard design choices. §4 is devoted to it. In one sentence: **you cannot afford a free parameter per source, redundancy per assertion is too thin to decide most assertions on votes alone, and therefore the structural prior — the OCEL topology — must do most of the inferential work.**

### 1.3 What we want

A **belief state**: a joint posterior `p(L* | source outputs, skeleton, priors)` over the latent true log `L*`, from which we can read

1. calibrated marginals per assertion,
2. *coherent joint samples* of complete OCEL realisations (downstream discovery/conformance under uncertainty consumes whole logs, not marginals — cf. arXiv:2108.08615),
3. per-assertion provenance/attribution ("which sources moved this belief, and by how much"),
4. an explicit **abstention** verdict where evidence is provably insufficient.

Item 4 is a first-class requirement in this regime, not a nicety.

---

## 2. Formal substrate: OCEL 2.0 as a random object

### 2.1 The assertion algebra

Lift each component of `L` to a random variable. This is the vocabulary the whole system is built on.

| # | assertion family | variable | domain | type |
|---|---|---|---|---|
| A1 | event existence | `X_e` | `{0,1}` | discrete |
| A2 | event type | `T_e` | `U_etype` (categorical, possibly taxonomic) | discrete |
| A3 | event time | `tau_e` | `R+` (ordered) | continuous |
| A4 | object existence | `X_o` | `{0,1}` | discrete |
| A5 | object type | `T_o` | `U_otype` | discrete |
| A6 | E2O link | `R_{e,q,o}` | `{0,1}` | discrete |
| A7 | O2O link | `S_{o,q,o'}` | `{0,1}` | discrete |
| A8 | event attribute | `V_{e,ea}` | mixed (real / count / ordinal / categorical / text-embedded) | mixed |
| A9 | object attribute trajectory | `V_{o,oa}(·)` | piecewise-constant path over mixed codomain + change-point set | mixed + temporal |
| A10 | identity / coreference | partition of mentions into latent objects | set partition | combinatorial |

Two readings of "we have the skeleton", both of which must be supported:

- **Schema-certain, instance-uncertain** (default): the *type system* (`U_etype`, `U_otype`, legal qualifiers, attribute-to-type assignment, cardinalities) is known and certain; instance-level A1–A10 are latent. This is the honest reading of "we know the structural skeleton of their signal too".
- **Instance-clamped**: some of A1/A4/A6/A7 are observed and clamped; only values and times are inferred.

The second must be a *special case obtained by clamping*, not a separate code path.

A10 (identity) is the expensive one and should be switchable: `identity ∈ {given, latent}`. See §8.

### 2.2 Hard structural constraints (deterministic factors)

These come from the spec and from domain schema; they carry no free parameters, and they are the cheapest information available — which matters enormously when evidence is thin.

1. **Type-attribute domain**: `V_{e,ea}` is defined only when `T_e = eatype(ea)`. Realised as an indicator gate, or as a "not-applicable" absorbing value.
2. **Referential integrity**: `R_{e,q,o} = 1 => X_e = 1 and X_o = 1`. Likewise for `S`.
3. **Qualifier signature**: qualifier `q` is legal only for certain `(evtype, objtype)` pairs. This prunes the E2O candidate space hard — often by orders of magnitude, which is the single most effective scalability lever available.
4. **Cardinality**: e.g. an event of type *Place Order* has exactly one `customer`-qualified object: `sum_o R_{e,customer,o} = 1`. A one-hot/Potts constraint, not a set of free Bernoullis.
5. **Temporal precedence / lifecycle**: per object type, an ordering automaton over event types (`create < approve < pay`). Yields `tau_e < tau_e'` inequality factors, conditioned on link and type variables.
6. **Attribute monotonicity / conservation**: quantities non-negative; `status` follows a state machine; running totals monotone.
7. **Uniqueness / functional dependency**: at most one true value for a functional attribute at an instant (mutual exclusion). **But see §3.1 on multi-truth**: many OCEL assertions are legitimately multi-valued (an event has *many* E2O edges), so exclusivity must be declared per assertion family and never assumed globally.

The encoding choice — hard zero factors vs. soft high-weight penalties — is a real trade-off; §7.3.

### 2.3 The noisy-channel frame

The cleanest formal container already exists in the database literature: **Probabilistic Unclean Databases** (arXiv:1801.06750), which factors

```
p(observed dirty instance, clean instance) = p_clean(intended) x p_noise(observed | intended)
```

Ours is the multi-channel generalisation:

```
p(L*, {Y_s}) = p_prior(L* | skeleton, constraints) x prod_s p_s(Y_s | L*, theta_s, scope_s)
```

- `p_prior` — OCEL-structural plus process-behavioural prior (§6),
- `p_s` — the **source channel model**, parameterised by reliability `theta_s`, gated by the source's known scope/topology,
- `Y_s` — that source's emitted signal.

Everything downstream is a choice of how to represent `p_prior`, how to parameterise `theta_s`, and how to do inference. HoloClean (arXiv:1702.00820) is this shape restricted to relational cells; PClean (arXiv:2007.11838) is this shape with a relational-database generative prior expressed in a domain-specific PPL; BClean (arXiv:2311.06517) and BayesWipe (arXiv:1506.08908) are further instances.

**Key consequence of sparsity:** because each `p_s` touches few variables, the factor graph is sparse and locally tree-like — exactly the regime where belief propagation is not merely tractable but **provably optimal** (§4.4).

---

## 3. Pillar I — truth discovery and multi-source fusion

### 3.1 The canonical line

| model | idea worth stealing | id |
|---|---|---|
| Majority / weighted vote | the baseline that is embarrassingly hard to beat | — |
| **TruthFinder** (Yin, Han & Yu, 2008) | fixed-point mutual reinforcement: source trust <-> fact confidence; "implication" similarity between competing values | — |
| **AccuCopy / source dependence** (Dong et al.) | sources *copy*; naive voting over-counts. Bayesian copy detection between source pairs | arXiv:0909.1776, arXiv:1503.00310, arXiv:1503.00309 |
| **LTM — Latent Truth Model** (Zhao et al., VLDB 2012) | **two-sided** source quality (separate false-positive and false-negative rates); multiple truths per object; Bayesian with Gibbs; truth is a latent Bernoulli per *(object, value) fact*, not a categorical per object | arXiv:1203.0058 |
| **CRH / CATD** (Li et al.) | (i) unified optimisation view `min_{x,w} sum_s w_s sum_a d(y_sa, x_a)` with a regulariser on `w`; (ii) heterogeneous distance functions per data type; (iii) **confidence-aware** weights via a chi-squared CI on the source residual — designed explicitly for **long-tail sources** | — |
| **Truth existence** | not every object has *any* true value; add an existence indicator | PMC4688015 |
| **Multi-truth** (SmartMTD) | one object legitimately has several true values; needs a different generative story than one-hot | arXiv:1708.02018 |
| **Hierarchies** | claimed values are not mutually exclusive when a taxonomy relates them (`Sedan ⊑ Car`) | arXiv:1904.10217 |
| **Correlations** | fusing with correlated (not merely copied) sources | arXiv:1503.00306 |
| **SLiMFast** | data fusion as *discriminative* learning: source accuracy as a **function of source features** (logistic regression) instead of a free per-source scalar; generalisation bounds; handles unseen sources | arXiv:1512.06474 |
| **Evolving truth** | truth changes over time: temporal smoothness on truth *and* decay on source reliability | PMC4688022 |
| **Community-structured reliability** | reliabilities correlated within latent communities; Laplace VI | arXiv:1806.02954, arXiv:1906.10470, arXiv:1210.0954 |
| **Trust propagation on source-object networks** | PageRank/HITS-style propagation over sparse, scale-free source-object graphs | arXiv:1603.02056, arXiv:1604.08407, arXiv:1509.00104 |
| **Latent-truth via RBMs / autoencoders** | neural amortisation of the LTM posterior | arXiv:1801.00283, arXiv:1807.10680 |

### 3.2 The sobering empirical results

Three papers to pin above the desk:

- **"Truth Finding on the Deep Web: Is the Problem Solved?"** (arXiv:1503.00303) — on real Stock and Flight data, sophisticated methods often barely beat weighted voting; the real error mass lives in *copying, extraction noise, and semantic mismatch*, not in the reliability estimator.
- **"Truth Discovery Algorithms: An Experimental Evaluation"** (arXiv:1409.6428) — 12 algorithms with reference implementations; no uniform winner; performance is regime-dependent.
- A 2026 replication of the same lesson in an LLM-evidence setting: **"Calibrated Trust, Not Sharper Prediction"** (arXiv:2608.14617) — evidence graphs with BP, sequential Bayesian odds, Dempster-Shafer combination and conformal prediction, stacked into one pipeline, did **not** sharpen prediction over the raw estimator; the value was in *calibration*.

**Design implication (load-bearing).** Budget spent on ever-fancier reliability parameterisations has poor ROI. Budget spent on (i) structural constraints, (ii) source-dependency modelling, (iii) calibration and abstention has good ROI. Weighted vote must be a first-class, always-reported baseline.

### 3.3 What truth discovery does *not* give us

None of the above models a **typed, relational, temporal** structure. They assume a flat set of `(object, attribute)` cells with independent truths. In OCEL:

- assertions are *coupled* (an E2O edge implies event and object existence; a timestamp is constrained by precedence);
- assertions are *typed*, so parameters should be shared across instances of a type (plate/template structure);
- assertions are *multi-valued and non-exclusive* by design.

That gap is what pillars II-IV fill.

---

## 4. The sparse / unreliable / numerous regime — the theory that actually binds

This section exists because the added constraint changes the answer, not merely the constants.

### 4.1 Identifiability is a property of the source-overlap graph, not of the model

Define the **source interaction graph** `G_S`: nodes are sources; edge `(s,s')` iff `s` and `s'` co-claim at least one assertion.

Two sharp results, both squarely about our regime:

- **Crowdsourcing with Sparsely Interacting Workers** (arXiv:1706.06660): for the single-coin symmetric-noise model, skills are asymptotically identifiable **iff** the limiting interaction graph is *irreducible* **and contains an odd cycle**. Bipartite (even-cycle-only) components leave a global sign ambiguity — the "everyone is anti-correlated with everyone" mirror solution cannot be broken.
- **Sparse rank-one matrix completion for skill estimation** (arXiv:1904.11608): the same problem viewed as completing a rank-one correlation matrix from observed co-occurrence entries; skills are recoverable **iff the sampling pattern has no bipartite connected component**.

**This is a checkable precondition and must be a diagnostic in the software, not an afterthought.** With sparse sources a real deployment easily produces a `G_S` that splits into islands; within an island reliabilities are jointly estimable, across islands they are comparable only through the prior. If an island is bipartite you get a sign flip. Detect, report, and fall back to the prior.

Supporting identifiability machinery for the general multi-class confusion-matrix case:

- **pairwise co-occurrence identifiability** — second-order statistics suffice under stated conditions, avoiding third-order tensor sample complexity (arXiv:1909.12325);
- **coupled / symmetric NMF with co-occurrence imputation** for *incomplete* annotation matrices, built explicitly for sparse annotator overlap (arXiv:2106.07193);
- **spectral initialisation plus one EM step** is provably optimal (arXiv:1406.3824); projected EM attains the minimax rate (arXiv:1310.5764).

### 4.2 Sample complexity per assertion, and a decidability test

Under Dawid-Skene the exact error exponent for aggregating `m` sources is `m·I(pi)`, where `I(pi)` is the average Chernoff information of the source pool (arXiv:1605.07696). To reach misclassification `eps` you need

```
m  >  log(1/eps) / I(pi)
```

sources on that assertion. In our regime `I(pi)` is small (unreliable) and `m = deg(a)` is small (sparse). Therefore:

> For a large fraction of assertions, **no aggregation rule whatsoever** can decide them from source votes alone.

Two architectural consequences:

1. **Abstention must be principled and per-assertion.** Estimate `deg(a)·Î(pi_a)` and expose it. Assertions below threshold get an explicit "undetermined" verdict rather than a confidently-wrong 0.51.
2. **The prior must carry the assertions the sources cannot.** This is the central justification for investing in the relational/structural layer: belief must flow *along the OCEL topology* from well-covered assertions into thinly-covered ones. Cross-assertion information here is not a refinement; it is the difference between decidable and not.

Refinements worth knowing: the minimax rate under practical assumptions, used to justify **worker clustering** (arXiv:1802.04551); and a **permutation-based** generalisation of Dawid-Skene with minimax rates and robustness to confusion-matrix misspecification (arXiv:1606.09632) — relevant because our sources' error structure is certainly misspecified.

### 4.3 One free parameter per source is unaffordable — pool

With `deg(s)` small, MLE of `theta_s` is hopeless (a source with 3 claims has an accuracy estimate with standard error ≈ 0.29). Four pooling strategies, all in the literature, all mutually compatible:

1. **Hierarchical / partial pooling.** `theta_s ~ p(· | phi)` with `phi` shared. Sources with few claims shrink toward the population; sources with many claims dominate their own estimate. Free in any Bayesian formulation, and the single highest-value structural choice in this regime.
2. **Source clustering / latent source types.** Bayesian nonparametric crowdsourcing (arXiv:1407.5017) motivates clustering precisely for *"early stages ... where the number of annotations is low ... not enough information to accurately estimate the bias introduced by each annotator separately"* — literally our situation. See also subgroup latent-factor models (arXiv:2302.02304), community-correlated reliability (arXiv:1806.02954), and the worker-clustering minimax analysis (arXiv:1802.04551).
3. **Worker-task specialisation.** A source is reliable only on its *matched* task type (arXiv:2004.00101, arXiv:2111.12550, arXiv:2302.07393). Very apt here: a barcode scanner is excellent at object identity and useless at activity semantics. In OCEL terms, reliability is indexed by `(source, assertion family, type)` — not by source alone.
4. **Feature-based reliability (SLiMFast, arXiv:1512.06474).** Regress accuracy on observable source features — modality, vendor, sampling rate, extractor version, sensor placement, upstream model, latency. Turns `|S|` free parameters into `d << |S|` shared ones and generalises to sources never seen before. **In the sparse-and-numerous regime this is arguably the most important single idea in the pillar.**
5. **Confidence-aware weighting (CATD).** Weight by a lower confidence bound on reliability rather than a point estimate; designed for long-tail sources. Cheap, and a good non-Bayesian fallback / initialiser.

These compose into one object: a **hierarchical GLM on source reliability**,

```
theta_{s, family, type}  =  f(source features; beta)  +  cluster effect  +  individual effect
```

That is the shape Stage 2 should adopt.

### 4.4 Sparsity is *good* for the inference algorithm

The one piece of luck in this regime: sparse claim graphs are locally tree-like, which is exactly where message passing is asymptotically exact.

- **Karger-Oh-Shah** (arXiv:1110.3564): on sparse `(l,r)`-regular random task-worker graphs an iterative message-passing / power-iteration estimator is order-optimal and analysable by density evolution; required redundancy scales as `1/q` in crowd quality `q`.
- **Optimal Inference in Crowdsourced Classification via BP** (arXiv:1602.03619): BP **exactly matches** the fundamental limit under Dawid-Skene — the strongest optimality statement in this literature.
- **Dense limit and regions of sub-optimality of message passing** (arXiv:1803.04924): the dense case maps to low-rank matrix estimation and identifies where AMP-style message passing is *sub*-optimal — worth knowing so we do not over-claim.
- **Robust deep learning from crowds with BP** (arXiv:2111.00734): *"due to the nature of sparsity in crowdsourcing, it is critical to exploit both probabilistic model ... and neural network"* — deepMF / deepBP.

**Design implication.** A factor-graph plus belief-propagation core is not an implementation convenience; in the sparse regime it is the theoretically indicated engine. Sampling (Gibbs / NUTS) is the right tool for the *low-dimensional parameter layer* (reliability hyper-parameters, calibration), not for millions of assertion variables.

### 4.5 Sparse coverage means missingness is probably informative

A source touching 0.1 % of assertions is not sampling uniformly; it looks where it looks. So `p(s speaks about a)` depends on the latent truth — a detector fires only on events that occurred, a scraper sees only records that exist.

- Treating silence as "no information" discards the strongest signal a detector-style source carries: its **false-negative rate**. LTM's two-sided quality (arXiv:1203.0058) is the minimal fix; truth-existence modelling (PMC4688015) is the complement.
- The general statistical frame is **MNAR / informative missingness**, with an explicit propensity or selection model estimated jointly (arXiv:2302.07540, arXiv:2206.14923, arXiv:2512.04392, arXiv:2608.23960). Recurring result: ignoring an informative missingness mechanism biases everything downstream; modelling it, or inverse-propensity-weighting by it, removes the bias.
- Practically, each source needs a declared **coverage semantics**:
  - *complete-over-scope* — silence inside scope is evidence of absence (full FN modelling),
  - *opportunistic* — silence is uninformative (MCAR within scope),
  - *selective* — silence is informative via an explicit propensity `pi_s(a | L*, features)`.

  This is part of the "known signal skeleton" the problem statement grants us, and it should be a **required field on every source adapter**.

### 4.6 Numerous sources means dependency is the dominant failure mode

`k` near-duplicate sources are not `k` independent votes; with weak sources, over-counting correlated evidence is the fastest route to confident error. Available tooling:

- copy detection (arXiv:0909.1776, arXiv:1503.00309); correlation-aware fusion (arXiv:1503.00306);
- **learning dependency structure without labels**: robust-PCA on the inverse generalised covariance (arXiv:1903.05844); static analysis when sources are programmatic (Coral, arXiv:1709.02477); LLM-based structure refinement for prompted sources (arXiv:2402.01867);
- the cost of getting it wrong is quantified in **Dependency Structure Misspecification in Multi-Source Weak Supervision** (arXiv:2106.10302).

Tension specific to our regime: dependency estimation *also* needs overlap. If `G_S` is thin, prefer a **hierarchical prior over declared source families** (same vendor, same upstream model, same physical site, same extractor version) over trying to learn the dependency graph from data. Declared provenance is cheap and, when sparse, strictly better than an under-determined estimate.

### 4.7 Summary of the regime

| property | consequence |
|---|---|
| sparse `deg(a)` | many assertions undecidable from votes -> **structural prior is primary**; abstention required |
| sparse `deg(s)` | per-source MLE hopeless -> **pool**: hierarchy + clusters + source features |
| thin overlap graph | identifiability is conditional and **must be diagnosed** (irreducible + odd cycle) |
| unreliable | small `I(pi)` -> large `m` needed; report decidability per assertion |
| numerous | dependency/copying dominates; prefer hierarchical source families over learned dependency graphs |
| sparse graph | **BP is provably near-optimal** -> factor-graph core; sampling only for the parameter layer |
| sparse coverage | missingness likely informative -> two-sided source quality + declared coverage semantics |

---

## 5. Pillar II — annotation aggregation and weak supervision (the same math, better theory and better engineering)

### 5.1 Crowdsourcing / annotation aggregation

- **Dawid & Skene (1979)**: per-annotator **confusion matrix** plus EM. Still the reference model. Confusion matrices beat scalar accuracy whenever a source has *systematic bias* — an OCR/NER extractor that reliably confuses activity `A` with `B` is not "70 % accurate", it is *structured*. In OCEL terms: per-source confusion over event types, over object types, and over qualifier assignment.
- **Truth inference at scale** (arXiv:1902.08918): Bayesian adjudication under *high* redundancy; useful contrast case — existing techniques beat simple baselines mainly at low redundancy or with adversarial workers.
- **Ground truth as a distribution, not a point** (arXiv:2003.00475; and arXiv:2301.01579 on ambiguity): some disagreement is *legitimate*, not noise. Highly relevant for OCEL: two sources reporting different timestamps for "the same" event may both be right about different physical sub-events. Do not force a point truth where the model should express a distribution.
- **Learning from crowds with sparse annotations** — coupled confusion correction (arXiv:2312.07331), directly on the "annotations are highly sparse so per-annotator expertise is hard to model" problem.
- Variational Bayes for continuous-valued crowd predictions (arXiv:2006.00778) — the regression analogue, needed for our continuous attributes and timestamps.

### 5.2 Weak supervision / data programming — the production-grade version of the same idea

This subfield is the closest existing engineering analogue to our problem: many noisy, cheap, *sparse-coverage* labelling functions, no ground truth, and a **label model** that recovers accuracies from agreement structure alone.

- **Data programming** (Ratner et al., 2016) and **MeTaL — multi-task weak supervision** (arXiv:1810.02840): sources label *different but related sub-tasks* at *different granularities*, with accuracies recovered by matrix completion on the inverse generalised covariance. This maps onto our problem almost perfectly: our "tasks" are event type, object type, E2O link, attribute value — related through the OCEL type hierarchy, with sources covering different subsets.
- **Learning dependency structures** (arXiv:1903.05844) — robust PCA, with improved recovery rates; sub-linear unlabelled-data requirements under stated conditions.
- **Hyper label model** (arXiv:2207.13545) — *amortised*: infers labels in a single forward pass without per-dataset parameter learning. Interesting as an eventual accelerator.
- **Triplet method** (implicit in the above): with three conditionally independent sources, per-source accuracies have a *closed form* from pairwise agreement rates alone. Extremely cheap, no labels, and a superb **initialiser** for the sparse regime — provided the required overlaps exist, which is exactly the §4.1 condition.
- Semi-supervised variants when a handful of gold labels exist (arXiv:2109.11410, arXiv:2008.09887) — worth wiring in, since in practice a small audited subset usually does exist and it breaks the identifiability symmetry immediately.

**Transfer to our design:** a two-stage pattern — (1) a *label model* over source votes and latent truth that needs no ground truth; (2) a *downstream model* trained on the resulting soft labels. Our Stage-2 design should keep the label model separable so it can be swapped, benchmarked, and initialised by the triplet method.

---

## 6. Pillar III — statistical relational learning: expressing OCEL topology

The OCEL type system is a **plate/template structure**: object types are classes, objects are instances, event types define attribute frames, qualifiers define legal roles. Statistical relational learning is exactly the technology for putting a probability distribution on that.

### 6.1 Object-oriented and relational Bayesian models

- **Object-Oriented Bayesian Networks** (arXiv:1302.1554) and **SPOOK** (arXiv:1301.6733): classes with BN fragments, instances, inter-object relations, and *uncertainty over structure*. This is essentially the OCEL metamodel with probabilities attached.
- **Relational Dynamic Bayesian Networks** (arXiv:1109.2137): stochastic processes that *create objects and relations over time* — this is what an object-centric process actually is.
- **Probabilistic relational models / CLP(BN)** (arXiv:1212.2519): relational-database-shaped BNs.
- Complexity of BNs specified by relational languages (arXiv:1612.01120): what gets hard, and when.

### 6.2 Templated / lifted factor graphs

- **Lifted graphical models: a survey** (arXiv:1107.4966) — the **par-factor graph** (parameterised factor graph) is the unifying formalism, and it is the right internal representation for us: define factor *templates* keyed by `(event type, qualifier, object type)` and *ground* them against the skeleton.
- Lifted first-order BP and its generalisation (arXiv:1606.09637); lifted marginal-MAP (arXiv:1807.00589); lifted inference for relational *continuous* models (arXiv:1203.3473).
- **Advanced Colour Passing / Approximate Lifted Model Construction** (arXiv:2504.20784): builds the lifted representation by detecting indistinguishable objects — and, importantly, *approximately*, since exact potential matching never happens in practice.
- Newer: lifted relational probabilistic inference via implicit learning (arXiv:2602.14890).

**Relevance to sparsity:** lifting is not just a speed trick here. Sharing parameters across indistinguishable groundings is the relational equivalent of §4.3's pooling — it is how you get statistical strength for a type from all its instances at once. **Lifting and pooling are the same idea viewed from two literatures.**

### 6.3 Probabilistic logic: MLN, PSL, ProbLog

| framework | inference | scale | why it matters here |
|---|---|---|---|
| **Markov Logic Networks** (Richardson & Domingos) | MaxWalkSAT / MC-SAT / lifted | Tuffy in an RDBMS (arXiv:1104.3216), Felix via Lagrangian task decomposition (arXiv:1108.0294); temporal MLNs (arXiv:2211.16414) | most expressive first-order weighted logic; grounding blows up |
| **PSL / Hinge-Loss MRFs** (arXiv:1505.04406) | **convex** MAP (consensus ADMM) | demonstrated on >1e9 ground rules | continuous truth values in [0,1]; the only framework in this family that is unambiguously scale-safe. Structure learning: arXiv:1807.00973; neural extension DeepPSL: arXiv:2109.13662 |
| **ProbLog** (distribution semantics) | weighted model counting over knowledge compilation (SDD/d-DNNF) | exact but #P-hard | exact semantics, great for *verification* of small subproblems and for compiling constraints |
| **DeepProbLog** (arXiv:1805.10872, arXiv:1907.08194) | WMC with neural predicates | limited | the right way to let a learned detector be a probabilistic predicate |
| **DeepSeaProbLog** (arXiv:2303.04660) | discrete-**continuous** neural probabilistic logic | limited | removes the finite-distribution restriction — relevant for timestamps/continuous attributes |
| **Scallop** (arXiv:2304.04812) | provenance semirings, top-k proofs | good | tunable exactness/scale trade-off via provenance |
| Scaling PNL | A-NeSI amortised inference (arXiv:2212.12393); DPNL exact-but-provenance-free (arXiv:2501.18202); GNN-accelerated MLN (arXiv:2001.11850); pLogicNet (arXiv:1906.08495) | — | the current frontier for making probabilistic logic scale |

**Assessment for our scenario.** PSL/HL-MRF is the strongest candidate for the *soft-constraint layer* at OCEL scale: convex MAP, billion-rule demonstrations, and hinge-loss rules express exactly the constraints of §2.2 (implication, cardinality, transitivity of identity). Its weakness is that it gives you a MAP point in `[0,1]^n`, **not calibrated marginals** — the hinge-loss "probabilities" are not posterior probabilities. Given §3.2's finding that *calibration* is where the value is, PSL cannot be the whole answer. It can be a constraint-propagation and MAP layer feeding a properly probabilistic layer.

ProbLog/pyAgrum earn their place as **exact oracles on small subgraphs** — for verifying that the approximate engine is right, and for compiling declarative constraints into factors.

---

## 7. Pillar IV — mixed graphical models and hybrid inference

Our variables are irreducibly mixed: binary existence and link indicators, categorical types, ordered continuous timestamps, and attributes spanning real / count / ordinal / categorical / text.

### 7.1 Pairwise mixed MRFs

- **Lee & Hastie** (arXiv:1205.5012, PMC4465824): pairwise MRF with Gaussian continuous nodes, Potts discrete nodes, and continuous-discrete cross terms; group-lasso structure learning; node-conditionals are Gaussian (continuous) and multinomial (discrete), enabling pseudo-likelihood estimation.
- **Chen, Witten & Shojaie** (arXiv:1311.0085, PMC5018402): node-conditional exponential families; identifies the **restrictions on parameter space required for a well-defined joint density** — a real trap, not a technicality.
- **Yang, Baker, Ravikumar, Allen & Liu — A General Framework for MGM** (arXiv:1411.0288): exponential-family MRFs across count/binary/continuous/skewed nodes. Known caveat: Poisson MRFs require *negative-only* edge parameters for normalisability — matters directly if we model counts (and OCEL attributes frequently are counts).
- **Pairwise Exponential MRF (PE-MRF)** (PMC6436845): ADMM over heterogeneous exponential-family domains.
- **Vector-Space MRFs** (arXiv:1505.05117): node domains in arbitrary vector spaces — multinomial, Dirichlet. Relevant when a "value" is itself a simplex (a source that emits a soft distribution rather than a hard label — which is exactly what a probabilistic source does).
- **Score-matching / regularised quadratic scoring** (arXiv:1809.05638): avoids the partition function entirely. Attractive because our normalising constant is hopeless.
- Bayesian MGM with zero-inflation (arXiv:2505.15464); semiparametric exponential-family graphical models (arXiv:1412.8697); high-dimensional MGM (arXiv:1304.2810).

### 7.2 The latent Gaussian copula route (the cleanest unification)

Model every non-structural variable as a monotone transform of a latent Gaussian:

- **nonparanormal / latent Gaussian copula for mixed data** (arXiv:1404.7236) — binary and mixed;
- **arbitrary mixed data** (arXiv:2211.11700) — continuous, count, binary, ordinal, truncated in one framework;
- **heterogeneous multi-group** copula graphical models (arXiv:2210.13140);
- estimation via **bridge functions** linking Kendall's tau to the latent correlation (arXiv:2205.06868, PMC10019899); penalised EM / copula-skeptic estimators (arXiv:1401.5264); Bayesian Gaussian copula factor models with missing values (arXiv:1806.04610); causal discovery on mixed data through the same device (PMID 35968913).

**Why this matters for us:** it collapses the entire mixed-attribute problem into a *single latent Gaussian precision matrix*, which then plugs directly into a Gaussian factor graph — where inference is exact, or handled by Gaussian belief propagation at enormous scale. It also degrades gracefully under sparsity: rank correlations are estimable from far fewer observations than a full parametric mixed MRF.

### 7.3 Hard vs. soft constraints

| encoding | pros | cons |
|---|---|---|
| **hard zero factors** | posterior is guaranteed coherent; sample any realisation and it is a *valid* OCEL | can make the graph disconnected/deterministic, break BP convergence, and make MCMC mixing terrible; one wrong constraint zeroes the truth |
| **soft high-weight penalties** (HL-MRF style) | convex, robust to a mis-stated constraint, degrades gracefully | posterior can put mass on invalid logs; downstream consumers must repair |

The literature's answer is *both*, layered: HoloClean (arXiv:1702.00820) relaxes denial constraints into factor weights; probabilistic databases under denial constraints study consistency directly (arXiv:1303.3233); Daisy relaxes DC violations on demand (arXiv:2002.06163); probabilistic unclean *graph* databases extend the frame to graph data (arXiv:2109.14112).

**Recommendation to carry into Stage 2:** hard factors for constraints that are definitional (spec-level type/attribute domains, referential integrity) and soft for constraints that are *domain beliefs* (lifecycle orderings, cardinalities that are usually-but-not-always true). Make the classification explicit and per-constraint, and let the modeller move a constraint between the two.

### 7.4 Hybrid discrete-continuous inference engines

- **DC-SAM** (arXiv:2204.11936): general MAP over discrete-continuous factor graphs by alternating minimisation; provides the library abstraction we would otherwise have to invent.
- **Variable elimination in hybrid factor graphs** (arXiv:2601.00545): a *new* framework producing a hybrid Bayes net supporting **exact** MAP and marginalisation over both variable kinds — the current state of the art in this line and the direction GTSAM's hybrid support is going.
- **iMHS** (arXiv:2103.13178): incremental multi-hypothesis smoother; eliminates a hybrid factor graph into a multi-hypothesis Bayes tree, keeping discrete-sequence hypotheses — directly analogous to keeping multiple structural interpretations of a log.
- **Gaussian belief propagation at scale**: Robot Web (arXiv:2202.03314) — distributed, asynchronous GaBP on a huge nonlinear factor graph with a trivially simple protocol; Hyperion (arXiv:2407.07074) — symbolic GaBP for continuous-time estimation fusing asynchronous multi-modal sensors. Both are strong evidence that GaBP handles our shape (huge, sparse, heterogeneous, asynchronous) in practice.
- **Chordal sparsity for global optimality** (arXiv:2605.30617): when local solvers land in bad minima, convex relaxation with chordal structure recovers or certifies the global optimum.

### 7.5 Learned / amortised message passing (optional later layer)

If loopiness or model misspecification hurts, message passing can be learned:

- **Neural Enhanced BP on factor graphs** (arXiv:2003.01998) — GNN correction on top of BP messages, precisely for "poor approximation of the data generating process, or loops".
- **Belief Propagation Neural Networks** (arXiv:2007.00295) — parameterised operators that provably retain BP's properties.
- **Factor-graph equivariant networks** (arXiv:2109.14218) and **Factor Graph Neural Networks** (arXiv:2308.00887) — the correct inductive biases (permutation equivariance over indices, orderings, assignments).
- Deep attentive BP (arXiv:2209.12000); differentiable nonparametric BP (arXiv:2303.04616); NSNet (arXiv:2211.03880).

**Position:** keep this as a *pluggable message-passing operator*, not a core dependency. It is the natural home for the GPU we have. It is also where the least theory and the most risk live, so it must not be on the critical path.

### 7.6 Probabilistic circuits as a tractable fusion backbone (2026 current)

- **Latent Posterior Factors** (arXiv:2603.15670): turns VAE latent posteriors into soft likelihood factors consumed by a Sum-Product Network — explicitly framed as the middle ground between "neural aggregation with no uncertainty" and "probabilistic logic with hand-engineered predicates". Very close in spirit to what we want for unstructured sources.
- **C²MF** (arXiv:2603.26629): context-specific, per-instance source credibility via a Conditional Probabilistic Circuit — reliability that varies with *context* rather than a static per-source scalar. Directly relevant to §4.3's specialisation point.

Circuits buy tractable exact marginals/MAP on a restricted-but-expressive family. Worth tracking as an alternative back end; not the first choice, because encoding OCEL relational constraints into a circuit is awkward.

---

## 8. Pillar V — identity, entity resolution, and cleaning

If A10 (identity) is latent, we are in Bayesian entity resolution territory.

- **Bipartite graphical record linkage** (arXiv:1312.4645, arXiv:1403.0211, arXiv:1409.0643): records link to *latent entities*, not to each other — which conveniently makes transitivity automatic instead of a constraint.
- **Exchangeable random partition priors** (arXiv:2301.02962): a principled and tractable prior over the linkage structure, plus a corrected distortion model for categorical attributes.
- **d-blink** (arXiv:1909.06039): distributed, partially-collapsed Gibbs — the scalability answer, since naive ER inference is quadratic in records.
- **Variational Bayes for merging noisy databases** (arXiv:1410.4792); post-hoc blocking (arXiv:1905.05337, arXiv:1710.10558).
- **Propagating ER uncertainty into the downstream task** (arXiv:1810.01538): the failure mode is picking a representative record and pretending it is certain.
- **PClean** (arXiv:2007.11838): a DSL for Bayesian cleaning over relational data with a non-parametric relational-DB model — the closest existing thing to "a language for writing our prior".
- Query-driven sampling for collective ER (arXiv:1508.03116) — do ER lazily, only where a query needs it. Attractive given our scale.

**Sparse-regime note:** ER is where thin evidence bites hardest, because the candidate space is quadratic while the evidence per pair is near zero. Blocking driven by the OCEL type system and qualifier signatures (§2.2 item 3) is not optional — it is the only thing that makes this tractable. Prefer *latent-entity* formulations (linear in records) over *pairwise-match* formulations (quadratic, and requiring transitivity constraints).

---

## 9. Pillar VI — the domain-native literature: uncertainty in (object-centric) process mining

This is where our problem is *named*, even though it is not solved at the joint level.

### 9.1 Uncertain event data

- **Pegoraro & van der Aalst, "Mining Uncertain Event Data in Process Mining"** (arXiv:1910.00089): the reference **taxonomy** of uncertain logs — *strong* uncertainty (a set of possible values) vs. *weak* (a probability distribution), applied to activity label, timestamp, and case membership. Use this taxonomy verbatim; it is the field's shared vocabulary.
- **Conformance checking over uncertain event data** (arXiv:2009.14452); **probability estimation of uncertain trace realisations** (arXiv:2108.08615) — estimating the probability of each possible realisation, which is exactly our "coherent joint samples" requirement.
- **Stochastically known logs** (Cohen & Gal): arXiv:2203.07507 defines stochastic trace models and alignment over them; arXiv:2106.03324 characterises the problem of relating a *stochastic observation* to a process model and argues explicitly against degrading probabilistic knowledge into interval bounds. That argument is ours too.
- **Probabilistic trace alignment** (arXiv:2107.03997); **alignment-based conformance over probabilistic events** (arXiv:2209.04309); **fuzzy logs against declarative temporal specs** (arXiv:2406.12078).
- Stochastic model quality: entropic relevance (arXiv:2007.09310); stochastic alignments (arXiv:2507.06472); optimisation-based stochastic process discovery (arXiv:2406.10817); model-driven stochastic trace clustering (arXiv:2506.23776).

### 9.2 Where the signals come from — event abstraction

Our "sources" are, in domain terms, event-abstraction pipelines:

- IoT-to-process-event abstraction: a DSL and architecture for detecting process activities from sensor streams (arXiv:2507.00686); IoT Miner (arXiv:2509.05769); the IoT/BPM framing paper (arXiv:2405.08528); EdgeMiner for distributed streaming abstraction (arXiv:2405.03426); the classic smart-home abstraction study (arXiv:1705.10202).
- LLM-based event abstraction and multi-source log integration (arXiv:2409.03478) — the modern "source" and a natural adapter type for us.
- Dempster-Shafer for uncertainty in CEP (PMC7962120) — the main non-Bayesian alternative for combining unreliable event detectors. Worth knowing; DS combination is known to behave badly under high conflict, and §3.2's arXiv:2608.14617 found no benefit from it in a fusion stack.

### 9.3 Object-centric specifics

- OCEL 2.0 spec and resources (arXiv:2403.01975, arXiv:2403.01982); DOCEL and datasets (arXiv:2309.14092); data-awareness of OCELs (arXiv:2212.02858); Dirigo / OCED extraction (arXiv:2411.07490).
- OCEL to temporal **event knowledge graphs** with attribute change (arXiv:2406.07596) — the graph view we will effectively be doing inference over.
- Case-notion selection via ER-schema (arXiv:2607.26384); multi-dimensional OCPM operations — drill-down, roll-up, unfold, fold (arXiv:2412.00393). These matter because our *granularity* is a modelling choice, and these operations define the legal moves between granularities.
- Object-centric predictive monitoring with heterogeneous graph encodings: HOEG (arXiv:2404.05316), EHHN hypergraph networks (arXiv:2607.01785), collaborative-process PPM (arXiv:2608.27671). These are the *discriminative* counterpart to our generative model, and a natural downstream consumer.
- Precision and fitness for object-centric models (arXiv:2110.05375) — evaluation vocabulary.

### 9.4 The gap

> There is no established framework that maintains a **joint probabilistic belief over a full OCEL 2.0 log**, fused from **many structurally-typed, sparse, unreliable sources**, with **calibrated marginals, coherent joint realisations, and principled abstention**.

Uncertain-log work is per-attribute or per-trace, largely single-source, and takes the uncertainty as *given input* rather than *inferring* it from source agreement. Truth-discovery work infers source quality but is flat, relation-free, and assumes dense-ish coverage. Weak supervision has the right label-model machinery but no relational/temporal structure. SRL has the structure but no source-reliability model and poor calibration at scale.

**That intersection is the contribution area, and it is well-defined.**

---

## 10. Theory: properties the model must satisfy

A checklist that Stage 2 must answer against, each with its literature anchor.

1. **Identifiability.** Without ground truth, source quality and truth are identifiable only up to symmetry (the "all sources adversarial" mirror). Breakers, in increasing order of strength: (a) priors placing mass on better-than-chance, (b) declared anchor/gold sources, (c) a small audited subset, (d) the odd-cycle/irreducibility condition on `G_S` (arXiv:1706.06660, arXiv:1904.11608), (e) three conditionally independent sources per assertion for the closed-form triplet solution.
2. **Conditional independence is the load-bearing assumption.** Violations (copying, shared upstream extractor) collapse effective sample size. Must be modelled (§4.6) or declared.
3. **Shrinkage under thin evidence.** Sources with few claims must shrink to the prior; assertions with few sources must shrink to the structural prior. Hierarchical Bayes delivers both for free; CATD delivers the first cheaply.
4. **Calibration, not just ranking.** Report reliability diagrams / ECE on any audited subset. arXiv:2608.14617 is the cautionary tale: a fusion pipeline that improves nothing but claims to.
5. **Coherence.** Posterior samples should be *valid* OCEL logs (types, referential integrity, precedence). See §7.3 for the hard/soft split.
6. **Marginals *and* joints.** Downstream process mining consumes whole logs (arXiv:2108.08615, arXiv:2203.07507). Expose both.
7. **Abstention.** Per-assertion decidability from the sample-complexity bound of §4.2. A three-valued output (`true` / `false` / `undetermined`) plus a posterior, not a bare probability.
8. **Attribution / provenance.** For each assertion, the decomposition of the posterior log-odds into contributions from prior, each source, and each structural factor. This is what makes the system auditable — and, with weak sources, auditability is the product.
9. **Robustness to misspecification.** Prefer permutation-model-style robustness (arXiv:1606.09632) and score-matching-style estimators (arXiv:1809.05638) over models that are only correct if the confusion structure is exactly right.
10. **Graceful degradation.** With zero sources on an assertion the answer must be the structural prior, not a crash or an arbitrary 0.5.

---

## 11. Candidate model families — comparison for *our* scenario

Legend: ++ strong, + adequate, o weak, - poor.

| | A. Hierarchical Bayesian generative (LTM/D-S + relational plates) | B. HL-MRF / PSL | C. Hybrid discrete-continuous factor graph (GTSAM / DC-SAM) | D. Probabilistic logic program (ProbLog / DeepProbLog) | E. Latent Gaussian copula MGM | F. Neural / amortised |
|---|---|---|---|---|---|---|
| expresses OCEL topology | + (via plates) | ++ (first-order rules) | + (templated factors) | ++ | o | + |
| mixed variable types | + | o (all in [0,1]) | ++ | o (finite; ++ with DeepSeaProbLog) | ++ | ++ |
| hard constraints | + | ++ (soft, convex) | + | ++ (exact) | - | o |
| scale (1e6+ assertions) | o (MCMC), + (VI) | ++ | ++ | - | ++ | ++ |
| **calibrated uncertainty** | ++ | - (MAP in [0,1], not posterior) | + (Gaussian/hybrid marginals) | ++ (exact where feasible) | + | o |
| learns source quality w/o labels | ++ | + (weight learning) | + | + | o | + |
| **behaviour in the sparse regime** | ++ (pooling is native) | + | ++ (BP optimality) | o | + | - (data-hungry) |
| tooling maturity (Python) | ++ (PyMC/NumPyro) | + (PSL is JVM; pslpython wrapper) | + (GTSAM Python) | + (ProbLog, pyAgrum) | + (statsmodels/custom) | ++ (torch) |
| GPU | + (NumPyro/JAX) | o | o | o | + | ++ |

**Reading of the table.** No single family wins. The columns that matter most given §4 are *behaviour in the sparse regime*, *calibrated uncertainty*, and *scale* — and those three are split across A, C, and B respectively.

**Therefore the indicated shape is layered, not monolithic:**

- a **par-factor-graph core** (§6.2) as the single internal representation, with pluggable factor families;
- **belief propagation** (loopy discrete + Gaussian) as the default assertion-level engine, justified by §4.4's optimality results, with exact variable elimination on small/tree-structured subgraphs;
- a **low-dimensional hierarchical parameter layer** for source reliability, calibration and structural hyper-parameters, fit by full Bayes (NUTS) or SVI — this is small enough for real MCMC, and it is where §4.3's pooling lives;
- a **convex constraint layer** (HL-MRF-style) available for constraint propagation and MAP at scale, feeding the probabilistic layer rather than replacing it;
- **exact oracles** (ProbLog / pyAgrum / exact VE) for verification on small subproblems and for compiling declarative constraints;
- an optional **learned message-passing** operator (§7.5) as a later accelerator, off the critical path.

The strong claim behind this: *the layers are not alternatives, they are different scales of the same factor graph.* Stage 2's main job is to define the single representation that all of them read and write.

---

## 12. Open questions to resolve in Stage 2

1. **Granularity of the assertion algebra.** Are candidate events enumerated up front (fixed universe with existence variables), or generated (open universe, transdimensional)? Fixed-universe is vastly cheaper and is the right v1; open-universe needs RJMCMC or a nonparametric prior and should be an explicit non-goal for v1.
2. **How much of the skeleton is clamped by default.** Proposal: schema always clamped; instance existence and links latent by default; identity clamped by default (latent behind a flag).
3. **Reliability parameterisation.** Concretely: which of {per-source scalar, per-source confusion matrix, per-(source, assertion-family) confusion, feature-regressed, cluster-pooled} is the default, and how they nest so the default can be relaxed.
4. **Coverage semantics per source.** The three-way declaration of §4.5 must be mandatory in the source adapter interface. What is the default, and can it be estimated?
5. **Behavioural prior strength.** Options in ascending cost: independent type-marginals; object-type-conditioned directly-follows counts; stochastic labelled Petri net per object type; marked temporal point process (arXiv:2501.14291, arXiv:2312.15045 for set-valued marks). Where is the knee in the effort/benefit curve, given that §4.2 says the prior is doing most of the work?
6. **Hard vs. soft per constraint class** (§7.3) — the explicit table.
7. **Copula vs. direct mixed MRF** for the attribute layer (§7.1 vs §7.2).
8. **Inference contract.** What exactly does the engine return: marginals, MAP, `k` posterior samples, decidability flags, attribution vectors? This is the API and it should be pinned before any modelling code.
9. **Evaluation protocol.** Synthetic generator with known ground truth (essential — nothing else lets us measure calibration); ablations over `deg(a)`, `deg(s)`, source accuracy, and copy structure; the mandatory weighted-vote baseline; ECE/reliability diagrams; sensitivity to a deliberately misspecified constraint.
10. **Diagnostics as first-class output.** The `G_S` connectivity/odd-cycle test (§4.1), per-assertion decidability (§4.2), effective-sample-size after dependency correction (§4.6). These are not logging; they are results.

---

## 13. References

**OCEL 2.0 / object-centric process mining**
arXiv:2403.01975 (OCEL 2.0 specification) · arXiv:2403.01982 (OCEL 2.0 resources) · arXiv:2309.14092 (DOCEL datasets) · arXiv:2212.02858 (data-awareness) · arXiv:2411.07490 (Dirigo) · arXiv:2406.07596 (OCEL to temporal EKG) · arXiv:2412.00393 (multi-dimensional OCPM operations) · arXiv:2607.26384 (ER-guided case-notion selection) · arXiv:2110.05375 (precision & fitness) · arXiv:2404.05316 (HOEG) · arXiv:2607.01785 (EHHN) · arXiv:2608.27671 (collaborative PPM) · PMID 38944260 (OMOP to OCEL)

**Uncertainty in process mining**
arXiv:1910.00089 · arXiv:2009.14452 · arXiv:2108.08615 · arXiv:2203.07507 · arXiv:2106.03324 · arXiv:2107.03997 · arXiv:2209.04309 · arXiv:2406.12078 · arXiv:2007.09310 · arXiv:2507.06472 · arXiv:2406.10817 · arXiv:2506.23776

**Event abstraction / IoT sources**
arXiv:2507.00686 · arXiv:2509.05769 · arXiv:2405.08528 · arXiv:2409.03478 · arXiv:2405.03426 · arXiv:1705.10202 · PMC7962120 (Dempster-Shafer CEP)

**Truth discovery / data fusion**
arXiv:1203.0058 (LTM) · arXiv:0909.1776 · arXiv:1503.00310 · arXiv:1503.00309 · arXiv:1503.00306 · arXiv:1503.00303 · arXiv:1409.6428 · arXiv:1512.06474 (SLiMFast) · arXiv:1708.02018 (SmartMTD) · arXiv:1904.10217 (hierarchies) · PMC4688015 (truth existence) · PMC4688022 (evolving truth) · arXiv:1806.02954 · arXiv:1906.10470 · arXiv:1210.0954 · arXiv:1603.02056 · arXiv:1604.08407 · arXiv:1509.00104 · arXiv:1801.00283 · arXiv:1807.10680 · arXiv:1611.01868 · arXiv:1702.00567

**Crowdsourcing theory — the sparse regime**
arXiv:1110.3564 (Karger-Oh-Shah) · arXiv:1602.03619 (BP optimality) · arXiv:1803.04924 (dense limit, sub-optimality) · arXiv:1706.06660 (sparsely interacting workers: irreducible + odd cycle) · arXiv:1904.11608 (rank-one completion identifiability) · arXiv:1909.12325 (pairwise co-occurrence identifiability) · arXiv:2106.07193 (CNMF with co-occurrence imputation) · arXiv:1406.3824 (spectral + EM) · arXiv:1310.5764 (minimax rates) · arXiv:1605.07696 (exact error exponent) · arXiv:1606.09632 (permutation model) · arXiv:1802.04551 (minimax + worker clustering) · arXiv:1407.5017 (Bayesian nonparametric crowdsourcing) · arXiv:2302.02304 (subgroup latent factors) · arXiv:2004.00101, arXiv:2111.12550, arXiv:2302.07393 (worker-task specialisation) · arXiv:2312.07331 (sparse annotations) · arXiv:1902.08918 · arXiv:2003.00475 · arXiv:2301.01579 · arXiv:2006.00778 · arXiv:2111.00734 (deepMF/deepBP) · arXiv:1602.03481, arXiv:1403.3080 (adaptive/budget allocation)

**Informative missingness / MNAR**
arXiv:2302.07540 · arXiv:2206.14923 · arXiv:2512.04392 · arXiv:2608.23960 · arXiv:2512.03322

**Weak supervision / data programming**
arXiv:1810.02840 (MeTaL) · arXiv:1903.05844 (dependency structure learning) · arXiv:1709.02477 (Coral) · arXiv:2106.10302 (misspecification) · arXiv:2207.13545 (hyper label model) · arXiv:2402.01867 (LLM structure refinement) · arXiv:2109.11410 · arXiv:2008.09887

**Statistical relational learning / probabilistic logic**
arXiv:1107.4966 (lifted graphical models survey) · arXiv:1302.1554 (OOBN) · arXiv:1301.6733 (SPOOK) · arXiv:1109.2137 (RDBN) · arXiv:1212.2519 (CLP(BN)) · arXiv:1612.01120 (complexity) · arXiv:1606.09637 (lifted region-based BP) · arXiv:1807.00589 (lifted marginal MAP) · arXiv:1203.3473 (lifted continuous) · arXiv:2504.20784 (approximate lifted model construction) · arXiv:2602.14890 (implicit learning) · arXiv:1505.04406 (HL-MRF/PSL) · arXiv:1807.00973 (PSL structure learning) · arXiv:2109.13662 (DeepPSL) · arXiv:1104.3216 (Tuffy) · arXiv:1108.0294 (Felix) · arXiv:2211.16414 (temporal MLN) · arXiv:1805.10872, arXiv:1907.08194 (DeepProbLog) · arXiv:2303.04660 (DeepSeaProbLog) · arXiv:2304.04812 (Scallop) · arXiv:2212.12393 (A-NeSI) · arXiv:2501.18202 (DPNL) · arXiv:2001.11850 (GNN + MLN) · arXiv:1906.08495 (pLogicNet) · arXiv:2509.07122 (NeSy framework comparison)

**Mixed graphical models**
arXiv:1205.5012, PMC4465824 (Lee & Hastie) · arXiv:1311.0085, PMC5018402 (Chen, Witten & Shojaie) · arXiv:1411.0288 (general MGM framework) · PMC6436845 (PE-MRF) · arXiv:1505.05117 (vector-space MRF) · arXiv:1304.2810 · arXiv:1412.8697 · arXiv:1809.05638 (quadratic scoring) · arXiv:2505.15464 (Bayesian MGM) · arXiv:1507.00039

**Gaussian copula / nonparanormal**
arXiv:1404.7236 · arXiv:2211.11700 · arXiv:2210.13140 · arXiv:2205.06868 · arXiv:1401.5264 · arXiv:1806.04610 · PMC10019899 · PMC7756188 · PMID 35968913

**Hybrid discrete-continuous inference**
arXiv:2204.11936 (DC-SAM) · arXiv:2601.00545 (hybrid variable elimination) · arXiv:2103.13178 (iMHS) · arXiv:2202.03314 (Robot Web / GaBP) · arXiv:2407.07074 (Hyperion) · arXiv:2605.30617 (chordal sparsity)

**Learned message passing**
arXiv:2003.01998 (NEBP) · arXiv:2007.00295 (BPNN) · arXiv:2109.14218 (equivariant factor-graph nets) · arXiv:2308.00887 (FGNN) · arXiv:2209.12000 (deep attentive BP) · arXiv:2303.04616, arXiv:2101.05948 (DNBP) · arXiv:2211.03880 (NSNet) · arXiv:2010.09283 · arXiv:2606.06344

**Entity resolution / data cleaning / probabilistic databases**
arXiv:1312.4645, arXiv:1403.0211, arXiv:1409.0643 (bipartite graphical RL) · arXiv:2301.02962 (exchangeable partition priors) · arXiv:1909.06039 (d-blink) · arXiv:1410.4792 (VB) · arXiv:1905.05337, arXiv:1710.10558 (post-hoc blocking) · arXiv:1810.01538 (downstream inference) · arXiv:1508.03116 (query-driven ER) · arXiv:1702.00820 (HoloClean) · arXiv:2007.11838 (PClean) · arXiv:1801.06750 (probabilistic unclean DBs) · arXiv:2109.14112 (unclean graph DBs) · arXiv:1303.3233 (DCs in probabilistic DBs) · arXiv:2002.06163 (Daisy) · arXiv:2311.06517 (BClean) · arXiv:1506.08908 (BayesWipe) · arXiv:1204.3677 · arXiv:2106.09764

**Temporal point processes**
arXiv:2501.14291 (survey) · arXiv:2312.15045 (set-valued marks) · arXiv:2605.17568 (structured neural MPP) · arXiv:2309.02868 (relational inference) · arXiv:2509.24762 (amortised/in-context TPP) · arXiv:2406.06149 · arXiv:2412.08590

**2025-26 fusion currents**
arXiv:2603.15670 (latent posterior factors + SPN) · arXiv:2603.26629 (C²MF conditional probabilistic circuits) · arXiv:2608.14617 (calibrated trust, not sharper prediction) · arXiv:2607.10491 (EvidentialRAG) · arXiv:2605.17435 (BELIEF) · arXiv:2608.14509 (evidence tuples: hypothesis, reliability, rationale, provenance) · arXiv:2607.20529 (uncertainty-aware trust for multi-LLM) · arXiv:2604.18576 (sequential Bayesian linguistic beliefs) · arXiv:2604.03925 (externalised Bayesian inference) · arXiv:2504.04128 (event-conditioned credibility)

**Classical, no arXiv id**
Dawid & Skene (1979), *Maximum likelihood estimation of observer error-rates using the EM algorithm*, JRSS-C.
Yin, Han & Yu (2008), *Truth discovery with multiple conflicting information providers on the web*, TKDE.
Li et al. (2014), *Resolving conflicts in heterogeneous data by truth discovery and source reliability estimation* (CRH), SIGMOD.
Li et al. (2014), *A confidence-aware approach for truth discovery on long-tail data* (CATD), PVLDB.
Richardson & Domingos (2006), *Markov logic networks*, Machine Learning.
Ratner et al. (2016), *Data programming: creating large training sets, quickly*, NeurIPS.
Lauritzen & Wermuth (1989), conditional Gaussian distributions for mixed graphical association models, Annals of Statistics.
