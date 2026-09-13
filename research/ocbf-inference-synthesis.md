# Inference for OCBF: research synthesis and architectural evaluation

**The recommended foundation is exact inference where tractable, complemented by constraint-aware blocked sampling for joint questions.** Belief propagation and expectation propagation remain useful approximations and sources of proposal distributions. Their outputs should not, by themselves, define the meaning of a coherent object-centric posterior.

This synthesis extends the [grounding synthesis](ocbf-grounding-synthesis.md) and [grounding proposal](ocbf-grounding-proposal.md). Its companion, the [inference architecture proposal](ocbf-inference-proposal.md), states the recommended system and algorithms. The scope is theoretical architecture and algorithm selection, with no implementation specification or delivery schedule.

## Abstract

OCBF's proposed target is a probability distribution over admissible object-centric histories, conditioned on heterogeneous, revisable evidence under a fixed semantic skeleton. This target requires more than accurate individual assertion marginals: conformance, temporal overlap, and priority comparisons often depend on joint histories. This review evaluates exact elimination, hybrid Gaussian methods, cutset conditioning, message passing, variational inference, blocked Monte Carlo, sequential Monte Carlo, incremental smoothing, weighted model integration, and probabilistic circuits against that requirement. The principal recommendation is to exploit structural and distributional tractability before approximation, preserve normalizing constants across competing interpretations, and use posterior-corrected joint sampling when exact inference becomes prohibitive. Computation must remain tied to one versioned evidence state and one declared model. Numerical diagnostics, model sensitivity, missing coverage, and physical provenance remain separate dimensions. No factory-scale runtime, calibration, or accuracy result is established by this analysis.

## Contents

1. [Inference target and decision criteria](#i1)
2. [Current substrate and implementation boundary](#i2)
3. [Structural decomposition and complexity](#i3)
4. [Exact and hybrid inference](#i4)
5. [Message passing and variational methods](#i5)
6. [Joint Monte Carlo and constrained sampling](#i6)
7. [Temporal and incremental inference](#i7)
8. [Compilation, lifting, and learned inference](#i8)
9. [Coherent process queries and hypothesis management](#i9)
10. [Evidence revision and trust sensitivity](#i10)
11. [Diagnostics and scientific validation](#i11)
12. [Comparative evaluation and recommended choices](#i12)
13. [Research questions and boundaries](#i13)
14. [Local evidence and sources](#sources)

<a id="i1"></a>
## 1. Inference target and decision criteria

### 1.1 One target, several computational methods

Retain the grounding's fixed skeleton $\mathcal S$, candidate universe $\mathcal U_\kappa$, knowledge time $\kappa$, and factory time $t$. Let $L$ be a random OCEL-like history, including event existence, activity, timestamps, qualified E2O/O2O relationships, and applicable attribute histories. Association and observation-system latent variables are auxiliary to this fixed semantics.

For one model configuration $\theta$, write a finite grounding as $x=(z,u)$, where $z$ collects discrete variables and $u$ continuous variables. The history is recovered by a declared map $L=\mathcal D(x)$. Define

$$
\pi_{\kappa,\theta}(x)
=\frac{\gamma_{\kappa,\theta}(x)}{Z_{\kappa,\theta}},
\qquad
\gamma_{\kappa,\theta}(x)
=\mathbf 1_{\Omega}(x)
\prod_{f\in F_\kappa}\psi_{f,\theta}(x_f),
\qquad
Z_{\kappa,\theta}=\int\gamma_{\kappa,\theta}(x)\,d\mu(x).
\tag{1}
$$

Here $\Omega$ is the admissible support, and $\mu$ combines the required counting and continuous measures. Factors include proper prior contributions and observation likelihoods with their dependence represented. Context conditioning is implicit. A positive finite $Z_{\kappa,\theta}$ is required.

The factor product is justified by the observation model; it is not obtained merely by giving each Kafka topic a factor. Common upstream measurements, association uncertainty, and source modes can induce dependencies across several Operations.

Mixed graphical models (MGMs) provide distributional families, not a single inference algorithm. A product of conditional terms used as a pseudolikelihood for estimation is not generally the joint likelihood required by Equation (1). The compatibility and normalizability requirements discussed in the grounding's [mixed-model analysis](ocbf-grounding-synthesis.md#s8) still apply.

The choice of algorithm must preserve this target or identify precisely the approximation made to it. A convenient inference engine does not justify silently changing source meaning, cardinality, occurrence scope, or missingness semantics.

### 1.2 The required output is query-dependent

For a process query $q$, the target is

$$
\mathbb E_{\pi_{\kappa,\theta}}[q(\mathcal D(X))]
=\frac{\int q(\mathcal D(x))\gamma_{\kappa,\theta}(x)\,d\mu(x)}
{Z_{\kappa,\theta}}.
\tag{2}
$$

Below, $q(x)$ abbreviates $q(\mathcal D(x))$ when no confusion arises.

| Question | Required information |
|---|---|
| Does a particular event exist? | A marginal probability. |
| How many of a fixed set of obligations are violated in expectation? | Corresponding violation probabilities; independence is unnecessary for the expected sum. |
| Did all required Operations complete? | A joint event probability, or justified bounds. |
| How long did a gap overlap downstream occupancy? | Joint occurrence, association, and time information. |
| Which Job has the greatest exception exposure? | A joint distribution across the compared Jobs, including shared influences. |
| Why did a result change after evidence correction? | Comparable inference under explicit old/new evidence states. |
| What would an intervention improve? | An additional causal model; ordinary posterior inference is insufficient. |

A sampler need not materialize an entire factory history if it can compute the necessary joint marginal or conditional expectation. Conversely, separate marginals cannot be treated as a substitute for joint information simply because their storage is convenient.

### 1.3 Evaluation criteria

The criteria are semantic fidelity, joint-query capability, handling of hard constraints, preservation of competing interpretations, evidence revisability, interpretable numerical error, and deliverability using existing assets. Runtime matters through the actual factor structure and query workload, not a raw message count.

The empirical constraints remain those in the grounding's [dated evidence register](ocbf-grounding-synthesis.md#evidence). Repeated reports and placeholder fields motivate the observation model; they do not establish how many independent latent variables a factory-wide graph must contain. The present analysis supplies source inspection and mathematical evaluation, not a new data extraction, benchmark, or calibrated factory fit.

<a id="i2"></a>
## 2. Current substrate and implementation boundary

The OCBF source baseline inspected for this evaluation is commit 1bb372b0fb6869fe27adda434336dca729a622f0. The findings below concern that code, not the whole proposed grounding.

| Existing element | Finding | Architectural consequence |
|---|---|---|
| Discrete BP | Damped sum-product with belief residuals, iteration limits, and oscillation diagnostics. | A reusable approximate marginal engine. |
| Gaussian EP | Cavity-based site updates, damping, and numerical diagnostics. | Useful local approximation, conditional on its factor and distribution assumptions. |
| Hybrid loop | Discrete beliefs influence continuous grounding; continuous feedback returns as unary factors. | Locally exact algebra inside an approximate loop is not global exact hybrid inference. |
| Exact backend | GTSAM discrete elimination with input-table and induced-clique guards; exact marginals and an MPE assignment. | A substantial existing asset for bounded exact problems and reference comparisons. |
| Exact result contract | No partition function or joint conditional representation is exposed by the current wrapper. | The required general joint-query interface is not already available. |
| Factor conversion | Each dense log table is independently max-shifted before exponentiation. | Harmless for normalized marginals of one fixed graph; discarded constants must be retained when comparing hypotheses or model evidence. |
| Degenerate normalization | The exact wrapper contains a uniform fallback when a marginal's total is nonpositive. | The proposed model instead requires an explicit invalid/undefined result when its evidence normalizer fails. |
| Reliability | High-level fusion initializes reliability from triplets; disabling fitting freezes that initialization. | Manual trust is a model input requirement, not an existing high-level switch. |
| Parameter fitting | When enabled, the final reliability fit occurs after the final belief computation. | Returned fitted parameters and beliefs must not be assumed to describe the same conditional posterior without checking their inference epoch. |
| Belief state | Discrete marginals, continuous summaries, and some local mixture information; no general joint OCEL sampler. | Marginal storage does not establish joint-query support. |

The source anchors are [discrete BP](C:/Users/Ryan/dada/ocbf/ocbf/inference/loopy_bp.py), [EP and hybrid feedback](C:/Users/Ryan/dada/ocbf/ocbf/inference/gabp_ep.py), [exact inference](C:/Users/Ryan/dada/ocbf/ocbf/inference/gtsam_exact.py), [fusion](C:/Users/Ryan/dada/ocbf/ocbf/pipeline.py), and [belief state](C:/Users/Ryan/dada/ocbf/ocbf/belief/state.py).

The earlier assessment also identified limitations in likelihood interpretation, silence, source dependence, and typed lifecycle/cardinality scope. Those are [model-compilation concerns](ocbf-grounding-synthesis.md#s13), not problems that can be solved merely by choosing a stronger numerical algorithm.

The current [inference explanation](C:/Users/Ryan/dada/ocbf/docs/explanation/inference.md) motivates BP using crowdsourcing optimality results. The relevant theorem concerns a Dawid–Skene model, random regular task–worker assignment graphs, and specified asymptotic and degree conditions. It does not establish BP optimality for arbitrary object-centric graphs with temporal variables and hard relational constraints.[^25] Likewise, there is no established theorem here that MCMC is always inferior on a large assertion graph.

<a id="i3"></a>
## 3. Structural decomposition and complexity

### 3.1 Complexity follows the representation

Dense discrete elimination has cost exponential in induced width. For an elimination ordering $\sigma$, an informative table-size quantity is

$$
C(\sigma)=\max_{B\in\mathcal B_\sigma}
\prod_{i\in B}|\mathcal X_i|,
\tag{3}
$$

where $\mathcal B_\sigma$ contains the intermediate variable scopes.[^1] The total number of variables, factor count, and retained archive size do not determine this quantity.

A high-degree shared Station is not automatically a hard inference problem: a star can remain a tree. Multiple overlapping cardinality, lifecycle, and shared-source factors can instead create expensive separators. The factor's representation also matters. An exactly-one constraint can have efficient dynamic-programming messages even when its dense table is large.

For continuous and mixed models, graph width is only part of the cost. Integration must stay in a manageable function family. Piecewise messages or mixture components can grow even on structurally simple graphs; weighted model integration has stricter tractability boundaries than the discrete case.[^17]

### 3.2 Normalize the computational representation

Clamped quantities and deterministic consequences can often be removed from the uncertain state, provided the transformation preserves support and weights. One categorical association variable may represent an exactly-one set of qualified links more efficiently than independently mutable Boolean links. Its decoded E2O assertions retain the original semantic meaning.

Such a transformation must preserve “none” or unresolved alternatives when they are admissible. It must not erase genuine multiplicity by forcing a one-of-many encoding where several participants are allowed. Nor should two labels for one physical event create two independent copies of prior mass.

Constraint propagation can rule out inconsistent candidates before numerical inference. It establishes logical impossibility relative to the model, not probability calibration or completeness of the candidate universe.

### 3.3 Regional inference needs boundary information

Suppose the target factors as
$\gamma(x_R,b,x_O)=\gamma_R(x_R,b)\gamma_O(b,x_O)$,
where $R$ is a region, $O$ its exterior, and $b$ their separator. Then

$$
\pi(x_R,b)\propto
\gamma_R(x_R,b)\,
m_{O\to R}(b),
\qquad
m_{O\to R}(b)=\int\gamma_O(b,x_O)\,d\mu(x_O).
\tag{4}
$$

This is the correct meaning of a posterior region. Dropping exterior factors corresponds to replacing the boundary message, not simply optimizing the same computation. Separate Job computations are independent only under an appropriate conditional-independence statement.

Shared source modes or resource states should remain common variables or enter through appropriate joint boundary messages. Independently redrawing a shared uncertainty for each Job changes the uncertainty of cross-Job rankings.

<a id="i4"></a>
## 4. Exact and hybrid inference

### 4.1 Enumeration, elimination, and junction trees

Enumeration provides an exact reference on a small finite support. Variable elimination sums out irrelevant variables while combining only factors that involve them; junction-tree methods organize such calculations for multiple marginal queries.[^1][^2]

For a block $a$ with remaining boundary $b$, elimination produces

$$
m(b)=\int\prod_{f\in F_a}\psi_f(x_f)\,d\mu(a).
\tag{5}
$$

Exact elimination can also retain normalized conditionals for backward sampling. A draw generated from those conditionals is a coherent joint draw from the represented distribution. Sampling independent returned marginals is a different procedure.

For repeated questions over one cohort and evidence state, the reusable product is an elimination representation, separator factors, or joint conditional structure. The existing exact wrapper supplies a useful foundation but does not currently expose all these outputs.

Exactness is relative to the factors and support actually compiled. Exact elimination of a relaxed type gate or incorrect lifecycle scope does not recover the intended semantics.

### 4.2 Discrete ambiguity with conditional continuous inference

Split the state into a discrete hypothesis $z$ and remaining quantities $u$. Define

$$
g(z)=\int\gamma(z,u)\,d\mu(u),
\qquad
\pi(z)=\frac{g(z)}{\sum_{z'}g(z')},
\qquad
\pi(u\mid z)=\frac{\gamma(z,u)}{g(z)}.
\tag{6}
$$

For a fully continuous $u$ on $\mathbb R^m$, suppose
$\log\gamma(z,u)=c_z-\frac12u^\top J_zu+h_z^\top u$
with $J_z$ positive definite. Then

$$
\log g(z)
=c_z+\frac12h_z^\top J_z^{-1}h_z
-\frac12\log\det J_z+\frac m2\log(2\pi).
\tag{7}
$$

The determinant and other mode-dependent normalizers matter. Comparing only optimized residuals or posterior peaks confuses mode height with integrated probability mass.

Recent hybrid factor-graph elimination work explicitly preserves these normalizers under conditional linear Gaussian assumptions. Its exact linear-Gaussian formulation is highly relevant; nonlinear linearization and hypothesis pruning introduce separate approximations.[^9] It is not a general exact solver for arbitrary timestamp censoring, inequality constraints, or observation likelihoods.

### 4.3 Temporal constraints break unrestricted Gaussian closure

If a hypothesis restricts the continuous state to a region $\mathcal C_z$, then its integrated mass becomes

$$
g_{\mathcal C}(z)
=g_{\mathbb R^m}(z)\,
\Pr\{U\in\mathcal C_z\mid z,\text{unrestricted Gaussian kernel}\}.
\tag{8}
$$

This truncation probability can change the relative weights of competing episode or association hypotheses. Independently clipping means, truncating marginal intervals, or multiplying separate order probabilities does not generally compute it.

Low-dimensional quadrature and recognized analytic distributions are reasonable numerical tools. Their error needs a separate qualification; an adaptive quadrature error estimate is not automatically a rigorous multidimensional posterior bound. Higher-dimensional or nonconjugate blocks may need constrained sampling.

Reported time intervals require the observation semantics established in the grounding. Replacing every interval by a uniform distribution or every duration by a Gaussian changes the model to suit the algorithm.

### 4.4 Cutset conditioning and Rao–Blackwellization

A cutset is a subset whose conditioning makes the remaining graph more tractable. A $w$-cutset bounds the induced width of that conditioned remainder.[^7]

If feasible cutset configurations can be enumerated, Equation (6) gives an exact mixture, provided its inner computations are exact. If not, sample the cutset while eliminating tractable remaining variables. For a query $q$,

$$
\mathbb E_\pi[q(X)]
=\mathbb E_{\pi(z)}
\left[\mathbb E[q(z,U)\mid z]\right].
\tag{9}
$$

Using the inner conditional expectation is Rao–Blackwellization. It can reduce estimator variance relative to additionally sampling those quantities, but increased per-sample cost and changed chain mixing prevent a universal runtime advantage.[^7]

The natural cutset variables here are unresolved episode assignments, linked occurrence alternatives, and discrete observation modes. Selection should follow the actual graph and distributional cost, not a rule that all continuous variables are always analytically solvable.

For a general cutset calculation, $z$ in Equations (6) and (9) denotes the selected cutset and $u$ the remainder, which can also contain discrete variables. Its integration measure then includes their counting measure. The Gaussian formula applies only to a wholly continuous Gaussian remainder.

<a id="i5"></a>
## 5. Message passing and variational methods

### 5.1 Loopy BP remains useful, with a narrower claim

Sum-product BP is exact on trees under its standard conditions.[^2] For loopy graphs it approximates marginal inference; convergence, local consistency, and posterior accuracy are distinct properties. The Bethe variational interpretation explains stationary points but does not make every converged solution a global optimum.[^3]

Damping, residual scheduling, and region selection can improve numerical behavior. They do not establish a bound on conformance probability or missing posterior modes. Generalized BP or join-graph propagation can retain stronger local dependencies, at increased region cost.[^8]

For this setting, BP has three defensible roles: approximate broad marginal screening, constructing candidate proposals, and diagnosing difficult regions by comparison with exact results. Its most important limitation is the lack of a general coherent joint distribution guaranteed by a set of loopy marginal beliefs.

### 5.2 EP and hybrid feedback

EP approximates a factor by projecting a tilted distribution formed from that factor and its cavity distribution. For a site approximation $\widetilde\psi_f$,

$$
q^{\setminus f}(x)\propto\frac{q(x)}{\widetilde\psi_f(x)},
\qquad
\widehat p_f(x)\propto\psi_f(x)\,q^{\setminus f}(x).
\tag{10}
$$

Moment matching updates the tractable approximation.[^5] A Gaussian projection can summarize one mode well while losing important alternative times or associations. An algebraically exact integral against an approximate cavity remains conditional on that approximate cavity.

Hybrid feedback should therefore have one explicit factor interpretation. A marginal posterior must not be sent back as if it were independent new evidence. Multi-variable boundary dependencies must not be silently replaced by unrelated unary messages.

EP is attractive for smooth or approximately unimodal continuous blocks. It is weaker as the sole representation for business-relevant discrete alternatives and discontinuous temporal obligations.

### 5.3 Variational inference and optimization-based alternatives

For a normalized trial density $q_\lambda$, conventional variational inference maximizes

$$
\mathcal L(\lambda)
=\mathbb E_{q_\lambda}[\log\gamma(X)]
 + H(q_\lambda)
=\log Z-\operatorname{KL}(q_\lambda\Vert\pi).
\tag{11}
$$

The family determines which dependencies and modes can be represented.[^6] A mean-field family can be particularly unsuitable when a hard constraint allows only coordinated assignments. If it assigns positive mass outside the target support, its KL divergence can be infinite.

Structured variational families mitigate that problem by retaining groups or using admissible parameterizations. A sequence of locally reasonable BP/EP and fitted-parameter updates should not be described as monotone optimization of one objective unless that objective and the required update conditions are actually established.

Tree-reweighted constructions provide upper bounds on the log partition function under their stated conditions.[^4] That does not automatically give upper and lower bounds on every posterior query. Bounding a probability ratio requires appropriate control of both its constrained numerator and its denominator.

MAP/MPE optimization, integer programming, and weighted constraint solving can produce feasible representative histories. They answer an optimization question. They do not produce posterior probabilities or posterior mean durations merely by returning a ranked list of solutions.

<a id="i6"></a>
## 6. Joint Monte Carlo and constrained sampling

### 6.1 Why the unit of a move matters

Consider an illustrative, nonempirical model with two Boolean links satisfying $x_1+x_2=1$. Its feasible states are $(1,0)$ and $(0,1)$. A single-site Gibbs update cannot move between them: holding either coordinate fixed forces the other.

If both true marginals are 0.5, independent marginal sampling also produces infeasible $(0,0)$ or $(1,1)$ with total probability 0.5. This is a mathematical counterexample, not a Mammut measurement.

Moves should instead update a categorical association or a coordinated block. More complex constraints may require multi-Operation reassignment, event-existence changes with their dependent links, or jointly proposed time changes. Constraint-aware sampling research directly addresses the rejection and support problems that arise from independent proposals.[^7][^8]

### 6.2 Blocked and collapsed Metropolis–Hastings

For a proposal density $r(z'\mid z)$ on a tractable cutset target $g(z)$, accept with

$$
a(z,z')
=\min\left\{1,
\frac{g(z')\,r(z\mid z')}
{g(z)\,r(z'\mid z)}
\right\}.
\tag{12}
$$

Where conditional block probabilities can be evaluated and normalized, blocked Gibbs is an alternative. Otherwise MH permits useful proposals without requiring their distribution to equal the posterior.

This is a suitable default sampling family for bounded retrospective cohorts: no artificial physical-time transition model is needed, and proposals can follow episode and association ambiguity. The requirements are substantial but clear: valid support, computable target ratios and proposal probabilities, appropriate irreducibility/aperiodicity, and acceptable exploration.

Local moves may still fail to connect remote modes. A combination of local and larger coordinated moves is useful only if their densities and invariant-target arguments remain valid. More chains improve the opportunity to discover disagreement; they do not certify that all modes were found.

### 6.3 BP-guided proposals must be corrected

An approximate belief can guide a sequential or block proposal, but its actual sampling density must be available. For importance sampling,

$$
z^{(s)}\sim r,\qquad
w_s=\frac{g(z^{(s)})}{r(z^{(s)})},
\qquad
\widehat{\mathbb E}[q]
=\frac{\sum_s w_s\,\mathbb E[q(z^{(s)},U)\mid z^{(s)}]}
{\sum_s w_s}.
\tag{13}
$$

The proposal must cover every target-positive configuration relevant to the claimed result. Drawing, rejecting, repairing, or renormalizing configurations can alter the proposal distribution. An unknown repair probability cannot simply be omitted from the correction.

Hybrid constrained-network research combines generalized message passing with Rao–Blackwellized importance sampling for this reason.[^8] Deterministic approximations can also guide SMC while the correction targets the original graphical model.[^10] Neither construction licenses uncorrected multiplication of approximate marginals.

Self-normalized importance estimates are generally biased at finite sample size, although they can be consistent under suitable conditions. A valid unbiased normalizing-constant estimator is a separate property.

### 6.4 Continuous blocks and approximate inner integration

Conditional Gaussian sampling is attractive when its assumptions hold. Smooth nonconjugate continuous blocks can use gradient-based methods such as HMC/NUTS; these do not directly sample unresolved discrete assignments, and standard smooth dynamics require appropriate treatment of support boundaries.[^24]

Censoring, hard order constraints, and discontinuous event predicates need distribution-aware treatment. A support-preserving reparameterization can help, but it must retain the intended prior and the appropriate Jacobian. Fitting an unconstrained Gaussian and rejecting afterward is not universally efficient.

If $g(z)$ is replaced by a deterministic approximate integral, MH targets the approximation unless further correction is supplied. Random inner estimates also require care. A pseudo-marginal construction can preserve the desired marginal target with a suitable nonnegative unbiased estimator and an extended state that retains the current estimator; independently recomputing noisy estimates on both sides is not generally the same algorithm.[^13]

This is a reason to keep analytically tractable or small numerically controlled blocks in the preferred scope. “Rao–Blackwellized” is not a guarantee when the inner marginalization is itself approximate.

A simpler fallback to approximate inner integration is to retain the difficult variables in the sampled state and evaluate the original joint target. This can cost more per effective draw, but avoids pretending that an approximate integrated likelihood is exact.

### 6.5 SMC for static graphical models

SMC does not require physical time. It can introduce variables or factors along a computational sequence, or traverse a sequence of tempered distributions ending at the full retrospective target.[^10]

For example, on a common admissible support,

$$
\pi_{\beta}(x)\propto
p_0(x)\,\Lambda(x)^\beta,
\qquad
0=\beta_0<\cdots<\beta_T=1.
\tag{14}
$$

When the likelihood is positive on that support, the incremental weight from $\beta_{r-1}$ to $\beta_r$ is proportional to $\Lambda(x)^{\beta_r-\beta_{r-1}}$. Hard support restrictions require explicit handling; tempering a hard zero does not automatically make disconnected modes reachable.

SMC populations offer several competing hypotheses simultaneously. Resampling can also remove rare modes, so rejuvenation and appropriate intermediate targets matter. Correctly constructed SMC can estimate normalizing constants without bias, but normalized query estimates remain finite Monte Carlo approximations.[^10]

Divide-and-conquer SMC exploits a decomposition tree and merges subproblem populations.[^26] Its relevance is structural; shared factors and merge corrections cannot be ignored. It is an attractive extension if sequential assembly or severe multimodality dominates, not a prerequisite for the smallest deliverable retrospective scope.

<a id="i7"></a>
## 7. Temporal and incremental inference

### 7.1 Three different meanings of “dynamic”

| Change | Example | Inference implication |
|---|---|---|
| Physical evolution in $t$ | An Operation starts or a tracked object changes place. | A trajectory model may describe state transitions. |
| Evidence evolution in $\kappa$ | A late completion or corrected tag association revises the past. | Recondition the affected historical model. |
| Computational evolution | More samples, changed factorization, or improved numerical precision. | Refine the approximation to a specified target. |

Only the first requires a physical transition model. Evidence revisions can be handled by recomputing a retrospective posterior at a new knowledge cutoff. Faster asynchronous updates do not make the model a DBN or CTBN.

### 7.2 Filtering and smoothing

When a justified transition model exists, filtering concerns the present conditional on evidence received so far, while smoothing revises earlier states using later evidence. A Rao–Blackwellized particle filter samples nonlinear/discrete components and analytically updates tractable components.[^11]

The relevance to Mammut is conditional: a tracking or source-mode model could support such a decomposition. It should not be inferred merely from event timestamps or telemetry frequency. Delayed evidence arriving outside the active window may require retrospective smoothing or replay.

Particle Gibbs with ancestor sampling is a research option for difficult latent trajectories, mitigating some path-degeneracy and mixing problems in state-space settings.[^12] It is more machinery than a bounded static cohort requires.

CTBN inference deals with finite-state trajectories in continuous time; clock-augmented models permit age-dependent residence times, and trajectory-based EP supports approximate inference under interval evidence.[^28][^29] Basic asynchronous CTBN semantics does not automatically represent one OCEL event synchronizing several objects. An operation-duration question also does not justify assuming exponential residence times.

### 7.3 Incremental factorization

Incremental multi-hypothesis smoothing represents shared trajectory history while retaining alternative discrete modes. Its explicit exponential hypothesis-growth problem is instructive for object-centric reconstruction.[^20] Non-Gaussian incremental approaches such as NF-iSAM use expressive learned conditional approximations, trading exact closed-form structure for more flexible density representations.[^21]

The useful architectural principle is dependency-aware reuse of elimination structure and cached conditionals. The robotics application does not establish that a factory model is Gaussian, that its hypotheses may be safely pruned, or that the same performance will transfer.

Incremental exactness requires that reused messages still represent the correct factors. A changed source interpretation can affect a larger region than the changed record's immediate object. A short loop can propagate that change through an entire connected component.

### 7.4 Fixed-lag limitations

Marginalizing past variables produces a boundary factor. Retaining that factor can preserve the information relevant to future variables for the original model. It does not retain arbitrary query access to the eliminated past.

If a late revision changes an old factor, the compressed boundary generally cannot be repaired by updating a single current marginal. Recovery requires sufficient retained conditionals, the relevant historical factors, or replay. A fixed-lag representation is therefore a declared query/retention restriction, not lossless bitemporal reasoning.

For the present delivery priorities, periodic retrospective inference over explicit evidence snapshots is the cleaner default. Incremental temporal methods become justified by a demonstrated workload or a prospective query requirement.

<a id="i8"></a>
## 8. Compilation, lifting, and learned inference

### 8.1 Weighted model counting and integration

Logical support and probabilistic weights can be combined through weighted model counting for discrete variables and weighted model integration for mixed variables.[^16] For a logical process predicate $A$,

$$
\Pr_\pi(A)=
\frac{\int\mathbf 1_A(x)\gamma(x)\,d\mu(x)}
{\int\gamma(x)\,d\mu(x)}.
\tag{15}
$$

This aligns well with occurrence, multiplicity, and bounded-order queries. It does not by itself make the numerator or denominator tractable.

Hybrid integration requires manageable constraint languages and function families. In particular, tree-shaped structure alone does not guarantee efficient integration in general WMI models.[^17] Arbitrary nonlinear source models cannot be treated as interchangeable leaves of an exact symbolic solver.

### 8.2 Probabilistic circuits

Probabilistic circuits trade representational restrictions for efficient supported queries.[^18] Smoothness, decomposability, determinism, and compatibility between representations determine which marginalization, product, maximum, and composite operations are tractable.[^19]

A fixed semantic skeleton creates opportunities to reuse compiled patterns, but not a guarantee that the grounded circuit stays fixed. New candidates, changed dependencies, or revised hard support can require structural changes. Evidence updates that only change compatible numerical weights are a simpler case.

Compiling a correct finite factor model into a tractable circuit is different from learning a compact circuit that approximates the factory distribution. Both can be useful; only the former preserves the specified target without an additional model approximation.

The strongest role here is a possible specialized accelerator for repeated finite conformance predicates. Compilation cost, circuit size, and changing grounded support make a universal circuit representation premature.

### 8.3 Lifted inference

Repeated object types do not imply exchangeability after conditioning on object-specific evidence. Lifted inference can exploit genuine symmetry, but relational evidence can break those symmetries and yield hard inference problems.[^22]

Template reuse is valuable even when probabilistic lifting is unavailable: the same operation-role factor can be instantiated many times. This is a software/model regularity, not proof that distinct Jobs may be averaged into one random variable.

### 8.4 Accelerators and learned proposals

PGMax demonstrates efficient accelerator-oriented loopy BP for discrete factor graphs.[^23] That establishes the feasibility of speeding up supported computations; it does not remove loopy approximation bias, mixed-domain restrictions, or incorrect evidence semantics.

Neural proposals and normalizing flows can be useful when many similar inference problems recur.[^21] The fixed vocabulary alone does not establish the training coverage needed for evolving source behavior. A learned proposal with an evaluable density can be corrected by MH or importance weights; an opaque learned marginal predictor cannot automatically be treated as a posterior sampler.

For the present priorities, hardware acceleration and amortization are secondary to reducing the relevant state and preserving the target. No GPU-first or learned-inference dependency is recommended.

<a id="i9"></a>
## 9. Coherent process queries and hypothesis management

### 9.1 One joint result for related answers

The useful common output is a representation of joint belief or a set of properly weighted coherent draws, together with supported conditional expectations. The same evaluated history should determine related conformance, duration, and exposure quantities.

For normalized weights $\bar w_s$ and coherent histories $L^{(s)}$,

$$
\widehat P(q(L)\in B)
=\sum_{s=1}^{S}\bar w_s
\mathbf 1\{q(L^{(s)})\in B\}.
\tag{16}
$$

Equal weights apply to appropriate posterior draws; importance or SMC samples require their weights. Dependence between MCMC samples affects Monte Carlo error. Conditional expectation may replace inner simulation where it is tractable.

The query's applicable reference, cohort, horizon, and handling of open/censored cases remain explicit. Normative reference factors must not enter the descriptive posterior merely to make the same reference appear conformant.

### 9.2 MAP, representative hypotheses, and omitted mass

The best-scoring history is useful for explanation and initialization. It is not the most representative value of every query. For hybrid models, a joint MAP assignment can even prefer a narrow high-density mode over a broader mode with more probability mass.

A list of feasible alternatives is not a posterior sample merely because its members are coherent. Uniform weighting is an additional assumption. Normalizing exact weights over only the top hypotheses produces the posterior conditional on retaining those hypotheses.

If the retained set is $A$, with known omitted posterior mass $\varepsilon=P(A^c)$, then for an event $B$,

$$
(1-\varepsilon)P(B\mid A)
\le P(B)
\le (1-\varepsilon)P(B\mid A)+\varepsilon.
\tag{17}
$$

For $q\in[a,b]$, the corresponding difference in expectations is at most $\varepsilon(b-a)$. These conclusions require a bound on omitted mass under the same target. A beam size, a low last-hypothesis score, or a stable ranking does not supply that bound.

A provable upper bound on the total unvisited unnormalized mass can support a conservative omitted-mass calculation. Ordinary unweighted search or a finite scenario set cannot.

Candidate events excluded before the model was defined are a different issue: the model has no assigned probability mass for them. Computational truncation bounds cannot repair unknown candidate coverage.

### 9.3 Probability and sensitivity remain separate

Coherent scenarios selected to explore trust or interpretation choices are useful executive evidence. They answer which conclusions are stable across those choices. Their agreement fraction is not a calibrated probability.

Posterior uncertainty within a configuration, differences across model configurations, and unknown physical provenance have different meanings. A report should preserve these distinctions even when they are summarized compactly.

<a id="i10"></a>
## 10. Evidence revision and trust sensitivity

### 10.1 Model and evidence epochs

Each inference result should identify the skeleton, candidate universe, interpretation versions, active factor versions, source parameters, temporal scope, and query/reference version that define it. This is a mathematical reproducibility requirement rather than a particular database design.

Cached messages and sample populations are computational state. They are not additional evidence. Warm-starting a revised calculation can save work, but its target must be reconstructed from the current evidence state.

### 10.2 Replacement, reweighting, and support expansion

For two targets on a common space, where the new posterior is absolutely continuous with respect to the old one,

$$
\pi'(x)\propto
\pi(x)\,\frac{\gamma'(x)}{\gamma(x)}.
\tag{18}
$$

This permits importance reweighting when the ratio is evaluable and overlap is adequate. A factor replacement can simplify the ratio to its new/old likelihood contributions.

Retraction can reopen states previously assigned zero probability. In that case the required absolute continuity fails: old draws cannot represent the newly admissible region. Reweighting cannot recover it. Recalculation, support-expanding proposals, or replay is required.

Even when support matches, weights can collapse. Pareto-smoothed importance sampling provides useful diagnostics and stabilization for heavy-tailed importance ratios.[^15] It does not prove that an unseen mode is absent, and smoothed weights introduce their own finite-sample approximation.

The highest-deliverability default is recomputation of affected bounded posteriors, with cache reuse where valid. Arbitrary exact online retraction should not be an assumed prerequisite.

### 10.3 Manual trust and optional learning

Baseline trust should remain a declared input to inference. A sensitivity family $\Theta$ is not a learned reliability posterior unless a probability model over $\Theta$ is specified.

When a scalar parameter $\lambda$ changes smoothly, under fixed support and conditions allowing differentiation under the integral,

$$
\frac{\partial}{\partial\lambda}\mathbb E_{\pi_\lambda}[q(X)]
=\operatorname{Cov}_{\pi_\lambda}
\left(q(X),\frac{\partial}{\partial\lambda}
\log\gamma_\lambda(X)\right).
\tag{19}
$$

This identity follows by differentiating the normalized expectation. It can identify local trust sensitivity for a fixed query. It does not replace finite changes, dependence-model alternatives, or support revisions.

Automated parameter inference is a separate scientific choice. In particular, NUTS applied to a reliability model with approximate soft counts is not thereby joint posterior sampling of the full object-centric model. A fully Bayesian treatment must specify how parameter and history uncertainty interact, including their shared use across Jobs.

### 10.4 Evidence influence

Source or evidence-group removal should use a defined deletion experiment, accounting for derived descendants and shared measurements. Compare the same query and reference under the resulting targets.

Neither local message magnitude nor trust sensitivity alone gives an evidence-removal effect. None of these quantities gives causal production loss or economic value of information without an additional decision/causal model.

<a id="i11"></a>
## 11. Diagnostics and scientific validation

### 11.1 Report the target and the computation separately

| Dimension | Examples |
|---|---|
| Model scope | Declared cohort, candidate coverage, conditioned boundaries, interpretation versions. |
| Posterior uncertainty | Probability of an occurrence, duration distribution, joint rank uncertainty. |
| Model sensitivity | Changes under trust, association, coverage, or dependence assumptions. |
| Computational uncertainty | Sampling error, numerical integration error, truncation, or variational approximation. |
| Empirical validity | Whether source likelihoods and outcomes have been checked against appropriate physical evidence. |

An exact result under a weakly grounded model is not a physically verified conclusion. A calibrated source likelihood does not guarantee that an approximate inference calculation is accurate.

### 11.2 Numerical evidence appropriate to each method

| Method | Relevant checks | What these checks do not prove |
|---|---|---|
| Exact elimination | Valid support, finite positive normalizer, factor-scale preservation, small-case comparisons. | Correct physical semantics or candidate completeness. |
| Quadrature | Stability under numerical refinement, domain treatment, applicable error control. | General exactness for a high-dimensional nonconjugate block. |
| BP/EP | Belief residuals, convergence status, initialization sensitivity, local exact comparisons. | Global marginal accuracy or discovery of all modes. |
| MCMC | Multiple dispersed chains, mode/association transitions, query-specific ESS and MCSE, rank/folded diagnostics. | A proof that all modes were found. |
| Importance sampling/SMC | Weight concentration, repeated populations, support coverage, normalizer behavior where applicable. | Absence of a missed target region. |
| Hypothesis pruning | A bound on omitted probability mass, if available. | Exhaustiveness from top-hypothesis stability alone. |

For an ergodic chain and a query satisfying the relevant central-limit conditions, a conventional approximation is

$$
\operatorname{MCSE}(\widehat{\mathbb E}[q])
\approx
\sqrt{\frac{\widehat{\operatorname{Var}}_\pi(q)}
{N_{\mathrm{eff},q}}}.
\tag{20}
$$

$N_{\mathrm{eff},q}$ must concern the estimator for that query, not merely a rank-normalized generic diagnostic. Modern rank-normalized/folded $\widehat R$, bulk ESS, and tail ESS address different exploration problems.[^14] Discrete structured states also motivate diagnostics beyond arbitrary numeric encodings of category identifiers.[^27]

For normalized importance weights, $1/\sum_s\bar w_s^2$ measures weight concentration. It is neither MCMC autocorrelation ESS nor OCBF's source-evidence ESS. Equal weights can still accompany complete failure to discover another mode.

Rare conformance violations require particular care. Observing no violation in a dependent or poorly mixed sample does not establish probability zero. A generic binomial confidence calculation cannot be applied without its sampling assumptions.

### 11.3 The appropriate scientific comparison

Algorithms should be compared against the same compiled target and query definitions. Meaningful criteria include marginal error, joint-event error, duration quantile error, semantic violation rate, mode coverage, query MCSE per unit work, and correctness after evidence revision.

Small exact cases can separate numerical errors from modeling errors. Deliberately constructed mathematical cases can expose exactly-one deadlocks, duplicated evidence, shared-source dependence, narrow timing modes, and retractions that expand support. Such cases are synthetic validation instruments and must not be presented as factory observations.

For real-data evaluation, the missing prerequisite is a defensible physical/reference chain for the claims being scored. Simulation-based numerical calibration would validate a method under a simulated model; it would not authenticate the retained factory archive.

No empirical method winner, fixed latency target, or required number of particles/chains is established here. A small belief residual or an unqualified “converged” flag is not an adequate acceptance criterion for executive rankings.

<a id="i12"></a>
## 12. Comparative evaluation and recommended choices

The following is an analytical judgment under the established priorities, not a scored benchmark.

| Approach | Joint capability | Main cost or failure | Recommended role |
|---|---|---|---|
| Finite enumeration | Exact over enumerated full support. | Exponential state count. | Small ambiguity sets and reference calculations. |
| Variable elimination / junction tree | Exact marginals, conditionals, supported joint queries. | Induced width and factor representation. | Preferred default where tractable; exploit the existing exact backend. |
| Conditional Gaussian hybrid elimination | Exact within its admitted Gaussian model. | Mixture growth; non-Gaussian constraints break closure. | Preferred continuous treatment where assumptions actually hold. |
| Cutset conditioning | Exact mixture if enumeration and inner solves are exact. | Number of cutset configurations. | Organizing principle for object/episode ambiguity. |
| Blocked Rao–Blackwellized MCMC | Consistent joint expectations under appropriate conditions. | Mode connectivity, mixing, and exact target evaluation. | Preferred general sampling fallback for bounded retrospective queries. |
| Importance sampling | Weighted joint samples. | Proposal mismatch and weight collapse. | Useful with strong, evaluable proposals and adequate overlap. |
| Static SMC / tempered SMC | Weighted joint populations and possible normalizer estimates. | Intermediate-target choice, resampling loss, rejuvenation cost. | Strong alternative for difficult multimodality or sequential assembly. |
| BP / generalized BP | Approximate local or regional beliefs. | Loops, local consistency, missing joint guarantees. | Broad approximation and proposal guidance. |
| EP / structured VI | Tractable approximating joint or local families. | Projection bias, mode loss, family restrictions. | Selected continuous/large-region approximations with explicit qualification. |
| MAP / constraint optimization | Coherent representative configurations. | Mode mass and uncertainty absent. | Explanation, feasibility, initialization, scenario generation. |
| Dynamic filtering / smoothing | Trajectory inference under a temporal model. | Transition assumptions, path degeneracy, late revisions. | Optional when prospective or continuous monitoring questions justify it. |
| WMC/WMI / circuits | Exact supported query classes after suitable compilation. | Compilation size and integration restrictions. | Specialized repeated-query extension. |
| Lifted inference | Reuse through actual probabilistic symmetry. | Object-specific evidence breaks symmetry. | Opportunistic specialization. |
| Learned / accelerator inference | Faster proposals or supported operations. | Training shift or unchanged approximation bias. | Secondary optimization, not the source of semantic correctness. |

### 12.1 Recommended default

The default should be a **query-oriented hybrid inference system**:

1. Interpret the evidence into one declared joint model and its hard support.
2. Reduce deterministic structure and identify relevant regions with their boundaries.
3. Use exact discrete and admitted continuous inference where feasible.
4. For harder joint questions, use blocked sampling with tractable variables integrated out.
5. Retain BP/EP as explicitly approximate calculations and, where corrected, proposal guidance.
6. Evaluate related business questions from the same joint representation and report the appropriate numerical qualification.

These are algorithmic responsibilities, not delivery phases.

### 12.2 Delivery emphasis

The strongest near-term scientific scope is bounded retrospective Job/Operation cohorts with explicit candidate alternatives, fixed manual trust settings, and a small set of process queries. This reduces the need to solve a factory-wide graph before obtaining useful insight.

If exact integration or adequately explored joint sampling is unavailable, coherent scenarios can still support sensitivity screens. Their outputs must remain scenario conclusions. A failed probability computation should not be disguised as a probability by normalizing a small hand-selected list.

This recommendation does not require automatic reliability learning, a new ontology, a generic continuous-time factory simulator, or universal incremental inference. It reuses the existing structure and inference assets while changing the contract from isolated confidence values to qualified process-query results.

<a id="i13"></a>
## 13. Research questions and boundaries

The major unresolved questions are empirical and structural:

- How large are the ambiguity cutsets and separators after semantically valid episode compression?
- Which temporal likelihoods admit analytic or low-dimensional numerical treatment without changing their meaning?
- Which block moves connect the feasible association/occurrence support efficiently?
- How much do shared-source latent states affect cross-Job priority uncertainty?
- When does static SMC outperform blocked MCMC for this workload at comparable query error?
- Which repeated conformance predicates benefit from compilation enough to justify its structural overhead?
- What posterior information must be retained to support late evidence revision and retrospective questions?
- Which real outcomes can support calibration independently of the reconstruction and reference being evaluated?

The 2026 hybrid-elimination work makes preservation of mixture weights and normalizers particularly relevant.[^9] It does not remove the need to characterize the factory model's non-Gaussian and logical constraints. Likewise, modern circuits, flows, and accelerators broaden the available choices without changing the obligation to identify the target and its support.

The proposed architecture is consequently a combination of established methods chosen for this setting. Its comparative performance, physical calibration, and factory-scale usability remain unvalidated. The [companion proposal](ocbf-inference-proposal.md) gives the narrower architectural recommendation.

<a id="sources"></a>
## Local evidence and sources

### Local evidence basis

The source boundary was checked on 11 September 2026 against the OCBF baseline 1bb372b0fb6869fe27adda434336dca729a622f0. Relevant local sources are:

- [Grounding synthesis](ocbf-grounding-synthesis.md), especially the dated empirical findings, exact-model semantics, and implementation boundary.
- [Grounding proposal](ocbf-grounding-proposal.md), which fixes the semantic target and the retrospective business emphasis.
- [Exact-elimination backend](C:/Users/Ryan/dada/ocbf/ocbf/inference/gtsam_exact.py), including result fields, factor rescaling, cost guards, and marginal normalization.
- [Discrete BP](C:/Users/Ryan/dada/ocbf/ocbf/inference/loopy_bp.py), [Gaussian EP](C:/Users/Ryan/dada/ocbf/ocbf/inference/gabp_ep.py), and [continuous grounding](C:/Users/Ryan/dada/ocbf/ocbf/model/continuous.py).
- [Fusion pipeline](C:/Users/Ryan/dada/ocbf/ocbf/pipeline.py), [graph construction](C:/Users/Ryan/dada/ocbf/ocbf/model/build.py), and [belief-state representation](C:/Users/Ryan/dada/ocbf/ocbf/belief/state.py).
- [Existing inference explanation](C:/Users/Ryan/dada/ocbf/docs/explanation/inference.md) and [declared limitations](C:/Users/Ryan/dada/ocbf/docs/about/limitations.md), evaluated against the source and literature rather than treated as independent guarantees.

The factory observations remain the explicitly dated 9 September inspection findings documented in the grounding synthesis. They have not been refreshed into claims about current factory conditions. No new physical provenance audit, end-to-end fit, or performance benchmark accompanies these documents.

### Research references

Numbered references identify primary papers or specifications. Persistent arXiv identifiers are used without treating upload dates as original publication dates. Recent preprints indicate research directions and conditional results, not established performance in Mammut. Literature coverage is bounded by material available at the 11 September 2026 evaluation.

[^1]: Dechter, R. [*Bucket Elimination: A Unifying Framework for Several Probabilistic Inference*](https://arxiv.org/abs/1302.3572). arXiv:1302.3572. Elimination, conditioning, and structural complexity.

[^2]: Kschischang, F. R., Frey, B. J., and Loeliger, H.-A. (2001). [*Factor Graphs and the Sum-Product Algorithm*](https://doi.org/10.1109/18.910572). IEEE Transactions on Information Theory, 47(2), 498–519.

[^3]: Heskes, T., Albers, K., and Kappen, H. [*Approximate Inference and Constrained Optimization*](https://arxiv.org/abs/1212.2480). arXiv:1212.2480. Bethe/Kikuchi optimization and convergence-oriented alternatives.

[^4]: Wainwright, M., Jaakkola, T. S., and Willsky, A. [*A New Class of Upper Bounds on the Log Partition Function*](https://arxiv.org/abs/1301.0610). arXiv:1301.0610. Tree-reweighted variational bounds.

[^5]: Minka, T. P. [*Expectation Propagation for Approximate Bayesian Inference*](https://arxiv.org/abs/1301.2294). arXiv:1301.2294. Cavity and tilted-distribution approximation.

[^6]: Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. [*Variational Inference: A Review for Statisticians*](https://arxiv.org/abs/1601.00670). arXiv:1601.00670. Variational objectives, families, and approximations.

[^7]: Bidyuk, B., and Dechter, R. [*Cutset Sampling for Bayesian Networks*](https://arxiv.org/abs/1110.2740). arXiv:1110.2740. Combining sampling, exact inference, and Rao–Blackwellization.

[^8]: Gogate, V., and Dechter, R. [*Approximate Inference Algorithms for Hybrid Bayesian Networks with Discrete Constraints*](https://arxiv.org/abs/1207.1385). arXiv:1207.1385. Join-graph propagation and constraint-aware importance proposals.

[^9]: Agrawal, V., and Dellaert, F. [*Variable Elimination in Hybrid Factor Graphs for Discrete-Continuous Inference & Estimation*](https://arxiv.org/abs/2601.00545). arXiv:2601.00545, 2026 preprint. Exact conditional linear Gaussian formulation, normalizers, and hypothesis management; nonlinear/pruned calculations require qualification.

[^10]: Lindsten, F., Helske, J., and Vihola, M. [*Graphical Model Inference: Sequential Monte Carlo Meets Deterministic Approximations*](https://arxiv.org/abs/1901.02374). arXiv:1901.02374. Corrected use of deterministic approximations in joint Monte Carlo inference.

[^11]: Doucet, A., de Freitas, N., Murphy, K., and Russell, S. [*Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks*](https://arxiv.org/abs/1301.3853). arXiv:1301.3853.

[^12]: Lindsten, F., Jordan, M. I., and Schön, T. B. [*Particle Gibbs with Ancestor Sampling*](https://arxiv.org/abs/1401.0604). arXiv:1401.0604.

[^13]: Andrieu, C., and Roberts, G. O. [*The Pseudo-Marginal Approach for Efficient Monte Carlo Computations*](https://arxiv.org/abs/0903.5480). arXiv:0903.5480. Extended-target treatment of random likelihood estimates.

[^14]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. [*Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC*](https://arxiv.org/abs/1903.08008). arXiv:1903.08008. Rank/folded diagnostics and query-dependent ESS.

[^15]: Vehtari, A., Simpson, D., Gelman, A., Yao, Y., and Gabry, J. [*Pareto Smoothed Importance Sampling*](https://arxiv.org/abs/1507.02646). arXiv:1507.02646. Importance-weight stabilization and diagnostics.

[^16]: Miosic, I., and Zuidberg Dos Martires, P. [*Measure Theoretic Weighted Model Integration*](https://arxiv.org/abs/2103.13901). arXiv:2103.13901.

[^17]: Zeng, Z., Yan, F., Morettin, P., Vergari, A., and Van den Broeck, G. [*Hybrid Probabilistic Inference with Logical Constraints: Tractability and Message Passing*](https://arxiv.org/abs/1909.09362). arXiv:1909.09362.

[^18]: Sidheekh, S., and Natarajan, S. [*Building Expressive and Tractable Probabilistic Generative Models: A Review*](https://arxiv.org/abs/2402.00759). arXiv:2402.00759.

[^19]: Wang, B., and Kwiatkowska, M. [*Compositional Probabilistic and Causal Inference Using Tractable Circuit Models*](https://arxiv.org/abs/2304.08278). arXiv:2304.08278. Structural conditions for compositional circuit queries.

[^20]: Jiang, F., Agrawal, V., Buchanan, R., Fallon, M., and Dellaert, F. [*iMHS: An Incremental Multi-Hypothesis Smoother*](https://arxiv.org/abs/2103.13178). arXiv:2103.13178.

[^21]: Huang, Q., Pu, C., Fourie, D., Khosoussi, K., How, J. P., and Leonard, J. J. [*NF-iSAM: Incremental Smoothing and Mapping via Normalizing Flows*](https://arxiv.org/abs/2105.05045). arXiv:2105.05045.

[^22]: Van den Broeck, G., and Darwiche, A. [*On the Complexity and Approximation of Binary Evidence in Lifted Inference*](https://arxiv.org/abs/1311.6591). arXiv:1311.6591.

[^23]: Zhou, G., Dedieu, A., Kumar, N., Lehrach, W., Lázaro-Gredilla, M., Kushagra, S., and George, D. [*PGMax: Factor Graphs for Discrete Probabilistic Graphical Models and Loopy Belief Propagation in JAX*](https://arxiv.org/abs/2202.04110). arXiv:2202.04110.

[^24]: Hoffman, M. D., and Gelman, A. [*The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*](https://arxiv.org/abs/1111.4246). arXiv:1111.4246.

[^25]: Ok, J., Oh, S., Shin, J., and Yi, Y. [*Optimal Inference in Crowdsourced Classification via Belief Propagation*](https://arxiv.org/abs/1602.03619). arXiv:1602.03619. Specific crowdsourcing guarantees and their assumptions.

[^26]: Lindsten, F., Johansen, A. M., Naesseth, C. A., Kirkpatrick, B., Schön, T. B., Aston, J., and Bouchard-Côté, A. [*Divide-and-Conquer with Sequential Monte Carlo*](https://arxiv.org/abs/1406.4993). arXiv:1406.4993.

[^27]: Duttweiler, L., Klus, J., Coull, B. A., Geller, R. J., Claus Henn, B., and Thurston, S. W. [*The Traceplot Thickens: Developing All-Purpose Convergence Diagnostics for Any Markov Chain Monte Carlo Algorithm*](https://arxiv.org/abs/2408.15392). arXiv:2408.15392. Research on diagnostics for structured or difficult sample spaces.

[^28]: Engelmann, N., Linzner, D., and Koeppl, H. [*Continuous-Time Bayesian Networks with Clocks*](https://arxiv.org/abs/2007.00347). arXiv:2007.00347.

[^29]: Nodelman, U., Koller, D., and Shelton, C. R. [*Expectation Propagation for Continuous Time Bayesian Networks*](https://arxiv.org/abs/1207.1401). arXiv:1207.1401.
