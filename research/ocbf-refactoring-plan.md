# OCBF high-level refactoring plan

**Date:** 11 September 2026.  
**Status:** Proposed execution plan; the stages below have not been implemented or validated by this document.  
**Basis:** [OCBF software architecture](ocbf-software-architecture.md), the [grounding proposal](ocbf-grounding-proposal.md), and the [inference proposal](ocbf-inference-proposal.md).  
**Planning level:** Responsibilities, sequence, deliverables, requirements and completion criteria. Estimates, staffing and deployment commitments are outside this plan.

## Purpose and delivery approach

Evolve the existing library into a reusable path from heterogeneous evidence to coherent object-centric belief and interpretable process questions. Preserve the fixed semantic skeleton and useful numerical assets. Concentrate new work on evidence interpretation, a common model, suitable belief representations and reusable queries.

**Stage 2 is the first business delivery milestone.** It should produce a bounded retrospective Mammut assessment from verified source records, with manual trust, meaningful process results and visible evidence limitations. Completing every proposed module or inference algorithm is not a prerequisite for that milestone.

| Stage | Main outcome | Required starting point |
|---|---|---|
| [1. Establish contracts around existing capabilities](#stage-1) | Stable boundaries, explicit configuration and truthful capability descriptions. | Recorded source/API baseline and known limitations. |
| [2. Deliver one complete retrospective business use case](#stage-2) | A reproducible evidence-to-insight workflow on a bounded real-data scope. | Stage 1 contracts sufficient for that workflow. |
| [3. Generalize the model and joint inference](#stage-3) | Reusable channels and queries, with joint sampling beyond tractable exact cases. | A functioning Stage 2 workflow and numerical references. |
| [4. Improve repeated execution and scale](#stage-4) | Measured improvements to revisions, reuse and resource use. | A correct recomputation baseline for the capability being optimized. |

The stages are outcome boundaries. Within each stage, work can proceed independently where contracts are stable. Compatibility, provenance and scientific validation accompany every stage.

## Contents

1. [Architectural scopes and ownership](#scopes)
2. [Requirements that apply throughout](#requirements)
3. [Stage 1: establish contracts](#stage-1)
4. [Stage 2: deliver a complete business use case](#stage-2)
5. [Stage 3: generalize modeling and joint inference](#stage-3)
6. [Stage 4: improve repeated execution and scale](#stage-4)
7. [Coordination and change discipline](#coordination)
8. [Completion records and separate extensions](#completion)

<a id="scopes"></a>
## 1. Architectural scopes and ownership

Ownership below denotes responsibility boundaries, not assignments to particular people.

| Scope | Owns | Boundary | Main architectural reference |
|---|---|---|---|
| **A. Core contracts and compatibility** | Semantic context, stable identities, public inputs/results, capability and error contracts, legacy adaptation. | Existing schema meanings remain fixed. Execution indices are not durable identity. | [Modules and dependencies](ocbf-software-architecture.md#a3), [data contracts](ocbf-software-architecture.md#a4) |
| **B. Evidence interpretation and trust** | Report lifecycle, provenance, applicability, coverage, dependence declarations, observation channels and resolved manual trust. | Raw extraction is external; unknown fields do not acquire probability semantics automatically. | [Evidence and trust](ocbf-software-architecture.md#a6) |
| **C. Canonical model and compilation** | Variables, factors, admissible support, decoding, dependency records and validated transformations. | Model construction consumes interpreted evidence; normative references remain query inputs. | [Canonical model](ocbf-software-architecture.md#a7) |
| **D. Inference and belief representations** | Engine adapters, capability/cost selection, exact computation, joint sampling, qualified approximations and retained belief representations. | Engines compute a declared target; they do not reinterpret sources or silently fit parameters. | [Inference](ocbf-software-architecture.md#a8), [beliefs](ocbf-software-architecture.md#a9) |
| **E. Process queries and explanations** | Conformance, descriptive exposure, execution projections, priorities and traceable query results. | Query evaluation does not fetch evidence or initiate hidden model changes. Evidence-removal comparisons use explicit orchestration. | [Queries and results](ocbf-software-architecture.md#a9) |
| **F. Integration and execution lifecycle** | Facade composition, Mammut/elastocel connections, repeatable runs, result export, recomputation and later caching. | Factory-specific interpretation/bindings stay in integration code; OCBF remains independent of factory storage and screens. | [Public API](ocbf-software-architecture.md#a5), [revisions](ocbf-software-architecture.md#a10), [runtime](ocbf-software-architecture.md#a11) |

Scopes B and C share the observation-channel boundary: B specifies the report's probability semantics and parameter needs; C owns construction of canonical factors under that contract. This boundary needs one explicit interface and owner for each channel, rather than duplicated likelihood logic.

Elastocel continues to derive structure from the catalogue and available log/candidates. The Mammut integration passes OCBF-owned schema, constraint and universe values into the library. Source-specific producer fields and business reference bindings remain outside OCBF's generic core.

<a id="requirements"></a>
## 2. Requirements that apply throughout

These requirements define proposed acceptance criteria for the work. The [architecture's scientific invariants](ocbf-software-architecture.md#a1) remain authoritative.

| ID | Requirement | Evidence expected during delivery |
|---|---|---|
| **R1 — Fixed semantics** | Preserve object/event types, qualified relationships and assertion meanings. Candidate instances and observation interpretations may change explicitly. | Mapping to existing semantic types; no producer-specific fields in generic process queries. |
| **R2 — Real evidence** | Verify origin and distinguish observed, derived, default, provisional and synthetic content. Preserve uninterpreted records without invented likelihoods. | Source/version and provenance register; explicit admission and exclusion reasons. |
| **R3 — Observation meaning** | Separate applicability, reliability, dependence and coverage. Copies and shared transformations do not automatically count as independent evidence; silence needs an observation-opportunity model. | Channel contracts and focused duplicate, scope and coverage checks. |
| **R4 — Time and revision** | Separate factory validity time from knowledge time. Define replacement, retraction, late arrival, closure and censoring for supported evidence. | Replayable snapshots and revision examples; explicit unsupported-action outcomes. |
| **R5 — Explicit model** | Preserve hard support and relevant normalization constants; separate descriptive assumptions from normative references. All routes identify their target or approximation. | Small reference models, decoding checks and a model/compilation manifest. |
| **R6 — Fixed trust per run** | Accept resolved manual parameters; keep fitting separate. Returned parameters and beliefs refer to the same model/evidence epoch. | Configuration and epoch checks; repeatable nominal and alternative settings. |
| **R7 — Honest query capabilities** | Use joint information where required. Distinguish posterior uncertainty, numerical error, sensitivity ranges, coverage limitations and scenario-only results. | Capability rejection cases and query results carrying the appropriate qualifications. |
| **R8 — Explicit failure** | Zero/undefined normalization, unsupported quantities and absent support do not become uniform belief, default 0.5 probabilities or zero durations. | Located failure/unresolved results and targeted numerical checks. |
| **R9 — Compatibility and modularity** | Preserve documented entry points or provide deliberate migration behavior. Keep optional backends and external applications outside core contracts. | Compatibility record, import checks and focused adapter tests. |
| **R10 — Reproducibility and validation** | Identify evidence, interpretation, model, parameters, query and computation. Validate numerical consistency separately from physical accuracy. | Reproducible run records, same-target comparisons and clearly labeled synthetic fixtures. |

Deliverability permits a narrow supported scope and explicit unresolved results. It does not permit hiding a changed target, invented evidence or an unsupported probability claim.

<a id="stage-1"></a>
## 3. Stage 1 — Establish contracts around existing capabilities

**Outcome:** Existing capabilities can participate in a consistent library workflow without overstating what they provide.

**Primary scopes:** A, with the contract portions of B–F.

### Boundary

Include the shared contracts, minimal orchestration, explicit parameter input and adapters needed for the first use case. Preserve existing schema/assertion assets and reuse current numerical kernels.

A full evidence catalogue, general hybrid compiler, joint sampler, broad query language and factory UI are outside this stage. Avoid reorganizing every file before an executable use case requires the new boundary.

### Work sequence

1. **Record the starting behavior.** Refresh the source/API baseline and relevant checks. Classify existing behavior as reusable, incomplete or needing a scientific correction. Use the [architecture's compatibility assessment](ocbf-software-architecture.md#a13) as the starting inventory.
2. **Define the smallest shared contracts.** Establish semantic context, evidence/model/parameter identity, interpreted-observation shape, engine capabilities, query requirements and neutral result/error forms. Follow the architecture's dependency direction.
3. **Make manual configuration explicit.** Provide an inference path that receives resolved parameters and keeps them fixed. Keep any fit-and-infer behavior explicit and ensure reported beliefs match the reported parameter epoch.
4. **Adapt existing engines and sources.** Wrap the current BP, EP, exact and claim-based entry points behind their actual capabilities. Missing provenance or joint conditionals remain declared gaps.
5. **Establish compatibility and validation seams.** Protect meaningful public behavior, check core imports without optional backends, and create small reference cases for subsequent model and query work.

### Deliverables

| ID | Deliverable | Minimum content |
|---|---|---|
| **1A** | Baseline and compatibility record | Relevant public entry points, current numerical behavior, known limitations and intended treatment. |
| **1B** | Shared contracts and composition boundary | Inputs/results, identities, error/capability vocabulary and dependency ownership. |
| **1C** | Explicit-parameter inference path and existing-engine adapters | Fixed parameter handling, model/run metadata and truthful capability descriptions. |
| **1D** | Focused reference and compatibility checks | Evidence that the new boundaries preserve admitted behavior and expose unsupported capabilities. |

### Completion criteria

- [ ] Existing semantic types remain usable through the new contracts.
- [ ] A supported example runs through an adapter with identifiable inputs, fixed parameters and a qualified result.
- [ ] Manual parameter injection is demonstrated; disabling fitting alone is not treated as that capability.
- [ ] BP marginals and the existing exact marginal/MPE output are not presented as generic joint-history support.
- [ ] Existing callers have documented continuity or an explicit migration path.
- [ ] Known model defects are recorded for correction before affected new functionality is admitted.

Stage 1 establishes boundaries and behavior. It does not certify legacy modeling assumptions merely because they reproduce previous outputs.

<a id="stage-2"></a>
## 4. Stage 2 — Deliver one complete retrospective business use case

**Outcome:** A bounded Mammut assessment links verified reports to useful, uncertainty-aware process results under manual trust.

**Primary scopes:** A–F, implementing only what the selected use case needs.

### Boundary

Choose a retrospective cohort, factory window, knowledge cutoff, supported observation families and a small query bundle. Use tractable inference on that declared scope. Deliver results through a readable report or an existing application surface.

Broad factory coverage, a redesigned dashboard, streaming infrastructure, general joint sampling and automatic trust learning are outside this stage.

### Work sequence

1. **Select and document the use case.** Record the Operation/Job cohort, relevant relationships, reference provenance, expected business question and evidence limitations. Prefer a scope containing meaningful heterogeneous observations whose interpretations can be verified.
2. **Verify the source basis.** Match retained data to its producer version/contract and document lineage. For current producer behavior, consult the active `amirhosseinniknam/factory-monitor` repository; do not assume a local checkout is current. Historical data still needs historically applicable interpretation.
3. **Implement the minimum evidence path.** Materialize effective revisions, interpret admitted reports, retain unresolved alternatives, declare coverage/dependence, and resolve manual trust. Unknown/default fields remain outside unjustified likelihoods.
4. **Build the bounded canonical model.** Compile only the required factor families, admissible support, association alternatives and temporal semantics. Record any cohort conditioning and exterior dependencies.
5. **Supply the required inference capability.** Prefer exact computation where tractable. Add the joint conditional or finite joint representation needed for the chosen queries; the existing marginal/MPE wrapper is not sufficient by itself. Preserve relevant constants and hard support.
6. **Evaluate and explain the query bundle.** Provide at least one informative conformance assessment, plus a descriptive exposure query where evidence and admitted inference support it. Add supporting/contradicting evidence, unresolved coverage and named manual-trust comparisons.
7. **Demonstrate correction and repeatability.** Replay a corrected or retracted observation through a new snapshot and full recomputation. Reproduce the original analysis from its recorded inputs.

### Deliverables

| ID | Deliverable | Minimum content |
|---|---|---|
| **2A** | Use-case and source register | Cohort/window/cutoff, source lineage/version, semantic bindings, query definitions and reference provenance. |
| **2B** | End-to-end evidence/model workflow | Admission report, effective snapshot, resolved trust, canonical model and required backend conversion. |
| **2C** | Business result artifact | Conformance results, supported exposure findings, evidence explanations, trust comparisons and explicit unavailable quantities. |
| **2D** | Replay and validation record | Reproducible nominal run, correction/retraction example, numerical reference checks and limitations. |

### Completion criteria

- [ ] The delivered findings trace to real retained reports; synthetic numerical fixtures are labeled separately.
- [ ] At least one meaningful process question is assessable from the admitted evidence and computation. An entirely unresolved result does not satisfy the business milestone.
- [ ] Heterogeneous observations contribute under explicit contracts; duplicated transformations are not advertised as independent corroboration.
- [ ] Joint predicates and interval calculations use appropriate joint information. Regional conditioning is visible.
- [ ] The result states applicability, denominator, units, incomplete-case treatment, computation status and evidence limitations.
- [ ] Nominal and alternative trust settings can be compared without silently changing the reference or interpretation.
- [ ] A revision produces a new identifiable calculation, with no duplicate contribution from the replaced evidence.
- [ ] Small exact/reference cases validate the supported numerical and query behavior; factory accuracy remains separately qualified.

If the proposed exposure quantity requires unsupported continuous integration or unverified readiness assumptions, report that limitation and choose another supported quantity for the first delivery. Do not manufacture time distributions or causal delay shares to complete the view. Record exposure as incomplete until its requirements are met.

The model/query interfaces established here must remain reusable. A producer-specific shortcut that bypasses them to compute a dashboard metric is not the intended Stage 2 deliverable.

<a id="stage-3"></a>
## 5. Stage 3 — Generalize the model and joint inference

**Outcome:** Additional sources and process questions reuse the same contracts, while principled joint inference extends the range of tractable analyses.

**Primary scopes:** B–E, supported by A and F.

### Boundary

Generalize the patterns demonstrated in Stage 2. Add observation families and factor capabilities in response to documented use cases. Implement blocked joint sampling for ambiguity that cannot be handled economically by the admitted exact route.

A universal probabilistic programming language, automatic source discovery, every PGM algorithm and a learned factory model are outside this stage.

### Work sequence

1. **Extract reusable interpretation/channel components.** Separate producer parsing from applicability, observation distributions, dependence and coverage. Onboard an additional source or producer representation through these extension points.
2. **Extend canonical factors and reductions.** Support the additional admitted discrete/continuous families, constraints and decoding needed by the next questions. Validate every new lowering against the same target.
3. **Add blocked joint inference.** Combine tractable elimination with coordinated sampling moves. Preserve proposal corrections, feasible support, continuous-variable treatment and query-relevant diagnostics.
4. **Extend process queries through shared primitives.** Reuse execution projections, obligations and interval expressions. Preserve shared uncertainty across Jobs for ranking or other comparative questions.
5. **Make solver selection explicit.** Select according to factor capabilities, induced complexity, query needs and recorded budgets. Preserve BP/EP as qualified approximate options.
6. **Compare methods and reuse.** Evaluate exact and sampling routes on the same small targets, then exercise the expanded real-data scope with its limitations visible.

### Deliverables

| ID | Deliverable | Minimum content |
|---|---|---|
| **3A** | Reusable observation and factor extensions | Versioned channel contracts, parameter needs, provenance/dependence semantics and supported operations. |
| **3B** | Joint inference capability | Blocked sampler, admitted elimination integration, joint result representation and method diagnostics. |
| **3C** | Expanded reusable query set and routing | Additional process/comparative queries with explicit scope and capability requirements. |
| **3D** | Generalization and numerical comparison record | Source onboarding evidence, unchanged reusable query logic where applicable, and same-target method comparisons. |

### Completion criteria

- [ ] An additional source family or representation is handled through interpretation/channel extensions without embedding its fields in generic queries.
- [ ] New factor operations preserve support, decoding and required normalizing constants.
- [ ] Joint sampling handles constrained moves and tests materially different modes; marginal agreement alone is insufficient.
- [ ] Exact-reference comparisons assess the actual query quantities, with Monte Carlo error reported.
- [ ] At least one query requiring joint information is demonstrated under the expanded inference path.
- [ ] Shared source/resource uncertainty remains shared in multi-Operation or multi-Job comparisons.
- [ ] Unsupported factor/query combinations are rejected or qualified under a declared policy.

Static SMC or other alternative engines can be evaluated if a demonstrated use case justifies them. They are not prerequisites for Stage 3 completion.

<a id="stage-4"></a>
## 6. Stage 4 — Improve repeated execution and scale

**Outcome:** Repeated analysis and evidence updates become more efficient while retaining the validated model and query semantics.

**Primary scopes:** F and D, with dependency support from B, C and E.

### Boundary

Optimize measured bottlenecks on declared workloads. Candidate targets include repeated trust settings, repeated query evaluation and corrections affecting a bounded portion of the model.

Distributed execution, continuous ingestion services, predictive filtering and production availability targets are separate scope decisions. Basic failure handling, provenance and revision correctness already belong to earlier stages.

### Work sequence

1. **Measure the baseline.** Select representative repeat/update workloads and record time, memory, numerical quality and recomputation behavior. Define resource goals from that evidence.
2. **Introduce dependency-aware reuse.** Distinguish reusable structure from invalidated numeric factors, messages, conditionals and query projections.
3. **Optimize revisions.** Add appropriate warm starts and affected-region recomputation. Retractions and expanded support require regeneration when old representations cannot cover the new target.
4. **Improve resource and result handling.** Add measured cost controls, cancellation/retention policies and portable result manifests appropriate to the supported workflow.
5. **Compare against fresh computation.** Validate reuse under corrections, trust changes, reordered arrivals and shared-dependency changes, then document the observed performance benefit.

### Deliverables

| ID | Deliverable | Minimum content |
|---|---|---|
| **4A** | Workload and performance record | Representative workloads, baseline costs, numerical criteria and selected improvement goals. |
| **4B** | Validated reuse/update mechanisms | Cache dependency rules, invalidation behavior, warm-start semantics and regeneration cases. |
| **4C** | Repeatable operational library workflow | Resource controls, retention/export behavior and clear completion/failure status. |
| **4D** | Reuse-versus-recomputation evidence | Numerical equivalence or assessed stochastic agreement, measured costs and supported limits. |

### Completion criteria

- [ ] Optimization is justified by measured workloads and shows a documented benefit within the chosen numerical criteria.
- [ ] Reuse agrees with fresh exact computation or appropriate stochastic comparisons for the supported cases.
- [ ] Changed parameters, interpretations, associations and shared source modes invalidate dependent artifacts correctly.
- [ ] Support expansion is not handled by merely reweighting samples that exclude the newly possible states.
- [ ] Cancellation, resource exhaustion and incomplete results have explicit status.
- [ ] Exported results preserve their model/query identities, qualifications and interpretation.

Stage 4 can target selected expensive capabilities independently once their recomputation baseline is established. It should not postpone the Stage 2 business delivery.

<a id="coordination"></a>
## 7. Coordination and change discipline

### Dependency and handoff rules

Core contracts precede dependent implementations. Once the Stage 2 observation, model and query contracts are stable, source integration, admitted engine work and query evaluation can proceed independently against those interfaces.

The main handoff chain is:

**Verified reports → effective evidence → interpreted observations and trust → canonical model → qualified belief → process results.**

Each handoff names its producer, consumer, schema/version, validation responsibilities and unsupported outcomes. The facade composes those handoffs; it does not become the owner of every domain rule.

### Keep changes reviewable

Separate module moves and interface adaptation from changes to probability semantics. For a behavioral change, record the previous behavior, intended new meaning, affected consumers and supporting checks.

When introducing a new route, compare it with the appropriate reference before replacing an existing path. Preserve the legacy route or a documented migration option until affected callers have an adequate replacement. Compatibility does not require retaining known scientific defects in the new model.

Use meaningful tests at the changing boundary. Broaden validation when a new factor family, engine capability, revision rule or query dependency creates a new concern. Avoid tests that only reproduce the implementation's current output without checking its meaning.

### Decisions to record as work proceeds

| Decision | When it must be resolved | Required record |
|---|---|---|
| Existing API continuity and intended corrections | Stage 1, before changing affected behavior | Compatibility decision and affected consumers. |
| First cohort, source contracts and business questions | Early Stage 2, before likelihood construction | Use-case/source register with evidence and reference provenance. |
| Trust, dependence, coverage and temporal assumptions | Before each model is admitted | Resolved model configuration, origins and supported alternatives. |
| Scope boundaries and required joint information | Before inference selection | Query requirements and conditioning statement. |
| Numerical/resource criteria | Before accepting the relevant engine/workload result | Criteria appropriate to the query and method; no arbitrary universal confidence threshold. |
| Reuse and invalidation policy | Before enabling an optimization | Dependency rule and fresh-computation comparison. |

These are implementation decisions to document, not requests to invent missing facts or obtain a new approval for every routine choice.

<a id="completion"></a>
## 8. Completion records and separate extensions

Use the following compact record at the end of each stage:

| Field | Required content |
|---|---|
| Delivered | Completed deliverable IDs, supported capabilities and artifact/API references. |
| Boundaries | Supported source/factor/query families, cohort/time assumptions and explicit exclusions. |
| Validation | Checks performed, reference basis, observed results and remaining uncertainty. |
| Compatibility | Preserved behavior, deliberate changes and migration instructions. |
| Evidence status | Real versus synthetic inputs, provenance limitations and interpretation versions. |
| Remaining work | Unmet completion criteria and the next dependent capability. |

A stage is complete when its stated outcome and completion criteria are supported by evidence. Partial functionality remains identified as partial. A functioning numerical demonstration does not, by itself, establish the Stage 2 business milestone or factory calibration.

Automatic reliability learning, predictive temporal models, causal attribution, learned proposals and distributed execution remain separate extensions. Each needs a use case, explicit assumptions and its own validation basis before joining this refactoring's required scope.

For detailed interfaces and mathematical commitments, follow the [software architecture](ocbf-software-architecture.md); for scientific justification, follow the [grounding synthesis](ocbf-grounding-synthesis.md) and [inference synthesis](ocbf-inference-synthesis.md). This plan organizes their delivery while keeping the first useful, evidence-backed business result early.
