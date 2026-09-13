# Proposal: a bitemporal object-centric belief model for OCBF

**Date:** 9 September 2026  
**Status:** Proposed theoretical grounding; not an adopted design or validated model  
**Research basis:** [Enhanced grounding of OCBF: research synthesis](ocbf-grounding-synthesis.md)  
**Scope:** Theory and high-level modeling. No software specification or delivery roadmap.

## Abstract

OCBF should represent beliefs about coherent object-centric histories under a fixed semantic skeleton. Heterogeneous signals enter through revisable observation models that distinguish applicability, reliability, dependence, and time. Process questions then operate on the same uncertain history, enabling interpretable conformance, descriptive attribution, and executive exception assessment. The recommended emphasis is retrospective operation evidence and explicit trust sensitivity, exploiting current Mammut and elastocel assets. Predictive dynamics and causal attribution remain optional extensions requiring additional assumptions.

## 1. Recommendation and empirical basis

**Adopt a joint-history interpretation as the enhanced grounding of OCBF, with bitemporal evidence and manually configured observation likelihoods.** Keep the ontology fixed; let observations, associations, coverage, and interpretations change.

This concentrates semantic work at the signal-to-assertion boundary. A corrected source interpretation can then update several business answers consistently, without rewriting each answer around producer-specific fields. Unknown signal semantics remain uninterpreted until their relationship to an existing assertion is established.

The current assets include a Mammut catalogue, elastocel's implemented structure bridge, retained source envelopes, reconstructed operation anchors, and conformance/attribution vocabulary. The inspected archive also contains repeated reports, shared upstream transformations, provisional identities, and default fields. SEE confidence 0.5 and worker count 1 are not established calibrated or measured quantities. These findings favor observation-aware interpretation over treating every record as a new independent fact.

The [empirical grounding](ocbf-grounding-synthesis.md#s2) and [evidence register](ocbf-grounding-synthesis.md#evidence) distinguish dated archive findings from physical factory truth. No new physical audit or model fit accompanies this proposal.

## 2. Formal objects

Use the synthesis's notation:

| Symbol | Meaning |
|---|---|
| $\mathcal S$ | Fixed object/event types, qualified relationships, attributes, and definitional constraints. |
| $\mathcal U_\kappa$ | Candidate instances under consideration at knowledge time $\kappa$. |
| $L$ | An uncertain OCEL-like history of events, objects, timestamps, E2O/O2O relations, and attributes. |
| $\Omega_{\mathcal S,\mathcal U}$ | Histories admissible under the skeleton and candidate grounding. |
| $D_\kappa$ | Interpreted evidence available by $\kappa$, with explicit revision semantics. |
| $\theta$ | A complete set of prior, observation, dependence, and temporal assumptions. |
| $\mathcal N$ | The applicable normative reference. |
| $q$ | A process question defined over histories. |

Known object identities and object types remain clamped in this grounding. Uncertain report-to-object association uses compatible alternatives and an unresolved outcome; a provisional device identifier is not thereby a verified Job or Operator. New candidate instances do not imply ontology evolution.

An unrepresented event is outside the model's support, not inferred absent. Definitional constraints, descriptive expectations, and normative obligations have separate roles. See [semantic worlds and grounding](ocbf-grounding-synthesis.md#s4).

## 3. Core probability model

Let $P_{0,\theta}$ be a proper descriptive prior supported on $\Omega_{\mathcal S,\mathcal U}$. Define an observation likelihood $\Lambda_{\kappa,\theta}$ and posterior

$$
P_{\kappa,\theta}(d\ell)
=
\frac{\Lambda_{\kappa,\theta}(\ell)P_{0,\theta}(d\ell)}
{\int_{\Omega_{\mathcal S,\mathcal U}}
\Lambda_{\kappa,\theta}(\ell')P_{0,\theta}(d\ell')}.
\tag{P1}
$$

The denominator must be positive and finite. A zero value signals incompatible model/evidence assumptions, not a usable uncertain result.

For dependent reports, let $K$ denote association variables and $H$ shared observation conditions or upstream errors. One admissible likelihood construction is

$$
\Lambda_{\kappa,\theta}(\ell)
=\int
\prod_{g\in\mathcal G_\kappa}
p_\theta(D_{\kappa,g}\mid\ell,k,h,c_\kappa)
\;p_\theta(dk,dh\mid\ell,c_\kappa).
\tag{P2}
$$

Groups are conditionally independent only given the stated history, latent variables, and context $c_\kappa$. Any prior dependence on this context is implicit in $P_{0,\theta}$. Different topics do not establish independence. Informative context and candidate selection must be accounted for in the conditioning model.

This is a probability specification, not a commitment to one inference algorithm. Mixed factor graphs can express the required variable domains and relational dependencies. BP is an inference option; loopy convergence is not an accuracy or calibration certificate. See [the joint model](ocbf-grounding-synthesis.md#s5) and [inference alternatives](ocbf-grounding-synthesis.md#s8).

## 4. Evidence and configurable trust

Every interpreted signal needs three distinct meanings:

1. **Applicability:** which assertion, object, episode, and factory interval it concerns.
2. **Reliability:** how its report is distributed under alternative truths.
3. **Dependence:** which other reports share its observations, transformations, or errors.

For a binary report $Y_s$, an applicable source contract can specify sensitivity $\alpha_s=P(Y_s=1\mid X=1)$ and false-positive rate $f_s=P(Y_s=1\mid X=0)$. Positive evidence then contributes likelihood ratio $\alpha_s/f_s$ under that contract.

Baseline parameters should be indexed by source family and assertion kind, with relevant mode/version context. They are declared assumptions until calibrated. A single generic confidence cannot resolve wrong scope, shared evidence, or unknown semantics.

Retransmission, revision, retraction, interval closure, and genuinely new observation are different evidence actions. Repeated stale state supplies no automatic extra independent support. Absence is negative evidence only under an explicit observation-opportunity model. An upstream posterior score requires understood prior and likelihood semantics; a placeholder score supplies neither. See [trust, dependence, and coverage](ocbf-grounding-synthesis.md#s6).

## 5. Temporal meaning

Use $t$ for factory validity time and $\kappa$ for knowledge time. The central assertion query is $P_{\kappa,\theta}(X_a(t)=x)$. Late evidence can revise belief about a past event without moving that event to ingestion time.

An Operation interval is anchored by its own start and completion events. An unobserved endpoint is not an infinite duration. Tracking loss changes coverage; it does not by itself imply departure or completion. A reported time bound is not automatically a uniform distribution.

Retrospective history beliefs are sufficient for the recommended conceptual scope. Prediction additionally requires transition assumptions. DBNs, irregular-time BNs, CTBNs, and semi-Markov or clock-augmented models offer different choices; synchronized object-centric events require particular care under asynchronous CTBN semantics.

Process executions and variants should retain a declared object/time projection. A variant can change through physical execution, revised evidence, or a changed extraction policy; these are different explanations. See [temporal variants](ocbf-grounding-synthesis.md#s7) and [object-centric executions](ocbf-grounding-synthesis.md#s10).

## 6. Common semantics for business questions

Evaluate a query through the posterior over histories:

$$
P_{\kappa,\theta}(q(L)\in B)
=\int \mathbf 1\{q(\ell)\in B\}
P_{\kappa,\theta}(d\ell).
\tag{P3}
$$

Every result specifies its cohort, period, knowledge cutoff, reference, units, and treatment of incomplete or inapplicable cases.

**Conformance.** Evaluate occurrence, order, multiplicity, duration obligations, or alignment cost against $\mathcal N$. Keep primary descriptive reconstruction separate from the normative reference being assessed. Preserve pending obligations and unknown coverage. Expected counts may use marginals; whole-execution probabilities, overlaps, and rankings generally need joint information.

**Descriptive attribution.** Measure exposure to conditions during a defined gap or interval, including downstream occupancy and observation loss. Overlapping conditions yield non-additive minutes. A gap is not necessarily ready-to-work waiting, and exposure is not causal delay allocation.

**Evidence explanation.** Trace supporting and contradicting observations, interpretation versions, and coverage. Evidence-removal effects require a separate, dependence-aware comparison; message contributions alone do not provide them.

Causal intervention effects require an additional causal model and identifying assumptions. The proposed business meaning is exception assessment and investigation prioritization. See [conformance](ocbf-grounding-synthesis.md#s10) and [attribution](ocbf-grounding-synthesis.md#s11).

## 7. Uncertainty commitments

Report nominal conclusions conditional on a declared $\theta_0$. For a defensible model family $\Theta$, robust support for an event $A$ is characterized by

$$
\left[
\inf_{\theta\in\Theta}P_{\kappa,\theta}(A),
\ \sup_{\theta\in\Theta}P_{\kappa,\theta}(A)
\right].
\tag{P4}
$$

A finite set of evaluated settings provides a scenario range, not an exhaustive bound or calibrated posterior interval. Computational approximation is a separate qualification. Missing physical provenance, unknown semantics, and absent candidates cannot be repaired by arbitrary probability assignments. See [robustness and identifiability](ocbf-grounding-synthesis.md#s9).

The intended exact model obeys semantic support, deterministic-copy invariance, neutral-evidence invariance, and order invariance for the same interpreted evidence state. Reductions should preserve the relevant likelihood or be labeled approximate. Related queries should be projections of the same joint belief. Assumptions and short arguments appear in [the formal properties](ocbf-grounding-synthesis.md#s12).

## 8. Recommended emphasis and boundaries

The highest-value conceptual emphasis is **retrospective Job/Operation exception assessment with traceable evidence and trust sensitivity**. It makes useful questions possible: which exceptions remain supported across reasonable assumptions, which intervals overlap relevant conditions, and which unresolved evidence could change priority.

This emphasis uses the current structural assets while avoiding unsupported claims of measured staffing productivity or recoverable throughput. It does not require making automatic reliability learning, a predictive digital twin, or causal allocation the foundation.

The remaining scientific choices concern admissible candidates, source interpretation contracts, dependence assumptions, coverage, normative references, and the joint information needed by each query. They remain explicit modeling decisions. The existing code provides structural derivation and assertion-level inference assets; it does not already supply the full evidence lifecycle and generic joint-history query model proposed here.

The [implementation boundary](ocbf-grounding-synthesis.md#s13) and [research questions](ocbf-grounding-synthesis.md#s14) document these limits. The proposal's claim is a coherent and interpretable grounding for deliverable insight, conditional on stated assumptions; practical accuracy and scalability remain unvalidated.
