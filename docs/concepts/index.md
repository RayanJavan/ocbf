# Concepts

OCBF assesses possible object-centric histories under supplied evidence and assumptions.
For example, reports can leave several plausible endpoint associations for an Operation.
A duration query then accounts for those alternatives instead of choosing one silently.

New term? See the [glossary](../reference/glossary.md).

## From reports to an answer

| Step | You supply | OCBF produces |
|---|---|---|
| [Define the context](semantics.md) | Fixed schema, candidate events and objects, semantic bindings | An immutable semantic context with stable assertion identities. |
| [Prepare evidence](evidence.md) | Versioned report envelopes, a knowledge cutoff, local interpreters | An effective snapshot, interpreted observations, and admission issues. |
| [Resolve trust](parameters.md) | Manual channel values, their origins, priors, and dependence assumptions | Explicit parameter values held fixed during inference. |
| [Compile the model](model.md) | Context, observations, parameters, support, and decoding | A canonical probability model with scientific identity and lineage. |
| [Plan and infer](inference.md) | Query requirements, an engine policy, budgets, and any required RNG | A posterior with declared capabilities and numerical qualifications. |
| [Evaluate questions](queries.md) | A query bundle with population, time scope, and normative reference | Estimates with denominators, unresolved outcomes, and evidence qualifications. |

The [first assessment](../getting-started/quickstart.md) runs this complete sequence with
synthetic evidence and core dependencies. Your application owns source access, persistence,
candidate construction, and presentation; see the
[integration boundary](../how-to/integrate-application.md).

## Choose the question before the calculation

Duration exceptions, whole-Job conformance, descriptive exposure, and priority distributions
can require dependencies across events and objects. Declare those query requirements before
selecting a posterior representation. A set of individual probabilities may not contain
enough information to answer a question about one joint history.

See [capabilities](../reference/capabilities.md) for the supported query/engine combinations
and [inference selection](../how-to/choose-inference.md) for the procedure.

## Core components

- **[Semantic context](semantics.md)** describes the types and candidates in an assessment.
  A `Schema` defines event/object meanings, a `Universe` contains particular candidates,
  and stable assertions identify facts such as an event's association with an Operation.
- **[Evidence and observations](evidence.md)** retain what sources reported and what those
  reports mean. Records carry provenance and revisions; interpreters produce observations
  and admission issues without silently deciding the underlying truth.
- **[Parameters and trust](parameters.md)** supply the observation-channel values, priors,
  and assumptions used by the model. A `ParameterSet` records the resolved values and their
  origins, making alternative manual settings reproducible.
- **[Models and constraints](model.md)** describe possible histories and their relationships.
  Compilation combines context, observations, and parameters into a `CompiledModel`;
  hard constraints exclude histories while evidence distinguishes the remaining ones.
- **[Inference and posteriors](inference.md)** calculate uncertainty over those histories.
  An `InferenceResult` retains a posterior with declared operations, such as individual
  marginals or joint draws, together with its numerical qualifications.
- **[Queries and result meaning](queries.md)** connect histories to process questions.
  A `QueryBundle` supplies projections, populations, and references; an `Estimate` records
  values, denominators, unresolved outcomes, and evidence qualifications.

## Revise and repeat

[Revisions and execution](execution.md) covers what happens when reports, parameters, or
questions change. A corrected report or changed assumption creates a new scientific target;
a changed query reference can reuse a sufficiently capable posterior. `ExecutionSession`
provides bounded reuse across calls, `ExecutionControl` provides cooperative resource
controls, and neutral exports support explicit replay.

Examples refer to a synthetic Operation with one possible start and two alternative end
reports, as used in the [first assessment](../getting-started/quickstart.md).
[How-to guides](../how-to/index.md) contain procedures; the
[reference](../reference/index.md) defines supported combinations and callable contracts.
