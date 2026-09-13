# Concepts

OCBF represents uncertain process histories as explicit library objects: a semantic
context, versioned evidence, parameter values, a model, and query results. Each topic
below describes those objects, their relationships, and their behavior in a calculation.

New term? See the [glossary](../reference/glossary.md).

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

## Execution support

[Revisions and execution](execution.md) covers what happens when reports, parameters, or
questions change. `ExecutionSession` provides bounded reuse across calls, `ExecutionControl`
provides cooperative resource controls, and neutral exports support explicit replay.

Examples refer to a synthetic Operation with one possible start and two alternative end
reports, as used in the [first assessment](../getting-started/quickstart.md).
The [workflow tour](../overview/workflow.md) describes the complete sequence.
[How-to guides](../how-to/index.md) contain procedures; the
[reference](../reference/index.md) defines supported combinations and callable contracts.
