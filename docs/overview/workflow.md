# The OCBF workflow

OCBF assesses possible object-centric histories under supplied evidence and assumptions.
For example, reports can leave several plausible endpoint associations for an Operation.
A duration query then accounts for those alternatives instead of choosing one silently.

## From reports to an answer

| Step | You supply | OCBF produces |
|---|---|---|
| Define the context | Fixed schema, candidate events and objects, semantic bindings | An immutable semantic context with stable assertion identities. |
| Prepare evidence | Versioned report envelopes, a knowledge cutoff, local interpreters | An effective snapshot, interpreted observations, and admission issues. |
| Resolve trust | Manual channel values, their origins, priors, and dependence assumptions | Explicit parameter values held fixed during inference. |
| Compile the model | Context, observations, parameters, support, and decoding | A canonical probability model with scientific identity and lineage. |
| Plan and infer | Query requirements, an engine policy, budgets, and any required RNG | A posterior with declared capabilities and numerical qualifications. |
| Evaluate questions | A query bundle with population, time scope, and normative reference | Estimates with denominators, unresolved outcomes, and evidence qualifications. |

The [first assessment](../getting-started/quickstart.md) runs this complete sequence with
synthetic evidence and core dependencies. [Concepts](../concepts/index.md) describes the
corresponding library objects and their behavior through concrete examples.

## Choose the question before the calculation

Duration exceptions, whole-Job conformance, descriptive exposure, and priority distributions
can require dependencies across events and objects. Declare those query requirements before
selecting a posterior representation. A set of individual probabilities may not contain
enough information to answer a question about one joint history.

See [capabilities](../reference/capabilities.md) for the supported query/engine combinations
and [inference selection](../how-to/choose-inference.md) for the procedure.

## Keep application responsibilities explicit

Your application owns source access, credentials, persistence, candidate construction,
producer-specific interpretation, and presentation. OCBF supplies the assessment machinery.
Its results are conditional on the declared model; numerical agreement cannot establish
physical evidence accuracy.

The [integration boundary](../integrations/index.md) describes the required handoffs.
The [Mammut example](../integrations/mammut.md) shows why admission prerequisites matter
even when the numerical workflow is runnable.

## Revise and repeat

A corrected report or changed assumption creates a new scientific target. A changed query
reference can reuse a sufficiently capable posterior. Sessions provide bounded reuse;
exports retain neutral inputs and results for explicit replay.

Continue with [revisions](../how-to/revise-evidence.md),
[sessions](../how-to/repeated-execution.md), or [replay](../how-to/export-replay.md).
