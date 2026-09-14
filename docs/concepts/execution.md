# Revisions and execution

OCBF calculations have separate input, model, run, and query identities. These distinguish
a changed scientific question from another execution of the same question.

## What changes a calculation

| Change | Effect |
|---|---|
| A positive endpoint report is replaced with a negative report | A new effective evidence state and model calculation. |
| Manual channel values change from nominal to cautious | A new parameter set and model target. |
| A third candidate end event is added | A changed semantic context, support, and model. |
| The duration reference changes from 30 to 45 minutes | A changed query; an adequate existing posterior can be reused. |
| The random seed changes | Another numerical run of the same target. |

Immutable values keep earlier inputs and results available. Updating a report does not
mutate the posterior previously calculated from its earlier revision.

## Sessions and reusable artifacts

An `ExecutionSession` owns a bounded in-memory artifact store. Repeated calls can reuse
artifacts whose dependencies still match, such as compiled structure or query work.
A changed trust value invalidates the affected numerical contributions even when the
candidate structure stays the same.

Reuse is transparent to scientific meaning: a fresh calculation and a reused calculation
address the same target. Unknown extension dependencies can cause work to run afresh.

Closing the session releases its stored references. An inference result or query result
still held by the application remains available. Session cache accounting describes
retained cache entries, not all process memory.

## Sampling after a revision

A compatible earlier sampled result can supply initialization hints for a revised target.
The sampler validates structure, support, and execution representation before using them.
It still performs warmup and retains an independently initialized comparison chain.

A warm start is not additional evidence. A support change can require fresh initialization
instead of reusing old assignments.

## Controls and incomplete results

`ExecutionControl` provides cooperative cancellation, deadlines, progress callbacks, and
declared workspace checks. A deadline shared across calls spans those calls. A native
operation may be checked only before and after its execution.

A stop before usable output produces a typed failure. When usable partial output exists,
it retains its incomplete execution status and numerical qualifications. Missing query
answers are not filled with zero.

## Replay and retention

Neutral exports retain supported inputs and results, while array artifacts hold supported
posterior arrays. Replay reconstructs declared values and requires any external extension
implementations to be supplied again.

Identical scientific inputs preserve their identities on supported replay. A new inference
execution receives a new run identity. Releasing posterior capabilities with
`summaries_only` preserves result metadata; separately held query answers remain distinct.

Procedures are documented in [session reuse](../how-to/reuse-sessions.md),
[warm-start sampling](../how-to/warm-start.md),
[bound execution](../how-to/bound-execution.md), and
[export/replay](../how-to/export-replay.md).
