# Revisions and execution

Real assessments change: a report is corrected, a trust value is revised, or a different reference
is asked about. OCBF uses identifiers to tell which results still apply after a change, and it can
reuse work that a change did not affect. Below, the objects built on the previous pages change one
at a time.

## Models and runs

Everything that determines the probabilities is part of the model: the context, the evidence, the
parameters, and the factors. The compiled model gets an identifier computed from all of these,
`model_id`. Each inference run also gets its own identifier, `run_id`:

```python
again: InferenceResult = infer(model, requirements=requirements, policy=policy)
print("same model:", again.model_id == result.model_id)
print("new run:", again.run_id != result.run_id)
print(
    "same probabilities:",
    float(again.posterior.marginal(links["short"]).probabilities[1])
    == float(result.posterior.marginal(links["short"]).probabilities[1]),
)
```

Running the block prints:

```text
same model: True
new run: True
same probabilities: True
```

`model_id` says *what* was calculated, and `run_id` says *which run* produced a particular result.
Two exact runs of the same model give the same probabilities.

## What changes a calculation

A change to the evidence or the parameters produces a different model:

```python
corrected: InterpretedEvidence = prepare_evidence(
    (*records, correction), as_of=AT, context=context, interpreters=interpreters  # (1)!
)
corrected_model: CompiledModel = compile_model(replace(spec, evidence=corrected))
cautious_model: CompiledModel = compile_model(replace(spec, parameters=cautious))  # (2)!
print("corrected report, new model:", corrected_model.model_id != model.model_id)
print("cautious trust, new model:", cautious_model.model_id != model.model_id)
```

1.  The negative correction of `long` from [Evidence and observations](evidence.md#revisions-and-knowledge-time).
2.  The cautious parameter set from [Parameters and trust](parameters.md#assumption-sensitivity).

Running the block prints:

```text
corrected report, new model: True
cautious trust, new model: True
```

A change to the question alone needs no new model or inference. The existing posterior is
evaluated again:

```python
for minutes in (10, 90):
    variant: QuerySpec = replace(
        duration,
        name=f"duration over {minutes} min",
        reference={
            "reference_id": "synthetic-reference-v1",
            "threshold_minutes": {"op": minutes},
        },
    )
    variant_answers: QueryResults = evaluate(result, QueryBundle((variant,)))
    print(
        f"{minutes} min: value {variant_answers.estimates[0].value:.1f},",
        "same run:", variant_answers.run_id == result.run_id,
    )
```

Running the block prints:

```text
10 min: value 1.0, same run: True
90 min: value 0.0, same run: True
```

With a 10-minute reference, both closed histories are violations. With a 90-minute reference,
neither is. Both answers come from the same inference run.

| Change | What must be computed again | Shown on |
|---|---|---|
| A corrected, retracted, or late report | New evidence, model, and inference | `corrected_model` above |
| Different trust values or assumptions | New parameter set, model, and inference | `cautious_model` above |
| An added candidate | New context, model, and inference | [Semantic context](semantics.md#candidate-boundaries) |
| A different reference or question | Only the query evaluation | The 10- and 90-minute references above |
| A different random seed | Only the inference run; the model stays the same | [Inference and posteriors](inference.md#posterior-representations) |

A change never modifies results computed earlier: `result` still holds the probabilities computed
from the original report.

## Sessions and reuse

Recomputing everything after every change wastes work. An `ExecutionSession` keeps intermediate
results in memory, up to a size limit you choose, and reuses them when their inputs have not
changed:

```python
from ocbf.api import ExecutionSession

with ExecutionSession(max_cache_bytes=64 * 1024 * 1024, policy=policy) as session:  # (1)!
    session_model: CompiledModel = compile_model(spec, session=session)
    session_result: InferenceResult = infer(
        session_model, requirements=requirements, session=session
    )
    first: QueryResults = evaluate(session_result, bundle, session=session)
    repeated: QueryResults = evaluate(session_result, bundle, session=session)  # (2)!
    print("reused stored work:", session.stats["hits"] > 0)
print(
    "same answers as without a session:",
    [e.value for e in repeated.estimates] == [e.value for e in answers.estimates],
)
```

1.  The size limit in bytes is required. Calls inside the session that pass no policy use the
    session's policy.
2.  Evaluating the same questions on the same result again reuses the stored evaluation.

Running the block prints:

```text
reused stored work: True
same answers as without a session: True
```

A result from a session equals the result computed without one. When an input changes, for
example a trust value, the work that depends on it is computed again. Closing the session
releases what it stored, and results you still hold, such as `repeated`, remain usable.

## Sampling after a revision

After a correction, a sampler for the new model can start from the histories drawn in an earlier
run, instead of from scratch. OCBF first checks that those histories are still possible in the new
model. It then still discards warmup draws, and it runs one additional chain started independently
for comparison.

!!! note "A warm start is not evidence"

    The earlier histories are only a starting point for the sampler, and they add nothing to what
    the reports say. When a change makes new histories possible, for example after a retraction,
    the sampler has to start fresh.

See [warm-start sampling](../how-to/warm-start.md) for the procedure.

## Controls and incomplete results

`ExecutionControl` lets you set a deadline, cancel a calculation, receive progress callbacks, and
check declared memory limits. OCBF checks these controls at fixed points during a calculation, so
an operation running inside a native library is checked only before it starts and after it
finishes.

A stopped calculation reports its status as `cancelled`, `resource-exhausted`, or `failed`; only
a finished one reports `complete`. See [bound execution](../how-to/bound-execution.md) and the
[rules for incomplete execution](../reference/results.md#incomplete-execution).

## Replay and retention

A posterior can be large. Once the questions are answered, you can drop it and keep a summary:

```python
from ocbf.api import summaries_only

summary: InferenceResult = summaries_only(result)  # (1)!
print("capabilities left:", summary.capabilities)
print(
    "identifiers kept:",
    summary.model_id == result.model_id and summary.run_id == result.run_id,
)
print("earlier answer unchanged:", answers.estimates[0].value)
```

1.  Returns a copy of the result without the posterior, keeping its identifiers and the record of
    how it was computed.

Running the block prints:

```text
capabilities left: ()
identifiers kept: True
earlier answer unchanged: 0.5
```

The summary can no longer answer new questions, because it has no capabilities left. It still
records what was calculated and how. Answers evaluated earlier are separate objects and are not
affected.

To make a calculation reproducible, export its inputs. **[Replaying](../reference/glossary.md)**
identical inputs gives the same `model_id` and the same exact answers, with a new `run_id`. Any
extension code that the original calculation used, such as a custom interpreter, must be supplied
again. See [export and replay](../how-to/export-replay.md).

## Summary

`model_id` identifies what is calculated, and `run_id` identifies one run of it. Changed evidence,
trust values, or candidates produce a new model and a new inference; a changed question only
evaluates the existing posterior again. A session reuses work whose inputs have not changed, and
warm starts, controls, and summaries change how work is done or kept, never what the evidence
says.

For procedures, see [reuse sessions](../how-to/reuse-sessions.md),
[warm-start sampling](../how-to/warm-start.md), [bound execution](../how-to/bound-execution.md),
and [export and replay](../how-to/export-replay.md).
