# Reuse analyses with bounded sessions

Use a caller-owned session when comparing manual trust settings or asking several questions
of one posterior. Reuse preserves the declared target, evidence lineage and common draws.
See [validation workloads](../development/validation.md#executable-workloads) for reproducible comparisons.

## Open a session and compare settings

This runnable repository example uses the synthetic inputs from the
[joint-inference guide](joint-inference.md). Supply your own validated `ModelSpec` and
`QueryBundle` for an admitted study.

```python
import numpy as np

from examples.joint_inference import inputs
from ocbf.api import (
    ExecutionSession, compare_settings, compile_model, evaluate, infer, requirements_for,
)
from ocbf.inference.contracts import InferencePolicy

spec, queries, settings, records = inputs()
policy = InferencePolicy(engine="reference_elimination")

with ExecutionSession(max_cache_bytes=64 * 1024 * 1024, policy=policy) as session:
    model = compile_model(spec, session=session)
    result = infer(model, requirements=requirements_for(queries), session=session)
    answers = evaluate(result, queries, session=session)
    repeated = evaluate(result, queries, session=session)
    comparison = compare_settings(
        spec, queries, settings, session=session, rng=np.random.default_rng(113),
    )
    print(dict(session.stats))
    print(session.last_assessment)
```

The byte budget is required and may be zero. It bounds retained cache entries, including
conservative container and array charges. Entries that cannot be frozen or accounted for,
or are larger than the budget, compute without caching. Eviction and session closure release
session references; results already held by callers remain usable. Sessions are intended
for one synchronous caller. `cancel()` may be signalled from another thread.

The optional session policy is a workflow default. An explicit `policy=` on `infer`,
`plan_inference` or `compare_settings` overrides it. With no session policy, the
default engine is `gtsam_exact`. Engines, evaluators, RNGs and controls remain explicit
per-call dependencies.

Every named setting is compiled and validated before the first numerical inference run.
Settings remain separate conditional answers. `comparison.execution["runs"]` records their
model/run identities, allocated setting seeds, chain seeds and inference assessments.
For stochastic comparisons, pass an explicit NumPy generator. All setting streams are
allocated in sorted-name order before any setting runs; sampling allocates all chain seeds
before initialization. Cache hits do not consume randomness.

## Force a fresh comparison

Pass `reuse=False` at every workflow call being compared:

```python
with ExecutionSession(max_cache_bytes=64 * 1024 * 1024, policy=policy) as session:
    fresh_model = compile_model(spec, session=session, reuse=False)
    fresh_result = infer(
        fresh_model, requirements=requirements_for(queries), session=session, reuse=False,
    )
    fresh_answers = evaluate(fresh_result, queries, session=session, reuse=False)
```

This bypasses cache lookup, insertion, dependency-history access and warm-start hints.
Outside a session, calls are uncached. `clear()` removes cache contents while keeping the
session open; `close()` ends its lifetime. Statistics are cumulative for the session.

## Change a reference or revise evidence

Changing normative thresholds requires a new query definition and fresh outcome evaluation.
Keep an adequate posterior and pass the new bundle to `evaluate`; decoded endpoints may be
reused. Pending/satisfied/violated outcomes, denominators, weights and ranks are evaluated
again. Query evaluation never fetches evidence or starts another inference run. If the
posterior cannot answer the scope or capability request, it raises `CapabilityError`.

For corrections, retractions and late arrivals, call `prepare_evidence` with the complete
revision history and the requested `as_of` cutoff, then resolve parameters and compile a new
`ModelSpec`. Pass the same session to those calls. The ordinary revision rules still detect
conflicts and select the effective records. Reuse requires matching record, context and
interpreter dependencies; cached admission issues and lineage are retained. Shared modes,
clocks and associations can make the affected region cross several Jobs.

## Initialize a revised sampling run

Warm starts initialize new chains from retained chain states. They do not reuse old samples
as new posterior draws or add the previous posterior as a likelihood.

```python
from dataclasses import replace
from ocbf.api import warm_start_from
from ocbf.inference.contracts import SamplingConfig

sampling = InferencePolicy(
    engine="blocked", sampling=SamplingConfig(chains=4, warmup=1000, draws=4000),
)
with ExecutionSession(max_cache_bytes=64 * 1024 * 1024, policy=sampling) as session:
    old_model = compile_model(spec, session=session)
    old_result = infer(
        old_model, requirements=requirements_for(queries),
        rng=np.random.default_rng(113), session=session,
    )
    hint = warm_start_from(old_model, old_result)
    new_model = compile_model(replace(spec, parameters=settings["cautious"]), session=session)
    new_result = infer(
        new_model, requirements=requirements_for(queries),
        rng=np.random.default_rng(114), session=session, warm_start=hint,
    )
    print(new_result.diagnostics["warm_start"])
```

States map through semantic keys and must satisfy the new domains, codec and support.
The configured warmup is always completed. At least one chain is independently initialized;
a one-chain configuration uses fresh initialization. Incompatible states are regenerated.
Changed structure, support expansion/contraction, changed codec or unknown support
compatibility regenerates the population. The run records the reason and source run/draw
identities. Importance reweighting is not provided. Assess query-specific MCSE and mixing
even when a run completes successfully.

## Bound work and handle interruption

```python
from ocbf.api import ExecutionControl
from ocbf.errors import OCBFError

control = ExecutionControl(timeout_seconds=30, max_work_bytes=64 * 1024 * 1024)
try:
    result = infer(model, requirements=requirements_for(queries), policy=policy, control=control)
except OCBFError as error:
    print(error.execution)  # status, stage, location, resources and reuse counters
else:
    print(result.execution.status)
```

The deadline spans all calls sharing a control. `max_work_bytes` checks declared workspace
estimates before allocations; it is **not a process RSS limit**. Table, clique, component
and retained-draw budgets remain part of `InferencePolicy` and are checked on cache hits.
`InferencePolicy.max_seconds` also bounds solving. Native calls such as GTSAM and dense
linear algebra are checked at their boundaries and cannot be interrupted mid-call.

Use `control.cancel()` or a `cancelled=` callback for cancellation. A `progress=` callback
receives structured stage, work and allocation information; it may request cancellation.
No root logger configuration, printing, thread pool or scheduling service is installed.

Execution status is `complete`, `cancelled`, `resource-exhausted` or `failed`, separate
from numerical quality and evidence limitations. No runtime stop selects another solver.
Before usable output exists, a typed failure carries `.execution`; unexpected implementation
errors become `ExecutionFailure` with their original cause. Unfinished normalization never
returns an exact posterior. Usable partial chains retain their incomplete status and
numerical qualifications. A partial query bundle lists completed and missing queries in
`answer.execution.details`; missing answers are never replaced with zero.

## Retain or export results

See [export and replay](export-replay.md) for result ownership, JSON/array artifacts, validation,
and summary-only retention. Extension-specific controls and reuse contracts belong in the
[extension guide](../development/extensions.md#add-optional-execution-support).
