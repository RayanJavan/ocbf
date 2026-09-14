# Bound execution and handle interruption

Give a run a deadline, a declared workspace budget, or a cancellation path, and read the
execution status it returns.

## Bound work and handle interruption

This runnable repository example uses the synthetic inputs from the
[joint-inference guide](joint-inference.md).

```python
from examples.joint_inference import inputs
from ocbf.api import ExecutionControl, compile_model, infer, requirements_for
from ocbf.errors import OCBFError
from ocbf.inference.contracts import InferencePolicy

spec, queries, settings, records = inputs()
policy = InferencePolicy(engine="reference_elimination")
model = compile_model(spec)

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

## Cancel or observe progress

Use `control.cancel()` or a `cancelled=` callback for cancellation. A `progress=` callback
receives structured stage, work and allocation information; it may request cancellation.
No root logger configuration, printing, thread pool or scheduling service is installed.

## Read the execution status

Execution status is `complete`, `cancelled`, `resource-exhausted` or `failed`, separate
from numerical quality and evidence limitations. No runtime stop selects another solver.
Before usable output exists, a typed failure carries `.execution`; unexpected implementation
errors become `ExecutionFailure` with their original cause. Unfinished normalization never
returns an exact posterior. Usable partial chains retain their incomplete status and
numerical qualifications. A partial query bundle lists completed and missing queries in
`answer.execution.details`; missing answers are never replaced with zero.

See [configuration and errors](../reference/configuration.md) for failure types and
[results](../reference/results.md#incomplete-execution) for incomplete output.
