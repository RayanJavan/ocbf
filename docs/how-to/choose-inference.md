# Choose inference

Define queries before choosing an engine so the request captures their joint scope.
The [inference topic](../concepts/inference.md) describes the posterior representations
that these requirements distinguish.

```python
import numpy as np
from examples.joint_inference import inputs
from ocbf.api import compile_model, evaluate, infer, plan_inference, requirements_for
from ocbf.inference.contracts import InferencePolicy, SamplingConfig

spec, queries, settings, records = inputs()
model = compile_model(spec)
requirements = requirements_for(queries)
policy = InferencePolicy(engine="auto", sampling=SamplingConfig())
plan = plan_inference(model, requirements=requirements, policy=policy)
print(plan.engine, plan.details)

rng = np.random.default_rng(20260912)
result = infer(model, requirements=requirements, policy=policy, rng=rng)
answers = evaluate(result, queries, rng=rng)
```

## Make selection explicit

`InferencePolicy()` defaults to `gtsam_exact`. A named route fails when it cannot perform
the requested calculation. The opt-in `auto` policy assesses `reference_elimination`,
`reference_hybrid`, and `blocked`, in that order.

Auto selection does not authorize BP, EP, or a change to the scientific target. A failure
during solving does not silently start another engine.

Use [the capability table](../reference/capabilities.md#inference-routes) to select a route.
Pass a NumPy generator for stochastic execution or additional exact joint draws. Blocked
sampling also requires a `SamplingConfig`.

## Interpret the plan

The plan records selected and rejected routes, admitted capabilities, factor/table costs,
and relevant sampled/eliminated scopes. Budgets bound tables, induced cliques, requested
joints, hybrid components, real dimensions, and retained draws.

A successful preflight is not a convergence guarantee, proof of a positive normalizer,
or proof that an optional native backend will load. See
[errors and configuration](../reference/configuration.md) for the actual defaults and
[execution controls](bound-execution.md).
