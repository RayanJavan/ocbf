# Warm-start sampling after a revision

Warm starts initialize new chains from retained chain states. They do not reuse old samples
as new posterior draws or add the previous posterior as a likelihood.

## Initialize a revised sampling run

This runnable repository example uses the synthetic inputs from the
[joint-inference guide](joint-inference.md) and changes their manual trust setting.

```python
from dataclasses import replace

import numpy as np

from examples.joint_inference import inputs
from ocbf.api import (
    ExecutionSession, compile_model, infer, requirements_for, warm_start_from,
)
from ocbf.inference.contracts import InferencePolicy, SamplingConfig

spec, queries, settings, records = inputs()
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

## Check what the warm start reused

States map through semantic keys and must satisfy the new domains, codec and support.
The configured warmup is always completed. At least one chain is independently initialized;
a one-chain configuration uses fresh initialization. Incompatible states are regenerated.
Changed structure, support expansion/contraction, changed codec or unknown support
compatibility regenerates the population. The run records the reason and source run/draw
identities. Importance reweighting is not provided. Assess query-specific MCSE and mixing
even when a run completes successfully.

See [revise evidence](revise-evidence.md) for building the revised inputs and
[read diagnostics](read-diagnostics.md) for assessing the new chains.
