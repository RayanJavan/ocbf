# Configuration and errors

The generated API reference owns callable signatures. This page explains the configuration
boundaries and the meaning of failures.

## Inspect defaults from the installed code

```python
from ocbf.inference.contracts import InferencePolicy, SamplingConfig

print(InferencePolicy())
print(SamplingConfig())
```

| Policy | Default |
|---|---|
| Engine | `gtsam_exact` |
| Largest input table | `2**22` states |
| Largest induced clique / requested joint | `2**18` states each |
| Approximate marginal permission | `False` |
| Sampling configuration | `None`; supply it explicitly for blocked execution |
| Hybrid components / continuous dimensions | `1024` / `64` |
| Retained draw bytes | `256 * 1024 * 1024` |
| Solver time budget | `None` |

`SamplingConfig()` supplies four chains, 1,000 warmup sweeps, 4,000 retained sweeps,
a 256-state attempted exact block conditional budget, and 0.1 global refresh probability.
The source reference documents the remaining settings.

## Parameters and controls

Manual resolution uses exact override, one matching family rule, then an explicit default.
Missing and ambiguous assignments fail. Inference never resolves missing trust by fitting.
The [parameter topic](../concepts/parameters.md#resolution-and-provenance) explains what
the resolved values represent; the [trust guide](../how-to/configure-trust.md) shows the calls.

`ExecutionSession` owns a bounded in-memory store. `ExecutionControl` supplies a cooperative
deadline, cancellation callback, progress callback, and workspace estimate budget.
A shared control's deadline spans the calls using it. Native calls are checked at their
boundaries; they cannot be interrupted midway by this mechanism.

## Typed failures

| Category | Meaning |
|---|---|
| `ValidationError` | Invalid references, configuration, units, identities, or declarations. |
| `EvidenceConflict` | Conflicting revisions or effective evidence actions. |
| `CapabilityError` | Unsupported factor, posterior requirement, route, control, or extension operation. |
| `BudgetExceeded` | A declared allocation/work budget cannot be met. |
| `BackendUnavailable` | An optional backend cannot be loaded. |
| `IncompatibleModel` | Contradictory support or zero model mass. |
| `NumericalFailure` | Invalid normalization or other numerical state. |
| `ExecutionCancelled` | Cooperative cancellation before usable output. |
| `ResourceExhausted` | A runtime deadline or workspace budget stops execution before usable output. |
| `ExecutionFailure` | Unexpected controlled-execution failure, preserving its cause. |

Failures can carry located keys and an execution assessment. An evidence admission issue
or unresolved estimate is an explicit result condition, not necessarily an exception.

No failure mode invents a posterior or silently substitutes a new solver target.
See [results](results.md) for partial computation and unresolved quantities.
