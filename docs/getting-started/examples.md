# Example studies

After the [first assessment](quickstart.md), these runnable studies show larger synthetic
workloads. Run them from the repository root with your environment activated. Each one
writes its results under the ignored `artifacts` directory; pass `--output` to choose
another location.

| Command | Purpose |
|---|---|
| `python -m examples.fixed_parameters` | Finite assessment, named trust, and deterministic revision/replay. |
| `python -m examples.joint_inference` | Same-target exact/sampling comparison and a constrained count reference. |
| `python -m examples.repeated_execution` | Paired fresh/cold/warm trust and query workloads. |
| `python -m examples.resource_validation` | Revision, hybrid, constrained sampling, resources, and warm-start measurements. |

All four run on the core dependencies. `fixed_parameters` also accepts
`--engine gtsam_exact` when the optional [GTSAM backend](installation.md#optional-gtsam-on-windows)
is installed.

## Read the measurements

`repeated_execution` records seeds, environment, repetitions, retained byte budgets,
numerical tolerances, and a median/MAD comparison. Add `--baseline` to record fresh-only
measurements. `resource_validation` checks correctness and resource behavior without a
universal speedup gate.

Inputs are synthetic. The studies validate numerical mechanics, not factory calibration
or universal speedups. Publish a performance claim only with its workload, environment,
inputs, and reproduction command.

## Continue

- [Reuse sessions](../how-to/reuse-sessions.md) and [warm-start sampling](../how-to/warm-start.md)
  use the `joint_inference` inputs.
- [Export and replay](../how-to/export-replay.md) reproduces a `fixed_parameters` bundle.
- [Mammut retrospective](../how-to/mammut-retrospective.md) runs admission against a
  retained archive instead of synthetic inputs.
