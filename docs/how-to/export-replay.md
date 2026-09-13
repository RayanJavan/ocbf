# Export and replay

Retain scientific inputs and qualified results in neutral formats. Standard OCEL files
and executable Python objects are different interchange products.

## Replay a finite study

```bash
python -m examples.fixed_parameters
```

```python
from examples._shared.study import reproduce
answers = reproduce("artifacts/fixed-parameters/nominal/inputs.json")
```

The example exports context, evidence, parameters, queries, and policy before inference.
It verifies that replay preserves scientific identities and deterministic answers.
Run identity changes for each execution.

`run_study` and `reproduce` are reusable checkout-example helpers. They are not part
of the installed library's public facade.

## Choose an interchange format

| API | Use |
|---|---|
| `dumps` / `loads`, `write_bundle` / `read_bundle` | Allowlisted, versioned JSON inputs and qualified results. |
| `write_artifact` / `read_artifact` | Typed JSON manifest plus non-executable NumPy array payloads. |
| `summaries_only` | Detached metadata-only inference result after evaluating queries. |

For an existing `result` and `answers`:

```python
from pathlib import Path
from uuid import uuid4
from ocbf.api import summaries_only
from ocbf.io import read_artifact, write_artifact

destination = Path("artifacts") / ("analysis-" + uuid4().hex)
write_artifact(destination, {"inference": result, "queries": answers})
restored = read_artifact(destination, max_bytes=256 * 1024 * 1024)
metadata = summaries_only(result)
```

Array loading validates format, paths, shapes, dtypes, checksums, and payload budgets.
Payload bytes do not include all Python container overhead. Exports do not overwrite an
existing destination. Interrupted `.partial` directories remain inspectable and are not
published results.

## Retain execution requirements

A custom extension must be supplied explicitly on replay; a bundle does not import
implementations named by input data or serialize executable closures. Retain the named
versions, configuration, seeds, and numerical environment.

Seeded stochastic replay is environment-dependent. Use numerical comparisons across
different backends or environments rather than promising universal bitwise equality.

Releasing a result's posterior does not release other caller references or session entries.
See [session ownership](repeated-execution.md). Exported posterior capabilities are
documented with [result meanings](../reference/results.md#interchange-and-retention).
