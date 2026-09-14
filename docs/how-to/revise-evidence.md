# Revise evidence

Append a revision record, select a knowledge cutoff, and recompute the effective model.
Do not overwrite the only copy of the original report.

## Replace a report

This example changes one synthetic endpoint report:

```python
from dataclasses import replace
from examples.fixed_parameters import AT, example_inputs
from ocbf.evidence import EvidenceAction

spec, queries, settings, records = example_inputs()
correction = replace(
    records[-1],
    revision_id="long:2",
    previous_revision="long:1",
    action=EvidenceAction.REPLACE,
    payload={"event": "long", "reported": False},
    known_at=AT,
)
updated_spec, updated_queries, updated_settings, history = example_inputs(
    (*records, correction), as_of=AT
)
```

Compile and infer `updated_spec` using the same explicit workflow as the quickstart.
The replaced report no longer contributes independently.

## Choose the action and clock deliberately

| Action | Meaning |
|---|---|
| `ASSERT` | Introduce a report chain with no predecessor. |
| `REPLACE` | Supply a successor revision with a predecessor identity. |
| `RETRACT` | Withdraw the report through an explicit successor action. |
| `CLOSE` | Close an effective record under the materialization contract. |

The default `explicit-chain-v1` policy uses knowledge time and explicit predecessor links.
Invalid identities and conflicting chains fail; arrival order is not an implicit conflict
resolution rule. `static-v1` supports explicitly static input with unavailable receipt time;
it must not be advertised as an “as known then” reconstruction.

Retractions may expand feasible support. Old samples cannot acquire newly possible states
by reweighting. Use fresh inference or a validated
[warm-start hint](warm-start.md).

The complete nominal/correction/retraction command is
`python -m examples.fixed_parameters`. See [export and replay](export-replay.md) for
recording the inputs and [revisions and execution](../concepts/execution.md) for identities.
