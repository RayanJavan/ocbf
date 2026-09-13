# Interpret evidence

Use an interpreter to bind understood report content to stable semantic assertions.
Keep extraction and producer-specific parsing in application or integration code.

## Supply a versioned contract

Implement `EvidenceInterpreter` from `ocbf.sources.contracts`. It declares `name`,
`version`, and `interpret(record, context)`, returning observations and admission issues.

An observation identifies its source, assertion family, observation channel, scope, reported
value, evidence revisions, interpretation identity, and information group. Its channel
determines how alternative truths could generate that report.

The [quickstart interpreter](../getting-started/quickstart.md#3-interpret-versioned-reports)
is a complete synthetic example.

## Materialize and inspect

```python
from examples.fixed_parameters import AT, SyntheticInterpreter, example_inputs
from ocbf.api import prepare_evidence

spec, queries, settings, records = example_inputs()
evidence = prepare_evidence(
    records,
    as_of=AT,
    context=spec.context,
    interpreters={("fixture", "1"): SyntheticInterpreter()},
)
print(evidence.issues)
print(evidence.observations)
```

An unknown producer version or uninterpreted field does not acquire a default likelihood.
Preserve the admission issue. A source profile, confidence field, or coverage enum does not
by itself establish observation meaning.

## Declare information ownership

Known copies must identify one information contribution. Repeated transport records and
effective revisions are handled separately from distinct measurements. Conflicting meanings
within one contribution require an explicit joint model, not a cluster label.

Use an explicit shared mode or other supported factor for shared error causes. Manual
trust and source-evidence ESS do not correct copied likelihoods.

Keep candidate expansion explicit: a report cannot silently create a new semantic type
or infer outside the supplied universe. See [evidence and observations](../concepts/evidence.md)
and the [channel capabilities](../reference/capabilities.md#observation-channels).
