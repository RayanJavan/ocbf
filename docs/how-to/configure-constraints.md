# Configure constraints

Decide what each rule means before putting it into a probability model.
The [model topic](../concepts/model.md#constraints-and-references) illustrates the difference
between excluding a history and assessing it against a duration reference.

| Role | Placement | Consequence |
|---|---|---|
| Definitional support | Canonical support factors | Violating states have zero probability. |
| Descriptive assumption | Explicit model factors/parameters | States receive the supplied relative weight. |
| Normative reference | Query specification | Histories can violate the reference and be assessed. |

Classify catalogue multiplicities through the semantic context and supplied
`cardinality_role` assumption. The [quickstart model](../getting-started/quickstart.md#5-define-support-and-process-questions)
uses normative catalogue multiplicities and separately declares an at-most-one end
association as hard support.

## Add a supported finite count factor

```python
from ocbf.model.spec import FactorSpec

factor = FactorSpec(
    "one-end-association",
    ("end-a-linked", "end-b-linked"),
    "count",
    {"lo": 0, "hi": 1},
    role="support",
)
```

This is a construction sketch: the two keys must name Boolean variables in the supplied
model. Factor parameters and roles are validated during compilation.

Typed qualifier rules must include compatible endpoint types. Lifecycle ordering requires
explicit endpoints belonging to the intended object; a shared Job does not justify pairing
every start with every completion.

Canonical hard support uses actual zero probability. The BP adapter admits only strictly
positive finite factors and therefore rejects such targets. Choose an admitted
[exact or sampling route](choose-inference.md).

A constraint-register configuration flag cannot certify a compiled posterior or an OCEL
file. Validate the actual support and [numerical route](../development/validation.md).
