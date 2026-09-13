# Semantic context

A semantic context describes the objects, events, and relationships that a calculation
can reason about. It combines a schema with a bounded set of candidates.

## Schema and universe

The schema defines types such as `Operation`, `start`, and `end`, together with qualified
relationships such as “this event has this Operation as its subject.” The universe
contains particular candidates: Operation `op` and events `start`, `short`, and `long`.

| Object | Role in the synthetic assessment |
|---|---|
| `Schema` | Declares the Operation type, start/end event types, and subject relationship. |
| `Universe` | Contains one Operation and three candidate events with their type support. |
| `SemanticContext` | Captures those inputs, constraint declarations, and provenance for a calculation. |
| `AssertionRef` | Identifies an assertion such as whether event `short` exists or belongs to `op`. |

A candidate event is a possibility. Its presence in the universe does not establish that
it happened. Object identities and types are supplied inputs; event existence, supported
event types, and associations can remain uncertain.

## Assertions and associations

The assertions “event `short` exists” and “event `short` belongs to Operation `op`” have
different meanings. An endpoint report can concern both assertions together. Other
evidence may support the event's existence without resolving its association.

Stable assertion references carry these meanings across evidence, models, and results:

```python
from ocbf.assertions import AssertionRef

existence = AssertionRef.event_exists("short")
association = AssertionRef.e2o("short", "subject", "op")
print(str(existence))
print(str(association))
```

Compiled array positions are local to a calculation. They are not persistent identifiers
for the assertions.

## Candidate boundaries

If the real endpoint is absent from the candidate universe, inference cannot discover it.
A model containing only `short` and `long` can assess those alternatives and any permitted
absence; it cannot assign probability to an unspecified third event.

Adding an alternative changes the context and model. Correcting a report about an existing
candidate changes the evidence while the type meanings remain fixed.

An event with an unknown time also differs from an event that did not occur. Its existence
and time require separate model treatment. The [model topic](model.md) describes how these
possibilities become histories; [query projections](queries.md#process-projections)
describe which parts of a history enter a question.
