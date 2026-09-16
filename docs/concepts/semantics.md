# Semantic context

A report such as "`op` ended at 10:15" only means something if OCBF knows what an Operation is,
what an end event is, and which events could be involved. This page fixes those meanings for the
[running example](index.md#the-running-example) and introduces the **assertions**, the
true-or-false facts that every later page works with.

## Schema: fixed meanings

A **schema** declares the types of events and objects, and the named relations allowed between
them. For the running example, it declares `start` and `end` events, `Operation` objects, and a
`subject` relation that says which Operation an event belongs to.

```python
from ocbf.schema import E2OQualifier, EventType, Multiplicity, ObjectType, Schema

schema: Schema = Schema(
    [EventType("start"), EventType("end")],  # (1)!
    [ObjectType("Operation")],
    e2o=[
        E2OQualifier("subject", kind, "Operation", Multiplicity(0, 1))  # (2)!
        for kind in ("start", "end")
    ],
)
```

1.  Event types name kinds of occurrence. Declaring `end` says nothing about whether any end
    happened.
2.  An event-to-object (E2O) relation: a `start` or `end` event can have an `Operation` as its
    `subject`, and `Multiplicity(0, 1)` allows at most one.

The schema says what *can* be stated, not what happened. It is also fixed for the whole
assessment: a correction changes a report, and the meaning of an `end` event stays as declared.

## Universe: the candidates

A **[candidate](../reference/glossary.md)** is a particular event or object that might be part of
the history. The **universe** lists all of them. The **semantic context** combines the schema, the
universe, and a note on where they came from (provenance) into one value that cannot be modified.

```python
from datetime import UTC, datetime, timedelta

from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext

AT = datetime(2026, 9, 8, 12, tzinfo=UTC)  # (1)!
TIMES: dict[str, datetime] = {  # (2)!
    "start": AT - timedelta(hours=2),
    "short": AT - timedelta(minutes=105),
    "long": AT - timedelta(hours=1),
}

builder: UniverseBuilder = UniverseBuilder(schema).add_object("op", "Operation")  # (3)!
builder.add_event("start", type_support={"start"})
builder.add_event("short", type_support={"end"})  # (4)!
builder.add_event("long", type_support={"end"})
context: SemanticContext = SemanticContext.from_universe(
    builder.build(), provenance={"status": "synthetic"}
)

support: dict[str, tuple[str, ...]] = {
    event.id: event.type_support for event in context.definition["events"]
}
for name, time in TIMES.items():
    minutes: int = (time - TIMES["start"]) // timedelta(minutes=1)
    print(name, support[name], f"+{minutes} min")
```

1.  The moment of the assessment, 12:00 UTC. Later pages use it as the cutoff for what is known
    and as the end of the period the question looks at.
2.  The time at which each candidate event would have happened. These are supplied inputs; the
    model attaches them to the events on the [model page](model.md).
3.  The Operation `op` is supplied as a fact: it exists and it is an `Operation`.
4.  `short` and `long` are both possible ends, and each may only have the type `end`. Listing
    them does not say that either one happened.

Running the block prints:

```text
start ('start',) +0 min
short ('end',) +15 min
long ('end',) +60 min
```

The universe contains three candidate events before any report is read. Each candidate's allowed
type is fixed. What remains uncertain is whether each event happened and whether it belongs to
`op`. The context also gets an identifier computed from its contents, `context.context_id`. Models
built from it store that identifier, and any result can be traced back to its context.

!!! note "A candidate is not evidence"

    The universe lists what *could* be part of the history. Reports, and the probabilities
    computed from them, decide how plausible each candidate is.

## Assertions: what can be true or false

An **[assertion](../reference/glossary.md)** is one fact about the candidates that each possible
history makes either true or false. OCBF refers to an assertion by a string that is the same
everywhere in the library:

```python
from ocbf.assertions import AssertionRef

exists: dict[str, str] = {
    event: str(AssertionRef.event_exists(event)) for event in TIMES  # (1)!
}
links: dict[str, str] = {
    event: str(AssertionRef.e2o(event, "subject", "op")) for event in TIMES  # (2)!
}
print(exists["short"])
print(links["short"])
```

1.  "Candidate event `short` happened."
2.  "Candidate event `short` has Operation `op` as its `subject`."

Running the block prints:

```text
event_exists(short)
e2o(short, subject, op)
```

These are two different facts. An end event can happen without belonging to `op`. A report can
also show that an event happened without showing which Operation it belongs to. Observations,
model variables, and query results all refer to assertions by these strings. A compiled model
numbers its variables internally as well, but those numbers can differ between calculations and
never identify an assertion.

## Candidate boundaries

OCBF computes probabilities only for the candidates in the universe:

```python
print(sorted(support))
print("third" in support)
```

Running the block prints:

```text
['long', 'short', 'start']
False
```

If the real end of `op` is not a candidate, no report can make OCBF consider it. This universe
allows OCBF to weigh `short`, `long`, and the possibility that neither belongs to `op`. It cannot
weigh an event named `third`. A report about `third` is rejected when the model is compiled, as
[Models and constraints](model.md#what-compilation-checks) shows. Adding `third` as a candidate
creates a different context with a different identifier, so results computed from the old context
do not apply to it.

An event with an unknown time is different from an event that did not happen; the model treats
*whether* an event happened and *when* as separate questions.

## Summary

The schema fixes what can be stated, the universe lists the candidates (`start`, `short`, `long`,
and `op`), and the semantic context holds both under one identifier, `context.context_id`. Each
history decides the assertions, such as `event_exists(short)` and `e2o(short, subject, op)`, and
every later object names them by these same strings. An event outside the universe, such as
`third`, is outside every calculation.

Next, [Evidence and observations](evidence.md) connects reports to these assertions.
