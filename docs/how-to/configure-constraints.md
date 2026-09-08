# Configure constraints

Structural constraints come from the schema and are the cheapest information the model has —
they carry no free parameters and they work on assertions no source ever mentioned. This
guide covers declaring them and choosing their strength.

## Declare the schema

```python
from ocbf.schema import E2OQualifier, EventType, Lifecycle, ObjectType, O2OQualifier, Schema
from ocbf.schema.core import AT_LEAST_ONE, MANY, ONE, OPTIONAL

schema = Schema(
    event_types=[EventType("PlaceOrder"), EventType("Ship")],
    object_types=[ObjectType("Order"), ObjectType("Item"), ObjectType("Customer")],
    e2o=[
        E2OQualifier("order",    "PlaceOrder", "Order",    ONE),
        E2OQualifier("customer", "PlaceOrder", "Customer", ONE),
        E2OQualifier("order",    "Ship",       "Order",    ONE),
        E2OQualifier("item",     "Ship",       "Item",     AT_LEAST_ONE),
    ],
    o2o=[O2OQualifier("contains", "Order", "Item", AT_LEAST_ONE)],
    lifecycles=[Lifecycle("Order", frozenset({("PlaceOrder", "Ship")}))],
)
```

The `e2o` list is doing more work than it looks. It is the **qualifier-signature prune** —
the single most effective scalability lever available, typically cutting the candidate link
space by two to four orders of magnitude. Declare every legal signature and no illegal ones.

!!! tip "Multiplicities are not decoration"

    They drive two things at once: the soft cardinality factor *and* the derived link prior.
    A group of `k` candidates under `ONE` gets a prior of `1/k`. Declaring `MANY` everywhere
    throws away both.

## Choose hard or soft

The register defaults implement the project's decision: **definitional rules are hard,
domain beliefs are soft.**

```python
from ocbf.schema import ConstraintClass, ConstraintRegister, Strength

register = ConstraintRegister()
register.is_hard(ConstraintClass.REFERENTIAL_INTEGRITY)   # True  — definitional
register.is_hard(ConstraintClass.CARDINALITY)             # False — domain belief
register.guarantees_valid_samples                          # True
```

| constraint | default | why |
| --- | --- | --- |
| `REFERENTIAL_INTEGRITY` | **hard** | a link to a non-existent event is not an unlikely log, it is not a log |
| `ATTRIBUTE_DOMAIN` | **hard** | definitional in the OCEL 2.0 specification |
| `QUALIFIER_LEGALITY` | **hard** | definitional in your schema |
| `TYPE_DISJOINTNESS` | **hard** | definitional |
| `CARDINALITY` | soft | schemas are aspirational; real logs violate them |
| `LIFECYCLE_PRECEDENCE` | soft | processes deviate |
| `ATTRIBUTE_MONOTONICITY` | soft | domain belief |
| `FUNCTIONAL_UNIQUENESS` | soft | and only where exclusivity is declared |

This distinction is measurable, not stylistic. Under a *false* multiplicity, links carrying
evidence keep a posterior of 0.72 when the constraint is soft and collapse to 0.14 when it
is effectively hard. A wrong hard constraint assigns probability zero to the truth and no
evidence recovers it.

## Adjust a weight

Soft weights are **dimensionless multipliers** on one claim's worth of evidence, not raw
nats:

```python
register = ConstraintRegister().override(ConstraintClass.CARDINALITY, weight=2.0)
# "one unit of violation costs about two confident claims"
```

The absolute scale is calibrated from your source pool at build time:

```python
from ocbf.model.build import calibrate_constraint_scale
calibrate_constraint_scale(claim_set, reliability)   # e.g. 0.178 nats per claim
```

A raw nat count could not hold its meaning here: the same number is a gentle nudge against
strong sources and an unbreakable rule against weak ones. Since the point of a soft
constraint is that evidence can win, the weight has to be denominated in evidence.

To pin an absolute scale instead:

```python
from ocbf.model import GraphSpec
spec = GraphSpec(constraint_weight_scale=1.0)   # weights are now literally nats
```

## Relax a definitional constraint

Permitted — it is how the sensitivity experiment runs — but it voids the guarantee that
posterior samples are valid OCEL logs:

```python
register = ConstraintRegister().override(
    ConstraintClass.REFERENTIAL_INTEGRITY, strength=Strength.SOFT, weight=5.0
)
register.guarantees_valid_samples    # False
```

Check that property before consuming samples as logs.

## Turn structure off for an ablation

```python
from ocbf.model import GraphSpec

bare = GraphSpec(
    include_referential_integrity=False,
    include_type_gate=False,
    include_cardinality=False,
)
```

This reduces the model to per-assertion voting with a prior, which is the right comparison
for measuring what the structure buys. Expect a substantial AUC drop.

## Tune the priors

```python
spec = GraphSpec(
    p_event_exists=0.6,          # how many candidate events are real?
    p_object_exists=0.9,         # registry entries are usually real
    link_prior_floor=0.005,      # bounds on the *derived* link prior
    link_prior_ceiling=0.5,
)
```

Note there is no `p_e2o_link`. Link priors are derived per `(event, qualifier)` group from
the declared multiplicity, so the prior and the cardinality factor agree by construction
rather than fighting. Making it a constant was a real bug: at `p = 0.08` over `k = 50`
candidates the prior expected four links while the constraint insisted on one, and BP failed
to converge as a result.

## See also

- [Hard versus soft, derived](../explanation/the-model.md#hard-and-soft) — the full argument.
- [`ocbf.schema`][ocbf.schema] and [`ocbf.model.build`][ocbf.model.build] — the APIs.
