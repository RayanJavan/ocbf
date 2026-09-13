"""Generate an object-centric order-to-cash process with retained synthetic truth.

Events can relate to several objects, including an order and multiple items. Optional
attributes and planted dependence support controlled numerical experiments. The generated
universe includes decoys and alternative event types so existence and association remain
uncertain inputs to a model.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from scipy.special import ndtr
from scipy.stats import poisson

from ocbf.assertions import AssertionRef, Family
from ocbf.schema import (
    AttributeKind,
    AttributeSpec,
    E2OQualifier,
    EventType,
    Lifecycle,
    ObjectType,
    O2OQualifier,
    Schema,
)
from ocbf.schema.core import AT_LEAST_ONE, MANY, ONE, OPTIONAL
from ocbf.universe import Universe, UniverseBuilder

E2OKey = tuple[str, str, str]
O2OKey = tuple[str, str, str]


PRIORITY_LEVELS = ("low", "normal", "high")
"""Ordered synthetic priority labels. Unordered nominal categories are excluded from copula transforms."""


def order_process_schema(*, attributes: bool = False) -> Schema:
    """An order-to-cash schema with genuine many-to-many structure.

    ``Ship`` takes one order and *many* items, which is what makes the log
    non-flattenable: no single case notion covers both the order and the item
    perspectives without duplication.

    ``attributes`` adds the attribute frames the continuous layer needs -- one per
    copula-eligible kind, so a run exercises the continuous, count, ordinal and truncated
    transforms rather than four copies of the easy case. It is off by default because
    declaring attributes registers latent variables, and the discrete benchmark's recorded
    figures are stated against a world without them.
    """
    order_attrs: tuple[AttributeSpec, ...] = ()
    item_attrs: tuple[AttributeSpec, ...] = ()
    invoice_attrs: tuple[AttributeSpec, ...] = ()
    place_attrs: tuple[AttributeSpec, ...] = ()
    ship_attrs: tuple[AttributeSpec, ...] = ()
    if attributes:
        order_attrs = (
            AttributeSpec("order_value", AttributeKind.CONTINUOUS, bounds=(0.0, None)),
            AttributeSpec("order_units", AttributeKind.COUNT, bounds=(0.0, None)),
        )
        item_attrs = (AttributeSpec("item_weight", AttributeKind.CONTINUOUS, bounds=(0.0, None)),)
        invoice_attrs = (
            AttributeSpec("invoice_total", AttributeKind.CONTINUOUS, bounds=(0.0, None)),
        )
        place_attrs = (AttributeSpec("order_priority", AttributeKind.ORDINAL, PRIORITY_LEVELS),)
        ship_attrs = (
            AttributeSpec("ship_delay_hours", AttributeKind.TRUNCATED, bounds=(0.0, None)),
        )

    return Schema(
        event_types=[
            EventType("PlaceOrder", place_attrs),
            EventType("ConfirmOrder"),
            EventType("PickItem"),
            EventType("PackItem"),
            EventType("Ship", ship_attrs),
            EventType("Invoice"),
            EventType("Pay"),
        ],
        object_types=[
            ObjectType("Order", order_attrs),
            ObjectType("Item", item_attrs),
            ObjectType("Customer"),
            ObjectType("Invoice", invoice_attrs),
        ],
        e2o=[
            E2OQualifier("order", "PlaceOrder", "Order", ONE),
            E2OQualifier("customer", "PlaceOrder", "Customer", ONE),
            E2OQualifier("order", "ConfirmOrder", "Order", ONE),
            E2OQualifier("item", "PickItem", "Item", ONE),
            E2OQualifier("order", "PickItem", "Order", ONE),
            E2OQualifier("item", "PackItem", "Item", ONE),
            E2OQualifier("order", "PackItem", "Order", ONE),
            # The many-to-many joint: one shipment, one order, several items.
            E2OQualifier("order", "Ship", "Order", ONE),
            E2OQualifier("item", "Ship", "Item", AT_LEAST_ONE),
            E2OQualifier("order", "Invoice", "Order", ONE),
            E2OQualifier("invoice", "Invoice", "Invoice", ONE),
            E2OQualifier("invoice", "Pay", "Invoice", ONE),
            E2OQualifier("customer", "Pay", "Customer", ONE),
        ],
        o2o=[
            O2OQualifier("contains", "Order", "Item", AT_LEAST_ONE),
            O2OQualifier("billed_to", "Invoice", "Customer", ONE),
            O2OQualifier("for_order", "Invoice", "Order", ONE),
        ],
        lifecycles=[
            Lifecycle(
                "Order",
                frozenset(
                    {
                        ("PlaceOrder", "ConfirmOrder"),
                        ("ConfirmOrder", "PickItem"),
                        ("PickItem", "PackItem"),
                        ("PackItem", "Ship"),
                        ("Ship", "Invoice"),
                    }
                ),
            ),
            Lifecycle("Item", frozenset({("PickItem", "PackItem"), ("PackItem", "Ship")})),
            Lifecycle("Invoice", frozenset({("Invoice", "Pay")})),
        ],
    )


@dataclass(slots=True)
class ProcessConfig:
    """Knobs on the generated world.

    ``decoy_event_rate`` and ``type_confusion_size`` are what keep the inference problem
    non-trivial. Without decoys, event existence is known; without a type support wider
    than the truth, ``T_e`` is determined. Both would quietly turn the benchmark into a
    much easier problem than the one the design targets.
    """

    n_orders: int = 40
    n_customers: int = 8
    items_per_order: tuple[int, int] = (1, 4)
    p_confirm: float = 0.85
    p_ship: float = 0.9
    p_invoice: float = 0.8
    p_pay: float = 0.7
    step_hours: float = 6.0
    jitter_hours: float = 1.5

    decoy_event_rate: float = 0.25
    """Extra candidate event slots that do not correspond to a real occurrence."""

    type_confusion_size: int = 3
    """Size of each event's latent type support, including the true type."""

    time_bracket_hours: float = 12.0
    """Half-width of the ``[time_lo, time_hi]`` bracket around the true timestamp."""

    attributes: bool = False
    """Generate synthetic attributes through a planted copula for numerical validation."""

    attribute_correlation: float = 0.6
    """True latent correlation between two attributes of the same object."""

    time_attribute_correlation: float = 0.5
    """Planted synthetic dependence between event times and event attributes."""

    p_type_known: float = 0.0
    """Fraction of candidate events whose activity label is certain.

    Not cosmetic: OCEL 2.0 requires attribute sets to be **disjoint** across types, so an
    event attribute is inherently type-specific, and under a type support wider than one it
    is never unambiguously in domain. Events with a determined type are therefore the only
    ones that carry event attributes into the copula (see
    [`PruneReport.attr_ambiguous`][ocbf.universe.core.PruneReport.attr_ambiguous]), and a
    world where every label is uncertain would leave that path untested.
    """

    seed: int = 0


@dataclass(slots=True)
class TrueEvent:
    id: str
    event_type: str
    time: float


class GroundTruth:
    """The generated world, plus the oracle that answers any assertion."""

    __slots__ = (
        "schema", "events", "objects", "object_types", "e2o", "o2o", "attributes", "config"
    )

    def __init__(
        self,
        schema: Schema,
        events: Mapping[str, TrueEvent],
        object_types: Mapping[str, str],
        e2o: frozenset[E2OKey],
        o2o: frozenset[O2OKey],
        config: ProcessConfig,
        attributes: Mapping[AssertionRef, float] | None = None,
    ) -> None:
        self.schema = schema
        self.events = dict(events)
        self.object_types = dict(object_types)
        self.e2o = e2o
        self.o2o = o2o
        self.attributes = dict(attributes or {})
        self.config = config

    def truth(self, ref: AssertionRef) -> bool | str | float | None:
        """Oracle for one assertion. ``None`` means "not applicable"."""
        match ref.family:
            case Family.EVENT_EXISTS:
                return ref.subject in self.events
            case Family.OBJECT_EXISTS:
                return ref.subject in self.object_types
            case Family.EVENT_TYPE:
                ev = self.events.get(ref.subject)
                return ev.event_type if ev else None
            case Family.EVENT_TIME:
                ev = self.events.get(ref.subject)
                return ev.time if ev else None
            case Family.E2O:
                return (ref.subject, ref.qualifier or "", ref.target or "") in self.e2o
            case Family.O2O:
                return (ref.subject, ref.qualifier or "", ref.target or "") in self.o2o
            case Family.EVENT_ATTR | Family.OBJECT_ATTR:
                # Ordinal attributes answer with the *level index*, which is what the copula
                # coordinate maps to and therefore what a source reports about.
                return self.attributes.get(ref)
            case _:
                return None

    def truth_map(self, refs: Iterable[AssertionRef]) -> dict[AssertionRef, bool | str | float | None]:
        return {r: self.truth(r) for r in refs}

    def build_universe(self, *, temporal_prune: bool = True) -> Universe:
        """Build candidate support around synthetic truth.

        Include decoy events, alternative event types and loose time brackets. Object identity/type remain supplied, while existence and qualified links are represented as uncertain assertions."""
        rng = np.random.default_rng(self.config.seed + 9973)
        cfg = self.config
        all_types = list(self.schema.event_type_names)
        builder = UniverseBuilder(self.schema)

        for ev in self.events.values():
            support = _confusion_support(
                ev.event_type, all_types, cfg.type_confusion_size, rng, cfg.p_type_known
            )
            builder.add_event(
                ev.id,
                type_support=support,
                time_lo=ev.time - cfg.time_bracket_hours,
                time_hi=ev.time + cfg.time_bracket_hours,
            )

        n_decoys = int(round(cfg.decoy_event_rate * len(self.events)))
        times = np.array([e.time for e in self.events.values()], dtype=float)
        lo, hi = (float(times.min()), float(times.max())) if times.size else (0.0, 1.0)
        for i in range(n_decoys):
            t = float(rng.uniform(lo, hi))
            fake_type = str(rng.choice(all_types))
            builder.add_event(
                f"decoy_{i}",
                type_support=_confusion_support(
                    fake_type, all_types, cfg.type_confusion_size, rng, cfg.p_type_known
                ),
                time_lo=t - cfg.time_bracket_hours,
                time_hi=t + cfg.time_bracket_hours,
            )

        for obj_id, obj_type in self.object_types.items():
            builder.add_object(obj_id, obj_type)

        # O2O candidates: the true links plus plausible-but-false ones, so the O2O layer
        # has something to discriminate.
        orders = [o for o, t in self.object_types.items() if t == "Order"]
        items = [o for o, t in self.object_types.items() if t == "Item"]
        for key in self.o2o:
            builder.add_o2o_candidate(*key)
        for order in orders:
            for item in rng.choice(items, size=min(3, len(items)), replace=False):
                builder.add_o2o_candidate(order, "contains", str(item))

        return builder.build(temporal_prune=temporal_prune)

    def summary(self) -> dict[str, int]:
        """Counts of generated events, objects, relations, and events per type."""
        by_type: dict[str, int] = {}
        for ev in self.events.values():
            by_type[ev.event_type] = by_type.get(ev.event_type, 0) + 1
        return {
            "events": len(self.events),
            "objects": len(self.object_types),
            "e2o": len(self.e2o),
            "o2o": len(self.o2o),
            **({"attributes": len(self.attributes)} if self.attributes else {}),
            **{f"n_{k}": v for k, v in sorted(by_type.items())},
        }

    def __repr__(self) -> str:
        return (
            f"GroundTruth(events={len(self.events)}, objects={len(self.object_types)}, "
            f"e2o={len(self.e2o)}, o2o={len(self.o2o)})"
        )


def _confusion_support(
    true_type: str,
    all_types: list[str],
    size: int,
    rng: np.random.Generator,
    p_known: float = 0.0,
) -> frozenset[str]:
    """A latent type support containing the truth plus ``size-1`` confusable alternatives.

    With probability ``p_known`` the support collapses to the truth alone, modelling an event
    whose activity label is certain while its links remain in doubt. That is a common real
    shape, and it is the only shape under which an event attribute is unambiguously in
    domain -- see [`ProcessConfig.p_type_known`][ocbf.synth.process.ProcessConfig.p_type_known].
    """
    if p_known > 0.0 and rng.random() < p_known:
        return frozenset([true_type])
    size = max(1, min(size, len(all_types)))
    others = [t for t in all_types if t != true_type]
    rng.shuffle(others)
    return frozenset([true_type, *others[: size - 1]])


def _sample_attributes(
    events: Mapping[str, TrueEvent],
    object_types: Mapping[str, str],
    config: ProcessConfig,
    rng: np.random.Generator,
) -> dict[AssertionRef, float]:
    """Sample synthetic attributes through a known latent Gaussian copula, retaining planted object and event dependence for numerical comparisons."""
    out: dict[AssertionRef, float] = {}
    rho_object = config.attribute_correlation
    rho_time = config.time_attribute_correlation

    def conditional(z: float, rho: float) -> float:
        return rho * z + math.sqrt(max(1.0 - rho * rho, 0.0)) * float(rng.standard_normal())

    for object_id, object_type in object_types.items():
        match object_type:
            case "Order":
                z = float(rng.standard_normal())
                out[AssertionRef.object_attr(object_id, "order_value")] = float(
                    np.exp(4.0 + 0.6 * z)
                )
                out[AssertionRef.object_attr(object_id, "order_units")] = float(
                    poisson.ppf(ndtr(conditional(z, rho_object)), 6.0)
                )
            case "Item":
                out[AssertionRef.object_attr(object_id, "item_weight")] = float(
                    np.exp(0.5 + 0.4 * rng.standard_normal())
                )
            case "Invoice":
                out[AssertionRef.object_attr(object_id, "invoice_total")] = float(
                    np.exp(4.5 + 0.5 * rng.standard_normal())
                )

    times = np.array([e.time for e in events.values()], dtype=np.float64)
    centre = float(times.mean()) if times.size else 0.0
    spread = float(times.std()) if times.size and times.std() > 0 else 1.0
    for event in events.values():
        z_time = (event.time - centre) / spread
        if event.event_type == "PlaceOrder":
            u = float(ndtr(conditional(z_time, rho_time)))
            level = int(np.searchsorted(np.array([0.55, 0.85]), u))
            out[AssertionRef.event_attr(event.id, "order_priority")] = float(level)
        elif event.event_type == "Ship":
            u = float(ndtr(conditional(z_time, rho_time)))
            # Zero-inflated: 60% of shipments leave on schedule, the rest are exponentially
            # late. The point mass is the truncated marginal's whole reason for existing.
            out[AssertionRef.event_attr(event.id, "ship_delay_hours")] = (
                0.0 if u < 0.6 else float(-2.0 * np.log((1.0 - u) / 0.4))
            )
    return out


def simulate_process(config: ProcessConfig | None = None) -> GroundTruth:
    """Generate one object-centric world."""
    cfg = config or ProcessConfig()
    rng = np.random.default_rng(cfg.seed)
    schema = order_process_schema(attributes=cfg.attributes)

    object_types: dict[str, str] = {}
    events: dict[str, TrueEvent] = {}
    e2o: set[E2OKey] = set()
    o2o: set[O2OKey] = set()

    customers = [f"cust_{i}" for i in range(cfg.n_customers)]
    for c in customers:
        object_types[c] = "Customer"

    clock = 0.0
    counter = 0

    def emit(event_type: str, at: float, links: Iterable[tuple[str, str]]) -> str:
        nonlocal counter
        eid = f"ev_{counter:05d}"
        counter += 1
        events[eid] = TrueEvent(eid, event_type, at)
        for qualifier, obj in links:
            e2o.add((eid, qualifier, obj))
        return eid

    def tick(t: float) -> float:
        return t + cfg.step_hours + float(rng.normal(0.0, cfg.jitter_hours))

    for k in range(cfg.n_orders):
        order = f"order_{k}"
        object_types[order] = "Order"
        customer = str(rng.choice(customers))

        n_items = int(rng.integers(cfg.items_per_order[0], cfg.items_per_order[1] + 1))
        items = [f"item_{k}_{j}" for j in range(n_items)]
        for it in items:
            object_types[it] = "Item"
            o2o.add((order, "contains", it))

        clock = tick(clock)
        t = clock
        emit("PlaceOrder", t, [("order", order), ("customer", customer)])

        if rng.random() > cfg.p_confirm:
            continue
        t = tick(t)
        emit("ConfirmOrder", t, [("order", order)])

        packed: list[str] = []
        for it in items:
            t = tick(t)
            emit("PickItem", t, [("item", it), ("order", order)])
            t = tick(t)
            emit("PackItem", t, [("item", it), ("order", order)])
            packed.append(it)

        if not packed or rng.random() > cfg.p_ship:
            continue
        t = tick(t)
        # The many-to-many joint: one Ship event, one order, every packed item.
        emit("Ship", t, [("order", order), *(("item", it) for it in packed)])

        if rng.random() > cfg.p_invoice:
            continue
        invoice = f"inv_{k}"
        object_types[invoice] = "Invoice"
        o2o.add((invoice, "billed_to", customer))
        o2o.add((invoice, "for_order", order))
        t = tick(t)
        emit("Invoice", t, [("order", order), ("invoice", invoice)])

        if rng.random() > cfg.p_pay:
            continue
        t = tick(t)
        emit("Pay", t, [("invoice", invoice), ("customer", customer)])

    attributes = (
        _sample_attributes(events, object_types, cfg, np.random.default_rng(cfg.seed + 7919))
        if cfg.attributes
        else {}
    )
    return GroundTruth(
        schema, events, object_types, frozenset(e2o), frozenset(o2o), cfg, attributes
    )
