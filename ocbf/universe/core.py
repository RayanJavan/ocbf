"""The fixed candidate universe, and the grounding pass that builds it.

Naive E2O grounding is ``|E| x |Q| x |O|`` -- 1e12 and hopeless at the scale of design doc
section 1.3. Three prunes, applied in order, bring it to the 1e5-1e7 range:

1. **Qualifier signature.** Because ``T_e`` is latent, a link survives if *some* event type
   still in the event's support makes the ``(event type, qualifier, object type)``
   signature legal. Typically cuts two to four orders of magnitude.
2. **Temporal windowing.** An object has a lifespan; links to events whose time bracket
   cannot overlap it are dropped.
3. **Claim-anchored blocking** ([`Universe.active_refs`][ocbf.universe.core.Universe.active_refs]). A candidate with no claim
   and no structural path to a claimed variable within ``k`` hops contributes only its
   prior, so it is never materialised into the graph.

The third is the one that makes the sparse regime work *for* us: sparse claims mean the
active graph is far smaller than the candidate universe.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef, Family, VariableRegistry
from ocbf.schema import Schema

E2OKey = tuple[str, str, str]
"""``(event id, qualifier, object id)``."""

O2OKey = tuple[str, str, str]
"""``(source object id, qualifier, target object id)``."""


@dataclass(frozen=True, slots=True)
class EventCandidate:
    """A potential occurrence: a *slot*, not a typed event.

    ``type_support`` is the set of event types this slot could be. Modelling one
    categorical ``T_e`` over the support is ``K`` times cheaper than ``K`` mutually
    exclusive binary candidates and gives belief propagation a far better-conditioned
    graph (design doc section 1.1).

    ``time_lo``/``time_hi`` bracket the timestamp. They drive the temporal prune only; the
    actual ``tau_e`` posterior is inferred in the continuous layer.
    """

    id: str
    type_support: frozenset[str]
    time_lo: float = -math.inf
    time_hi: float = math.inf

    def __post_init__(self) -> None:
        if not self.type_support:
            raise ValueError(f"event {self.id!r} has an empty type support")
        if self.time_lo > self.time_hi:
            raise ValueError(f"event {self.id!r} has an empty time bracket")

    @property
    def type_domain(self) -> tuple[str, ...]:
        """Support in a stable order -- this *is* the categorical domain of ``T_e``."""
        return tuple(sorted(self.type_support))


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    """A registry entry. Type is clamped; existence is latent.

    The asymmetry with [`EventCandidate`][ocbf.universe.core.EventCandidate] is deliberate and argued in design doc
    section 1.2: identity being clamped hands us a registry, and an OCEL 2.0 registry entry
    cannot name its own attributes without its type.
    """

    id: str
    object_type: str
    alive_from: float = -math.inf
    alive_to: float = math.inf

    def __post_init__(self) -> None:
        if self.alive_from > self.alive_to:
            raise ValueError(f"object {self.id!r} has an empty lifespan")


@dataclass(slots=True)
class PruneReport:
    """What grounding discarded, and why.

    Reported rather than logged: a pruned candidate is a modelling assertion (prior-only
    belief), so the counts belong in the results.
    """

    e2o_signature_legal: int = 0
    e2o_after_temporal: int = 0
    o2o_signature_legal: int = 0
    e2o_naive: int = 0
    o2o_naive: int = 0

    attr_registered: int = 0
    """Attribute variables that entered the continuous layer."""

    attr_ambiguous: int = 0
    """Event attributes dropped because the event's type support disagrees about them.

    An attribute is in-domain for some event types and not for others, and ``T_e`` is
    latent, so whether the attribute *applies* is itself uncertain. Design doc section 2.2
    resolves that with an explicit ``NA`` state, which is available to a discrete attribute
    variable and not to a Gaussian coordinate -- a latent Gaussian has no "does not apply"
    value, and there is no well-posed way to weigh a density against the probability of
    absence.

    So the conservative reading is taken, exactly as the cardinality factor does where types
    disagree: the attribute is registered only where *every* type in the support declares
    it, and is otherwise prior-only and counted here. That confines the relaxation to
    assertions the belief state makes no claim about anyway.
    """

    attr_categorical: int = 0
    """Attributes left in the discrete layer because they are unordered categoricals.

    Design doc section 4.2's honest boundary, counted rather than assumed: a categorical
    attribute has no monotone image in a Gaussian, so the copula does not cover it.
    """

    @property
    def e2o_reduction(self) -> float:
        """Fraction of the naive E2O space removed. The headline scalability number."""
        if self.e2o_naive == 0:
            return 0.0
        return 1.0 - self.e2o_after_temporal / self.e2o_naive

    def summary(self) -> dict[str, float | int]:
        """Candidate counts before and after each prune, with the reduction achieved."""
        return {
            "e2o_naive": self.e2o_naive,
            "e2o_after_signature": self.e2o_signature_legal,
            "e2o_after_temporal": self.e2o_after_temporal,
            "e2o_reduction": round(self.e2o_reduction, 6),
            "o2o_naive": self.o2o_naive,
            "o2o_after_signature": self.o2o_signature_legal,
            **(
                {
                    "attr_registered": self.attr_registered,
                    "attr_ambiguous": self.attr_ambiguous,
                    "attr_categorical": self.attr_categorical,
                }
                if self.attr_registered or self.attr_ambiguous or self.attr_categorical
                else {}
            ),
        }


class Universe:
    """A grounded candidate world plus its variable registry.

    Built by [`UniverseBuilder`][ocbf.universe.core.UniverseBuilder]; treat as immutable.
    """

    __slots__ = (
        "schema",
        "_events",
        "_objects",
        "_e2o",
        "_o2o",
        "registry",
        "prune_report",
        "_e2o_by_event",
        "_e2o_by_object",
        "_o2o_by_object",
    )

    def __init__(
        self,
        schema: Schema,
        events: Mapping[str, EventCandidate],
        objects: Mapping[str, ObjectRecord],
        e2o: tuple[E2OKey, ...],
        o2o: tuple[O2OKey, ...],
        registry: VariableRegistry,
        prune_report: PruneReport,
    ) -> None:
        self.schema = schema
        self._events = dict(events)
        self._objects = dict(objects)
        self._e2o = e2o
        self._o2o = o2o
        self.registry = registry
        self.prune_report = prune_report

        self._e2o_by_event: dict[str, list[int]] = defaultdict(list)
        self._e2o_by_object: dict[str, list[int]] = defaultdict(list)
        for i, (e, _q, o) in enumerate(e2o):
            self._e2o_by_event[e].append(i)
            self._e2o_by_object[o].append(i)

        self._o2o_by_object: dict[str, list[int]] = defaultdict(list)
        for i, (s, _q, t) in enumerate(o2o):
            self._o2o_by_object[s].append(i)
            self._o2o_by_object[t].append(i)

    # -- accessors ------------------------------------------------------------------

    @property
    def events(self) -> Mapping[str, EventCandidate]:
        return self._events

    @property
    def objects(self) -> Mapping[str, ObjectRecord]:
        return self._objects

    @property
    def e2o_candidates(self) -> tuple[E2OKey, ...]:
        return self._e2o

    @property
    def o2o_candidates(self) -> tuple[O2OKey, ...]:
        return self._o2o

    def event_type_domain(self, event: str) -> tuple[str, ...]:
        """Ordered type support of an event -- the domain of its `T_e` variable."""
        return self._events[event].type_domain

    def e2o_of_event(self, event: str) -> list[E2OKey]:
        """All candidate E2O links attached to one event."""
        return [self._e2o[i] for i in self._e2o_by_event.get(event, ())]

    def e2o_of_object(self, obj: str) -> list[E2OKey]:
        """All candidate E2O links attached to one object."""
        return [self._e2o[i] for i in self._e2o_by_object.get(obj, ())]

    def e2o_group(self, event: str, qualifier: str) -> list[E2OKey]:
        """All candidate links for one ``(event, qualifier)`` pair.

        This is the scope of a cardinality counting factor (design doc section 3.1), and
        the block resampled jointly by blocked Gibbs (section 6.4).
        """
        return [k for k in self.e2o_of_event(event) if k[1] == qualifier]

    def qualifiers_of_event(self, event: str) -> frozenset[str]:
        """Qualifiers with at least one surviving candidate link on this event."""
        return frozenset(k[1] for k in self.e2o_of_event(event))

    # -- prune 3: claim-anchored blocking --------------------------------------------

    def active_refs(
        self, claimed: Iterable[AssertionRef], hops: int = 1
    ) -> frozenset[AssertionRef]:
        """Refs within ``hops`` structural steps of a claimed ref.

        Adjacency is *entity co-occurrence*: two refs are neighbours when they touch a
        common event or object id. That is the relation along which the hard schema
        factors and the structural prior connect variables, so it is the right notion of
        "the prior can move this belief from that evidence".

        Anything outside the returned set is answered with the structural prior. This is
        the graceful-degradation guarantee of design doc section 10, made operational.
        """
        if hops < 0:
            raise ValueError("hops must be non-negative")

        by_entity: dict[str, list[AssertionRef]] = defaultdict(list)
        for ref in self.registry:
            for entity in ref.touches:
                by_entity[entity].append(ref)

        seed = [r for r in claimed if r in self.registry]
        seen: set[AssertionRef] = set(seed)
        frontier: deque[tuple[AssertionRef, int]] = deque((r, 0) for r in seed)

        while frontier:
            ref, depth = frontier.popleft()
            if depth >= hops:
                continue
            for entity in ref.touches:
                for neighbour in by_entity[entity]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        frontier.append((neighbour, depth + 1))
        return frozenset(seen)

    # -- reporting ------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        """Universe size, per-family variable counts, and the prune report."""
        return {
            "events": len(self._events),
            "objects": len(self._objects),
            "e2o_candidates": len(self._e2o),
            "o2o_candidates": len(self._o2o),
            "variables": len(self.registry),
            "by_family": self.registry.summary(),
            "prune": self.prune_report.summary(),
        }

    def __repr__(self) -> str:
        return (
            f"Universe(events={len(self._events)}, objects={len(self._objects)}, "
            f"e2o={len(self._e2o)}, o2o={len(self._o2o)}, vars={len(self.registry)})"
        )


class UniverseBuilder:
    """Accumulates candidates, then grounds them into a [`Universe`][ocbf.universe.core.Universe]."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self._events: dict[str, EventCandidate] = {}
        self._objects: dict[str, ObjectRecord] = {}
        self._explicit_o2o: set[O2OKey] = set()

    # -- accumulation ---------------------------------------------------------------

    def add_event(
        self,
        event_id: str,
        type_support: Iterable[str] | None = None,
        time_lo: float = -math.inf,
        time_hi: float = math.inf,
    ) -> UniverseBuilder:
        """Register a candidate event slot.

        ``type_support=None`` means "could be any declared event type" -- maximal
        uncertainty, and the honest default when a slot comes from a detector that
        localises an occurrence without labelling it.
        """
        support = (
            frozenset(self.schema.event_type_names)
            if type_support is None
            else frozenset(type_support)
        )
        unknown = support - set(self.schema.event_type_names)
        if unknown:
            raise ValueError(f"event {event_id!r} names unknown event types {sorted(unknown)}")
        if event_id in self._events:
            raise ValueError(f"duplicate event candidate {event_id!r}")
        self._events[event_id] = EventCandidate(event_id, support, time_lo, time_hi)
        return self

    def add_object(
        self,
        object_id: str,
        object_type: str,
        alive_from: float = -math.inf,
        alive_to: float = math.inf,
    ) -> UniverseBuilder:
        if object_type not in self.schema.object_types:
            raise ValueError(f"object {object_id!r} has unknown type {object_type!r}")
        if object_id in self._objects:
            raise ValueError(f"duplicate object {object_id!r}")
        self._objects[object_id] = ObjectRecord(object_id, object_type, alive_from, alive_to)
        return self

    def add_o2o_candidate(self, source: str, qualifier: str, target: str) -> UniverseBuilder:
        """Propose one O2O link explicitly.

        O2O candidates are *not* auto-generated from the signature table: the object-object
        space is quadratic and, unlike E2O, has no temporal prune to shrink it. Callers
        supply candidates from blocking or from source scopes.
        """
        self._explicit_o2o.add((source, qualifier, target))
        return self

    # -- grounding ------------------------------------------------------------------

    def build(self, *, temporal_prune: bool = True) -> Universe:
        """Apply the prunes, register variables, and return the grounded universe."""
        report = PruneReport()
        e2o = self._ground_e2o(report, temporal_prune=temporal_prune)
        o2o = self._ground_o2o(report)
        registry = self._register(e2o, o2o, report)
        return Universe(
            self.schema, self._events, self._objects, e2o, o2o, registry, report
        )

    def _ground_e2o(self, report: PruneReport, *, temporal_prune: bool) -> tuple[E2OKey, ...]:
        """Prunes 1 and 2.

        Implementation note: the loop is over ``(object type, legal (event type, qualifier)
        pairs)`` rather than over the naive triple product, and the temporal test is a
        vectorised mask over per-event-type numpy buckets. That keeps grounding roughly
        linear in the number of *surviving* candidates instead of in the naive space.
        """
        n_qualifier_names = len({q.qualifier for q in self.schema.e2o_qualifiers})
        report.e2o_naive = len(self._events) * len(self._objects) * max(n_qualifier_names, 1)

        # Bucket events by each type in their support, as numpy arrays for the mask.
        buckets: dict[str, list[str]] = defaultdict(list)
        for ev in self._events.values():
            for et in ev.type_support:
                buckets[et].append(ev.id)

        bucket_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for et, ids in buckets.items():
            lo = np.fromiter((self._events[i].time_lo for i in ids), dtype=np.float64, count=len(ids))
            hi = np.fromiter((self._events[i].time_hi for i in ids), dtype=np.float64, count=len(ids))
            bucket_arrays[et] = (np.array(ids, dtype=object), lo, hi)

        # Legal (event type, qualifier) pairs per object type.
        legal_by_otype: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for spec in self.schema.e2o_qualifiers:
            legal_by_otype[spec.object_type].append((spec.event_type, spec.qualifier))

        survivors: set[E2OKey] = set()
        signature_hits = 0
        for obj in self._objects.values():
            for event_type, qualifier in legal_by_otype.get(obj.object_type, ()):
                bucket = bucket_arrays.get(event_type)
                if bucket is None:
                    continue
                ids, lo, hi = bucket
                signature_hits += len(ids)
                if temporal_prune:
                    mask = (lo <= obj.alive_to) & (hi >= obj.alive_from)
                    selected = ids[mask]
                else:
                    selected = ids
                for event_id in selected:
                    survivors.add((str(event_id), qualifier, obj.id))

        # ``signature_hits`` counts before de-duplication across event types in a support;
        # the reported figure is the de-duplicated one, which is what actually matters.
        report.e2o_signature_legal = signature_hits
        report.e2o_after_temporal = len(survivors)
        return tuple(sorted(survivors))

    def _ground_o2o(self, report: PruneReport) -> tuple[O2OKey, ...]:
        report.o2o_naive = len(self._objects) ** 2 * max(
            len({q.qualifier for q in self.schema.o2o_qualifiers}), 1
        )
        survivors: set[O2OKey] = set()
        for source, qualifier, target in self._explicit_o2o:
            if source not in self._objects or target not in self._objects:
                raise ValueError(f"O2O candidate ({source}, {qualifier}, {target}) names an unknown object")
            st = self._objects[source].object_type
            tt = self._objects[target].object_type
            if self.schema.is_legal_o2o(st, qualifier, tt):
                survivors.add((source, qualifier, target))
        report.o2o_signature_legal = len(survivors)
        return tuple(sorted(survivors))

    def _register(
        self, e2o: tuple[E2OKey, ...], o2o: tuple[O2OKey, ...], report: PruneReport
    ) -> VariableRegistry:
        reg = VariableRegistry()
        for ev in self._events.values():
            reg.add_binary(AssertionRef.event_exists(ev.id))
            domain = ev.type_domain
            if len(domain) > 1:
                reg.add_categorical(AssertionRef.event_type(ev.id), len(domain))
            # A singleton support means the type is determined; no variable is created and
            # queries fall through to the clamped value.
            reg.add_continuous(AssertionRef.event_time(ev.id))
            self._register_event_attributes(reg, ev, report)
        for obj in self._objects.values():
            reg.add_binary(AssertionRef.object_exists(obj.id))
            self._register_object_attributes(reg, obj, report)
        for event_id, qualifier, object_id in e2o:
            reg.add_binary(AssertionRef.e2o(event_id, qualifier, object_id))
        for source, qualifier, target in o2o:
            reg.add_binary(AssertionRef.o2o(source, qualifier, target))
        return reg.freeze()

    def _register_event_attributes(
        self, reg: VariableRegistry, event: EventCandidate, report: PruneReport
    ) -> None:
        """Register the attributes every type in this event's support agrees it has.

        Agreement across the support is the condition, not membership in any one type. With
        ``T_e`` latent, an attribute declared by only some of the candidate types is an
        assertion whose very applicability is uncertain, and the reasons for handling that
        conservatively rather than with an ``NA`` state are recorded on
        [`PruneReport.attr_ambiguous`][ocbf.universe.core.PruneReport.attr_ambiguous].
        """
        declared = [self.schema.event_types[t].attributes for t in sorted(event.type_support)]
        shared = set.intersection(*(set(specs) for specs in declared)) if declared else set()
        union = set().union(*(set(specs) for specs in declared)) if declared else set()
        for spec in sorted(union, key=lambda s: s.name):
            if not spec.kind.in_copula:
                report.attr_categorical += 1
            elif spec in shared:
                reg.add_continuous(AssertionRef.event_attr(event.id, spec.name))
                report.attr_registered += 1
            else:
                report.attr_ambiguous += 1

    def _register_object_attributes(
        self, reg: VariableRegistry, obj: ObjectRecord, report: PruneReport
    ) -> None:
        """Register an object's copula-eligible attributes.

        No ambiguity arises here: object type is clamped (design doc section 1.2), so what an
        object's attributes *are* is known even though their values are not.
        """
        for spec in self.schema.object_types[obj.object_type].attributes:
            if not spec.kind.in_copula:
                report.attr_categorical += 1
                continue
            reg.add_continuous(AssertionRef.object_attr(obj.id, spec.name))
            report.attr_registered += 1
