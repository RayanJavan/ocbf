"""Core OCEL 2.0 schema objects.

Mirrors Definition 2 of the OCEL 2.0 specification (arXiv:2403.01975), with three
additions the specification leaves to the modeller and that we need in order to build a
probabilistic model on top of it:

* **multiplicities** on qualified relations -- the cardinality factors of design doc
  section 3.1 have nothing to ground against otherwise;
* **lifecycle orderings** per object type -- the soft precedence factors of section 3.3;
* **attribute kinds** -- the latent Gaussian copula of section 4 needs to know which
  marginal transform applies to each attribute.

Everything here is frozen and hashable: the schema is clamped, so nothing in it should
ever be mutated after construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum


class AttributeKind(str, Enum):
    """How an attribute enters the model.

    The first five kinds are handled by the latent Gaussian copula layer (design doc
    section 4.1). ``CATEGORICAL`` deliberately is not: unordered categoricals have no
    monotone transform to a latent Gaussian and stay in the discrete layer. That boundary
    is a real limitation of the copula approach, and naming it here keeps it visible.
    """

    CONTINUOUS = "continuous"
    COUNT = "count"
    ORDINAL = "ordinal"
    BINARY = "binary"
    TRUNCATED = "truncated"
    CATEGORICAL = "categorical"

    @property
    def in_copula(self) -> bool:
        """Whether this kind is representable as a monotone transform of a Gaussian."""
        return self is not AttributeKind.CATEGORICAL


@dataclass(frozen=True, slots=True)
class AttributeSpec:
    """One attribute of one event type or object type.

    ``levels`` is required for ``CATEGORICAL`` and ``ORDINAL`` (ordered, for the latter).
    ``bounds`` optionally constrains ``CONTINUOUS``/``COUNT``/``TRUNCATED`` domains.
    """

    name: str
    kind: AttributeKind
    levels: tuple[str, ...] | None = None
    bounds: tuple[float | None, float | None] = (None, None)

    def __post_init__(self) -> None:
        needs_levels = self.kind in (AttributeKind.CATEGORICAL, AttributeKind.ORDINAL)
        if needs_levels and not self.levels:
            raise ValueError(f"attribute {self.name!r} of kind {self.kind} requires levels")
        if not needs_levels and self.levels:
            raise ValueError(f"attribute {self.name!r} of kind {self.kind} must not have levels")
        if self.levels and len(set(self.levels)) != len(self.levels):
            raise ValueError(f"attribute {self.name!r} has duplicate levels")

    @property
    def cardinality(self) -> int | None:
        """Number of levels, or ``None`` for continuous-valued kinds."""
        return len(self.levels) if self.levels else None


@dataclass(frozen=True, slots=True)
class EventType:
    """An OCEL event type (an "activity")."""

    name: str
    attributes: tuple[AttributeSpec, ...] = ()

    def attribute(self, name: str) -> AttributeSpec:
        for spec in self.attributes:
            if spec.name == name:
                return spec
        raise KeyError(f"event type {self.name!r} has no attribute {name!r}")

    @property
    def attribute_names(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.attributes)


@dataclass(frozen=True, slots=True)
class ObjectType:
    """An OCEL object type."""

    name: str
    attributes: tuple[AttributeSpec, ...] = ()

    def attribute(self, name: str) -> AttributeSpec:
        for spec in self.attributes:
            if spec.name == name:
                return spec
        raise KeyError(f"object type {self.name!r} has no attribute {name!r}")

    @property
    def attribute_names(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.attributes)


@dataclass(frozen=True, slots=True)
class Multiplicity:
    """Declared cardinality of a qualified relation, as a closed interval.

    ``hi=None`` means unbounded. Per design doc section 3.4 multiplicity is a *domain
    belief*, not a definitional rule, so this becomes a soft counting factor rather than a
    hard constraint -- real logs violate their own schemas.
    """

    lo: int = 0
    hi: int | None = None

    def __post_init__(self) -> None:
        if self.lo < 0:
            raise ValueError("multiplicity lower bound must be non-negative")
        if self.hi is not None and self.hi < self.lo:
            raise ValueError(f"multiplicity [{self.lo}, {self.hi}] is empty")

    def contains(self, n: int) -> bool:
        return self.lo <= n and (self.hi is None or n <= self.hi)

    @property
    def is_exactly_one(self) -> bool:
        return self.lo == 1 and self.hi == 1

    def __str__(self) -> str:
        return f"[{self.lo}..{'*' if self.hi is None else self.hi}]"


ONE = Multiplicity(1, 1)
OPTIONAL = Multiplicity(0, 1)
MANY = Multiplicity(0, None)
AT_LEAST_ONE = Multiplicity(1, None)


@dataclass(frozen=True, slots=True)
class E2OQualifier:
    """A legal ``(event type, qualifier, object type)`` signature.

    The set of these *is* the qualifier-legality prune of design doc section 2.3, which is
    the single most effective scalability lever available: it cuts the naive
    ``|E| x |Q| x |O|`` E2O candidate space by two to four orders of magnitude.
    """

    qualifier: str
    event_type: str
    object_type: str
    multiplicity: Multiplicity = MANY

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.event_type, self.qualifier, self.object_type)


@dataclass(frozen=True, slots=True)
class O2OQualifier:
    """A legal ``(source object type, qualifier, target object type)`` signature."""

    qualifier: str
    source_type: str
    target_type: str
    multiplicity: Multiplicity = MANY

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source_type, self.qualifier, self.target_type)


@dataclass(frozen=True, slots=True)
class Lifecycle:
    """Soft precedence structure for one object type.

    ``precedes`` holds ``(before, after)`` event-type pairs that are expected to occur in
    that order among the events attached to a single object of this type. Per design doc
    section 3.4 this is a domain belief, so it becomes a soft truncation factor.
    """

    object_type: str
    precedes: frozenset[tuple[str, str]] = frozenset()

    def ordered_pairs(self) -> Iterator[tuple[str, str]]:
        yield from sorted(self.precedes)


class Schema:
    """The clamped type system of the world.

    Construction validates internal consistency: every qualifier signature must reference
    declared types, every lifecycle must reference declared event types, and attribute
    names must be disjoint across types (an OCEL 2.0 requirement -- the specification
    mandates a disjoint attribute set across all types so that an attribute name uniquely
    determines its owning type).
    """

    def __init__(
        self,
        event_types: Iterable[EventType],
        object_types: Iterable[ObjectType],
        e2o: Iterable[E2OQualifier] = (),
        o2o: Iterable[O2OQualifier] = (),
        lifecycles: Iterable[Lifecycle] = (),
    ) -> None:
        self._event_types: dict[str, EventType] = {et.name: et for et in event_types}
        self._object_types: dict[str, ObjectType] = {ot.name: ot for ot in object_types}
        self._e2o: tuple[E2OQualifier, ...] = tuple(e2o)
        self._o2o: tuple[O2OQualifier, ...] = tuple(o2o)
        self._lifecycles: dict[str, Lifecycle] = {lc.object_type: lc for lc in lifecycles}

        # Signature lookup tables, built once. These are hit once per candidate during
        # grounding, so they must be O(1).
        self._e2o_legal: dict[tuple[str, str, str], E2OQualifier] = {q.key: q for q in self._e2o}
        self._o2o_legal: dict[tuple[str, str, str], O2OQualifier] = {q.key: q for q in self._o2o}

        self._validate()

    # -- validation ---------------------------------------------------------------

    def _validate(self) -> None:
        for q in self._e2o:
            if q.event_type not in self._event_types:
                raise ValueError(f"E2O qualifier {q.qualifier!r} names unknown event type {q.event_type!r}")
            if q.object_type not in self._object_types:
                raise ValueError(f"E2O qualifier {q.qualifier!r} names unknown object type {q.object_type!r}")
        for q in self._o2o:
            for role, name in (("source", q.source_type), ("target", q.target_type)):
                if name not in self._object_types:
                    raise ValueError(f"O2O qualifier {q.qualifier!r} names unknown {role} type {name!r}")
        for lc in self._lifecycles.values():
            if lc.object_type not in self._object_types:
                raise ValueError(f"lifecycle names unknown object type {lc.object_type!r}")
            for before, after in lc.precedes:
                for name in (before, after):
                    if name not in self._event_types:
                        raise ValueError(
                            f"lifecycle for {lc.object_type!r} names unknown event type {name!r}"
                        )

        # OCEL 2.0 requires attribute names to be disjoint across types, so that an
        # attribute name determines its owner. We enforce it because the assertion algebra
        # addresses attributes by name alone.
        seen: dict[str, str] = {}
        for owner, specs in (
            *((et.name, et.attributes) for et in self._event_types.values()),
            *((ot.name, ot.attributes) for ot in self._object_types.values()),
        ):
            for spec in specs:
                if spec.name in seen:
                    raise ValueError(
                        f"attribute {spec.name!r} is declared on both {seen[spec.name]!r} "
                        f"and {owner!r}; OCEL 2.0 requires disjoint attribute sets across types"
                    )
                seen[spec.name] = owner

    # -- accessors ----------------------------------------------------------------

    @property
    def event_types(self) -> Mapping[str, EventType]:
        return self._event_types

    @property
    def object_types(self) -> Mapping[str, ObjectType]:
        return self._object_types

    @property
    def event_type_names(self) -> tuple[str, ...]:
        """Event type names in a stable order -- this defines the categorical domain of ``T_e``."""
        return tuple(sorted(self._event_types))

    @property
    def object_type_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._object_types))

    @property
    def e2o_qualifiers(self) -> tuple[E2OQualifier, ...]:
        return self._e2o

    @property
    def o2o_qualifiers(self) -> tuple[O2OQualifier, ...]:
        return self._o2o

    def lifecycle(self, object_type: str) -> Lifecycle | None:
        """Declared event ordering for an object type, if any."""
        return self._lifecycles.get(object_type)

    # -- signature queries (the pruning primitives) --------------------------------

    def is_legal_e2o(self, event_type: str, qualifier: str, object_type: str) -> bool:
        """Whether this event-to-object signature is declared.

        The predicate behind the qualifier-signature prune, which is the single most
        effective scalability lever in grounding.
        """
        return (event_type, qualifier, object_type) in self._e2o_legal

    def is_legal_o2o(self, source_type: str, qualifier: str, target_type: str) -> bool:
        """Whether this object-to-object signature is declared."""
        return (source_type, qualifier, target_type) in self._o2o_legal

    def e2o_spec(self, event_type: str, qualifier: str, object_type: str) -> E2OQualifier:
        return self._e2o_legal[(event_type, qualifier, object_type)]

    def o2o_spec(self, source_type: str, qualifier: str, target_type: str) -> O2OQualifier:
        return self._o2o_legal[(source_type, qualifier, target_type)]

    def legal_e2o_for(
        self, event_type_support: Iterable[str], object_type: str
    ) -> frozenset[str]:
        """Qualifiers legal for *some* event type in the support.

        Because ``T_e`` is latent (design doc section 1.1), a candidate link survives the
        signature prune if any type still in the event's support makes it legal. This is
        the exact predicate used by ``ocbf.universe``.
        """
        support = set(event_type_support)
        return frozenset(
            q.qualifier
            for q in self._e2o
            if q.object_type == object_type and q.event_type in support
        )

    def attribute_owner(self, attribute: str) -> tuple[str, AttributeSpec]:
        """Return ``(owning type name, spec)`` for an attribute name.

        Well-defined because ``_validate`` enforces disjointness.
        """
        for et in self._event_types.values():
            if attribute in et.attribute_names:
                return et.name, et.attribute(attribute)
        for ot in self._object_types.values():
            if attribute in ot.attribute_names:
                return ot.name, ot.attribute(attribute)
        raise KeyError(f"no type declares attribute {attribute!r}")

    def event_types_declaring(self, attribute: str) -> frozenset[str]:
        """Event types for which ``attribute`` is in-domain.

        Used by the attribute-domain gate: because ``T_e`` is latent, an event attribute
        variable carries an explicit ``NA`` state that is forced whenever ``T_e`` falls
        outside this set (design doc section 2.2).
        """
        return frozenset(
            et.name for et in self._event_types.values() if attribute in et.attribute_names
        )

    def __repr__(self) -> str:
        return (
            f"Schema(event_types={len(self._event_types)}, object_types={len(self._object_types)}, "
            f"e2o={len(self._e2o)}, o2o={len(self._o2o)}, lifecycles={len(self._lifecycles)})"
        )
