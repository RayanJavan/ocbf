"""Typed addresses into the latent world.

An [`AssertionRef`][ocbf.assertions.refs.AssertionRef] names one latent quantity: "does event ``e17`` exist", "what type
is event ``e17``", "is object ``item_3`` linked to event ``e17`` under qualifier
``ships``". Sources emit claims *about* refs; the belief state answers queries *about*
refs; attribution vectors are indexed *by* ref.

Refs are frozen, slotted and hashable, and their string form round-trips through
[`AssertionRef.parse`][ocbf.assertions.refs.AssertionRef.parse], so they can be used as stable keys in files and reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Family(str, Enum):
    """Supported semantic assertion families. Object identity and type are supplied by the schema and universe; event type, existence, links, timestamps and attributes have distinct references."""

    EVENT_EXISTS = "event_exists"
    EVENT_TYPE = "event_type"
    EVENT_TIME = "event_time"
    OBJECT_EXISTS = "object_exists"
    E2O = "e2o"
    O2O = "o2o"
    EVENT_ATTR = "event_attr"
    OBJECT_ATTR = "object_attr"

    @property
    def is_structural(self) -> bool:
        """Existence, type and link families -- the discrete backbone of the graph."""
        return self in _STRUCTURAL

    @property
    def is_link(self) -> bool:
        """Whether this family describes a qualified E2O or O2O relation."""
        return self in (Family.E2O, Family.O2O)

    @property
    def is_continuous(self) -> bool:
        """Families living in the latent Gaussian copula layer."""
        return self is Family.EVENT_TIME


_STRUCTURAL = frozenset(
    {
        Family.EVENT_EXISTS,
        Family.EVENT_TYPE,
        Family.OBJECT_EXISTS,
        Family.E2O,
        Family.O2O,
    }
)


@dataclass(frozen=True, slots=True, order=True)
class AssertionRef:
    """A typed address into the latent world.

    Fields are interpreted per family:

    ========================  ==========  ===========  ==========  ===========
    family                    subject     qualifier    target      attribute
    ========================  ==========  ===========  ==========  ===========
    ``EVENT_EXISTS``          event id    --           --          --
    ``EVENT_TYPE``            event id    --           --          --
    ``EVENT_TIME``            event id    --           --          --
    ``OBJECT_EXISTS``         object id   --           --          --
    ``E2O``                   event id    qualifier    object id   --
    ``O2O``                   object id   qualifier    object id   --
    ``EVENT_ATTR``            event id    --           --          attr name
    ``OBJECT_ATTR``           object id   --           --          attr name
    ========================  ==========  ===========  ==========  ===========

    Use the classmethod constructors rather than the raw initialiser; they enforce the
    field-shape contract that `_validate` checks.
    """

    family: Family
    subject: str
    qualifier: str | None = None
    target: str | None = None
    attribute: str | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        needs_link = self.family.is_link
        has_link = self.qualifier is not None and self.target is not None
        if needs_link and not has_link:
            raise ValueError(f"{self.family.value} ref requires qualifier and target")
        if not needs_link and (self.qualifier is not None or self.target is not None):
            raise ValueError(f"{self.family.value} ref must not carry qualifier/target")

        needs_attr = self.family in (Family.EVENT_ATTR, Family.OBJECT_ATTR)
        if needs_attr and self.attribute is None:
            raise ValueError(f"{self.family.value} ref requires an attribute name")
        if not needs_attr and self.attribute is not None:
            raise ValueError(f"{self.family.value} ref must not carry an attribute name")

    # -- constructors --------------------------------------------------------------

    @classmethod
    def event_exists(cls, event: str) -> AssertionRef:
        return cls(Family.EVENT_EXISTS, event)

    @classmethod
    def event_type(cls, event: str) -> AssertionRef:
        return cls(Family.EVENT_TYPE, event)

    @classmethod
    def event_time(cls, event: str) -> AssertionRef:
        return cls(Family.EVENT_TIME, event)

    @classmethod
    def object_exists(cls, obj: str) -> AssertionRef:
        return cls(Family.OBJECT_EXISTS, obj)

    @classmethod
    def e2o(cls, event: str, qualifier: str, obj: str) -> AssertionRef:
        return cls(Family.E2O, event, qualifier=qualifier, target=obj)

    @classmethod
    def o2o(cls, source: str, qualifier: str, target: str) -> AssertionRef:
        return cls(Family.O2O, source, qualifier=qualifier, target=target)

    @classmethod
    def event_attr(cls, event: str, attribute: str) -> AssertionRef:
        return cls(Family.EVENT_ATTR, event, attribute=attribute)

    @classmethod
    def object_attr(cls, obj: str, attribute: str) -> AssertionRef:
        return cls(Family.OBJECT_ATTR, obj, attribute=attribute)

    # -- serialisation --------------------------------------------------------------

    def __str__(self) -> str:
        if self.family.is_link:
            return f"{self.family.value}({self.subject}, {self.qualifier}, {self.target})"
        if self.attribute is not None:
            return f"{self.family.value}({self.subject}.{self.attribute})"
        return f"{self.family.value}({self.subject})"

    @classmethod
    def parse(cls, text: str) -> AssertionRef:
        """Inverse of `__str__`.

        Raises ``ValueError`` on anything it did not produce, rather than guessing.
        """
        head, sep, rest = text.partition("(")
        if not sep or not rest.endswith(")"):
            raise ValueError(f"malformed assertion ref: {text!r}")
        try:
            family = Family(head)
        except ValueError as exc:
            raise ValueError(f"unknown assertion family in {text!r}") from exc

        body = rest[:-1]
        if family.is_link:
            parts = [p.strip() for p in body.split(",")]
            if len(parts) != 3:
                raise ValueError(f"{family.value} ref needs 3 components: {text!r}")
            return cls(family, parts[0], qualifier=parts[1], target=parts[2])
        if family in (Family.EVENT_ATTR, Family.OBJECT_ATTR):
            subject, dot, attribute = body.rpartition(".")
            if not dot:
                raise ValueError(f"{family.value} ref needs 'subject.attribute': {text!r}")
            return cls(family, subject, attribute=attribute)
        return cls(family, body)

    # -- convenience ----------------------------------------------------------------

    @property
    def touches(self) -> tuple[str, ...]:
        """Entity ids this assertion involves -- used for locality and blocking."""
        return (self.subject, self.target) if self.target is not None else (self.subject,)
