"""The constraint register: which structural rules are hard, and which are soft.

Design doc section 3.4 (Stage 1 decision 6): **hard zero factors for definitional schema
rules, soft high-weight penalties for domain beliefs**. The distinction is not cosmetic --
a hard factor that is wrong assigns probability zero to the truth and no amount of
evidence recovers it, whereas a soft one degrades gracefully. Section 8.3 makes that
claim testable by deliberately misspecifying a constraint and measuring the damage.

The defaults below encode decision 6. They are overridable per constraint class, because
the decision fixes the defaults, not a straitjacket.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Strength(str, Enum):
    """How a constraint enters the factor graph."""

    HARD = "hard"
    """A ``-inf`` log-potential. Posterior support excludes violating configurations."""

    SOFT = "soft"
    """A finite penalty of magnitude ``weight``. Violations are improbable, not impossible."""


class ConstraintClass(str, Enum):
    """The structural rules the model knows how to express.

    The first four are definitional -- they come from the OCEL 2.0 specification or from
    the clamped schema, so violating them does not produce an unlikely log, it produces a
    thing that is not a log. The rest are beliefs about a particular domain.
    """

    REFERENTIAL_INTEGRITY = "referential_integrity"
    """``R[e,q,o]=1`` implies ``X_e=1`` and ``X_o=1``; likewise for O2O."""

    ATTRIBUTE_DOMAIN = "attribute_domain"
    """``V[e,a]`` is ``NA`` unless ``T_e`` declares ``a``."""

    QUALIFIER_LEGALITY = "qualifier_legality"
    """Only declared ``(event type, qualifier, object type)`` signatures may link.

    Applied as a *prune* at grounding time rather than as a factor, since it does not
    depend on any latent value beyond the event-type support.
    """

    TYPE_DISJOINTNESS = "type_disjointness"
    """An event/object has exactly one type; attribute sets are disjoint across types."""

    CARDINALITY = "cardinality"
    """Declared multiplicity of a qualified relation. Soft: schemas are aspirational."""

    LIFECYCLE_PRECEDENCE = "lifecycle_precedence"
    """Expected event-type ordering within one object's trace. Soft: processes deviate."""

    ATTRIBUTE_MONOTONICITY = "attribute_monotonicity"
    """Monotone or state-machine-constrained attribute trajectories. Soft."""

    FUNCTIONAL_UNIQUENESS = "functional_uniqueness"
    """At most one true value for a declared-exclusive attribute at an instant. Soft.

    Only applied where exclusivity is *declared*. Stage 1 section 3.1 is emphatic that
    many object-centric assertions are legitimately multi-valued -- an event has many E2O
    edges -- so exclusivity is never assumed globally.
    """


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """How one constraint class is realised.

    ``weight`` is **dimensionless**: a multiplier on the evidence scale computed by
    [`calibrate_constraint_scale`][ocbf.model.build.calibrate_constraint_scale], which is the Chernoff information
    of a typical single claim. So ``weight=1.0`` means "one unit of violation costs about
    what one confident claim is worth", and a handful of claims can overrule a misstated
    schema belief.

    Expressing the weight in raw nats would tie its meaning to a particular source pool: the
    same number would be a gentle nudge against strong sources and an unbreakable rule
    against weak ones. Since the whole point of registering domain beliefs as SOFT is that
    evidence can win, the weight has to be denominated in units of evidence.

    Ignored for hard constraints.
    """

    constraint: ConstraintClass
    strength: Strength
    weight: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.strength is Strength.SOFT and self.weight <= 0.0:
            raise ValueError(
                f"soft constraint {self.constraint.value!r} needs a positive weight; "
                f"got {self.weight}"
            )

    @property
    def is_hard(self) -> bool:
        return self.strength is Strength.HARD


_DEFAULTS: tuple[ConstraintSpec, ...] = (
    # -- definitional: hard ---------------------------------------------------------
    ConstraintSpec(ConstraintClass.REFERENTIAL_INTEGRITY, Strength.HARD),
    ConstraintSpec(ConstraintClass.ATTRIBUTE_DOMAIN, Strength.HARD),
    ConstraintSpec(ConstraintClass.QUALIFIER_LEGALITY, Strength.HARD),
    ConstraintSpec(ConstraintClass.TYPE_DISJOINTNESS, Strength.HARD),
    # -- domain beliefs: soft -------------------------------------------------------
    #
    # Weights are multipliers on one claim's worth of evidence, so 1.0 means a single
    # confident claim roughly offsets one unit of violation. Cardinality sits slightly above
    # the others because a declared multiplicity is a stronger statement than an expected
    # ordering -- schemas are wrong about *how many* less often than processes deviate from
    # their intended sequence.
    ConstraintSpec(ConstraintClass.CARDINALITY, Strength.SOFT, weight=1.2),
    ConstraintSpec(ConstraintClass.LIFECYCLE_PRECEDENCE, Strength.SOFT, weight=1.0),
    ConstraintSpec(ConstraintClass.ATTRIBUTE_MONOTONICITY, Strength.SOFT, weight=1.0),
    ConstraintSpec(ConstraintClass.FUNCTIONAL_UNIQUENESS, Strength.SOFT, weight=1.0),
)


class ConstraintRegister:
    """Per-constraint-class strength and weight, with decision-6 defaults."""

    def __init__(self, specs: tuple[ConstraintSpec, ...] = _DEFAULTS) -> None:
        self._specs: dict[ConstraintClass, ConstraintSpec] = {s.constraint: s for s in specs}
        missing = set(ConstraintClass) - set(self._specs)
        if missing:
            raise ValueError(f"constraint register is missing: {sorted(c.value for c in missing)}")

    def __getitem__(self, constraint: ConstraintClass) -> ConstraintSpec:
        return self._specs[constraint]

    def is_hard(self, constraint: ConstraintClass) -> bool:
        """Whether this class is a hard zero factor rather than a finite penalty."""
        return self._specs[constraint].is_hard

    def is_enabled(self, constraint: ConstraintClass) -> bool:
        return self._specs[constraint].enabled

    def weight(self, constraint: ConstraintClass) -> float:
        """Penalty multiplier in units of one claim's evidence. Ignored when hard."""
        return self._specs[constraint].weight

    def override(
        self,
        constraint: ConstraintClass,
        *,
        strength: Strength | None = None,
        weight: float | None = None,
        enabled: bool | None = None,
    ) -> ConstraintRegister:
        """Return a new register with one class changed.

        Relaxing a definitional constraint to ``SOFT`` is permitted -- it is how the
        section 8.3 sensitivity experiment is run -- but it means posterior samples may no
        longer be valid OCEL logs, so callers get a loud name rather than a silent flag.
        """
        current = self._specs[constraint]
        updated = replace(
            current,
            strength=strength if strength is not None else current.strength,
            weight=weight if weight is not None else (1.0 if strength is Strength.SOFT and current.weight == 0.0 else current.weight),
            enabled=enabled if enabled is not None else current.enabled,
        )
        specs = dict(self._specs)
        specs[constraint] = updated
        return ConstraintRegister(tuple(specs.values()))

    @property
    def guarantees_valid_samples(self) -> bool:
        """Whether every definitional constraint is still hard and enabled.

        Design doc section 6.4 promises that posterior samples are valid OCEL 2.0 logs.
        That promise holds exactly when this is ``True``.
        """
        definitional = (
            ConstraintClass.REFERENTIAL_INTEGRITY,
            ConstraintClass.ATTRIBUTE_DOMAIN,
            ConstraintClass.QUALIFIER_LEGALITY,
            ConstraintClass.TYPE_DISJOINTNESS,
        )
        return all(self._specs[c].is_hard and self._specs[c].enabled for c in definitional)

    def __repr__(self) -> str:
        hard = sum(1 for s in self._specs.values() if s.is_hard and s.enabled)
        soft = sum(1 for s in self._specs.values() if not s.is_hard and s.enabled)
        off = sum(1 for s in self._specs.values() if not s.enabled)
        return f"ConstraintRegister(hard={hard}, soft={soft}, disabled={off})"
