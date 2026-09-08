"""Schema, assertions, universe, claims: the contracts the rest of the system assumes."""

from __future__ import annotations

import numpy as np
import pytest

from ocbf.assertions import AssertionRef, Family, VariableRegistry, VarKind
from ocbf.schema import (
    AttributeKind,
    AttributeSpec,
    ConstraintClass,
    ConstraintRegister,
    E2OQualifier,
    EventType,
    Lifecycle,
    ObjectType,
    O2OQualifier,
    Schema,
    Strength,
)
from ocbf.schema.core import MANY, ONE
from ocbf.sources import (
    ChannelFamily,
    Claim,
    ClaimSet,
    CoverageSemantics,
    SourceProfile,
    StaticSource,
)
from ocbf.universe import UniverseBuilder


@pytest.fixture
def schema() -> Schema:
    return Schema(
        event_types=[EventType("A"), EventType("B")],
        object_types=[ObjectType("X"), ObjectType("Y")],
        e2o=[
            E2OQualifier("q1", "A", "X", ONE),
            E2OQualifier("q2", "B", "Y", MANY),
        ],
        o2o=[O2OQualifier("rel", "X", "Y", MANY)],
    )


# -- schema ------------------------------------------------------------------------


def test_schema_rejects_unknown_type_in_qualifier():
    with pytest.raises(ValueError, match="unknown event type"):
        Schema(
            event_types=[EventType("A")],
            object_types=[ObjectType("X")],
            e2o=[E2OQualifier("q", "MISSING", "X")],
        )


def test_schema_enforces_disjoint_attribute_names():
    """OCEL 2.0 requires an attribute name to determine its owning type."""
    spec = AttributeSpec("shared", AttributeKind.CONTINUOUS)
    with pytest.raises(ValueError, match="disjoint attribute sets"):
        Schema(
            event_types=[EventType("A", (spec,))],
            object_types=[ObjectType("X", (spec,))],
        )


def test_attribute_spec_requires_levels_for_categorical():
    with pytest.raises(ValueError, match="requires levels"):
        AttributeSpec("colour", AttributeKind.CATEGORICAL)


def test_categorical_is_outside_the_copula():
    """The copula boundary is a documented limitation; keep it asserted, not assumed."""
    assert not AttributeKind.CATEGORICAL.in_copula
    assert AttributeKind.ORDINAL.in_copula
    assert AttributeKind.COUNT.in_copula


def test_legal_e2o_respects_latent_type_support(schema: Schema):
    # 'q1' is legal only for event type A, so a support of {B} must not admit it.
    assert schema.legal_e2o_for({"A"}, "X") == {"q1"}
    assert schema.legal_e2o_for({"B"}, "X") == frozenset()
    assert schema.legal_e2o_for({"A", "B"}, "X") == {"q1"}


# -- constraint register ------------------------------------------------------------


def test_default_register_matches_decision_six():
    reg = ConstraintRegister()
    assert reg.is_hard(ConstraintClass.REFERENTIAL_INTEGRITY)
    assert reg.is_hard(ConstraintClass.ATTRIBUTE_DOMAIN)
    assert not reg.is_hard(ConstraintClass.CARDINALITY)
    assert not reg.is_hard(ConstraintClass.LIFECYCLE_PRECEDENCE)
    assert reg.guarantees_valid_samples


def test_relaxing_a_definitional_constraint_voids_the_sample_guarantee():
    """Design doc section 6.4 promises valid OCEL samples only while these stay hard."""
    reg = ConstraintRegister().override(
        ConstraintClass.REFERENTIAL_INTEGRITY, strength=Strength.SOFT, weight=5.0
    )
    assert not reg.guarantees_valid_samples


def test_soft_constraint_requires_positive_weight():
    with pytest.raises(ValueError, match="positive weight"):
        ConstraintRegister().override(ConstraintClass.CARDINALITY, weight=0.0)


# -- assertion refs -----------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        AssertionRef.event_exists("e1"),
        AssertionRef.event_type("e1"),
        AssertionRef.event_time("e1"),
        AssertionRef.object_exists("o1"),
        AssertionRef.e2o("e1", "q", "o1"),
        AssertionRef.o2o("o1", "rel", "o2"),
        AssertionRef.event_attr("e1", "amount"),
        AssertionRef.object_attr("o1", "status"),
    ],
)
def test_ref_string_round_trip(ref: AssertionRef):
    assert AssertionRef.parse(str(ref)) == ref


def test_link_ref_requires_qualifier_and_target():
    with pytest.raises(ValueError, match="requires qualifier and target"):
        AssertionRef(Family.E2O, "e1")


def test_non_link_ref_rejects_qualifier():
    with pytest.raises(ValueError, match="must not carry qualifier"):
        AssertionRef(Family.EVENT_EXISTS, "e1", qualifier="q", target="o")


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        AssertionRef.parse("not a ref")


# -- registry -----------------------------------------------------------------------


def test_registry_groups_families_contiguously():
    reg = VariableRegistry()
    reg.add_binary(AssertionRef.e2o("e1", "q", "o1"))
    reg.add_binary(AssertionRef.event_exists("e1"))
    reg.add_binary(AssertionRef.e2o("e2", "q", "o1"))
    reg.add_binary(AssertionRef.event_exists("e2"))
    reg.freeze()

    for family in (Family.EVENT_EXISTS, Family.E2O):
        s = reg.family_slice(family)
        assert s.stop - s.start == 2
        for i in range(s.start, s.stop):
            assert reg.family_of(i) is family


def test_registry_rejects_conflicting_reregistration():
    reg = VariableRegistry()
    ref = AssertionRef.event_type("e1")
    reg.add_categorical(ref, 3)
    with pytest.raises(ValueError, match="cannot re-register"):
        reg.add_categorical(ref, 4)


def test_registry_is_immutable_once_frozen():
    reg = VariableRegistry()
    reg.add_binary(AssertionRef.event_exists("e1"))
    reg.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        reg.add_binary(AssertionRef.event_exists("e2"))


def test_get_returns_none_for_unregistered():
    """Pruned assertions answer with the prior, not a KeyError."""
    reg = VariableRegistry().freeze()
    assert reg.get(AssertionRef.event_exists("nope")) is None


# -- universe -----------------------------------------------------------------------


def test_signature_prune_removes_illegal_pairs(schema: Schema):
    b = UniverseBuilder(schema)
    b.add_event("e1", type_support=["A"])
    b.add_object("x1", "X")
    b.add_object("y1", "Y")
    u = b.build()
    # Only (e1, q1, x1) is legal: q2 needs event type B, and q1 needs object type X.
    assert u.e2o_candidates == (("e1", "q1", "x1"),)


def test_latent_type_support_keeps_candidates_alive(schema: Schema):
    """A candidate survives if *some* type in the support permits it."""
    b = UniverseBuilder(schema)
    b.add_event("e1", type_support=["A", "B"])
    b.add_object("x1", "X")
    b.add_object("y1", "Y")
    u = b.build()
    assert set(u.e2o_candidates) == {("e1", "q1", "x1"), ("e1", "q2", "y1")}


def test_temporal_prune_drops_non_overlapping(schema: Schema):
    b = UniverseBuilder(schema)
    b.add_event("early", type_support=["A"], time_lo=0.0, time_hi=1.0)
    b.add_event("late", type_support=["A"], time_lo=100.0, time_hi=101.0)
    b.add_object("x1", "X", alive_from=0.0, alive_to=10.0)
    u = b.build()
    assert u.e2o_candidates == (("early", "q1", "x1"),)


def test_prune_report_records_the_reduction(schema: Schema):
    b = UniverseBuilder(schema)
    for i in range(5):
        b.add_event(f"e{i}", type_support=["A"])
    b.add_object("x1", "X")
    b.add_object("y1", "Y")
    u = b.build()
    assert u.prune_report.e2o_naive > u.prune_report.e2o_after_temporal
    assert 0.0 < u.prune_report.e2o_reduction < 1.0


def test_determined_event_type_creates_no_variable(schema: Schema):
    """A singleton type support is clamped, so it needs no categorical variable."""
    b = UniverseBuilder(schema)
    b.add_event("e1", type_support=["A"])
    b.add_object("x1", "X")
    u = b.build()
    assert u.registry.get(AssertionRef.event_type("e1")) is None


def test_o2o_candidates_are_not_auto_generated(schema: Schema):
    """The object-object space is quadratic with no temporal prune; callers must supply it."""
    b = UniverseBuilder(schema)
    b.add_object("x1", "X")
    b.add_object("y1", "Y")
    assert b.build().o2o_candidates == ()


def test_active_refs_expands_by_entity_locality(schema: Schema):
    b = UniverseBuilder(schema)
    b.add_event("e1", type_support=["A", "B"])
    b.add_object("x1", "X")
    b.add_object("y1", "Y")
    u = b.build()

    seed = AssertionRef.e2o("e1", "q1", "x1")
    zero = u.active_refs([seed], hops=0)
    one = u.active_refs([seed], hops=1)
    assert zero == {seed}
    assert seed in one and len(one) > len(zero)
    # Everything reached shares an entity with the seed.
    assert all(set(r.touches) & {"e1", "x1"} for r in one)


# -- sources and claims -------------------------------------------------------------


def _profile(sid: str, coverage: CoverageSemantics, cluster: str = "c0") -> SourceProfile:
    return SourceProfile(sid, coverage, ChannelFamily.BINARY, cluster)


def test_non_opportunistic_source_needs_silence_to_interpret():
    """Declaring detector semantics with no silence would fabricate certainty."""
    ref = AssertionRef.e2o("e1", "q", "o1")
    with pytest.raises(ValueError, match="no silence to interpret"):
        StaticSource(
            _profile("s1", CoverageSemantics.COMPLETE_OVER_SCOPE),
            [Claim("s1", ref, True)],
        )


def test_source_rejects_claims_outside_its_scope():
    a, b = AssertionRef.event_exists("e1"), AssertionRef.event_exists("e2")
    with pytest.raises(ValueError, match="outside its declared scope"):
        StaticSource(_profile("s1", CoverageSemantics.OPPORTUNISTIC), [Claim("s1", a, True)], [b])


def test_source_rejects_foreign_claims():
    ref = AssertionRef.event_exists("e1")
    with pytest.raises(ValueError, match="claims from"):
        StaticSource(_profile("s1", CoverageSemantics.OPPORTUNISTIC), [Claim("other", ref, True)])


def test_silent_refs_is_scope_minus_claims():
    a, b = AssertionRef.event_exists("e1"), AssertionRef.event_exists("e2")
    src = StaticSource(
        _profile("s1", CoverageSemantics.COMPLETE_OVER_SCOPE), [Claim("s1", a, True)], [a, b]
    )
    assert src.silent_refs() == {b}


def test_coverage_silence_semantics():
    assert not CoverageSemantics.OPPORTUNISTIC.silence_is_evidence
    assert CoverageSemantics.COMPLETE_OVER_SCOPE.silence_is_evidence
    assert CoverageSemantics.SELECTIVE.silence_is_evidence


def test_claim_requires_a_value():
    with pytest.raises(ValueError, match="carries no value"):
        Claim("s1", AssertionRef.event_exists("e1"))


def test_degree_counts_distinct_sources_not_claims():
    """Two claims from one source are not two votes."""
    ref = AssertionRef.event_exists("e1")
    cs = ClaimSet([Claim("s1", ref, True), Claim("s1", ref, True), Claim("s2", ref, False)])
    assert cs.deg_assertion(ref) == 2


def test_channel_family_defaults_by_assertion_family():
    assert ChannelFamily.for_family(Family.EVENT_TYPE) is ChannelFamily.CATEGORICAL
    assert ChannelFamily.for_family(Family.EVENT_TIME) is ChannelFamily.CONTINUOUS
    assert ChannelFamily.for_family(Family.E2O) is ChannelFamily.BINARY
