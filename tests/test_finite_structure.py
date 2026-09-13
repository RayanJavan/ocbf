"""Canonical finite support invariants against independent reference cases."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from ocbf.api import compile_model, infer
from ocbf.assertions import AssertionRef as Ref
from ocbf.errors import CapabilityError, IncompatibleModel, ValidationError
from ocbf.evidence import InterpretedEvidence, materialize
from ocbf.inference.contracts import InferencePolicy
from ocbf.model.compile import structural_variables
from ocbf.model.spec import FactorSpec, LifecycleBinding, ModelSpec
from ocbf.reliability.config import ParameterSet
from ocbf.schema import E2OQualifier, EventType, Multiplicity, O2OQualifier, ObjectType, Schema
from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext

POLICY = InferencePolicy(engine="reference_elimination")
NOW = datetime(2026, 9, 9, tzinfo=UTC)


def specification(universe, *, role="support", factors=(), bindings=()):
    context = SemanticContext.from_universe(universe)
    priors = {
        v.key: tuple([1 / len(v.domain)] * len(v.domain))
        for v in structural_variables(context)
        if len(v.domain) > 1 and not v.key.startswith("event_type(")
    }
    parameters = ParameterSet(priors=priors, assumptions={"cardinality_role": role})
    evidence = InterpretedEvidence(materialize([], as_of=NOW), ())
    return ModelSpec(context, evidence, parameters, factors=factors, lifecycle_bindings=bindings)


def force(key, value=True):
    return FactorSpec(
        "force:" + key, (key,), log_values=[-np.inf, 0] if value else [0, -np.inf], role="support"
    )


def test_cardinality_counts_each_target_type_separately():
    schema = Schema(
        [EventType("started")],
        [ObjectType("Job"), ObjectType("Operation")],
        e2o=[
            E2OQualifier("subject", "started", t, Multiplicity(1, 1)) for t in ("Job", "Operation")
        ],
    )
    universe = (
        UniverseBuilder(schema)
        .add_event("e")
        .add_object("j", "Job")
        .add_object("op", "Operation")
        .build()
    )
    exists = str(Ref.event_exists("e"))
    result = infer(compile_model(specification(universe, factors=(force(exists),))), policy=POLICY)
    for obj in ("j", "op"):
        assert result.posterior.marginal(str(Ref.e2o("e", "subject", obj))).probabilities[1] == 1
    # A required typed relation with no candidate is a contradiction when the event occurs.
    missing = UniverseBuilder(schema).add_event("e").add_object("j", "Job").build()
    with pytest.raises(IncompatibleModel):
        infer(compile_model(specification(missing, factors=(force(exists),))), policy=POLICY)


def test_normative_cardinality_does_not_change_support_and_classification_is_required():
    schema = Schema(
        [EventType("e")],
        [ObjectType("Object")],
        e2o=[E2OQualifier("subject", "e", "Object", Multiplicity(1, 1))],
    )
    universe = UniverseBuilder(schema).add_event("e").build()
    spec = specification(universe, role="normative", factors=(force(str(Ref.event_exists("e"))),))
    assert (
        infer(compile_model(spec), policy=POLICY)
        .posterior.marginal(str(Ref.event_exists("e")))
        .probabilities[1]
        == 1
    )
    with pytest.raises(ValidationError):
        compile_model(replace(spec, parameters=replace(spec.parameters, assumptions={})))


def test_typed_o2o_counts_do_not_conflate_same_qualifier():
    schema = Schema(
        [],
        [ObjectType(t) for t in ("Operation", "Job", "Station")],
        o2o=[
            O2OQualifier("context", "Operation", t, Multiplicity(1, 1)) for t in ("Job", "Station")
        ],
    )
    builder = (
        UniverseBuilder(schema)
        .add_object("op", "Operation")
        .add_object("job", "Job")
        .add_object("s", "Station")
    )
    builder.add_o2o_candidate("op", "context", "job").add_o2o_candidate("op", "context", "s")
    result = infer(compile_model(specification(builder.build())), policy=POLICY)
    for target in ("job", "s"):
        assert (
            result.posterior.marginal(str(Ref.o2o("op", "context", target))).probabilities[1] == 1
        )


def test_lifecycle_is_bound_to_one_operation_not_all_job_events():
    schema = Schema(
        [EventType("start"), EventType("end")],
        [ObjectType("Operation")],
        e2o=[E2OQualifier("subject", kind, "Operation") for kind in ("start", "end")],
    )
    builder = UniverseBuilder(schema).add_object("op1", "Operation").add_object("op2", "Operation")
    for event, kind in (("a", "start"), ("b", "end"), ("c", "start"), ("d", "end")):
        builder.add_event(event, type_support={kind})
    bindings = (
        LifecycleBinding("op1", "a", "b", "subject", NOW, NOW + timedelta(hours=1), role="support"),
        LifecycleBinding(
            "op2",
            "c",
            "d",
            "subject",
            NOW - timedelta(hours=3),
            NOW - timedelta(hours=2),
            role="support",
        ),
    )
    model = compile_model(specification(builder.build(), bindings=bindings))
    precedence = [f for f in model.factors if f.family == "precedence"]
    assert len(precedence) == 2
    assert all(model.log_value(f, (1, 1)) == 0 for f in precedence)
    reversed_binding = replace(bindings[0], after=NOW - timedelta(minutes=1))
    model = compile_model(specification(builder.build(), bindings=(reversed_binding,)))
    factor = next(f for f in model.factors if f.family == "precedence")
    assert model.log_value(factor, (1, 1)) == -np.inf
    assert model.log_value(factor, (1, 0)) == 0  # not applicable to unassociated endpoints


def test_uncertain_type_never_gives_illegal_qualifier_mass():
    schema = Schema(
        [EventType("a"), EventType("b")],
        [ObjectType("Object")],
        e2o=[E2OQualifier("subject", "a", "Object")],
    )
    universe = UniverseBuilder(schema).add_event("e").add_object("o", "Object").build()
    link, kind = str(Ref.e2o("e", "subject", "o")), str(Ref.event_type("e"))
    model = compile_model(specification(universe, role="normative"))
    table = infer(model, policy=POLICY).posterior.joint((link, kind))
    assert table.probabilities[1, table.domains[1].index("b")] == 0
    assert table.probabilities[1, table.domains[1].index("__inactive__")] == 0
    with pytest.raises(ValidationError):
        compile_model(
            replace(
                specification(universe),
                ground_structure=False,
                variables=structural_variables(SemanticContext.from_universe(universe)),
            )
        )


def test_identity_codec_preserves_supported_assertions_and_missing_time():
    from ocbf.io import dumps, loads
    from ocbf.model.decoding import decode_assignment

    schema = Schema(
        [EventType("start")],
        [ObjectType("Operation")],
        e2o=[E2OQualifier("subject", "start", "Operation")],
    )
    universe = UniverseBuilder(schema).add_event("e").add_object("op", "Operation").build()
    model = compile_model(specification(universe, role="normative"))
    state = {key: domain[-1] for key, domain in model.domains.items()}
    history = loads(dumps(decode_assignment(model, state)))
    assert history.events["e"]["type"] == "start"
    assert history.events["e"]["time"] is None
    assert history.objects["op"] == "Operation"
    assert history.e2o == (("e", "subject", "op"),)
    state[str(Ref.event_exists("e"))] = False
    with pytest.raises(ValidationError):
        decode_assignment(model, state)


def test_extra_semantic_variable_cannot_bypass_declared_support():
    from ocbf.model.spec import VariableSpec

    schema = Schema([EventType("event")], [])
    universe = UniverseBuilder(schema).add_event("e").build()
    spec = specification(universe)
    variables = (
        *structural_variables(spec.context),
        VariableSpec(str(Ref.event_exists("undeclared")), (False, True)),
    )
    with pytest.raises(CapabilityError):
        compile_model(replace(spec, variables=variables))
