"""Independent numerical references and lifecycle/ownership checks for the new workflow."""

import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy.special import logsumexp

from ocbf.api import compile_model, evaluate, infer, requirements_for
from ocbf.belief.posterior import QueryRequirements
from ocbf.errors import (
    BudgetExceeded,
    CapabilityError,
    EvidenceConflict,
    IncompatibleModel,
    ValidationError,
)
from ocbf.evidence import (
    EvidenceAction,
    EvidenceRecord,
    InterpretedEvidence,
    Observation,
    materialize,
)
from ocbf.inference.contracts import InferencePolicy
from ocbf.model.spec import FactorSpec, ModelSpec, VariableSpec
from ocbf.queries import ConditionInterval, Endpoint, ExecutionProjection, QueryBundle, QuerySpec
from ocbf.reliability.config import ChannelValues, ParameterSet, TrustRule, resolve_parameters
from ocbf.schema import Schema
from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext

NOW = datetime(2026, 9, 9, tzinfo=UTC)
REFERENCE = InferencePolicy(engine="reference_elimination")


def context():
    return SemanticContext.from_universe(UniverseBuilder(Schema([], [])).build())


def record(key="r", value=True, **kwargs):
    return EvidenceRecord(
        key,
        kwargs.pop("revision_id", key),
        "s",
        "1",
        {"value": value},
        kwargs.pop("known_at", NOW),
        **kwargs,
    )


def evidence(records=(), observations=()):
    return InterpretedEvidence(materialize(records, as_of=NOW + timedelta(days=1)), observations)


def toy(*, factors=(), observations=(), records=(), params=None, domains=None):
    domains = domains or {"latent:a": (False, True), "latent:b": (False, True)}
    variables = tuple(VariableSpec(k, v) for k, v in domains.items())
    params = params or ParameterSet(
        priors={k: tuple([1 / len(v)] * len(v)) for k, v in domains.items()}
    )
    return ModelSpec(
        context(),
        evidence(records, observations),
        params,
        variables,
        tuple(factors),
        ground_structure=False,
    )


def brute(model):
    domains = model.domains
    scope = tuple(domains)
    values = np.empty(tuple(map(len, domains.values())))
    for indices in np.ndindex(values.shape):
        state = dict(zip(scope, indices, strict=True))
        values[indices] = sum(
            model.log_value(f, tuple(state[k] for k in f.scope)) for f in model.factors
        )
    z = logsumexp(values)
    return np.exp(values - z), z


def test_owned_values_identity_and_static_semantics():
    payload = {"value": [1, 2]}
    r = EvidenceRecord("r", "r", "s", "1", payload, NOW)
    payload["value"].append(3)
    assert r.payload["value"] == (1, 2)
    with pytest.raises(TypeError):
        r.payload["x"] = 2
    with pytest.raises(ValidationError):
        record(known_at=datetime(2026, 9, 9))  # noqa: DTZ001 -- intentionally rejected naive clock
    static = record(known_at=None)
    with pytest.raises(ValidationError):
        materialize([static], as_of=NOW)
    assert materialize([static], as_of=None, policy="static-v1").effective == (static,)


def test_revisions_replay_duplicates_conflicts_late_arrival():
    first = record()
    corrected = replace(
        first,
        revision_id="r2",
        previous_revision="r",
        action=EvidenceAction.REPLACE,
        known_at=NOW + timedelta(hours=1),
        payload={"value": False},
    )
    removed = replace(
        corrected,
        revision_id="r3",
        previous_revision="r2",
        action=EvidenceAction.RETRACT,
        known_at=NOW + timedelta(hours=2),
    )
    initial = materialize([removed, corrected, first, first], as_of=NOW)
    assert initial.effective == (first,)
    assert initial.snapshot_id == materialize([first], as_of=NOW).snapshot_id
    later = materialize([removed, first, corrected], as_of=NOW + timedelta(hours=1))
    assert later.effective == (corrected,)
    assert materialize([first, removed, corrected], as_of=NOW + timedelta(hours=3)).effective == ()
    with pytest.raises(EvidenceConflict):
        materialize([first, replace(first, payload={"different": True})], as_of=NOW)
    with pytest.raises(EvidenceConflict):
        materialize(
            [first, corrected, replace(corrected, revision_id="fork")],
            as_of=NOW + timedelta(days=1),
        )


def test_manual_resolution_precedence_no_clipping_and_ambiguity():
    observation = Observation(
        "o", "s", "exists", "1", "binary", ("latent:a",), True, ("r",), "i@1", "g"
    )
    exact = TrustRule(ChannelValues(0.999, 0.001), "manual exact", "s", "exists", "binary", "1")
    family = TrustRule(ChannelValues(0.6, 0.4), "manual family", family="exists")
    params = resolve_parameters([observation], rules=[family, exact])
    assert params.for_observation(observation).sensitivity == 0.999
    with pytest.raises(ValidationError):
        resolve_parameters([observation], rules=[family, family])
    with pytest.raises(ValidationError):
        resolve_parameters([observation])


@pytest.mark.parametrize("engine", ["reference_elimination", "gtsam_exact"])
def test_exact_joint_and_offsets_against_independent_enumeration(engine):
    if engine == "gtsam_exact":
        from ocbf.backends import gtsam_backend

        if not gtsam_backend().available:
            pytest.skip("optional GTSAM unavailable")
    f = FactorSpec(
        "couple",
        ("latent:b", "latent:a"),
        log_values=np.log([[2.0, 1.0], [3.0, 9.0]]),
        log_offset=13.0,
    )
    model = compile_model(toy(factors=[f]))
    result = infer(
        model,
        policy=InferencePolicy(engine=engine),
        requirements=QueryRequirements((("latent:a", "latent:b"),), ("joint", "normalizer")),
    )
    p, z = brute(model)
    assert np.allclose(
        result.posterior.joint(("latent:a", "latent:b")).probabilities, p, atol=1e-11
    )
    assert result.posterior.log_normalizer == pytest.approx(z)
    assert np.allclose(result.posterior.marginal("latent:b").probabilities, p.sum(axis=0))
    with pytest.raises(ValueError):
        result.posterior.joint(("latent:a",)).probabilities.setflags(write=True)


def test_no_uniform_fallback_for_contradictions_or_empty_support():
    a = FactorSpec("a", ("latent:a",), log_values=[0.0, -np.inf], role="support")
    b = FactorSpec("b", ("latent:a",), log_values=[-np.inf, 0.0], role="support")
    with pytest.raises(IncompatibleModel):
        infer(compile_model(toy(factors=[a, b])), policy=REFERENCE)
    with pytest.raises(IncompatibleModel):
        compile_model(toy(factors=[replace(a, log_values=[-np.inf, -np.inf])]))


def test_two_level_categorical_channel_and_duplicate_information():
    obs = Observation("o", "s", "type", "1", "categorical", ("latent:a",), "B", ("r",), "i@1", "g")
    second = replace(obs, observation_id="copy", evidence_ids=("copy",))
    params = resolve_parameters(
        [obs],
        default=TrustRule(ChannelValues(hit_rate=0.9), "manual"),
        priors={"latent:a": (0.5, 0.5)},
    )
    spec = toy(
        observations=(obs,), records=(record(),), params=params, domains={"latent:a": ("A", "B")}
    )
    p = infer(compile_model(spec), policy=REFERENCE).posterior.marginal("latent:a").probabilities
    copied = replace(spec, evidence=evidence((record(), record("copy")), (obs, second)))
    q = infer(compile_model(copied), policy=REFERENCE).posterior.marginal("latent:a").probabilities
    assert p == pytest.approx([0.1, 0.9])
    assert q == pytest.approx(p)
    with pytest.raises(CapabilityError):
        compile_model(
            replace(
                copied,
                evidence=evidence((record(), record("copy")), (obs, replace(second, value="A"))),
            )
        )


def test_preflight_rejects_before_kernel_evaluation():
    from ocbf.model.kernels import builtin_kernels

    class Exploding:
        version = "1"

        def log_value(self, *args):
            raise AssertionError("allocated/evaluated before budget gate")

    kernels = builtin_kernels()
    kernels["explode"] = Exploding()
    factor = FactorSpec("huge", ("latent:a", "latent:b"), family="explode")
    model = compile_model(toy(factors=[factor]), factors=kernels)
    with pytest.raises(BudgetExceeded):
        infer(model, policy=replace(REFERENCE, max_table_states=2))


def test_model_input_order_identity_and_reference_separation():
    spec = toy()
    first = compile_model(spec)
    second = compile_model(replace(spec, variables=tuple(reversed(spec.variables))))
    assert first.model_id == second.model_id
    changed = compile_model(
        replace(
            spec, parameters=ParameterSet(priors={"latent:a": (0.1, 0.9), "latent:b": (0.5, 0.5)})
        )
    )
    assert first.model_id != changed.model_id


def duration_query(threshold=30, *, kind="duration_exception", conditions=(), coverage=False):
    start = NOW - timedelta(hours=2)
    projection = ExecutionProjection(
        "op",
        "job",
        (Endpoint("latent:a", start),),
        (Endpoint("latent:b", start + timedelta(hours=1)),),
    )
    q = QuerySpec(
        "q",
        kind,
        (projection,),
        NOW - timedelta(days=1),
        NOW,
        {"threshold_minutes": {"op": threshold}, "origin": "synthetic test reference"},
        conditions,
        coverage,
    )
    return QueryBundle((q,))


def test_joint_queries_denominators_count_exposure_and_reference_reuse():
    f = FactorSpec(
        "equal", ("latent:a", "latent:b"), log_values=[[0, -np.inf], [-np.inf, 0]], role="support"
    )
    model = compile_model(toy(factors=[f]))
    queries = duration_query()
    result = infer(model, requirements=requirements_for(queries), policy=REFERENCE)
    estimate = evaluate(result, queries).estimates[0]
    assert estimate.denominator == pytest.approx(
        0.5
    )  # product of marginals would incorrectly be .25
    assert estimate.value == 1
    assert estimate.outcomes["inapplicable"] == pytest.approx(0.5)
    changed = duration_query(90)
    assert changed.bundle_id != queries.bundle_id
    assert evaluate(result, changed).estimates[0].value == 0
    assert evaluate(result, duration_query(kind="expected_exception_count")).estimates[
        0
    ].value == pytest.approx(0.5)
    intervals = (
        ConditionInterval(NOW - timedelta(hours=2), NOW - timedelta(minutes=90)),
        ConditionInterval(NOW - timedelta(minutes=105), NOW - timedelta(hours=1)),
    )
    exposed = evaluate(
        result, duration_query(kind="exposure", conditions=intervals, coverage=True)
    ).estimates[0]
    assert exposed.value == pytest.approx(60)  # union, not 75 minutes of double counting
    assert evaluate(result, duration_query(kind="exposure")).estimates[0].value is None


def test_missing_endpoints_and_zero_applicability_are_unresolved_not_zero():
    queries = duration_query()
    q = queries.queries[0]
    missing = replace(q, executions=(replace(q.executions[0], ends=()),))
    result = infer(compile_model(toy()), policy=REFERENCE)
    answer = evaluate(result, QueryBundle((missing,))).estimates[0]
    assert answer.value is None and answer.outcomes["unresolved"] == pytest.approx(0.5)
    f = FactorSpec("absent", ("latent:a",), log_values=[0, -np.inf])
    result = infer(compile_model(toy(factors=[f])), policy=REFERENCE)
    answer = evaluate(result, queries).estimates[0]
    assert answer.value is None and answer.denominator == 0


def test_bp_is_qualified_and_cannot_answer_joint_query():
    model = compile_model(toy())
    with pytest.raises(CapabilityError):
        infer(model, policy=InferencePolicy(engine="bp"))
    result = infer(model, policy=InferencePolicy(engine="bp", allow_approximate=True))
    assert result.posterior.marginal("latent:a").probabilities == pytest.approx([0.5, 0.5])
    with pytest.raises(CapabilityError):
        evaluate(result, duration_query())


def test_contract_imports_do_not_load_optional_backends():
    code = """
import sys
import ocbf.evidence, ocbf.reliability.config, ocbf.belief.posterior, ocbf.queries.expressions
assert not any(name in sys.modules for name in ('gtsam','torch','pymc','arviz'))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_clamped_boolean_report_keeps_its_likelihood_constant():
    observation = Observation(
        "o", "s", "exists", "1", "binary", ("known",), True, ("r",), "i@1", "g"
    )
    parameters = resolve_parameters(
        [observation], default=TrustRule(ChannelValues(0.8, 0.2), "manual")
    )
    model = compile_model(
        toy(
            observations=(observation,),
            records=(record(),),
            params=parameters,
            domains={"known": (True,)},
        )
    )
    result = infer(model, policy=REFERENCE)
    assert result.posterior.marginal("known").probabilities == pytest.approx([1])
    assert result.posterior.log_normalizer == pytest.approx(np.log(0.8))


def test_horizon_boundary_and_absent_candidates_remain_distinct():
    result = infer(compile_model(toy()), policy=REFERENCE)
    query = duration_query().queries[0]
    execution = query.executions[0]
    missing = replace(execution, starts=())
    answer = evaluate(result, QueryBundle((replace(query, executions=(missing,)),))).estimates[0]
    assert answer.value is None and answer.outcomes["unresolved"] == 1
    at_horizon = replace(execution, starts=(replace(execution.starts[0], time=query.horizon),))
    answer = evaluate(result, QueryBundle((replace(query, executions=(at_horizon,)),))).estimates[0]
    assert answer.value is None and answer.outcomes["inapplicable"] == 1
    beyond = replace(
        execution, ends=(replace(execution.ends[0], time=query.horizon + timedelta(seconds=1)),)
    )
    answer = evaluate(result, QueryBundle((replace(query, executions=(beyond,)),))).estimates[0]
    assert answer.value is None and answer.outcomes["pending"] > 0


def test_closure_supersedes_open_payload_without_becoming_second_report():
    first = record()
    closed = replace(
        first,
        revision_id="closed",
        previous_revision=first.revision_id,
        action=EvidenceAction.CLOSE,
        payload={"value": True, "closed": True},
    )
    snapshot = materialize([closed, first], as_of=NOW)
    assert snapshot.effective == (closed,)
    assert snapshot.issues[0].status == "superseded"
