"""Independent scientific and architectural references for the bounded Joint-inference routes."""

import ast
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from ocbf.api import (
    compile_model,
    evaluate,
    infer,
    plan_inference,
    prepare_evidence,
    requirements_for,
)
from ocbf.belief.posterior import QueryRequirements
from ocbf.diagnostics.monte_carlo import assessment
from ocbf.errors import BudgetExceeded, CapabilityError, ValidationError
from ocbf.evidence import EvidenceRecord, InterpretedEvidence
from ocbf.evidence.snapshots import materialize
from ocbf.inference.contracts import InferencePolicy, SamplingConfig
from ocbf.inference.sampling import metropolis_log_acceptance
from ocbf.io.bundle import dumps, loads
from ocbf.model.reductions import reduction_codec
from ocbf.model.spec import ContinuousVariableSpec, FactorSpec, ModelSpec, VariableSpec
from ocbf.queries import Endpoint, ExecutionProjection, QueryBundle, QuerySpec
from ocbf.reliability.config import ParameterSet, TrustRule, resolve_parameters
from ocbf.schema import Schema
from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext

AT = datetime(2026, 9, 8, 12, tzinfo=UTC)
JOINT = QueryRequirements((), (), (("joint",), ("joint_draws",)))


def toy(variables, factors=(), priors=None):
    context = SemanticContext.from_universe(UniverseBuilder(Schema([], [])).build())
    evidence = InterpretedEvidence(materialize((), as_of=AT), ())
    return ModelSpec(
        context,
        evidence,
        ParameterSet(priors=priors or {}),
        tuple(variables),
        tuple(factors),
        ground_structure=False,
    )


def draw_policy(**kwargs):
    return InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=4, warmup=100, draws=500, **kwargs)
    )


def scalar_hybrid(epoch=0):
    return toy(
        (
            VariableSpec("mode", ("a", "b")),
            ContinuousVariableSpec("time", epoch, 2, "seconds", "synthetic reference"),
        ),
        (
            FactorSpec(
                "report",
                ("mode", "time"),
                "linear_gaussian",
                {
                    "coefficients": {"time": 1.0},
                    "observed": epoch + 1,
                    "sd": 1.0,
                    "mode_key": "mode",
                    "by_mode": {"a": {"sd": 0.5}, "b": {"sd": 3.0}},
                },
            ),
        ),
        {"mode": (0.4, 0.6)},
    )


def test_gaussian_mode_constants_and_epoch_translation():
    for epoch in (0, 1_800_000_000):
        model = compile_model(scalar_hybrid(epoch))
        result = infer(model, requirements=JOINT, policy=InferencePolicy(engine="auto"))
        assert result.manifest["execution_plan"].engine == "reference_hybrid"
        components = result.posterior.conditional.components
        expected = np.array(
            [0.4 * norm.pdf(1, scale=np.sqrt(4 + 0.25)), 0.6 * norm.pdf(1, scale=np.sqrt(4 + 9))]
        )
        assert result.posterior.log_normalizer == pytest.approx(np.log(expected.sum()), abs=1e-10)
        assert np.exp(
            [c.log_mass - result.posterior.log_normalizer for c in components]
        ) == pytest.approx(expected / expected.sum())
        for c, sd in zip(components, (0.5, 3), strict=True):
            variance = 1 / (0.25 + 1 / sd**2)
            assert c.mean[0] == pytest.approx(epoch + variance / sd**2, abs=1e-7)
            assert c.covariance[0, 0] == pytest.approx(variance)
        samples = result.posterior.draw(("mode", "time"), rng=np.random.default_rng(3), size=20000)
        assert (samples.values["mode"] == "a").mean() == pytest.approx(
            expected[0] / expected.sum(), abs=0.015
        )
        restored = loads(dumps(result))
        assert restored.posterior.log_normalizer == result.posterior.log_normalizer


def test_inactive_coordinate_integrates_to_one():
    spec = toy(
        (
            VariableSpec("exists", (False, True)),
            ContinuousVariableSpec("time", 12, 3, "seconds", "manual", (("exists", True),)),
        ),
        priors={"exists": (0.3, 0.7)},
    )
    result = infer(
        compile_model(spec), requirements=JOINT, policy=InferencePolicy(engine="reference_hybrid")
    )
    assert result.posterior.log_normalizer == pytest.approx(0, abs=1e-12)
    assert np.exp([c.log_mass for c in result.posterior.conditional.components]) == pytest.approx(
        [0.3, 0.7]
    )


def test_exact_joint_draws_preserve_dependence_without_dense_query_table():
    variables = tuple(VariableSpec(k, (False, True)) for k in ("a", "b", "c"))
    factors = tuple(
        FactorSpec(k + "equal", ("a", k), log_values=[[0, -np.inf], [-np.inf, 0]], role="support")
        for k in ("b", "c")
    )
    model = compile_model(toy(variables, factors, {k: (0.5, 0.5) for k in ("a", "b", "c")}))
    policy = InferencePolicy(engine="auto", max_joint_states=2)
    result = infer(model, requirements=JOINT, policy=policy)
    with pytest.raises(BudgetExceeded):
        result.posterior.joint(("a", "b", "c"))
    draws = result.posterior.draw(("a", "b", "c"), rng=np.random.default_rng(2), size=20000)
    assert np.array_equal(draws.values["a"], draws.values["b"])
    assert np.array_equal(draws.values["a"], draws.values["c"])
    assert draws.values["a"].mean() == pytest.approx(0.5, abs=0.015)


def test_reduction_preserves_nonuniform_original_mass_and_decoding():
    spec = toy(
        (
            VariableSpec("a", (False, True)),
            VariableSpec("b", (False, True)),
            VariableSpec("fixed", (True,)),
        ),
        (FactorSpec("one", ("a", "b"), "count", {"lo": 1, "hi": 1}, role="support"),),
        {"a": (0.2, 0.8), "b": (0.7, 0.3)},
    )
    model = compile_model(spec)
    codec = reduction_codec(model)
    key = next(iter(codec.choices))
    states = [codec.expand({key: label}) for label in codec.domains[key]]
    logs = np.array([model.log_density(s) for s in states])
    assert np.exp(logs) == pytest.approx([0.56, 0.06])
    assert all(s["a"] != s["b"] and s["fixed"] for s in states)
    result = infer(model, requirements=JOINT, policy=draw_policy(), rng=np.random.default_rng(13))
    draws = result.posterior.draw(("a", "b"))
    assert np.all(draws.values["a"] != draws.values["b"])
    assert draws.values["a"].mean() == pytest.approx(0.56 / 0.62, abs=0.04)
    assert all(next(iter(t.values())) > 0 for t in result.diagnostics["mode_transitions"])
    assert (
        loads(dumps(result)).posterior.draw_set.draw_set_id == result.posterior.draw_set.draw_set_id
    )


def test_mh_asymmetric_transition_matrix_has_correct_stationary_target():
    target = np.array([0.1, 0.3, 0.6])
    proposal = np.array([[0.1, 0.6, 0.3], [0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
    transition = np.zeros((3, 3))
    for old in range(3):
        for new in range(3):
            if old != new:
                transition[old, new] = proposal[old, new] * np.exp(
                    metropolis_log_acceptance(
                        np.log(target[old]),
                        np.log(target[new]),
                        np.log(proposal[old, new]),
                        np.log(proposal[new, old]),
                    )
                )
        transition[old, old] = 1 - transition[old].sum()
    assert target @ transition == pytest.approx(target, abs=1e-14)
    assert metropolis_log_acceptance(0, -np.inf, 0, 0) == -np.inf


def test_constrained_continuous_sampling_matches_half_normal_difference():
    spec = toy(
        (
            ContinuousVariableSpec("start", 0, 1, "seconds", "synthetic"),
            ContinuousVariableSpec("end", 0, 1, "seconds", "synthetic"),
        ),
        (
            FactorSpec(
                "order",
                ("start", "end"),
                "temporal_order",
                {"before": "start", "after": "end"},
                role="support",
            ),
        ),
    )
    model = compile_model(spec)
    with pytest.raises(CapabilityError):
        infer(model, requirements=JOINT, policy=InferencePolicy(engine="reference_hybrid"))
    policy = draw_policy(proposal_scale=0.8, refresh_probability=0.3)
    result = infer(model, requirements=JOINT, policy=policy, rng=np.random.default_rng(18))
    draws = result.posterior.draw(("start", "end"))
    duration = draws.values["end"] - draws.values["start"]
    assert np.all(duration >= 0)
    assert duration.mean() == pytest.approx(2 / np.sqrt(np.pi), abs=0.12)
    assert np.mean(duration > 1) == pytest.approx(2 * norm.sf(1 / np.sqrt(2)), abs=0.06)


def comparison_spec():
    keys = ("start", "a_short", "a_long", "b_short", "b_long")
    variables = tuple(VariableSpec(k, (True,) if k == "start" else (False, True)) for k in keys)
    factors = (
        FactorSpec("a_one", ("a_short", "a_long"), "count", {"lo": 1, "hi": 1}, role="support"),
        FactorSpec("b_one", ("b_short", "b_long"), "count", {"lo": 1, "hi": 1}, role="support"),
        FactorSpec(
            "shared", ("a_long", "b_long"), log_values=[[0, -np.inf], [-np.inf, 0]], role="support"
        ),
    )
    spec = toy(variables, factors, {k: (0.5, 0.5) for k in keys if k != "start"})
    executions = tuple(
        ExecutionProjection(
            job,
            job,
            (Endpoint("start", AT),),
            (
                Endpoint(job + "_short", AT + timedelta(minutes=10)),
                Endpoint(job + "_long", AT + timedelta(minutes=60)),
            ),
        )
        for job in ("a", "b")
    )
    query = QuerySpec(
        "rank",
        "priority_distribution",
        executions,
        AT - timedelta(hours=1),
        AT + timedelta(hours=2),
        {"threshold_minutes": {"a": 30, "b": 30}},
    )
    whole = replace(query, name="job", kind="whole_job_conformance", executions=(executions[0],))
    return spec, QueryBundle((query, whole))


def test_queries_share_exact_and_sample_semantics_and_common_uncertainty():
    spec, queries = comparison_spec()
    model = compile_model(spec)
    exact = infer(
        model,
        requirements=requirements_for(queries),
        policy=InferencePolicy(engine="reference_elimination"),
    )
    expected = evaluate(exact, queries)
    ranking, job = expected.estimates
    assert job.value == pytest.approx(0.5)
    assert ranking.distribution["top_tie_probability"] == 1
    assert all(r["1"]["probability"] == 1 for r in ranking.distribution["ranks"].values())
    sampled = infer(
        model,
        requirements=requirements_for(queries),
        policy=draw_policy(),
        rng=np.random.default_rng(4),
    )
    answers = evaluate(sampled, queries)
    assert answers.estimates[1].value == pytest.approx(0.5, abs=0.05)
    assert answers.estimates[0].distribution["top_tie_probability"] == 1
    assert len({e.metadata["draw_set_id"] for e in answers.estimates}) == 1
    assert answers.estimates[1].numerical["mcse"] is not None


def test_diagnostics_detect_stuck_chains_and_do_not_certify_zero_rare_events():
    stuck = np.vstack([np.zeros(100), np.ones(100)])
    assert assessment(stuck)["status"] == "between_chain_disagreement"
    assert assessment(np.zeros((4, 100)))["mcse"] is None
    a = np.random.default_rng(33).binomial(1, 0.3, (4, 2000))
    b = np.maximum(a, np.random.default_rng(12).binomial(1, 0.7, a.shape))
    result = assessment(a, b, method="iid")
    influence = (a - a.mean() / b.mean() * b) / b.mean()
    assert result["mcse"] == pytest.approx(np.std(influence, ddof=1) / np.sqrt(a.size))


def test_rtIs_representations_and_shared_mode_prior():
    from examples.mammut_retrospective.rtls import ZoneBinding, ZoneInterpreter, wrapped_zone
    from ocbf.reliability.channels import AssociationParameters

    spec = toy(
        (VariableSpec("association", ("station_a", "station_b")),),
        priors={"association": (0.5, 0.5)},
    )
    binding = ZoneBinding(
        "tag1",
        "association",
        ("station_a", "station_b"),
        {"zone1": "station_a"},
        AT - timedelta(days=1),
        AT + timedelta(days=1),
        ("synthetic dossier",),
        "1",
        spec.context.context_id,
        True,
    )
    payload = {
        "entity_id": "tag1",
        "zone": "zone1",
        "timestamp": AT.isoformat(),
        "information_id": "measurement1",
    }
    record = EvidenceRecord(
        "zone",
        "zone:1",
        "rtls",
        "1",
        payload,
        AT,
        provenance={"physical_origin": "synthetic_fixture"},
    )
    a = prepare_evidence(
        (record,),
        as_of=AT,
        context=spec.context,
        interpreters={("rtls", "1"): ZoneInterpreter((binding,))},
    )
    b = prepare_evidence(
        (replace(record, payload={"observation": payload}),),
        as_of=AT,
        context=spec.context,
        interpreters={("rtls", "1"): ZoneInterpreter((binding,), wrapped_zone)},
    )
    assert a.observations == b.observations
    values = AssociationParameters(
        ("good", "bad"),
        (0.8, 0.2),
        (((0.9, 0.1), (0.1, 0.9)), ((0.5, 0.5), (0.5, 0.5))),
        "source_mode",
        "manual",
    )
    params = resolve_parameters(
        a.observations, default=TrustRule(values, "manual"), priors={"association": (0.5, 0.5)}
    )
    model = compile_model(replace(spec, evidence=a, parameters=params))
    result = infer(model, policy=InferencePolicy(engine="reference_elimination"))
    assert result.posterior.marginal("association").probabilities == pytest.approx([0.82, 0.18])
    duplicate = replace(a.observations[0], observation_id="copy")
    copied = compile_model(
        replace(
            spec, evidence=replace(a, observations=(*a.observations, duplicate)), parameters=params
        )
    )
    assert len([f for f in copied.factors if f.key == "prior:source_mode"]) == 1
    assert infer(copied, policy=InferencePolicy(engine="reference_elimination")).posterior.marginal(
        "association"
    ).probabilities == pytest.approx([0.82, 0.18])
    assert ZoneInterpreter().interpret(record, spec.context)[1][0].status == "uninterpreted"
    assert loads(dumps(params)).parameter_id == params.parameter_id


def test_timestamp_channel_shared_bias_and_conflicting_declarations():
    from ocbf.evidence import Observation
    from ocbf.reliability.channels import TimestampParameters

    spec = toy((ContinuousVariableSpec("t", 0, 2, "seconds", "manual"),))
    record = EvidenceRecord("time", "time:1", "clock", "1", {}, AT)
    observation = Observation(
        "o",
        "clock",
        "time",
        "1",
        "timestamp_gaussian",
        ("t",),
        1.0,
        ("time:1",),
        "timestamp-test@1",
        "g",
    )
    evidence = InterpretedEvidence(
        materialize((record,), as_of=AT), (observation, replace(observation, observation_id="copy"))
    )
    parameters = resolve_parameters(
        evidence.observations,
        default=TrustRule(
            TimestampParameters(1, 0, "seconds", "manual", "clock_bias", 3), "manual"
        ),
    )
    model = compile_model(replace(spec, evidence=evidence, parameters=parameters))
    assert len([f for f in model.factors if f.key == "prior:clock_bias"]) == 1
    result = infer(model, requirements=JOINT, policy=InferencePolicy(engine="reference_hybrid"))
    assert result.posterior.log_normalizer == pytest.approx(norm.logpdf(1, scale=np.sqrt(14)))
    conflicting = replace(
        spec,
        variables=(
            *spec.variables,
            ContinuousVariableSpec("clock_bias", 0, 99, "seconds", "manual"),
        ),
        evidence=evidence,
        parameters=parameters,
    )
    with pytest.raises(ValidationError, match="conflicting shared variable"):
        compile_model(conflicting)
    from ocbf.model.channels import TimestampChannel

    with pytest.raises(ValidationError, match="own latent key"):
        TimestampChannel().contribute(
            observation,
            TimestampParameters(1, 0, "seconds", "manual", "t", 2),
            {"t": spec.variables[0]},
        )


def test_contract_modules_do_not_import_concrete_implementations():
    modules = (
        "ocbf.model.contracts",
        "ocbf.sources.contracts",
        "ocbf.queries.contracts",
        "ocbf.inference.contracts",
        "ocbf.belief.posterior",
    )
    script = (
        "import sys; "
        + "; ".join("import " + m for m in modules)
        + "; assert not any(m in sys.modules for m in ('torch','gtsam','pymc','ocbf.model.kernels','ocbf.queries.aggregation','ocbf.inference.sampling'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    for module in modules:
        tree = ast.parse(Path(module.replace(".", "/") + ".py").read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(
            "adapters" in name or "kernels" in name or "aggregation" in name for name in imports
        )


def test_vectorized_factor_lowering_matches_scalar_reference():
    from examples.fixed_parameters import example_inputs
    from ocbf.model.batch import finite_table

    model = compile_model(example_inputs()[0])
    for factor in model.factors:
        _, table = finite_table(model, factor, {}, max_states=100000)
        expected = np.empty(table.shape)
        for idx in np.ndindex(table.shape):
            expected[idx] = model.log_value(factor, idx)
        assert np.allclose(table, expected)


def test_auto_rejects_before_sampling_without_config_and_rng():
    spec, query = comparison_spec()
    model = compile_model(spec)
    with pytest.raises(CapabilityError):
        plan_inference(
            model,
            requirements=requirements_for(query),
            policy=InferencePolicy(engine="auto", max_clique_states=1),
        )
    with pytest.raises(CapabilityError, match="random stream"):
        infer(model, requirements=requirements_for(query), policy=draw_policy())
    with pytest.raises(BudgetExceeded):
        infer(
            model,
            requirements=JOINT,
            policy=replace(draw_policy(), max_draw_bytes=1),
            rng=np.random.default_rng(1),
        )


def test_whole_job_uses_joint_obligations_and_priority_excludes_unrankable_draws():
    spec, bundle = comparison_spec()
    rank, single = bundle.queries
    execution = single.executions[0]
    whole = replace(
        single,
        executions=(execution, replace(execution, execution_id="second_obligation")),
        reference={"threshold_minutes": {"a": 30, "second_obligation": 30}},
    )
    model = compile_model(spec)
    result = infer(
        model, requirements=JOINT, policy=InferencePolicy(engine="reference_elimination")
    )
    answer = evaluate(result, QueryBundle((whole,))).estimates[0]
    assert answer.value == pytest.approx(0.5)  # two perfectly dependent obligations, not .25
    unavailable = replace(
        rank, executions=(rank.executions[0], replace(rank.executions[1], starts=()))
    )
    answer = evaluate(result, QueryBundle((unavailable,))).estimates[0]
    assert answer.denominator == 0 and answer.status == "unresolved"
    assert answer.distribution["unrankable_mass"] == 1


def test_modeled_endpoint_nonlinear_query_matches_normal_cdf():
    spec = toy(
        (
            VariableSpec("start", (True,)),
            VariableSpec("end", (True,)),
            ContinuousVariableSpec(
                "end_time", AT.timestamp() + 1800, 600, "UTC seconds", "synthetic"
            ),
        )
    )
    execution = ExecutionProjection(
        "op", "job", (Endpoint("start", AT),), (Endpoint("end", None, time_key="end_time"),)
    )
    query = QuerySpec(
        "duration",
        "duration_exception",
        (execution,),
        AT - timedelta(hours=1),
        AT + timedelta(hours=2),
        {"threshold_minutes": {"op": 30}},
    )
    bundle = QueryBundle((query,))
    result = infer(
        compile_model(spec),
        requirements=requirements_for(bundle),
        policy=InferencePolicy(engine="auto"),
    )
    answer = evaluate(result, bundle, rng=np.random.default_rng(15)).estimates[0]
    expected = (norm.cdf(9) - norm.cdf(0)) / (norm.cdf(9) - norm.cdf(-3))
    assert answer.value == pytest.approx(expected, abs=0.025)
    assert answer.numerical["mcse"] is not None
    assert "iid" in answer.computation
    with pytest.raises(ValidationError):
        Endpoint("end", AT, time_key="end_time")


def test_large_custom_kernel_uses_batch_port_without_scalar_state_loop():
    from ocbf.model.batch import finite_table

    class IndependentKernel:
        name = "external_sum"
        version = "1"

        def log_values(self, factor, state, domains):
            return -0.2 * sum(v.astype(float) for v in state.values())

        def log_value(self, *args):
            raise AssertionError("large lowering must not loop over scalar states")

    variables = tuple(VariableSpec(f"b{i}", (False, True)) for i in range(18))
    factor = FactorSpec("large", tuple(v.key for v in variables), "external_sum")
    from ocbf.model.kernels import builtin_kernels

    kernels = {**builtin_kernels(), "external_sum": IndependentKernel()}
    model = compile_model(
        toy(variables, (factor,), {v.key: (0.5, 0.5) for v in variables}), factors=kernels
    )
    _, table = finite_table(model, factor, {}, max_states=1 << 18)
    assert table.shape == (2,) * 18
    assert table[(1,) * 18] == pytest.approx(-3.6)


def test_exact_expected_sum_retains_local_joint_budget():
    variables = (
        VariableSpec("s", (True,)),
        *(VariableSpec(f"e{i}", (False, True)) for i in range(12)),
    )
    spec = toy(variables, priors={f"e{i}": (0.5, 0.5) for i in range(12)})
    executions = tuple(
        ExecutionProjection(
            str(i), str(i), (Endpoint("s", AT),), (Endpoint(f"e{i}", AT + timedelta(hours=1)),)
        )
        for i in range(12)
    )
    query = QuerySpec(
        "count",
        "expected_exception_count",
        executions,
        AT - timedelta(hours=1),
        AT + timedelta(hours=2),
        {"threshold_minutes": {str(i): 30 for i in range(12)}},
    )
    bundle = QueryBundle((query,))
    result = infer(
        compile_model(spec),
        requirements=requirements_for(bundle),
        policy=InferencePolicy(engine="reference_elimination", max_joint_states=2),
    )
    answer = evaluate(result, bundle).estimates[0]
    assert answer.value == pytest.approx(6)
    assert answer.numerical["status"] == "exact"


def test_semantic_time_decoding_and_fixed_time_conflict():
    from ocbf.assertions import AssertionRef as Ref
    from ocbf.model.compile import structural_variables
    from ocbf.model.decoding import decode_assignment
    from ocbf.schema import EventType

    context = SemanticContext.from_universe(
        UniverseBuilder(Schema([EventType("E")], [])).add_event("e", type_support={"E"}).build()
    )
    exists, time_key = str(Ref.event_exists("e")), str(Ref.event_time("e"))
    variables = (
        *structural_variables(context),
        ContinuousVariableSpec(
            time_key, AT.timestamp(), 1, "UTC seconds", "manual", ((exists, True),)
        ),
    )
    evidence = InterpretedEvidence(materialize((), as_of=AT), ())
    spec = ModelSpec(
        context,
        evidence,
        ParameterSet(priors={exists: (0.5, 0.5)}, assumptions={"cardinality_role": "normative"}),
        variables,
    )
    model = compile_model(spec)
    result = infer(model, requirements=JOINT, policy=InferencePolicy(engine="reference_hybrid"))
    for component in result.posterior.conditional.components:
        state = component.draw_assignment(np.random.default_rng(1))
        decoded = decode_assignment(model, state)
        assert ("e" in decoded.events) == state[exists]
        if state[exists]:
            assert decoded.events["e"]["time"].timestamp() == pytest.approx(state[time_key])
    with pytest.raises(ValidationError, match="fixed and modeled"):
        compile_model(replace(spec, decoding={"fixed_endpoints": {"e": AT}}))


def test_recorded_rng_replay_and_retraction_regenerate_draws(tmp_path):
    from examples._shared.study import reproduce, run_study, verify_reproduction
    from examples.joint_inference import inputs
    from ocbf.evidence import EvidenceAction

    spec, bundle, _, records = inputs()
    policy = InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=2, warmup=5, draws=30)
    )
    result = run_study(
        spec, bundle, {}, policy=policy, seed=7, source_status="synthetic", output=tmp_path
    )
    verify_reproduction(result, reproduce(tmp_path / "inputs.json"))
    retraction = replace(
        records[0], action=EvidenceAction.RETRACT, revision_id="a:2", previous_revision="a:1"
    )
    updated, _, _, _ = inputs((*records, retraction))
    new_result = infer(
        compile_model(updated),
        requirements=requirements_for(bundle),
        policy=policy,
        rng=np.random.default_rng(7),
    )
    assert new_result.model_id != result.model_id
    assert new_result.posterior.draw_set.draw_set_id != result.estimates[0].metadata["draw_set_id"]


def test_bad_gaussian_parameters_and_shared_configuration_fail_before_inference():
    spec = scalar_hybrid()
    factor = spec.factors[0]
    for parameters in (
        {**factor.parameters, "sd": 0},
        {**factor.parameters, "observed": float("nan")},
        {**factor.parameters, "mode_key": "missing"},
    ):
        with pytest.raises(ValidationError):
            compile_model(replace(spec, factors=(replace(factor, parameters=parameters),)))


def test_custom_proposal_is_substitutable_and_corrected():
    from ocbf.inference.contracts import Proposal
    from ocbf.inference.sampling import BlockedEngine

    class BiasedProposal:
        name = "independent-biased"
        version = "1"

        def propose(self, state, rng):
            value = bool(rng.random() < 0.2)
            return Proposal(
                {"a": value}, np.log(0.2 if value else 0.8), np.log(0.2 if state["a"] else 0.8)
            )

    model = compile_model(toy((VariableSpec("a", (False, True)),), priors={"a": (0.2, 0.8)}))
    engine = BlockedEngine(proposals={("a",): BiasedProposal()})
    result = infer(
        model,
        requirements=JOINT,
        policy=draw_policy(),
        engines={"blocked": engine},
        rng=np.random.default_rng(551),
    )
    assert result.posterior.draw(("a",)).values["a"].mean() == pytest.approx(0.8, abs=0.06)


def test_budget_exhaustion_retains_explicit_incomplete_draws(monkeypatch):
    import itertools

    from ocbf.inference import sampling

    ticks = itertools.count()
    monkeypatch.setattr(sampling.time, "monotonic", lambda: next(ticks))
    model = compile_model(toy((VariableSpec("a", (False, True)),), priors={"a": (0.5, 0.5)}))
    policy = InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=4, warmup=0, draws=100), max_seconds=12
    )
    result = infer(model, requirements=JOINT, policy=policy, rng=np.random.default_rng(18))
    assert result.diagnostics["status"] == "budget_exhausted"
    assert result.posterior.draw_set.metadata["status"] == "budget_exhausted"
    assert result.posterior.draw_set.shape[1] < 100


def test_rank_distribution_matches_exact_shared_source_reference():
    from examples.joint_inference import inputs

    spec, queries, _, _ = inputs()
    model = compile_model(spec)
    exact = evaluate(
        infer(
            model,
            requirements=requirements_for(queries),
            policy=InferencePolicy(engine="reference_elimination"),
        ),
        queries,
    )
    sampled = evaluate(
        infer(
            model,
            requirements=requirements_for(queries),
            policy=draw_policy(),
            rng=np.random.default_rng(42),
        ),
        queries,
    )
    for job, ranks in exact.estimates[2].distribution["ranks"].items():
        for rank, reference in ranks.items():
            value = sampled.estimates[2].distribution["ranks"][job][rank]
            assert abs(value["probability"] - reference["probability"]) < 0.05
            assert value["numerical"]["mcse"] is not None


def test_draws_reject_lossy_category_conversion_and_budget_ancestral_workspace():
    model = compile_model(toy((VariableSpec("a", (False, "unknown")),), priors={"a": (0.5, 0.5)}))
    result = infer(model, policy=InferencePolicy(engine="reference_elimination"))
    assert result.posterior.joint(("a",)).domains == ((False, "unknown"),)
    from ocbf.queries.aggregation import exact_data

    assert exact_data(result.posterior, ("a",)).state["a"].tolist() == [[False, "unknown"]]
    with pytest.raises(CapabilityError, match="losslessly"):
        result.posterior.draw(("a",), rng=np.random.default_rng(1))
    with pytest.raises(CapabilityError, match="losslessly"):
        plan_inference(model, requirements=JOINT, policy=draw_policy())
    numeric = compile_model(toy((VariableSpec("a", (False, True)),), priors={"a": (0.5, 0.5)}))
    result = infer(
        numeric, policy=InferencePolicy(engine="reference_elimination", max_draw_bytes=100)
    )
    with pytest.raises(BudgetExceeded):
        result.posterior.draw(("a",), rng=np.random.default_rng(1), size=10)


def test_larger_case_rejects_exact_budget_and_preserves_constrained_support():
    from examples.joint_inference import larger_inputs

    model, requirements = larger_inputs()
    policy = InferencePolicy(
        engine="auto",
        max_table_states=512,
        max_clique_states=512,
        sampling=SamplingConfig(chains=2, warmup=5, draws=10),
    )
    plan = plan_inference(model, requirements=requirements, policy=policy)
    assert plan.engine == "blocked"
    result = infer(model, requirements=requirements, policy=policy, rng=np.random.default_rng(12))
    counts = np.stack(tuple(result.posterior.draw_set.values.values())).sum(axis=0)
    assert ((counts >= 3) & (counts <= 9)).all()
