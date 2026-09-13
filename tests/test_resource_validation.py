"""Independent numerical, invalidation and resource boundaries for Repeated-execution.

These are synthetic numerical references, not empirical factory calibration.
"""

import json
from dataclasses import dataclass, replace
from datetime import timedelta

import numpy as np
import pytest
from scipy.special import logsumexp

from examples.joint_inference import AT, inputs
from ocbf._values import canonical_json
from ocbf.api import (
    ExecutionControl,
    ExecutionSession,
    compare_settings,
    compile_model,
    evaluate,
    infer,
    prepare_evidence,
    warm_start_from,
)
from ocbf.belief.posterior import QueryRequirements
from ocbf.diagnostics.monte_carlo import assessment
from ocbf.errors import (
    BudgetExceeded,
    ExecutionCancelled,
    IncompatibleModel,
    ResourceExhausted,
    ValidationError,
)
from ocbf.evidence import EvidenceAction, InterpretedEvidence
from ocbf.evidence.snapshots import materialize
from ocbf.inference.contracts import InferencePolicy, SamplingConfig
from ocbf.io import dumps, loads, read_artifact, write_artifact
from ocbf.model.dependencies import dependency_index
from ocbf.model.kernels import builtin_kernels
from ocbf.model.spec import ContinuousVariableSpec, FactorSpec, ModelSpec, VariableSpec
from ocbf.queries import Endpoint, ExecutionProjection, QueryBundle, QuerySpec
from ocbf.reliability.config import ParameterSet
from ocbf.runtime.cache import MemoryArtifactStore, artifact_key

EXACT = InferencePolicy(engine="reference_elimination")
DRAW = QueryRequirements(capabilities=("joint_draws",))
BUDGET = 8 * 1024 * 1024


def toy(variables, factors=(), priors=None):
    base = inputs()[0]
    return ModelSpec(
        base.context,
        InterpretedEvidence(materialize((), as_of=AT), ()),
        ParameterSet(priors=priors or {}),
        tuple(variables),
        tuple(factors),
        ground_structure=False,
    )


def assert_enumerated(model, result):
    keys = tuple(model.domains)
    logs = np.array(
        [
            model.log_density(dict(zip(keys, values, strict=True)))
            for values in __import__("itertools").product(*model.domains.values())
        ]
    )
    expected = np.exp(logs - logsumexp(logs))
    assert result.posterior.log_normalizer == pytest.approx(logsumexp(logs), abs=1e-10)
    assert result.posterior.joint(keys).probabilities.ravel() == pytest.approx(expected, abs=1e-10)


def test_store_eviction_ownership_refusal_and_fresh_bypass():
    store = MemoryArtifactStore(3200)
    array = np.arange(64.0)
    assert store.put("test", "one", array, slot="value")
    array[:] = -1
    retained = store.get("test", "one")
    assert np.array_equal(retained, np.arange(64.0))
    with pytest.raises(ValueError):
        retained.setflags(write=True)
    for i in range(10):
        store.put("test", str(i), np.arange(64.0) + i, slot="value")
    assert store.stats["bytes"] <= 3200 and store.stats["evictions"] > 0
    assert np.array_equal(retained, np.arange(64.0))
    assert not store.put("test", "huge", np.ones(10000))
    assert not store.put("test", "mutable", object())
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        spec = inputs()[0]
        model = compile_model(spec, session=session)
        before = dict(session.stats)
        fresh = compile_model(spec, session=session, reuse=False)
        infer(fresh, policy=EXACT, session=session, reuse=False)
        assert session.stats == before
        assert model.model_id == fresh.model_id


def test_dependency_explanations_are_bounded_and_precise_on_cached_compilation():
    spec, _, _, _ = inputs()
    with ExecutionSession(max_cache_bytes=150_000) as session:
        model = compile_model(spec, session=session)
        original = dependency_index(model)
        repeated = compile_model(spec, session=session)
        assert dependency_index(repeated).parents == original.parents
        assert any(
            any(p.startswith("channel:") for p in parents) for parents in original.parents.values()
        )
        for i in range(60):
            changed = replace(original, structure_id=f"structure:{i}")
            session.explain_dependencies(changed)
        assert session.stats["bytes"] <= 150_000
        assert not hasattr(session, "_indices")


def test_artifact_keys_are_typed_and_sensitive_to_numeric_layout():
    values = (True, 1, 1.0, "1", {"x": 1}, ("x", 1), np.array([1]), np.array([1.0]))
    assert len({artifact_key(v) for v in values}) == len(values)
    assert artifact_key({"a": 1, "b": 2}) == artifact_key({"b": 2, "a": 1})
    assert artifact_key(np.array([0.0])) != artifact_key(np.array([-0.0]))
    assert artifact_key(np.array([[1, 2]])) != artifact_key(np.array([1, 2]))


@pytest.mark.parametrize("change", ["offset", "weights", "support", "scope", "domain"])
def test_elimination_dependency_closure_matches_independent_enumeration(change):
    variables = tuple(VariableSpec(k, (False, True)) for k in ("a", "b", "shared"))
    factors = (
        FactorSpec("a/shared", ("a", "shared"), log_values=np.log([[0.9, 0.2], [0.1, 0.8]])),
        FactorSpec("b/shared", ("b", "shared"), log_values=np.log([[0.7, 0.3], [0.3, 0.7]])),
    )
    spec = toy(variables, factors, {k: (0.5, 0.5) for k in ("a", "b", "shared")})
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        original = compile_model(spec, session=session)
        result = infer(original, policy=EXACT, session=session)
        assert_enumerated(original, result)
        if change == "offset":
            updated = replace(spec, factors=(replace(factors[0], log_offset=2.3), factors[1]))
        elif change == "weights":
            updated = replace(
                spec,
                parameters=ParameterSet(
                    priors={"a": (0.5, 0.5), "b": (0.5, 0.5), "shared": (0.1, 0.9)}
                ),
            )
        elif change == "support":
            updated = replace(
                spec,
                factors=(
                    replace(factors[0], log_values=np.array([[0.0, -np.inf], [-np.inf, 0.0]])),
                    factors[1],
                ),
            )
        elif change == "scope":
            updated = replace(spec, scope={"conditioning": "declared cohort"})
        else:
            updated = toy(
                (VariableSpec("a", ("left", "right", "unresolved")),), priors={"a": (0.2, 0.3, 0.5)}
            )
        model = compile_model(updated, session=session)
        changed = infer(model, policy=EXACT, session=session)
        assert model.model_id != original.model_id
        assert_enumerated(model, changed)
        if change == "weights":
            assert not np.allclose(
                changed.posterior.marginal("b").probabilities,
                result.posterior.marginal("b").probabilities,
            )
        with pytest.raises(BudgetExceeded):
            infer(original, policy=replace(EXACT, max_clique_states=1), session=session)


def test_cached_positive_target_cannot_hide_a_new_zero_normalizer():
    a = VariableSpec("a", (False, True))
    spec = toy(
        (a,), (FactorSpec("zero", ("a",), log_values=np.array([0.0, -np.inf])),), {"a": (0.5, 0.5)}
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        infer(compile_model(spec, session=session), policy=EXACT, session=session)
        impossible = replace(
            spec,
            factors=(*spec.factors, FactorSpec("one", ("a",), log_values=np.array([-np.inf, 0.0]))),
        )
        with pytest.raises(IncompatibleModel):
            infer(compile_model(impossible, session=session), policy=EXACT, session=session)
        assert session.last_assessment.status == "failed"


def test_external_kernel_without_reuse_contract_is_recomputed():
    class Kernel:
        name, version = "external", "1"

        def __init__(self):
            self.weight = 0.2

        def log_value(self, factor, indices, domains):
            return np.log(self.weight if indices[0] else 1 - self.weight)

    kernel = Kernel()
    registry = {**builtin_kernels(), "external": kernel}
    spec = toy(
        (VariableSpec("a", (False, True)),),
        (FactorSpec("report", ("a",), "external"),),
        {"a": (0.5, 0.5)},
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        model = compile_model(spec, factors=registry, session=session)
        first = infer(model, policy=EXACT, session=session)
        kernel.weight = 0.8
        second = infer(
            compile_model(spec, factors=registry, session=session), policy=EXACT, session=session
        )
        assert first.posterior.marginal("a").probabilities[1] == pytest.approx(0.2)
        assert second.posterior.marginal("a").probabilities[1] == pytest.approx(0.8)
        assert dependency_index(model).support_id is None


def test_interpreter_configuration_and_unresolved_admission_survive_reuse():
    @dataclass(frozen=True)
    class Interpreter:
        value: str
        name: str = "configured"
        version: str = "1"

        @property
        def reuse_key(self):
            return (self.name, self.version, self.value)

        def interpret(self, record, context):
            from ocbf.evidence import AdmissionIssue, Observation

            return (
                Observation(
                    record.record_id,
                    record.source_id,
                    "association",
                    record.producer_version,
                    "categorical",
                    ("candidate",),
                    self.value,
                    (record.revision_id,),
                    "configured@1",
                    record.record_id,
                ),
            ), (
                AdmissionIssue(
                    record.revision_id, "uninterpreted", "auxiliary field has no likelihood"
                ),
            )

    spec, _, _, records = inputs()
    with ExecutionSession(max_cache_bytes=BUDGET) as session:

        def prepare(value):
            return prepare_evidence(
                records,
                as_of=AT,
                context=spec.context,
                interpreters={("rtls", "1"): Interpreter(value)},
                session=session,
            )

        first, same, changed = prepare("left"), prepare("left"), prepare("right")
        assert first.interpretation_id == same.interpretation_id != changed.interpretation_id
        assert first.issues == same.issues == changed.issues
        assert session.stats["hits"] > 0


def test_conditional_cache_includes_declared_extension_configuration():
    from ocbf.inference.conditional import condition_target

    @dataclass(frozen=True)
    class Kernel:
        weight: float
        name: str = "configured"
        version: str = "1"

        @property
        def reuse_key(self):
            return (self.version, self.weight)

        def log_value(self, factor, indices, domains):
            return np.log(self.weight if indices[0] else 1 - self.weight)

    spec = toy(
        (VariableSpec("a", (False, True)),),
        (FactorSpec("report", ("a",), "configured"),),
        {"a": (0.5, 0.5)},
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        for weight in (0.2, 0.8):
            model = compile_model(
                spec, factors={**builtin_kernels(), "configured": Kernel(weight)}, session=session
            )
            actual = condition_target(model, {}, EXACT, store=session.store_for())
            assert actual.finite.marginal("a").probabilities[1] == pytest.approx(weight)


def test_retraction_reordered_arrivals_and_cutoffs_recompute_same_target():
    original, _, _, records = inputs()
    correction = replace(
        records[0],
        revision_id="a:2",
        previous_revision="a:1",
        action=EvidenceAction.REPLACE,
        known_at=AT + timedelta(seconds=1),
        payload={**records[0].payload, "zone": "z2"},
    )
    retract = replace(
        correction,
        revision_id="a:3",
        previous_revision="a:2",
        action=EvidenceAction.RETRACT,
        known_at=AT + timedelta(seconds=2),
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        infer(compile_model(original, session=session), policy=EXACT, session=session)
        for cutoff in (AT, AT + timedelta(seconds=1), AT + timedelta(seconds=2)):
            action_set = (*records, correction, retract)
            spec = inputs(action_set, as_of=cutoff)[0]
            shuffled = inputs(tuple(reversed(action_set)), as_of=cutoff)[0]
            model = compile_model(spec, session=session)
            assert model.model_id == compile_model(shuffled).model_id
            actual = infer(model, policy=EXACT, session=session)
            assert_enumerated(model, actual)


def test_reference_changes_recompute_pending_denominators_on_retained_draws():
    spec = toy(
        (VariableSpec("start", (True,)), VariableSpec("end", (False, True))),
        priors={"end": (0.5, 0.5)},
    )
    projection = ExecutionProjection(
        "op", "job", (Endpoint("start", AT - timedelta(minutes=20)),), (Endpoint("end", AT),)
    )
    query = QuerySpec(
        "op",
        "duration_exception",
        (projection,),
        AT - timedelta(hours=1),
        AT + timedelta(minutes=1),
        {"threshold_minutes": {"op": 30}},
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        result = infer(compile_model(spec), policy=EXACT, session=session)
        first = evaluate(result, QueryBundle((query,)), session=session)
        stricter = replace(query, reference={"threshold_minutes": {"op": 10}})
        second = evaluate(result, QueryBundle((stricter,)), session=session)
        fresh = evaluate(result, QueryBundle((stricter,)))
        assert first.estimates[0].outcomes["pending"] == 0.5
        assert second.estimates[0].outcomes["unresolved"] == 0.5
        assert canonical_json(second.estimates) == canonical_json(fresh.estimates)


@pytest.mark.parametrize("seed", [113, 229])
def test_warm_start_new_trust_matches_exact_and_independent_chains(seed):
    original = compile_model(toy((VariableSpec("a", (False, True)),), priors={"a": (0.6, 0.4)}))
    changed = compile_model(toy((VariableSpec("a", (False, True)),), priors={"a": (0.3, 0.7)}))
    policy = InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=4, warmup=60, draws=700)
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        old = infer(
            original,
            requirements=DRAW,
            policy=policy,
            rng=np.random.default_rng(seed),
            session=session,
        )
        hint = warm_start_from(original, old)
        warm = infer(
            changed,
            requirements=DRAW,
            policy=policy,
            rng=np.random.default_rng(seed + 1),
            session=session,
            warm_start=hint,
        )
        cold = infer(
            changed,
            requirements=DRAW,
            policy=policy,
            rng=np.random.default_rng(seed + 2),
            session=session,
        )
    estimates = []
    for result in (warm, cold):
        values = result.posterior.draw_set.values["a"].astype(float)
        quality = assessment(values)
        error = abs(values.mean() - 0.7)
        assert error <= 0.03 and error <= 4 * quality["mcse"] + 0.002
        estimates.append((values.mean(), quality["mcse"]))
    assert (
        abs(estimates[0][0] - estimates[1][0])
        <= 4 * np.hypot(estimates[0][1], estimates[1][1]) + 0.002
    )
    assert warm.diagnostics["warm_start"]["mode"] == "warm"
    assert warm.diagnostics["warm_start"]["independent_chain"] == 3
    assert warm.posterior.draw_set.draw_set_id != hint.draw_set_id
    assert warm.model_id != hint.model_id


def test_support_expansion_forces_regeneration_and_fresh_mode_ignores_hint():
    policy = InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=2, warmup=5, draws=80)
    )
    original = compile_model(toy((VariableSpec("a", (False, True)),), priors={"a": (1.0, 0.0)}))
    changed = compile_model(toy((VariableSpec("a", (False, True)),), priors={"a": (0.5, 0.5)}))
    hint = warm_start_from(
        original, infer(original, requirements=DRAW, policy=policy, rng=np.random.default_rng(1))
    )
    regenerated = infer(
        changed, requirements=DRAW, policy=policy, rng=np.random.default_rng(2), warm_start=hint
    )
    assert regenerated.diagnostics["warm_start"]["mode"] == "fresh"
    assert "support" in regenerated.diagnostics["warm_start"]["reason"]
    assert regenerated.posterior.draw_set.values["a"].any()
    fresh = infer(
        changed,
        requirements=DRAW,
        policy=policy,
        rng=np.random.default_rng(2),
        reuse=False,
        warm_start=hint,
    )
    assert np.array_equal(
        fresh.posterior.draw_set.values["a"], regenerated.posterior.draw_set.values["a"]
    )


def test_partial_sampler_and_query_cancellation_preserve_explicit_statuses():
    spec, queries, _, _ = inputs()
    model = compile_model(spec)
    control = ExecutionControl()

    def progress(event):
        if event["stage"] == "sampling.sweep" and event["completed"] == 12:
            control.cancel()

    control.progress = progress
    policy = InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=2, warmup=3, draws=30)
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        result = infer(
            model,
            requirements=DRAW,
            policy=policy,
            rng=np.random.default_rng(9),
            control=control,
            session=session,
        )
        assert result.execution.status == "cancelled"
        assert result.posterior.draw_set.shape == (1, 9)
        assert session.last_assessment.status == "cancelled"
        answer = evaluate(result, queries)
        assert all(e.numerical["status"] == "incomplete" for e in answer.estimates)
        exact = infer(model, policy=EXACT)
        query_control = ExecutionControl()
        query_control.progress = lambda event: (
            query_control.cancel()
            if event["stage"] == "query.evaluate" and event["completed"] == 1
            else None
        )
        partial = evaluate(exact, queries, session=session, control=query_control)
        assert partial.execution.status == "cancelled" and len(partial.estimates) == 1
        assert partial.execution.details["missing_queries"] == tuple(
            q.name for q in queries.queries[1:]
        )


def test_failure_assessments_and_runtime_stops_never_trigger_solver_fallback():
    spec = inputs()[0]
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        control = ExecutionControl()
        control.cancel()
        with pytest.raises(ExecutionCancelled) as exc:
            compile_model(spec, session=session, control=control)
        assert session.last_assessment == exc.value.execution
        control = ExecutionControl(deadline=0)
        with pytest.raises(ResourceExhausted) as exc:
            compile_model(spec, session=session, control=control)
        assert session.last_assessment == exc.value.execution

    class FailingEngine:
        name, version = "reference_elimination", "1"

        def assess(self, model, req, policy):
            from ocbf.inference.adapters.exact import ExactEngine

            return ExactEngine(self.name).assess(model, req, policy)

        def solve(self, *args):
            raise MemoryError("test allocation failure")

    with pytest.raises(ResourceExhausted) as exc:
        infer(
            compile_model(spec),
            policy=InferencePolicy(engine="auto"),
            engines={"reference_elimination": FailingEngine()},
        )
    assert exc.value.execution.status == "resource-exhausted"


def test_shared_clock_change_and_hybrid_controls_preserve_analytic_normalizers():
    variables = (
        ContinuousVariableSpec("time", 0, 2, "seconds", "synthetic"),
        ContinuousVariableSpec("clock", 0, 1, "seconds", "synthetic"),
    )
    report = FactorSpec(
        "report",
        ("time", "clock"),
        "linear_gaussian",
        {"coefficients": {"time": 1.0, "clock": 1.0}, "observed": 1.0, "sd": 0.5},
    )
    spec = toy(variables, (report,))
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        for clock_sd in (1.0, 3.0):
            updated = replace(
                spec, variables=(variables[0], replace(variables[1], prior_sd=clock_sd))
            )
            model = compile_model(updated, session=session)
            result = infer(
                model,
                requirements=QueryRequirements(capabilities=("normalizer",)),
                policy=InferencePolicy(engine="reference_hybrid"),
                session=session,
            )
            variance = 4 + clock_sd**2 + 0.25
            expected = -0.5 * (np.log(2 * np.pi * variance) + 1 / variance)
            assert result.posterior.log_normalizer == pytest.approx(expected, abs=1e-10)


def test_session_comparisons_allocate_streams_before_runs_and_hold_epochs():
    spec, queries, settings, _ = inputs()
    policy = InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=2, warmup=5, draws=40)
    )
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        reused = compare_settings(
            spec, queries, settings, policy=policy, session=session, rng=np.random.default_rng(44)
        )
        fresh = compare_settings(
            spec,
            queries,
            settings,
            policy=policy,
            session=session,
            reuse=False,
            rng=np.random.default_rng(44),
        )
    for a, b in zip(reused.settings, fresh.settings, strict=True):
        assert a[:2] == b[:2]
        assert canonical_json(a[2].estimates) == canonical_json(b[2].estimates)
    runs = reused.execution["runs"]
    assert tuple(r["name"] for r in runs) == tuple(sorted(settings))
    assert len({r["setting_seed"] for r in runs}) == len(settings)
    assert all(len(r["chain_seeds"]) == 2 for r in runs)
    assert loads(dumps(reused)).execution == reused.execution


def test_resource_exhaustion_after_usable_draws_and_unexpected_failure_assessments():
    from ocbf.errors import ExecutionFailure
    from ocbf.runtime.control import operation

    with pytest.raises(ExecutionFailure) as failure, operation("extension"):
        raise RuntimeError("broken extension")
    assert failure.value.execution.status == "failed"
    assert isinstance(failure.value.__cause__, RuntimeError)

    control = ExecutionControl()

    def progress(event):
        if event["stage"] == "sampling.sweep" and event["completed"] == 12:
            control.deadline = 0

    control.progress = progress
    model = compile_model(inputs()[0])
    policy = InferencePolicy(
        engine="blocked", sampling=SamplingConfig(chains=2, warmup=3, draws=30)
    )
    result = infer(
        model, requirements=DRAW, policy=policy, rng=np.random.default_rng(9), control=control
    )
    assert result.execution.status == "resource-exhausted"
    assert result.posterior.draw_set.shape == (1, 9)
    assert all(
        e.numerical["status"] == "incomplete" for e in evaluate(result, inputs()[1]).estimates
    )


@pytest.mark.parametrize(
    "damage", ["duplicate", "inline", "header", "shape", "path", "version", "budget"]
)
def test_array_artifact_validation_precedes_loading(tmp_path, monkeypatch, damage):
    path = tmp_path / "arrays"
    write_artifact(path, {"values": np.arange(32.0)})
    manifest = path / "manifest.json"
    document = json.loads(manifest.read_text())
    name = next(iter(document["arrays"]))
    if damage == "duplicate":
        document["value"]["map"]["copy"] = {"npy": name}
    elif damage == "inline":
        document["value"]["map"]["copy"] = {
            "array": {"sequence": [0]},
            "dtype": "V1000000000",
            "shape": [1],
        }
    elif damage == "header":
        document["arrays"][name]["dtype"] = "<i8"
    elif damage == "shape":
        document["arrays"][name]["shape"] = [2**60]
    elif damage == "path":
        document["arrays"]["../outside.npy"] = document["arrays"].pop(name)
        document["value"]["map"]["values"] = {"npy": "../outside.npy"}
    elif damage == "version":
        document["format"] = "unknown"
    manifest.write_text(json.dumps(document))
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("array allocated before validation")
    )
    with pytest.raises(ValidationError):
        read_artifact(path, max_bytes=1 if damage == "budget" else BUDGET)


def test_session_policy_is_explicit_and_call_policy_overrides_it():
    with ExecutionSession(max_cache_bytes=BUDGET, policy=EXACT) as session:
        model = compile_model(inputs()[0], session=session)
        result = infer(model, session=session)
        assert result.manifest["execution_plan"].engine == "reference_elimination"
        with pytest.raises(BudgetExceeded):
            infer(model, session=session, policy=replace(EXACT, max_clique_states=1))


def test_exports_retain_common_draw_identity_and_stop_cleanly(tmp_path):
    spec = toy((VariableSpec("a", (False, True)),), priors={"a": (0.4, 0.6)})
    result = infer(
        compile_model(spec),
        requirements=DRAW,
        policy=InferencePolicy(
            engine="blocked", sampling=SamplingConfig(chains=2, warmup=3, draws=15)
        ),
        rng=np.random.default_rng(10),
    )
    path = tmp_path / "sampled"
    write_artifact(path, result)
    restored = read_artifact(path)
    assert restored.posterior.draw_set.draw_set_id == result.posterior.draw_set.draw_set_id
    assert loads(dumps(result)).model_id == result.model_id
    control = ExecutionControl()
    control.progress = lambda event: (
        control.cancel() if event["stage"] == "export.array_chunk" else None
    )
    with pytest.raises(ExecutionCancelled) as exc:
        write_artifact(tmp_path / "cancelled", result, control=control)
    assert exc.value.execution.status == "cancelled" and not (tmp_path / "cancelled").exists()
    control = ExecutionControl()
    control.progress = lambda event: control.cancel() if event["stage"] == "import.header" else None
    with pytest.raises(ExecutionCancelled) as exc:
        read_artifact(path, control=control)
    assert exc.value.execution.status == "cancelled"
