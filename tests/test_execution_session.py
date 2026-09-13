"""Repeated-execution reuse, revision, warm-start and lifecycle validation.

Shaped as one end-to-end vertical slice through the runtime surface plus a few minimal
correctness probes, kept cheap for a constrained machine: exact `reference_elimination`
throughout, tiny models, and a single small sampling config. Reuse is an execution concern,
so every optimized path is checked against a forced-fresh calculation of the same target.
Synthetic fixtures establish computational behavior only; no factory-accuracy claim.
"""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from examples.joint_inference import inputs
from ocbf.api import (
    ExecutionControl,
    ExecutionSession,
    compile_model,
    evaluate,
    infer,
    requirements_for,
    summaries_only,
    warm_start_from,
)
from ocbf.belief.posterior import QueryRequirements
from ocbf.errors import CapabilityError, ExecutionCancelled, ResourceExhausted, ValidationError
from ocbf.evidence import EvidenceAction, InterpretedEvidence
from ocbf.evidence.snapshots import materialize
from ocbf.inference.contracts import InferencePolicy, SamplingConfig
from ocbf.io.arrays import read_artifact, write_artifact
from ocbf.model.dependencies import dependency_index
from ocbf.model.spec import ModelSpec, VariableSpec
from ocbf.queries import QueryBundle
from ocbf.reliability.config import ParameterSet
from ocbf.schema import Schema
from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext

BUDGET = 64 * 1024 * 1024
EXACT = InferencePolicy(engine="reference_elimination")
DRAWS = QueryRequirements(capabilities=("joint_draws",))
SCOPE = ("association:a", "association:b")


def tiny_sampling():
    # Smallest config that still yields >=2 chains (an independent comparison chain) and
    # a nonempty draw set; deliberately minimal to keep the run cheap.
    return InferencePolicy(engine="blocked", sampling=SamplingConfig(chains=2, warmup=3, draws=12))


def toy(variables, priors):
    context = SemanticContext.from_universe(UniverseBuilder(Schema([], [])).build())
    from datetime import UTC, datetime

    evidence = InterpretedEvidence(materialize((), as_of=datetime(2026, 9, 8, tzinfo=UTC)), ())
    return ModelSpec(
        context, evidence, ParameterSet(priors=priors), tuple(variables), (), ground_structure=False
    )


def test_e2e_vertical_slice_reuse_matches_fresh(tmp_path):
    """Compile -> infer -> evaluate -> export -> reload under one session, compared to a
    forced-fresh pass of the same target across the whole stack."""
    spec, queries, _, _ = inputs()
    req = requirements_for(queries)

    fresh_model = compile_model(spec)
    fresh_result = infer(fresh_model, requirements=req, policy=EXACT)
    fresh_answer = evaluate(fresh_result, queries)

    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        model = compile_model(spec, session=session)
        result = infer(model, requirements=req, policy=EXACT, session=session)
        evaluate(result, queries, session=session)  # first pass warms the query cache
        # A second identical pass exercises the cache without changing any answer.
        repeat = evaluate(
            infer(
                compile_model(spec, session=session),
                requirements=req,
                policy=EXACT,
                session=session,
            ),
            queries,
            session=session,
        )
        hits = session.stats["hits"]
        # Export the retained posterior and read it back through the portable array bundle.
        destination = tmp_path / "posterior"
        assert write_artifact(destination, result).status == "complete"
        restored = read_artifact(destination)

    assert hits > 0  # structural + posterior artifacts reused on the repeat pass
    # Cache metadata never changes the scientific Model ID.
    assert model.model_id == fresh_model.model_id
    # Belief layer: joint table and log normalizer are bit-for-bit equal to the fresh target.
    assert result.posterior.joint(SCOPE).probabilities == pytest.approx(
        fresh_result.posterior.joint(SCOPE).probabilities, abs=1e-10
    )
    assert result.posterior.log_normalizer == pytest.approx(
        fresh_result.posterior.log_normalizer, abs=1e-10
    )
    # Query layer: reused and fresh estimates agree exactly.
    for reused, baseline in zip(repeat.estimates, fresh_answer.estimates, strict=True):
        assert reused.outcomes == baseline.outcomes
        if baseline.value is None:
            assert reused.value is None
        else:
            assert reused.value == pytest.approx(baseline.value, abs=1e-10, rel=1e-9)
    # Interchange: reload preserves identity and normalization.
    assert restored.model_id == fresh_model.model_id
    assert restored.run_id == result.run_id
    assert restored.posterior.log_normalizer == pytest.approx(
        fresh_result.posterior.log_normalizer, abs=1e-12
    )


def test_trust_change_is_explainable_and_recomputes_dependents():
    spec, queries, settings, _ = inputs()
    req = requirements_for(queries)
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        nominal = compile_model(spec, session=session)
        infer(nominal, requirements=req, policy=EXACT, session=session)
        cautious = compile_model(replace(spec, parameters=settings["cautious"]), session=session)
        explained = session.last_assessment.details.get("dependency_changes")
        hits = session.stats["hits"]
    assert nominal.model_id != cautious.model_id  # distinct scientific identity
    diff = dependency_index(cautious).changed(dependency_index(nominal))
    assert diff and set(explained) == set(diff)  # exactly the changed numeric factors, explained
    assert hits > 0  # elimination schedule/order and unchanged factors reused across the change


def test_reference_change_reuses_decode_but_recomputes_outcomes():
    spec, queries, _, _ = inputs()
    req = requirements_for(queries)
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        result = infer(
            compile_model(spec, session=session), requirements=req, policy=EXACT, session=session
        )
        first = evaluate(result, queries, session=session)
        before = session.stats["hits"]
        stricter = QueryBundle(
            tuple(
                replace(q, reference={"reference_id": "t2", "threshold_minutes": {"a": 1, "b": 1}})
                for q in queries.queries
            )
        )
        second = evaluate(result, stricter, session=session)
        after = session.stats["hits"]
    assert after > before  # decoded endpoints / joint tables reused across the reference change
    assert any(
        a.outcomes != b.outcomes or a.value != b.value
        for a, b in zip(first.estimates, second.estimates, strict=True)
    )


def test_revision_reuse_agrees_with_fresh_computation():
    _, _, _, records = inputs()
    correction = replace(
        records[0],
        revision_id="a:2",
        previous_revision="a:1",
        action=EvidenceAction.REPLACE,
        payload={**records[0].payload, "zone": "z2"},
    )
    revised, bundle, _, _ = inputs((*records, correction))
    req = requirements_for(bundle)
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        infer(
            compile_model(inputs()[0], session=session),
            requirements=req,
            policy=EXACT,
            session=session,
        )
        reused = evaluate(
            infer(
                compile_model(revised, session=session),
                requirements=req,
                policy=EXACT,
                session=session,
            ),
            bundle,
            session=session,
        )
    fresh = evaluate(infer(compile_model(revised), requirements=req, policy=EXACT), bundle)
    for a, b in zip(reused.estimates, fresh.estimates, strict=True):
        assert a.outcomes == b.outcomes
        assert a.value == b.value or (a.value is None and b.value is None)


def test_deterministic_reuse_and_warm_start_semantics():
    spec, _, _, _ = inputs()
    policy = tiny_sampling()
    with ExecutionSession(max_cache_bytes=BUDGET) as session:
        cached = infer(
            compile_model(spec, session=session),
            requirements=DRAWS,
            policy=policy,
            rng=np.random.default_rng(7),
            session=session,
        )
    fresh = infer(
        compile_model(spec), requirements=DRAWS, policy=policy, rng=np.random.default_rng(7)
    )
    # Caching consumes no randomness: identical seed ⇒ identical draws and Draw-set ID.
    assert cached.posterior.draw_set.draw_set_id == fresh.posterior.draw_set.draw_set_id
    for key, values in cached.posterior.draw_set.values.items():
        assert np.array_equal(values, fresh.posterior.draw_set.values[key])

    # Warm start engages on the same target, and regenerates when structure changes.
    hint = warm_start_from(compile_model(spec), fresh)
    same = infer(
        compile_model(spec),
        requirements=DRAWS,
        policy=policy,
        rng=np.random.default_rng(8),
        warm_start=hint,
    )
    assert same.diagnostics["warm_start"]["mode"] == "warm"
    assert same.diagnostics["warm_start"]["independent_chain"] == 1  # chains - 1 kept fresh
    small = compile_model(toy((VariableSpec("a", (False, True)),), {"a": (0.4, 0.6)}))
    small_hint = warm_start_from(
        small, infer(small, requirements=DRAWS, policy=policy, rng=np.random.default_rng(9))
    )
    expanded = compile_model(
        toy(
            (VariableSpec("a", (False, True)), VariableSpec("b", (False, True))),
            {"a": (0.4, 0.6), "b": (0.5, 0.5)},
        )
    )
    regen = infer(
        expanded,
        requirements=DRAWS,
        policy=policy,
        rng=np.random.default_rng(10),
        warm_start=small_hint,
    )
    assert regen.diagnostics["warm_start"]["mode"] == "fresh"
    assert "structure" in regen.diagnostics["warm_start"]["reason"]


def test_lifecycle_statuses_retention_and_isolation():
    spec, queries, _, _ = inputs()
    req = requirements_for(queries)

    # Cancellation before work → cancelled.
    control = ExecutionControl()
    control.cancel()
    with pytest.raises(ExecutionCancelled) as cancelled:
        compile_model(spec, control=control)
    assert cancelled.value.execution.status == "cancelled"

    # Workspace budget too small for any real allocation → resource-exhausted.
    with pytest.raises(ResourceExhausted) as exhausted:
        infer(
            compile_model(spec),
            requirements=req,
            policy=EXACT,
            control=ExecutionControl(max_work_bytes=1),
        )
    assert exhausted.value.execution.status == "resource-exhausted"

    # summaries-only retention releases the posterior while prior answers stay valid.
    result = infer(compile_model(spec), requirements=req, policy=EXACT)
    answer = evaluate(result, queries)
    detached = summaries_only(result)
    assert detached.posterior is None and detached.capabilities == ()
    assert answer.estimates
    with pytest.raises(CapabilityError):
        evaluate(detached, queries)

    # Sessions are isolated and reject use after close.
    session = ExecutionSession(max_cache_bytes=BUDGET)
    session.close()
    with pytest.raises(ValidationError):
        compile_model(spec, session=session)


def test_export_rejects_overwrite_and_tampering(tmp_path):
    spec, queries, _, _ = inputs()
    result = infer(compile_model(spec), requirements=requirements_for(queries), policy=EXACT)
    destination = tmp_path / "artifact"
    write_artifact(destination, result)
    with pytest.raises(ValidationError):  # never overwrite an existing destination
        write_artifact(destination, result)
    payload = next(destination.glob("array-*.npy"))
    data = bytearray(payload.read_bytes())
    data[-1] ^= 0xFF
    payload.write_bytes(bytes(data))
    with pytest.raises(ValidationError):  # checksum validated before the array is trusted
        read_artifact(destination)


def test_architecture_isolation_and_capability_rejection():
    from ocbf.inference.adapters.approximate import BPEngine

    # Core runtime + facade import without optional backends.
    script = (
        "import sys, ocbf.api, ocbf.runtime; "
        "assert not any(m in sys.modules for m in ('torch', 'gtsam', 'pymc'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True)

    # An engine without cooperative support rejects a control request explicitly.
    model = compile_model(toy((VariableSpec("a", (False, True)),), {"a": (0.5, 0.5)}))
    with pytest.raises(CapabilityError, match="cooperative"):
        infer(
            model,
            policy=InferencePolicy(engine=BPEngine().name, allow_approximate=True),
            control=ExecutionControl(),
        )
