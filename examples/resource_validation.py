"""Supplementary revision, hybrid, constrained-sampling and warm-start measurements.

Run ``python -m examples.resource_validation --output artifacts/resource-validation``.
These synthetic cases verify correctness and resource behavior, with no speedup gate.
The separate frozen priority benchmark remains the performance acceptance gate.
"""

import argparse
import json
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.stats import norm

from examples.joint_inference import AT, inputs
from examples.repeated_execution import PROTOCOL, array_bytes, peak_rss
from ocbf._values import canonical_json, portable
from ocbf.api import (
    ExecutionSession,
    compile_model,
    evaluate,
    infer,
    plan_inference,
    prepare_evidence,
    requirements_for,
    warm_start_from,
)
from ocbf.belief.posterior import QueryRequirements
from ocbf.diagnostics.monte_carlo import assessment
from ocbf.evidence import EvidenceAction
from ocbf.inference.contracts import InferencePolicy, SamplingConfig
from ocbf.io import dumps
from ocbf.model.spec import ContinuousVariableSpec, FactorSpec, VariableSpec
from ocbf.reliability.config import ParameterSet

VALIDATION_PROTOCOL = {
    "version": "resource-workloads-v1",
    "source_status": PROTOCOL["source_status"],
    "cache_bytes": PROTOCOL["cache_bytes"],
    "seeds": [113, 229],
    "exact_atol": 1e-10,
    "exact_rtol": 1e-9,
    "sampling": {"chains": 4, "warmup": 1000, "draws": 4000},
    "probability_tolerance": 0.03,
    "count_tolerance": 0.06,
    "numerical_gate": "absolute tolerance AND error <= 4 * MCSE + .002",
    "workloads": [
        "replacement",
        "retraction",
        "late-arrival",
        "support-expansion",
        "bounded-hybrid",
        "constrained-sampling",
        "warm-start-trust",
    ],
    "timing_scope": "per-stage measurements; infer includes mandatory plan revalidation",
    "performance_gate": "none; the frozen priority benchmark owns that gate",
}


def measure(spec, queries, policy, *, session=None, seed=113, warm_start=None, preparation=0):
    phases = {"prepare_evidence": preparation}

    def timed(name, fn):
        started = time.perf_counter()
        value = fn()
        phases[name] = time.perf_counter() - started
        return value

    requirements = (
        requirements_for(queries) if queries else QueryRequirements(capabilities=("joint_draws",))
    )
    model = timed("compile", lambda: compile_model(spec, session=session))
    plan = timed(
        "plan",
        lambda: plan_inference(model, requirements=requirements, policy=policy, session=session),
    )
    result = timed(
        "infer",
        lambda: infer(
            model,
            requirements=requirements,
            policy=policy,
            rng=np.random.default_rng(seed),
            session=session,
            warm_start=warm_start,
        ),
    )
    answer = timed("query", lambda: evaluate(result, queries, session=session)) if queries else None
    encoded = timed("export", lambda: dumps(answer if answer else result))
    return (
        model,
        result,
        answer,
        {
            "phases_seconds": phases,
            "total_seconds": sum(phases.values()),
            "peak_rss_bytes": peak_rss(),
            "retained_array_bytes": array_bytes(result),
            "export_bytes": len(encoded.encode()),
            "variables": len(model.variables),
            "factors": len(model.factors),
            "model_id": model.model_id,
            "plan_id": result.plan_id,
            "run_id": result.run_id,
            "policy": portable(policy),
            "plan": portable(plan),
            "seed": seed,
            "engine_version": "1",
            "execution": portable(result.execution),
            "cache": dict(session.stats) if session else None,
            "estimates": portable(answer.estimates) if answer else None,
        },
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Freeze these supplemental criteria before running them; the original priority
    # workload baseline is preserved in artifacts/repeated-execution/baseline.json.
    protocol_path = output / "validation-protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != VALIDATION_PROTOCOL:
        raise ValueError("existing validation protocol differs; select a new output directory")
    protocol_path.write_text(json.dumps(VALIDATION_PROTOCOL, indent=2) + "\n")
    spec, queries, settings, records = inputs()
    exact_policy = InferencePolicy(engine="reference_elimination")
    checks, measurements = [], []
    correction = replace(
        records[0],
        revision_id="a:2",
        previous_revision="a:1",
        action=EvidenceAction.REPLACE,
        known_at=AT + timedelta(seconds=1),
        payload={**records[0].payload, "zone": "z2"},
    )
    retraction = replace(
        correction,
        revision_id="a:3",
        previous_revision="a:2",
        action=EvidenceAction.RETRACT,
        known_at=AT + timedelta(seconds=2),
    )
    scenarios = [
        ("replacement", (*records, correction), AT + timedelta(seconds=1)),
        ("retraction", (*records, correction, retraction), AT + timedelta(seconds=2)),
        ("late-arrival", records[1:], AT),
        ("late-arrival", records, AT + timedelta(seconds=3)),
    ]
    with ExecutionSession(max_cache_bytes=PROTOCOL["cache_bytes"]) as session:
        measure(spec, queries, exact_policy, session=session)
        for name, history, cutoff in scenarios:
            before = time.perf_counter()
            updated = inputs(history, as_of=cutoff)[0]
            preparation = time.perf_counter() - before
            fresh = measure(updated, queries, exact_policy, preparation=preparation)
            reused = measure(
                updated, queries, exact_policy, session=session, preparation=preparation
            )
            for mode, record in (("fresh", fresh[3]), ("session", reused[3])):
                measurements.append({"workload": name, "mode": mode, **record})
            joint_scope = ("association:a", "association:b")
            np.testing.assert_allclose(
                fresh[1].posterior.joint(joint_scope).probabilities,
                reused[1].posterior.joint(joint_scope).probabilities,
                atol=1e-10,
                rtol=1e-9,
            )
            assert canonical_json(fresh[2].estimates) == canonical_json(reused[2].estimates)
            checks.append(
                {
                    "workload": name,
                    "same_target_and_answers": True,
                    "normalizer_error": abs(
                        fresh[1].posterior.log_normalizer - reused[1].posterior.log_normalizer
                    ),
                }
            )

        empty = prepare_evidence((), as_of=AT, context=spec.context, interpreters={})
        expanded = replace(
            spec,
            evidence=empty,
            variables=(VariableSpec("association", ("a", "b")),),
            factors=(),
            parameters=ParameterSet(priors={"association": (0.4, 0.6)}),
        )
        for domain, weights in ((("a", "b"), (0.4, 0.6)), (("a", "b", "late"), (0.2, 0.3, 0.5))):
            changed = replace(
                expanded,
                variables=(VariableSpec("association", domain),),
                parameters=ParameterSet(priors={"association": weights}),
            )
            _, result, _, record = measure(changed, None, exact_policy, session=session)
            np.testing.assert_allclose(
                result.posterior.marginal("association").probabilities, weights, atol=1e-10
            )
            measurements.append({"workload": "support-expansion", "mode": "session", **record})
        checks.append({"workload": "support-expansion", "matches_analytic_probabilities": True})

        hybrid = replace(
            spec,
            evidence=empty,
            variables=(
                VariableSpec("mode", ("a", "b")),
                ContinuousVariableSpec("time", 0, 2, "seconds", "synthetic"),
            ),
            parameters=ParameterSet(priors={"mode": (0.4, 0.6)}),
            factors=(
                FactorSpec(
                    "report",
                    ("mode", "time"),
                    "linear_gaussian",
                    {
                        "coefficients": {"time": 1.0},
                        "observed": 1.0,
                        "sd": 1.0,
                        "mode_key": "mode",
                        "by_mode": {"a": {"sd": 0.5}, "b": {"sd": 3.0}},
                    },
                ),
            ),
        )
        hp = InferencePolicy(engine="reference_hybrid", max_components=2, max_continuous_dim=1)
        expected = np.array(
            [0.4 * norm.pdf(1, scale=np.sqrt(4.25)), 0.6 * norm.pdf(1, scale=np.sqrt(13))]
        )
        for mode, cache in (("fresh", None), ("cold", session), ("warm", session)):
            _, result, _, record = measure(hybrid, None, hp, session=cache)
            np.testing.assert_allclose(
                result.posterior.log_normalizer, np.log(expected.sum()), atol=1e-10, rtol=1e-9
            )
            measurements.append({"workload": "bounded-hybrid", "mode": mode, **record})
        checks.append({"workload": "bounded-hybrid", "matches_analytic_normalizer": True})

    config = SamplingConfig(**VALIDATION_PROTOCOL["sampling"])
    sample_policy = InferencePolicy(engine="blocked", sampling=config)
    changed = replace(spec, parameters=settings["cautious"])
    exact = evaluate(
        infer(compile_model(changed), requirements=requirements_for(queries), policy=exact_policy),
        queries,
    )
    for seed in VALIDATION_PROTOCOL["seeds"]:
        with ExecutionSession(max_cache_bytes=PROTOCOL["cache_bytes"]) as session:
            old = measure(spec, queries, sample_policy, session=session, seed=seed)
            hint = warm_start_from(old[0], old[1])
            cold = measure(changed, queries, sample_policy, seed=seed + 1)
            warm = measure(
                changed, queries, sample_policy, session=session, seed=seed + 2, warm_start=hint
            )
            for mode, measured in (("fresh", cold), ("warm-start", warm)):
                measurements.append({"workload": "warm-start-trust", "mode": mode, **measured[3]})
                for actual, reference in zip(measured[2].estimates, exact.estimates, strict=True):
                    if actual.value is not None:
                        quantities = [
                            (
                                actual.name,
                                actual.value,
                                reference.value,
                                actual.numerical,
                                0.06 if actual.unit == "executions" else 0.03,
                            )
                        ]
                    else:
                        quantities = [
                            (
                                f"priority:{job}:{rank}",
                                r["probability"],
                                reference.distribution["ranks"][job][rank]["probability"],
                                r["numerical"],
                                0.03,
                            )
                            for job, ranks in actual.distribution["ranks"].items()
                            for rank, r in ranks.items()
                        ]
                    for quantity, value, expected_value, numerical, tolerance in quantities:
                        error = abs(value - expected_value)
                        mcse = numerical["mcse"]
                        passed = (
                            error <= tolerance and mcse is not None and error <= 4 * mcse + 0.002
                        )
                        checks.append(
                            {
                                "workload": "warm-start-trust",
                                "seed": seed,
                                "mode": mode,
                                "quantity": quantity,
                                "absolute_error": error,
                                "numerical": portable(numerical),
                                "tolerance": tolerance,
                                "passed": passed,
                            }
                        )

    keys = tuple(f"item:{i}" for i in range(12))
    constrained = replace(
        spec,
        evidence=empty,
        variables=tuple(VariableSpec(k, (False, True)) for k in keys),
        parameters=ParameterSet(priors={k: (0.5, 0.5) for k in keys}),
        factors=(FactorSpec("bounded-count", keys, "count", {"lo": 3, "hi": 9}, role="support"),),
    )
    policy = replace(sample_policy, engine="auto", max_table_states=512, max_clique_states=512)
    with ExecutionSession(max_cache_bytes=PROTOCOL["cache_bytes"]) as session:
        _, result, _, record = measure(constrained, None, policy, session=session, seed=12012)
        counts = np.stack(tuple(result.posterior.draw_set.values.values())).sum(axis=0)
        numerical = assessment(counts)
        error = abs(float(counts.mean()) - 6.0)
        checks.append(
            {
                "workload": "constrained-sampling",
                "quantity": "expected-count",
                "absolute_error": error,
                "tolerance": 0.08,
                "numerical": portable(numerical),
                "passed": error <= 0.08
                and error <= 4 * numerical["mcse"] + 0.002
                and bool(((counts >= 3) & (counts <= 9)).all()),
            }
        )
        measurements.append({"workload": "constrained-sampling", "mode": "session", **record})
    report = {"protocol": VALIDATION_PROTOCOL, "measurements": measurements, "checks": checks}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    failed = [c for c in checks if c.get("passed") is False]
    print(
        json.dumps(
            {"measurements": len(measurements), "checks": len(checks), "failed": failed}, indent=2
        )
    )
    if failed:
        raise RuntimeError("supplemental numerical acceptance gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/resource-validation")
    run(parser.parse_args().output)
