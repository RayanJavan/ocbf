"""Synthetic same-target comparison of exact elimination and joint sampling.

Run python -m examples.joint_inference --output artifacts/joint-inference.
All inputs are synthetic. Real source admission is a separate integration command.
"""

import argparse
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from examples._shared.study import reproduce, run_study, verify_reproduction
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
from ocbf.evidence import AdmissionIssue, EvidenceAction, EvidenceRecord, Observation
from ocbf.inference.contracts import InferencePolicy, SamplingConfig
from ocbf.io import write_bundle
from ocbf.model.spec import FactorSpec, ModelSpec, VariableSpec
from ocbf.queries import Endpoint, ExecutionProjection, QueryBundle, QuerySpec
from ocbf.reliability.channels import AssociationParameters
from ocbf.reliability.config import TrustRule, resolve_parameters
from ocbf.schema import Schema
from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext

AT = datetime(2026, 9, 8, 12, tzinfo=UTC)


@dataclass(frozen=True)
class SyntheticAssociationInterpreter:
    """Bind the two fixture labels to candidate durations; no producer contract is implied."""

    name: str = "synthetic-association"
    version: str = "1"

    def interpret(self, record, context):
        report = record.payload
        if report.get("entity_id") not in ("a", "b") or report.get("zone") not in ("z1", "z2"):
            return (), (AdmissionIssue(record.revision_id, "uninterpreted", "unknown fixture label"),)
        return (Observation(
            record.record_id, record.source_id, "zone_association", record.producer_version,
            "association_mode", ("association:" + report["entity_id"],),
            {"z1": "short", "z2": "long"}[report["zone"]], (record.revision_id,),
            "synthetic-association@1", report["information_id"],
        ),), ()


def inputs(records=None, *, as_of=AT):
    context = SemanticContext.from_universe(
        UniverseBuilder(Schema([], [])).build(),
        provenance={"status": "synthetic numerical fixture"},
    )
    variables = [VariableSpec("start", (True,))]
    priors, factors, executions = {}, [], []
    for job in ("a", "b"):
        key = "association:" + job
        variables.append(VariableSpec(key, ("short", "long")))
        priors[key] = (0.5, 0.5)
        for end in ("short", "long"):
            flag = job + ":" + end
            variables.append(VariableSpec(flag, (False, True)))
            priors[flag] = (0.5, 0.5)
            values = np.where(
                np.array([[False, True]]) == (np.array(["short", "long"])[:, None] == end),
                0.0,
                -np.inf,
            )
            factors.append(
                FactorSpec("bind:" + flag, (key, flag), log_values=values, role="support")
            )
        executions.append(
            ExecutionProjection(
                job,
                job,
                (Endpoint("start", AT - timedelta(hours=1)),),
                (
                    Endpoint(job + ":short", AT - timedelta(minutes=50)),
                    Endpoint(job + ":long", AT),
                ),
            )
        )
    if records is None:
        records = tuple(
            EvidenceRecord(
                job,
                job + ":1",
                "rtls",
                "1",
                {
                    "entity_id": job,
                    "zone": "z1",
                    "timestamp": AT.isoformat(),
                    "information_id": "measurement:" + job,
                },
                AT,
                provenance={"physical_origin": "synthetic_fixture"},
            )
            for job in ("a", "b")
        )
    evidence = prepare_evidence(
        records,
        as_of=as_of,
        context=context,
        interpreters={("rtls", "1"): SyntheticAssociationInterpreter()},
    )
    assumptions = {
        "dependence": "one manually specified source mode shared by both reports",
        "status": "synthetic; zone-to-execution bindings are illustrative assumptions",
    }
    settings = {}
    for name, modes in (("nominal", (0.8, 0.2)), ("cautious", (0.5, 0.5))):
        values = AssociationParameters(
            ("good", "reversed"),
            modes,
            (((0.95, 0.05), (0.05, 0.95)), ((0.05, 0.95), (0.95, 0.05))),
            "source_mode",
            "manual synthetic assumption",
        )
        settings[name] = resolve_parameters(
            evidence.observations,
            default=TrustRule(values, "manual synthetic setting"),
            priors=priors,
            assumptions=assumptions,
        )
    spec = ModelSpec(
        context,
        evidence,
        settings["nominal"],
        tuple(variables),
        tuple(factors),
        scope=assumptions,
        ground_structure=False,
    )
    rank = QuerySpec(
        "priority",
        "priority_distribution",
        tuple(executions),
        AT - timedelta(hours=2),
        AT + timedelta(minutes=1),
        {"reference_id": "synthetic-30min", "threshold_minutes": {"a": 30, "b": 30}},
    )
    whole = replace(
        rank, name="whole_job", kind="whole_job_conformance", executions=(executions[0],)
    )
    count = replace(rank, name="exception_count", kind="expected_exception_count")
    return spec, QueryBundle((whole, count, rank)), settings, records


def larger_inputs():
    """Twelve coupled bits, conditioned on 3 <= count <= 9; 4096 input states."""
    from ocbf.reliability.config import ParameterSet

    base, _, _, _ = inputs()
    keys = tuple(f"item:{i}" for i in range(12))
    spec = replace(
        base,
        evidence=prepare_evidence((), as_of=AT, context=base.context, interpreters={}),
        parameters=ParameterSet(priors={k: (0.5, 0.5) for k in keys}),
        variables=tuple(VariableSpec(k, (False, True)) for k in keys),
        factors=(FactorSpec("bounded-count", keys, "count", {"lo": 3, "hi": 9}, role="support"),),
        scope={"status": "synthetic twelve-variable constrained numerical fixture"},
    )
    return compile_model(spec), QueryRequirements((keys,), (), (("joint_draws",),))


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec, queries, settings, records = inputs()
    model = compile_model(spec)
    requirements = requirements_for(queries)
    exact = infer(
        model, requirements=requirements, policy=InferencePolicy(engine="reference_elimination")
    )
    expected = evaluate(exact, queries)
    config = SamplingConfig()
    policy = InferencePolicy(engine="blocked", sampling=config)
    started = time.perf_counter()
    sampled = infer(
        model, requirements=requirements, policy=policy, rng=np.random.default_rng(20260912)
    )
    answers = evaluate(sampled, queries)
    seconds = time.perf_counter() - started
    comparisons = []
    for a, b in zip(answers.estimates, expected.estimates, strict=True):
        if a.value is not None:
            error = abs(a.value - b.value)
            # Predeclared demonstration tolerances: .03 probability / .06 expected count.
            tolerance = 0.06 if a.unit == "executions" else 0.03
            mcse = a.numerical["mcse"]
            passed = error <= tolerance and mcse is not None and error <= 4 * mcse + 0.002
            comparisons.append(
                {
                    "query": a.name,
                    "exact": b.value,
                    "sampled": a.value,
                    "absolute_error": error,
                    "tolerance": tolerance,
                    "mcse": mcse,
                    "passed": passed,
                }
            )
    plan = plan_inference(
        model, requirements=requirements, policy=replace(policy, engine="auto", max_clique_states=2)
    )
    budgeted = infer(
        model,
        requirements=requirements,
        policy=replace(policy, engine="auto", max_clique_states=2),
        rng=np.random.default_rng(321),
    )
    write_bundle(
        output / "comparison.json",
        {
            "exact": expected,
            "sampled": answers,
            "checks": tuple(comparisons),
            "seconds": seconds,
            "policy": policy,
            "exceeds_exact_budget": plan,
            "budgeted_result": evaluate(budgeted, queries),
        },
    )
    write_bundle(output / "posterior.json", sampled)
    larger_model, larger_requirements = larger_inputs()
    larger_policy = replace(policy, engine="auto", max_table_states=512, max_clique_states=512)
    larger_plan = plan_inference(
        larger_model, requirements=larger_requirements, policy=larger_policy
    )
    larger = infer(
        larger_model,
        requirements=larger_requirements,
        policy=larger_policy,
        rng=np.random.default_rng(12012),
    )
    values = np.stack(tuple(larger.posterior.draw_set.values.values()))
    counts = values.sum(axis=0)
    # Symmetry gives E[count]=6 exactly under this independent binomial reference.
    numerical = assessment(counts)
    count_error = abs(float(counts.mean()) - 6)
    larger_passed = (
        count_error <= 0.08
        and numerical["mcse"] is not None
        and count_error <= 4 * numerical["mcse"] + 0.002
        and bool(((counts >= 3) & (counts <= 9)).all())
    )
    write_bundle(
        output / "larger-comparison.json",
        {
            "variables": 12,
            "unconditioned_input_states": 4096,
            "plan": larger_plan,
            "policy": larger_policy,
            "result": larger,
            "exact_expected_count": 6.0,
            "sampled_expected_count": float(counts.mean()),
            "tolerance_count": 0.08,
            "numerical": numerical,
            "passed": larger_passed,
        },
    )
    status = (
        "Synthetic numerical reference only; no physical calibration or business admission claim."
    )
    nominal = run_study(
        spec,
        queries,
        settings,
        policy=policy,
        output=output / "replay/nominal",
        source_status=status,
        seed=99,
    )
    verify_reproduction(nominal, reproduce(output / "replay/nominal/inputs.json"))
    correction = replace(
        records[0],
        revision_id="a:2",
        previous_revision="a:1",
        action=EvidenceAction.REPLACE,
        payload={**records[0].payload, "zone": "z2"},
    )
    retraction = replace(
        correction,
        revision_id="a:3",
        previous_revision="a:2",
        action=EvidenceAction.RETRACT,
        known_at=AT + timedelta(hours=1),
    )
    for name, history, cutoff in (
        ("correction", (*records, correction), AT),
        ("retraction", (*records, correction, retraction), AT + timedelta(hours=1)),
    ):
        updated, bundle, trust, _ = inputs(history, as_of=cutoff)
        run_study(
            updated,
            bundle,
            trust,
            policy=InferencePolicy(engine="reference_elimination"),
            output=output / "replay" / name,
            source_status=status,
        )
    report = [
        "# Joint-inference numerical comparison",
        "",
        status,
        "",
        f"Four chains, {config.warmup} warmup and {config.draws} retained sweeps per chain.",
        f"Sampling plus query evaluation: {seconds:.3f} seconds.",
        "",
        "| Query | Exact | Sampled | MCSE | Passed |",
        "|---|---:|---:|---:|---|",
    ]
    report.extend(
        f"| {r['query']} | {r['exact']:.6f} | {r['sampled']:.6f} | {r['mcse']:.6f} | {r['passed']} |"
        for r in comparisons
    )
    report.extend(
        [
            "",
            "The bounded auto route rejected finite exact elimination under the stated clique budget and selected blocked inference.",
            f"Twelve-variable constrained case: 4096 input states exceed a 512-state exact budget; sampled count {counts.mean():.6f}, exact 6, MCSE {numerical['mcse']:.6f}, passed {larger_passed}.",
            "Nominal stochastic replay uses recorded seeds; correction/retraction demonstrations fully recompute the changed target.",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not all(r["passed"] for r in comparisons) or not larger_passed:
        raise RuntimeError("same-target numerical comparison did not meet predeclared tolerances")
    print(
        "comparison passed",
        {
            "seconds": seconds,
            "checks": comparisons,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/joint-inference")
    options = parser.parse_args()
    main(options.output)
