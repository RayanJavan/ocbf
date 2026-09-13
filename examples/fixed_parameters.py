"""Synthetic finite Operation assessment with manual trust and reproducible revisions.

Run ``python -m examples.fixed_parameters`` from the checkout. No automatic estimation,
source access, or optional backend is required; use ``--engine gtsam_exact`` to exercise
the retained GTSAM conditionals instead of the local log-space reference.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from examples._shared.study import reproduce, run_study, verify_reproduction
from ocbf.api import prepare_evidence
from ocbf.assertions import AssertionRef as Ref
from ocbf.evidence import EvidenceAction, EvidenceRecord, Observation
from ocbf.inference.contracts import InferencePolicy
from ocbf.io import write_bundle
from ocbf.model.compile import structural_variables
from ocbf.model.spec import FactorSpec, ModelSpec
from ocbf.queries import ConditionInterval, Endpoint, ExecutionProjection, QueryBundle, QuerySpec
from ocbf.reliability.config import ChannelValues, TrustRule, resolve_parameters
from ocbf.schema import E2OQualifier, EventType, Multiplicity, ObjectType, Schema
from ocbf.universe import UniverseBuilder
from ocbf.universe.context import SemanticContext

AT = datetime(2026, 9, 8, 12, tzinfo=UTC)


# --8<-- [start:interpreter]
@dataclass(frozen=True)
class SyntheticInterpreter:
    name: str = "synthetic-endpoint"
    version: str = "1"

    def interpret(self, record, context):
        event = record.payload["event"]
        scope = (str(Ref.event_exists(event)), str(Ref.e2o(event, "subject", "op")))
        return (
            Observation(
                record.record_id,
                record.source_id,
                "endpoint",
                record.producer_version,
                "conjunction",
                scope,
                record.payload["reported"],
                (record.revision_id,),
                "synthetic-endpoint@1",
                record.record_id,
                {"expected": (True, True)},
            ),
        ), ()
# --8<-- [end:interpreter]


def example_inputs(records=None, *, as_of=AT):
    # --8<-- [start:context]
    schema = Schema(
        [EventType("start"), EventType("end")],
        [ObjectType("Operation")],
        e2o=[
            E2OQualifier("subject", kind, "Operation", Multiplicity(0, 1))
            for kind in ("start", "end")
        ],
    )
    builder = UniverseBuilder(schema).add_object("op", "Operation")
    for event, kind in (("start", "start"), ("short", "end"), ("long", "end")):
        builder.add_event(event, type_support={kind})
    context = SemanticContext.from_universe(
        builder.build(),
        provenance={
            "status": "synthetic numerical fixture",
            "candidate_support": "three explicit endpoint alternatives",
        },
    )
    # --8<-- [end:context]
    # --8<-- [start:evidence]
    if records is None:
        records = tuple(
            EvidenceRecord(
                event,
                event + ":1",
                "fixture",
                "1",
                {"event": event, "reported": True},
                AT - timedelta(hours=1),
                provenance={"status": "synthetic"},
            )
            for event in ("start", "short", "long")
        )
    evidence = prepare_evidence(
        records,
        as_of=as_of,
        context=context,
        interpreters={("fixture", "1"): SyntheticInterpreter()},
    )
    # --8<-- [end:evidence]
    # --8<-- [start:parameters]
    priors = {
        v.key: (0.5, 0.5)
        for v in structural_variables(context)
        if len(v.domain) == 2 and not v.key.startswith("event_type(")
    }
    assumptions = {
        "cardinality_role": "normative",
        "endpoint_times": "fixed supplied UTC times",
        "dependence": "distinct synthetic information groups have conditionally independent report errors",
        "association_support": "at most one end linked to this Operation",
        "cohort_conditioning": "complete synthetic universe retained; no exterior dependencies",
    }
    settings = {
        name: resolve_parameters(
            evidence.observations,
            rules=[
                TrustRule(
                    ChannelValues(a, f), "named manual synthetic assumption", family="endpoint"
                )
            ],
            priors=priors,
            assumptions=assumptions,
        )
        for name, a, f in (("nominal", 0.85, 0.15), ("cautious", 0.65, 0.35))
    }
    # --8<-- [end:parameters]
    # --8<-- [start:model]
    end_links = tuple(str(Ref.e2o(e, "subject", "op")) for e in ("short", "long"))
    spec = ModelSpec(
        context,
        evidence,
        settings["nominal"],
        factors=(
            FactorSpec(
                "exclusive_endpoint_association",
                end_links,
                "count",
                {"lo": 0, "hi": 1},
                role="support",
            ),
        ),
        scope={"knowledge_time": as_of, **assumptions},
        decoding={
            "kind": "identity finite assertions",
            "time_unit": "UTC seconds",
            "fixed_endpoints": {
                "start": AT - timedelta(hours=2),
                "short": AT - timedelta(minutes=105),
                "long": AT - timedelta(hours=1),
            },
        },
    )
    # --8<-- [end:model]
    # --8<-- [start:queries]
    endpoint = lambda event, time: Endpoint(
        str(Ref.event_exists(event)), time, ((str(Ref.e2o(event, "subject", "op")), True),)
    )
    execution = ExecutionProjection(
        "op",
        "synthetic-job",
        (endpoint("start", AT - timedelta(hours=2)),),
        (endpoint("short", AT - timedelta(minutes=105)), endpoint("long", AT - timedelta(hours=1))),
    )
    q = QuerySpec(
        "duration",
        "duration_exception",
        (execution,),
        AT - timedelta(days=1),
        AT,
        {"reference_id": "synthetic-reference-v1", "threshold_minutes": {"op": 30}},
    )
    queries = QueryBundle(
        (
            q,
            replace(q, name="count", kind="expected_exception_count"),
            replace(
                q,
                name="tracking overlap",
                kind="exposure",
                coverage_verified=True,
                conditions=(
                    ConditionInterval(
                        AT - timedelta(minutes=110),
                        AT - timedelta(minutes=100),
                        evidence_ids=("synthetic-tracking-interval",),
                    ),
                ),
            ),
        )
    )
    # --8<-- [end:queries]
    return spec, queries, settings, tuple(records)


def run_example(output="artifacts/fixed-parameters", *, engine="reference_elimination"):
    output = Path(output)
    spec, queries, settings, records = example_inputs()
    policy = InferencePolicy(engine=engine)
    status = "Synthetic numerical example only; no factory accuracy or verified factory result."
    nominal = run_study(
        spec, queries, settings, policy=policy, output=output / "nominal", source_status=status
    )
    verify_reproduction(nominal, reproduce(output / "nominal/inputs.json"))
    correction = replace(
        records[-1],
        revision_id="long:2",
        previous_revision="long:1",
        action=EvidenceAction.REPLACE,
        payload={"event": "long", "reported": False},
        known_at=AT,
    )
    retraction = replace(
        correction,
        revision_id="long:3",
        previous_revision="long:2",
        action=EvidenceAction.RETRACT,
        known_at=AT + timedelta(hours=1),
    )
    answers = {"nominal": nominal}
    for name, history, as_of in (
        ("correction", (*records, correction), AT),
        ("retraction", (*records, correction, retraction), AT + timedelta(hours=1)),
    ):
        revised, bundle, trust, _ = example_inputs(history, as_of=as_of)
        answers[name] = run_study(
            revised,
            bundle,
            trust,
            policy=policy,
            output=output / name,
            source_status=status + " Injected replay demonstration, not a real correction.",
        )
    write_bundle(
        output / "replay.json",
        {
            "kind": "injected synthetic replay",
            "answers": answers,
            "reproduction": "identical scientific inputs and numerical outputs",
        },
    )
    return answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/fixed-parameters")
    parser.add_argument(
        "--engine",
        default="reference_elimination",
        choices=("reference_elimination", "gtsam_exact"),
    )
    args = parser.parse_args()
    answers = run_example(args.output, engine=args.engine)
    for name, result in answers.items():
        print(name, {estimate.name: estimate.value for estimate in result.estimates})
