"""Reproducible workflow, evidence admission, and independent empirical references."""

import json
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from examples.fixed_parameters import AT, example_inputs, run_example
from examples.mammut_retrospective.admission import (
    EntryBinding,
    ProductionCommand,
    reconcile_production,
)
from examples.mammut_retrospective.reference import (
    ClosedBaselineOperation,
    bind_thresholds,
    freeze_reference,
)
from ocbf.api import compare_settings, compile_model, infer, prepare_evidence
from ocbf.errors import CapabilityError, EvidenceConflict, ValidationError
from ocbf.inference.contracts import InferencePolicy
from ocbf.io import dumps, loads
from ocbf.model.spec import FactorSpec

POLICY = InferencePolicy(engine="reference_elimination")


def test_action_specific_production_identifiers_require_a_crosswalk():
    opened = ProductionCommand("open-record", "open", "chassis", station="S", entered_at=AT)
    closed = ProductionCommand(
        "close-record", "close", "chassis", entry_id="db-17", exited_at=AT + timedelta(minutes=10)
    )
    missing = reconcile_production([opened, closed, opened])
    assert len(missing["unresolved"]) == 1
    binding = EntryBinding("db-17", "open-record", "verified-db-row-17")
    linked = reconcile_production([opened, closed], [binding])
    assert len(linked["episodes"]["open-record"]) == 2 and not linked["unresolved"]
    assert (
        linked["reconciliation_id"]
        == reconcile_production([closed, opened, opened], [binding])["reconciliation_id"]
    )
    with pytest.raises(EvidenceConflict):
        reconcile_production([opened, replace(closed, chassis_id="other")], [binding])


def test_frozen_reference_deduplication_missing_groups_and_baseline_isolation():
    operations = tuple(
        ClosedBaselineOperation(
            str(i),
            "s",
            "p",
            AT - timedelta(days=3),
            AT - timedelta(days=3) + timedelta(minutes=m),
            (str(i),),
        )
        for i, m in enumerate((10, 20, 40))
    )
    kwargs = {
        "start": AT - timedelta(days=4),
        "end": AT - timedelta(days=2),
        "provenance": "synthetic reference test",
    }
    reference = freeze_reference(operations, **kwargs)
    assert loads(dumps(reference))["reference_id"] == reference["reference_id"]
    assert reference["groups"][0]["threshold_minutes"] == pytest.approx(36)
    assert reference["groups"][0]["sample_size"] == 3
    assert (
        reference["reference_id"]
        == freeze_reference((*reversed(operations), operations[0]), **kwargs)["reference_id"]
    )
    bound = bind_thresholds(reference, {"op": ("s", "p"), "unknown": ("missing", "group")})
    assert dict(bound["threshold_minutes"]) == {"op": pytest.approx(36)}
    with pytest.raises(ValidationError):
        freeze_reference([replace(operations[0], start=AT, end=AT)], **kwargs)


def test_owned_portable_round_trip_keeps_scientific_identities_and_no_code_import():
    spec, queries, settings, _ = example_inputs()
    restored = loads(dumps({"spec": spec, "queries": queries, "settings": settings}))
    assert compile_model(restored["spec"]).model_id == compile_model(spec).model_id
    assert restored["queries"].bundle_id == queries.bundle_id
    with pytest.raises(TypeError):
        restored["spec"].parameters.priors["bad"] = (1,)
    with pytest.raises(ValidationError):
        loads(
            json.dumps(
                {"format": "ocbf-neutral-bundle-v1", "value": {"type": "os.system", "fields": {}}}
            )
        )
    table = FactorSpec("scalar", (), log_values=np.asarray(3.0), log_offset=-1)
    copied = loads(dumps(table))
    assert copied.log_values.shape == () and copied.log_offset == -1
    with pytest.raises(ValueError):
        copied.log_values.flags.writeable = True


def test_trust_comparison_holds_interpretation_priors_reference_and_support_fixed():
    spec, queries, settings, _ = example_inputs()
    comparison = compare_settings(spec, queries, settings, policy=POLICY)
    assert len({answers.bundle_id for _, _, answers in comparison.settings}) == 1
    assert len({answers.model_id for _, _, answers in comparison.settings}) == 2
    with pytest.raises(ValidationError):
        compare_settings(
            spec, queries, {"different-prior": replace(spec.parameters, priors={})}, policy=POLICY
        )


def test_full_replay_recomputes_and_replaced_observation_never_contributes_twice(tmp_path):
    answers = run_example(tmp_path)
    assert answers["nominal"].estimates[0].value == pytest.approx(0.5)
    assert (
        0
        < answers["correction"].estimates[0].value
        < answers["retraction"].estimates[0].value
        < 0.5
    )
    assert "long:1" not in answers["correction"].estimates[0].evidence_ids
    assert "long:2" in answers["correction"].estimates[0].evidence_ids
    assert "long:2" not in answers["retraction"].estimates[0].evidence_ids
    assert len({answer.model_id for answer in answers.values()}) == 3


def test_unknown_producer_and_unsupported_silence_do_not_become_likelihoods():
    spec, _, _, records = example_inputs()
    interpreted = prepare_evidence(records, as_of=AT, context=spec.context, interpreters={})
    assert not interpreted.observations and len(interpreted.issues) == 3
    original = spec.evidence.observations[0]
    bad = replace(
        original, applicability={"expected": (True, True), "silence_without_opportunity": True}
    )
    with pytest.raises(CapabilityError):
        compile_model(replace(spec, evidence=replace(spec.evidence, observations=(bad,))))


def test_joint_functional_uses_retained_dependence():
    spec, _, _, _ = example_inputs()
    result = infer(compile_model(spec), policy=POLICY)
    factor = spec.factors[0]
    assert result.posterior.expectation(factor.scope, lambda s: all(s.values())) == 0


def test_canonical_identities_distinguish_typed_values_from_lookalike_payloads():
    from ocbf._values import fingerprint, portable

    for typed in (AT, np.asarray([1.0]), float("-inf")):
        assert fingerprint("test", typed) != fingerprint("test", portable(typed))
    assert fingerprint("test", True) != fingerprint("test", 1)


def test_retransmission_and_future_reports_do_not_change_the_current_model():
    spec, _, _, records = example_inputs()
    future = replace(
        records[0], record_id="future", revision_id="future:1", known_at=AT + timedelta(days=1)
    )
    reordered, _, _, _ = example_inputs((*reversed(records), records[0], future))
    assert reordered.evidence.snapshot.snapshot_id == spec.evidence.snapshot.snapshot_id
    assert compile_model(reordered).model_id == compile_model(spec).model_id


def test_all_settings_validate_before_any_inference(monkeypatch):
    from ocbf import api

    spec, queries, settings, _ = example_inputs()
    calls = []
    monkeypatch.setattr(api, "infer", lambda *args, **kwargs: calls.append(True))
    invalid = {"a-valid": settings["nominal"], "z-invalid": replace(spec.parameters, priors={})}
    with pytest.raises(ValidationError):
        api.compare_settings(spec, queries, invalid, policy=POLICY)
    assert not calls


def test_endpoint_time_change_requires_new_model_while_reference_change_does_not():
    from ocbf.api import evaluate
    from ocbf.queries import QueryBundle

    spec, queries, _, _ = example_inputs()
    model = compile_model(spec)
    result = infer(model, policy=POLICY)
    query = queries.queries[0]
    execution = query.executions[0]
    changed = replace(
        execution, starts=(replace(execution.starts[0], time=AT - timedelta(hours=3)),)
    )
    with pytest.raises(CapabilityError):
        evaluate(result, QueryBundle((replace(query, executions=(changed,)),)))
    reference = replace(query, reference={"threshold_minutes": {"op": 120}})
    assert evaluate(result, QueryBundle((reference,))).estimates[0].value == 0
    assert model.model_id == result.model_id
