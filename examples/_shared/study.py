"""Batch composition over resolved contracts. No Mammut fields enter inference or queries."""

from pathlib import Path

import numpy as np

from ocbf._values import canonical_json
from ocbf.api import compare_settings, compile_model, evaluate, infer, requirements_for
from ocbf.errors import ValidationError
from ocbf.io import read_bundle, write_bundle

from .report import render_results


def run_study(
    spec,
    queries,
    settings,
    *,
    policy,
    output,
    source_status,
    seed=None,
    channels=None,
    factors=None,
    engines=None,
    evaluators=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Export before running: the calculation can be repeated without opening a source.
    inputs = {
        "spec": spec,
        "queries": queries,
        "settings": settings,
        "policy": policy,
        "source_status": source_status,
        "seed": seed,
        "extensions": {
            name: tuple(sorted(registry)) if registry is not None else "builtins"
            for name, registry in (
                ("channels", channels),
                ("factors", factors),
                ("engines", engines),
                ("evaluators", evaluators),
            )
        },
    }
    write_bundle(output / "inputs.json", inputs)
    restored = read_bundle(output / "inputs.json")
    rng = np.random.default_rng(seed) if seed is not None else None
    model = compile_model(restored["spec"], channels=channels, factors=factors)
    result = infer(
        model,
        requirements=requirements_for(queries, evaluators=evaluators),
        policy=policy,
        engines=engines,
        rng=rng,
    )
    answers = evaluate(result, queries, evaluators=evaluators, rng=rng)
    comparisons = compare_settings(
        spec,
        queries,
        settings,
        policy=policy,
        channels=channels,
        factors=factors,
        engines=engines,
        evaluators=evaluators,
        rng=rng,
    )
    write_bundle(
        output / "results.json",
        {"posterior": result, "answers": answers, "comparison": comparisons},
    )
    write_bundle(
        output / "manifest.json",
        {
            "model_id": model.model_id,
            "manifest": model.manifest,
            "variables": model.variables,
            "factors": model.factors,
            "decoding": model.decoding,
        },
    )
    (output / "report.md").write_text(
        render_results(answers, label="Operation assessment", source_status=source_status),
        encoding="utf-8",
    )
    return answers


def reproduce(input_path, *, channels=None, factors=None, engines=None, evaluators=None):
    inputs = read_bundle(input_path)
    for name, registry in (
        ("channels", channels),
        ("factors", factors),
        ("engines", engines),
        ("evaluators", evaluators),
    ):
        if inputs.get("extensions", {}).get(name, "builtins") != "builtins" and registry is None:
            raise ValidationError(
                "replay requires explicitly supplied extension registry", key=name
            )
    seed = inputs.get("seed")
    rng = np.random.default_rng(seed) if seed is not None else None
    model = compile_model(inputs["spec"], channels=channels, factors=factors)
    result = infer(
        model,
        requirements=requirements_for(inputs["queries"], evaluators=evaluators),
        policy=inputs["policy"],
        engines=engines,
        rng=rng,
    )
    return evaluate(result, inputs["queries"], evaluators=evaluators, rng=rng)


def verify_reproduction(first, second):
    # Run identity deliberately changes; scientific identities and numeric estimates must not.
    a = (first.model_id, first.bundle_id, first.estimates)
    b = (second.model_id, second.bundle_id, second.estimates)
    if canonical_json(a) != canonical_json(b):
        raise ValidationError("reproduction changed scientific identities or numerical answers")
