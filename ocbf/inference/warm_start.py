"""Validated initialization hints; prior posteriors never become likelihood factors."""

from dataclasses import dataclass

import numpy as np

from ocbf._values import freeze
from ocbf.errors import CapabilityError, NumericalFailure, ValidationError
from ocbf.model.dependencies import dependency_index
from ocbf.runtime.cache import artifact_key


@dataclass(frozen=True)
class WarmStart:
    """Immutable final-chain states and identities used only as initialization hints."""

    model_id: str
    source_run_id: str
    draw_set_id: str
    structure_id: str
    support_id: str | None
    codec_id: str
    states: tuple

    def __post_init__(self):
        object.__setattr__(self, "states", freeze(self.states))


def warm_start_from(model, result):
    """Extract final assignments from retained blocked chains for explicit reuse.

    Args:
        model (CompiledModel): The exact model that produced ``result``.
        result (InferenceResult): Result retaining MCMC draws and an execution codec.

    Returns:
        hint (WarmStart): Detached final state per chain and compatibility identities.

    Raises:
        ValidationError: The model identity differs from the producing result.
        CapabilityError: Retained blocked-chain assignments or their codec are absent.

    Pass the hint explicitly to a subsequent inference call. That call validates support,
    structure and codec, retains an independently initialized chain, and performs full
    warmup. This helper does not turn a previous posterior into new evidence.
    """
    if result.model_id != model.model_id:
        raise ValidationError("warm-start model and producing result do not match")
    draws = getattr(result.posterior, "draw_set", None)
    plan = result.manifest.get("execution_plan")
    if draws is None or draws.method != "mcmc" or plan is None or "codec" not in plan.details:
        raise CapabilityError("warm starts require retained blocked-chain assignments")
    index = dependency_index(model)
    states = tuple(
        {k: v[chain, -1].item() for k, v in draws.values.items()} for chain in range(draws.shape[0])
    )
    return WarmStart(
        model.model_id,
        result.run_id,
        draws.draw_set_id,
        index.structure_id,
        index.support_id,
        artifact_key(plan.details["codec"]),
        states,
    )


def initializations(model, plan, config, hint):
    """Conservative support equality; expansion, contraction and unknown cases regenerate."""
    if hint is None:
        return {}, {"mode": "fresh", "reason": "no warm start requested"}
    if config.initial_states:
        raise ValidationError("explicit initial_states and warm_start are mutually exclusive")
    index = dependency_index(model)
    details = {
        "source_model_id": hint.model_id,
        "source_run_id": hint.source_run_id,
        "source_draw_set_id": hint.draw_set_id,
        "full_warmup": config.warmup,
    }
    checks = (
        (config.chains >= 2, "warm starts require an independent comparison chain"),
        (index.structure_id == hint.structure_id, "variable/factor structure changed"),
        (
            index.support_id is not None and index.support_id == hint.support_id,
            "support changed or is unknown; regenerate sample population",
        ),
        (artifact_key(plan.details["codec"]) == hint.codec_id, "execution codec changed"),
    )
    for valid, reason in checks:
        if not valid:
            return {}, {**details, "mode": "fresh", "reason": reason}
    codec, selected = plan.details["codec"], {}
    for chain, original in enumerate(hint.states[: config.chains - 1]):
        try:
            if set(original) != {v.key for v in model.variables} or not np.isfinite(
                model.log_density(original)
            ):
                continue
            encoded = dict(original)
            for key, value in codec.clamped.items():
                if encoded.pop(key) != value:
                    raise ValueError("clamped assignment changed")
            for key, choices in codec.choices.items():
                active = [c for c in choices if encoded.pop(c)]
                if len(active) != 1:
                    raise ValueError("invalid categorical encoding")
                encoded[key] = active[0]
            selected[chain] = {k: encoded[k] for k in plan.details["sampled"]}
        except NumericalFailure:
            raise
        except (KeyError, ValueError, TypeError):
            continue
    return selected, {
        **details,
        "mode": "warm" if selected else "fresh",
        "warm_chains": tuple(selected),
        "independent_chain": config.chains - 1,
        "reason": "validated support and semantic assignments; remaining chains initialize afresh",
    }
