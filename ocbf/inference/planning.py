"""Capability/cost selection over supplied interfaces, without concrete engine imports."""

from dataclasses import replace

from ocbf.belief.posterior import QueryRequirements
from ocbf.errors import CapabilityError

from .contracts import InferencePolicy


def plan_inference(model, *, requirements=None, policy=None, engines, store=None, control=None):
    policy = policy or InferencePolicy()
    requirements = requirements or QueryRequirements()
    names = (
        ("reference_elimination", "reference_hybrid", "blocked")
        if policy.engine == "auto"
        else (policy.engine,)
    )
    considered = []
    for name in names:
        engine = engines.get(name)
        if engine is None:
            error = CapabilityError("engine not registered", key=name)
        else:
            try:
                if control is not None and not hasattr(engine, "assess_with_context"):
                    raise CapabilityError(
                        "engine does not declare cooperative planning controls", key=name
                    )
                plan = (
                    engine.assess_with_context(
                        model, requirements, policy, store=store, control=control
                    )
                    if hasattr(engine, "assess_with_context")
                    else engine.assess(model, requirements, policy)
                )
                return replace(
                    plan,
                    details={
                        **plan.details,
                        "selection": tuple(considered),
                        "selected_reason": "first compatible route within declared budgets",
                        "requested_policy": policy.engine,
                    },
                )
            except CapabilityError as exc:
                error = exc
        if policy.engine != "auto":
            raise error
        considered.append({"engine": name, "reason": str(error)})
    raise CapabilityError(
        "no admitted engine: " + "; ".join(f"{r['engine']}: {r['reason']}" for r in considered)
    )
