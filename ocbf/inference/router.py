"""Composition facade for explicit routing and opt-in capability-based selection."""

from ocbf.belief.posterior import QueryRequirements
from ocbf.errors import CapabilityError
from ocbf.inference.contracts import InferencePolicy
from ocbf.runtime.control import ExecutionControl, checkpoint


def plan_inference(
    model, *, requirements=None, policy=None, engines=None, control=None, store=None
):
    from .planning import plan_inference as plan

    if engines is None:
        from .registry import builtin_engines

        engines = builtin_engines()
    checkpoint(control, "inference.plan")
    result = plan(
        model,
        requirements=requirements,
        policy=policy,
        engines=engines,
        store=store,
        control=control,
    )
    checkpoint(control, "inference.planned")
    return result


def infer(
    model,
    *,
    requirements=None,
    policy=None,
    engines=None,
    rng=None,
    store=None,
    control=None,
    warm_start=None,
):
    policy = policy or InferencePolicy()
    requirements = requirements or QueryRequirements()
    if engines is None:
        from .registry import builtin_engines

        engines = builtin_engines()
    plan = plan_inference(
        model,
        requirements=requirements,
        policy=policy,
        engines=engines,
        control=control,
        store=store,
    )
    engine = engines[plan.engine]
    # The sampler owns max_seconds itself and preserves its partial-draw behavior.
    if policy.max_seconds is not None and plan.engine != "blocked":
        control = (
            control.limited(policy.max_seconds)
            if control is not None
            else ExecutionControl(timeout_seconds=policy.max_seconds)
        )
    if hasattr(engine, "solve_with_context"):
        return engine.solve_with_context(
            model, plan, policy, rng, store=store, control=control, warm_start=warm_start
        )
    if control is not None or warm_start is not None:
        raise CapabilityError(
            "engine does not declare cooperative controls or warm starts", key=engine.name
        )
    return engine.solve(model, plan, policy, rng)
