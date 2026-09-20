"""Route selection and solve dispatch over supplied engines."""

from dataclasses import replace

from ocbf.belief.posterior import QueryRequirements
from ocbf.errors import CapabilityError
from ocbf.inference.contracts import InferencePolicy
from ocbf.runtime.control import ExecutionControl, checkpoint

from .registry import AUTO_ROUTE


def _ready(policy, requirements, engines):
    """Apply the defaults shared by both public entries: policy, requirements, engine set."""
    if engines is None:
        from .registry import builtin_engines

        engines = builtin_engines()
    return policy or InferencePolicy(), requirements or QueryRequirements(), engines


def _plan(model, requirements, policy, engines, *, store=None, control=None):
    """First engine, in requested/auto order, compatible within budgets; inputs already defaulted."""
    checkpoint(control, "inference.plan")
    names = AUTO_ROUTE if policy.engine == "auto" else (policy.engine,)
    considered = []
    for name in names:
        engine = engines.get(name)
        if engine is None:
            error = CapabilityError("engine not registered", key=name)
        else:
            try:
                cooperative = getattr(engine, "cooperative", False)
                if control is not None and not cooperative:
                    raise CapabilityError(
                        "engine does not declare cooperative planning controls", key=name
                    )
                plan = (
                    engine.assess(model, requirements, policy, store=store, control=control)
                    if cooperative
                    else engine.assess(model, requirements, policy)
                )
                checkpoint(control, "inference.planned")
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


def plan_inference(
    model, *, requirements=None, policy=None, engines=None, control=None, store=None
):
    policy, requirements, engines = _ready(policy, requirements, engines)
    return _plan(model, requirements, policy, engines, store=store, control=control)


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
    policy, requirements, engines = _ready(policy, requirements, engines)
    plan = _plan(model, requirements, policy, engines, store=store, control=control)
    engine = engines[plan.engine]
    # The sampler owns max_seconds itself and preserves its partial-draw behavior.
    if policy.max_seconds is not None and plan.engine != "blocked":
        control = (
            control.limited(policy.max_seconds)
            if control is not None
            else ExecutionControl(timeout_seconds=policy.max_seconds)
        )
    if getattr(engine, "cooperative", False):
        return engine.solve(
            model, plan, policy, rng, store=store, control=control, warm_start=warm_start
        )
    if control is not None or warm_start is not None:
        raise CapabilityError(
            "engine does not declare cooperative controls or warm starts", key=engine.name
        )
    return engine.solve(model, plan, policy, rng)
