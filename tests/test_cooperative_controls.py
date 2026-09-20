"""Declared cooperative-execution capability.

The dispatchers thread execution context (store/control, and warm starts for solving) to
targets that *declare* cooperation, and refuse it -- loudly, with CapabilityError -- to those
that do not. Capability is read from a declared ``cooperative`` attribute, never sniffed from
the presence of a ``*_with_context`` method.
"""

from dataclasses import dataclass, field

import pytest

from ocbf.errors import CapabilityError
from ocbf.inference.contracts import ExecutionPlan, InferencePolicy
from ocbf.inference.registry import builtin_engines
from ocbf.inference.router import infer, plan_inference
from ocbf.runtime.control import ExecutionControl


@dataclass
class _StubEngine:
    """Engine that records the context it is handed and names itself in the plan it yields."""

    name: str = "gtsam_exact"
    cooperative: bool = True
    seen: dict = field(default_factory=dict)

    def assess(self, model, requirements, policy, *, store=None, control=None):
        self.seen["assess_control"] = control
        return ExecutionPlan(self.name, (), 1, 0, ("marginal",))

    def solve(self, model, plan, policy, rng, *, store=None, control=None, warm_start=None):
        self.seen["solve_control"] = control
        self.seen["warm_start"] = warm_start
        return "result:" + self.name


def test_builtin_engines_declare_cooperative_capability():
    """Every built-in engine declares cooperation explicitly; a forgotten declaration fails here."""
    engines = builtin_engines()
    assert all(isinstance(engine.cooperative, bool) for engine in engines.values())
    declined = {name for name, engine in engines.items() if not engine.cooperative}
    assert declined == {"bp"}


def test_plan_threads_control_to_cooperative_engine():
    stub = _StubEngine()
    control = ExecutionControl()
    plan_inference(
        object(),
        policy=InferencePolicy(engine="gtsam_exact"),
        engines={"gtsam_exact": stub},
        control=control,
    )
    assert stub.seen["assess_control"] is control


def test_plan_rejects_control_for_noncooperative_engine():
    stub = _StubEngine(cooperative=False)
    with pytest.raises(CapabilityError):
        plan_inference(
            object(),
            policy=InferencePolicy(engine="gtsam_exact"),
            engines={"gtsam_exact": stub},
            control=ExecutionControl(),
        )


def test_infer_threads_control_and_warm_start_to_cooperative_engine():
    stub = _StubEngine()
    control = ExecutionControl()
    infer(
        object(),
        policy=InferencePolicy(engine="gtsam_exact"),
        engines={"gtsam_exact": stub},
        control=control,
        warm_start={"seed": 1},
    )
    assert stub.seen["solve_control"] is control
    assert stub.seen["warm_start"] == {"seed": 1}


def test_infer_rejects_warm_start_for_noncooperative_engine():
    stub = _StubEngine(name="bp", cooperative=False)
    with pytest.raises(CapabilityError):
        infer(
            object(),
            policy=InferencePolicy(engine="bp"),
            engines={"bp": stub},
            warm_start={"seed": 1},
        )


def test_blocked_engine_is_exempt_from_max_seconds_control_wrapping():
    """The sampler owns its own time budget, so max_seconds must not wrap a control around it."""
    blocked = _StubEngine(name="blocked")
    infer(
        object(),
        policy=InferencePolicy(engine="blocked", max_seconds=5.0),
        engines={"blocked": blocked},
    )
    assert blocked.seen["solve_control"] is None

    other = _StubEngine(name="gtsam_exact")
    infer(
        object(),
        policy=InferencePolicy(engine="gtsam_exact", max_seconds=5.0),
        engines={"gtsam_exact": other},
    )
    assert other.seen["solve_control"] is not None
    assert other.seen["solve_control"].deadline is not None
