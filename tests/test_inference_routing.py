"""Route selection over the registry and injected engine mappings, exercised through the router."""

from dataclasses import dataclass

import pytest

from ocbf.errors import CapabilityError
from ocbf.inference.contracts import ExecutionPlan, InferencePolicy
from ocbf.inference.registry import AUTO_ROUTE, builtin_engines
from ocbf.inference.router import plan_inference
from ocbf.runtime.control import ExecutionControl


@dataclass
class _StubEngine:
    """Minimal engine: yields a plan naming itself, or declines, ignoring the model."""

    name: str
    admits: bool = True

    def assess(self, model, requirements, policy):
        if not self.admits:
            raise CapabilityError("stub declines route", key=self.name)
        return ExecutionPlan(self.name, (), 1, 0, ("marginal",))


def test_auto_route_covers_every_registered_engine_except_the_documented_two():
    registered = builtin_engines()
    assert set(AUTO_ROUTE) <= set(registered)
    # A newly registered engine silently left out of (or wrongly added to) the route fails here.
    assert set(registered) - set(AUTO_ROUTE) == {"gtsam_exact", "bp"}


def test_auto_selects_first_admitting_engine_over_injected_mapping():
    engines = {name: _StubEngine(name, admits=(name == "blocked")) for name in AUTO_ROUTE}
    plan = plan_inference(object(), policy=InferencePolicy(engine="auto"), engines=engines)
    assert plan.engine == "blocked"
    declined = [entry["engine"] for entry in plan.details["selection"]]
    assert declined == list(AUTO_ROUTE[: AUTO_ROUTE.index("blocked")])


def test_auto_raises_when_every_route_declines():
    engines = {name: _StubEngine(name, admits=False) for name in AUTO_ROUTE}
    with pytest.raises(CapabilityError, match="no admitted engine"):
        plan_inference(object(), policy=InferencePolicy(engine="auto"), engines=engines)


def test_default_policy_routes_to_gtsam_exact():
    """A call with no policy uses the default policy, whose engine is gtsam_exact."""
    plan = plan_inference(object(), engines={"gtsam_exact": _StubEngine("gtsam_exact")})
    assert plan.engine == "gtsam_exact"


def test_control_with_control_incapable_engine_is_rejected():
    engines = {"gtsam_exact": _StubEngine("gtsam_exact")}
    with pytest.raises(CapabilityError):
        plan_inference(
            object(),
            policy=InferencePolicy(engine="gtsam_exact"),
            engines=engines,
            control=ExecutionControl(),
        )


def test_explicit_unregistered_engine_raises():
    with pytest.raises(CapabilityError):
        plan_inference(object(), policy=InferencePolicy(engine="ghost"), engines={})
