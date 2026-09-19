"""Route selection over the registry and injected engine mappings, exercised through the router."""

from dataclasses import dataclass

import pytest

from ocbf.errors import CapabilityError
from ocbf.inference.contracts import ExecutionPlan, InferencePolicy
from ocbf.inference.registry import AUTO_ROUTE, builtin_engines
from ocbf.inference.router import plan_inference


@dataclass
class _StubEngine:
    """Minimal engine: yields a plan naming itself, or declines, ignoring the model."""

    name: str
    admits: bool = True

    def assess(self, model, requirements, policy):
        if not self.admits:
            raise CapabilityError("stub declines route", key=self.name)
        return ExecutionPlan(self.name, (), 1, 0, ("marginal",))


def test_auto_route_names_are_registered():
    """The auto route can only name engines the registry actually builds."""
    registered = builtin_engines()
    assert set(AUTO_ROUTE) <= set(registered)
    # gtsam_exact (optional native) and bp (approximate marginals) stay out of auto by design.
    assert "gtsam_exact" not in AUTO_ROUTE
    assert "bp" not in AUTO_ROUTE


def test_auto_selects_first_admitting_engine_over_injected_mapping():
    engines = {name: _StubEngine(name, admits=(name == "blocked")) for name in AUTO_ROUTE}
    plan = plan_inference(object(), policy=InferencePolicy(engine="auto"), engines=engines)
    assert plan.engine == "blocked"
    declined = [entry["engine"] for entry in plan.details["selection"]]
    assert declined == list(AUTO_ROUTE[: AUTO_ROUTE.index("blocked")])


def test_default_policy_routes_to_gtsam_exact():
    """A call with no policy is defaulted once in the entry to the gtsam_exact route."""
    plan = plan_inference(object(), engines={"gtsam_exact": _StubEngine("gtsam_exact")})
    assert plan.engine == "gtsam_exact"


def test_explicit_unregistered_engine_raises():
    with pytest.raises(CapabilityError):
        plan_inference(object(), policy=InferencePolicy(engine="ghost"), engines={})
