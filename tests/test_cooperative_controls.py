"""Declared cooperative-execution capability.

The dispatchers thread execution context (store/control, and warm starts for solving) to
targets that *declare* cooperation, and refuse it -- loudly, with CapabilityError -- to those
that do not. Capability is read from a declared ``cooperative`` attribute, never sniffed from
the presence of a ``*_with_context`` method.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from ocbf.belief.posterior import InferenceResult, JointDrawSet, JointTable, QueryRequirements
from ocbf.errors import CapabilityError
from ocbf.inference.contracts import ExecutionPlan, InferencePolicy
from ocbf.inference.registry import builtin_engines
from ocbf.inference.router import infer, plan_inference
from ocbf.queries.aggregation import exact_data
from ocbf.queries.evaluate import builtin_evaluators, evaluate
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


# --- posterior and evaluator (query) seam --------------------------------------------------


def _bundle(kind="k"):
    query = SimpleNamespace(kind=kind, version="1", name="q")
    return SimpleNamespace(queries=(query,), bundle_id="b")


@dataclass
class _StubPosterior:
    """Posterior recording the control handed to its joint/draw operations."""

    cooperative: bool = True
    seen: dict = field(default_factory=dict)

    def joint(self, scope, *, store=None, control=None):
        self.seen["joint_control"] = control
        scope = tuple(scope)
        return JointTable(scope, tuple((0,) for _ in scope), np.ones((1,) * len(scope)))

    def draw(self, scope, *, rng=None, size=None, control=None):
        self.seen["draw_control"] = control
        return JointDrawSet({k: np.zeros((1, 1), dtype=np.int64) for k in scope}, "iid")


@dataclass
class _StubEvaluator:
    """Evaluator recording the control handed to its evaluate/evaluate_on operations."""

    cooperative: bool = True
    version: str = "1"
    seen: dict = field(default_factory=dict)

    def requirements(self, query):
        return QueryRequirements((("x",),), ("joint_draws",), ())

    def evaluate(self, result, query, *, store=None, control=None):
        self.seen["evaluate_control"] = control
        return "estimate"

    def evaluate_on(self, result, query, data, *, store=None, control=None):
        self.seen["evaluate_on_control"] = control
        return "estimate"


def test_real_posteriors_and_evaluator_declare_cooperative():
    """A forgotten declaration on a real posterior or evaluator fails here."""
    from ocbf.belief.samples import SamplePosterior
    from ocbf.inference.adapters.approximate import MarginalPosterior
    from ocbf.inference.adapters.hybrid import HybridPosterior
    from ocbf.inference.elimination import EliminationPosterior
    from ocbf.queries.aggregation import ExecutionEvaluator

    assert EliminationPosterior.cooperative is True
    assert HybridPosterior.cooperative is True
    assert SamplePosterior.cooperative is True
    assert MarginalPosterior.cooperative is False
    assert ExecutionEvaluator.cooperative is True


def test_builtin_evaluators_declare_cooperative_capability():
    evaluators = builtin_evaluators()
    assert all(isinstance(e.cooperative, bool) and e.cooperative for e in evaluators.values())


def test_exact_data_threads_control_to_cooperative_posterior():
    posterior = _StubPosterior(cooperative=True)
    control = ExecutionControl()
    exact_data(posterior, ("x",), control=control)
    assert posterior.seen["joint_control"] is control


def test_exact_data_rejects_control_for_noncooperative_posterior():
    with pytest.raises(CapabilityError):
        exact_data(_StubPosterior(cooperative=False), ("x",), control=ExecutionControl())


def test_evaluate_threads_control_to_cooperative_posterior_and_evaluator():
    posterior = _StubPosterior(cooperative=True)
    evaluator = _StubEvaluator(cooperative=True)
    result = InferenceResult("m", "p", "r", posterior, ("joint_draws",))
    control = ExecutionControl()
    evaluate(
        result,
        _bundle(),
        evaluators={"k": evaluator},
        rng=np.random.default_rng(0),
        control=control,
    )
    assert posterior.seen["draw_control"] is control
    assert evaluator.seen["evaluate_on_control"] is control


def test_evaluate_rejects_control_for_noncooperative_evaluator():
    result = InferenceResult("m", "p", "r", _StubPosterior(), ("joint_draws",))
    with pytest.raises(CapabilityError):
        evaluate(
            result,
            _bundle(),
            evaluators={"k": _StubEvaluator(cooperative=False)},
            rng=np.random.default_rng(0),
            control=ExecutionControl(),
        )


def test_evaluate_rejects_control_for_noncooperative_posterior():
    result = InferenceResult("m", "p", "r", _StubPosterior(cooperative=False), ("joint_draws",))
    with pytest.raises(CapabilityError):
        evaluate(
            result,
            _bundle(),
            evaluators={"k": _StubEvaluator(cooperative=True)},
            rng=np.random.default_rng(0),
            control=ExecutionControl(),
        )
