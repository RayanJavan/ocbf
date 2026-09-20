"""Bounded exact discrete-mode enumeration and analytical Gaussian integration."""

import uuid
from dataclasses import dataclass

import numpy as np

from ocbf._values import fingerprint
from ocbf.belief.posterior import InferenceResult, JointDrawSet, draw_dtype
from ocbf.errors import BudgetExceeded, CapabilityError
from ocbf.inference.conditional import assess_conditioning, condition_target
from ocbf.inference.contracts import ExecutionPlan
from ocbf.model.dependencies import reusable_model
from ocbf.runtime.control import checkpoint


@dataclass(frozen=True)
class HybridPosterior:
    conditional: object
    policy: object

    @property
    def log_normalizer(self):
        return self.conditional.log_normalizer

    def draw(self, scope, *, rng=None, size=None):
        return self.draw_with_context(scope, rng=rng, size=size)

    def draw_with_context(self, scope, *, rng=None, size=None, control=None):
        if rng is None:
            raise CapabilityError("hybrid joint drawing requires an explicit random stream")
        size = 4000 if size is None else size
        if type(size) is not int or size < 1:
            raise CapabilityError("hybrid drawing requires a positive integer size")
        components = self.conditional.components
        first = components[0]
        if not set(scope) <= set(first.state) | set(first.keys):
            raise CapabilityError("query names absent candidate variables")
        dtypes = {
            k: np.dtype(float)
            if k in first.keys
            else draw_dtype(tuple(c.state[k] for c in components))
            for k in scope
        }
        if (
            size * (sum(d.itemsize for d in dtypes.values()) + 8 * len(first.keys) + 16)
            > self.policy.max_draw_bytes
        ):
            raise BudgetExceeded("hybrid draw allocation exceeds budget")
        checkpoint(
            control,
            "draw.hybrid.allocate",
            allocation_bytes=size
            * (sum(d.itemsize for d in dtypes.values()) + 8 * len(first.keys) + 16),
        )
        weights = np.exp(np.array([c.log_mass for c in components]) - self.log_normalizer)
        choices = rng.choice(len(components), size=size, p=weights)
        values = {k: np.empty(size, dtype=dtypes[k]) for k in scope}
        for i in np.unique(choices):
            checkpoint(control, "draw.hybrid.component", component=int(i))
            component, selected = components[i], choices == i
            samples = (
                rng.multivariate_normal(
                    component.mean, component.covariance, size=int(selected.sum())
                )
                if component.keys
                else None
            )
            for key in scope:
                values[key][selected] = (
                    samples[:, component.keys.index(key)]
                    if key in component.keys
                    else component.state[key]
                )
        return JointDrawSet(
            {k: v[None, :] for k, v in values.items()},
            "iid",
            {"origin": "exact conditional Gaussian mixture"},
        )


@dataclass(frozen=True)
class HybridEngine:
    name: str = "reference_hybrid"
    version: str = "1"
    cooperative = True

    def assess(self, model, requirements, policy, *, store=None, control=None):
        capabilities = ("joint_draws", "normalizer")
        if not requirements.satisfied_by(capabilities):
            raise CapabilityError("hybrid route supplies joint draws and a normalizer")
        if not model.continuous:
            raise CapabilityError("hybrid route requires a continuous target")
        for scope in requirements.scopes:
            if not set(scope) <= set(model.domains) | set(model.continuous):
                raise CapabilityError("query names absent variables")
        order, inputs, clique = assess_conditioning(
            model, set(), policy, store=store, control=control
        )
        return ExecutionPlan(
            self.name,
            order,
            inputs,
            clique,
            capabilities,
            {"integration": "analytical conditional Gaussian; no truncation approximation"},
        )

    def solve(self, model, plan, policy, rng, *, store=None, control=None, warm_start=None):
        if warm_start is not None:
            raise CapabilityError("hybrid exact inference does not use sampler warm starts")
        if store is not None and not reusable_model(model):
            store = None
        conditional = condition_target(model, {}, policy, store=store, control=control)
        return InferenceResult(
            model.model_id,
            fingerprint("plan", (model.model_id, plan, policy, self.version)),
            "run:" + str(uuid.uuid4()),
            HybridPosterior(conditional, policy),
            plan.capabilities,
            "exact-on-admitted-hybrid-model",
            {"log_normalizer": conditional.log_normalizer},
            {**model.manifest, "decoding": model.decoding, "execution_plan": plan},
        )
