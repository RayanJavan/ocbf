"""Exact finite inference with retained GTSAM elimination conditionals and log scales."""

import uuid
from dataclasses import dataclass

import numpy as np

from ocbf._values import fingerprint
from ocbf.belief.posterior import InferenceResult
from ocbf.errors import CapabilityError, NumericalFailure
from ocbf.inference.contracts import ExecutionPlan
from ocbf.inference.elimination import (
    EliminationPosterior,
    LogTable,
    materialize_factors,
    normalizer,
    preflight,
)
from ocbf.model.dependencies import reusable_model
from ocbf.runtime.control import checkpoint


@dataclass(frozen=True)
class ExactEngine:
    name: str = "gtsam_exact"
    version: str = "1"
    cooperative = True

    def assess(self, model, requirements, policy, *, store=None, control=None):
        if model.continuous:
            raise CapabilityError("finite exact adapter does not integrate continuous variables")
        capabilities = ("marginal", "joint", "expectation", "normalizer", "joint_draws")
        if not requirements.satisfied_by(capabilities):
            raise CapabilityError("exact finite adapter cannot satisfy requested capabilities")
        scopes = [f.scope for f in model.factors]
        order, inputs, clique = preflight(
            model.domains, scopes, policy, store=store, control=control
        )
        for scope in requirements.scopes:
            if not set(scope) <= model.domains.keys():
                raise CapabilityError("query names absent candidate variables")
            if "joint" in requirements.capabilities or "expectation" in requirements.capabilities:
                preflight(model.domains, scopes, policy, keep=scope, store=store, control=control)
        if self.name not in ("gtsam_exact", "reference_elimination"):
            raise CapabilityError("unknown exact engine", key=self.name)
        return ExecutionPlan(self.name, order, inputs, clique, capabilities)

    def solve(self, model, plan, policy, rng, *, store=None, control=None, warm_start=None):
        if warm_start is not None:
            raise CapabilityError("exact inference does not use sampler warm starts")
        # Unknown kernel implementations may depend on unrecorded state.
        if store is not None and not reusable_model(model):
            store = None
        tables = materialize_factors(model, policy, store=store, control=control)
        log_z, reference_conditionals = normalizer(
            tables, model.domains, plan.ordering, store=store, control=control
        )
        conditionals, backend_version = reference_conditionals, "numpy/scipy reference elimination"
        offsets = []
        if self.name == "gtsam_exact":
            from ocbf.backends import gtsam_backend, import_gtsam

            g = import_gtsam()
            backend_version = gtsam_backend().version
            keys = {v.key: i for i, v in enumerate(model.variables)}
            reverse = {i: k for k, i in keys.items()}
            graph = g.DiscreteFactorGraph()
            for table in tables:
                checkpoint(control, "gtsam.convert")
                maximum = float(np.max(table.values))
                # Scalar factors affect Z; they need not be passed to the normalized solver.
                offsets.append(maximum)
                if not table.scope:
                    continue
                probabilities = np.exp(table.values - maximum)
                if np.any(np.isfinite(table.values) & (probabilities == 0)):
                    raise NumericalFailure("finite likelihood underflows during GTSAM conversion")
                graph.add(
                    [(keys[k], len(model.domains[k])) for k in table.scope],
                    probabilities.ravel().tolist(),
                )
            ordering = g.Ordering()
            for k in plan.ordering:
                ordering.push_back(keys[k])
            checkpoint(control, "gtsam.native.begin")
            bayes = graph.eliminateSequential(ordering)
            checkpoint(control, "gtsam.native.end")
            converted = []
            for i in range(bayes.size()):
                checkpoint(control, "gtsam.conditionals", completed=i, total=bayes.size())
                conditional = bayes.at(i)
                indices = list(conditional.keys())
                scope = tuple(reverse[k] for k in indices)
                p = np.zeros(tuple(len(model.domains[k]) for k in scope))
                for assignment, probability in conditional.enumerate():
                    p[tuple(int(assignment[k]) for k in indices)] = probability
                if not np.isfinite(p).all() or (p < 0).any():
                    raise NumericalFailure("GTSAM returned invalid conditionals")
                with np.errstate(divide="ignore"):
                    converted.append(LogTable(scope, np.log(p)))
            conditionals = tuple(converted)
            # Validate each retained local joint against the canonical reference, including
            # parent contexts. Native 0/0 normalization cannot silently enter the result.
            native = EliminationPosterior(model.domains, conditionals, log_z, policy)
            reference = EliminationPosterior(model.domains, reference_conditionals, log_z, policy)
            for t in conditionals:
                checkpoint(control, "gtsam.validate")
                if not np.allclose(
                    native.joint(t.scope, store=store, control=control).probabilities,
                    reference.joint(t.scope, store=store, control=control).probabilities,
                    atol=1e-10,
                    rtol=1e-9,
                ):
                    raise NumericalFailure("GTSAM conditionals disagree with canonical target")
        posterior = EliminationPosterior(model.domains, conditionals, log_z, policy)
        return InferenceResult(
            model.model_id,
            fingerprint("plan", (model.model_id, plan, policy, self.version)),
            "run:" + str(uuid.uuid4()),
            posterior,
            plan.capabilities,
            "exact-on-finite-model",
            {
                "backend": self.name,
                "backend_version": backend_version,
                "log_normalizer": log_z,
                "largest_input_states": plan.largest_input,
                "largest_clique_states": plan.largest_clique,
            },
            {
                **model.manifest,
                "execution_plan": plan,
                "decoding": model.decoding,
                "conversion": {
                    "factor_log_offsets": offsets,
                    "normalization": "canonical log-space elimination",
                    "retention": "neutral exact conditional tables",
                },
            },
        )
