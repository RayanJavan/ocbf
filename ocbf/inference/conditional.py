"""Concrete conditioned target evaluator. No query or source dependencies."""

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.special import logsumexp

from ocbf._values import freeze
from ocbf.errors import BudgetExceeded, CapabilityError, IncompatibleModel, NumericalFailure
from ocbf.model.dependencies import execution_identity, reusable_model
from ocbf.runtime.cache import cached
from ocbf.runtime.control import checkpoint

from .elimination import EliminationPosterior, LogTable, normalizer, preflight


@dataclass(frozen=True)
class GaussianComponent:
    state: object
    keys: tuple
    mean: np.ndarray
    covariance: np.ndarray
    log_mass: float

    def __post_init__(self):
        for name in ("state", "mean", "covariance"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def draw_assignment(self, rng):
        values = rng.multivariate_normal(self.mean, self.covariance) if self.keys else ()
        return {**self.state, **dict(zip(self.keys, values, strict=True))}


@dataclass(frozen=True)
class ConditionedPosterior:
    log_normalizer: float
    fixed: object
    finite: object = None
    components: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "fixed", freeze(self.fixed))
        object.__setattr__(self, "components", tuple(self.components))

    def draw_assignment(self, rng):
        if self.finite is not None:
            return {**self.fixed, **self.finite.draw_assignment(rng)}
        weights = np.exp(np.array([c.log_mass for c in self.components]) - self.log_normalizer)
        return self.components[rng.choice(len(weights), p=weights)].draw_assignment(rng)


def assess_conditioning(model, fixed_keys, policy, *, store=None, control=None):
    checkpoint(control, "conditioning.assess")
    remaining_finite = {k: d for k, d in model.domains.items() if k not in fixed_keys}
    remaining_real = tuple(k for k in model.continuous if k not in fixed_keys)
    if len(remaining_real) > policy.max_continuous_dim:
        raise BudgetExceeded("conditional Gaussian dimension exceeds budget")
    scopes = [tuple(k for k in f.scope if k in remaining_finite) for f in model.factors]
    if not model.continuous:
        return preflight(remaining_finite, scopes, policy, store=store, control=control)
    for factor in model.factors:
        if set(factor.scope) & set(remaining_real) and not hasattr(
            model._kernels[factor.family], "gaussian_terms"
        ):
            raise CapabilityError("factor cannot be analytically integrated", key=factor.key)
    count = math.prod(map(len, remaining_finite.values()))
    if count > policy.max_components or count > policy.max_table_states:
        raise BudgetExceeded("conditional Gaussian mixture exceeds component budget")
    return tuple(remaining_finite), count, count


def condition_target(model, fixed, policy, *, store=None, control=None):
    store = store if store is None or reusable_model(model) else None
    order, _, _ = assess_conditioning(model, fixed.keys(), policy, store=store, control=control)
    if not model.continuous:
        return cached(
            store,
            "condition.finite",
            (
                execution_identity(model) if store is not None else model.model_id,
                fixed,
                order,
                policy.max_table_states,
                policy.max_clique_states,
                policy.max_joint_states,
            ),
            lambda: _condition_target(model, fixed, policy, order, store, control),
        )
    return _condition_target(model, fixed, policy, order, store, control)


def _condition_target(model, fixed, policy, order, store, control):
    if not model.continuous:
        domains = {k: d for k, d in model.domains.items() if k not in fixed}
        tables = []
        for factor in model.factors:
            checkpoint(control, "conditioning.factor", key=factor.key)
            from ocbf.model.batch import finite_table

            scope, table = finite_table(model, factor, fixed, max_states=policy.max_table_states)
            if np.isnan(table).any() or np.isposinf(table).any():
                raise NumericalFailure("nonfinite conditioned factor", key=factor.key)
            tables.append(LogTable(scope, table))
        z, conditionals = normalizer(tables, domains, order, store=store, control=control)
        return ConditionedPosterior(
            z, fixed, EliminationPosterior(domains, conditionals, z, policy)
        )
    keys = tuple(k for k in model.continuous if k not in fixed)
    finite = tuple(k for k in model.domains if k not in fixed)
    components = []
    for values in product(*(model.domains[k] for k in finite)):
        checkpoint(
            control,
            "conditioning.gaussian",
            completed=len(components),
            allocation_bytes=8 * (4 * len(keys) ** 2 + 4 * len(keys)),
        )
        state = {**fixed, **dict(zip(finite, values, strict=True))}
        centers = {k: model.continuous[k].prior_mean for k in keys}
        q, h, constant = np.zeros((len(keys), len(keys))), np.zeros(len(keys)), 0.0
        for factor in model.factors:
            kernel = model._kernels[factor.family]
            if hasattr(kernel, "gaussian_terms"):
                fq, fh, fc = kernel.gaussian_terms(factor, {**state, **centers}, keys)
                q += fq
                h += fh
                constant += fc
            else:
                constant += model.factor_log_density(factor, state)
        if constant == -math.inf:
            continue
        if not np.isfinite(q).all() or not np.isfinite(h).all() or not math.isfinite(constant):
            raise NumericalFailure("nonfinite Gaussian canonical terms")
        try:
            chol = np.linalg.cholesky(q)
            mean = np.linalg.solve(chol.T, np.linalg.solve(chol, h))
            covariance = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(len(keys))))
        except np.linalg.LinAlgError as exc:
            raise NumericalFailure(
                "conditional Gaussian precision is not positive definite"
            ) from exc
        mass = (
            constant
            + 0.5 * h @ mean
            + 0.5 * len(keys) * math.log(2 * math.pi)
            - np.log(np.diag(chol)).sum()
        )
        components.append(
            GaussianComponent(
                state, keys, mean + np.array([centers[k] for k in keys]), covariance, float(mass)
            )
        )
    if not components:
        raise IncompatibleModel("conditioned target has zero mass")
    z = float(logsumexp([c.log_mass for c in components]))
    if not math.isfinite(z):
        raise NumericalFailure("undefined conditional normalizer")
    return ConditionedPosterior(z, fixed, components=tuple(components))
