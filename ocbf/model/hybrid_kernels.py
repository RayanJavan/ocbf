"""State evaluation and exact canonical Gaussian algebra for bounded hybrid factors."""

import math
from dataclasses import dataclass

import numpy as np

from ocbf.errors import CapabilityError, ValidationError


@dataclass(frozen=True)
class LinearGaussianKernel:
    name: str = "linear_gaussian"
    version: str = "1"
    operations: tuple = ("state", "gaussian")

    def _parameters(self, factor, state):
        p = dict(factor.parameters)
        if "mode_key" in p:
            mode = state[p["mode_key"]]
            p.update(p["by_mode"][str(mode)])
        if not all(state[k] == v for k, v in p.get("active_when", ())):
            return None, float(p["inactive_log_likelihood"])
        sd = p["sd"]
        if not math.isfinite(sd) or sd <= 0 or not math.isfinite(p["observed"]):
            raise ValidationError(
                "Gaussian factor requires finite location and positive scale", key=factor.key
            )
        return p, 0.0

    def log_density(self, factor, state):
        p, inactive = self._parameters(factor, state)
        if p is None:
            return inactive + factor.log_offset
        residual = p["observed"] - sum(a * state[k] for k, a in p["coefficients"].items())
        return (
            -0.5 * (residual / p["sd"]) ** 2
            - math.log(p["sd"])
            - 0.5 * math.log(2 * math.pi)
            + factor.log_offset
        )

    def gaussian_terms(self, factor, state, keys):
        p, inactive = self._parameters(factor, state)
        d = len(keys)
        if p is None:
            return np.zeros((d, d)), np.zeros(d), inactive + factor.log_offset
        coefficients = p["coefficients"]
        a = np.array([coefficients.get(k, 0.0) for k in keys])
        # Integrated coordinates are centered on supplied prior means, avoiding cancellation
        # of epoch-second quadratic terms. The returned Gaussian is in delta coordinates.
        residual = p["observed"] - sum(v * state[k] for k, v in coefficients.items())
        precision = 1 / p["sd"] ** 2
        return (
            precision * np.outer(a, a),
            precision * residual * a,
            -0.5 * precision * residual**2
            - math.log(p["sd"])
            - 0.5 * math.log(2 * math.pi)
            + factor.log_offset,
        )


@dataclass(frozen=True)
class TemporalOrderKernel:
    name: str = "temporal_order"
    version: str = "1"
    operations: tuple = ("state", "hard_order")

    def log_density(self, factor, state):
        p = factor.parameters
        if not all(state[k] == v for k, v in p.get("active_when", ())):
            return factor.log_offset
        return factor.log_offset if state[p["before"]] <= state[p["after"]] else -math.inf


def validate_hybrid_factor(factor, variables):
    p = factor.parameters
    if factor.family == "linear_gaussian":
        if not p.get("coefficients") or not set(p["coefficients"]) <= set(factor.scope):
            raise ValidationError(
                "Gaussian coefficients must name scoped real variables", key=factor.key
            )
        units = {variables[k].unit for k in p["coefficients"]}
        if len(units) != 1 or any(hasattr(variables[k], "domain") for k in p["coefficients"]):
            raise ValidationError(
                "Gaussian coordinates must have compatible real units", key=factor.key
            )
        if any(not math.isfinite(v) for v in p["coefficients"].values()):
            raise ValidationError("nonfinite Gaussian coefficient", key=factor.key)
        for overrides in p.get("by_mode", {}).values():
            if not set(overrides) <= {"sd", "observed"}:
                raise CapabilityError(
                    "finite Gaussian modes may change location and scale only", key=factor.key
                )
        for q in (p, *({**p, **v} for v in p.get("by_mode", {}).values())):
            if (
                not {"sd", "observed"} <= q.keys()
                or not math.isfinite(q["sd"])
                or q["sd"] <= 0
                or not math.isfinite(q["observed"])
            ):
                raise ValidationError(
                    "Gaussian factor requires finite location and positive scale", key=factor.key
                )
        if p.get("active_when") and (
            "inactive_log_likelihood" not in p or not math.isfinite(p["inactive_log_likelihood"])
        ):
            raise ValidationError(
                "gated Gaussian needs an explicit finite inactive likelihood", key=factor.key
            )
        if "mode_key" in p:
            if p["mode_key"] not in factor.scope or not hasattr(variables[p["mode_key"]], "domain"):
                raise ValidationError(
                    "Gaussian mode must be a scoped finite variable", key=factor.key
                )
            mode = variables[p["mode_key"]]
            if set(p["by_mode"]) != {str(v) for v in mode.domain}:
                raise ValidationError("Gaussian mode parameters are incomplete", key=factor.key)
    elif factor.family == "temporal_order":
        if p["before"] not in factor.scope or p["after"] not in factor.scope:
            raise ValidationError("temporal order endpoints outside factor scope", key=factor.key)
        if variables[p["before"]].unit != variables[p["after"]].unit:
            raise ValidationError("temporal order units disagree", key=factor.key)
        if factor.role != "support":
            raise CapabilityError("temporal_order is an exact support constraint", key=factor.key)
    for key, expected in p.get("active_when", ()):
        if (
            key not in factor.scope
            or not hasattr(variables[key], "domain")
            or expected not in variables[key].domain
        ):
            raise ValidationError("invalid activation condition", key=factor.key)
