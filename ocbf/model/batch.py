"""Vectorized finite table construction. Loops scale with factors/variables, not states."""

import math

import numpy as np

from ocbf.errors import BudgetExceeded, CapabilityError


def finite_table(model, factor, fixed, *, max_states):
    scope = tuple(k for k in factor.scope if k not in fixed)
    shape = tuple(len(model.domains[k]) for k in scope)
    if math.prod(shape) > max_states:
        raise BudgetExceeded("conditioned factor table exceeds budget", key=factor.key)
    if factor.family == "table":
        index = tuple(
            model.domains[k].index(fixed[k]) if k in fixed else slice(None) for k in factor.scope
        )
        return scope, factor.log_values[index] + factor.log_offset
    arrays = {}
    for key in factor.scope:
        if key in fixed:
            arrays[key] = np.asarray(fixed[key])
        else:
            axes = [1] * len(scope)
            axes[scope.index(key)] = len(model.domains[key])
            arrays[key] = np.asarray(model.domains[key]).reshape(axes)
    kernel = model._kernels[factor.family]
    if hasattr(kernel, "log_values"):
        return scope, np.broadcast_to(kernel.log_values(factor, arrays, model.domains), shape)
    # Extension fallback is deliberately bounded; large extensions must supply a batch port.
    if math.prod(shape) > 4096:
        raise CapabilityError("large factor extension needs vectorized log_values", key=factor.key)
    out = np.empty(shape)
    for index in np.ndindex(shape):
        state = {**fixed, **{k: model.domains[k][i] for k, i in zip(scope, index, strict=True)}}
        out[index] = model.factor_log_density(factor, state)
    return scope, out
