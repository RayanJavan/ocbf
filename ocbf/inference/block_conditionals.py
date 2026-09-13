"""Vectorized finite block conditionals over exact execution codecs."""

import numpy as np
from scipy.special import logsumexp

from ocbf.errors import CapabilityError, NumericalFailure
from ocbf.model.dependencies import execution_identity, reusable_model
from ocbf.runtime.cache import cached
from ocbf.runtime.control import checkpoint

from .elimination import LogTable, eliminate, preflight


def draw_finite_block(model, codec, state, block, policy, rng, *, store=None, control=None):
    store = store if store is None or reusable_model(model) else None
    fixed = {k: v for k, v in state.items() if k not in block}
    domains = {k: d for k, d in codec.domains.items() if k not in fixed}
    scopes = [
        tuple(sorted({codec.encoded_key(k) for k in f.scope} & domains.keys()))
        for f in model.factors
    ]
    order, _, _ = preflight(domains, scopes, policy, keep=block, store=store, control=control)
    table = cached(
        store,
        "condition.block",
        (
            execution_identity(model) if store is not None else model.model_id,
            codec,
            fixed,
            block,
            order,
            policy.max_table_states,
            policy.max_clique_states,
            policy.max_joint_states,
        ),
        lambda: _block_table(model, codec, fixed, domains, scopes, order, store, control),
    )
    z = logsumexp(table.values)
    if not np.isfinite(z):
        raise NumericalFailure("valid current state has no finite block conditional")
    choice = rng.choice(table.values.size, p=np.exp(table.values.ravel() - z))
    index = np.unravel_index(choice, table.values.shape)
    return {**state, **{k: domains[k][i] for k, i in zip(table.scope, index, strict=True)}}


def _block_table(model, codec, fixed, domains, scopes, order, store, control):
    tables = []
    for factor, scope in zip(model.factors, scopes, strict=True):
        checkpoint(control, "conditioning.block_factor", key=factor.key)
        shape = tuple(len(domains[k]) for k in scope)
        encoded = dict(fixed)
        for axis, key in enumerate(scope):
            axes = [1] * len(scope)
            axes[axis] = len(domains[key])
            encoded[key] = np.asarray(domains[key]).reshape(axes)
        original = {}
        for key in factor.scope:
            if key in codec.clamped:
                original[key] = np.asarray(codec.clamped[key])
            else:
                mapped = codec.encoded_key(key)
                value = np.asarray(encoded[mapped])
                original[key] = value == key if mapped in codec.choices else value
        if factor.family == "table":
            index = tuple(
                np.argmax(original[k][..., None] == np.asarray(model.domains[k]), axis=-1)
                for k in factor.scope
            )
            values = factor.log_values[index] + factor.log_offset
        else:
            kernel = model._kernels[factor.family]
            if not hasattr(kernel, "log_values"):
                raise CapabilityError(
                    "block conditional requires batch-capable finite kernels", key=factor.key
                )
            values = kernel.log_values(factor, original, model.domains)
        values = np.broadcast_to(values, shape)
        if np.isnan(values).any() or np.isposinf(values).any():
            raise NumericalFailure("nonfinite block conditional factor", key=factor.key)
        tables.append(LogTable(scope, values))
    table, _ = eliminate(tables, domains, order, store=store, control=control)
    return table
