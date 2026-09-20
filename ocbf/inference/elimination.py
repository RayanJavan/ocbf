"""Finite log-space elimination, symbolic cost checks and neutral retained conditionals.

This bounded reference also checks normalization before invoking a native backend. It
never treats an impossible model as a uniform distribution.
"""

from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from ocbf._values import freeze
from ocbf.belief.posterior import JointDrawSet, JointTable, draw_dtype
from ocbf.errors import BudgetExceeded, CapabilityError, IncompatibleModel, NumericalFailure
from ocbf.runtime.cache import artifact_key, cached, extension_key
from ocbf.runtime.control import checkpoint


@dataclass(frozen=True)
class LogTable:
    scope: tuple[str, ...]
    values: np.ndarray
    _artifact_id: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "values", freeze(np.asarray(self.values, dtype=float)))

    @property
    def artifact_id(self):
        if self._artifact_id is None:
            object.__setattr__(self, "_artifact_id", artifact_key(self.scope, self.values))
        return self._artifact_id


def ordering_and_cost(domains, scopes, *, keep=()):
    """Deterministic minimum-fill ordering, used unchanged by the selected lowering."""
    graph = {k: set() for k in domains}
    largest_input = max((prod(len(domains[k]) for k in s) for s in scopes), default=1)
    for scope in scopes:
        for key in scope:
            graph[key].update(set(scope) - {key})
    order, largest = [], largest_input
    candidates = set(domains) - set(keep)
    while candidates:

        def score(k):
            n = sorted(graph[k])
            missing = sum(b not in graph[a] for i, a in enumerate(n) for b in n[i + 1 :])
            return missing, prod(len(domains[v]) for v in (k, *n)), k

        key = min(candidates, key=score)
        adjacent = graph.pop(key)
        largest = max(largest, prod(len(domains[k]) for k in (key, *adjacent)))
        for other in adjacent:
            graph[other].discard(key)
            graph[other].update(adjacent - {other})
        order.append(key)
        candidates.remove(key)
    return tuple(order), largest_input, largest


def preflight(domains, scopes, policy, *, keep=(), store=None, control=None):
    checkpoint(control, "elimination.preflight")
    if not set(keep) <= domains.keys():
        raise CapabilityError("query names absent candidate variables")
    order, input_size, clique = cached(
        store,
        "elimination.order",
        (domains, scopes, tuple(keep)),
        lambda: ordering_and_cost(domains, scopes, keep=keep),
    )
    if input_size > policy.max_table_states:
        raise BudgetExceeded(f"input table needs {input_size} states")
    if clique > policy.max_clique_states:
        raise BudgetExceeded(f"elimination clique needs {clique} states")
    if prod(len(domains[k]) for k in keep) > policy.max_joint_states:
        raise BudgetExceeded("requested joint table exceeds max_joint_states")
    checkpoint(control, "elimination.workspace", allocation_bytes=32 * clique)
    return order, input_size, clique


def materialize_factors(model, policy, *, check_elimination=True, store=None, control=None):
    domains = model.domains
    if check_elimination:
        preflight(domains, [f.scope for f in model.factors], policy, store=store, control=control)
    elif any(
        prod(len(domains[k]) for k in f.scope) > policy.max_table_states for f in model.factors
    ):
        raise BudgetExceeded("input factor exceeds max_table_states")
    tables = []
    for f in model.factors:
        checkpoint(
            control,
            "elimination.factor",
            key=f.key,
            allocation_bytes=32 * prod(len(domains[k]) for k in f.scope),
        )
        from ocbf.model.batch import finite_table

        token = extension_key(model._kernels[f.family])

        def compute(f=f):
            _, values = finite_table(model, f, {}, max_states=policy.max_table_states)
            return LogTable(f.scope, values)

        table = cached(
            store if token else None,
            "elimination.factor",
            (f, {k: domains[k] for k in f.scope}, token),
            compute,
            slot=f.key,
        )
        values = table.values
        if np.isnan(values).any() or np.isposinf(values).any():
            raise NumericalFailure("factor evaluation is nonfinite", key=f.key)
        if np.isneginf(values).all():
            raise IncompatibleModel("factor has empty support", key=f.key)
        tables.append(table)
    return tuple(tables)


def _combine(tables, domains):
    scope = tuple(sorted({k for t in tables for k in t.scope}))
    out = np.zeros(tuple(len(domains[k]) for k in scope))
    for table in tables:
        ordered = tuple(k for k in scope if k in table.scope)
        axes = tuple(table.scope.index(k) for k in ordered)
        values = table.values.transpose(axes) if axes else table.values
        shape = tuple(len(domains[k]) if k in table.scope else 1 for k in scope)
        out += values.reshape(shape)
    return LogTable(scope, out)


@dataclass(frozen=True)
class EliminationBucket:
    frontal: str
    incoming: tuple[int, ...]
    scope: tuple[str, ...]
    separator: tuple[str, ...]


@dataclass(frozen=True)
class EliminationSchedule:
    buckets: tuple[EliminationBucket, ...]
    remaining: tuple[int, ...]


def elimination_schedule(scopes, ordering):
    known, active, buckets = list(scopes), list(range(len(scopes))), []
    for key in ordering:
        incoming = tuple(i for i in active if key in known[i])
        scope = tuple(sorted({k for i in incoming for k in known[i]})) or (key,)
        separator = tuple(k for k in scope if k != key)
        buckets.append(EliminationBucket(key, incoming, scope, separator))
        active = [i for i in active if i not in incoming] + [len(known)]
        known.append(separator)
    return EliminationSchedule(tuple(buckets), tuple(active))


def eliminate(tables, domains, ordering, *, retain=False, store=None, control=None):
    if store is not None:
        return _eliminate_reusing(
            tables, domains, ordering, retain=retain, store=store, control=control
        )
    working, conditionals = list(tables), []
    for key in ordering:
        checkpoint(control, "elimination.bucket", key=key)
        bucket = [t for t in working if key in t.scope]
        if not bucket:
            bucket = [LogTable((key,), np.zeros(len(domains[key])))]
        working = [t for t in working if key not in t.scope]
        checkpoint(
            control,
            "elimination.workspace",
            allocation_bytes=32
            * prod(len(domains[k]) for k in {v for t in bucket for v in t.scope}),
        )
        joined = _combine(bucket, domains)
        axis = joined.scope.index(key)
        reduced = logsumexp(joined.values, axis=axis)
        parents = tuple(k for k in joined.scope if k != key)
        working.append(LogTable(parents, reduced))
        if retain:
            # Unreachable parent contexts remain unsupported; no invented conditional mass.
            denominator = np.expand_dims(reduced, axis)
            normalized = np.full(joined.values.shape, -np.inf)
            np.subtract(joined.values, denominator, out=normalized, where=np.isfinite(denominator))
            conditionals.append(LogTable(joined.scope, normalized))
    return _combine(working, domains), tuple(conditionals)


def _eliminate_reusing(tables, domains, ordering, *, retain, store, control):
    scopes = tuple(t.scope for t in tables)
    schedule = cached(
        store,
        "elimination.schedule",
        (scopes, ordering),
        lambda: elimination_schedule(scopes, ordering),
    )
    known, conditionals = list(tables), []
    for bucket in schedule.buckets:
        checkpoint(
            control,
            "elimination.bucket",
            key=bucket.frontal,
            allocation_bytes=32 * prod(len(domains[k]) for k in bucket.scope),
        )
        incoming = tuple(known[i] for i in bucket.incoming) or (
            LogTable((bucket.frontal,), np.zeros(len(domains[bucket.frontal]))),
        )

        def compute(incoming=incoming, bucket=bucket):
            joined = _combine(incoming, domains)
            axis = joined.scope.index(bucket.frontal)
            reduced = logsumexp(joined.values, axis=axis)
            if not retain:
                return LogTable(bucket.separator, reduced), ()
            denominator = np.expand_dims(reduced, axis)
            normalized = np.full(joined.values.shape, -np.inf)
            np.subtract(joined.values, denominator, out=normalized, where=np.isfinite(denominator))
            return LogTable(bucket.separator, reduced), (LogTable(joined.scope, normalized),)

        message, conditional = cached(
            store,
            "elimination.bucket",
            (
                bucket.frontal,
                tuple(t.artifact_id for t in incoming),
                {k: domains[k] for k in bucket.scope},
                retain,
            ),
            compute,
            slot=bucket.frontal,
        )
        known.append(message)
        conditionals.extend(conditional)
    return _combine([known[i] for i in schedule.remaining], domains), tuple(conditionals)


def normalizer(tables, domains, order, *, store=None, control=None):
    scalar, conditionals = eliminate(
        tables, domains, order, retain=True, store=store, control=control
    )
    z = float(scalar.values)
    if z == -np.inf:
        raise IncompatibleModel("joint hard support or likelihoods have zero mass")
    if not np.isfinite(z):
        raise NumericalFailure("undefined model normalizer")
    return z, conditionals


@dataclass(frozen=True)
class EliminationPosterior:
    """Owned neutral conditional tables. Native backend objects are not retained."""

    domains: object
    conditionals: tuple[LogTable, ...]
    log_normalizer: float
    policy: object
    _artifact_id: str | None = field(default=None, init=False, repr=False, compare=False)
    cooperative = True

    def __post_init__(self):
        object.__setattr__(self, "domains", freeze(self.domains))
        object.__setattr__(self, "conditionals", tuple(self.conditionals))

    @property
    def artifact_id(self):
        if self._artifact_id is None:
            object.__setattr__(
                self,
                "_artifact_id",
                artifact_key(
                    self.domains,
                    tuple(t.artifact_id for t in self.conditionals),
                    self.log_normalizer,
                ),
            )
        return self._artifact_id

    def joint(self, scope, *, store=None, control=None):
        scope = tuple(scope)
        if len(set(scope)) != len(scope):
            raise CapabilityError("joint scope contains duplicate variables")
        order, _, _ = preflight(
            self.domains,
            [t.scope for t in self.conditionals],
            self.policy,
            keep=scope,
            store=store,
            control=control,
        )
        if store is not None:
            key = artifact_key(self.artifact_id, scope)
            existing = store.get("posterior.joint", key)
            if existing is not None:
                return existing
        table, _ = eliminate(self.conditionals, self.domains, order, store=store, control=control)
        z = float(logsumexp(table.values))
        if not np.isfinite(z):
            raise NumericalFailure("retained posterior has invalid normalization")
        probs = np.exp(table.values - z)
        axes = tuple(table.scope.index(k) for k in scope)
        if axes:
            probs = probs.transpose(axes)
        result = JointTable(scope, tuple(self.domains[k] for k in scope), probs)
        if store is not None:
            store.put("posterior.joint", key, result)
        return result

    def marginal(self, key):
        return self.joint((key,))

    def draw_assignment(self, rng):
        state = {}
        for conditional in reversed(self.conditionals):
            frontal = tuple(k for k in conditional.scope if k not in state)
            if len(frontal) != 1:
                raise NumericalFailure("conditional reconstruction needs one frontal variable")
            key = frontal[0]
            index = tuple(
                slice(None) if k == key else self.domains[k].index(state[k])
                for k in conditional.scope
            )
            logs = conditional.values[index]
            z = logsumexp(logs)
            if not np.isfinite(z):
                raise NumericalFailure("unreachable conditional context")
            state[key] = self.domains[key][rng.choice(len(logs), p=np.exp(logs - z))]
        return state

    def draw(self, scope, *, rng=None, size=None, control=None):
        if rng is None:
            raise CapabilityError("joint drawing requires an explicit random stream")
        if not set(scope) <= self.domains.keys():
            raise CapabilityError("draw scope names absent variables")
        size = 4000 if size is None else size
        if type(size) is not int or size < 1:
            raise CapabilityError("joint drawing requires a positive integer size")
        dtypes = {k: draw_dtype(self.domains[k]) for k in scope}
        # All ancestral indices and a batched conditional workspace are allocated too.
        bytes_per_draw = (
            sum(dtype.itemsize for dtype in dtypes.values())
            + 8 * len(self.domains)
            + 32 * max(map(len, self.domains.values()), default=1)
        )
        if size * bytes_per_draw > self.policy.max_draw_bytes:
            raise BudgetExceeded("joint draw allocation exceeds budget")
        checkpoint(control, "draw.exact.allocate", allocation_bytes=size * bytes_per_draw)
        indices = {}
        for conditional in reversed(self.conditionals):
            checkpoint(control, "draw.exact.conditional")
            frontal = tuple(k for k in conditional.scope if k not in indices)
            if len(frontal) != 1:
                raise NumericalFailure("invalid conditional reconstruction")
            key = frontal[0]
            index = tuple(
                np.arange(len(self.domains[k]))[None, :] if k == key else indices[k][:, None]
                for k in conditional.scope
            )
            logs = conditional.values[index]
            probabilities = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
            indices[key] = np.minimum(
                (rng.random(size)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1),
                len(self.domains[key]) - 1,
            )
        return JointDrawSet(
            {k: np.asarray(self.domains[k], dtype=dtypes[k])[indices[k]][None, :] for k in scope},
            "iid",
            {"origin": "retained exact finite conditionals"},
        )

    def expectation(self, scope, functional):
        """Integrate a finite, deterministic functional of the requested joint states."""
        table = self.joint(scope)
        total = 0.0
        for index in np.ndindex(table.probabilities.shape):
            probability = float(table.probabilities[index])
            if not probability:
                continue
            state = MappingProxyType(
                {k: d[i] for k, d, i in zip(table.scope, table.domains, index, strict=True)}
            )
            value = float(functional(state))
            if not np.isfinite(value):
                raise NumericalFailure("expectation functional must be finite on posterior support")
            total += probability * value
        return total
