"""Vectorized semantic execution/interval primitives, independent of posterior storage."""

import numpy as np

from ocbf.errors import CapabilityError, ValidationError

SATISFIED, VIOLATED, PENDING, INAPPLICABLE, UNRESOLVED = range(5)
OUTCOMES = ("satisfied", "violated", "pending", "inapplicable", "unresolved")


def condition(tests, state, shape):
    mask = np.ones(shape, dtype=bool)
    for key, expected in tests:
        if key not in state:
            raise CapabilityError("query names absent variable", key=key)
        mask &= state[key] == expected
    return mask


def endpoints(endpoints, state, shape):
    count, time = np.zeros(shape, dtype=int), np.full(shape, np.nan)
    for endpoint in endpoints:
        presence = state[endpoint.event_key]
        if presence.dtype.kind != "b":
            raise CapabilityError(
                "endpoint occurrence requires a Boolean variable", key=endpoint.event_key
            )
        present = presence & condition(endpoint.association, state, shape)
        count += present
        value = (
            state[endpoint.time_key]
            if endpoint.time_key
            else (endpoint.time.timestamp() if endpoint.time is not None else np.nan)
        )
        time = np.where(present, value, time)
    return count, time


def project_execution(execution, state, shape):
    """Reference-independent endpoints/applicability; outcomes are evaluated separately."""
    count, start = endpoints(execution.starts, state, shape)
    end_count, end = endpoints(execution.ends, state, shape)
    applicable = condition(execution.applicable_when, state, shape)
    return count, start, end_count, end, applicable


def interval(execution, state, query, shape, *, projected=None):
    count, start, end_count, end, applicable = (
        project_execution(execution, state, shape) if projected is None else projected
    )
    status = np.full(shape, UNRESOLVED)
    valid_start = applicable & (count == 1) & np.isfinite(start)
    in_window = (
        valid_start
        & (start >= query.window_start.timestamp())
        & (start < query.horizon.timestamp())
    )
    if execution.starts:
        status[applicable & (count == 0)] = INAPPLICABLE
    status[~applicable | (valid_start & ~in_window)] = INAPPLICABLE
    threshold = query.reference.get("threshold_minutes", {}).get(execution.execution_id)
    if threshold is not None:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValidationError("duration reference must be finite and nonnegative")
        status[
            in_window & (end_count == 0) & ((query.horizon.timestamp() - start) / 60 <= threshold)
        ] = PENDING
    known_end = in_window & (end_count == 1) & np.isfinite(end) & (end >= start)
    status[known_end & (end > query.horizon.timestamp())] = PENDING
    closed = known_end & (end <= query.horizon.timestamp())
    status[closed] = SATISFIED
    return status, start, end


def overlap(start, end, conditions, state, shape, horizon):
    """Union of clipped intervals; vectorized across histories, loop only over intervals."""
    total = np.zeros(shape)
    previous = np.full(shape, -np.inf)
    for item in sorted(conditions, key=lambda c: c.start):
        present = condition(item.present_when, state, shape)
        left = np.maximum(np.maximum(start, item.start.timestamp()), previous)
        right = np.minimum(end, (item.end or horizon).timestamp())
        total += np.where(present, np.maximum(0, right - left), 0)
        previous = np.where(present, np.maximum(previous, right), previous)
    return total / 60


def obligation(execution, state, query, shape, *, projected=None):
    status, start, end = interval(execution, state, query, shape, projected=projected)
    closed = status == SATISFIED
    threshold = query.reference.get("threshold_minutes", {}).get(execution.execution_id)
    if threshold is None:
        status[closed] = UNRESOLVED
    else:
        status[closed & ((end - start) / 60 > threshold)] = VIOLATED
    return status


def job_status(statuses):
    statuses = np.stack(statuses)
    result = np.full(statuses.shape[1:], INAPPLICABLE)
    result[np.any(statuses == SATISFIED, axis=0)] = SATISFIED
    result[np.any(statuses == PENDING, axis=0)] = PENDING
    result[np.any(statuses == UNRESOLVED, axis=0)] = UNRESOLVED
    result[np.any(statuses == VIOLATED, axis=0)] = VIOLATED
    return result
