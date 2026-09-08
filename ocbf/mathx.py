"""Numeric primitives for the message-passing hot path.

``scipy.special.logsumexp`` is correct and general -- it dispatches through the array API,
handles complex inputs, and supports weights. In this codebase it is called several times
per belief-propagation iteration on plain real ``float64`` arrays, and profiling put that
generality at roughly two thirds of total engine runtime. This module trades the generality
we do not use for the speed we do.
"""

from __future__ import annotations

import numpy as np


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """``log(sum(exp(a)))`` computed stably along ``axis``.

    Shifts by the row maximum before exponentiating, which is what makes the computation
    stable for the very negative log-potentials that hard factors produce. A non-finite
    maximum is replaced by zero so that an all-[`NEG_INF`][ocbf.model.graph.NEG_INF] row
    returns a finite value rather than ``nan``.
    """
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    total = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    if keepdims or axis is None:
        return total if keepdims else total.reshape(())
    return np.squeeze(total, axis=axis)
