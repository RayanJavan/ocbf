"""Cooperative execution control and assessments. No process killing or root logging."""

import math
import time
from contextlib import contextmanager
from threading import Event

from ocbf.errors import ExecutionCancelled, ExecutionStopped, ResourceExhausted, ValidationError

from .contracts import ExecutionAssessment


class ExecutionControl:
    """A deadline applies across calls sharing this control; workspace limits are advisory.

    max_work_bytes bounds each declared allocation estimate, not process RSS. Native
    uninterruptible calls are checked on entry/return; the caller owns scheduling.
    """

    def __init__(
        self,
        *,
        timeout_seconds=None,
        deadline=None,
        max_work_bytes=None,
        progress=None,
        cancelled=None,
    ):
        if timeout_seconds is not None:
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValidationError("timeout must be finite and positive")
            deadline = (
                min(deadline, time.monotonic() + timeout_seconds)
                if deadline is not None
                else time.monotonic() + timeout_seconds
            )
        if deadline is not None and not math.isfinite(deadline):
            raise ValidationError("deadline must be finite monotonic time")
        if max_work_bytes is not None and (type(max_work_bytes) is not int or max_work_bytes < 1):
            raise ValidationError("workspace byte budget must be a positive integer")
        self.deadline, self.max_work_bytes = deadline, max_work_bytes
        self.progress, self.cancelled = progress, cancelled
        self._event = Event()

    def cancel(self):
        self._event.set()

    @property
    def is_cancelled(self):
        return self._event.is_set() or bool(self.cancelled is not None and self.cancelled())

    def limited(self, seconds):
        """A child deadline preserves its parent's cancellation and allocation policy."""
        return ExecutionControl(
            timeout_seconds=seconds,
            deadline=self.deadline,
            max_work_bytes=self.max_work_bytes,
            progress=self.progress,
            cancelled=lambda: self.is_cancelled,
        )

    def checkpoint(self, stage, *, allocation_bytes=0, **progress):
        if self.is_cancelled:
            raise ExecutionCancelled("execution cancelled", key=stage)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise ResourceExhausted("execution deadline exceeded", key=stage)
        if self.max_work_bytes is not None and allocation_bytes > self.max_work_bytes:
            raise ResourceExhausted("workspace allocation exceeds declared budget", key=stage)
        if self.progress is not None:
            self.progress({"stage": stage, "allocation_bytes": allocation_bytes, **progress})
            if self.is_cancelled:
                raise ExecutionCancelled("execution cancelled", key=stage)
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise ResourceExhausted("execution deadline exceeded", key=stage)


def checkpoint(control, stage, **kwargs):
    if control is not None:
        control.checkpoint(stage, **kwargs)


class Operation:
    def __init__(self, stage, store):
        self.stage, self.store = stage, store
        self.started = time.monotonic()
        self.before = dict(store.stats) if store else {}
        self.assessment = None

    def finish(self, status="complete", **details):
        after = dict(self.store.stats) if self.store else {}
        counters = ("hits", "misses", "invalidations", "evictions", "refused")
        self.assessment = ExecutionAssessment(
            status,
            self.stage,
            time.monotonic() - self.started,
            {k: after[k] for k in ("bytes", "peak_bytes", "max_bytes") if k in after},
            {k: after.get(k, 0) - self.before.get(k, 0) for k in counters},
            details,
        )
        return self.assessment


@contextmanager
def operation(stage, *, control=None, store=None):
    run = Operation(stage, store)
    try:
        checkpoint(control, stage)
        yield run
    except MemoryError as exc:
        error = ResourceExhausted("allocation failed", key=stage)
        error.execution = run.finish(error.status, failure=str(error), location=stage)
        raise error from exc
    except Exception as exc:
        status = exc.status if isinstance(exc, ExecutionStopped) else "failed"
        from ocbf.errors import BudgetExceeded

        if isinstance(exc, BudgetExceeded):
            status = "resource-exhausted"
        assessment = run.finish(status, failure=str(exc), location=getattr(exc, "key", stage))
        # Expected typed failures retain the original exception type/compatibility.
        from ocbf.errors import ExecutionFailure, OCBFError

        if isinstance(exc, OCBFError):
            exc.execution = assessment
            raise
        error = ExecutionFailure(str(exc), key=stage)
        error.execution = assessment
        raise error from exc
    else:
        if run.assessment is None:
            run.finish()
