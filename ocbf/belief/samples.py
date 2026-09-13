"""Concrete posterior backed by retained MCMC draws."""

from dataclasses import dataclass

from ocbf.errors import CapabilityError
from ocbf.runtime.control import checkpoint

from .posterior import JointDrawSet


@dataclass(frozen=True)
class SamplePosterior:
    draw_set: JointDrawSet

    def draw(self, scope, *, rng=None, size=None):
        return self.draw_with_context(scope, rng=rng, size=size)

    def draw_with_context(self, scope, *, rng=None, size=None, control=None):
        if size is not None and size != self.draw_set.shape[1]:
            raise CapabilityError("retained MCMC draws cannot be silently resampled or truncated")
        if not set(scope) <= self.draw_set.values.keys():
            raise CapabilityError("draw scope names absent variables")
        checkpoint(
            control,
            "posterior.project",
            allocation_bytes=2 * sum(self.draw_set.values[k].nbytes for k in scope),
        )
        return self.draw_set.project(scope)
