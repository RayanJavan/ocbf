"""Neutral posterior capabilities and immutable result values; no engine imports."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ocbf._values import canonical_json, fingerprint, freeze
from ocbf.errors import CapabilityError, ValidationError


def draw_dtype(domain):
    """Admit portable scalar arrays only when category values survive conversion exactly."""
    array = np.asarray(domain)
    if (
        array.ndim != 1
        or array.dtype.hasobject
        or canonical_json(tuple(array.tolist())) != canonical_json(tuple(domain))
    ):
        raise CapabilityError("joint draws require losslessly representable scalar categories")
    return array.dtype


@dataclass(frozen=True)
class QueryRequirements:
    scopes: tuple[tuple[str, ...], ...] = ()
    capabilities: tuple[str, ...] = ("marginal",)
    alternatives: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "scopes", tuple(sorted({tuple(s) for s in self.scopes})))
        object.__setattr__(self, "capabilities", tuple(sorted(set(self.capabilities))))
        object.__setattr__(self, "alternatives", freeze(self.alternatives))

    def satisfied_by(self, capabilities):
        available = set(capabilities)
        return set(self.capabilities) <= available and (
            not self.alternatives or any(set(option) <= available for option in self.alternatives)
        )


@dataclass(frozen=True)
class JointTable:
    scope: tuple[str, ...]
    domains: tuple[tuple, ...]
    probabilities: np.ndarray

    def __post_init__(self):
        p = np.asarray(self.probabilities, dtype=float)
        if (
            p.shape != tuple(map(len, self.domains))
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.isclose(p.sum(), 1, atol=1e-10, rtol=0)
        ):
            raise ValidationError("invalid normalized joint table")
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "domains", freeze(self.domains))
        object.__setattr__(self, "probabilities", freeze(p))


class MarginalProvider(Protocol):
    def marginal(self, key: str) -> JointTable: ...


class JointProvider(MarginalProvider, Protocol):
    def joint(self, scope: tuple[str, ...]) -> JointTable: ...


@dataclass(frozen=True)
class JointDrawSet:
    """Owned equal-weight joint draws: every array has shape (chains, iterations)."""

    values: Mapping
    method: str
    metadata: Mapping = field(default_factory=dict)
    parent_id: str | None = None
    _draw_set_id: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        values = {k: np.asarray(v) for k, v in self.values.items()}
        shapes = {v.shape for v in values.values()}
        if (
            not values
            or len(shapes) != 1
            or any(v.ndim != 2 or not v.size for v in values.values())
        ):
            raise ValidationError("draw arrays need one nonempty chain/iteration shape")
        if self.method not in ("iid", "mcmc"):
            raise ValidationError("unsupported draw weighting semantics")
        if any(v.dtype.kind in "fc" and not np.isfinite(v).all() for v in values.values()):
            raise ValidationError("nonfinite joint draw")
        object.__setattr__(self, "values", freeze(values))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    @property
    def draw_set_id(self):
        if self._draw_set_id is None:
            object.__setattr__(
                self,
                "_draw_set_id",
                self.parent_id
                or fingerprint("draw-set", (self.values, self.method, self.metadata)),
            )
        return self._draw_set_id

    @property
    def shape(self):
        return next(iter(self.values.values())).shape

    def project(self, scope):
        return JointDrawSet(
            {k: self.values[k] for k in scope}, self.method, self.metadata, self.draw_set_id
        )


class JointDrawProvider(Protocol):
    def draw(self, scope, *, rng=None, size=None) -> JointDrawSet: ...


class ConditionalReconstruction(Protocol):
    log_normalizer: float

    def draw_assignment(self, rng) -> Mapping: ...


@dataclass(frozen=True)
class InferenceResult:
    model_id: str
    plan_id: str
    run_id: str
    posterior: object = field(repr=False, compare=False)
    capabilities: tuple[str, ...] = ()
    computation: str = "unavailable"
    diagnostics: Mapping = field(default_factory=dict)
    manifest: Mapping = field(default_factory=dict)
    execution: object = field(default=None, metadata={"identity_omit_default": True})

    def __post_init__(self):
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))
        object.__setattr__(self, "manifest", freeze(self.manifest))
