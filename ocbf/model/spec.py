"""Canonical finite and bounded hybrid targets, independent of solver representations."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ocbf._values import canonical_json, fingerprint, freeze, utc
from ocbf.errors import ValidationError
from ocbf.evidence import InterpretedEvidence
from ocbf.reliability.config import ParameterSet
from ocbf.universe.context import SemanticContext


@dataclass(frozen=True)
class VariableSpec:
    key: str
    domain: tuple
    unit: str = "category"
    measure: str = "counting"

    def __post_init__(self):
        if not self.key:
            raise ValidationError("variable key is required")
        domain = freeze(self.domain)
        if not domain or len({canonical_json(v) for v in domain}) != len(domain):
            raise ValidationError("a finite unique domain is required", key=self.key)
        if self.measure != "counting":
            raise ValidationError("finite variables require counting measure", key=self.key)
        object.__setattr__(self, "domain", domain)


@dataclass(frozen=True)
class ContinuousVariableSpec:
    """Real coordinate with an explicit proper prior and semantic activation.

    When inactive the coordinate is auxiliary with this same normalized prior.
    Decoding omits its value; integrating the inactive coordinate contributes one.
    """

    key: str
    prior_mean: float
    prior_sd: float
    unit: str
    origin: str
    active_when: tuple[tuple[str, object], ...] = ()
    measure: str = "lebesgue"

    def __post_init__(self):
        if not self.key or not self.unit or not self.origin or self.measure != "lebesgue":
            raise ValidationError("continuous key, unit, origin and Lebesgue measure required")
        if (
            not math.isfinite(self.prior_mean)
            or not math.isfinite(self.prior_sd)
            or self.prior_sd <= 0
        ):
            raise ValidationError("continuous prior must be a proper Gaussian", key=self.key)
        object.__setattr__(self, "active_when", freeze(self.active_when))


@dataclass(frozen=True)
class FactorSpec:
    key: str
    scope: tuple[str, ...]
    family: str = "table"
    parameters: Mapping = field(default_factory=dict)
    log_values: np.ndarray | None = None
    role: str = "descriptive"
    evidence_ids: tuple[str, ...] = ()
    version: str = "1"
    log_offset: float = 0.0

    def __post_init__(self):
        if not self.key:
            raise ValidationError("factor key is required")
        if len(set(self.scope)) != len(self.scope):
            raise ValidationError("duplicate factor variable", key=self.key)
        if self.role not in ("support", "descriptive", "observation"):
            raise ValidationError("normative references cannot be model factors", key=self.key)
        if not math.isfinite(self.log_offset):
            raise ValidationError("factor offset must be finite", key=self.key)
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "parameters", freeze(self.parameters))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        if self.log_values is not None:
            a = np.asarray(self.log_values, dtype=np.float64)
            if np.isnan(a).any() or np.isposinf(a).any():
                raise ValidationError("factor contains NaN or positive infinity", key=self.key)
            object.__setattr__(self, "log_values", freeze(a))


@dataclass(frozen=True)
class LifecycleBinding:
    """Explicit same-object endpoint binding; never all starts/completions of a Job."""

    object_id: str
    before_event: str
    after_event: str
    qualifier: str
    before: datetime
    after: datetime
    role: str = "descriptive"
    penalty_nats: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "before", utc(self.before))
        object.__setattr__(self, "after", utc(self.after))
        if self.before_event == self.after_event or self.role not in (
            "support",
            "descriptive",
            "normative",
        ):
            raise ValidationError("invalid lifecycle binding")
        if not math.isfinite(self.penalty_nats) or self.penalty_nats <= 0:
            raise ValidationError("lifecycle penalty must be positive and finite")


@dataclass(frozen=True)
class ModelSpec:
    context: SemanticContext
    evidence: InterpretedEvidence
    parameters: ParameterSet
    variables: tuple[VariableSpec | ContinuousVariableSpec, ...] = ()
    factors: tuple[FactorSpec, ...] = ()
    scope: Mapping = field(default_factory=dict)
    decoding: Mapping = field(default_factory=dict)
    ground_structure: bool = True
    lifecycle_bindings: tuple[LifecycleBinding, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "scope", freeze(self.scope))
        object.__setattr__(self, "decoding", freeze(self.decoding))
        object.__setattr__(self, "lifecycle_bindings", tuple(self.lifecycle_bindings))


@dataclass(frozen=True)
class CompiledModel:
    variables: tuple[VariableSpec | ContinuousVariableSpec, ...]
    factors: tuple[FactorSpec, ...]
    manifest: Mapping
    decoding: Mapping
    _kernels: Mapping = field(repr=False, compare=False)
    _domains: Mapping = field(init=False, repr=False, compare=False)
    _model_id: str = field(init=False, repr=False, compare=False)
    _dependency_index: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        from types import MappingProxyType

        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "manifest", freeze(self.manifest))
        object.__setattr__(self, "decoding", freeze(self.decoding))
        object.__setattr__(self, "_kernels", MappingProxyType(dict(self._kernels)))
        object.__setattr__(
            self,
            "_domains",
            freeze({v.key: v.domain for v in self.variables if hasattr(v, "domain")}),
        )
        object.__setattr__(
            self,
            "_model_id",
            fingerprint("model", (self.variables, self.factors, self.manifest, self.decoding)),
        )

    @property
    def model_id(self):
        return self._model_id

    @property
    def domains(self):
        return self._domains

    def log_value(self, factor, indices):
        return self._kernels[factor.family].log_value(factor, indices, self.domains)

    @property
    def continuous(self):
        return {v.key: v for v in self.variables if isinstance(v, ContinuousVariableSpec)}

    def factor_log_density(self, factor, state):
        kernel = self._kernels[factor.family]
        if hasattr(kernel, "log_density"):
            return float(kernel.log_density(factor, state))
        return self.log_value(factor, tuple(self.domains[k].index(state[k]) for k in factor.scope))

    def log_density(self, state):
        values = [self.factor_log_density(f, state) for f in self.factors]
        if any(math.isnan(v) or v == math.inf for v in values):
            from ocbf.errors import NumericalFailure

            raise NumericalFailure("nonfinite factor density")
        return sum(values)
