"""Observation/factor ports. This module imports no concrete kernels or channels."""

from dataclasses import dataclass
from typing import Protocol

from ocbf._values import freeze

from .spec import FactorSpec


@dataclass(frozen=True)
class ChannelContribution:
    factors: tuple[FactorSpec, ...] = ()
    variables: tuple = ()
    version: str = "1"

    def __post_init__(self):
        object.__setattr__(self, "factors", freeze(self.factors))
        object.__setattr__(self, "variables", freeze(self.variables))


class FactorKernel(Protocol):
    name: str
    version: str

    def log_value(self, factor, indices, domains) -> float: ...


class StateFactorKernel(Protocol):
    name: str
    version: str
    operations: tuple[str, ...]

    def log_density(self, factor, state) -> float: ...


class BatchFiniteFactorKernel(FactorKernel, Protocol):
    """Broadcasted state arrays; implementations must avoid iteration over configurations."""

    def log_values(self, factor, state, domains): ...


class GaussianFactorKernel(StateFactorKernel, Protocol):
    """Canonical (Q,h,c) in delta coordinates around state[key] for integrated keys."""

    def gaussian_terms(self, factor, state, keys): ...


class ObservationChannel(Protocol):
    name: str
    version: str

    def factors(self, observation, parameters, domains) -> tuple[FactorSpec, ...]: ...


class ContributingChannel(Protocol):
    name: str
    version: str

    def contribute(self, observation, parameters, variables) -> ChannelContribution: ...
