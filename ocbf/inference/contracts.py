"""Execution policies and adapter interfaces independent of process query implementations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ocbf._values import freeze
from ocbf.belief.posterior import InferenceResult, QueryRequirements
from ocbf.errors import ValidationError


@dataclass(frozen=True)
class SamplingConfig:
    chains: int = 4
    warmup: int = 1000
    draws: int = 4000
    max_block_states: int = 256
    max_initialization: int = 2000
    refresh_probability: float = 0.1
    proposal_scale: float = 0.25
    blocks: tuple[tuple[str, ...], ...] = ()
    initial_states: tuple = ()

    def __post_init__(self):
        if any(
            type(v) is not int
            for v in (
                self.chains,
                self.draws,
                self.warmup,
                self.max_block_states,
                self.max_initialization,
            )
        ):
            raise ValidationError("sampling iteration budgets must be integers")
        if (
            min(self.chains, self.draws, self.max_block_states, self.max_initialization) < 1
            or self.warmup < 0
        ):
            raise ValidationError("invalid sampling iteration budgets")
        if not 0 < self.refresh_probability <= 1 or not 0 < self.proposal_scale < float("inf"):
            raise ValidationError("invalid sampling proposal configuration")
        object.__setattr__(self, "blocks", freeze(self.blocks))
        object.__setattr__(self, "initial_states", freeze(self.initial_states))


@dataclass(frozen=True)
class Proposal:
    state: Mapping
    log_forward: float
    log_reverse: float

    def __post_init__(self):
        object.__setattr__(self, "state", freeze(self.state))


class ProposalKernel(Protocol):
    name: str
    version: str

    def propose(self, state, rng) -> Proposal: ...


@dataclass(frozen=True)
class InferencePolicy:
    engine: str = "gtsam_exact"
    max_table_states: int = 1 << 22
    max_clique_states: int = 1 << 18
    max_joint_states: int = 1 << 18
    allow_approximate: bool = False
    sampling: SamplingConfig | None = None
    max_components: int = 1024
    max_continuous_dim: int = 64
    max_draw_bytes: int = 256 * 1024 * 1024
    max_seconds: float | None = None

    def __post_init__(self):
        if any(
            type(v) is not int
            for v in (
                self.max_table_states,
                self.max_clique_states,
                self.max_joint_states,
                self.max_components,
                self.max_continuous_dim,
                self.max_draw_bytes,
            )
        ):
            raise ValidationError("allocation budgets must be integers")
        if (
            min(
                self.max_table_states,
                self.max_clique_states,
                self.max_joint_states,
                self.max_components,
                self.max_continuous_dim,
                self.max_draw_bytes,
            )
            < 1
        ):
            raise ValidationError("resource budgets must be positive")
        if self.max_seconds is not None and not 0 < self.max_seconds < float("inf"):
            raise ValidationError("time budget must be finite and positive")


@dataclass(frozen=True)
class ExecutionPlan:
    engine: str
    ordering: tuple[str, ...]
    largest_input: int
    largest_clique: int
    capabilities: tuple[str, ...]
    details: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "details", freeze(self.details))


class InferenceEngine(Protocol):
    name: str
    version: str

    def assess(
        self, model, requirements: QueryRequirements, policy: InferencePolicy
    ) -> ExecutionPlan: ...
    def solve(self, model, plan, policy, rng) -> InferenceResult: ...
