"""Explicit manual trust resolution; no estimator imports or data-dependent defaults."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from ocbf._values import fingerprint, freeze
from ocbf.errors import ValidationError


@dataclass(frozen=True, order=True)
class ChannelKey:
    source_id: str
    family: str
    channel: str
    producer_version: str

    @classmethod
    def of(cls, observation):
        return cls(
            observation.source_id,
            observation.family,
            observation.channel,
            observation.producer_version,
        )


@dataclass(frozen=True)
class ChannelValues:
    """Binary alpha/f; categorical hit rate or explicit row-stochastic confusion matrix."""

    sensitivity: float = 0.7
    false_positive: float = 0.3
    hit_rate: float = 0.7
    confusion: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self):
        for value in (self.sensitivity, self.false_positive, self.hit_rate):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValidationError("channel probabilities must be finite in [0, 1]")
        matrix = tuple(tuple(float(v) for v in row) for row in self.confusion)
        if matrix and (
            any(len(row) != len(matrix) for row in matrix)
            or any(not math.isfinite(v) or v < 0 for row in matrix for v in row)
            or any(not math.isclose(sum(row), 1, abs_tol=1e-12) for row in matrix)
        ):
            raise ValidationError("confusion matrix must be square and row stochastic")
        object.__setattr__(self, "confusion", matrix)

    @classmethod
    def from_source_params(cls, params):
        return cls(params.sensitivity, 1 - params.specificity, params.rho)


@dataclass(frozen=True)
class TrustRule:
    # The selected channel validates its own immutable parameter contract.
    values: object
    origin: str
    source_id: str | None = None
    family: str | None = None
    channel: str | None = None
    producer_version: str | None = None

    def matches(self, key):
        return all(
            getattr(self, name) in (None, getattr(key, name))
            for name in ("source_id", "family", "channel", "producer_version")
        )

    @property
    def exact(self):
        return all(
            getattr(self, name) is not None
            for name in ("source_id", "family", "channel", "producer_version")
        )


@dataclass(frozen=True)
class ResolvedChannel:
    key: ChannelKey
    values: object
    origin: str


@dataclass(frozen=True)
class ParameterSet:
    channels: tuple[ResolvedChannel, ...] = ()
    priors: Mapping = field(default_factory=dict)
    assumptions: Mapping = field(default_factory=dict)

    def __post_init__(self):
        channels = tuple(sorted(self.channels, key=lambda c: c.key))
        if len({c.key for c in channels}) != len(channels):
            raise ValidationError("duplicate resolved channel key")
        if any(not c.origin for c in channels):
            raise ValidationError("parameter origin is required")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "priors", freeze(self.priors))
        object.__setattr__(self, "assumptions", freeze(self.assumptions))

    @property
    def parameter_id(self):
        return fingerprint("parameters", self)

    def for_observation(self, observation):
        key = ChannelKey.of(observation)
        for channel in self.channels:
            if channel.key == key:
                return channel.values
        raise ValidationError("missing resolved channel parameters", key=observation.observation_id)


def resolve_parameters(observations, *, rules=(), default=None, priors=None, assumptions=None):
    """Exact override > compatible family rule > explicitly supplied default; ties fail."""
    for rule in rules:
        if not rule.exact and rule.family is None:
            raise ValidationError("non-exact rules must name a family; use default explicitly")
    resolved = []
    for key in sorted({ChannelKey.of(o) for o in observations}):
        matching = [r for r in rules if r.matches(key)]
        candidates = [r for r in matching if r.exact] or [r for r in matching if not r.exact]
        if len(candidates) > 1:
            raise ValidationError("ambiguous manual trust rules", key=str(key))
        rule = candidates[0] if candidates else default
        if rule is None:
            raise ValidationError("missing manual trust rule", key=str(key))
        resolved.append(ResolvedChannel(key, rule.values, rule.origin))
    return ParameterSet(tuple(resolved), priors or {}, assumptions or {})
