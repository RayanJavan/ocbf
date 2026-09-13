"""Parameter values for static source diagnostics and numerical utilities.

Binary sensitivity/specificity, categorical hit rates and continuous error parameters are
explicit mathematical assumptions. ``ChannelValues.from_source_params`` converts binary
and categorical values for the canonical evidence workflow; it does not fit or calibrate
those values from physical evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # the estimators depend on these types, so the arrow points this way
    from ocbf.reliability.moments import TripletEstimate


@dataclass(frozen=True, slots=True)
class SourceParams:
    """The channel parameters of one source.

    ``sensitivity`` and ``specificity`` are kept separate rather than collapsed into an
    accuracy. Two-sided quality is what lets a detector's *silence* carry its false-negative
    rate, and under ``COMPLETE_OVER_SCOPE`` coverage that silence is most of the evidence.
    """

    sensitivity: float = 0.7
    specificity: float = 0.7
    rho: float = 0.7
    """Categorical hit rate: ``P(reported type = true type)``."""

    temperature: float = 1.0
    """Explicit multiplier for a supplied distributional log likelihood. Choosing a temperature does not establish empirical calibration."""

    noise_df: float = 3.0
    """Student-t degrees of freedom for synthetic or caller-supplied continuous noise assumptions."""

    def __post_init__(self) -> None:
        for name in ("sensitivity", "specificity", "rho"):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {v}")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.noise_df <= 0.0:
            raise ValueError("noise_df must be positive")

    def clipped(self, floor: float = 0.02, ceiling: float = 0.98) -> SourceParams:
        """Bound the parameters away from certainty.

        A source estimated from three claims can land at 0.999 and would then override every
        genuinely-informed source in the pool. Clipping does not fix that -- pooling does --
        but it keeps a thin estimate from becoming an oracle.
        """
        c = lambda v: float(np.clip(v, floor, ceiling))  # noqa: E731
        return replace(
            self, sensitivity=c(self.sensitivity), specificity=c(self.specificity), rho=c(self.rho)
        )

    @property
    def binary_loglik(self) -> np.ndarray:
        """``[[log p(y=0|t=0), log p(y=1|t=0)], [log p(y=0|t=1), log p(y=1|t=1)]]``."""
        a, b = self.sensitivity, self.specificity
        return np.log(np.array([[b, 1.0 - b], [1.0 - a, a]]))


class ReliabilityTable:
    """Source id to [`SourceParams`][ocbf.reliability.params.SourceParams], with a pooled default.

    The default is not a fallback detail -- it *is* the pooling for every source too thin to
    estimate, which in the sparse regime is most of them. Sources flagged ``PRIOR_ONLY`` by
    the overlap diagnostic get exactly this and nothing else.
    """

    __slots__ = ("_params", "default")

    def __init__(
        self, params: Mapping[str, SourceParams] | None = None, default: SourceParams | None = None
    ) -> None:
        self._params = dict(params or {})
        self.default = default or SourceParams()

    def __getitem__(self, source_id: str) -> SourceParams:
        return self._params.get(source_id, self.default)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._params

    def __len__(self) -> int:
        return len(self._params)

    def set(self, source_id: str, params: SourceParams) -> None:
        self._params[source_id] = params

    def items(self):
        return self._params.items()

    @classmethod
    def from_triplet(
        cls, estimate: "TripletEstimate", *, floor: float = 0.02, ceiling: float = 0.98
    ) -> ReliabilityTable:
        """Seed from the label-free moment estimator.

        Sensitivity and specificity are both set to the triplet accuracy: the triplet method
        estimates a single symmetric correlation and cannot separate the two sides. That is a
        limitation of the initialiser, not of the model -- the EM parameter block splits them
        on the first pass.
        """
        params = {
            s: SourceParams(a, a, a).clipped(floor, ceiling)
            for s, a in estimate.accuracy.items()
        }
        d = estimate.default_accuracy
        return cls(params, SourceParams(d, d, d).clipped(floor, ceiling))

    def sensitivities(self) -> dict[str, float]:
        """Per-source sensitivity, for sources with a fitted entry."""
        return {s: p.sensitivity for s, p in self._params.items()}

    def specificities(self) -> dict[str, float]:
        """Per-source specificity, for sources with a fitted entry."""
        return {s: p.specificity for s, p in self._params.items()}

    def summary(self) -> dict[str, float | int]:
        """Distribution of the fitted sensitivities and specificities."""
        if not self._params:
            return {"sources": 0, "default_sensitivity": self.default.sensitivity}
        sens = np.array([p.sensitivity for p in self._params.values()])
        spec = np.array([p.specificity for p in self._params.values()])
        return {
            "sources": len(self._params),
            "sensitivity_mean": round(float(sens.mean()), 4),
            "specificity_mean": round(float(spec.mean()), 4),
            "sensitivity_min": round(float(sens.min()), 4),
            "sensitivity_max": round(float(sens.max()), 4),
        }

    def __repr__(self) -> str:
        return f"ReliabilityTable({len(self._params)} sources, default={self.default})"


@dataclass(frozen=True, slots=True)
class ContinuousChannel:
    """One source's ``y = v + bias + eps`` channel for one continuous template.

    Bias and scale are in the template's **observed** units -- hours for a timestamp, euros
    for a price -- which is why they cannot be shared across templates the way an accuracy
    can. A source's sensitivity means the same thing wherever it speaks; its clock offset in
    hours means nothing about its price errors.
    """

    bias: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not (self.scale > 0.0 and np.isfinite(self.scale)):
            raise ValueError(f"channel scale must be positive and finite, got {self.scale}")


class ContinuousChannelTable:
    """Continuous error parameters indexed by template and source. This allows one source to have different bias/noise assumptions across assertion templates."""

    __slots__ = ("_channels", "_defaults", "_fallback")

    def __init__(
        self,
        channels: Mapping[tuple[str, str], ContinuousChannel] | None = None,
        defaults: Mapping[str, ContinuousChannel] | None = None,
        fallback: ContinuousChannel | None = None,
    ) -> None:
        self._channels = dict(channels or {})
        self._defaults = dict(defaults or {})
        self._fallback = fallback or ContinuousChannel()

    def __getitem__(self, key: tuple[str, str]) -> ContinuousChannel:
        """Channel for ``(template, source)``, falling back to the template's pool."""
        template, _source = key
        return self._channels.get(key, self._defaults.get(template, self._fallback))

    def __len__(self) -> int:
        return len(self._channels)

    def __contains__(self, key: object) -> bool:
        return key in self._channels

    def set(self, template: str, source_id: str, channel: ContinuousChannel) -> None:
        self._channels[(template, source_id)] = channel

    def set_default(self, template: str, channel: ContinuousChannel) -> None:
        self._defaults[template] = channel

    def templates(self) -> tuple[str, ...]:
        return tuple(sorted(self._defaults))

    def summary(self) -> dict[str, object]:
        """Per-template pooled scale, and how many sources were individually estimated."""
        return {
            "fitted_channels": len(self._channels),
            "template_scales": {
                t: round(c.scale, 4) for t, c in sorted(self._defaults.items())
            },
        }

    def __repr__(self) -> str:
        return (
            f"ContinuousChannelTable({len(self._channels)} fitted, "
            f"{len(self._defaults)} templates)"
        )
