"""The source protocol and its declarations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ocbf.assertions import AssertionRef, Family
from ocbf.sources.claims import Claim


class CoverageSemantics(str, Enum):
    """What a source's silence means.

    Design doc section 5.2 shows these are one family, not three unrelated modes:

    ```text
    pi_s(a) = sigmoid( w_s . f(a) + gamma_s * truth(a) )
    ```

    with ``gamma_s = 0`` recovering ``OPPORTUNISTIC`` and ``gamma_s -> inf`` recovering
    ``COMPLETE_OVER_SCOPE``. So ``SELECTIVE`` with a fitted ``gamma_s`` is the general
    case, and a *mis-declared* mode is detectable by fitting ``gamma_s`` and comparing.
    """

    OPPORTUNISTIC = "opportunistic"
    """Silence is uninformative: the propensity does not depend on the truth, so it
    factorises out of the likelihood entirely."""

    COMPLETE_OVER_SCOPE = "complete_over_scope"
    """Silence inside the declared scope is a full negative claim, contributing
    ``log(1 - alpha_s)`` or ``log(beta_s)``. This is where a detector's false-negative
    rate does its work -- treating its silence as "no information" throws away the
    strongest signal it carries."""

    SELECTIVE = "selective"
    """Silence is informative through an explicit propensity model. The general case."""

    @property
    def silence_is_evidence(self) -> bool:
        return self is not CoverageSemantics.OPPORTUNISTIC


class ChannelFamily(str, Enum):
    """The likelihood form ``p(y | truth, theta_s)`` for this source's claims."""

    BINARY = "binary"
    """Two-sided quality ``(alpha_s, beta_s)`` per LTM. Two-sided is non-negotiable:
    under ``COMPLETE_OVER_SCOPE``, ``beta_s`` is what silence means."""

    CATEGORICAL = "categorical"
    """One-parameter spread model ``rho_s`` with the off-diagonal confusion profile shared
    across the source cluster. A per-source ``K x K`` confusion matrix is exactly the
    parameterisation the sparse regime forbids (design doc section 5.1)."""

    CONTINUOUS = "continuous"
    """``y = v + bias_s + eps``, ``eps ~ StudentT(nu_s, sigma_s)``. Student-t rather than
    Gaussian for robustness to the gross outliers unreliable sources produce; as a scale
    mixture of normals it stays conditionally Gaussian and so composes with the single
    moment-matching mechanism of design doc section 4.3."""

    DISTRIBUTIONAL = "distributional"
    """The source emits a distribution, consumed as ``lambda_s * log q_s(v)``. The
    ``lambda_s`` calibration temperature is where "calibrated marginals" is actually
    enforced, and it is the cheapest high-value parameter in the model."""

    @classmethod
    def for_family(cls, family: Family) -> ChannelFamily:
        """The natural channel for an assertion family, absent an explicit declaration."""
        if family is Family.EVENT_TYPE:
            return cls.CATEGORICAL
        if family is Family.EVENT_TIME:
            return cls.CONTINUOUS
        if family in (Family.EVENT_ATTR, Family.OBJECT_ATTR):
            return cls.CONTINUOUS
        return cls.BINARY


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Everything the reliability model needs to know about a source *before* seeing data.

    ``features`` feeds ``beta_feat`` in the hierarchical GLM -- the SLiMFast idea of
    regressing accuracy on observable source properties (modality, vendor, model version,
    placement, sampling rate, latency). It turns ``|S|`` free parameters into ``d << |S|``
    and generalises to sources never seen before, which at 1e5 sources is the difference
    between a fittable model and an unfittable one.
    """

    source_id: str
    coverage: CoverageSemantics
    channel: ChannelFamily
    cluster_id: str = "default"
    features: tuple[float, ...] = ()
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.features) != len(self.feature_names):
            raise ValueError(
                f"source {self.source_id!r}: {len(self.features)} features but "
                f"{len(self.feature_names)} names"
            )

    def feature_map(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.features, strict=True))


@runtime_checkable
class Source(Protocol):
    """An evidence provider.

    ``scope`` and ``claims`` are separate on purpose. The scope is what the source *could*
    have spoken about; the claims are what it *did*. Their difference is the silence, and
    under a non-opportunistic coverage semantics that silence is evidence.
    """

    @property
    def profile(self) -> SourceProfile: ...

    def scope(self) -> Iterable[AssertionRef]:
        """Assertions this source is capable of addressing."""
        ...

    def claims(self) -> Iterable[Claim]:
        """Assertions this source actually spoke about."""
        ...


@dataclass(slots=True)
class StaticSource:
    """A source backed by a materialised list of claims.

    The common adapter shape: an upstream pipeline has already run, and its outputs are
    handed over as claims. ``scope`` defaults to the claimed refs, which forces
    ``OPPORTUNISTIC`` semantics to be the only coherent reading -- if the scope equals the
    claims there is no silence to interpret. A non-opportunistic source **must** declare a
    scope wider than its claims, and the constructor enforces that rather than letting a
    silently-empty silence set produce a confidently wrong answer.
    """

    profile: SourceProfile
    _claims: tuple[Claim, ...]
    _scope: frozenset[AssertionRef]

    def __init__(
        self,
        profile: SourceProfile,
        claims: Iterable[Claim],
        scope: Iterable[AssertionRef] | None = None,
    ) -> None:
        self.profile = profile
        self._claims = tuple(claims)

        wrong_source = {c.source_id for c in self._claims} - {profile.source_id}
        if wrong_source:
            raise ValueError(
                f"source {profile.source_id!r} was given claims from {sorted(wrong_source)}"
            )

        claimed = frozenset(c.ref for c in self._claims)
        self._scope = claimed if scope is None else frozenset(scope)

        missing = claimed - self._scope
        if missing:
            raise ValueError(
                f"source {profile.source_id!r} claims {len(missing)} refs outside its "
                f"declared scope, e.g. {next(iter(missing))}"
            )
        if profile.coverage.silence_is_evidence and self._scope == claimed:
            raise ValueError(
                f"source {profile.source_id!r} declares {profile.coverage.value} coverage "
                "but its scope equals its claims, so it has no silence to interpret; "
                "declare a wider scope or use OPPORTUNISTIC"
            )

    def scope(self) -> frozenset[AssertionRef]:
        return self._scope

    def claims(self) -> tuple[Claim, ...]:
        return self._claims

    def silent_refs(self) -> frozenset[AssertionRef]:
        """Scope minus claims -- the refs whose silence carries evidence."""
        return self._scope - frozenset(c.ref for c in self._claims)

    def __len__(self) -> int:
        return len(self._claims)

    def __repr__(self) -> str:
        return (
            f"StaticSource({self.profile.source_id!r}, claims={len(self._claims)}, "
            f"scope={len(self._scope)}, coverage={self.profile.coverage.value})"
        )
