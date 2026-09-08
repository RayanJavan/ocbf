"""Claims and the indexed claim set.

[`ClaimSet`][ocbf.sources.claims.ClaimSet] is the bipartite claim-incidence structure ``C ⊆ S x A`` of Stage 1
section 1.2. Nearly every diagnostic reads it: the source overlap graph (section 7.1) is
its projection onto sources, per-assertion decidability (7.2) needs ``deg(a)``, and
effective sample size (7.3) needs the per-family breakdown of who covers what.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef


@dataclass(frozen=True, slots=True)
class Claim:
    """One source speaking about one assertion.

    ``value`` is interpreted by the source's channel family:

    * ``BINARY``        -- ``bool``
    * ``CATEGORICAL``   -- ``str``, a level of the assertion's domain
    * ``CONTINUOUS``    -- ``float``
    * ``DISTRIBUTIONAL``-- ``value`` may be the argmax; ``soft`` carries the distribution

    ``soft`` lets any channel attach a distribution rather than a point. It is the honest
    representation for a learned detector, and it is what the ``lambda_s`` calibration
    temperature acts on.

    ``observed_features`` are per-claim covariates for the ``SELECTIVE`` propensity model
    (``w_s . f(a)`` in design doc section 5.2) -- e.g. signal strength, distance to sensor.
    """

    source_id: str
    ref: AssertionRef
    value: bool | str | float | None = None
    soft: Mapping[str, float] | None = None
    observed_features: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.value is None and self.soft is None:
            raise ValueError(f"claim by {self.source_id!r} on {self.ref} carries no value")
        if self.soft is not None:
            total = sum(self.soft.values())
            if not (0.0 < total < float("inf")):
                raise ValueError(f"claim by {self.source_id!r} on {self.ref} has unnormalisable soft mass")
            if any(p < 0.0 for p in self.soft.values()):
                raise ValueError(f"claim by {self.source_id!r} on {self.ref} has negative soft mass")

    def normalised_soft(self) -> dict[str, float] | None:
        if self.soft is None:
            return None
        total = sum(self.soft.values())
        return {k: v / total for k, v in self.soft.items()}


class ClaimSet:
    """An indexed, immutable collection of claims.

    Built once from all sources, then queried. The indices exist because the sparse regime
    makes ``deg(a)`` and ``deg(s)`` first-class quantities: they appear in the decidability
    test, the identifiability diagnostic, and the pooling strength of the reliability GLM.
    """

    __slots__ = ("_claims", "_by_ref", "_by_source", "_sources", "_refs")

    def __init__(self, claims: Iterable[Claim]) -> None:
        self._claims: tuple[Claim, ...] = tuple(claims)
        self._by_ref: dict[AssertionRef, list[int]] = defaultdict(list)
        self._by_source: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self._claims):
            self._by_ref[c.ref].append(i)
            self._by_source[c.source_id].append(i)
        self._sources: tuple[str, ...] = tuple(sorted(self._by_source))
        self._refs: tuple[AssertionRef, ...] = tuple(sorted(self._by_ref))

    @classmethod
    def from_sources(cls, sources: Iterable) -> ClaimSet:
        """Collect claims from anything satisfying the [`Source`][ocbf.sources.Source] protocol."""
        return cls(c for s in sources for c in s.claims())

    # -- access ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._claims)

    def __iter__(self):
        return iter(self._claims)

    def __getitem__(self, i: int) -> Claim:
        return self._claims[i]

    @property
    def claims(self) -> tuple[Claim, ...]:
        return self._claims

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self._sources

    @property
    def refs(self) -> tuple[AssertionRef, ...]:
        """Every assertion touched by at least one claim."""
        return self._refs

    def for_ref(self, ref: AssertionRef) -> list[Claim]:
        """Every claim about one assertion."""
        return [self._claims[i] for i in self._by_ref.get(ref, ())]

    def for_source(self, source_id: str) -> list[Claim]:
        """Every claim made by one source."""
        return [self._claims[i] for i in self._by_source.get(source_id, ())]

    def sources_covering(self, ref: AssertionRef) -> frozenset[str]:
        """Distinct sources that spoke about this assertion."""
        return frozenset(self._claims[i].source_id for i in self._by_ref.get(ref, ()))

    def refs_of_source(self, source_id: str) -> frozenset[AssertionRef]:
        """Distinct assertions this source spoke about."""
        return frozenset(self._claims[i].ref for i in self._by_source.get(source_id, ()))

    # -- degrees: the quantities the sparse regime turns on --------------------------

    def deg_assertion(self, ref: AssertionRef) -> int:
        """``deg(a)`` -- how many *distinct sources* cover this assertion.

        Distinct sources, not claims: two claims from one source are not two votes.
        """
        return len(self.sources_covering(ref))

    def deg_source(self, source_id: str) -> int:
        """``deg(s)`` -- how many distinct assertions this source touches.

        Small ``deg(s)`` is exactly why per-source MLE is hopeless and why the reliability
        GLM pools (design doc section 5.3).
        """
        return len(self.refs_of_source(source_id))

    def assertion_degrees(self) -> dict[AssertionRef, int]:
        return {ref: self.deg_assertion(ref) for ref in self._refs}

    def source_degrees(self) -> dict[str, int]:
        return {sid: self.deg_source(sid) for sid in self._sources}

    def density(self) -> float:
        """``|C| / (|S| x |A|)`` -- the sparsity figure of Stage 1 section 1.2."""
        denom = len(self._sources) * len(self._refs)
        return len(self._claims) / denom if denom else 0.0

    def summary(self) -> dict[str, float | int]:
        """Claim counts and degree statistics.

        `density`, `deg_a_median` and `deg_s_median` are the numbers that characterise
        the regime; read them before interpreting any result.
        """
        a_deg = np.fromiter(
            (self.deg_assertion(r) for r in self._refs), dtype=np.int64, count=len(self._refs)
        )
        s_deg = np.fromiter(
            (self.deg_source(s) for s in self._sources), dtype=np.int64, count=len(self._sources)
        )
        return {
            "claims": len(self._claims),
            "sources": len(self._sources),
            "assertions_touched": len(self._refs),
            "density": round(self.density(), 8),
            "deg_a_mean": round(float(a_deg.mean()), 3) if a_deg.size else 0.0,
            "deg_a_median": int(np.median(a_deg)) if a_deg.size else 0,
            "deg_a_max": int(a_deg.max()) if a_deg.size else 0,
            "deg_s_mean": round(float(s_deg.mean()), 3) if s_deg.size else 0.0,
            "deg_s_median": int(np.median(s_deg)) if s_deg.size else 0,
            "deg_s_max": int(s_deg.max()) if s_deg.size else 0,
        }

    def __repr__(self) -> str:
        return (
            f"ClaimSet(claims={len(self._claims)}, sources={len(self._sources)}, "
            f"assertions={len(self._refs)})"
        )
