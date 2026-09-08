"""Effective sample size: how many *independent* votes an assertion really has.

``k`` near-duplicate sources are not ``k`` votes. With weak sources, over-counting
correlated evidence is the fastest route to confident error, and it is the dominant failure
mode when sources are numerous (Stage 1 section 4.6). This module makes the over-count
visible.

The design-effect correction, per declared source family ``c``:

```text
ESS_c  = m_c / ( 1 + (m_c - 1) * rho_c )
ESS(a) = sum over families of ESS_c
```

with ``rho_c`` the within-family residual correlation -- the agreement between cluster-mates
*beyond* what their individual accuracies already explain. Under conditional independence
two sources with +/-1 accuracies ``a_i``, ``a_j`` agree at rate ``a_i a_j``; anything above
that is shared error, which is exactly what must not be counted twice.

Reporting ``ESS(a)`` next to ``deg(a)`` is what turns "twenty sources agree" from a
reassuring number into a checkable one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef
from ocbf.reliability.moments import BinaryClaimTable, pairwise_agreement
from ocbf.sources import ClaimSet


@dataclass(slots=True)
class ESSReport:
    """Per-assertion effective sample size and the correlations behind it."""

    ess: dict[AssertionRef, float]
    deg: dict[AssertionRef, int]
    cluster_rho: dict[str, float]

    def deflation(self, ref: AssertionRef) -> float:
        """``ESS(a) / deg(a)`` in ``(0, 1]``. 1.0 means fully independent evidence."""
        d = self.deg.get(ref, 0)
        return self.ess.get(ref, 0.0) / d if d else 0.0

    def summary(self) -> dict[str, object]:
        """Mean degree, mean ESS, deflation, and the fitted within-cluster correlations."""
        if not self.ess:
            return {"assertions": 0}
        e = np.fromiter(self.ess.values(), dtype=np.float64, count=len(self.ess))
        d = np.fromiter((self.deg[r] for r in self.ess), dtype=np.float64, count=len(self.ess))
        with np.errstate(invalid="ignore", divide="ignore"):
            defl = np.where(d > 0, e / d, np.nan)
        return {
            "assertions": len(self.ess),
            "deg_mean": round(float(d.mean()), 3),
            "ess_mean": round(float(e.mean()), 3),
            "deflation_mean": round(float(np.nanmean(defl)), 4),
            "deflation_min": round(float(np.nanmin(defl)), 4),
            "clusters": len(self.cluster_rho),
            "rho_mean": round(float(np.mean(list(self.cluster_rho.values()))), 4)
            if self.cluster_rho
            else 0.0,
            "rho_max": round(float(max(self.cluster_rho.values())), 4)
            if self.cluster_rho
            else 0.0,
        }


def _cluster_correlations(
    claim_set: ClaimSet,
    clusters: Mapping[str, str],
    accuracy: Mapping[str, float],
    *,
    min_overlap: int,
    default_accuracy: float,
) -> dict[str, float]:
    """Residual within-cluster correlation, over and above what accuracy explains."""
    table = BinaryClaimTable(claim_set)
    if not table.sources:
        return {}
    agree = pairwise_agreement(table, min_overlap=min_overlap)

    # +/-1 correlation scale: accuracy a maps to 2a - 1.
    corr = np.array(
        [2.0 * accuracy.get(s, default_accuracy) - 1.0 for s in table.sources], dtype=np.float64
    )
    expected = np.outer(corr, corr)

    by_cluster: dict[str, list[float]] = {}
    for i, si in enumerate(table.sources):
        ci = clusters.get(si)
        if ci is None:
            continue
        for j in range(i + 1, len(table.sources)):
            if clusters.get(table.sources[j]) != ci:
                continue
            observed = agree[i, j]
            if not np.isfinite(observed):
                continue
            exp = expected[i, j]
            denom = 1.0 - exp
            if denom <= 1e-6:
                continue
            by_cluster.setdefault(ci, []).append(float(np.clip((observed - exp) / denom, 0.0, 1.0)))

    return {c: float(np.median(v)) for c, v in by_cluster.items() if v}


def effective_sample_sizes(
    claim_set: ClaimSet,
    clusters: Mapping[str, str],
    accuracy: Mapping[str, float] | None = None,
    *,
    refs: Sequence[AssertionRef] | None = None,
    min_overlap: int = 3,
    default_accuracy: float = 0.6,
    default_rho: float = 0.3,
) -> ESSReport:
    """Compute ESS per assertion from declared source families and estimated correlations.

    ``clusters`` maps source id to its **declared** family. Declared rather than learned is
    deliberate (design doc section 5.3): learning a dependency graph also needs overlap,
    and in a thin claim graph there is not enough of it -- so provenance the operator
    already knows beats an under-determined estimate.

    ``default_rho`` applies to clusters with too little overlap to estimate. It is
    deliberately non-zero: assuming independence within a declared family is the optimistic
    error, and the optimistic error here is the one that produces confident wrongness.
    """
    accuracy = accuracy or {}
    rho = _cluster_correlations(
        claim_set, clusters, accuracy, min_overlap=min_overlap, default_accuracy=default_accuracy
    )

    targets = list(refs) if refs is not None else list(claim_set.refs)
    ess: dict[AssertionRef, float] = {}
    deg: dict[AssertionRef, int] = {}

    for ref in targets:
        covering = claim_set.sources_covering(ref)
        deg[ref] = len(covering)
        if not covering:
            ess[ref] = 0.0
            continue
        per_cluster: dict[str, int] = {}
        for s in covering:
            per_cluster[clusters.get(s, s)] = per_cluster.get(clusters.get(s, s), 0) + 1
        total = 0.0
        for cluster, m in per_cluster.items():
            r = rho.get(cluster, default_rho if m > 1 else 0.0)
            total += m / (1.0 + (m - 1) * r)
        ess[ref] = total

    return ESSReport(ess=ess, deg=deg, cluster_rho=rho)
