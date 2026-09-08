"""Per-assertion decidability from the exact error exponent.

Under Dawid-Skene the error exponent for aggregating ``m`` sources is ``m * I(pi)``, where
``I`` is the average Chernoff information of the pool (arXiv:1605.07696). Reaching
misclassification ``eps`` therefore requires

```text
sum over sources covering a of C_s   >   log(1 / eps)
```

In our regime ``C_s`` is small (unreliable sources) and the covering set is small (sparse
claims), so **for a large fraction of assertions no aggregation rule whatsoever can decide
them**. That is not a defect of any particular estimator; it is an information-theoretic
statement about the evidence.

The practical consequence, and the reason this module exists: an assertion below threshold
gets an explicit ``UNDETERMINED`` verdict rather than a confidently-wrong 0.51. A posterior
that does not distinguish "balanced evidence" from "no evidence" is hiding the difference
that matters most when sources are weak.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef
from ocbf.sources import ClaimSet

_T_GRID = np.linspace(0.0, 1.0, 201)


def chernoff_binary(sensitivity: float, specificity: float) -> float:
    """Chernoff information in nats for a binary channel with two-sided quality.

    Under truth=1 the source emits 1 with probability ``sensitivity``; under truth=0 it
    emits 0 with probability ``specificity``. The Chernoff information is

    ```text
    C = -min_t  log sum_y  p0(y)^t  p1(y)^(1-t)
    ```

    evaluated on a grid. A grid rather than an optimiser because this is called once per
    source and the objective is smooth, unimodal in ``t``, and cheap -- a closed-form
    solver would add a dependency and a failure mode for no accuracy gain.

    A channel at chance (0.5, 0.5) returns exactly 0: it carries no information, and any
    method that treats such a source as a vote is fooling itself.
    """
    a = float(np.clip(sensitivity, 1e-6, 1 - 1e-6))
    b = float(np.clip(specificity, 1e-6, 1 - 1e-6))
    p1 = np.array([1.0 - a, a])
    p0 = np.array([b, 1.0 - b])
    with np.errstate(divide="ignore"):
        logs = np.log(np.sum(p0[None, :] ** _T_GRID[:, None] * p1[None, :] ** (1 - _T_GRID[:, None]), axis=1))
    return float(-logs.min())


@dataclass(slots=True)
class DecidabilityReport:
    """Evidence in nats per assertion, and the resulting verdicts."""

    evidence: dict[AssertionRef, float]
    target_nats: float
    eps: float

    def is_decidable(self, ref: AssertionRef) -> bool:
        """Whether this assertion's accumulated evidence clears the target."""
        return self.evidence.get(ref, 0.0) >= self.target_nats

    @property
    def decidable_refs(self) -> list[AssertionRef]:
        return [r for r, e in self.evidence.items() if e >= self.target_nats]

    def summary(self) -> dict[str, object]:
        """Decidable fraction and the distribution of per-assertion evidence."""
        vals = np.fromiter(self.evidence.values(), dtype=np.float64, count=len(self.evidence))
        n_dec = int((vals >= self.target_nats).sum()) if vals.size else 0
        return {
            "assertions": len(self.evidence),
            "target_eps": self.eps,
            "target_nats": round(self.target_nats, 4),
            "decidable": n_dec,
            "undecidable": len(self.evidence) - n_dec,
            "decidable_fraction": round(n_dec / len(self.evidence), 4) if self.evidence else 0.0,
            "evidence_mean": round(float(vals.mean()), 4) if vals.size else 0.0,
            "evidence_median": round(float(np.median(vals)), 4) if vals.size else 0.0,
            "evidence_max": round(float(vals.max()), 4) if vals.size else 0.0,
        }


def decidability(
    claim_set: ClaimSet,
    sensitivity: Mapping[str, float],
    specificity: Mapping[str, float] | None = None,
    *,
    eps: float = 0.1,
    default_accuracy: float = 0.6,
    refs: Sequence[AssertionRef] | None = None,
    ess: Mapping[AssertionRef, float] | None = None,
) -> DecidabilityReport:
    """Total Chernoff information per assertion, against the ``log(1/eps)`` threshold.

    When ``ess`` is supplied, each assertion's evidence is scaled by
    ``ESS(a) / deg(a)`` -- the design-effect correction. This is the composition that makes
    the diagnostics honest: twenty correlated sources supply twenty Chernoff terms but only
    a few independent votes, and without the correction a copied cluster would certify its
    own assertions as decidable.
    """
    specificity = specificity if specificity is not None else sensitivity
    target = float(np.log(1.0 / eps))
    cache: dict[str, float] = {}

    def info(source_id: str) -> float:
        if source_id not in cache:
            cache[source_id] = chernoff_binary(
                sensitivity.get(source_id, default_accuracy),
                specificity.get(source_id, default_accuracy),
            )
        return cache[source_id]

    targets = list(refs) if refs is not None else list(claim_set.refs)
    evidence: dict[AssertionRef, float] = {}
    for ref in targets:
        covering = claim_set.sources_covering(ref)
        total = sum(info(s) for s in covering)
        if ess is not None and covering:
            deflation = ess.get(ref, float(len(covering))) / len(covering)
            total *= max(0.0, min(1.0, deflation))
        evidence[ref] = total

    return DecidabilityReport(evidence=evidence, target_nats=target, eps=eps)
