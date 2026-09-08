"""Scoring rules, discrimination, and calibration for binary assertions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef
from ocbf.belief import BeliefState

_EPS = 1e-12


@dataclass(slots=True)
class BinaryReport:
    """Discrimination and calibration on one set of binary assertions."""

    n: int
    base_rate: float
    accuracy: float
    balanced_accuracy: float
    auc: float
    brier: float
    log_loss: float
    ece: float
    n_decidable: int
    accuracy_decidable: float

    def as_dict(self) -> dict[str, float | int]:
        """The report as a flat, rounded mapping suitable for tabulating."""
        return {
            "n": self.n,
            "base_rate": round(self.base_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 4),
            "log_loss": round(self.log_loss, 4),
            "ece": round(self.ece, 4),
            "n_decidable": self.n_decidable,
            "accuracy_decidable": round(self.accuracy_decidable, 4),
        }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by rank statistic, with ties handled by average ranks."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def reliability_diagram(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Per-bin confidence vs. observed frequency -- the picture behind the ECE number."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float | int]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (scores >= lo) & (scores < hi if b < n_bins - 1 else scores <= hi)
        count = int(sel.sum())
        if not count:
            continue
        out.append(
            {
                "bin_lo": round(float(lo), 3),
                "bin_hi": round(float(hi), 3),
                "count": count,
                "mean_confidence": round(float(scores[sel].mean()), 4),
                "observed_frequency": round(float(labels[sel].mean()), 4),
            }
        )
    return out


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    total = 0.0
    for row in reliability_diagram(scores, labels, n_bins):
        gap = abs(float(row["mean_confidence"]) - float(row["observed_frequency"]))
        total += (int(row["count"]) / len(scores)) * gap
    return total


def evaluate_binary(
    belief: BeliefState,
    truth: Mapping[AssertionRef, object],
    refs: Sequence[AssertionRef] | None = None,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> BinaryReport:
    """Score a belief state against ground truth on binary assertions.

    ``accuracy_decidable`` is reported separately from ``accuracy`` on purpose. The
    decidability flag (design doc section 7.2) claims that assertions above the evidence
    threshold are the ones we can actually call; if that claim is true, accuracy on the
    decidable subset should be materially higher than overall. If it is not, the
    decidability test is miscalibrated and the flag is worse than useless.
    """
    candidates = list(refs) if refs is not None else [r for r in truth if isinstance(truth[r], bool)]
    pairs = [
        (r, bool(truth[r]))
        for r in candidates
        if r in truth and isinstance(truth[r], bool) and belief.registry.get(r) is not None
    ]
    if not pairs:
        return BinaryReport(0, 0.0, *(float("nan"),) * 6, 0, float("nan"))

    kept_refs = [r for r, _ in pairs]
    labels = np.fromiter((int(v) for _, v in pairs), dtype=np.int64, count=len(pairs))
    scores = belief.binary_scores(kept_refs)
    clipped = np.clip(scores, _EPS, 1 - _EPS)
    predicted = (scores >= threshold).astype(np.int64)

    correct = predicted == labels
    pos, neg = labels == 1, labels == 0
    balanced = np.mean(
        [correct[pos].mean() if pos.any() else np.nan, correct[neg].mean() if neg.any() else np.nan]
    )

    decidable = belief.decidable_mask(kept_refs)
    acc_dec = float(correct[decidable].mean()) if decidable.any() else float("nan")

    return BinaryReport(
        n=len(pairs),
        base_rate=float(labels.mean()),
        accuracy=float(correct.mean()),
        balanced_accuracy=float(balanced),
        auc=_auc(scores, labels),
        brier=float(np.mean((scores - labels) ** 2)),
        log_loss=float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        ece=_ece(scores, labels, n_bins),
        n_decidable=int(decidable.sum()),
        accuracy_decidable=acc_dec,
    )


def compare(
    beliefs: Mapping[str, BeliefState],
    truth: Mapping[AssertionRef, object],
    refs: Sequence[AssertionRef] | None = None,
    **kwargs: object,
) -> dict[str, dict[str, float | int]]:
    """Score several belief states on identical assertions.

    The evaluation set is intersected across methods so the comparison is like-for-like: a
    method that simply declines to answer on hard assertions must not thereby appear to win.
    """
    if refs is None:
        common: set[AssertionRef] | None = None
        for belief in beliefs.values():
            touched = {r for r in truth if belief.registry.get(r) is not None}
            common = touched if common is None else (common & touched)
        refs = sorted(common or set())
    return {
        name: evaluate_binary(belief, truth, refs, **kwargs).as_dict()  # type: ignore[arg-type]
        for name, belief in beliefs.items()
    }
