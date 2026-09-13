"""Independent moment estimators for static source reliability.

For conditionally independent binary reports in +/-1 encoding, pairwise agreement satisfies ``E[x_i*x_j] = beta_i*beta_j``. Triplets estimate oriented source accuracies under an explicit better-than-chance assumption. Missing overlap falls back to the supplied assumption, not measured evidence. Pairwise continuous comparisons estimate relative bias and noise under their own assumptions. Canonical inference does not invoke these estimators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ocbf.assertions import AssertionRef
from ocbf.reliability.params import ContinuousChannel, ContinuousChannelTable
from ocbf.sources import ClaimSet


class BinaryClaimTable:
    """A sparse ``sources x assertions`` table of +/-1 votes.

    Only binary-valued claims enter. The table is the shared substrate for the moment
    estimators here and for the overlap-graph diagnostic, so it is built once.
    """

    __slots__ = ("sources", "refs", "_source_index", "_ref_index", "votes", "mask")

    def __init__(self, claim_set: ClaimSet, refs: Sequence[AssertionRef] | None = None) -> None:
        usable = [c for c in claim_set if isinstance(c.value, bool)]
        allowed = None if refs is None else set(refs)
        if allowed is not None:
            usable = [c for c in usable if c.ref in allowed]

        self.sources: tuple[str, ...] = tuple(sorted({c.source_id for c in usable}))
        self.refs: tuple[AssertionRef, ...] = tuple(sorted({c.ref for c in usable}))
        self._source_index = {s: i for i, s in enumerate(self.sources)}
        self._ref_index = {r: i for i, r in enumerate(self.refs)}

        n_s, n_a = len(self.sources), len(self.refs)
        self.votes = np.zeros((n_s, n_a), dtype=np.int8)
        self.mask = np.zeros((n_s, n_a), dtype=bool)
        for c in usable:
            i = self._source_index[c.source_id]
            j = self._ref_index[c.ref]
            self.votes[i, j] = 1 if c.value else -1
            self.mask[i, j] = True

    @property
    def shape(self) -> tuple[int, int]:
        return self.votes.shape

    def source_index(self, source_id: str) -> int:
        return self._source_index[source_id]

    def ref_index(self, ref: AssertionRef) -> int:
        return self._ref_index[ref]

    def overlap_counts(self) -> np.ndarray:
        """Return the pairwise source overlap counts as an integer matrix. Its nonzero support defines the static-source overlap graph."""
        m = self.mask.astype(np.int32)
        return m @ m.T

    def __repr__(self) -> str:
        return f"BinaryClaimTable(sources={len(self.sources)}, assertions={len(self.refs)})"


def pairwise_agreement(table: BinaryClaimTable, min_overlap: int = 3) -> np.ndarray:
    """``M[i,j] = E[y_i y_j]`` over co-claimed assertions; ``NaN`` where overlap is too thin.

    ``NaN`` rather than 0 is deliberate: 0 means "these sources are uncorrelated", whereas
    absent overlap means "we cannot say", and collapsing the two would silently fabricate
    independence -- the exact error that the effective-sample-size diagnostic exists to
    prevent.
    """
    v = table.votes.astype(np.float64) * table.mask
    m = table.mask.astype(np.float64)
    both = m @ m.T
    prod = v @ v.T
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = np.where(both >= min_overlap, prod / np.maximum(both, 1.0), np.nan)
    np.fill_diagonal(agree, np.nan)
    return agree


@dataclass(slots=True)
class TripletEstimate:
    """Per-source accuracy from the triplet method, with its own provenance."""

    accuracy: dict[str, float]
    n_triplets: dict[str, int]
    covered: tuple[str, ...]
    uncovered: tuple[str, ...]
    default_accuracy: float

    def weight(self, source_id: str, cap: float = 4.0) -> float:
        """Log-odds vote weight ``log(a / (1-a))``, clipped.

        Clipping matters: an accuracy estimated from two triplets can land at 0.99 and
        would otherwise dominate every genuinely-informed source in the pool.
        """
        a = self.accuracy.get(source_id, self.default_accuracy)
        a = float(np.clip(a, 1e-3, 1 - 1e-3))
        return float(np.clip(np.log(a / (1.0 - a)), -cap, cap))

    def summary(self) -> dict[str, float | int]:
        vals = np.array(list(self.accuracy.values())) if self.accuracy else np.array([0.0])
        return {
            "covered": len(self.covered),
            "uncovered": len(self.uncovered),
            "accuracy_mean": round(float(vals.mean()), 4),
            "accuracy_min": round(float(vals.min()), 4),
            "accuracy_max": round(float(vals.max()), 4),
            "default_accuracy": round(self.default_accuracy, 4),
        }


def triplet_accuracies(
    claim_set: ClaimSet,
    *,
    min_overlap: int = 3,
    max_triplets_per_source: int = 64,
    default_accuracy: float = 0.6,
    floor: float = 0.5,
    ceiling: float = 0.99,
) -> TripletEstimate:
    """Estimate oriented static-source accuracy from pairwise agreements. Sources with insufficient overlap receive ``default_accuracy`` as an explicit assumption. Prior strength is not additional observed evidence."""
    table = BinaryClaimTable(claim_set)
    n = len(table.sources)
    if n < 3:
        return TripletEstimate({}, {}, (), table.sources, default_accuracy)

    agree = pairwise_agreement(table, min_overlap=min_overlap)
    valid = np.isfinite(agree)

    accuracy: dict[str, float] = {}
    counts: dict[str, int] = {}
    covered: list[str] = []
    uncovered: list[str] = []

    for i, source in enumerate(table.sources):
        partners = np.flatnonzero(valid[i])
        estimates: list[float] = []
        for j, k in combinations(partners.tolist(), 2):
            if not valid[j, k]:
                continue
            denom = agree[j, k]
            if abs(denom) < 1e-6:
                continue
            sq = agree[i, j] * agree[i, k] / denom
            if sq <= 0.0:
                continue
            estimates.append(float(np.sqrt(sq)))
            if len(estimates) >= max_triplets_per_source:
                break

        if estimates:
            # Median, not mean: individual triplet estimates are ratios of small-sample
            # correlations and are heavy-tailed, so the mean is dominated by the worst one.
            a_hat = float(np.median(estimates))
            # +/-1 correlation -> accuracy. Sign is resolved by assuming better-than-chance,
            # which is the identifiability symmetry-breaker, not a derived fact.
            accuracy[source] = float(np.clip(0.5 * (1.0 + a_hat), floor, ceiling))
            counts[source] = len(estimates)
            covered.append(source)
        else:
            accuracy[source] = default_accuracy
            counts[source] = 0
            uncovered.append(source)

    return TripletEstimate(
        accuracy=accuracy,
        n_triplets=counts,
        covered=tuple(covered),
        uncovered=tuple(uncovered),
        default_accuracy=default_accuracy,
    )


def pairwise_channels(
    claim_set: ClaimSet,
    template_of: Callable[[AssertionRef], str | None],
    *,
    prior_weight: float = 4.0,
    default_df: float = 3.0,
) -> ContinuousChannelTable:
    """Continuous channel bias and scale from pairwise disagreement, with no labels.

    The continuous sibling of `triplet_accuracies`. Where the triplet method reads accuracies
    off agreement *rates*, this reads a scale off the spread of the differences between two
    sources speaking about the same quantity. For independent channels

    ```text
    Var(y_i - y_j) = sigma_i^2 + sigma_j^2         E[y_i - y_j] = bias_i - bias_j
    ```

    so both parameters follow from co-claims alone. Biases are identified only up to a common
    offset -- adding a constant to every source's clock is unobservable -- which is fixed here
    by centring the pool. That is the same kind of stated symmetry-breaker as the
    better-than-chance prior on the binary side, and it deserves the same visibility: it is a
    choice the data cannot make.

    **Why not residuals against a point estimate of the truth?** Because with a median
    assertion degree near one, most assertions have a single witness, and a source's residual
    against its own claim is identically zero. That estimator reports a confident scale of
    zero -- which would make every single-witness claim infinitely sharp, and is worse than
    reporting nothing. Pairwise differences simply do not exist without overlap, so the thin
    case falls through to the template's pooled default by construction.

    Medians and a median-absolute-deviation scale throughout, because the quantities being
    summarised are exactly the gross outliers the Student-t channel exists to survive.
    """
    differences: dict[tuple[str, str], list[float]] = defaultdict(list)
    pooled: dict[str, list[float]] = defaultdict(list)

    for ref in claim_set.refs:
        template = template_of(ref)
        if template is None:
            continue
        values = [
            (c.source_id, float(c.value))
            for c in claim_set.for_ref(ref)
            if isinstance(c.value, (int, float)) and not isinstance(c.value, bool)
        ]
        for i, (left, left_value) in enumerate(values):
            for right, right_value in values[i + 1 :]:
                delta = left_value - right_value
                differences[(template, left)].append(delta)
                differences[(template, right)].append(-delta)
                pooled[template].append(delta)

    table = ContinuousChannelTable()
    if not pooled:
        return table

    # A difference of two independent channels carries the variance of both, hence sqrt(2).
    scales = {t: max(_mad(np.asarray(d)) / np.sqrt(2.0), 1e-6) for t, d in pooled.items()}
    for template, scale in scales.items():
        table.set_default(template, ContinuousChannel(0.0, scale))

    raw: dict[tuple[str, str], tuple[float, float]] = {}
    for key, deltas in differences.items():
        d = np.asarray(deltas, dtype=np.float64)
        shrink = d.size / (d.size + prior_weight)
        # Halved: the median difference against the pool estimates twice this source's own
        # offset from it, since the comparison set carries the mirrored offset.
        bias = shrink * float(np.median(d)) * 0.5
        scale = shrink * _mad(d) / np.sqrt(2.0) + (1.0 - shrink) * scales[key[0]]
        raw[key] = (bias, max(scale, 1e-6))

    by_template: dict[str, list[float]] = defaultdict(list)
    for (template, _source), (bias, _scale) in raw.items():
        by_template[template].append(bias)
    centre = {t: float(np.median(b)) for t, b in by_template.items()}

    for (template, source), (bias, scale) in raw.items():
        table.set(template, source, ContinuousChannel(bias - centre[template], scale))
    return table


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, rescaled to estimate a Gaussian standard deviation."""
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))
