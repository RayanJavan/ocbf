"""Continuous calibration and scoring against supplied reference values.

CRPS, probability integral transforms and interval coverage compare predictive summaries
with reference truth. The reference's physical quality must be established separately.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from ocbf.assertions import AssertionRef
from ocbf.belief import BeliefState

_SQRT_PI = float(np.sqrt(np.pi))


@dataclass(slots=True)
class ContinuousReport:
    """Accuracy, sharpness and calibration on one set of continuous assertions."""

    n: int
    rmse: float
    mae: float
    crps: float
    coverage: float
    nominal_coverage: float
    interval_width: float
    n_prior_only: int

    @property
    def coverage_error(self) -> float:
        """Signed gap between realised and nominal coverage.

        The continuous analogue of expected calibration error, and read the same way. Below
        nominal means overconfident -- the intervals are too narrow and the stated
        probabilities overstate what the model knows. Above nominal means the opposite, which
        is the safer failure but still a miscalibration.
        """
        return self.coverage - self.nominal_coverage

    def as_dict(self) -> dict[str, float | int]:
        """The report as a flat, rounded mapping suitable for tabulating."""
        return {
            "n": self.n,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "crps": round(self.crps, 4),
            "coverage": round(self.coverage, 4),
            "coverage_error": round(self.coverage_error, 4),
            "interval_width": round(self.interval_width, 4),
            "n_prior_only": self.n_prior_only,
        }


def gaussian_crps(mean: np.ndarray, var: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score of ``N(mean, var)`` against an observation.

    ```text
    CRPS = sd * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],    z = (y - mean) / sd
    ```

    A **proper** scoring rule, which is the whole reason it is here rather than a squared
    error: it is minimised only by the true predictive distribution, so it cannot be gamed by
    reporting a confident point. It is the continuous generalisation of the Brier score
    [`ocbf.eval.metrics`][ocbf.eval.metrics] uses on binary assertions, and it rewards
    sharpness and calibration together instead of trading one against the other.
    """
    sd = np.sqrt(np.maximum(var, 1e-300))
    z = (np.asarray(truth, dtype=np.float64) - mean) / sd
    density = np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)
    return sd * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * density - 1.0 / _SQRT_PI)


def evaluate_continuous(
    belief: BeliefState,
    truth: Mapping[AssertionRef, float | None],
    refs: Sequence[AssertionRef] | None = None,
    *,
    mass: float = 0.9,
) -> ContinuousReport:
    """Score a belief state's continuous posteriors against known values.

    Assertions the belief state cannot place in observed units -- no marginal attached, so no
    units to be right or wrong in -- are skipped rather than scored against a latent
    coordinate. Prior-only assertions *are* scored and counted separately: a posterior that
    fell back to its prior is still an answer the caller receives, and hiding it would
    flatter the model exactly where it knows least.
    """
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    keys = list(refs) if refs is not None else list(truth)

    latent_error: list[float] = []
    observed_error: list[float] = []
    crps: list[float] = []
    covered: list[bool] = []
    widths: list[float] = []
    prior_only = 0

    for ref in keys:
        actual = truth.get(ref)
        if actual is None or isinstance(actual, bool) or not isinstance(actual, (int, float)):
            continue
        try:
            mean, var = belief.gaussian(ref)
            lo, hi = belief.value_interval(ref, mass)
            predicted = belief.value(ref)
        except (KeyError, TypeError):
            continue
        if not np.isfinite(mean) or not np.isfinite(var):
            continue

        z_truth = float(np.asarray(belief.transform(ref).to_latent(float(actual))))
        latent_error.append(z_truth - mean)
        observed_error.append(predicted - float(actual))
        crps.append(float(gaussian_crps(np.array(mean), np.array(var), np.array(z_truth))))
        covered.append(lo <= float(actual) <= hi)
        widths.append(hi - lo)
        prior_only += int(belief.is_prior_only(ref))

    if not crps:
        blank = float("nan")
        return ContinuousReport(0, blank, blank, blank, blank, mass, blank, 0)

    observed = np.asarray(observed_error)
    return ContinuousReport(
        n=len(crps),
        rmse=float(np.sqrt(np.mean(np.square(observed)))),
        mae=float(np.mean(np.abs(observed))),
        crps=float(np.mean(crps)),
        coverage=float(np.mean(covered)),
        nominal_coverage=mass,
        interval_width=float(np.mean(widths)),
        n_prior_only=prior_only,
    )


def compare_continuous(
    beliefs: Mapping[str, BeliefState],
    truth: Mapping[AssertionRef, float | None],
    refs: Sequence[AssertionRef] | None = None,
    *,
    mass: float = 0.9,
) -> dict[str, dict[str, float | int]]:
    """Score methods on the same supplied assertion set and reference truth. Resolve references once so missing predictions cannot silently change the comparison population."""
    keys = list(refs) if refs is not None else list(truth)
    return {
        name: evaluate_continuous(b, truth, keys, mass=mass).as_dict()
        for name, b in beliefs.items()
    }


def pit_values(
    belief: BeliefState,
    truth: Mapping[AssertionRef, float | None],
    refs: Iterable[AssertionRef],
) -> np.ndarray:
    """Probability-integral transforms ``Phi((z_true - mean) / sd)``.

    Uniform on ``[0, 1]`` exactly when the posterior is calibrated, which makes their
    histogram the continuous reliability diagram: a hump in the middle means intervals that
    are too wide, mass at both ends means too narrow, and a tilt means a systematic bias.
    Computed on latent coordinates because the transform is monotone, so the PIT is the same
    number in both spaces and the latent one needs no inverse.
    """
    out: list[float] = []
    for ref in refs:
        actual = truth.get(ref)
        if actual is None or isinstance(actual, bool) or not isinstance(actual, (int, float)):
            continue
        try:
            mean, var = belief.gaussian(ref)
            z_truth = float(np.asarray(belief.transform(ref).to_latent(float(actual))))
        except (KeyError, TypeError):
            continue
        out.append(float(ndtr((z_truth - mean) / np.sqrt(max(var, 1e-300)))))
    return np.asarray(out, dtype=np.float64)


def normal_quantile(mass: float) -> float:
    """Half-width in standard deviations of a central interval of the given mass."""
    return float(ndtri(0.5 * (1.0 + mass)))
