"""Query-estimator diagnostics, distinct from source-evidence overlap/ESS."""

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _rhat(x):
    chains, draws = x.shape
    if chains < 2 or draws < 4:
        return None
    half = draws // 2
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)

    def value(a):
        ranks = rankdata(a, method="average").reshape(a.shape)
        z = ndtri((ranks - 0.375) / (ranks.size + 0.25))
        within = np.var(z, axis=1, ddof=1).mean()
        between = half * np.var(z.mean(axis=1), ddof=1)
        if within == 0:
            return None if between == 0 else float("inf")
        return float(np.sqrt(((half - 1) / half * within + between / half) / within))

    values = (value(split), value(np.abs(split - np.median(split))))
    return max(v for v in values if v is not None) if any(v is not None for v in values) else None


def assessment(numerator, denominator=None, *, method="mcmc"):
    """Delta-method ratio influence series with joint numerator/denominator covariance.

    FFT autocovariances and Geyer initial positive paired sequence assess serial error.
    A constant observed series cannot certify that a rare event is impossible.
    """
    a = np.asarray(numerator, dtype=float)
    b = np.ones_like(a) if denominator is None else np.asarray(denominator, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("diagnostics require finite aligned chain/iteration arrays")
    if b.mean() <= 0:
        return {"status": "undefined_denominator", "mcse": None, "mcmc_ess": None}
    estimate = a.mean() / b.mean()
    influence = (a - estimate * b) / b.mean()
    _, draws = a.shape
    variance = np.var(influence, ddof=1) if a.size > 1 else 0.0
    rhat = _rhat(influence) if method == "mcmc" else None
    if variance == 0 or draws < 4:
        return {
            "status": "constant_or_insufficient_draws",
            "mcse": None,
            "mcmc_ess": None,
            "rhat": rhat,
            "samples": a.size,
        }
    if method == "iid":
        ess = float(a.size)
    else:
        centered = influence - influence.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(centered, n=2 * draws, axis=1)
        autocov = (
            np.fft.irfft(spectrum * np.conjugate(spectrum), axis=1)[:, :draws].mean(axis=0) / draws
        )
        if autocov[0] <= 0:
            return {
                "status": "between_chain_disagreement",
                "mcse": None,
                "mcmc_ess": None,
                "rhat": rhat,
            }
        rho = autocov / autocov[0]
        pairs = rho[1 : 1 + 2 * ((draws - 1) // 2)].reshape(-1, 2).sum(axis=1)
        positive = np.flatnonzero(pairs <= 0)
        pairs = pairs[: positive[0]] if positive.size else pairs
        pairs = np.minimum.accumulate(pairs)
        ess = float(min(a.size, a.size / max(1.0, 1 + 2 * pairs.sum())))
    adequate = method == "iid" or (rhat is not None and rhat <= 1.01 and ess >= 100)
    return {
        "status": "assessed" if adequate else "numerically_inadequate",
        "mcse": float(np.sqrt(variance / ess)),
        "mcmc_ess": ess if method == "mcmc" else None,
        "iid_samples": a.size if method == "iid" else None,
        "rhat": rhat,
        "interval_meaning": "estimated numerical standard error; not posterior spread or calibration",
    }
