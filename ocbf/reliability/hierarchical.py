"""The hierarchical reliability GLM -- the parameter block of the two-block scheme.

Stage 1 decision 3. Reliability is indexed by ``(source, assertion family)``, not by source
alone, and it is a pooled regression rather than a free parameter per source:

```text
logit(theta_{s,f}) =  b0                 # global intercept, prior mass above chance
                    + x_s . beta         # source features (SLiMFast)
                    + u_{c(s)}           # declared source-family random effect
                    + m_f                # assertion-family main effect
                    + delta_{c(s), f}    # SPECIALISATION: cluster x family interaction
                    + eps_s              # individual, shrunk toward the population
```

Each term answers a specific finding from Stage 1:

``x_s . beta``
    SLiMFast's feature-regressed reliability. Turns ``|S|`` free parameters into ``d << |S|``
    and generalises to sources never seen before -- at 1e5 sources this is the difference
    between a fittable model and an unfittable one.

``eps_s`` with a small ``sigma_eps``
    Partial pooling. A source with three claims shrinks to its cluster; one with ten
    thousand dominates its own estimate. Automatic, no thresholds.

``delta_{c,f}``
    A barcode scanner is excellent at object identity and useless at activity semantics. A
    single scalar per source cannot say that; this can.

``u_{c(s)}``
    Shared within a *declared* source family, this **is** the dependency correction. Sources
    sharing a vendor or an upstream model share the effect, so their agreement is partly
    explained rather than counted as independent confirmation.

The likelihood uses **soft counts** from the current belief state, which makes this the
exact M-step of the variational EM in design doc section 6.1. Because the counts are
fractional, the observation enters as a weighted log-likelihood ``Potential`` rather than a
Binomial -- the same objective, without pretending the counts are integers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import Family
from ocbf.belief import BeliefState
from ocbf.reliability.params import ReliabilityTable, SourceParams
from ocbf.sources import ClaimSet, CoverageSemantics, Source

_EPS = 1e-9


@dataclass(slots=True)
class SoftCounts:
    """Expected confusion counts per ``(source, assertion family)`` under the posterior."""

    source_ids: list[str]
    families: list[str]
    source_index: np.ndarray
    family_index: np.ndarray
    cluster_index: np.ndarray
    features: np.ndarray
    tp: np.ndarray
    fn: np.ndarray
    tn: np.ndarray
    fp: np.ndarray
    cluster_names: list[str]

    def __len__(self) -> int:
        return len(self.tp)

    @property
    def n_sources(self) -> int:
        return len(self.source_ids)

    @property
    def n_families(self) -> int:
        return len(self.families)

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_names)

    def empirical(self) -> tuple[np.ndarray, np.ndarray]:
        """Unpooled per-cell rates, for comparison against the pooled fit."""
        sens = (self.tp + 1.0) / (self.tp + self.fn + 2.0)
        spec = (self.tn + 1.0) / (self.tn + self.fp + 2.0)
        return sens, spec


def soft_counts(
    claim_set: ClaimSet,
    belief: BeliefState,
    sources: Mapping[str, Source],
    *,
    include_silence: bool = True,
) -> SoftCounts:
    """Accumulate expected confusion counts under the current posterior.

    Silence from non-opportunistic sources is counted as a negative claim, which is what
    lets a detector's false-negative rate be estimated at all.
    """
    profiles = {sid: s.profile for sid, s in sources.items()}
    cluster_names = sorted({p.cluster_id for p in profiles.values()}) or ["default"]
    cluster_of = {c: i for i, c in enumerate(cluster_names)}

    source_ids = sorted(set(claim_set.source_ids) | set(profiles))
    src_of = {s: i for i, s in enumerate(source_ids)}
    fam_names = sorted({r.family.value for r in claim_set.refs})
    fam_of = {f: i for i, f in enumerate(fam_names)}

    n_feat = max((len(p.features) for p in profiles.values()), default=0)
    features = np.zeros((len(source_ids), n_feat), dtype=np.float64)
    clusters = np.zeros(len(source_ids), dtype=np.int64)
    for sid, p in profiles.items():
        i = src_of[sid]
        clusters[i] = cluster_of.get(p.cluster_id, 0)
        if n_feat:
            features[i, : len(p.features)] = p.features

    cells: dict[tuple[int, int], np.ndarray] = {}

    def bump(source_id: str, family: str, said_true: bool, p_true: float) -> None:
        key = (src_of[source_id], fam_of[family])
        cell = cells.setdefault(key, np.zeros(4))
        if said_true:
            cell[0] += p_true          # tp
            cell[3] += 1.0 - p_true    # fp
        else:
            cell[1] += p_true          # fn
            cell[2] += 1.0 - p_true    # tn

    for claim in claim_set:
        if not isinstance(claim.value, bool):
            continue
        if claim.ref.family.value not in fam_of:
            continue
        idx = belief.registry.get(claim.ref)
        if idx is None:
            continue
        bump(claim.source_id, claim.ref.family.value, claim.value, belief.prob_true(claim.ref))

    if include_silence:
        for source in sources.values():
            if not source.profile.coverage.silence_is_evidence:
                continue
            sid = source.profile.source_id
            claimed = {c.ref for c in source.claims()}
            for ref in source.scope():
                if ref in claimed or ref.family.value not in fam_of:
                    continue
                idx = belief.registry.get(ref)
                # Only binary assertions have a two-sided confusion to accumulate; a
                # categorical assertion's silence is not a "no" about anything in
                # particular, so it carries no information for this channel.
                if idx is None or int(belief.registry.cardinalities[idx]) != 2:
                    continue
                bump(sid, ref.family.value, False, belief.prob_true(ref))

    keys = sorted(cells)
    counts = np.array([cells[k] for k in keys]) if keys else np.zeros((0, 4))
    return SoftCounts(
        source_ids=source_ids,
        families=fam_names,
        source_index=np.array([k[0] for k in keys], dtype=np.int64),
        family_index=np.array([k[1] for k in keys], dtype=np.int64),
        cluster_index=clusters,
        features=features,
        tp=counts[:, 0] if len(counts) else np.zeros(0),
        fn=counts[:, 1] if len(counts) else np.zeros(0),
        tn=counts[:, 2] if len(counts) else np.zeros(0),
        fp=counts[:, 3] if len(counts) else np.zeros(0),
        cluster_names=cluster_names,
    )


@dataclass(slots=True)
class HierarchicalFit:
    """Fitted reliability, plus the variance components that reveal how much pooling ran."""

    table: ReliabilityTable
    sigma_cluster: float
    sigma_individual: float
    sigma_specialisation: float
    intercept_sensitivity: float
    intercept_specificity: float
    method: str
    n_cells: int

    def summary(self) -> dict[str, object]:
        return {
            "method": self.method,
            "cells": self.n_cells,
            "intercept_sens": round(self.intercept_sensitivity, 4),
            "intercept_spec": round(self.intercept_specificity, 4),
            "sigma_cluster": round(self.sigma_cluster, 4),
            "sigma_specialisation": round(self.sigma_specialisation, 4),
            "sigma_individual": round(self.sigma_individual, 4),
            **{f"table_{k}": v for k, v in self.table.summary().items()},
        }


def fit_hierarchical(
    counts: SoftCounts,
    *,
    method: str = "map",
    draws: int = 500,
    tune: int = 500,
    seed: int = 0,
    prior_intercept: float = 1.0,
    progressbar: bool = False,
) -> HierarchicalFit:
    """Fit the reliability hierarchy with PyMC.

    ``method="map"`` gives a penalised point estimate and is the default: the design's
    two-block loop calls this once per outer iteration, and a full posterior every time
    would dominate the runtime for little gain in the E-step. ``method="nuts"`` gives the
    full posterior when the reliability estimates are themselves the object of interest.

    ``prior_intercept`` is positive on the logit scale, putting prior mass on
    better-than-chance sources. This is not a convenience -- it is the symmetry-breaker for
    the "all sources are adversarial" mirror solution, which has identical likelihood and
    which the data alone cannot rule out (Stage 1 section 10, item 1).
    """
    if len(counts) == 0:
        return HierarchicalFit(
            ReliabilityTable(), 0.0, 0.0, 0.0, prior_intercept, prior_intercept, method, 0
        )

    import pymc as pm
    import pytensor

    n_src = counts.n_sources
    n_fam = counts.n_families
    n_clu = counts.n_clusters
    n_feat = counts.features.shape[1]

    cell_src = counts.source_index
    cell_fam = counts.family_index
    cell_clu = counts.cluster_index[cell_src]

    # Standardise features so one shared prior scale is meaningful across columns.
    feats = counts.features
    if n_feat:
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True)
        feats = (feats - mu) / np.where(sd > 1e-9, sd, 1.0)

    # Compile through PyTensor's C linker rather than whatever backend happens to be
    # default. The numba backend pulls in llvmlite's executable-memory allocator, which on
    # this platform intermittently faults once other libraries have already claimed
    # address space -- a crash with nothing to do with the model. The graph here is a few
    # hundred parameters, so the C linker is fast enough that the backend choice costs
    # nothing. Scoped via change_flags so the process-wide config is left alone.
    with pytensor.config.change_flags(mode="FAST_COMPILE"), pm.Model():
        b0_sens = pm.Normal("b0_sens", mu=prior_intercept, sigma=1.0)
        b0_spec = pm.Normal("b0_spec", mu=prior_intercept, sigma=1.0)

        shared = 0.0
        if n_feat:
            beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=n_feat)
            shared = shared + pm.math.dot(feats, beta)[cell_src]

        # Variance components get Gamma priors whose mode is away from zero, and the random
        # effects are non-centred.
        #
        # Both choices are forced by using MAP. The joint mode of a hierarchical model sits
        # at sigma = 0 with every random effect collapsed -- a well-known pathology, and a
        # silent one: it reports a clean fit while switching off exactly the partial
        # pooling that the sparse regime depends on. A HalfNormal has its mode at zero and
        # so invites it. Gamma(2, r) has mode 1/r > 0, encoding the thing we already know
        # to be true, namely that sources do differ from one another. The non-centred form
        # additionally decouples each effect from its scale, which conditions the geometry
        # for both the optimiser and NUTS.
        sigma_u = pm.Gamma("sigma_cluster", alpha=2.0, beta=6.0)
        u = pm.Normal("u_raw", mu=0.0, sigma=1.0, shape=n_clu) * sigma_u
        shared = shared + u[cell_clu]

        m = pm.Normal("m", mu=0.0, sigma=0.5, shape=n_fam)
        shared = shared + m[cell_fam]

        sigma_d = pm.Gamma("sigma_specialisation", alpha=2.0, beta=8.0)
        delta = pm.Normal("delta_raw", mu=0.0, sigma=1.0, shape=(n_clu, n_fam)) * sigma_d
        shared = shared + delta[cell_clu, cell_fam]

        sigma_e = pm.Gamma("sigma_individual", alpha=2.0, beta=8.0)
        eps = pm.Normal("eps_raw", mu=0.0, sigma=1.0, shape=n_src) * sigma_e
        shared = shared + eps[cell_src]

        sens = pm.Deterministic("sens", pm.math.sigmoid(b0_sens + shared))
        spec = pm.Deterministic("spec", pm.math.sigmoid(b0_spec + shared))

        # Weighted log-likelihood over fractional counts: the exact EM M-step objective.
        pm.Potential(
            "soft_confusion",
            (
                counts.tp * pm.math.log(sens + _EPS)
                + counts.fn * pm.math.log(1.0 - sens + _EPS)
                + counts.tn * pm.math.log(spec + _EPS)
                + counts.fp * pm.math.log(1.0 - spec + _EPS)
            ).sum(),
        )

        if method == "nuts":
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=2,
                random_seed=seed,
                progressbar=progressbar,
                compute_convergence_checks=False,
            )
            post = idata.posterior
            point = {k: post[k].mean(dim=("chain", "draw")).values for k in post.data_vars}
        else:
            point = pm.find_MAP(progressbar=progressbar, seed=seed)

    sens_hat = np.atleast_1d(point["sens"])
    spec_hat = np.atleast_1d(point["spec"])

    # Average a source's cells into one parameter pair. The per-family split is retained in
    # the fitted effects; the current channel model consumes one pair per source.
    acc_s = np.full(n_src, np.nan)
    acc_p = np.full(n_src, np.nan)
    for cell, s in enumerate(cell_src):
        acc_s[s] = sens_hat[cell] if np.isnan(acc_s[s]) else 0.5 * (acc_s[s] + sens_hat[cell])
        acc_p[s] = spec_hat[cell] if np.isnan(acc_p[s]) else 0.5 * (acc_p[s] + spec_hat[cell])

    default_sens = float(np.nanmean(acc_s)) if np.isfinite(acc_s).any() else 0.6
    default_spec = float(np.nanmean(acc_p)) if np.isfinite(acc_p).any() else 0.6
    params = {
        sid: SourceParams(
            sensitivity=float(acc_s[i]) if np.isfinite(acc_s[i]) else default_sens,
            specificity=float(acc_p[i]) if np.isfinite(acc_p[i]) else default_spec,
            rho=float(acc_s[i]) if np.isfinite(acc_s[i]) else default_sens,
        ).clipped()
        for i, sid in enumerate(counts.source_ids)
    }
    table = ReliabilityTable(
        params, SourceParams(default_sens, default_spec, default_sens).clipped()
    )

    return HierarchicalFit(
        table=table,
        sigma_cluster=float(np.atleast_1d(point["sigma_cluster"])[0]),
        sigma_individual=float(np.atleast_1d(point["sigma_individual"])[0]),
        sigma_specialisation=float(np.atleast_1d(point["sigma_specialisation"])[0]),
        intercept_sensitivity=float(np.atleast_1d(point["b0_sens"])[0]),
        intercept_specificity=float(np.atleast_1d(point["b0_spec"])[0]),
        method=method,
        n_cells=len(counts),
    )
