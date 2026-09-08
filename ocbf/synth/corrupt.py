"""Source simulation: reproducing the sparse / unreliable / numerous regime.

Design doc section 8.2. Every axis of Stage 1 section 4.7 gets an independent knob, because
the whole point of the evaluation is to sweep them and find where the model stops working:

* ``deg(a)`` -- sources per assertion, controlling decidability;
* ``deg(s)`` -- assertions per source, drawn long-tailed, controlling how hard pooling has
  to work;
* accuracy -- barely-above-chance upward;
* **copy structure** -- within-cluster copying, which is what the effective-sample-size
  diagnostic exists to catch;
* coverage semantics, including a deliberately mis-declared one;
* specialisation -- sources strong on one assertion family and weak on others.

Source scopes are deliberately **localised** (a contiguous window over the sorted assertion
pool) rather than uniform random subsets. Real sparse sources look at a region -- a time
window, one object type, one sensor's neighbourhood -- and that locality is precisely what
produces the thin, islanded overlap graph that the identifiability diagnostic of design doc
section 7.1 is meant to detect. A uniform-random scope would manufacture an artificially
well-connected overlap graph and hide the failure mode.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef, Family
from ocbf.model.copula import marginal_key
from ocbf.sources import (
    ChannelFamily,
    Claim,
    CoverageSemantics,
    SourceProfile,
    StaticSource,
)
from ocbf.synth.process import GroundTruth
from ocbf.universe import Universe

_DEFAULT_FAMILIES: tuple[Family, ...] = (
    Family.E2O,
    Family.EVENT_EXISTS,
    Family.EVENT_TYPE,
    Family.O2O,
)


@dataclass(slots=True)
class SourceRegime:
    """Knobs on the evidence layer."""

    n_sources: int = 200
    n_clusters: int = 6

    accuracy_mean: float = 0.62
    """Population mean of per-source accuracy. 0.62 is deliberately near chance."""

    cluster_accuracy_sd: float = 0.10
    individual_accuracy_sd: float = 0.05
    accuracy_floor: float = 0.5
    accuracy_ceiling: float = 0.97

    deg_s_alpha: float = 1.8
    """Pareto exponent for assertions-per-source. Smaller means a heavier tail."""

    deg_s_min: int = 8
    deg_s_max: int = 400

    n_hotspots: int = 24
    """Number of regions sources concentrate on.

    Sources do not scan the candidate space uniformly -- they observe reality, and the
    sensors are somewhere. Several cameras watch the packing station; nobody watches the
    space of links that never happen. Hotspots are anchored on *true* assertions, which
    gives realistic overlap: regions covered by several sources and regions covered by
    none. That islanded structure is exactly what the identifiability diagnostic of design
    doc section 7.1 must detect, and a uniform scope would manufacture an artificially
    well-connected overlap graph that hides the failure mode.

    Lower means more concentration and higher ``deg(a)``; raise it to sweep toward the
    undecidable extreme.
    """

    hotspot_spread: float = 2.5
    """Window width as a multiple of the source's target scope size."""

    copy_rate: float = 0.30
    """Fraction of sources that copy a cluster-mate rather than observing independently."""

    copy_fidelity: float = 0.9
    """Fraction of a copied source's claims taken verbatim from its exemplar."""

    specialisation_strength: float = 0.25
    """How much a cluster's accuracy moves between its strong and weak families."""

    coverage_mix: tuple[tuple[CoverageSemantics, float], ...] = (
        (CoverageSemantics.OPPORTUNISTIC, 0.6),
        (CoverageSemantics.COMPLETE_OVER_SCOPE, 0.3),
        (CoverageSemantics.SELECTIVE, 0.1),
    )

    misdeclared_coverage_rate: float = 0.0
    """Fraction of sources whose declared coverage differs from their true behaviour.

    Used to test whether fitting ``gamma_s`` detects a mis-declaration (design doc
    section 5.2).
    """

    families: tuple[Family, ...] = _DEFAULT_FAMILIES
    time_bias_sd: float = 2.0
    time_noise_sd: float = 4.0

    attribute_bias_fraction: float = 0.15
    attribute_noise_fraction: float = 0.35
    """Attribute bias and noise, as fractions of the attribute's own spread.

    Relative rather than absolute because attributes are not commensurable: an order value in
    the hundreds and a three-level priority cannot share a noise scale in observed units, and
    giving them one would make the benchmark a test of unit choice. A fraction of the
    template's spread means the same difficulty for every attribute -- which is what the
    latent standardisation of the copula expresses on the model's side.
    """

    seed: int = 0

    def __post_init__(self) -> None:
        total = sum(w for _, w in self.coverage_mix)
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(f"coverage_mix weights must sum to 1, got {total}")
        if not 0.0 <= self.copy_rate <= 1.0:
            raise ValueError("copy_rate must be in [0, 1]")


@dataclass(slots=True)
class SourceTruth:
    """The true generating parameters of one simulated source.

    Returned so evaluation can compare *recovered* reliability against *true* reliability
    -- without this the benchmark can only score assertions, not the reliability model,
    and the reliability model is the part the sparse regime stresses.
    """

    source_id: str
    cluster_id: str
    sensitivity: float
    specificity: float
    rho: float
    bias_z: float
    """Standardised systematic offset of this source's continuous claims.

    Standardised so one draw serves every template: multiplied by a template's own bias
    scale it becomes a clock offset in hours or a systematic over-reporting of order value,
    and a source that runs fast on one is the same source that runs fast on the other.
    """

    time_bias: float
    declared_coverage: CoverageSemantics
    true_coverage: CoverageSemantics
    copies: str | None
    n_claims: int

    @property
    def is_misdeclared(self) -> bool:
        return self.declared_coverage is not self.true_coverage


@dataclass(slots=True)
class SimulatedSources:
    sources: list[StaticSource]
    truth: dict[str, SourceTruth]

    def summary(self) -> dict[str, float | int]:
        acc = np.array([t.sensitivity for t in self.truth.values()])
        n = np.array([t.n_claims for t in self.truth.values()])
        return {
            "sources": len(self.sources),
            "claims": int(n.sum()),
            "copiers": sum(1 for t in self.truth.values() if t.copies),
            "misdeclared": sum(1 for t in self.truth.values() if t.is_misdeclared),
            "sensitivity_mean": round(float(acc.mean()), 4),
            "sensitivity_min": round(float(acc.min()), 4),
            "deg_s_median": int(np.median(n)),
            "deg_s_max": int(n.max()),
        }


def simulate_sources(
    universe: Universe, truth: GroundTruth, regime: SourceRegime | None = None
) -> SimulatedSources:
    """Generate a population of sparse, unreliable, mutually-dependent sources."""
    reg = regime or SourceRegime()
    rng = np.random.default_rng(reg.seed + 104729)

    pools = _assertion_pools(universe, reg.families)
    if not pools:
        raise ValueError("no assertions available for the requested families")
    channel_scales = _channel_scales(pools, truth, reg)

    clusters = _make_clusters(reg, rng)
    hotspots = _make_hotspots(pools, truth, reg, rng)
    profiles: list[tuple[SourceProfile, SourceTruth, list[AssertionRef]]] = []

    for i in range(reg.n_sources):
        sid = f"src_{i:05d}"
        cluster = clusters[int(rng.integers(len(clusters)))]
        family = cluster.pick_family(rng)
        if family not in pools:
            family = next(iter(pools))
        pool = pools[family]

        scope = _localised_scope(pool, hotspots[family], reg, rng)
        if not scope:
            continue

        base = cluster.accuracy_for(family, reg)
        sens = _clip(base + rng.normal(0.0, reg.individual_accuracy_sd), reg)
        spec = _clip(base + rng.normal(0.0, reg.individual_accuracy_sd), reg)
        rho = _clip(base + rng.normal(0.0, reg.individual_accuracy_sd), reg)

        true_cov = _pick_coverage(reg, rng)
        declared = true_cov
        if rng.random() < reg.misdeclared_coverage_rate:
            options = [c for c, _ in reg.coverage_mix if c is not true_cov]
            declared = options[int(rng.integers(len(options)))]

        profile = SourceProfile(
            source_id=sid,
            coverage=declared,
            channel=ChannelFamily.for_family(family),
            cluster_id=cluster.name,
            features=(cluster.modality_code, float(len(scope)), cluster.base_accuracy),
            feature_names=("modality", "scope_size", "cluster_prior_accuracy"),
        )
        bias_z = float(rng.normal(0.0, 1.0))
        st = SourceTruth(
            source_id=sid,
            cluster_id=cluster.name,
            sensitivity=sens,
            specificity=spec,
            rho=rho,
            bias_z=bias_z,
            time_bias=bias_z * reg.time_bias_sd,
            declared_coverage=declared,
            true_coverage=true_cov,
            copies=None,
            n_claims=0,
        )
        profiles.append((profile, st, scope))

    # -- copying: this is what makes agreement spurious and ESS < deg(a) ------------
    by_cluster: dict[str, list[int]] = {}
    for idx, (profile, _st, _scope) in enumerate(profiles):
        by_cluster.setdefault(profile.cluster_id, []).append(idx)

    copy_from: dict[int, int] = {}
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        for idx in members:
            if rng.random() < reg.copy_rate:
                exemplar = int(rng.choice([m for m in members if m != idx]))
                copy_from[idx] = exemplar
                profiles[idx][1].copies = profiles[exemplar][0].source_id

    sources: list[StaticSource] = []
    truths: dict[str, SourceTruth] = {}
    emitted: dict[int, list[Claim]] = {}

    # Independent observers first, so copiers have something to copy.
    order = sorted(range(len(profiles)), key=lambda i: i in copy_from)
    for idx in order:
        profile, st, scope = profiles[idx]
        exemplar = copy_from.get(idx)
        if exemplar is not None and emitted.get(exemplar):
            claims = _copy_claims(profile, emitted[exemplar], scope, reg, rng)
        else:
            claims = _observe(profile, st, scope, universe, truth, rng, reg, channel_scales)
        emitted[idx] = claims

        st.n_claims = len(claims)
        declared_scope = frozenset(scope) if st.declared_coverage.silence_is_evidence else None
        if declared_scope is not None and declared_scope == frozenset(c.ref for c in claims):
            # No silence to interpret; degrade the declaration rather than raise.
            profile = SourceProfile(
                source_id=profile.source_id,
                coverage=CoverageSemantics.OPPORTUNISTIC,
                channel=profile.channel,
                cluster_id=profile.cluster_id,
                features=profile.features,
                feature_names=profile.feature_names,
            )
            st.declared_coverage = CoverageSemantics.OPPORTUNISTIC
            declared_scope = None
        sources.append(StaticSource(profile, claims, declared_scope))
        truths[profile.source_id] = st

    return SimulatedSources(sources, truths)


# -- internals -------------------------------------------------------------------------


@dataclass(slots=True)
class _Cluster:
    name: str
    base_accuracy: float
    strong_family: Family
    weak_family: Family
    modality_code: float
    families: tuple[Family, ...]

    def pick_family(self, rng: np.random.Generator) -> Family:
        """Clusters mostly work in their strong family -- that is what specialisation means."""
        if rng.random() < 0.7:
            return self.strong_family
        return self.families[int(rng.integers(len(self.families)))]

    def accuracy_for(self, family: Family, reg: SourceRegime) -> float:
        if family is self.strong_family:
            return self.base_accuracy + reg.specialisation_strength
        if family is self.weak_family:
            return self.base_accuracy - reg.specialisation_strength
        return self.base_accuracy


def _make_clusters(reg: SourceRegime, rng: np.random.Generator) -> list[_Cluster]:
    fams = reg.families
    out: list[_Cluster] = []
    for c in range(reg.n_clusters):
        base = float(rng.normal(reg.accuracy_mean, reg.cluster_accuracy_sd))
        strong = fams[c % len(fams)]
        weak = fams[(c + 1) % len(fams)]
        out.append(
            _Cluster(
                name=f"cluster_{c}",
                base_accuracy=base,
                strong_family=strong,
                weak_family=weak,
                modality_code=float(c),
                families=fams,
            )
        )
    return out


def _assertion_pools(
    universe: Universe, families: Sequence[Family]
) -> dict[Family, list[AssertionRef]]:
    """Sorted assertion lists per family.

    Sorted order *is* the locality metric: refs sort by subject, subjects are event ids,
    and event ids are issued in process order, so adjacency in this list approximates
    adjacency in the process. That is what makes a contiguous window a physically
    meaningful notion of "what one sensor can see".
    """
    pools: dict[Family, list[AssertionRef]] = {}
    for fam in families:
        refs = [universe.registry.ref(i) for i in universe.registry.family_ids(fam)]
        if refs:
            pools[fam] = sorted(refs)
    return pools


def _make_hotspots(
    pools: Mapping[Family, list[AssertionRef]],
    truth: GroundTruth,
    reg: SourceRegime,
    rng: np.random.Generator,
) -> dict[Family, np.ndarray]:
    """Pool positions that sources concentrate on, anchored on true assertions.

    Anchoring on truth is about *where the sensors are*, not about leaking the answer: a
    hotspot only says "sources look here", and the window around it still contains mostly
    false candidates that the source must get right or wrong on its own merits.
    """
    hotspots: dict[Family, np.ndarray] = {}
    for family, pool in pools.items():
        # Identity rather than equality: ``0.0 == False`` in Python, so a genuine zero-valued
        # attribute would be read as a false assertion and quietly excluded from every hotspot.
        positive = [
            i
            for i, ref in enumerate(pool)
            if (v := truth.truth(ref)) is not None and v is not False
        ]
        candidates = positive or list(range(len(pool)))
        k = min(reg.n_hotspots, len(candidates))
        picked = rng.choice(len(candidates), size=k, replace=False)
        hotspots[family] = np.array(sorted(candidates[int(i)] for i in picked), dtype=np.int64)
    return hotspots


def _localised_scope(
    pool: list[AssertionRef],
    hotspots: np.ndarray,
    reg: SourceRegime,
    rng: np.random.Generator,
) -> list[AssertionRef]:
    """A contiguous window around a hotspot, thinned to a long-tailed size."""
    target = int(min(reg.deg_s_max, max(reg.deg_s_min, rng.pareto(reg.deg_s_alpha) * reg.deg_s_min)))
    target = min(target, len(pool))
    window = min(len(pool), max(target, int(target * reg.hotspot_spread)))

    anchor = int(hotspots[int(rng.integers(len(hotspots)))]) if len(hotspots) else int(
        rng.integers(len(pool))
    )
    start = int(np.clip(anchor - window // 2, 0, max(0, len(pool) - window)))
    segment = pool[start : start + window]
    if target >= len(segment):
        return list(segment)
    picked = rng.choice(len(segment), size=target, replace=False)
    return [segment[int(i)] for i in sorted(picked)]


def _pick_coverage(reg: SourceRegime, rng: np.random.Generator) -> CoverageSemantics:
    modes = [c for c, _ in reg.coverage_mix]
    weights = np.array([w for _, w in reg.coverage_mix], dtype=float)
    return modes[int(rng.choice(len(modes), p=weights / weights.sum()))]


def _clip(x: float, reg: SourceRegime) -> float:
    return float(min(reg.accuracy_ceiling, max(reg.accuracy_floor, x)))


def _observe(
    profile: SourceProfile,
    st: SourceTruth,
    scope: Sequence[AssertionRef],
    universe: Universe,
    truth: GroundTruth,
    rng: np.random.Generator,
    reg: SourceRegime,
    scales: Mapping[str, tuple[float, float]],
) -> list[Claim]:
    claims: list[Claim] = []
    for ref in scope:
        actual = truth.truth(ref)
        match profile.channel:
            case ChannelFamily.BINARY:
                claim = _observe_binary(profile, st, ref, bool(actual), rng)
            case ChannelFamily.CATEGORICAL:
                claim = _observe_categorical(profile, st, ref, actual, universe, rng)
            case ChannelFamily.CONTINUOUS:
                claim = _observe_continuous(profile, st, ref, actual, rng, scales)
            case _:
                claim = None
        if claim is not None:
            claims.append(claim)
    return claims


def _observe_binary(
    profile: SourceProfile,
    st: SourceTruth,
    ref: AssertionRef,
    actual: bool,
    rng: np.random.Generator,
) -> Claim | None:
    if actual:
        observed = rng.random() < st.sensitivity
    else:
        observed = rng.random() >= st.specificity

    match st.true_coverage:
        case CoverageSemantics.COMPLETE_OVER_SCOPE:
            # Detector semantics: it speaks only when it fires. Its silence is the
            # negative claim, and that silence is what carries the false-negative rate.
            return Claim(profile.source_id, ref, True) if observed else None
        case CoverageSemantics.SELECTIVE:
            # Truth-dependent propensity: more likely to report on true assertions.
            propensity = 0.75 if actual else 0.25
            if rng.random() > propensity:
                return None
            return Claim(profile.source_id, ref, observed)
        case _:
            return Claim(profile.source_id, ref, observed)


def _observe_categorical(
    profile: SourceProfile,
    st: SourceTruth,
    ref: AssertionRef,
    actual: object,
    universe: Universe,
    rng: np.random.Generator,
) -> Claim | None:
    if not isinstance(actual, str):
        return None  # decoy event: no true type to report
    domain = universe.event_type_domain(ref.subject)
    if actual not in domain:
        return None
    if rng.random() < st.rho:
        return Claim(profile.source_id, ref, actual)
    others = [t for t in domain if t != actual]
    if not others:
        return None
    return Claim(profile.source_id, ref, others[int(rng.integers(len(others)))])


def _observe_continuous(
    profile: SourceProfile,
    st: SourceTruth,
    ref: AssertionRef,
    actual: object,
    rng: np.random.Generator,
    scales: Mapping[str, tuple[float, float]],
) -> Claim | None:
    """``y = v + bias_s + eps`` with Student-t noise -- design doc section 5.1's channel.

    Student-t rather than Gaussian because unreliable sources produce gross outliers, and a
    benchmark whose noise was Gaussian would never test the robustness the channel is chosen
    for.
    """
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return None
    bias_scale, noise_scale = scales.get(marginal_key(ref), (0.0, 1.0))
    noise = float(rng.standard_t(df=3.0)) * noise_scale
    return Claim(profile.source_id, ref, float(actual) + st.bias_z * bias_scale + noise)


def _channel_scales(
    pools: Mapping[Family, list[AssertionRef]],
    truth: GroundTruth,
    reg: SourceRegime,
) -> dict[str, tuple[float, float]]:
    """``(bias scale, noise scale)`` per continuous template, in that template's own units.

    Timestamps keep their absolute scales, which is how the regime has always stated them.
    Attributes get scales proportional to their own observed spread, because an order value
    and a three-level priority are not commensurable and a shared absolute noise would make
    one trivial and the other pure noise.
    """
    out: dict[str, tuple[float, float]] = {}
    for family, refs in pools.items():
        if family is Family.EVENT_TIME:
            out[marginal_key(refs[0])] = (reg.time_bias_sd, reg.time_noise_sd)
            continue
        if family not in (Family.EVENT_ATTR, Family.OBJECT_ATTR):
            continue
        by_template: dict[str, list[float]] = defaultdict(list)
        for ref in refs:
            value = truth.truth(ref)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                by_template[marginal_key(ref)].append(float(value))
        for key, values in by_template.items():
            spread = float(np.std(values)) if len(values) > 1 else 0.0
            spread = spread if spread > 0.0 else 1.0
            out[key] = (reg.attribute_bias_fraction * spread, reg.attribute_noise_fraction * spread)
    return out


def _copy_claims(
    profile: SourceProfile,
    exemplar_claims: Sequence[Claim],
    scope: Sequence[AssertionRef],
    reg: SourceRegime,
    rng: np.random.Generator,
) -> list[Claim]:
    """Take an exemplar's claims verbatim, restricted to this source's own scope.

    This is the dependency the model must not mistake for independent confirmation.
    """
    in_scope = set(scope)
    candidates = [c for c in exemplar_claims if c.ref in in_scope]
    if not candidates:
        return []
    keep = max(1, int(round(reg.copy_fidelity * len(candidates))))
    picked = rng.choice(len(candidates), size=min(keep, len(candidates)), replace=False)
    return [
        Claim(profile.source_id, candidates[int(i)].ref, candidates[int(i)].value)
        for i in sorted(picked)
    ]
