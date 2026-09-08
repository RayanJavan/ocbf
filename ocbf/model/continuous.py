"""Assembling the Gaussian block from a universe, a claim set, and a fitted copula.

The continuous counterpart of [`ocbf.model.build`][ocbf.model.build]. The banks grounded
here, and what each is for:

============================  ==========================================================
bank                          role
============================  ==========================================================
``time_bracket``              the universe's coarse time bracket, as interval censoring
``time_prior``                event-type-conditioned time mean, where the type is known
``cg_time``                   CG coupling of a timestamp to its event's latent type
``continuous_channel``        one site per continuous claim, through the copula warp
``copula_correlation``        the latent precision structure of design doc section 4.1
``precedence``                SOFT: declared lifecycle orderings, as truncation
============================  ==========================================================

Three of these are the reason the layer earns its keep. ``cg_time`` makes a timestamp
inform an event's *type* and the type inform the timestamp -- the coupling design doc
section 4.2 asks for. ``precedence`` is the first factor in the system that constrains
*when* things happened relative to each other. ``copula_correlation`` is what makes two
attributes of the same object more than two independent scalars.

Everything here is grounded per template, in the par-factor sense of design doc
section 2.1: one bank per marginal, so all of a template's groundings share its parameters
(which is pooling) and its message shape (which is vectorisation).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ocbf.assertions import AssertionRef, Family, VariableRegistry
from ocbf.belief import BeliefState
from ocbf.model.copula import (
    EVENT_TIME_KEY,
    CopulaSpec,
    coupling_precision,
    fit_copula,
    marginal_key,
)
from ocbf.model.gaussian import GaussianGraph
from ocbf.model.gaussian_banks import (
    DEFAULT_QUADRATURE,
    CGMeanBank,
    CorrelationBank,
    GaussianEvidenceBank,
    IntervalBank,
    ObservationBank,
    PrecedenceBank,
    student_t_loglik,
    tempered_loglik,
)
from ocbf.reliability.params import ContinuousChannelTable, ReliabilityTable
from ocbf.reliability.moments import pairwise_channels
from ocbf.schema import ConstraintClass, ConstraintRegister
from ocbf.sources import ClaimSet
from ocbf.universe import Universe


@dataclass(slots=True)
class ContinuousSpec:
    """Settings for the continuous layer.

    Every default is chosen so that a universe with no continuous evidence grounds an empty
    block and the layer costs nothing. The continuous layer is additive: switching it on
    where there is nothing continuous to say must not change any answer.
    """

    time_bracket: bool = True
    """Use each candidate event's ``[time_lo, time_hi]`` bracket as interval evidence.

    The bracket already exists for the temporal prune, where it informs only which links are
    possible. Read as censoring evidence it also gives every timestamp an informative
    posterior where no source spoke -- which is most of them.
    """

    precedence: bool = True
    precedence_slack: float = 1.0
    """Slack ``s`` in the soft precedence ``log sigmoid(delta / s)``, in observed units.

    Small relative to a typical inter-event gap makes the factor nearly a hard ordering;
    large makes it a gentle preference. It is the width over which "out of order" fades
    into "unusually close together".
    """

    precedence_min_weight: float = 0.05
    """Skip a precedence pair whose structural weight falls below this.

    The weight is the posterior probability of the whole configuration -- both events really
    link to the shared object, *and* both carry the two declared types -- so a pair below the
    threshold is one the discrete layer does not believe in. One in twenty is the reading.

    **This threshold is not a tuning knob, and lowering it to make factors appear is a
    mistake.** How many pairs survive is a property of the *link layer*: an unclaimed link
    sits at its derived prior of ``1/k`` over a group of ``k`` candidates, so a pair of them
    weighs about ``1/k^2`` wherever the threshold sits. Where the sources never mention the
    links, no threshold produces a factor worth grounding, because the configuration itself is
    not believed.

    So precedence is claim-anchored in effect rather than by construction -- the conclusion
    design doc section 11.1 reached for the type gate, arrived at from the other direction,
    and recorded with its measurements in section 11.11.
    """

    max_precedence_pairs: int = 200_000

    type_conditioned_time: bool = True
    """Ground the conditional-Gaussian coupling between an event's type and its timestamp.

    Design doc section 4.2's homogeneous CG, in the one place in this schema where mean
    modulation is unambiguously real: activities happen at characteristic points in a
    process, so ``Pay`` events are late and ``PlaceOrder`` events are early.
    """

    type_mean_prior_weight: float = 5.0
    """Pseudo-observations pulling a type's fitted time mean toward the global mean.

    Partial pooling, for the same reason the reliability GLM pools: a type seen three times
    should not be handed a sharp mean of its own. Larger shrinks harder.
    """

    min_conditional_sd: float = 0.25
    """Floor on the CG conditional standard deviation, in latent units.

    Without it a type observed at nearly one moment gets a near-zero conditional variance and
    its log-partition dominates every other term in the discrete graph -- the continuous
    analogue of the variance collapse design doc section 11.4 records for the parameter block.
    """

    n_quadrature: int = DEFAULT_QUADRATURE


@dataclass(slots=True)
class ContinuousGrounding:
    """A grounded Gaussian block, plus what the hybrid loop needs to talk to it.

    ``cg_targets`` maps a block position to the discrete variable its conditional-Gaussian
    factor is coupled to, and ``report`` records what grounding built and skipped. Both are
    returned rather than hidden inside the graph because the coupling is the *hybrid* loop's
    business: the Gaussian engine holds no discrete variables, and handing it any would blur
    the split that keeps both engines simple.
    """

    graph: GaussianGraph
    copula: CopulaSpec
    cg_targets: dict[int, int] = field(default_factory=dict)
    report: dict[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether the block has no variables or no factors worth running an engine on."""
        return self.graph.n_vars == 0 or self.graph.n_edges == 0

    def summary(self) -> dict[str, object]:
        return {**self.graph.summary(), **self.copula.summary(), **self.report}


# -- membership -------------------------------------------------------------------------


def continuous_refs(universe: Universe) -> list[AssertionRef]:
    """Every registered continuous assertion, in registry order.

    Timestamps and copula-eligible attributes. Unordered categorical attributes are absent
    by construction: they have no monotone image in a Gaussian and stay in the discrete
    layer (design doc section 4.2).
    """
    registry = universe.registry
    out: list[AssertionRef] = []
    for family in (Family.EVENT_TIME, Family.EVENT_ATTR, Family.OBJECT_ATTR):
        for idx in registry.family_ids(family):
            if int(registry.cardinalities[idx]) == 0:
                out.append(registry.ref(int(idx)))
    return out


# -- fitting the copula -------------------------------------------------------------------


def point_estimates(claim_set: ClaimSet, refs: Iterable[AssertionRef]) -> dict[AssertionRef, float]:
    """Median claim value per continuous assertion.

    The median rather than the mean, and for the same reason the channel is Student-t: one
    gross outlier from an unreliable source should not move the estimate that the marginals
    and correlations are fitted from. These are inputs to *fitting*, not beliefs -- the
    posterior comes from the engine.
    """
    out: dict[AssertionRef, float] = {}
    for ref in refs:
        values = [
            float(c.value)
            for c in claim_set.for_ref(ref)
            if isinstance(c.value, (int, float)) and not isinstance(c.value, bool)
        ]
        if values:
            out[ref] = float(np.median(values))
    return out


def fit_copula_from_claims(
    universe: Universe, claim_set: ClaimSet, *, seed: int = 0
) -> CopulaSpec:
    """Fit marginals and latent correlations from what the sources reported.

    The values are claims, not truth, so a fitted marginal is the marginal of the *reported*
    values. That approximation is stated in
    [`fit_copula`][ocbf.model.copula.structure.fit_copula] and it is benign in the direction
    that matters: centred source noise widens a marginal without moving its shape, and a
    slightly over-dispersed marginal makes the layer more conservative rather than less.

    The event-time marginal falls back to the midpoints of the universe's time brackets when
    no source speaks about time. Those brackets are a genuine, if coarse, statement about
    when events happened, and using them keeps latent units meaningful even in a
    time-claim-free run.
    """
    refs = continuous_refs(universe)
    estimates = point_estimates(claim_set, refs)

    observations: dict[str, list[float]] = defaultdict(list)
    for ref in refs:
        value = estimates.get(ref)
        if value is not None:
            observations[marginal_key(ref)].append(value)

    if EVENT_TIME_KEY not in observations:
        midpoints = [
            0.5 * (ev.time_lo + ev.time_hi)
            for ev in universe.events.values()
            if np.isfinite(ev.time_lo) and np.isfinite(ev.time_hi)
        ]
        if midpoints:
            observations[EVENT_TIME_KEY] = midpoints

    return fit_copula(
        observations,
        schema=universe.schema,
        pairs=_schema_pairs(universe, estimates),
        seed=seed,
    )


def _schema_pairs(
    universe: Universe, estimates: Mapping[AssertionRef, float]
) -> list[tuple[str, str, list[float], list[float]]]:
    """Co-observed values for the template pairs design doc section 4.1 calls schema-given.

    Those are: an event attribute and its own event's timestamp, two attributes of one
    event, and two attributes of one object. Every other pair is left independent, because
    learning it would need co-observation this regime does not supply -- and a zero in the
    precision matrix is a statement, not a gap.
    """
    by_entity: dict[str, dict[str, float]] = defaultdict(dict)
    for ref, value in estimates.items():
        by_entity[ref.subject][marginal_key(ref)] = value

    paired: dict[tuple[str, str], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for templates in by_entity.values():
        keys = sorted(templates)
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                xs, ys = paired[(left, right)]
                xs.append(templates[left])
                ys.append(templates[right])
    return [(left, right, xs, ys) for (left, right), (xs, ys) in paired.items() if len(xs) >= 3]


# -- grounding ----------------------------------------------------------------------------


def build_continuous_graph(
    universe: Universe,
    claim_set: ClaimSet,
    reliability: ReliabilityTable,
    *,
    channels: ContinuousChannelTable | None = None,
    copula: CopulaSpec | None = None,
    discrete: BeliefState | None = None,
    spec: ContinuousSpec | None = None,
    register: ConstraintRegister | None = None,
    seed: int = 0,
) -> ContinuousGrounding:
    """Ground every continuous factor template against the universe.

    ``discrete`` is the current belief over the discrete backbone. The factors that need it
    -- precedence, which is grounded on links and types the discrete layer believes in, and
    the conditional-Gaussian coupling, whose mixture weights *are* a type posterior -- are
    omitted when it is absent. That is why the pipeline runs belief propagation first: the
    continuous layer is grounded against a discrete belief, not beside one.
    """
    spec = spec or ContinuousSpec()
    register = register or ConstraintRegister()
    registry = universe.registry
    refs = continuous_refs(universe)
    var_ids = np.array(
        [idx for idx in (registry.get(r) for r in refs) if idx is not None], dtype=np.int64
    )
    position = {int(v): i for i, v in enumerate(var_ids)}

    if copula is None:
        copula = fit_copula_from_claims(universe, claim_set, seed=seed)
    if channels is None:
        channels = pairwise_channels(claim_set, _template_of(universe))

    banks: list[object] = []
    report: dict[str, int] = {}

    if spec.time_bracket:
        bank = _time_bracket_bank(universe, copula, position)
        if bank is not None:
            banks.append(bank)

    channel_banks = _channel_banks(
        universe, claim_set, reliability, channels, copula, position, spec
    )
    banks.extend(channel_banks)
    report["continuous_claims"] = sum(b.n_factors() for b in channel_banks)

    correlation = _correlation_bank(universe, copula, position)
    if correlation is not None:
        banks.append(correlation)

    cg_targets: dict[int, int] = {}
    if spec.type_conditioned_time and discrete is not None:
        cg_bank, fixed_bank, cg_targets = _type_time_banks(
            universe, claim_set, copula, discrete, position, spec
        )
        if cg_bank is not None:
            banks.append(cg_bank)
        if fixed_bank is not None:
            banks.append(fixed_bank)

    if spec.precedence and discrete is not None and register.is_enabled(
        ConstraintClass.LIFECYCLE_PRECEDENCE
    ):
        bank, skipped = _precedence_bank(universe, copula, discrete, position, spec, register)
        report["precedence_skipped"] = skipped
        if bank is not None:
            banks.append(bank)

    graph = GaussianGraph(registry, var_ids, banks)  # type: ignore[arg-type]
    return ContinuousGrounding(graph, copula, cg_targets, report)


def _time_bracket_bank(
    universe: Universe, copula: CopulaSpec, position: Mapping[int, int]
) -> IntervalBank | None:
    """Each event's coarse time bracket, read as interval censoring on its timestamp."""
    if EVENT_TIME_KEY not in copula.marginals:
        return None
    marginal = copula.marginals[EVENT_TIME_KEY]
    positions: list[int] = []
    lo: list[float] = []
    hi: list[float] = []
    for event in universe.events.values():
        idx = universe.registry.get(AssertionRef.event_time(event.id))
        if idx is None or int(idx) not in position:
            continue
        if not (np.isfinite(event.time_lo) or np.isfinite(event.time_hi)):
            continue  # an unbounded bracket says nothing, so it grounds no factor
        positions.append(position[int(idx)])
        low, high = event.time_lo, event.time_hi
        lo.append(float(marginal.to_latent(low)) if np.isfinite(low) else -np.inf)
        hi.append(float(marginal.to_latent(high)) if np.isfinite(high) else np.inf)
    if not positions:
        return None
    return IntervalBank(positions, lo, hi, name="time_bracket")


def _template_of(universe: Universe):
    """Template key of a ref, or ``None`` where the ref is not in the Gaussian block.

    Passed to [`pairwise_channels`][ocbf.reliability.moments.pairwise_channels] so that the
    estimator groups claims exactly as the banks will, without the reliability package
    needing to know what a copula is.
    """
    registry = universe.registry

    def key(ref: AssertionRef) -> str | None:
        idx = registry.get(ref)
        if idx is None or int(registry.cardinalities[idx]) != 0:
            return None
        try:
            return marginal_key(ref)
        except ValueError:
            return None

    return key


def _channel_banks(
    universe: Universe,
    claim_set: ClaimSet,
    reliability: ReliabilityTable,
    channels: ContinuousChannelTable,
    copula: CopulaSpec,
    position: Mapping[int, int],
    spec: ContinuousSpec,
) -> list[ObservationBank]:
    """One bank per marginal template, carrying every continuous claim against it.

    Grouping by template is the par-factor principle rather than a micro-optimisation: all
    the groundings of one template share a marginal, so the copula warp is one vectorised
    call per bank instead of one per claim.
    """
    grouped: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for claim in claim_set:
        if not isinstance(claim.value, (int, float)) or isinstance(claim.value, bool):
            continue
        idx = universe.registry.get(claim.ref)
        if idx is None or int(idx) not in position:
            continue
        try:
            key = marginal_key(claim.ref)
        except ValueError:
            continue
        if key not in copula.marginals:
            continue
        grouped[key].append((position[int(idx)], float(claim.value), claim.source_id))

    banks: list[ObservationBank] = []
    for key, entries in sorted(grouped.items()):
        positions = np.array([e[0] for e in entries], dtype=np.int64)
        values = np.array([e[1] for e in entries], dtype=np.float64)
        labels = [e[2] for e in entries]
        channel = [channels[(key, sid)] for sid in labels]
        params = [reliability[sid] for sid in labels]
        loglik = student_t_loglik(
            values,
            np.array([c.scale for c in channel], dtype=np.float64),
            np.array([p.noise_df for p in params], dtype=np.float64),
            np.array([c.bias for c in channel], dtype=np.float64),
        )
        temperatures = np.array([p.temperature for p in params], dtype=np.float64)
        if not np.allclose(temperatures, 1.0):
            loglik = tempered_loglik(loglik, temperatures)
        banks.append(
            ObservationBank(
                positions,
                loglik,
                copula.marginals[key],
                name="continuous_channel",
                labels=labels,
                n_quadrature=spec.n_quadrature,
            )
        )
    return banks


def _correlation_bank(
    universe: Universe, copula: CopulaSpec, position: Mapping[int, int]
) -> CorrelationBank | None:
    """Ground the copula's fitted correlations onto co-located pairs of coordinates."""
    if not copula.correlations:
        return None
    registry = universe.registry
    by_entity: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for pos_id, block_pos in position.items():
        ref = registry.ref(int(pos_id))
        by_entity[ref.subject].append((marginal_key(ref), block_pos))

    a_ids: list[int] = []
    b_ids: list[int] = []
    blocks: list[np.ndarray] = []
    for entries in by_entity.values():
        entries.sort()
        for i, (left_key, left_pos) in enumerate(entries):
            for right_key, right_pos in entries[i + 1 :]:
                rho = copula.correlation(left_key, right_key)
                if abs(rho) < 1e-6:
                    continue
                a_ids.append(left_pos)
                b_ids.append(right_pos)
                blocks.append(coupling_precision(rho))
    if not a_ids:
        return None
    return CorrelationBank(a_ids, b_ids, np.stack(blocks), name="copula_correlation")


def _type_time_banks(
    universe: Universe,
    claim_set: ClaimSet,
    copula: CopulaSpec,
    discrete: BeliefState,
    position: Mapping[int, int],
    spec: ContinuousSpec,
) -> tuple[CGMeanBank | None, GaussianEvidenceBank | None, dict[int, int]]:
    """The conditional-Gaussian coupling between an event's type and its timestamp.

    Events whose type support is a singleton have no ``T_e`` variable, so their type is
    known and its time mean enters as ordinary Gaussian evidence. Both banks read the same
    fitted means, so a determined event and an uncertain one are treated consistently
    rather than by two different models.
    """
    if EVENT_TIME_KEY not in copula.marginals:
        return None, None, {}
    fit = fit_type_conditioned_means(universe, claim_set, copula, discrete, spec)
    if fit is None:
        return None, None, {}
    means_by_type, conditional_var = fit

    registry = universe.registry
    latent: list[tuple[int, int, tuple[str, ...]]] = []
    determined: list[tuple[int, float]] = []
    for event in universe.events.values():
        time_idx = registry.get(AssertionRef.event_time(event.id))
        if time_idx is None or int(time_idx) not in position:
            continue
        block_pos = position[int(time_idx)]
        domain = universe.event_type_domain(event.id)
        type_idx = registry.get(AssertionRef.event_type(event.id))
        if type_idx is None:
            determined.append((block_pos, means_by_type.get(domain[0], 0.0)))
        else:
            latent.append((block_pos, int(type_idx), domain))

    cg_bank = None
    cg_targets: dict[int, int] = {}
    if latent:
        width = max(len(d) for _, _, d in latent)
        state_means = np.zeros((len(latent), width))
        state_probs = np.zeros((len(latent), width))
        positions = np.empty(len(latent), dtype=np.int64)
        labels: list[str] = []
        for row, (block_pos, type_idx, domain) in enumerate(latent):
            probs = discrete.marginal(registry.ref(type_idx))
            positions[row] = block_pos
            state_means[row, : len(domain)] = [means_by_type.get(t, 0.0) for t in domain]
            state_probs[row, : len(probs)] = probs
            labels.append(registry.ref(type_idx).subject)
            cg_targets[block_pos] = type_idx
        cg_bank = CGMeanBank(
            positions, state_means, state_probs, conditional_var, name="cg_time", labels=labels
        )

    fixed_bank = None
    if determined:
        fixed_bank = GaussianEvidenceBank.from_moments(
            [p for p, _ in determined],
            [m for _, m in determined],
            np.full(len(determined), conditional_var),
            name="time_prior",
        )
    return cg_bank, fixed_bank, cg_targets


def fit_type_conditioned_means(
    universe: Universe,
    claim_set: ClaimSet,
    copula: CopulaSpec,
    discrete: BeliefState,
    spec: ContinuousSpec,
) -> tuple[dict[str, float], float] | None:
    """Latent time mean per event type, and the shared conditional variance.

    The prior block of design doc section 6.1, applied to the continuous layer: refit each
    outer iteration from **soft** posterior counts, so an event contributes to a type's mean
    in proportion to the current belief that it has that type. Shrunk toward the global mean
    by ``type_mean_prior_weight`` pseudo-observations, because a type seen three times should
    not be given a sharp mean of its own.

    Homogeneous conditional Gaussian, so the variance is shared across types (design doc
    section 4.2) -- discrete states shift the mean and nothing else. Returns ``None`` when no
    event has a usable time estimate, which is the honest outcome of a run in which nobody
    mentioned time.
    """
    marginal = copula.marginals[EVENT_TIME_KEY]
    registry = universe.registry
    latent_estimates: dict[str, float] = {}
    for ref, value in point_estimates(
        claim_set, (r for r in continuous_refs(universe) if r.family is Family.EVENT_TIME)
    ).items():
        latent_estimates[ref.subject] = float(marginal.to_latent(value))
    if not latent_estimates:
        return None

    global_mean = float(np.mean(list(latent_estimates.values())))
    weight_sum: dict[str, float] = defaultdict(float)
    value_sum: dict[str, float] = defaultdict(float)
    residuals: list[float] = []

    for event_id, z in latent_estimates.items():
        domain = universe.event_type_domain(event_id)
        type_idx = registry.get(AssertionRef.event_type(event_id))
        probs = (
            discrete.marginal(registry.ref(type_idx))
            if type_idx is not None
            else np.ones(len(domain)) / len(domain)
        )
        for t, p in zip(domain, probs[: len(domain)], strict=False):
            weight_sum[t] += float(p)
            value_sum[t] += float(p) * z

    means: dict[str, float] = {}
    for t, w in weight_sum.items():
        means[t] = (value_sum[t] + spec.type_mean_prior_weight * global_mean) / (
            w + spec.type_mean_prior_weight
        )

    for event_id, z in latent_estimates.items():
        domain = universe.event_type_domain(event_id)
        type_idx = registry.get(AssertionRef.event_type(event_id))
        probs = (
            discrete.marginal(registry.ref(type_idx))
            if type_idx is not None
            else np.ones(len(domain)) / len(domain)
        )
        for t, p in zip(domain, probs[: len(domain)], strict=False):
            residuals.append(float(p) * (z - means.get(t, global_mean)) ** 2)

    total_weight = sum(weight_sum.values())
    variance = float(np.sum(residuals) / total_weight) if total_weight > 0 else 1.0
    return means, max(variance, spec.min_conditional_sd**2)


def _precedence_bank(
    universe: Universe,
    copula: CopulaSpec,
    discrete: BeliefState,
    position: Mapping[int, int],
    spec: ContinuousSpec,
    register: ConstraintRegister,
) -> tuple[PrecedenceBank | None, int]:
    """Declared lifecycle orderings, as soft truncations on latent time differences.

    Whether a declared ordering *applies* to a pair of candidate events depends on discrete
    variables -- do both events really link to the shared object, and do they carry the two
    declared types. The exact treatment is a higher-order conditional-Gaussian factor over
    ``(T_e, T_e', R, R', tau, tau')``. This grounds the mean-field form instead: the pair's
    weight is the posterior probability that the configuration holds, which is precisely the
    structured mean-field split design doc section 6.1 already commits to between the belief
    and parameter blocks.

    The approximation is bounded and stated: a pair the discrete layer is unsure about
    exerts a proportionally weaker ordering pressure, and one it disbelieves exerts none.
    """
    constraint = register[ConstraintClass.LIFECYCLE_PRECEDENCE]
    if EVENT_TIME_KEY not in copula.marginals:
        return None, 0
    marginal = copula.marginals[EVENT_TIME_KEY]
    registry = universe.registry

    before_ids: list[int] = []
    after_ids: list[int] = []
    weights: list[float] = []
    skipped = 0
    floor = spec.precedence_min_weight

    for obj in universe.objects.values():
        lifecycle = universe.schema.lifecycle(obj.object_type)
        if lifecycle is None:
            continue
        # Each of the four terms is a probability, so the product cannot exceed any one of
        # them: dropping an event whose link posterior is already below the floor is exact,
        # not a heuristic. It is also what makes this affordable -- a candidate universe puts
        # a hundred or more candidate events on one object, and the unpruned double loop is
        # quadratic in that.
        attached = {
            e: p for e, p in _attached_events(universe, discrete, obj.id).items() if p >= floor
        }
        if len(attached) < 2:
            continue
        for before_type, after_type in lifecycle.ordered_pairs():
            early_side = _typed(universe, discrete, attached, before_type, floor)
            late_side = _typed(universe, discrete, attached, after_type, floor)
            for early, early_weight in early_side.items():
                for late, late_weight in late_side.items():
                    if early == late:
                        continue
                    weight = early_weight * late_weight
                    if weight < floor:
                        skipped += 1
                        continue
                    early_pos = _time_position(registry, position, early)
                    late_pos = _time_position(registry, position, late)
                    if early_pos is None or late_pos is None:
                        continue
                    before_ids.append(early_pos)
                    after_ids.append(late_pos)
                    weights.append(weight * constraint.weight)
                    if len(before_ids) >= spec.max_precedence_pairs:
                        break

    if not before_ids:
        return None, skipped
    # The slack is stated in observed units and the factor acts on latent ones, so it is
    # divided by the marginal's local scale -- the same conversion that makes a tolerance
    # in hours mean the same thing as one in euros.
    slack = max(spec.precedence_slack / max(marginal.scale, 1e-9), 1e-6)
    return (
        PrecedenceBank(
            before_ids,
            after_ids,
            slack,
            np.asarray(weights, dtype=np.float64),
            hard=constraint.is_hard,
            name="precedence",
            n_quadrature=spec.n_quadrature,
        ),
        skipped,
    )


def _typed(
    universe: Universe,
    discrete: BeliefState,
    attached: Mapping[str, float],
    event_type: str,
    floor: float,
) -> dict[str, float]:
    """Attached events plausibly of one type, carrying ``P(link) * P(type)``.

    Folding the type probability in here rather than at the innermost loop turns a quadratic
    scan into two linear ones: an event that cannot be the ``before`` type is dropped once,
    not once per candidate partner.
    """
    out: dict[str, float] = {}
    for event_id, link in attached.items():
        combined = link * _type_probability(universe, discrete, event_id, event_type)
        if combined >= floor:
            out[event_id] = combined
    return out


def _attached_events(
    universe: Universe, discrete: BeliefState, object_id: str
) -> dict[str, float]:
    """Candidate events linked to an object, with the posterior that some link holds.

    Several qualifiers can connect one event to one object; the strongest is taken, because
    the lifecycle ordering is about the event being *about* this object at all, not about
    the role it plays.
    """
    out: dict[str, float] = {}
    for event_id, qualifier, _obj in universe.e2o_of_object(object_id):
        p = discrete.prob_true(AssertionRef.e2o(event_id, qualifier, object_id), default=0.0)
        out[event_id] = max(out.get(event_id, 0.0), p)
    return out


def _type_probability(
    universe: Universe, discrete: BeliefState, event_id: str, event_type: str
) -> float:
    """``P(T_e = event_type)``, reading a determined type as certainty."""
    domain = universe.event_type_domain(event_id)
    if event_type not in domain:
        return 0.0
    idx = universe.registry.get(AssertionRef.event_type(event_id))
    if idx is None:
        return 1.0 if domain[0] == event_type else 0.0
    return float(discrete.marginal(universe.registry.ref(idx))[domain.index(event_type)])


def _time_position(
    registry: VariableRegistry, position: Mapping[int, int], event_id: str
) -> int | None:
    idx = registry.get(AssertionRef.event_time(event_id))
    return None if idx is None else position.get(int(idx))
