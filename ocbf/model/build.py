"""Assembling the factor graph from a universe, a claim set, and reliability parameters.

The banks built here, and what each is for:

============================  ==========================================================
bank                          role
============================  ==========================================================
``prior``                     structural priors on existence, links and event type
``channel``                   one unary per claim: the source likelihood
``silence``                   negative evidence from non-opportunistic sources' silence
``referential_integrity``     HARD: a link implies both endpoints exist
``type_gate``                 HARD: a link is impossible under event types that forbid it
``cardinality``               SOFT: declared multiplicity of a qualified relation
============================  ==========================================================

The last three are what distinguish this from parallel per-assertion voting. They are the
edges along which belief reaches assertions no source ever covered, which Stage 1
section 4.2 shows is where most assertions live.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ocbf.assertions import AssertionRef, Family, VariableRegistry
from ocbf.model.banks import CardinalityBank, PairwiseBank, UnaryBank
from ocbf.model.graph import NEG_INF, FactorGraph
from ocbf.reliability.params import ReliabilityTable
from ocbf.schema import ConstraintClass, ConstraintRegister
from ocbf.schema.core import Multiplicity
from ocbf.sources import ClaimSet, CoverageSemantics, Source
from ocbf.universe import Universe


@dataclass(slots=True)
class GraphSpec:
    """Structural prior strengths.

    These are the beliefs that carry an assertion when the sources cannot. The defaults are
    deliberately weak-but-informative: strong enough to break ties and propagate structure,
    weak enough that a few good claims override them.
    """

    p_event_exists: float = 0.6
    p_object_exists: float = 0.9

    link_prior_floor: float = 0.005
    link_prior_ceiling: float = 0.5
    """Bounds on the *derived* link prior (see `_e2o_link_priors`).

    The E2O link prior is not a fixed constant. It is computed per
    ``(event, qualifier)`` group from the declared multiplicity and the number of surviving
    candidates: a group of ``k`` candidates under an ``exactly-one`` qualifier gets ``1/k``,
    not some global default.

    A flat prior would be incoherent with the cardinality factor rather than merely
    imprecise. At ``p = 0.08`` over ``k = 50`` candidates the prior expects four links while
    the counting factor insists on one, so the two fight each other and the result depends
    on their relative weights instead of on the evidence. Deriving the prior from the same
    multiplicity the constraint uses makes them agree by construction.
    """

    o2o_prior_floor: float = 0.01
    o2o_prior_ceiling: float = 0.9

    constraint_weight_scale: float | None = None
    """Nats that one unit of soft-constraint violation costs.

    ``None`` means "calibrate from the sources": the scale is set to the median Chernoff
    information of a single claim, so a soft constraint costs about what one confident claim
    of evidence is worth and a few claims can overrule it. That is the behaviour design doc
    section 3.4 asks for, and pinning it to a raw nat count instead would silently change
    meaning whenever the source pool's quality changes.

    Set a float to override with an absolute scale.
    """

    include_referential_integrity: bool = True
    include_type_gate: bool = True
    include_cardinality: bool = True

    type_gate_claimed_only: bool = True
    """Attach the type gate only to links that carry at least one claim.

    Without this, every candidate link of an event contributes a hard gate factor to that
    event's single latent type variable -- routinely a hundred or more. Two things go wrong.

    The gate messages from *unclaimed* links are driven entirely by the link prior, so a
    type that permits many candidate links accumulates an advantage merely for having many
    candidates, which is not evidence about anything. And because all those links share the
    event's existence variable and its cardinality factors, they are far from independent,
    so loopy BP double-counts that spurious signal and drives the type posterior into a
    saturated corner it then oscillates between.

    Restricting to claimed links is the claim-anchoring principle of design doc section 2.3
    applied to this factor. The cost is precise and bounded: an unclaimed link retains its
    prior mass even under an event type that forbids it. Those assertions are prior-only
    and already flagged as such, so the relaxation is confined to exactly the assertions
    where the belief state makes no claim to begin with.
    """

    def __post_init__(self) -> None:
        for name in (
            "p_event_exists",
            "p_object_exists",
            "link_prior_floor",
            "link_prior_ceiling",
            "o2o_prior_floor",
            "o2o_prior_ceiling",
        ):
            v = getattr(self, name)
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {v}")
        if self.link_prior_floor >= self.link_prior_ceiling:
            raise ValueError("link_prior_floor must be below link_prior_ceiling")
        if self.constraint_weight_scale is not None and self.constraint_weight_scale <= 0:
            raise ValueError("constraint_weight_scale must be positive")


def _binary_logprior(p: float, width: int) -> np.ndarray:
    row = np.full(width, NEG_INF)
    row[0] = np.log1p(-p)
    row[1] = np.log(p)
    return row


def _expected_links(multiplicity: Multiplicity, k: int) -> float:
    """How many of ``k`` candidates the declared multiplicity expects to be true.

    Bounded multiplicities use the midpoint of their interval. Unbounded ones cannot be
    read off the schema at all -- ``[1..*]`` says "at least one" and nothing more -- so the
    lower bound is used, which is the weakest commitment the declaration supports.
    """
    lo = multiplicity.lo
    if multiplicity.hi is not None:
        return 0.5 * (lo + min(multiplicity.hi, k))
    return float(max(lo, 1)) if lo > 0 else 1.0


def _e2o_link_priors(universe: Universe, spec: GraphSpec) -> dict[int, float]:
    """Per-link prior derived from the multiplicity of its ``(event, qualifier)`` group.

    Deriving rather than fixing keeps the prior coherent with the cardinality factor: both
    are read off the same declaration, so they push in the same direction instead of
    trading off against each other.

    Where the event's latent type support disagrees about the multiplicity, the most
    permissive reading is used -- the same conservative rule as the cardinality bank, and
    for the same reason: never penalise a configuration that some plausible type allows.
    """
    reg = universe.registry
    schema = universe.schema
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for event_id, qualifier, object_id in universe.e2o_candidates:
        idx = reg.get(AssertionRef.e2o(event_id, qualifier, object_id))
        if idx is not None:
            grouped[(event_id, qualifier)].append(int(idx))

    out: dict[int, float] = {}
    for (event_id, qualifier), members in grouped.items():
        domain = universe.event_type_domain(event_id)
        specs = [
            q for q in schema.e2o_qualifiers if q.qualifier == qualifier and q.event_type in domain
        ]
        k = len(members)
        if specs:
            expected = max(_expected_links(q.multiplicity, k) for q in specs)
        else:
            expected = 1.0
        p = float(
            np.clip(expected / max(k, 1), spec.link_prior_floor, spec.link_prior_ceiling)
        )
        for idx in members:
            out[idx] = p
    return out


def _o2o_link_priors(universe: Universe, spec: GraphSpec) -> dict[int, float]:
    """Same derivation for O2O, grouped by ``(source object, qualifier)``."""
    reg = universe.registry
    schema = universe.schema
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for source_id, qualifier, target_id in universe.o2o_candidates:
        idx = reg.get(AssertionRef.o2o(source_id, qualifier, target_id))
        if idx is not None:
            grouped[(source_id, qualifier)].append(int(idx))

    out: dict[int, float] = {}
    for (source_id, qualifier), members in grouped.items():
        source_type = universe.objects[source_id].object_type
        specs = [
            q
            for q in schema.o2o_qualifiers
            if q.qualifier == qualifier and q.source_type == source_type
        ]
        k = len(members)
        expected = max((_expected_links(q.multiplicity, k) for q in specs), default=1.0)
        p = float(np.clip(expected / max(k, 1), spec.o2o_prior_floor, spec.o2o_prior_ceiling))
        for idx in members:
            out[idx] = p
    return out


def calibrate_constraint_scale(
    claim_set: ClaimSet, reliability: ReliabilityTable, *, fallback: float = 2.0
) -> float:
    """Nats of evidence carried by a typical single claim.

    Soft-constraint weights are expressed against this scale so that "soft" keeps its
    meaning as the source pool changes. A weight of 1.0 then costs about one confident
    claim, and a handful of claims can overrule a misstated schema belief -- which is the
    behaviour design doc section 3.4 is asking for, and which a hardcoded nat count cannot
    deliver across pools of differing quality.
    """
    from ocbf.diagnostics.decidability import chernoff_binary

    values = [
        chernoff_binary(reliability[s].sensitivity, reliability[s].specificity)
        for s in claim_set.source_ids
    ]
    positive = [v for v in values if v > 1e-6]
    return float(np.median(positive)) if positive else fallback


def build_graph(
    universe: Universe,
    claim_set: ClaimSet,
    reliability: ReliabilityTable,
    *,
    sources: Mapping[str, Source] | None = None,
    spec: GraphSpec | None = None,
    register: ConstraintRegister | None = None,
    extra_banks: Sequence[object] = (),
) -> FactorGraph:
    """Ground every factor template against the universe.

    ``extra_banks`` are appended after the templated ones. The hybrid loop uses this to hand
    back the continuous layer's message to the discrete backbone -- the exact, closed-form
    conditional-Gaussian log-partition per state of design doc section 4.3. Passing it as an
    ordinary bank rather than wiring a special case keeps that message on the same footing as
    a source channel, including in the attribution vector, which is where a reader will look
    to find out why a timestamp changed an event's type.
    """
    spec = spec or GraphSpec()
    register = register or ConstraintRegister()
    reg = universe.registry
    width = max(reg.max_cardinality, 1)
    banks: list[object] = []

    # -- structural priors ----------------------------------------------------------
    prior_vars: list[int] = []
    prior_pots: list[np.ndarray] = []
    for family, p in (
        (Family.EVENT_EXISTS, spec.p_event_exists),
        (Family.OBJECT_EXISTS, spec.p_object_exists),
    ):
        for idx in reg.family_ids(family):
            prior_vars.append(int(idx))
            prior_pots.append(_binary_logprior(p, width))

    for idx, p in _e2o_link_priors(universe, spec).items():
        prior_vars.append(idx)
        prior_pots.append(_binary_logprior(p, width))
    for idx, p in _o2o_link_priors(universe, spec).items():
        prior_vars.append(idx)
        prior_pots.append(_binary_logprior(p, width))

    for idx in reg.family_ids(Family.EVENT_TYPE):
        c = int(reg.cardinalities[idx])
        row = np.full(width, NEG_INF)
        row[:c] = -np.log(c)
        prior_vars.append(int(idx))
        prior_pots.append(row)

    if prior_vars:
        banks.append(
            UnaryBank(np.array(prior_vars), np.array(prior_pots), name="prior")
        )

    # -- source channels ------------------------------------------------------------
    ch_vars, ch_pots, ch_labels = _channel_bank(universe, claim_set, reliability, width)
    if ch_vars:
        banks.append(
            UnaryBank(np.array(ch_vars), np.array(ch_pots), name="channel", labels=ch_labels)
        )

    if sources:
        si_vars, si_pots, si_labels = _silence_bank(universe, sources, reliability, width)
        if si_vars:
            banks.append(
                UnaryBank(np.array(si_vars), np.array(si_pots), name="silence", labels=si_labels)
            )

    # -- hard structural factors ----------------------------------------------------
    if spec.include_referential_integrity and register.is_enabled(
        ConstraintClass.REFERENTIAL_INTEGRITY
    ):
        bank = _referential_integrity_bank(universe, register)
        if bank is not None:
            banks.append(bank)

    if spec.include_type_gate and register.is_enabled(ConstraintClass.QUALIFIER_LEGALITY):
        anchored = set(claim_set.refs) if spec.type_gate_claimed_only else None
        bank = _type_gate_bank(universe, width, register, anchored)
        if bank is not None:
            banks.append(bank)

    # -- soft cardinality -----------------------------------------------------------
    # Soft weights are multipliers on one claim's worth of evidence, so the scale is read
    # off the source pool rather than fixed in nats (see calibrate_constraint_scale).
    weight_scale = (
        spec.constraint_weight_scale
        if spec.constraint_weight_scale is not None
        else calibrate_constraint_scale(claim_set, reliability)
    )
    if spec.include_cardinality and register.is_enabled(ConstraintClass.CARDINALITY):
        bank = _cardinality_bank(universe, register, weight_scale)
        if bank is not None:
            banks.append(bank)

    banks.extend(extra_banks)

    log_prior = np.full((len(reg), width), 0.0)
    card = reg.cardinalities
    for i in range(len(reg)):
        c = int(card[i])
        if c < width:
            log_prior[i, c:] = NEG_INF
        if c == 0:
            log_prior[i, :] = NEG_INF

    return FactorGraph(reg, log_prior, banks)  # type: ignore[arg-type]


# -- channel and silence ---------------------------------------------------------------


def _channel_bank(
    universe: Universe, claim_set: ClaimSet, reliability: ReliabilityTable, width: int
) -> tuple[list[int], list[np.ndarray], list[str]]:
    reg = universe.registry
    ids: list[int] = []
    pots: list[np.ndarray] = []
    labels: list[str] = []

    for claim in claim_set:
        idx = reg.get(claim.ref)
        if idx is None:
            continue
        params = reliability[claim.source_id].clipped()
        c = int(reg.cardinalities[idx])
        row = np.full(width, NEG_INF)

        if c == 2 and isinstance(claim.value, bool):
            # log p(y | truth) for truth in {0, 1}
            row[:2] = params.binary_loglik[:, 1 if claim.value else 0]
        elif c > 2 and isinstance(claim.value, str):
            domain = universe.event_type_domain(claim.ref.subject)
            if claim.value not in domain:
                continue
            hit = domain.index(claim.value)
            # One-parameter spread: rho on the reported level, the rest spread uniformly.
            row[:c] = np.log((1.0 - params.rho) / max(c - 1, 1))
            row[hit] = np.log(params.rho)
        else:
            continue

        ids.append(int(idx))
        pots.append(row)
        labels.append(claim.source_id)

    return ids, pots, labels


def _silence_bank(
    universe: Universe,
    sources: Mapping[str, Source],
    reliability: ReliabilityTable,
    width: int,
) -> tuple[list[int], list[np.ndarray], list[str]]:
    """Negative evidence from what non-opportunistic sources did *not* say.

    Stage 1 section 4.5: for a detector-style source, silence carries the false-negative
    rate, and discarding it throws away that source's strongest signal.
    """
    reg = universe.registry
    ids: list[int] = []
    pots: list[np.ndarray] = []
    labels: list[str] = []

    for source in sources.values():
        profile = source.profile
        if not profile.coverage.silence_is_evidence:
            continue
        params = reliability[profile.source_id].clipped()
        neg = params.binary_loglik[:, 0]  # log p(y=0 | truth)
        claimed = {c.ref for c in source.claims()}
        for ref in source.scope():
            if ref in claimed:
                continue
            idx = reg.get(ref)
            if idx is None or int(reg.cardinalities[idx]) != 2:
                continue
            row = np.full(width, NEG_INF)
            row[:2] = neg
            ids.append(int(idx))
            pots.append(row)
            labels.append(profile.source_id)

    return ids, pots, labels


# -- structural factors ----------------------------------------------------------------


def _implication_table(hard: bool, weight: float) -> np.ndarray:
    """``T[link, endpoint_exists]``; the ``(1, 0)`` corner is forbidden."""
    penalty = NEG_INF if hard else -weight
    return np.array([[0.0, 0.0], [penalty, 0.0]])


def _referential_integrity_bank(
    universe: Universe, register: ConstraintRegister
) -> PairwiseBank | None:
    """A link implies both of its endpoints exist.

    This is the edge that turns "several sources claim links into event ``e``" into
    "``e`` probably happened", and conversely lets confidence that ``e`` did *not* happen
    suppress every link into it. Cheap, definitional, and one of the strongest structural
    signals available.
    """
    reg = universe.registry
    spec = register[ConstraintClass.REFERENTIAL_INTEGRITY]
    table = _implication_table(spec.is_hard, spec.weight)[None, :, :]

    a_ids: list[int] = []
    b_ids: list[int] = []
    for event_id, qualifier, object_id in universe.e2o_candidates:
        link = reg.get(AssertionRef.e2o(event_id, qualifier, object_id))
        if link is None:
            continue
        for endpoint in (
            reg.get(AssertionRef.event_exists(event_id)),
            reg.get(AssertionRef.object_exists(object_id)),
        ):
            if endpoint is not None:
                a_ids.append(int(link))
                b_ids.append(int(endpoint))

    for source_id, qualifier, target_id in universe.o2o_candidates:
        link = reg.get(AssertionRef.o2o(source_id, qualifier, target_id))
        if link is None:
            continue
        for endpoint in (
            reg.get(AssertionRef.object_exists(source_id)),
            reg.get(AssertionRef.object_exists(target_id)),
        ):
            if endpoint is not None:
                a_ids.append(int(link))
                b_ids.append(int(endpoint))

    if not a_ids:
        return None
    return PairwiseBank(
        np.array(a_ids),
        np.array(b_ids),
        table,
        np.zeros(len(a_ids), dtype=np.int64),
        name="referential_integrity",
    )


def _type_gate_bank(
    universe: Universe,
    width: int,
    register: ConstraintRegister,
    anchored: set[AssertionRef] | None = None,
) -> PairwiseBank | None:
    """A link is impossible under event types whose signature forbids it.

    Because ``T_e`` is latent, qualifier legality cannot be fully resolved by pruning: a
    candidate survives if *some* type in the support permits it. This factor carries the
    rest of the constraint, and it is bidirectional -- evidence for a link is evidence about
    the event's type, and vice versa. That coupling is a large part of how event-type belief
    forms at all when few sources speak about types directly.
    """
    reg = universe.registry
    schema = universe.schema
    spec = register[ConstraintClass.QUALIFIER_LEGALITY]
    penalty = NEG_INF if spec.is_hard else -spec.weight

    tables: list[np.ndarray] = []
    table_key: dict[tuple[bool, ...], int] = {}
    a_ids: list[int] = []
    b_ids: list[int] = []
    tids: list[int] = []

    for event_id, qualifier, object_id in universe.e2o_candidates:
        ref = AssertionRef.e2o(event_id, qualifier, object_id)
        if anchored is not None and ref not in anchored:
            continue
        type_var = reg.get(AssertionRef.event_type(event_id))
        link_var = reg.get(ref)
        if type_var is None or link_var is None:
            continue

        domain = universe.event_type_domain(event_id)
        object_type = universe.objects[object_id].object_type
        legal = tuple(schema.is_legal_e2o(t, qualifier, object_type) for t in domain)
        if all(legal):
            continue  # nothing to constrain

        # The legality pattern alone keys the table: its length is the domain size, and
        # states beyond it are already NEG_INF-padded to the shared width.
        tid = table_key.get(legal)
        if tid is None:
            tab = np.full((width, 2), NEG_INF)
            for t, ok in enumerate(legal):
                tab[t, 0] = 0.0
                tab[t, 1] = 0.0 if ok else penalty
            tid = len(tables)
            tables.append(tab)
            table_key[legal] = tid

        a_ids.append(int(type_var))
        b_ids.append(int(link_var))
        tids.append(tid)

    if not a_ids:
        return None
    return PairwiseBank(
        np.array(a_ids),
        np.array(b_ids),
        np.stack(tables),
        np.array(tids, dtype=np.int64),
        name="type_gate",
    )


def _cardinality_bank(
    universe: Universe, register: ConstraintRegister, weight_scale: float
) -> CardinalityBank | None:
    """Declared multiplicity per ``(event, qualifier)`` group.

    ``T_e`` is latent, so the applicable multiplicity is type-dependent. v0.1 resolves this
    conservatively: if every type in the event's support that admits this qualifier agrees
    on the multiplicity, use it; otherwise fall back to the permissive union. The
    conservative branch never penalises a configuration that some plausible type would
    allow, at the cost of dropping the constraint where types disagree.

    The exact treatment is a factor over ``(T_e, {R})`` -- higher-order, and deferred.
    """
    reg = universe.registry
    schema = universe.schema
    weight = register.weight(ConstraintClass.CARDINALITY) * weight_scale

    by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    for event_id, qualifier, object_id in universe.e2o_candidates:
        idx = reg.get(AssertionRef.e2o(event_id, qualifier, object_id))
        if idx is not None:
            by_group[(event_id, qualifier)].append(int(idx))

    groups: list[list[int]] = []
    lows: list[int] = []
    highs: list[int | None] = []

    for (event_id, qualifier), members in by_group.items():
        domain = universe.event_type_domain(event_id)
        specs = [
            q
            for q in schema.e2o_qualifiers
            if q.qualifier == qualifier and q.event_type in domain
        ]
        if not specs:
            continue
        mults = {(q.multiplicity.lo, q.multiplicity.hi) for q in specs}
        if len(mults) == 1:
            lo, hi = next(iter(mults))
        else:
            lo = min(m[0] for m in mults)
            hi = None if any(m[1] is None for m in mults) else max(m[1] for m in mults)
        if lo == 0 and hi is None:
            continue  # vacuous
        groups.append(members)
        lows.append(lo)
        highs.append(hi)

    if not groups:
        return None
    return CardinalityBank(groups, lows, highs, weight, name="cardinality")
