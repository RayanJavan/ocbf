"""Blocked finite Gibbs and corrected joint MH with exact conditional elimination."""

import math
import platform
import time
import uuid
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType

import numpy as np
import scipy
from scipy.special import logsumexp

from ocbf._values import fingerprint, freeze
from ocbf.belief.posterior import InferenceResult, JointDrawSet, draw_dtype
from ocbf.belief.samples import SamplePosterior
from ocbf.errors import (
    BudgetExceeded,
    CapabilityError,
    ExecutionStopped,
    IncompatibleModel,
    NumericalFailure,
    ValidationError,
)
from ocbf.model.dependencies import reusable_model
from ocbf.model.reductions import reduction_codec
from ocbf.runtime.cache import cached
from ocbf.runtime.control import checkpoint

from .block_conditionals import draw_finite_block
from .conditional import assess_conditioning, condition_target
from .contracts import ExecutionPlan, Proposal
from .warm_start import initializations


def metropolis_log_acceptance(old, new, forward, reverse):
    terms = (old, new, forward, reverse)
    if (
        any(math.isnan(x) or x == math.inf for x in terms)
        or not math.isfinite(old)
        or not math.isfinite(forward)
    ):
        raise NumericalFailure("invalid target/proposal law in MH transition")
    return min(0.0, new - old + reverse - forward)


def _blocks(model, codec, sampled):
    groups = [{k} for k in sampled]
    for factor in model.factors:
        if factor.role != "support":
            continue
        scope = {codec.encoded_key(k) for k in factor.scope} & set(sampled)
        joined = set(scope)
        rest = []
        for group in groups:
            if group & joined:
                joined.update(group)
            else:
                rest.append(group)
        groups = [*rest, joined] if joined else rest
    # Occurrences and their time coordinates are updated together when both are sampled.
    for variable in model.continuous.values():
        scope = {variable.key, *(codec.encoded_key(k) for k, _ in variable.active_when)} & set(
            sampled
        )
        touching = [g for g in groups if g & scope]
        if touching:
            groups = [g for g in groups if not g & scope] + [set.union(*touching)]
    return tuple(sorted(tuple(sorted(g)) for g in groups if g))


@dataclass(frozen=True)
class BlockedEngine:
    name: str = "blocked"
    version: str = "1"
    cooperative = True
    proposals: object = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "proposals", MappingProxyType(dict(self.proposals)))

    def assess(self, model, requirements, policy, *, store=None, control=None):
        checkpoint(control, "sampling.assess")
        config = policy.sampling
        if config is None:
            raise CapabilityError("blocked inference requires explicit SamplingConfig")
        if not requirements.satisfied_by(("joint_draws",)):
            raise CapabilityError(
                "blocked inference supplies joint draws, not exact tables or a normalizer"
            )
        keys = set(model.domains) | set(model.continuous)
        if any(not set(s) <= keys for s in requirements.scopes):
            raise CapabilityError("query names absent variables")
        codec = cached(
            store,
            "sampling.codec",
            (model.domains, tuple(f for f in model.factors if f.role == "support")),
            lambda: reduction_codec(model),
        )
        sampled = set()
        for factor in model.factors:
            if not hasattr(model._kernels[factor.family], "gaussian_terms"):
                sampled.update(set(factor.scope) & model.continuous.keys())
            if (
                factor.role == "support"
                and factor.family
                not in (
                    "table",
                    "implication",
                    "type_presence",
                    "type_gate",
                    "cardinality",
                    "precedence",
                    "count",
                    "temporal_order",
                )
                and not self.proposals
            ):
                raise CapabilityError(
                    "custom hard support requires registered proposal", key=factor.key
                )
        if config.blocks:
            sampled.update(k for block in config.blocks for k in block)
        sampled.update(k for block in self.proposals for k in block)
        if not sampled <= set(codec.domains) | set(model.continuous):
            raise CapabilityError("sampling block names an absent or reduced variable")
        if not sampled and codec.domains:
            sampled.add(min(codec.domains))
        while True:
            fixed_keys = codec.expand(
                {k: codec.domains[k][0] if k in codec.domains else 0.0 for k in sampled}
            ).keys()
            try:
                order, inputs, clique = assess_conditioning(
                    model, fixed_keys, policy, store=store, control=control
                )
                break
            except BudgetExceeded:
                candidates = set(codec.domains) - sampled
                if not candidates:
                    # Excess Gaussian dimensions remain explicit sampled coordinates.
                    candidates = set(model.continuous) - sampled
                if not candidates:
                    raise
                degree = lambda k: sum(
                    codec.encoded_key(v) == k for f in model.factors for v in f.scope
                )
                sampled.add(min(candidates, key=lambda k: (-degree(k), k)))
        if not sampled <= set(codec.domains) | set(model.continuous):
            raise CapabilityError("sampling block names an absent or reduced variable")
        blocks = config.blocks or cached(
            store,
            "sampling.blocks",
            (
                codec,
                tuple(sorted(sampled)),
                tuple(f for f in model.factors if f.role == "support"),
                tuple(model.continuous.values()),
            ),
            lambda: _blocks(model, codec, sampled),
        )
        if {k for b in blocks for k in b} != sampled or any(
            not b or len(set(b)) != len(b) for b in blocks
        ):
            raise ValidationError("sampling blocks must cover the planned sampled variables")
        if not set(self.proposals) <= {*blocks, tuple(sorted(sampled))}:
            raise CapabilityError("custom proposal keys must match a planned block or full refresh")
        bytes_per_draw = sum(draw_dtype(d).itemsize for d in model.domains.values()) + 8 * len(
            model.continuous
        )
        if config.chains * config.draws * bytes_per_draw > policy.max_draw_bytes:
            raise BudgetExceeded("retained joint draws exceed allocation budget")
        return ExecutionPlan(
            self.name,
            order,
            inputs,
            clique,
            ("joint_draws",),
            {
                "codec": codec,
                "sampled": tuple(sorted(sampled)),
                "blocks": blocks,
                "eliminated": tuple(sorted(keys - set(fixed_keys))),
                "conditional_target": "exact elimination and admitted Gaussian integration",
                "proposals": {str(k): f"{p.name}@{p.version}" for k, p in self.proposals.items()},
            },
        )

    def solve(self, model, plan, policy, rng, *, store=None, control=None, warm_start=None):
        if rng is None:
            raise CapabilityError("blocked inference requires an explicit random stream")
        config, codec = policy.sampling, plan.details["codec"]
        if store is not None and not reusable_model(model):
            store = None
        warm_states, warm_details = initializations(model, plan, config, warm_start)
        sampled, blocks = plan.details["sampled"], plan.details["blocks"]
        seeds = rng.integers(0, 2**63 - 1, size=config.chains)
        started = time.monotonic()
        accepted = {str(b): [0, 0] for b in (*blocks, tuple(sampled))}
        chains, initial, transitions = [], [], []
        status = "complete"
        block_fallbacks = {}
        dtypes = {
            v.key: draw_dtype(model.domains[v.key]) if v.key in model.domains else np.dtype(float)
            for v in model.variables
        }
        chain_lengths = []
        stop_error = None
        checkpoint(
            control,
            "sampling.allocate",
            allocation_bytes=3
            * config.chains
            * config.draws
            * sum(d.itemsize for d in dtypes.values()),
        )

        def evaluate(state):
            try:
                posterior = condition_target(
                    model, codec.expand(state), policy, store=store, control=control
                )
                return posterior.log_normalizer, posterior
            except IncompatibleModel:
                return -math.inf, None

        def independent(state, block, generator):
            proposed = dict(state)
            forward = reverse = 0.0
            for key in block:
                if key in codec.domains:
                    domain = codec.domains[key]
                    proposed[key] = domain[generator.integers(len(domain))]
                    forward -= math.log(len(domain))
                    reverse -= math.log(len(domain))
                else:
                    v = model.continuous[key]
                    proposed[key] = generator.normal(v.prior_mean, v.prior_sd)
                    logp = lambda x, v=v: (
                        -0.5 * ((x - v.prior_mean) / v.prior_sd) ** 2
                        - math.log(v.prior_sd)
                        - 0.5 * math.log(2 * math.pi)
                    )
                    forward += logp(proposed[key])
                    reverse += logp(state[key])
            return Proposal(proposed, forward, reverse)

        def validate(state):
            if set(state) != set(sampled):
                raise ValidationError("proposal/initial state has incorrect scope")
            for k, value in state.items():
                if k in codec.domains:
                    if value not in codec.domains[k]:
                        raise ValidationError("proposal outside finite domain", key=k)
                elif not math.isfinite(value):
                    raise ValidationError("nonfinite continuous proposal", key=k)

        for chain_index, seed in enumerate(seeds):
            retained, retained_count, changed = None, 0, {k: 0 for k in sampled}
            try:
                checkpoint(control, "sampling.chain", chain=chain_index)
                generator = np.random.default_rng(int(seed))
                base = {
                    k: codec.domains[k][0] if k in codec.domains else model.continuous[k].prior_mean
                    for k in sampled
                }
                if chain_index in warm_states:
                    state = dict(warm_states[chain_index])
                    validate(state)
                    log_mass, posterior = evaluate(state)
                elif config.initial_states:
                    if len(config.initial_states) != config.chains:
                        raise ValidationError("one initial state is required per chain")
                    state = dict(config.initial_states[chain_index])
                    validate(state)
                    log_mass, posterior = evaluate(state)
                else:
                    posterior = None
                    for _ in range(config.max_initialization):
                        checkpoint(control, "sampling.initialize", chain=chain_index)
                        state = dict(independent(base, sampled, generator).state)
                        log_mass, posterior = evaluate(state)
                        if posterior is not None:
                            break
                if posterior is None:
                    raise CapabilityError(
                        "bounded initialization found no admissible state; supply valid initial states"
                    )
                initial.append(state.copy())
                retained = {k: np.empty(config.draws, dtype=dtype) for k, dtype in dtypes.items()}
                retained_count, changed = 0, {k: 0 for k in sampled}
                for sweep in range(config.warmup + config.draws):
                    checkpoint(
                        control,
                        "sampling.sweep",
                        chain=chain_index,
                        completed=sweep,
                        total=config.warmup + config.draws,
                    )
                    if (
                        policy.max_seconds is not None
                        and time.monotonic() - started > policy.max_seconds
                    ):
                        status = "budget_exhausted"
                        break
                    for block_index, block in enumerate((*blocks, tuple(sampled))):
                        checkpoint(control, "sampling.block", chain=chain_index, block=block_index)
                        refresh = block_index == len(blocks)
                        # The final transition is a global independence refresh, including mode jumps.
                        if refresh and generator.random() >= config.refresh_probability:
                            continue
                        old_state = state.copy()
                        custom = self.proposals.get(tuple(block))
                        size = (
                            math.prod(len(codec.domains[k]) for k in block)
                            if all(k in codec.domains for k in block)
                            else math.inf
                        )
                        gibbs = custom is None and not refresh and size <= config.max_block_states
                        if gibbs and not model.continuous:
                            try:
                                state = draw_finite_block(
                                    model,
                                    codec,
                                    state,
                                    block,
                                    policy,
                                    generator,
                                    store=store,
                                    control=control,
                                )
                            except CapabilityError as exc:
                                block_fallbacks[str(block)] = str(exc)
                                gibbs = False
                            else:
                                log_mass, posterior = evaluate(state)
                                accept = True
                        elif gibbs:
                            candidates, masses = [], []
                            for values in product(*(codec.domains[k] for k in block)):
                                candidate = {**state, **dict(zip(block, values, strict=True))}
                                mass, _ = evaluate(candidate)
                                candidates.append(candidate)
                                masses.append(mass)
                            chosen = generator.choice(
                                len(masses), p=np.exp(np.asarray(masses) - logsumexp(masses))
                            )
                            state = candidates[chosen]
                            log_mass, posterior = evaluate(state)
                            accept = True
                        if not gibbs:
                            if custom is not None:
                                proposal = custom.propose(freeze(state), generator)
                            elif refresh or size != math.inf:
                                proposal = independent(state, block, generator)
                            else:
                                proposed = dict(state)
                                for key in block:
                                    if key in codec.domains:
                                        proposed[key] = codec.domains[key][
                                            generator.integers(len(codec.domains[key]))
                                        ]
                                    else:
                                        proposed[key] += generator.normal(
                                            0,
                                            model.continuous[key].prior_sd * config.proposal_scale,
                                        )
                                proposal = Proposal(
                                    proposed, 0.0, 0.0
                                )  # symmetric joint random walk
                            validate(proposal.state)
                            if any(
                                proposal.state[k] != state[k] for k in set(sampled) - set(block)
                            ):
                                raise ValidationError(
                                    "proposal changed variables outside its declared block"
                                )
                            new_mass, new_posterior = evaluate(proposal.state)
                            accept = math.log(
                                max(generator.random(), np.finfo(float).tiny)
                            ) < metropolis_log_acceptance(
                                log_mass, new_mass, proposal.log_forward, proposal.log_reverse
                            )
                            if accept:
                                state, log_mass, posterior = (
                                    dict(proposal.state),
                                    new_mass,
                                    new_posterior,
                                )
                        accepted[str(block)][1] += 1
                        accepted[str(block)][0] += int(accept)
                        if sweep >= config.warmup:
                            for key in sampled:
                                changed[key] += int(state[key] != old_state[key])
                    if sweep >= config.warmup:
                        history = posterior.draw_assignment(generator)
                        for key, values in retained.items():
                            values[retained_count] = history[key]
                        retained_count += 1
            except ExecutionStopped as exc:
                status, stop_error = exc.status, exc
            if (retained is None or retained_count == 0) and status != "complete":
                break
            chains.append(retained)
            chain_lengths.append(retained_count)
            transitions.append(changed)
            if status != "complete":
                break
        retained_count = min(chain_lengths, default=0)
        if retained_count == 0:
            if stop_error is not None:
                raise stop_error
            raise BudgetExceeded("sampling exhausted its budget before retaining usable draws")
        values = {k: np.stack([chain[k][:retained_count] for chain in chains]) for k in dtypes}
        draws = JointDrawSet(
            values,
            "mcmc",
            {
                "chain_seeds": tuple(int(s) for s in seeds[: len(chains)]),
                "warmup": config.warmup,
                "retained_per_chain": retained_count,
                "status": status,
                "initial_states": tuple(initial),
                "transitions": tuple(transitions),
                "proposal_settings": config,
                "adaptation": "none; fixed transitions throughout",
            },
        )
        return InferenceResult(
            model.model_id,
            fingerprint("plan", (model.model_id, plan, policy, self.version)),
            "run:" + str(uuid.uuid4()),
            SamplePosterior(draws),
            ("joint_draws",),
            "mcmc-on-declared-target",
            {
                "status": status,
                "acceptance": accepted,
                "mode_transitions": tuple(transitions),
                "block_conditional_fallbacks": block_fallbacks,
                "elapsed_seconds": time.monotonic() - started,
                "warm_start": warm_details,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "python_version": platform.python_version(),
            },
            {
                **model.manifest,
                "variable_domains": model.domains,
                "decoding": model.decoding,
                "execution_plan": plan,
            },
        )
