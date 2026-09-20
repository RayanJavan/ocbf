"""Qualified adapters around existing BP/EP numerical kernels."""

import uuid
from dataclasses import dataclass
from math import prod

import numpy as np
from scipy.special import logsumexp

from ocbf._values import fingerprint, freeze
from ocbf.belief.posterior import InferenceResult, JointTable
from ocbf.errors import BudgetExceeded, CapabilityError, NumericalFailure
from ocbf.inference.contracts import ExecutionPlan
from ocbf.inference.elimination import materialize_factors


@dataclass(frozen=True)
class MarginalPosterior:
    tables: tuple[JointTable, ...]
    cooperative = False

    def marginal(self, key):
        for table in self.tables:
            if table.scope == (key,):
                return table
        raise CapabilityError("absent marginal", key=key)


class _Layout:
    """Dense execution indices without inventing semantic AssertionRef identities."""

    is_frozen = True

    def __init__(self, variables):
        self.cardinalities = np.array([len(v.domain) for v in variables])
        self.max_cardinality = int(max(self.cardinalities, default=1))

    def __len__(self):
        return len(self.cardinalities)


class _TableBank:
    name = "canonical_tables"

    def __init__(self, tables, keys, width):
        self.tables, self.width = tables, width
        self.edges = np.array([keys[k] for t in tables for k in t.scope], dtype=np.int64)

    def edge_vars(self):
        return self.edges

    def n_factors(self):
        return len(self.tables)

    def factor_to_var(self, incoming):
        from ocbf.model.graph import NEG_INF

        result = np.full(incoming.shape, NEG_INF)
        offset = 0
        for table in self.tables:
            n = len(table.scope)
            for axis in range(n):
                message = table.values.copy()
                for other in range(n):
                    if other != axis:
                        shape = [1] * n
                        shape[other] = table.values.shape[other]
                        message += incoming[offset + other, : shape[other]].reshape(shape)
                axes = tuple(i for i in range(n) if i != axis)
                result[offset + axis, : table.values.shape[axis]] = logsumexp(message, axis=axes)
            offset += n
        return result


@dataclass(frozen=True)
class BPEngine:
    name: str = "bp"
    version: str = "1"
    cooperative = False

    def assess(self, model, requirements, policy, *, store=None, control=None):
        if control is not None:
            raise CapabilityError("BP does not honor cooperative controls", key=self.name)
        if model.continuous:
            raise CapabilityError("BP adapter admits finite targets only")
        if not policy.allow_approximate:
            raise CapabilityError("BP requires allow_approximate=True")
        if not requirements.satisfied_by(("marginal",)):
            raise CapabilityError("BP marginals do not supply joint histories")
        for scope in requirements.scopes:
            if len(scope) > 1 or not set(scope) <= model.domains.keys():
                raise CapabilityError("BP supplies admitted single-variable marginals only")
        largest = max(
            (prod(len(model.domains[k]) for k in f.scope) for f in model.factors), default=1
        )
        if largest > policy.max_table_states:
            raise BudgetExceeded("BP table exceeds max_table_states")
        return ExecutionPlan(self.name, (), largest, 0, ("marginal",))

    def solve(self, model, plan, policy, rng, *, store=None, control=None, warm_start=None):
        if control is not None or warm_start is not None:
            raise CapabilityError(
                "BP does not honor cooperative controls or warm starts", key=self.name
            )
        from ocbf.inference.loopy_bp import run_bp
        from ocbf.model.graph import FactorGraph

        # Bounded compatibility adapter: until support-aware BP is supplied, reject exact
        # zeros rather than translating them into a finite penalty convention.
        tables = materialize_factors(model, policy, check_elimination=False)
        if any(not np.isfinite(t.values).all() for t in tables):
            raise CapabilityError("this BP adapter admits strictly positive finite factors only")
        layout = _Layout(model.variables)
        keys = {v.key: i for i, v in enumerate(model.variables)}
        bank = _TableBank(tables, keys, layout.max_cardinality)
        graph = FactorGraph(layout, FactorGraph.padded_log_prior(layout, uniform=False), [bank])
        bp = run_bp(graph)
        posterior = MarginalPosterior(
            tuple(
                JointTable((v.key,), (v.domain,), bp.beliefs[i, : len(v.domain)])
                for i, v in enumerate(model.variables)
            )
        )
        return InferenceResult(
            model.model_id,
            fingerprint("plan", (model.model_id, plan, policy)),
            "run:" + str(uuid.uuid4()),
            posterior,
            ("marginal",),
            "approximate-BP-marginals",
            bp.diagnostics(),
            {**model.manifest, "approximation": "loopy BP; convergence is not accuracy"},
        )


@dataclass(frozen=True)
class GaussianMarginals:
    keys: tuple[str, ...]
    mean: np.ndarray
    variance: np.ndarray
    unit: str = "latent standardized coordinates"

    def __post_init__(self):
        object.__setattr__(self, "mean", freeze(self.mean))
        object.__setattr__(self, "variance", freeze(self.variance))


def run_ep_adapter(graph, *, model_id, evidence_id, parameter_id, context_id, config=None):
    """Run a caller-grounded Gaussian graph; identity/grounding remain caller-owned.

    This caller-grounded adapter does not compile hybrid targets or fit any parameters. Units remain
    latent; callers need their supplied copula transforms to report observed-unit moments.
    """
    from copy import deepcopy

    from ocbf.inference.gabp_ep import EPConfig, run_ep
    from ocbf.model.gaussian import MAX_PRECISION, MIN_PRECISION

    cfg = config or EPConfig()
    owned = deepcopy(graph)
    if (
        not np.isfinite(owned.prior).all()
        or (owned.prior[:, 0] <= MIN_PRECISION).any()
        or (owned.prior[:, 0] >= MAX_PRECISION).any()
    ):
        raise CapabilityError("EP adapter requires proper priors inside the kernel precision range")
    owned.banks = [_CheckedGaussianBank(bank) for bank in owned.banks]
    ep = run_ep(owned, cfg)
    natural = owned.prior.copy()
    np.add.at(natural, owned.edge_var, ep.messages)
    if (
        not np.isfinite(natural).all()
        or (natural[:, 0] <= MIN_PRECISION).any()
        or (natural[:, 0] >= MAX_PRECISION).any()
    ):
        raise NumericalFailure("Gaussian moments would require precision clipping")
    if not np.isfinite(ep.mean).all() or not np.isfinite(ep.var).all() or (ep.var <= 0).any():
        raise NumericalFailure("invalid Gaussian marginal moments")
    posterior = GaussianMarginals(
        tuple(str(owned.registry.ref(int(i))) for i in owned.var_ids), ep.mean, ep.var
    )
    return InferenceResult(
        model_id,
        fingerprint("plan", (model_id, cfg)),
        "run:" + str(uuid.uuid4()),
        posterior,
        ("gaussian_marginal",),
        "approximate-EP-marginals",
        ep.diagnostics(),
        {
            "evidence_id": evidence_id,
            "parameter_id": parameter_id,
            "context_id": context_id,
            "grounding": "caller-supplied Gaussian graph",
            "joint_history": "unsupported",
            "parameter_fitting": "none",
        },
    )


class _CheckedGaussianBank:
    """Surface numerical failures the low-level engine otherwise skips as site updates."""

    def __init__(self, bank):
        self.bank = bank

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def factor_to_var(self, cavity):
        messages = self.bank.factor_to_var(cavity)
        if not np.isfinite(messages).all():
            raise NumericalFailure(
                "Gaussian factor returned an invalid site update", key=self.bank.name
            )
        return messages
