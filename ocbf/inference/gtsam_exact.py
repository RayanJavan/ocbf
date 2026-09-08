"""Exact discrete inference through GTSAM -- the verification oracle.

Design doc section 6.3 puts an exact engine at tier 1 and states its job plainly: a system
whose core is an approximation needs ground truth to measure the approximation against.
Until now the only oracle was brute-force enumeration in ``tests/test_exactness.py``, which
is exact but costs ``prod(cardinalities)`` and so runs out at a handful of variables --
small enough that the loopy structure the engine actually meets is out of reach.

GTSAM's `DiscreteFactorGraph` eliminates instead of enumerating, so its cost is exponential
in the *treewidth* rather than in the variable count. A dozen cardinality groups coupled
through shared endpoints is hopeless to enumerate and unremarkable to eliminate, and that
is precisely the shape where loopy BP stops being exact and starts needing to be checked.

Scope, stated so it is not overread:

* **Discrete backbone only.** The tier-4 role design doc section 6.3 reserves for GTSAM --
  hybrid discrete-continuous MAP -- would need the continuous block expressed as a GTSAM
  hybrid factor graph, which [`ocbf.inference.gabp_ep`][ocbf.inference.gabp_ep] has no reason
  to produce. This is the tier-1 role instead; see design doc section 11.13.
* **CPU.** GTSAM's CUDA acceleration is in its nonlinear least-squares solvers. Discrete
  elimination does not touch the GPU, and [`ocbf.backends.gtsam_backend`][ocbf.backends.gtsam_backend] reports the two
  facts separately for exactly this reason.
* **Optional.** Nothing in the pipeline calls this. It is imported lazily, and a machine
  without GTSAM loses a check, not a capability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # gtsam is optional and is imported lazily; see ocbf.backends
    from gtsam import DiscreteFactorGraph

from ocbf.assertions import AssertionRef
from ocbf.backends import import_gtsam
from ocbf.inference.loopy_bp import BPResult
from ocbf.model.graph import NEG_INF, FactorGraph

__all__ = [
    "EliminationCost",
    "ExactConfig",
    "ExactResult",
    "OracleComparison",
    "compare_to_exact",
    "elimination_cost",
    "exact_marginals",
    "to_discrete_factor_graph",
]


@dataclass(slots=True)
class ExactConfig:
    """Limits for a solver that has no iteration cap to stop it.

    Elimination either finishes or exhausts memory, so the limits have to be predictive
    rather than reactive. `max_clique_states` is the one that matters and the one to move:
    it bounds the largest intermediate factor elimination will build, which is what actually
    determines whether a graph is solvable, and it is checked by
    [`elimination_cost`][ocbf.inference.gtsam_exact.elimination_cost] before any work is
    done.

    Variable count is *not* that quantity, and the gap is not academic. Under
    exactly-one constraints along every row and column of a square of binary variables,
    measured on this backend:

    | variables | predicted clique | outcome        |
    |-----------|------------------|----------------|
    | 16 (4x4)  | 1e3 states       | 0.1 s          |
    | 25 (5x5)  | 6.6e4 states     | 6 s            |
    | 36 (6x6)  | 1.7e7 states     | 11 GB, killed  |

    Nine more variables is four orders of magnitude of clique. `max_variables` is kept only
    as a cheap gate that fires before the graph is even translated; it is not a safety
    limit and should not be read as one.
    """

    max_clique_states: int = 1 << 18
    """Ceiling on the largest intermediate factor, in table entries.

    Four times the 5x5 case above, which already takes seconds -- deliberately sized for an
    oracle rather than an engine. Raise it knowingly.
    """

    max_variables: int = 1_000
    max_table_states: int = 1 << 22
    compute_map: bool = True
    """Also run max-product for the MPE assignment. Cheap next to the marginals."""


@dataclass(slots=True)
class ExactResult:
    """Exact marginals, and what it took to get them.

    Deliberately shaped like [`BPResult`][ocbf.inference.loopy_bp.BPResult] -- same ``(n_vars, max_card)`` log-space
    layout, same ``NEG_INF`` padding, same `diagnostics` method -- so that comparing the two
    is subtraction rather than translation.
    """

    log_beliefs: np.ndarray
    map_assignment: np.ndarray | None
    cost: EliminationCost
    seconds: float
    version: str | None

    @property
    def beliefs(self) -> np.ndarray:
        return np.exp(self.log_beliefs)

    def diagnostics(self) -> dict[str, object]:
        """Backend identity and cost. There is no convergence to report -- it is exact."""
        return {
            "exact_backend": f"gtsam {self.version}",
            "exact_factors": self.cost.n_factors,
            "exact_largest_clique": self.cost.largest_clique_states,
            "exact_seconds": round(self.seconds, 4),
        }


@dataclass(frozen=True, slots=True)
class EliminationCost:
    """What elimination will cost on a graph, computed before paying it.

    `largest_clique_states` simulates the fill-in GTSAM's own COLAMD ordering will produce,
    so this is a prediction of what GTSAM will actually do rather than a generic treewidth
    bound. It is cheap -- one ordering and a pass over the factor scopes -- which is what
    makes it usable as a guard rather than as a post-mortem.
    """

    largest_clique_states: int
    largest_input_table: int
    n_factors: int

    def summary(self) -> str:
        return (
            f"{self.n_factors} factors, largest input table {self.largest_input_table}, "
            f"largest clique {self.largest_clique_states} states"
        )


@dataclass(frozen=True, slots=True)
class OracleComparison:
    """How far the approximation sits from exact, on one graph.

    Both an ``L-infinity`` and a mean error, because they answer different questions: the
    maximum is what a decidability verdict can turn on, the mean is what an aggregate metric
    will feel. `worst_ref` names the assertion behind the maximum, so a regression points at
    a variable rather than at a number.
    """

    max_abs_error: float
    mean_abs_error: float
    max_kl: float
    worst_variable: int
    worst_ref: AssertionRef | None
    n_compared: int

    def summary(self) -> str:
        return (
            f"max |dp| = {self.max_abs_error:.2e} at variable {self.worst_variable} "
            f"({self.worst_ref}), mean {self.mean_abs_error:.2e}, "
            f"max KL {self.max_kl:.2e} over {self.n_compared} variables"
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "oracle_max_abs_error": round(self.max_abs_error, 10),
            "oracle_mean_abs_error": round(self.mean_abs_error, 10),
            "oracle_max_kl": round(self.max_kl, 10),
            "oracle_worst_variable": self.worst_variable,
        }


def to_discrete_factor_graph(
    graph: FactorGraph, *, config: ExactConfig | None = None, include_prior: bool = True
) -> DiscreteFactorGraph:
    """Translate a [`FactorGraph`][ocbf.model.graph.FactorGraph] into GTSAM's `DiscreteFactorGraph`.

    Two conventions have to line up, and both are checked by the exactness tests rather than
    assumed:

    * **Table order.** GTSAM lays a factor's table out row-major over its keys *in the order
      they were given*, last key varying fastest -- which is C-order over the axes of
      `dense_factors`' tables. Pushing the keys in `var_ids` order is therefore the whole of
      the translation, with no sorting or transposing.
    * **Scale.** GTSAM discrete factors hold non-negative potentials, not log-potentials, so
      each table is exponentiated after subtracting its own maximum. Per-factor shifts
      cancel in the normalised marginals, and the shift is what keeps a
      [`NEG_INF`][ocbf.model.graph.NEG_INF] entry landing on a clean zero instead of an
      underflow warning.

    Variables with cardinality 0 -- the continuous ones -- are skipped. If a bank connects
    to one, that is a modelling error rather than something to paper over, and it raises.

    Args:
        graph: the graph to translate.
        config: limits to enforce while translating; defaults to [`ExactConfig`][ocbf.inference.gtsam_exact.ExactConfig].
        include_prior: add each variable's row of `log_prior` as a unary factor. Keep it on
            unless the prior is being supplied some other way -- it is also what guarantees
            every variable appears in the graph at all.

    Returns:
        The translated `DiscreteFactorGraph`.
    """
    cfg = config or ExactConfig()
    gtsam = import_gtsam()

    card = graph.registry.cardinalities
    n_discrete = int(np.count_nonzero(card))
    if n_discrete > cfg.max_variables:
        raise ValueError(
            f"{n_discrete} discrete variables exceeds max_variables={cfg.max_variables}; "
            "exact elimination is an oracle for small graphs, not an engine"
        )

    dfg = gtsam.DiscreteFactorGraph()

    if include_prior:
        for var in range(graph.n_vars):
            c = int(card[var])
            if c:
                dfg.add([(var, c)], _potentials(graph.log_prior[var, :c]))

    for var_ids, log_table in graph.dense_factors():
        keys = []
        for var in var_ids:
            c = int(card[int(var)])
            if c == 0:
                raise ValueError(
                    f"variable {int(var)} is continuous (cardinality 0) but a bank connects "
                    "a discrete factor to it"
                )
            keys.append((int(var), c))
        if log_table.size > cfg.max_table_states:
            raise ValueError(
                f"a factor over {len(keys)} variables needs {log_table.size} table entries, "
                f"over max_table_states={cfg.max_table_states}"
            )
        dfg.add(keys, _potentials(log_table))

    return dfg


def elimination_cost(graph: FactorGraph, *, config: ExactConfig | None = None) -> EliminationCost:
    """Predict what [`exact_marginals`][ocbf.inference.gtsam_exact.exact_marginals] will cost on this graph.

    Worth calling on its own before pointing the oracle at an unfamiliar graph: the answer
    is the difference between a check that returns in a second and one that exhausts memory,
    and nothing about the graph's size tells you which.
    """
    gtsam = import_gtsam()
    dfg = to_discrete_factor_graph(graph, config=config)
    return _elimination_cost(dfg, graph.registry.cardinalities, gtsam)


def _elimination_cost(dfg, cardinalities: np.ndarray, gtsam) -> EliminationCost:
    """Simulate fill-in along GTSAM's own ordering, and report the largest clique.

    Standard induced-width computation: eliminate in order, connect each eliminated
    variable's surviving neighbours into a clique, and record the biggest clique met. The
    ordering comes from `Ordering.ColamdDiscreteFactorGraph` because that is the one GTSAM
    will use -- asking a different heuristic would give a number about a computation nobody
    is going to run.
    """
    card = {var: int(c) for var, c in enumerate(cardinalities) if int(c)}
    adjacency: dict[int, set[int]] = {var: set() for var in card}
    largest_input = 1

    for i in range(dfg.size()):
        scope = list(dfg.at(i).keys())
        states = 1
        for key in scope:
            states *= card[key]
            adjacency[key].update(other for other in scope if other != key)
        largest_input = max(largest_input, states)

    ordering = gtsam.Ordering.ColamdDiscreteFactorGraph(dfg)
    alive = set(adjacency)
    largest_clique = 1
    for position in range(ordering.size()):
        var = ordering.at(position)
        if var not in alive:
            continue
        neighbours = (adjacency[var] & alive) - {var}
        states = card[var]
        for other in neighbours:
            states *= card[other]
        largest_clique = max(largest_clique, states)
        for other in neighbours:
            adjacency[other] |= neighbours - {other}
        alive.discard(var)

    return EliminationCost(
        largest_clique_states=largest_clique,
        largest_input_table=largest_input,
        n_factors=dfg.size(),
    )


def _potentials(log_table: np.ndarray) -> list[float]:
    """Log-potentials to the flat, non-negative, max-shifted list GTSAM wants."""
    shifted = log_table - log_table.max()
    return np.exp(shifted).ravel(order="C").tolist()


def exact_marginals(graph: FactorGraph, config: ExactConfig | None = None) -> ExactResult:
    """Exact marginals for the discrete backbone, by variable elimination.

    Raises:
        BackendUnavailable: if GTSAM is not installed or will not load.
        ValueError: if the graph trips one of [`ExactConfig`][ocbf.inference.gtsam_exact.ExactConfig]'s limits.
    """
    cfg = config or ExactConfig()
    gtsam = import_gtsam()

    started = time.perf_counter()
    dfg = to_discrete_factor_graph(graph, config=cfg)

    cost = _elimination_cost(dfg, graph.registry.cardinalities, gtsam)
    if cost.largest_clique_states > cfg.max_clique_states:
        raise ValueError(
            f"elimination would build a clique of {cost.largest_clique_states} states, over "
            f"max_clique_states={cfg.max_clique_states} ({cost.summary()}). This graph's "
            "treewidth, not its size, is what makes it unsolvable exactly."
        )

    card = graph.registry.cardinalities
    log_beliefs = np.full((graph.n_vars, graph.max_card), NEG_INF, dtype=np.float64)
    marginals = gtsam.DiscreteMarginals(dfg)
    for var in range(graph.n_vars):
        c = int(card[var])
        if c == 0:
            continue
        probs = np.asarray(marginals.marginalProbabilities((var, c)), dtype=np.float64).ravel()
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(c, 1.0 / c)
        # Back to log space with the graph's own floor, so an impossible state reads as
        # NEG_INF here exactly as it does everywhere else rather than as -inf.
        with np.errstate(divide="ignore"):
            log_beliefs[var, :c] = np.maximum(np.log(probs), NEG_INF)

    assignment: np.ndarray | None = None
    if cfg.compute_map:
        mpe = dfg.optimize()
        assignment = np.array(
            [int(mpe[var]) if int(card[var]) else -1 for var in range(graph.n_vars)],
            dtype=np.int64,
        )

    return ExactResult(
        log_beliefs=log_beliefs,
        map_assignment=assignment,
        cost=cost,
        seconds=time.perf_counter() - started,
        version=_version(),
    )


def compare_to_exact(
    graph: FactorGraph,
    approximate: BPResult | np.ndarray,
    config: ExactConfig | None = None,
) -> OracleComparison:
    """Measure an approximate posterior against the exact one on the same graph.

    This is what the oracle is *for*. BP is exact on trees and an approximation on
    everything else, and the size of that approximation is a number the design record asks
    for rather than a thing to assume is small.

    Args:
        graph: the graph both posteriors are over.
        approximate: a [`BPResult`][ocbf.inference.loopy_bp.BPResult], or any ``(n_vars, max_card)`` array of
            probabilities in the same layout.
        config: passed through to [`exact_marginals`][ocbf.inference.gtsam_exact.exact_marginals].
    """
    beliefs = approximate.beliefs if isinstance(approximate, BPResult) else np.asarray(approximate)
    exact = exact_marginals(graph, config).beliefs

    if beliefs.shape != exact.shape:
        raise ValueError(f"expected beliefs of shape {exact.shape}, got {beliefs.shape}")

    card = graph.registry.cardinalities
    errors: list[float] = []
    per_variable: list[tuple[float, int]] = []
    max_kl = 0.0
    compared = 0

    for var in range(graph.n_vars):
        c = int(card[var])
        if c == 0:
            continue
        compared += 1
        p, q = exact[var, :c], beliefs[var, :c]
        diff = np.abs(p - q)
        errors.extend(diff.tolist())
        per_variable.append((float(diff.max()), var))
        # KL(exact || approximate): the divergence a consumer of the approximation pays.
        # Only states the exact posterior gives mass to contribute, which is also what
        # keeps a hard-zero state out of the logarithm.
        support = p > 0.0
        if support.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = p[support] * np.log(p[support] / np.maximum(q[support], 1e-300))
            max_kl = max(max_kl, float(np.sum(terms)))

    worst_error, worst_variable = max(per_variable, default=(0.0, -1))
    worst_ref = graph.registry.ref(worst_variable) if worst_variable >= 0 else None

    return OracleComparison(
        max_abs_error=worst_error,
        mean_abs_error=float(np.mean(errors)) if errors else 0.0,
        max_kl=max_kl,
        worst_variable=worst_variable,
        worst_ref=worst_ref,
        n_compared=compared,
    )


def _version() -> str | None:
    from ocbf.backends import gtsam_backend

    return gtsam_backend().version
