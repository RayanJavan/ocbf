"""Query composition and capability negotiation; numerical aggregation is separate."""

from itertools import product
from math import prod

from ocbf.belief.estimates import QueryResults
from ocbf.belief.posterior import QueryRequirements
from ocbf.errors import BudgetExceeded, CapabilityError, ExecutionStopped
from ocbf.runtime.contracts import ExecutionAssessment
from ocbf.runtime.control import checkpoint

from .contracts import QueryEvaluator

__all__ = ["QueryEvaluator", "builtin_evaluators", "evaluate", "requirements_for"]


def builtin_evaluators():
    from .aggregation import ExecutionEvaluator

    return {
        name: ExecutionEvaluator()
        for name in (
            "duration_exception",
            "expected_exception_count",
            "exposure",
            "whole_job_conformance",
            "priority_distribution",
        )
    }


def requirements_for(queries, *, evaluators=None):
    """Combine query scopes and capability alternatives before inference.

    Args:
        queries (QueryBundle): Versioned query declarations to evaluate together.
        evaluators (Mapping[str, QueryEvaluator] | None): Local registry, defaulting
            to the built-in process evaluators.

    Returns:
        result (QueryRequirements): Union of scopes and necessary capabilities, with
            minimal alternative combinations that can satisfy the complete bundle.

    Raises:
        CapabilityError: A query kind or version has no matching evaluator.

    This call inspects declarations. It neither runs inference nor samples a posterior.
    """
    evaluators = builtin_evaluators() if evaluators is None else evaluators
    scopes, capabilities, alternatives = set(), set(), [frozenset()]
    for query in queries.queries:
        evaluator = evaluators.get(query.kind)
        if evaluator is None or evaluator.version != query.version:
            raise CapabilityError("missing query evaluator/version", key=query.kind)
        requirements = evaluator.requirements(query)
        scopes.update(requirements.scopes)
        capabilities.update(requirements.capabilities)
        alternatives = {
            a | frozenset(b) for a, b in product(alternatives, requirements.alternatives or ((),))
        }
        # Remove supersets; they do not add another route to satisfy the bundle.
        alternatives = [a for a in alternatives if not any(b < a for b in alternatives)]
    return QueryRequirements(
        tuple(sorted(scopes)),
        tuple(sorted(capabilities)),
        tuple(sorted(tuple(sorted(a)) for a in alternatives if a)),
    )


def evaluate(result, queries, *, evaluators=None, rng=None, store=None, control=None):
    evaluators = builtin_evaluators() if evaluators is None else evaluators
    requirements = requirements_for(queries, evaluators=evaluators)
    if not requirements.satisfied_by(result.capabilities):
        raise CapabilityError("posterior lacks the joint information required by this query bundle")
    if control is not None and any(
        not getattr(evaluators[q.kind], "cooperative", False) for q in queries.queries
    ):
        raise CapabilityError("evaluator does not declare cooperative controls")
    from .aggregation import draw_data

    def calculate(data=None):
        estimates = []
        try:
            for query in queries.queries:
                checkpoint(
                    control,
                    "query.evaluate",
                    key=query.name,
                    completed=len(estimates),
                    total=len(queries.queries),
                )
                evaluator = evaluators[query.kind]
                args = (result, query) if data is None else (result, query, data)
                base = evaluator.evaluate if data is None else evaluator.evaluate_on
                estimate = (
                    base(*args, store=store, control=control)
                    if getattr(evaluator, "cooperative", False)
                    else base(*args)
                )
                estimates.append(estimate)
        except ExecutionStopped as exc:
            if not estimates:
                raise
            return QueryResults(
                result.model_id,
                result.run_id,
                queries.bundle_id,
                tuple(estimates),
                ExecutionAssessment(
                    exc.status,
                    "evaluate",
                    details={
                        "failure": str(exc),
                        "completed_queries": tuple(e.name for e in estimates),
                        "missing_queries": tuple(q.name for q in queries.queries[len(estimates) :]),
                    },
                ),
            )
        return QueryResults(result.model_id, result.run_id, queries.bundle_id, tuple(estimates))

    # Try admitted exact tables first. No numerical or validation failure licenses sampling.
    if "joint" in result.capabilities:
        try:
            return calculate()
        except BudgetExceeded:
            if "joint_draws" not in result.capabilities:
                raise
    scope = tuple(sorted({k for s in requirements.scopes for k in s}))
    if any(not hasattr(evaluators[q.kind], "evaluate_on") for q in queries.queries):
        raise CapabilityError("draw evaluator extension must implement evaluate_on")
    if control is not None and any(
        not getattr(evaluators[q.kind], "cooperative", False) for q in queries.queries
    ):
        raise CapabilityError("draw evaluator does not declare cooperative controls")
    checkpoint(control, "query.draws.begin")
    if control is not None and not getattr(result.posterior, "cooperative", False):
        raise CapabilityError("posterior does not declare cooperative draw controls")
    common = (
        result.posterior.draw(scope, rng=rng, control=control)
        if getattr(result.posterior, "cooperative", False)
        else result.posterior.draw(scope, rng=rng)
    )
    checkpoint(control, "query.draws.end", allocation_bytes=8 * prod(common.shape))
    data = draw_data(common)
    return calculate(data)
