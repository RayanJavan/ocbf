"""Concrete vectorized process evaluators over neutral tables or common draw sets."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import rankdata

from ocbf.assertions import AssertionRef, Family
from ocbf.belief.estimates import Estimate
from ocbf.belief.posterior import QueryRequirements, draw_dtype
from ocbf.diagnostics.monte_carlo import assessment
from ocbf.errors import CapabilityError, ValidationError
from ocbf.runtime.cache import artifact_key, cached
from ocbf.runtime.control import checkpoint

from .history import (
    INAPPLICABLE,
    OUTCOMES,
    SATISFIED,
    UNRESOLVED,
    VIOLATED,
    interval,
    job_status,
    obligation,
    overlap,
    project_execution,
)


@dataclass(frozen=True)
class EvaluationData:
    state: object
    weights: np.ndarray
    method: str
    draw_set_id: str | None = None
    draw_status: str = "complete"
    representation_id: str | None = None

    @property
    def shape(self):
        return self.weights.shape

    def mean(self, values):
        return float(np.sum(values * self.weights))

    def numerical(self, numerator, denominator=None):
        if self.method == "exact":
            return {"status": "exact", "mcse": 0.0}
        result = assessment(numerator, denominator, method=self.method)
        if self.draw_status != "complete":
            result = {**result, "status": "incomplete", "draw_status": self.draw_status}
        return result


def exact_data(posterior, scope, *, store=None, control=None):
    if control is not None and not getattr(posterior, "cooperative", False):
        raise CapabilityError("posterior does not declare cooperative joint-table controls")
    table = (
        posterior.joint(scope, store=store, control=control)
        if getattr(posterior, "cooperative", False)
        else posterior.joint(scope)
    )
    size = table.probabilities.size
    checkpoint(control, "query.exact_data", allocation_bytes=size * (16 + 8 * len(scope)))
    state = {}
    for axis, (key, domain) in enumerate(zip(table.scope, table.domains, strict=True)):
        try:
            labels = np.asarray(domain, dtype=draw_dtype(domain))
        except CapabilityError:
            # Exact finite domains retain their broader contract, including mixed labels.
            labels = np.empty(len(domain), dtype=object)
            labels[:] = domain
        shape = [1] * len(table.scope)
        shape[axis] = len(domain)
        state[key] = np.broadcast_to(labels.reshape(shape), table.probabilities.shape).reshape(
            1, size
        )
    return EvaluationData(
        state,
        table.probabilities.reshape(1, size),
        "exact",
        representation_id=artifact_key("exact-grid", table.scope, table.domains),
    )


def draw_data(draws):
    shape = draws.shape
    return EvaluationData(
        draws.values,
        np.full(shape, 1 / np.prod(shape)),
        draws.method,
        draws.draw_set_id,
        draws.metadata.get("status", "complete"),
        draws.draw_set_id,
    )


def validate_bindings(result, query):
    domains = getattr(result.posterior, "domains", result.manifest.get("variable_domains", {}))
    for execution in query.executions:
        for endpoint in (*execution.starts, *execution.ends):
            try:
                ref = AssertionRef.parse(endpoint.event_key)
            except ValueError:
                continue
            if ref.family is not Family.EVENT_EXISTS:
                raise CapabilityError("semantic endpoint must reference event existence")
            if endpoint.time_key:
                if endpoint.time_key != str(AssertionRef.event_time(ref.subject)):
                    raise CapabilityError("modeled time is bound to another event")
                continuous = result.manifest.get("decoding", {}).get("continuous", {})
                if endpoint.time_key not in continuous:
                    raise CapabilityError("query time is not modeled", key=endpoint.time_key)
            elif endpoint.time is not None:
                fixed = result.manifest.get("decoding", {}).get("fixed_endpoints", {})
                if fixed.get(ref.subject) != endpoint.time:
                    raise CapabilityError(
                        "query endpoint differs from fixed-time conditioning",
                        key=endpoint.event_key,
                    )
        tests = (
            *execution.applicable_when,
            *(t for e in (*execution.starts, *execution.ends) for t in e.association),
            *(t for c in query.conditions for t in c.present_when),
        )
        for key, expected in tests:
            if key in domains and expected not in domains[key]:
                raise CapabilityError(
                    "query condition names a state outside its candidate domain", key=key
                )


def lineage(result, scope):
    closure, used, evidence = set(scope), set(), set()
    factors = result.manifest.get("factor_scopes", {})
    dependencies = result.manifest.get("dependencies", {})
    changed = True
    while changed:
        changed = False
        for key, keys in factors.items():
            if key not in used and closure.intersection(keys):
                closure.update(keys)
                used.add(key)
                evidence.update(dependencies.get(key, ()))
                changed = True
    return used, evidence


@dataclass(frozen=True)
class ExecutionEvaluator:
    version: str = "1"
    cooperative = True

    def requirements(self, query):
        extra = {k for c in query.conditions for k, _ in c.present_when}
        scopes = tuple(tuple(sorted(set(e.scope) | extra)) for e in query.executions)
        if query.kind in ("whole_job_conformance", "priority_distribution"):
            scopes = (tuple(sorted({k for s in scopes for k in s})),)
        return QueryRequirements(scopes, (), (("joint",), ("joint_draws",)))

    def evaluate(self, result, query, *, store=None, control=None):
        if query.kind in ("expected_exception_count", "exposure") and len(query.executions) > 1:
            parts = [
                self.evaluate(
                    result, replace(query, executions=(execution,)), store=store, control=control
                )
                for execution in query.executions
            ]
            denominator = sum(p.denominator for p in parts)
            outcomes = {label: sum(p.outcomes[label] for p in parts) for label in OUTCOMES}
            if query.kind == "expected_exception_count":
                value = sum(p.value or 0 for p in parts) if denominator else None
            else:
                value = (
                    sum((p.value or 0) * p.denominator for p in parts) / denominator
                    if denominator
                    else None
                )
            scope = {k for s in self.requirements(query).scopes for k in s}
            factors, evidence = lineage(result, scope)
            return replace(
                parts[0],
                query_id=query.query_id,
                value=value,
                denominator=denominator,
                outcomes=outcomes,
                status="assessed" if value is not None else "unresolved",
                evidence_ids=tuple(
                    evidence | {r for c in query.conditions for r in c.evidence_ids}
                ),
                metadata={
                    **parts[0].metadata,
                    "population": tuple(e.execution_id for e in query.executions),
                    "population_ids": tuple(e.population_id for e in query.executions),
                    "population_size": len(query.executions),
                    "factors": tuple(sorted(factors)),
                },
            )
        scope = tuple(sorted({k for s in self.requirements(query).scopes for k in s}))
        return self.evaluate_on(
            result,
            query,
            exact_data(result.posterior, scope, store=store, control=control),
            store=store,
            control=control,
        )

    def evaluate_on(self, result, query, data, *, store=None, control=None):
        checkpoint(control, "query.bindings", key=query.name)
        checkpoint(
            control,
            "query.workspace",
            allocation_bytes=int(np.prod(data.shape)) * (96 * len(query.executions) + 128),
        )
        validate_bindings(result, query)
        if not query.executions:
            raise ValidationError("process query requires an explicit execution population")
        scope = {k for s in self.requirements(query).scopes for k in s}
        factors, evidence = lineage(result, scope)
        evidence.update(r for c in query.conditions for r in c.evidence_ids)
        projections, statuses = {}, []
        for execution in query.executions:
            checkpoint(
                control,
                "query.projection",
                key=execution.execution_id,
                allocation_bytes=int(np.prod(data.shape)) * 64,
            )
            projected = cached(
                store if data.representation_id else None,
                "query.projection",
                (data.representation_id, execution, result.manifest.get("decoding", {})),
                lambda execution=execution: project_execution(execution, data.state, data.shape),
            )
            projections[execution.execution_id] = projected
            statuses.append(
                obligation(execution, data.state, query, data.shape, projected=projected)
            )
        numerator = np.zeros(data.shape)
        denominator_series = np.zeros(data.shape)
        distribution = {}
        qualifications = [
            "Conditional on declared candidate support, time model and evidence coverage.",
            "Missing endpoints and unresolved associations are not zero durations.",
            "Configured reference is a query input, not an observed production standard.",
        ]
        quantity, unit = "", "probability"
        if query.kind == "priority_distribution":
            jobs = tuple(sorted({e.population_id for e in query.executions}))
            eligible = np.ones(data.shape, dtype=bool)
            scores = []
            for job in jobs:
                checkpoint(
                    control,
                    "query.rank",
                    key=job,
                    allocation_bytes=int(np.prod(data.shape)) * len(query.executions) * 32,
                )
                selected = np.stack(
                    [
                        s
                        for e, s in zip(query.executions, statuses, strict=True)
                        if e.population_id == job
                    ]
                )
                eligible &= np.all(
                    np.isin(selected, (SATISFIED, VIOLATED, INAPPLICABLE)), axis=0
                ) & np.any(selected != INAPPLICABLE, axis=0)
                scores.append(np.sum(selected == VIOLATED, axis=0))
            scores = np.stack(scores)
            # Sorting preserves competition ties without a jobs-by-jobs-by-draw tensor.
            ranks = rankdata(-scores, method="min", axis=0)
            total = data.mean(eligible)
            rank_distributions, score_means = {}, {}
            for i, job in enumerate(jobs):
                rank_distributions[job] = {
                    str(rank): {
                        "probability": data.mean(eligible & (ranks[i] == rank)) / total
                        if total
                        else None,
                        "numerical": data.numerical(
                            (eligible & (ranks[i] == rank)).astype(float), eligible
                        ),
                    }
                    for rank in range(1, len(jobs) + 1)
                }
                score_means[job] = data.mean(eligible * scores[i]) / total if total else None
            maximum = scores.max(axis=0)
            ties = (scores == maximum).sum(axis=0) > 1
            distribution = {
                "ranks": rank_distributions,
                "posterior_mean_scores": score_means,
                "ranks_of_posterior_means": {
                    job: 1 + sum(score_means[other] > score_means[job] for other in jobs)
                    for job in jobs
                }
                if total
                else {},
                "top_tie_probability": data.mean(eligible & ties) / total if total else None,
                "score": "evaluable violation count",
                "ranking": "competition; tied rank-one probabilities need not sum to one",
                "unrankable_mass": 1 - total,
            }
            outcomes = {"rankable": total, "unrankable": 1 - total}
            value, denominator, numerical = None, total, data.numerical(eligible.astype(float))
            numerical = {
                **numerical,
                "rank_estimates": "per-Job, per-rank numerical assessments are in distribution.ranks",
            }
            quantity, unit = "conditional priority rank distributions", "rank"
        else:
            if query.kind == "whole_job_conformance":
                if len({e.population_id for e in query.executions}) != 1:
                    raise ValidationError("whole-Job conformance requires one declared Job")
                statuses = [job_status(statuses)]
                numerator = (statuses[0] == SATISFIED).astype(float)
                quantity = "conformance conditional on decidable applicable Job"
            elif query.kind == "exposure":
                statuses = []
                for execution in query.executions:
                    checkpoint(control, "query.exposure", key=execution.execution_id)
                    status, start, end = interval(
                        execution,
                        data.state,
                        query,
                        data.shape,
                        projected=projections[execution.execution_id],
                    )
                    closed = status == SATISFIED
                    if query.coverage_verified:
                        duration = overlap(
                            start, end, query.conditions, data.state, data.shape, query.horizon
                        )
                        numerator += np.where(closed, duration, 0)
                    else:
                        status[closed] = UNRESOLVED
                    statuses.append(status)
                quantity, unit = (
                    "expected observed overlap conditional on evaluable execution",
                    "minutes",
                )
                qualifications.append(
                    "Descriptive overlap is not causal delay or ready-to-work waiting."
                )
            elif query.kind in ("duration_exception", "expected_exception_count"):
                if query.kind == "duration_exception" and len(statuses) != 1:
                    raise ValidationError(
                        "duration probability requires one execution; use expected count"
                    )
                numerator = sum((s == VIOLATED).astype(float) for s in statuses)
                quantity = "violation conditional on applicable closed evaluable execution"
                if query.kind == "expected_exception_count":
                    quantity, unit = "expected count of evaluable violations", "executions"
            else:
                raise CapabilityError("unsupported query kind", key=query.kind)
            denominator_series = sum(
                np.isin(s, (SATISFIED, VIOLATED)).astype(float) for s in statuses
            )
            outcomes = {
                label: sum(data.mean(s == code) for s in statuses)
                for code, label in enumerate(OUTCOMES)
            }
            denominator = data.mean(denominator_series)
            conditional = query.kind != "expected_exception_count"
            value = (
                (data.mean(numerator) / denominator if conditional else data.mean(numerator))
                if denominator
                else None
            )
            numerical = data.numerical(numerator, denominator_series if conditional else None)
        status = "assessed" if value is not None or (distribution and denominator) else "unresolved"
        if status == "assessed" and numerical["status"] not in ("exact", "assessed"):
            status = "numerically_inadequate"
        computation = (
            result.computation
            if data.method == "exact"
            else f"{data.method} query evaluation of {result.computation}"
        )
        return Estimate(
            query.query_id,
            query.name,
            quantity,
            value,
            unit,
            status,
            denominator,
            computation,
            outcomes,
            tuple(qualifications),
            tuple(evidence),
            {
                "population": tuple(e.execution_id for e in query.executions),
                "population_ids": tuple(e.population_id for e in query.executions),
                "population_size": len(query.executions),
                "window_start": query.window_start,
                "horizon": query.horizon,
                "reference": query.reference,
                "factors": tuple(sorted(factors)),
                "knowledge_time": result.manifest.get("knowledge_cutoff"),
                "model_scope": result.manifest.get("scope", {}),
                "interval_convention": "half-open",
                "incomplete_policy": "separate outcomes",
                "draw_set_id": data.draw_set_id,
            },
            distribution,
            numerical,
        )
