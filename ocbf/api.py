"""Small synchronous composition facade. Scientific work remains in its owning modules."""

from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from ocbf._values import canonical_json, freeze
from ocbf.errors import ValidationError
from ocbf.evidence.snapshots import materialize
from ocbf.inference.router import infer as _infer
from ocbf.inference.router import plan_inference as _plan_inference
from ocbf.inference.warm_start import warm_start_from
from ocbf.model.compile import compile_model as _compile_model
from ocbf.model.dependencies import dependency_index
from ocbf.queries.evaluate import evaluate as _evaluate
from ocbf.queries.evaluate import requirements_for
from ocbf.runtime.control import ExecutionControl, checkpoint, operation
from ocbf.runtime.session import ExecutionSession
from ocbf.sources.interpretation import interpret_snapshot


@contextmanager
def _execution(stage, session, control, reuse):
    store = session.store_for(reuse) if session is not None else None
    try:
        with operation(stage, store=store, control=control) as run:
            yield store, run
    except Exception as exc:
        if session is not None:
            session.last_assessment = getattr(exc, "execution", None)
        raise
    else:
        if session is not None:
            session.last_assessment = run.assessment


def prepare_evidence(
    records,
    *,
    as_of,
    interpreters,
    context,
    revision_policy="explicit-chain-v1",
    session=None,
    control=None,
    reuse=True,
):
    """Materialize effective revisions and interpret them within a semantic context.

    Args:
        records (Iterable[EvidenceRecord]): Immutable evidence envelopes, including their revision chains.
        as_of (datetime | None): Knowledge cutoff; use ``None`` with an explicitly static policy.
        interpreters (Mapping[tuple[str, str], EvidenceInterpreter]): Local ``(source_id, producer_version)`` interpreter registry.
        context (SemanticContext): Immutable semantic context and candidate support.
        revision_policy (str): Materialization policy; defaults to ``explicit-chain-v1``.
        session (ExecutionSession | None): Optional caller-owned execution session for bounded reuse.
        control (ExecutionControl | None): Optional cooperative deadline, cancellation and progress control.
        reuse (bool): Whether to use the supplied session's artifact store.

    Returns:
        result (InterpretedEvidence): Effective evidence with its effective snapshot, observations and admission issues.

    Raises:
        EvidenceConflict: Revision identities or actions cannot be resolved consistently.
        ValidationError: Records, clocks or policy inputs are invalid.

    Unknown report meanings remain admission issues. This call does not fetch external
    records, infer missing provenance or construct likelihoods. A session records execution
    assessment; the caller owns its lifetime.
    """
    with _execution("prepare_evidence", session, control, reuse) as (store, run):
        snapshot = materialize(records, as_of=as_of, policy=revision_policy, control=control)
        result = interpret_snapshot(snapshot, context, interpreters, store=store, control=control)
        run.finish(evidence_id=snapshot.snapshot_id)
        return result


def compile_model(spec, *, channels=None, factors=None, session=None, control=None, reuse=True):
    """Validate scientific inputs and compile their canonical probability model.

    Args:
        spec (ModelSpec): Model specification containing context, evidence, resolved parameters,
            variables, factors, support and decoding declarations.
        channels (Mapping[str, ObservationChannel] | None): Optional local observation-channel registry; defaults to built-ins.
        factors (Mapping[str, FactorKernel] | None): Optional local factor-kernel registry; defaults to built-ins.
        session (ExecutionSession | None): Optional caller-owned session for bounded artifact reuse.
        control (ExecutionControl | None): Optional cooperative execution control.
        reuse (bool): Whether to use the supplied session's artifact store.

    Returns:
        result (CompiledModel): Immutable model with scientific identity, factors and manifest.

    Raises:
        ValidationError: References, parameters, units or declarations are invalid.
        CapabilityError: A requested observation/factor construction is unsupported.
        IncompatibleModel: An admitted factor has no supported states.

    No source parsing, parameter fitting or solver selection occurs here. Normalization
    can discover further incompatibility during inference. Normative references remain
    query inputs rather than silently constraining this target.
    """
    with _execution("compile", session, control, reuse) as (store, run):
        model = _compile_model(
            spec, channels=channels, factors=factors, store=store, control=control
        )
        details = {}
        if store is not None:
            # Explainable cache decision: which numeric factors changed since a prior model
            # of the same structure. Content addressing still drives the actual reuse.
            changed = session.explain_dependencies(dependency_index(model))
            if changed:
                details["dependency_changes"] = changed
        run.finish(model_id=model.model_id, **details)
        return model


def plan_inference(
    model, *, requirements=None, policy=None, engines=None, session=None, control=None, reuse=True
):
    """Assess the model and requested posterior capabilities before solving.

    Args:
        model (CompiledModel): Compiled canonical model.
        requirements (QueryRequirements | None): Declarative scopes and required capabilities; defaults to an empty
            request. Derive process-query requirements with ``requirements_for``.
        policy (InferencePolicy | None): Explicit engine and budgets. Falls back to the session policy, then the
            default ``InferencePolicy`` whose engine is ``gtsam_exact``.
        engines (Mapping[str, InferenceEngine] | None): Optional local engine registry; defaults to built-ins.
        session (ExecutionSession | None): Optional caller-owned session for reusable planning artifacts.
        control (ExecutionControl | None): Optional cooperative execution control.
        reuse (bool): Whether to use the supplied session's artifact store.

    Returns:
        result (ExecutionPlan): Plan identifying the admitted route and its cost/capability assessment.

    Raises:
        CapabilityError: The requested route cannot supply the required capabilities.
        BudgetExceeded: The admitted computation exceeds a declared planning budget.

    ``engine="auto"`` explicitly enables ordered route assessment. A plan is a preflight
    assessment, not proof of physical accuracy or successful native backend execution.
    """
    with _execution("plan", session, control, reuse) as (store, _run):
        policy = policy if policy is not None else session.policy if session is not None else None
        return _plan_inference(
            model,
            requirements=requirements,
            policy=policy,
            engines=engines,
            control=control,
            store=store,
        )


def infer(
    model,
    *,
    requirements=None,
    policy=None,
    engines=None,
    rng=None,
    session=None,
    control=None,
    reuse=True,
    warm_start=None,
):
    """Compute a qualified posterior for one fixed canonical target.

    Args:
        model (CompiledModel): Compiled model with resolved immutable parameters.
        requirements (QueryRequirements | None): Required posterior scopes/capabilities, normally derived from queries.
        policy (InferencePolicy | None): Engine and budgets; uses the session policy when supplied, otherwise the
            default ``gtsam_exact`` policy. Automatic routing is opt-in.
        engines (Mapping[str, InferenceEngine] | None): Optional local registry of inference engines.
        rng (numpy.random.Generator | None): NumPy generator required for stochastic execution and joint drawing.
        session (ExecutionSession | None): Optional caller-owned execution session.
        control (ExecutionControl | None): Optional cooperative deadline, cancellation and progress control.
        reuse (bool): Enable validated reuse; ``False`` also ignores warm-start hints.
        warm_start (WarmStart | None): Explicit sampler initialization hint for a compatible revised target.

    Returns:
        result (InferenceResult): Neutral result with scientific/run identities, posterior capabilities,
            computation qualifications and diagnostics. Controlled execution adds an assessment.
            Usable partial draws retain an incomplete execution status.

    Raises:
        CapabilityError: The route, controls or warm start are unsupported.
        BackendUnavailable: A requested optional numerical backend cannot load.
        IncompatibleModel: The target has zero supported mass.
        NumericalFailure: Normalization or other numerical state is invalid.
        BudgetExceeded: No usable result exists within the declared resource budget.

    Parameters are never fitted and evidence is never fetched by inference. Results retain
    their posterior by default. Reuse cannot change the scientific target; sampling hints
    still require fresh streams, warmup and support validation.
    """
    with _execution("infer", session, control, reuse) as (store, run):
        policy = policy if policy is not None else session.policy if session is not None else None
        result = _infer(
            model,
            requirements=requirements,
            policy=policy,
            engines=engines,
            rng=rng,
            store=store,
            control=control,
            warm_start=warm_start if reuse else None,
        )
        status = result.diagnostics.get("status", "complete")
        status = {"budget_exhausted": "resource-exhausted"}.get(status, status)
        assessment = run.finish(
            status,
            model_id=model.model_id,
            run_id=result.run_id,
            **(
                {"warm_start": result.diagnostics["warm_start"]}
                if "warm_start" in result.diagnostics
                else {}
            ),
        )
        return (
            replace(result, execution=assessment)
            if session is not None or control is not None or status != "complete"
            else result
        )


def evaluate(result, queries, *, evaluators=None, rng=None, session=None, control=None, reuse=True):
    """Evaluate a query bundle using the result's declared posterior capabilities.

    Args:
        result (InferenceResult): Qualified inference result with a retained posterior representation.
        queries (QueryBundle): Versioned query bundle with references, projections and time scope.
        evaluators (Mapping[str, QueryEvaluator] | None): Optional local query-evaluator registry; defaults to built-ins.
        rng (numpy.random.Generator | None): NumPy generator when evaluation needs additional joint draws.
        session (ExecutionSession | None): Optional caller-owned session for reusable query artifacts.
        control (ExecutionControl | None): Optional cooperative execution control.
        reuse (bool): Whether to use the supplied session's artifact store.

    Returns:
        result (QueryResults): Results preserving model/run identity, estimates, denominators, evidence
            qualifications and numerical assessments. Partial controlled evaluation identifies
            completed and missing queries in its execution assessment.

    Raises:
        CapabilityError: The posterior lacks required joint/table/draw capabilities.
        ValidationError: Query bindings, population, references or scope are invalid.

    Related sampled queries use common draws. An evaluator does not fit parameters,
    rebuild the model or launch hidden inference. Unassessable quantities remain explicit
    unresolved results; they do not receive invented probabilities or durations.
    """
    with _execution("evaluate", session, control, reuse) as (store, run):
        answer = _evaluate(
            result, queries, evaluators=evaluators, rng=rng, store=store, control=control
        )
        assessment = run.finish(
            answer.execution.status if answer.execution else "complete",
            model_id=result.model_id,
            run_id=result.run_id,
            **(dict(answer.execution.details) if answer.execution else {}),
        )
        run.assessment = assessment
        return (
            replace(answer, execution=assessment)
            if session is not None or control is not None
            else answer
        )


def summaries_only(result):
    """Return a detached metadata-only result after evaluating the desired queries.

    Caller-owned copies and other returned results remain valid. Future posterior access
    requires another explicit inference call or loading a retained posterior export.
    """
    return replace(
        result,
        posterior=None,
        capabilities=(),
        manifest={
            **result.manifest,
            "retention": "summaries-only; posterior capabilities released",
        },
    )


@dataclass(frozen=True)
class SensitivityResult:
    """Separate conditional calculations, not a probability mixture or credible interval."""

    settings: tuple
    meaning: str = "named manual-assumption sensitivity; no mixture weights implied"
    execution: object = field(default=None, metadata={"identity_omit_default": True})

    def __post_init__(self):
        object.__setattr__(self, "settings", tuple(self.settings))
        object.__setattr__(self, "execution", freeze(self.execution))


def compare_settings(
    spec,
    queries,
    settings,
    *,
    channels=None,
    factors=None,
    engines=None,
    evaluators=None,
    policy=None,
    rng=None,
    session=None,
    control=None,
    reuse=True,
):
    """Compare named manual-channel settings through separate conditional calculations.

    Args:
        spec (ModelSpec): Baseline model specification.
        queries (QueryBundle): Query bundle and normative reference held fixed across settings.
        settings (Mapping[str, ParameterSet]): Mapping from setting name to a resolved parameter set.
        channels (Mapping[str, ObservationChannel] | None): Optional observation-channel registry.
        factors (Mapping[str, FactorKernel] | None): Optional factor-kernel registry.
        engines (Mapping[str, InferenceEngine] | None): Optional inference-engine registry.
        evaluators (Mapping[str, QueryEvaluator] | None): Optional query-evaluator registry.
        policy (InferencePolicy | None): Engine and budgets, with the same precedence as ``infer``.
        rng (numpy.random.Generator | None): NumPy generator for stochastic calculations.
        session (ExecutionSession | None): Optional caller-owned session for validated reuse across settings.
        control (ExecutionControl | None): Optional cooperative execution control shared across the comparison.
        reuse (bool): Whether to use the supplied session's artifact store.

    Returns:
        result (SensitivityResult): Comparison containing sorted setting names, parameter identities and query
            results. Session-backed runs record per-setting execution and random-stream details.

    Raises:
        ValidationError: Settings alter baseline priors or structural/temporal assumptions.

    Every named target is compiled before numerical execution starts. Results are separate
    assumption-conditional answers, not mixture probabilities or a credible interval.
    Compilation, inference and evaluation failures retain their typed behavior.
    """
    requirements = requirements_for(queries, evaluators=evaluators)
    outputs = []
    compiled = []
    for name, parameters in sorted(settings.items()):
        if canonical_json(parameters.priors) != canonical_json(spec.parameters.priors):
            raise ValidationError("trust comparison must hold priors fixed", key=name)
        if canonical_json(parameters.assumptions) != canonical_json(spec.parameters.assumptions):
            raise ValidationError(
                "trust comparison must hold structural/temporal assumptions fixed", key=name
            )
        model = compile_model(
            replace(spec, parameters=parameters),
            channels=channels,
            factors=factors,
            session=session,
            control=control,
            reuse=reuse,
        )
        compiled.append((name, parameters, model))
    # Validate every named target before starting any numerical run.
    streams = {}
    if session is not None and rng is not None:
        import numpy as np

        # Allocate every setting stream first; cache operations never consume randomness.
        streams = {
            name: int(seed)
            for (name, _, _), seed in zip(
                compiled, rng.integers(0, 2**63 - 1, size=len(compiled)), strict=True
            )
        }
    runs = []
    for name, parameters, model in compiled:
        checkpoint(control, "compare.setting", setting=name)
        generator = np.random.default_rng(streams[name]) if streams else rng
        result = infer(
            model,
            requirements=requirements,
            policy=policy,
            engines=engines,
            rng=generator,
            session=session,
            control=control,
            reuse=reuse,
        )
        if streams:
            result = replace(
                result,
                manifest={
                    **result.manifest,
                    "setting_stream": {
                        "name": name,
                        "seed": streams[name],
                        "allocation": "sorted names before execution",
                    },
                },
            )
        if session is not None:
            runs.append(
                {
                    "name": name,
                    "model_id": result.model_id,
                    "run_id": result.run_id,
                    "setting_seed": streams.get(name),
                    "chain_seeds": result.posterior.draw_set.metadata.get("chain_seeds", ())
                    if hasattr(result.posterior, "draw_set")
                    else (),
                    "assessment": result.execution,
                }
            )
        outputs.append(
            (
                name,
                parameters.parameter_id,
                evaluate(
                    result,
                    queries,
                    evaluators=evaluators,
                    rng=generator,
                    session=session,
                    control=control,
                    reuse=reuse,
                ),
            )
        )
    return SensitivityResult(
        tuple(outputs),
        execution={
            "runs": tuple(runs),
            "stream_allocation": "sorted setting names before execution; chain seeds before sampling",
        }
        if session is not None
        else None,
    )


__all__ = [
    "ExecutionControl",
    "ExecutionSession",
    "SensitivityResult",
    "compare_settings",
    "compile_model",
    "evaluate",
    "infer",
    "plan_inference",
    "prepare_evidence",
    "requirements_for",
    "summaries_only",
    "warm_start_from",
]
