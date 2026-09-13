"""Validate and ground a canonical target; numerical lowering belongs to engine adapters."""

import math
from collections import defaultdict
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ocbf._values import canonical_json, freeze
from ocbf.assertions import AssertionRef as Ref
from ocbf.errors import CapabilityError, IncompatibleModel, ValidationError
from ocbf.runtime.cache import cached, extension_key
from ocbf.runtime.control import checkpoint
from ocbf.schema.constraints import ConstraintClass, Strength

from .kernels import builtin_channels, builtin_kernels
from .spec import CompiledModel, ContinuousVariableSpec, FactorSpec, VariableSpec


def structural_variables(context):
    d = context.definition
    variables = []
    for o in d["objects"]:
        variables.append(VariableSpec(str(Ref.object_exists(o.id)), (True,)))
    for e in d["events"]:
        if "__inactive__" in e.type_domain:
            raise ValidationError("event type collides with reserved inactive state", key=e.id)
        variables.extend(
            (
                VariableSpec(str(Ref.event_exists(e.id)), (False, True)),
                VariableSpec(str(Ref.event_type(e.id)), ("__inactive__", *e.type_domain)),
            )
        )
    variables.extend(VariableSpec(str(Ref.e2o(*key)), (False, True)) for key in d["e2o"])
    variables.extend(VariableSpec(str(Ref.o2o(*key)), (False, True)) for key in d["o2o"])
    return tuple(variables)


def _structure(spec):
    d, schema = spec.context.definition, spec.context.schema
    register = {c.constraint: c for c in d["constraints"]}
    for cls in (
        ConstraintClass.REFERENTIAL_INTEGRITY,
        ConstraintClass.QUALIFIER_LEGALITY,
        ConstraintClass.TYPE_DISJOINTNESS,
    ):
        if not register[cls].enabled or register[cls].strength is not Strength.HARD:
            raise CapabilityError(
                "new structural target requires definitional hard support", key=cls.value
            )
    factors = []
    objects = {o.id: o.object_type for o in d["objects"]}
    for e in d["events"]:
        factors.append(
            FactorSpec(
                "type_presence:" + e.id,
                (str(Ref.event_exists(e.id)), str(Ref.event_type(e.id))),
                "type_presence",
                role="support",
            )
        )
    groups = defaultdict(list)
    for event, qualifier, obj in d["e2o"]:
        key = str(Ref.e2o(event, qualifier, obj))
        factors.append(
            FactorSpec(
                "integrity:" + key,
                (key, str(Ref.event_exists(event))),
                "implication",
                role="support",
            )
        )
        allowed = tuple(
            q.event_type
            for q in d["e2o_qualifiers"]
            if q.qualifier == qualifier and q.object_type == objects[obj]
        )
        factors.append(
            FactorSpec(
                "legality:" + key,
                (key, str(Ref.event_type(event))),
                "type_gate",
                {"allowed": allowed},
                role="support",
            )
        )
        groups[(event, qualifier, objects[obj])].append(key)
    for source, qualifier, target in d["o2o"]:
        if not schema.is_legal_o2o(objects[source], qualifier, objects[target]):
            raise ValidationError(
                "illegal O2O candidate", key=str(Ref.o2o(source, qualifier, target))
            )
    cardinality = register[ConstraintClass.CARDINALITY]
    # Include zero-candidate groups: a required relation cannot disappear silently.
    for event in d["events"]:
        for qualifier in d["e2o_qualifiers"]:
            if qualifier.event_type in event.type_support:
                groups[(event.id, qualifier.qualifier, qualifier.object_type)]
    # The classification must be explicit; catalogue norms must not silently constrain belief.
    classification = spec.parameters.assumptions.get("cardinality_role")
    if classification not in ("normative", "descriptive", "support"):
        raise ValidationError("cardinality_role must explicitly classify catalogue multiplicities")
    if cardinality.enabled and classification != "normative":
        scale = spec.parameters.assumptions.get("constraint_scale_nats")
        if classification == "descriptive" and (
            scale is None or not math.isfinite(scale) or scale <= 0
        ):
            raise ValidationError(
                "descriptive cardinality needs fixed positive constraint_scale_nats"
            )
        for (event, qualifier, target_type), links in sorted(groups.items()):
            bounds = {
                q.event_type: (q.multiplicity.lo, q.multiplicity.hi)
                for q in d["e2o_qualifiers"]
                if q.qualifier == qualifier and q.object_type == target_type
            }
            factors.append(
                FactorSpec(
                    f"cardinality:{event}:{qualifier}:{target_type}",
                    (str(Ref.event_type(event)), *sorted(links)),
                    "cardinality",
                    {
                        "bounds": bounds,
                        "penalty": -math.inf
                        if classification == "support"
                        else -cardinality.weight * scale,
                    },
                    role=classification,
                )
            )
        for source, source_type in sorted(objects.items()):
            for q in d["o2o_qualifiers"]:
                if q.source_type != source_type:
                    continue
                links = tuple(
                    sorted(
                        str(Ref.o2o(a, b, c))
                        for a, b, c in d["o2o"]
                        if a == source and b == q.qualifier and objects[c] == q.target_type
                    )
                )
                factors.append(
                    FactorSpec(
                        f"cardinality:{source}:{q.qualifier}:{q.target_type}:o2o",
                        links,
                        "count",
                        {
                            "lo": q.multiplicity.lo,
                            "hi": q.multiplicity.hi,
                            "penalty": -math.inf
                            if classification == "support"
                            else -cardinality.weight * scale,
                        },
                        role=classification,
                    )
                )
    for binding in spec.lifecycle_bindings:
        if binding.object_id not in objects:
            raise ValidationError("lifecycle object is absent", key=binding.object_id)
        before = str(Ref.e2o(binding.before_event, binding.qualifier, binding.object_id))
        after = str(Ref.e2o(binding.after_event, binding.qualifier, binding.object_id))
        candidates = {str(Ref.e2o(*k)) for k in d["e2o"]}
        if before not in candidates or after not in candidates:
            raise ValidationError(
                "lifecycle endpoints must link to the same declared object", key=binding.object_id
            )
        if binding.role != "normative":
            factors.append(
                FactorSpec(
                    f"precedence:{binding.object_id}:{binding.before_event}:{binding.after_event}",
                    (before, after),
                    "precedence",
                    {
                        "before": binding.before.timestamp(),
                        "after": binding.after.timestamp(),
                        "penalty": -math.inf
                        if binding.role == "support"
                        else -binding.penalty_nats,
                    },
                    role=binding.role,
                )
            )
    return factors


def compile_model(spec, *, channels=None, factors=None, store=None, control=None):
    checkpoint(control, "compile.validate")
    channels = builtin_channels() if channels is None else dict(channels)
    kernels = builtin_kernels() if factors is None else dict(factors)
    tokens = (
        (
            {k: extension_key(v) for k, v in channels.items()},
            {k: extension_key(v) for k, v in kernels.items()},
        )
        if store is not None
        else None
    )
    from ocbf.runtime.cache import artifact_key

    key = None
    if store is not None and all(t is not None for registry in tokens for t in registry.values()):
        # Reuse a completed validation only for identical, fully specified immutable inputs.
        # A new evidence/interpretation/parameter epoch necessarily gets a distinct key.
        key = artifact_key("canonical-compiler-v1", spec, tokens)
        cached_model = store.get("compile.validated", key)
        if cached_model is not None:
            existing, index = cached_model
            object.__setattr__(existing, "_dependency_index", index)
            for factor in existing.factors:
                checkpoint(
                    control,
                    "compile.factor",
                    key=factor.key,
                    allocation_bytes=factor.log_values.nbytes
                    if factor.log_values is not None
                    else 0,
                )
            return existing
    model = _compile(spec, channels, kernels, store, control)
    if key is not None:
        store.put("compile.validated", key, (model, model._dependency_index))
    return model


def _compile(spec, channels, kernels, store, control):
    interpreted_context = spec.evidence.manifest.get("context_id")
    if interpreted_context is not None and interpreted_context != spec.context.context_id:
        raise ValidationError("evidence was interpreted against a different semantic context")
    variables = tuple(
        sorted(
            spec.variables
            or cached(
                store,
                "structure.variables",
                spec.context.context_id,
                lambda: structural_variables(spec.context),
            ),
            key=lambda v: v.key,
        )
    )
    declared = {v.key: v for v in variables}
    if len(declared) != len(variables):
        raise ValidationError("duplicate variable identity")
    contributions = {}
    for observation in spec.evidence.observations:
        checkpoint(control, "compile.observation", key=observation.observation_id)
        if not set(observation.scope) <= declared.keys():
            raise ValidationError(
                "observation outside candidate support", key=observation.observation_id
            )
        channel = channels.get(observation.channel)
        if channel is not None and hasattr(channel, "contribute"):
            parameters = spec.parameters.for_observation(observation)
            token = extension_key(channel)
            contribution = cached(
                store if token else None,
                "channel.contribution",
                (token, observation, parameters, declared),
                lambda channel=channel, observation=observation, parameters=parameters: (
                    channel.contribute(observation, parameters, declared)
                ),
                slot=observation.observation_id,
            )
            if contribution.version != "1":
                raise CapabilityError("unsupported channel contribution version")
            contributions[observation.observation_id] = contribution
            for variable in contribution.variables:
                if variable.key in declared and canonical_json(
                    declared[variable.key]
                ) != canonical_json(variable):
                    raise ValidationError(
                        "conflicting shared variable declaration", key=variable.key
                    )
                declared[variable.key] = variable
    variables = tuple(sorted(declared.values(), key=lambda v: v.key))
    domains = {v.key: v.domain for v in variables if hasattr(v, "domain")}
    continuous = {v.key: v for v in variables if isinstance(v, ContinuousVariableSpec)}
    for v in continuous.values():
        if any(k not in domains or value not in domains[k] for k, value in v.active_when):
            raise ValidationError("invalid continuous activation", key=v.key)
    if not spec.ground_structure:
        if spec.lifecycle_bindings:
            raise CapabilityError("lifecycle grounding requires structural compilation")
        # Custom finite latents are allowed, but semantic assertions cannot bypass support.
        for key in declared:
            try:
                Ref.parse(key)
            except ValueError:
                continue
            raise ValidationError("semantic assertions require structural compilation", key=key)
    result = list(spec.factors)
    if spec.ground_structure:
        expected = {v.key: v.domain for v in structural_variables(spec.context)}
        if any(domains.get(k) != v for k, v in expected.items()):
            raise ValidationError(
                "structural variables must preserve all declared candidate domains"
            )
        for key in declared.keys() - expected.keys():
            try:
                ref = Ref.parse(key)
            except ValueError:
                continue
            if (
                key in continuous
                and ref == Ref.event_time(ref.subject)
                and str(Ref.event_exists(ref.subject)) in expected
            ):
                if ref.subject in spec.decoding.get("fixed_endpoints", {}):
                    raise ValidationError("event cannot have fixed and modeled time", key=key)
                if continuous[key].unit != "UTC seconds":
                    raise ValidationError("semantic event time uses UTC seconds", key=key)
                if continuous[key].active_when != ((str(Ref.event_exists(ref.subject)), True),):
                    raise ValidationError("event time must activate with event existence", key=key)
                continue
            raise CapabilityError(
                "semantic variable is outside the admitted context/support", key=key
            )
        result.extend(
            cached(
                store,
                "structure.factors",
                (
                    spec.context.context_id,
                    spec.lifecycle_bindings,
                    spec.decoding,
                    spec.parameters.assumptions,
                ),
                lambda: _structure(spec),
            )
        )
    shared_priors = {}
    for contribution in contributions.values():
        for factor in contribution.factors:
            if factor.key.startswith("prior:"):
                key = factor.key.removeprefix("prior:")
                if key not in domains or factor.scope != (key,) or factor.family != "table":
                    raise ValidationError(
                        "shared finite prior must name its declared variable", key=factor.key
                    )
                if (
                    factor.log_values is None
                    or factor.log_values.shape != (len(domains[key]),)
                    or factor.log_offset != 0
                    or not np.isclose(np.exp(factor.log_values).sum(), 1, atol=1e-12, rtol=0)
                ):
                    raise ValidationError(
                        "shared prior must be normalized exactly once", key=factor.key
                    )
                previous = shared_priors.get(factor.key)
                if previous is not None and canonical_json(previous) != canonical_json(factor):
                    raise ValidationError("conflicting shared prior", key=factor.key)
                shared_priors[factor.key] = factor
    for v in variables:
        checkpoint(control, "compile.prior", key=v.key)
        if v.key in continuous:
            result.append(
                FactorSpec(
                    "prior:" + v.key,
                    (v.key,),
                    "linear_gaussian",
                    {"coefficients": {v.key: 1.0}, "observed": v.prior_mean, "sd": v.prior_sd},
                )
            )
            continue
        if "prior:" + v.key in shared_priors:
            factor = shared_priors["prior:" + v.key]
            if v.key in spec.parameters.priors and not np.allclose(
                np.exp(factor.log_values), spec.parameters.priors[v.key], atol=1e-12, rtol=0
            ):
                raise ValidationError("shared prior conflicts with supplied prior", key=v.key)
            result.append(factor)
            continue
        if spec.ground_structure and v.key.startswith("event_type("):
            if v.key in spec.parameters.priors:
                raise CapabilityError(
                    "active event-type prior is uniform; custom type priors are not admitted",
                    key=v.key,
                )
            continue  # type_presence owns the conditional active-type prior
        if len(v.domain) == 1:
            probabilities = (1.0,)
        else:
            probabilities = spec.parameters.priors.get(v.key)
            if probabilities is None:
                raise ValidationError("missing explicit prior", key=v.key)
        p = np.asarray(probabilities, dtype=float)
        if (
            p.shape != (len(v.domain),)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)
        ):
            raise ValidationError("prior must be a normalized finite domain vector", key=v.key)
        with np.errstate(divide="ignore"):
            result.append(FactorSpec("prior:" + v.key, (v.key,), log_values=np.log(p)))
    groups = defaultdict(list)
    for o in spec.evidence.observations:
        if not set(o.scope) <= declared.keys():
            raise ValidationError("observation outside candidate support", key=o.observation_id)
        groups[o.information_id].append(o)
    for group_id, observations in sorted(groups.items()):
        checkpoint(control, "compile.group", key=group_id)
        first = observations[0]
        semantics = lambda o: canonical_json(
            (
                o.source_id,
                o.family,
                o.producer_version,
                o.channel,
                o.scope,
                o.value,
                o.interpretation,
                o.applicability,
            )
        )
        if any(semantics(o) != semantics(first) for o in observations[1:]):
            raise CapabilityError(
                "dependent reports require one explicitly joint observation channel", key=group_id
            )
        first = replace(
            first, evidence_ids=tuple(sorted({r for o in observations for r in o.evidence_ids}))
        )
        channel = channels.get(first.channel)
        if channel is None:
            raise CapabilityError("unsupported observation channel", key=first.channel)
        if first.observation_id in contributions:
            result.extend(
                replace(f, evidence_ids=first.evidence_ids) if f.role == "observation" else f
                for f in contributions[first.observation_id].factors
                if not f.key.startswith("prior:")
            )
        else:
            parameters = spec.parameters.for_observation(first)
            token = extension_key(channel)
            result.extend(
                cached(
                    store if token else None,
                    "channel.factors",
                    (token, first, parameters, domains),
                    lambda channel=channel, first=first, parameters=parameters: channel.factors(
                        first, parameters, domains
                    ),
                    slot=group_id,
                )
            )
    result = tuple(sorted(result, key=lambda f: f.key))
    if len({f.key for f in result}) != len(result):
        raise ValidationError("duplicate factor identity")
    for f in result:
        checkpoint(
            control,
            "compile.factor",
            key=f.key,
            allocation_bytes=f.log_values.nbytes if f.log_values is not None else 0,
        )
        if not set(f.scope) <= declared.keys():
            raise ValidationError("factor outside candidate support", key=f.key)
        if f.family not in kernels or kernels[f.family].version != f.version:
            raise CapabilityError("missing factor kernel/version", key=f.key)
        if f.family == "table":
            if not set(f.scope) <= domains.keys():
                raise CapabilityError("table factors require finite variables", key=f.key)
            if f.log_values is None or f.log_values.shape != tuple(
                len(domains[k]) for k in f.scope
            ):
                raise ValidationError("table shape does not match ordered factor scope", key=f.key)
            if np.isneginf(f.log_values).all():
                raise IncompatibleModel("factor has empty support", key=f.key)
        if not set(f.evidence_ids) <= {r.revision_id for r in spec.evidence.snapshot.effective}:
            raise ValidationError("factor references ineffective evidence", key=f.key)
        if f.family in ("linear_gaussian", "temporal_order"):
            from .hybrid_kernels import validate_hybrid_factor

            validate_hybrid_factor(f, declared)
    manifest = freeze(
        {
            "contract": "hybrid-v1" if continuous else "finite-v1",
            "context_id": spec.context.context_id,
            "evidence_id": spec.evidence.snapshot.snapshot_id,
            "knowledge_cutoff": spec.evidence.snapshot.as_of,
            "interpretation_id": spec.evidence.interpretation_id,
            "parameter_id": spec.parameters.parameter_id,
            "parameters": spec.parameters,
            "scope": spec.scope,
            "lifecycle_bindings": spec.lifecycle_bindings,
            "evidence_issues": tuple(
                i for i in spec.evidence.issues if i.status not in ("duplicate", "out_of_scope")
            ),
            "kernels": {k: kernels[k].version for k in sorted({f.family for f in result})},
            "channels": {
                k: channels[k].version
                for k in sorted({o.channel for o in spec.evidence.observations})
            },
            "dependencies": {f.key: f.evidence_ids for f in result},
            "factor_scopes": {f.key: f.scope for f in result},
            "validation": "finite domains, parameters, scopes and kernel versions checked",
        }
    )
    decoding = {
        **spec.decoding,
        "contract": "hybrid-assertions-v1" if continuous else "finite-assertions-v1",
        "object_types": {o.id: o.object_type for o in spec.context.definition["objects"]},
        "candidate_support": "declared context only; missing candidates are not negative assertions",
    }
    if continuous:
        decoding["continuous"] = continuous
        manifest = freeze(
            {
                **manifest,
                "variable_domains": domains,
                "constraint_dependencies": {f.key: f.key for f in result if f.role == "support"},
                "parameter_dependencies": {f.key: spec.parameters.parameter_id for f in result},
            }
        )
    if contributions:
        manifest = freeze(
            {
                **manifest,
                "parameter_dependencies": {f.key: spec.parameters.parameter_id for f in result},
                "shared_variables": tuple(
                    sorted(
                        set(declared)
                        - {v.key for v in spec.variables or structural_variables(spec.context)}
                    )
                ),
            }
        )
    model = CompiledModel(variables, result, manifest, decoding, MappingProxyType(kernels))
    from .dependencies import dependency_index

    if store is not None:
        dependency_index(model, spec, contributions)
    checkpoint(control, "compile.complete")
    return model
