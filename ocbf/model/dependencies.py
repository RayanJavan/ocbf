"""Factor dependency sidecars and support identities, independent of solver layout."""

import numpy as np

from ocbf.runtime.cache import artifact_key, extension_key
from ocbf.runtime.contracts import DependencyIndex


def dependency_index(model, spec=None, contributions=None):
    existing = model._dependency_index
    if existing is not None:
        return existing
    kernels = {name: extension_key(kernel) for name, kernel in model._kernels.items()}
    variables = {v.key: v for v in model.variables}
    domains = {
        v.key: (v.domain, v.unit, v.measure)
        if hasattr(v, "domain")
        else ("real", v.unit, v.measure, v.active_when)
        for v in model.variables
    }
    structure = artifact_key(
        domains,
        tuple((f.key, f.family, f.version, f.scope, f.role) for f in model.factors),
        kernels,
    )
    tokens = {
        "interpretation": model.manifest.get("interpretation_id"),
        "decoding": artifact_key(model.decoding),
        "context": model.manifest.get("context_id"),
        "scope": artifact_key(model.manifest.get("scope")),
        "assumptions": artifact_key(model.manifest.get("parameters")),
    }
    parents, numerics, supports = {}, {}, []
    known_support = all(
        kernels[name] is not None and type(kernel).__module__.startswith("ocbf.")
        for name, kernel in model._kernels.items()
    )
    channel_parents = {}
    if spec is not None:
        tokens["assumptions"] = artifact_key(spec.parameters.assumptions)
        for observation in spec.evidence.observations:
            group = "observation:" + observation.observation_id
            parameter = "channel:" + artifact_key(
                (
                    observation.source_id,
                    observation.family,
                    observation.channel,
                    observation.producer_version,
                )
            )
            tokens[group] = artifact_key(observation)
            tokens[parameter] = artifact_key(spec.parameters.for_observation(observation))
            contribution = (contributions or {}).get(observation.observation_id)
            keys = [f.key for f in contribution.factors] if contribution else [group]
            if contribution:
                keys.extend("prior:" + v.key for v in contribution.variables)
            for key in keys:
                channel_parents.setdefault(key, set()).update((group, parameter))
        for key, values in spec.parameters.priors.items():
            tokens["prior-value:" + key] = artifact_key(values)
    for factor in model.factors:
        variable_values = {k: variables[k] for k in factor.scope}
        kernel = kernels[factor.family]
        numerics[factor.key] = artifact_key(factor, variable_values, kernel)
        deps = {"variable:" + k for k in factor.scope}
        deps.update("evidence:" + k for k in factor.evidence_ids)
        deps.update(channel_parents.get(factor.key, ()))
        if factor.key.startswith("prior:"):
            deps.add("prior-value:" + factor.key.removeprefix("prior:"))
        if factor.role == "support":
            deps.add("constraint:" + factor.key)
        if kernel is None:
            # This extension has not declared a reproducible execution implementation.
            deps.update(("whole-model:" + model.model_id, "interpretation", "assumptions"))
            known_support = False
        parents[factor.key] = tuple(sorted(deps))
        if factor.family == "table":
            supports.append((factor.key, factor.scope, np.isneginf(factor.log_values)))
        elif factor.family == "linear_gaussian" and type(
            model._kernels[factor.family]
        ).__module__.startswith("ocbf."):
            p = factor.parameters
            supports.append(
                (
                    factor.key,
                    p.get("active_when", ()),
                    p.get("inactive_log_likelihood", 0) == -np.inf,
                )
            )
        elif type(model._kernels[factor.family]).__module__.startswith("ocbf."):
            # Conservatively treat all parameters of other built-in rules as support.
            supports.append((factor.key, factor.family, factor.parameters, factor.role))
        else:
            known_support = False
    tokens.update({"variable:" + k: artifact_key(v) for k, v in variables.items()})
    for f in model.factors:
        tokens.update({"evidence:" + k: k for k in f.evidence_ids})
        if f.role == "support":
            tokens["constraint:" + f.key] = numerics[f.key]
    tokens["execution"] = artifact_key(model.model_id, numerics)
    tokens["reusable"] = all(token is not None for token in kernels.values())
    index = DependencyIndex(
        structure,
        artifact_key(domains, supports, kernels) if known_support else None,
        numerics,
        parents,
        tokens,
    )
    object.__setattr__(model, "_dependency_index", index)
    return index


def reusable_model(model):
    return dependency_index(model).tokens["reusable"]


def execution_identity(model):
    """Full numerical inputs, including explicitly declared extension configuration."""
    return dependency_index(model).tokens["execution"]
