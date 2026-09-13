"""Versioned factor and observation-channel protocols with explicit local registries."""

import math
from dataclasses import dataclass

import numpy as np

from ocbf.errors import CapabilityError, ValidationError

from .contracts import FactorKernel, ObservationChannel
from .spec import FactorSpec

__all__ = [
    "ConjunctionChannel",
    "FactorKernel",
    "ObservationChannel",
    "PointChannel",
    "RuleKernel",
    "TableKernel",
    "builtin_channels",
    "builtin_kernels",
]


@dataclass(frozen=True)
class TableKernel:
    name: str = "table"
    version: str = "1"

    def log_value(self, factor, indices, domains):
        return float(factor.log_values[indices]) + factor.log_offset


@dataclass(frozen=True)
class RuleKernel:
    name: str
    version: str = "1"

    def log_values(self, factor, state, domains):
        v = [state[k] for k in factor.scope]
        p = factor.parameters
        if self.name == "implication":
            valid = np.logical_or(~v[0], v[1])
        elif self.name == "type_presence":
            active = v[1] != "__inactive__"
            return (
                np.where(
                    v[0] == active,
                    np.where(v[0], -math.log(len(domains[factor.scope[1]]) - 1), 0),
                    -np.inf,
                )
                + factor.log_offset
            )
        elif self.name == "type_gate":
            valid = np.logical_or(~v[0], np.isin(v[1], p["allowed"]))
        elif self.name == "cardinality":
            count = sum(x.astype(int) for x in v[1:])
            valid = (v[0] == "__inactive__") & (count == 0)
            for label in domains[factor.scope[0]]:
                if label == "__inactive__":
                    continue
                bounds = p["bounds"].get(label)
                allowed = (
                    count == 0
                    if bounds is None
                    else (
                        (count >= bounds[0]) & (True if bounds[1] is None else count <= bounds[1])
                    )
                )
                valid = valid | ((v[0] == label) & allowed)
        elif self.name == "precedence":
            truth = np.asarray(True)
            for x in v:
                truth = truth & x
            valid = ~truth | (p["before"] <= p["after"])
        elif self.name == "count":
            count = sum(x.astype(int) for x in v)
            valid = (count >= p["lo"]) & (True if p["hi"] is None else count <= p["hi"])
        elif self.name == "conjunction_report":
            truth = np.asarray(True)
            for x, expected in zip(v, p["expected"], strict=True):
                truth = truth & (x == expected)
            probability = np.where(truth, p["sensitivity"], p["false_positive"])
            if not p["report"]:
                probability = 1 - probability
            with np.errstate(divide="ignore"):
                return np.log(probability) + factor.log_offset
        else:
            raise CapabilityError("unknown rule kernel", key=self.name)
        return np.where(valid, 0.0, p.get("penalty", -np.inf)) + factor.log_offset

    def log_value(self, factor, indices, domains):
        values = [domains[k][i] for k, i in zip(factor.scope, indices, strict=True)]
        p = factor.parameters
        if self.name == "implication":
            valid = not values[0] or bool(values[1])
        elif self.name == "type_presence":
            valid = bool(values[0]) == (values[1] != "__inactive__")
            # Conditional active-type prior, normalized separately for each existence state.
            if valid:
                return (
                    -math.log(len(domains[factor.scope[1]]) - 1) if values[0] else 0.0
                ) + factor.log_offset
        elif self.name == "type_gate":
            valid = not values[0] or values[1] in p["allowed"]
        elif self.name == "cardinality":
            event_type = values[0]
            count = sum(bool(v) for v in values[1:])
            bound = p["bounds"].get(event_type)
            valid = (
                count == 0
                if event_type == "__inactive__"
                else (
                    count == 0
                    if bound is None
                    else count >= bound[0] and (bound[1] is None or count <= bound[1])
                )
            )
        elif self.name == "precedence":
            valid = not all(values) or p["before"] <= p["after"]
        elif self.name == "count":
            count = sum(bool(v) for v in values)
            valid = count >= p["lo"] and (p["hi"] is None or count <= p["hi"])
        elif self.name == "conjunction_report":
            truth = all(v == expected for v, expected in zip(values, p["expected"], strict=True))
            probability = p["sensitivity"] if truth else p["false_positive"]
            if not p["report"]:
                probability = 1 - probability
            return (math.log(probability) if probability else -math.inf) + factor.log_offset
        else:
            raise CapabilityError("unknown rule kernel", key=self.name)
        return (0.0 if valid else float(p.get("penalty", -math.inf))) + factor.log_offset


@dataclass(frozen=True)
class PointChannel:
    name: str
    version: str = "1"

    def factors(self, observation, parameters, domains):
        if observation.applicability:
            raise CapabilityError(
                "point channel does not implement opportunity/applicability conditions",
                key=observation.observation_id,
            )
        if len(observation.scope) != 1:
            raise CapabilityError(
                "point channels have one variable; use a joint channel",
                key=observation.observation_id,
            )
        domain = domains[observation.scope[0]]
        value = observation.value
        if self.name == "binary":
            if any(type(v) is not bool for v in domain) or type(value) is not bool:
                raise ValidationError(
                    "binary channel requires Boolean report/domain", key=observation.observation_id
                )
            a, f = parameters.sensitivity, parameters.false_positive
            probabilities = np.array(
                [(a if truth else f) if value else (1 - a if truth else 1 - f) for truth in domain]
            )
        else:
            if value not in domain or ("__inactive__" in domain and not parameters.confusion):
                raise CapabilityError(
                    "inactive categorical states require an explicit confusion matrix",
                    key=observation.observation_id,
                )
            k, hit = len(domain), domain.index(value)
            if parameters.confusion:
                matrix = np.asarray(parameters.confusion)
                if matrix.shape != (k, k):
                    raise ValidationError(
                        "confusion matrix/domain mismatch", key=observation.observation_id
                    )
                probabilities = matrix[:, hit]
            elif k > 1:
                probabilities = np.full(k, (1 - parameters.hit_rate) / (k - 1))
                probabilities[hit] = parameters.hit_rate
            else:
                probabilities = np.ones(1)
        with np.errstate(divide="ignore"):
            table = np.log(probabilities)
        return (
            FactorSpec(
                "observation:" + observation.observation_id,
                observation.scope,
                log_values=table,
                role="observation",
                evidence_ids=observation.evidence_ids,
            ),
        )


def builtin_kernels():
    from .hybrid_kernels import LinearGaussianKernel, TemporalOrderKernel

    return {
        k.name: k
        for k in [
            TableKernel(),
            LinearGaussianKernel(),
            TemporalOrderKernel(),
            *(
                RuleKernel(n)
                for n in (
                    "implication",
                    "type_presence",
                    "type_gate",
                    "cardinality",
                    "precedence",
                    "count",
                    "conjunction_report",
                )
            ),
        ]
    }


@dataclass(frozen=True)
class ConjunctionChannel:
    """One report about a joint proposition, including known deterministic descendants.

    Sensitivity/false-positive rates describe this whole report. They are never multiplied
    once per field. ``expected`` follows the explicitly ordered observation scope.
    """

    name: str = "conjunction"
    version: str = "1"

    def factors(self, observation, parameters, domains):
        if set(observation.applicability) != {"expected"}:
            raise CapabilityError("conjunction channel admits only its ordered expected states")
        expected = observation.applicability.get("expected", ())
        if type(observation.value) is not bool or len(expected) != len(observation.scope):
            raise ValidationError("joint report needs a Boolean value and ordered expected states")
        if any(v not in domains[k] for k, v in zip(observation.scope, expected, strict=True)):
            raise ValidationError("joint report names a state outside its domain")
        return (
            FactorSpec(
                "observation:" + observation.observation_id,
                observation.scope,
                "conjunction_report",
                {
                    "expected": expected,
                    "report": observation.value,
                    "sensitivity": parameters.sensitivity,
                    "false_positive": parameters.false_positive,
                },
                role="observation",
                evidence_ids=observation.evidence_ids,
            ),
        )


def builtin_channels():
    from .channels import AssociationChannel, TimestampChannel

    return {
        **{n: PointChannel(n) for n in ("binary", "categorical")},
        "conjunction": ConjunctionChannel(),
        "association_mode": AssociationChannel(),
        "timestamp_gaussian": TimestampChannel(),
    }
