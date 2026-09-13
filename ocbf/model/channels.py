"""Concrete reusable channels; producer field names do not belong here."""

from dataclasses import dataclass

import numpy as np

from ocbf.errors import ValidationError
from ocbf.reliability.channels import AssociationParameters, TimestampParameters

from .contracts import ChannelContribution
from .spec import ContinuousVariableSpec, FactorSpec, VariableSpec


@dataclass(frozen=True)
class AssociationChannel:
    name: str = "association_mode"
    version: str = "1"

    def contribute(self, observation, parameters, variables):
        if not isinstance(parameters, AssociationParameters):
            raise ValidationError(
                "association channel requires AssociationParameters", key=observation.observation_id
            )
        if len(observation.scope) != 1 or observation.applicability:
            raise ValidationError("association report requires one declared candidate variable")
        target = variables[observation.scope[0]]
        if not isinstance(target, VariableSpec):
            raise ValidationError("association channel requires finite candidates", key=target.key)
        reports = target.domain
        if observation.value not in reports:
            raise ValidationError("report outside declared association labels")
        matrices = np.asarray(parameters.matrices)
        if matrices.shape != (len(parameters.modes), len(reports), len(reports)):
            raise ValidationError("association matrix does not match candidate/report labels")
        mode = VariableSpec(parameters.mode_key, parameters.modes)
        with np.errstate(divide="ignore"):
            table = np.log(matrices[:, :, reports.index(observation.value)]).T
            prior = np.log(parameters.mode_prior)
        return ChannelContribution(
            (
                FactorSpec("prior:" + mode.key, (mode.key,), log_values=prior),
                FactorSpec(
                    "observation:" + observation.observation_id,
                    (target.key, mode.key),
                    log_values=table,
                    role="observation",
                    evidence_ids=observation.evidence_ids,
                ),
            ),
            (mode,),
        )


@dataclass(frozen=True)
class TimestampChannel:
    name: str = "timestamp_gaussian"
    version: str = "1"

    def contribute(self, observation, parameters, variables):
        if not isinstance(parameters, TimestampParameters):
            raise ValidationError(
                "timestamp channel requires TimestampParameters", key=observation.observation_id
            )
        if len(observation.scope) != 1 or observation.applicability:
            raise ValidationError("timestamp channel requires one bound time variable")
        target = variables[observation.scope[0]]
        if not isinstance(target, ContinuousVariableSpec) or parameters.unit != target.unit:
            raise ValidationError("timestamp channel/domain units disagree")
        if target.active_when and parameters.inactive_log_likelihood is None:
            raise ValidationError("inactive report likelihood must be supplied explicitly")
        coefficients = {target.key: 1.0}
        extra = ()
        if parameters.clock_key:
            if parameters.clock_key == target.key:
                raise ValidationError("clock bias must have its own latent key", key=target.key)
            clock = ContinuousVariableSpec(
                parameters.clock_key, 0.0, parameters.clock_sd, target.unit, parameters.origin
            )
            coefficients[clock.key] = 1.0
            extra = (clock,)
        scope = tuple(coefficients) + tuple(k for k, _ in target.active_when)
        factor = FactorSpec(
            "observation:" + observation.observation_id,
            scope,
            "linear_gaussian",
            {
                "coefficients": coefficients,
                "observed": float(observation.value) - parameters.bias,
                "sd": parameters.sd,
                "active_when": target.active_when,
                "inactive_log_likelihood": parameters.inactive_log_likelihood or 0.0,
            },
            role="observation",
            evidence_ids=observation.evidence_ids,
        )
        return ChannelContribution((factor,), extra)


@dataclass(frozen=True)
class FactorOnlyChannelAdapter:
    """Explicit adapter for a v1 factor-only extension."""

    channel: object

    @property
    def name(self):
        return self.channel.name

    @property
    def version(self):
        return self.channel.version

    def contribute(self, observation, parameters, variables):
        domains = {k: v.domain for k, v in variables.items() if hasattr(v, "domain")}
        return ChannelContribution(self.channel.factors(observation, parameters, domains))
