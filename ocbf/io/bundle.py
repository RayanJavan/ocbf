"""Allowlisted JSON codec for reproducible finite/hybrid inputs and qualified results.

Only declared OCBF value types are reconstructed. Payload mappings are tagged, so their
contents cannot be confused with codec instructions. Extension types require an explicit
caller-supplied registry; loading never imports a class named by an input document.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from ocbf._values import freeze, utc
from ocbf.errors import ValidationError

FORMAT = "ocbf-neutral-bundle-v1"


def value_types():
    from ocbf.api import SensitivityResult
    from ocbf.belief.estimates import Estimate, QueryResults
    from ocbf.belief.posterior import InferenceResult, JointDrawSet, JointTable, QueryRequirements
    from ocbf.belief.samples import SamplePosterior
    from ocbf.evidence.observations import InterpretedEvidence, Observation
    from ocbf.evidence.records import AdmissionIssue, EvidenceAction, EvidenceRecord
    from ocbf.evidence.snapshots import EvidenceSnapshot
    from ocbf.inference.adapters.hybrid import HybridPosterior
    from ocbf.inference.conditional import ConditionedPosterior, GaussianComponent
    from ocbf.inference.contracts import ExecutionPlan, InferencePolicy, Proposal, SamplingConfig
    from ocbf.inference.elimination import (
        EliminationBucket,
        EliminationPosterior,
        EliminationSchedule,
        LogTable,
    )
    from ocbf.inference.warm_start import WarmStart
    from ocbf.model.contracts import ChannelContribution
    from ocbf.model.decoding import DecodedHistory
    from ocbf.model.reductions import ReductionCodec
    from ocbf.model.spec import (
        ContinuousVariableSpec,
        FactorSpec,
        LifecycleBinding,
        ModelSpec,
        VariableSpec,
    )
    from ocbf.queries.expressions import (
        ConditionInterval,
        Endpoint,
        ExecutionProjection,
        QueryBundle,
        QuerySpec,
    )
    from ocbf.reliability.channels import AssociationParameters, TimestampParameters
    from ocbf.reliability.config import (
        ChannelKey,
        ChannelValues,
        ParameterSet,
        ResolvedChannel,
        TrustRule,
    )
    from ocbf.runtime.contracts import DependencyIndex, ExecutionAssessment
    from ocbf.schema.constraints import ConstraintClass, ConstraintSpec, Strength
    from ocbf.schema.core import (
        AttributeKind,
        AttributeSpec,
        E2OQualifier,
        EventType,
        Lifecycle,
        Multiplicity,
        O2OQualifier,
        ObjectType,
    )
    from ocbf.universe.context import SemanticContext
    from ocbf.universe.core import EventCandidate, ObjectRecord

    return {
        cls.__name__: cls
        for cls in (
            DependencyIndex,
            WarmStart,
            ExecutionAssessment,
            EliminationBucket,
            EliminationSchedule,
            ContinuousVariableSpec,
            ChannelContribution,
            ReductionCodec,
            AssociationParameters,
            TimestampParameters,
            SamplingConfig,
            Proposal,
            GaussianComponent,
            ConditionedPosterior,
            HybridPosterior,
            JointDrawSet,
            SamplePosterior,
            AttributeKind,
            AttributeSpec,
            EventType,
            ObjectType,
            Multiplicity,
            E2OQualifier,
            O2OQualifier,
            Lifecycle,
            ConstraintClass,
            Strength,
            ConstraintSpec,
            EventCandidate,
            ObjectRecord,
            SemanticContext,
            EvidenceAction,
            EvidenceRecord,
            AdmissionIssue,
            EvidenceSnapshot,
            Observation,
            InterpretedEvidence,
            ChannelKey,
            ChannelValues,
            ResolvedChannel,
            ParameterSet,
            TrustRule,
            VariableSpec,
            DecodedHistory,
            FactorSpec,
            LifecycleBinding,
            ModelSpec,
            InferencePolicy,
            ExecutionPlan,
            LogTable,
            EliminationPosterior,
            QueryRequirements,
            JointTable,
            InferenceResult,
            Estimate,
            QueryResults,
            Endpoint,
            ExecutionProjection,
            ConditionInterval,
            QuerySpec,
            QueryBundle,
            SensitivityResult,
        )
    }


def _encode(value, registry, *, arrays=None):
    if isinstance(value, Enum):
        return {"enum": _type_name(value, registry), "value": value.value}
    if isinstance(value, datetime):
        return {"utc": utc(value).isoformat()}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValidationError("object arrays are not interchangeable")
        if arrays is not None:
            return {"npy": arrays(value)}
        return {
            "array": _encode(value.tolist(), registry, arrays=arrays),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return _encode(value.item(), registry)
    if is_dataclass(value):
        return {
            "type": _type_name(value, registry),
            "fields": {
                f.name: _encode(getattr(value, f.name), registry, arrays=arrays)
                for f in fields(value)
                if f.init
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValidationError("bundle mappings require string keys")
        return {"map": {k: _encode(v, registry, arrays=arrays) for k, v in sorted(value.items())}}
    if isinstance(value, (tuple, list)):
        return {"sequence": [_encode(v, registry, arrays=arrays) for v in value]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValidationError("unsupported bundle value", key=type(value).__name__)


def _type_name(value, registry):
    for name, cls in registry.items():
        if type(value) is cls:
            return name
    raise ValidationError("type is not in the explicit bundle registry", key=type(value).__name__)


def _decode(value, registry, *, arrays=None):
    if not isinstance(value, dict):
        if isinstance(value, list):
            raise ValidationError("untagged bundle sequence")
        return value
    if set(value) == {"npy"} and arrays is not None:
        return arrays(value["npy"])
    if set(value) == {"map"}:
        return {k: _decode(v, registry, arrays=arrays) for k, v in value["map"].items()}
    if set(value) == {"sequence"}:
        return tuple(_decode(v, registry, arrays=arrays) for v in value["sequence"])
    if set(value) == {"utc"}:
        return utc(datetime.fromisoformat(value["utc"]))
    if set(value) == {"float"} and value["float"] in ("nan", "+inf", "-inf"):
        return float(value["float"])
    if set(value) == {"array", "dtype", "shape"}:
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValidationError("object arrays are not interchangeable")
        array = np.asarray(_decode(value["array"], registry, arrays=arrays), dtype=dtype)
        if array.size != math.prod(value["shape"]):
            raise ValidationError("array shape/content mismatch")
        return freeze(array.reshape(value["shape"]))
    if set(value) == {"enum", "value"}:
        cls = registry[value["enum"]]
        if not issubclass(cls, Enum):
            raise ValidationError("expected enum type")
        return cls(value["value"])
    if set(value) == {"type", "fields"}:
        cls = registry[value["type"]]
        if not is_dataclass(cls):
            raise ValidationError("expected value type")
        return cls(**{k: _decode(v, registry, arrays=arrays) for k, v in value["fields"].items()})
    raise ValidationError("unknown bundle encoding")


def dumps(value, *, types=None):
    registry = value_types() if types is None else dict(types)
    return (
        json.dumps(
            {"format": FORMAT, "value": _encode(value, registry)},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )


def loads(text, *, types=None):
    try:
        document = json.loads(text, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        if set(document) != {"format", "value"} or document["format"] != FORMAT:
            raise ValidationError("unsupported bundle version")
        return _decode(document["value"], value_types() if types is None else dict(types))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid neutral bundle: {exc}") from exc


def write_bundle(path, value, *, types=None):
    Path(path).write_text(dumps(value, types=types), encoding="utf-8")


def read_bundle(path, *, types=None, max_bytes=64 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValidationError("bundle exceeds configured read budget")
    return loads(path.read_text(encoding="utf-8"), types=types)
