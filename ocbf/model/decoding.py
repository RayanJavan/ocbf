"""Decode supported finite assignments without inventing timestamps or missing candidates."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ocbf._values import canonical_json, freeze
from ocbf.assertions import AssertionRef, Family
from ocbf.errors import CapabilityError, ValidationError


@dataclass(frozen=True)
class DecodedHistory:
    events: Mapping
    objects: Mapping
    e2o: tuple
    o2o: tuple
    latents: Mapping
    model_id: str

    def __post_init__(self):
        for name in ("events", "objects", "e2o", "o2o", "latents"):
            object.__setattr__(self, name, freeze(getattr(self, name)))


def decode_assignment(model, assignment):
    """Push forward one complete supported state to a bounded OCEL-like history.

    This is an identity codec, not a sampler or full OCEL 2.0 file serializer. Unavailable
    fixed endpoint times remain ``None``. Auxiliary finite latents remain separate.
    """
    if model.decoding.get("contract") not in ("finite-assertions-v1", "hybrid-assertions-v1"):
        raise CapabilityError("unsupported decoding contract")
    if set(assignment) != set(model.domains) | set(model.continuous):
        raise ValidationError("decoding requires exactly the compiled assignment scope")
    indices = {}
    for key in model.continuous:
        if not isinstance(assignment[key], (int, float)) or not math.isfinite(assignment[key]):
            raise ValidationError("continuous assignment must be finite", key=key)
    for key, domain in model.domains.items():
        labels = [canonical_json(v) for v in domain]
        value = canonical_json(assignment[key])
        if value not in labels:
            raise ValidationError("assignment state outside typed domain", key=key)
        indices[key] = labels.index(value)
    for factor in model.factors:
        value = model.factor_log_density(factor, assignment)
        if not math.isfinite(value):
            raise ValidationError("cannot decode an unsupported assignment", key=factor.key)
    events, objects, e2o, o2o, latents = {}, {}, [], [], {}
    refs = {}
    for key, value in assignment.items():
        try:
            refs[key] = AssertionRef.parse(key)
        except ValueError:
            latents[key] = value
    for key, ref in refs.items():
        value = assignment[key]
        if ref.family is Family.EVENT_EXISTS and value:
            time_key = str(AssertionRef.event_time(ref.subject))
            if time_key in model.continuous:
                from datetime import UTC, datetime

                time = datetime.fromtimestamp(assignment[time_key], tz=UTC)
                semantics = "joint modeled event time"
            else:
                time = model.decoding.get("fixed_endpoints", {}).get(ref.subject)
                semantics = "conditioned fixed report time; missing is unresolved"
            events[ref.subject] = {
                "type": assignment[str(AssertionRef.event_type(ref.subject))],
                "time": time,
                "time_semantics": semantics,
            }
        elif ref.family is Family.OBJECT_EXISTS and value:
            objects[ref.subject] = model.decoding["object_types"][ref.subject]
        elif ref.family is Family.E2O and value:
            e2o.append((ref.subject, ref.qualifier, ref.target))
        elif ref.family is Family.O2O and value:
            o2o.append((ref.subject, ref.qualifier, ref.target))
    return DecodedHistory(
        events, objects, tuple(sorted(e2o)), tuple(sorted(o2o)), latents, model.model_id
    )
