"""Owned immutable values and versioned, deterministic scientific fingerprints."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType

import numpy as np

from ocbf.errors import ValidationError


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("a timezone-aware datetime is required")
    return value.astimezone(UTC)


def freeze(value):
    """Copy nested values; arrays are backed by immutable bytes, not writable owners."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValidationError("object arrays are not portable immutable values")
        a = np.ascontiguousarray(value)
        return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(value.shape)
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValidationError("manifest mapping keys must be strings")
        return MappingProxyType({k: freeze(v) for k, v in sorted(value.items())})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze(v) for v in value), key=canonical_json))
    if isinstance(value, datetime):
        return utc(value)
    if isinstance(value, (str, bool, int, float, Enum)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        # A frozen dataclass may still own mutable fields. Copy those recursively.
        from dataclasses import replace

        return replace(
            value, **{f.name: freeze(getattr(value, f.name)) for f in fields(value) if f.init}
        )
    raise ValidationError(f"unsupported immutable value: {type(value).__name__}")


def portable(value):
    """Neutral JSON encoding; special floats are tagged, never emitted as invalid JSON."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": portable(value.tolist()),
        }
    if isinstance(value, np.generic):
        return portable(value.item())
    if is_dataclass(value):
        return {
            f.name: portable(getattr(value, f.name))
            for f in fields(value)
            if not f.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {k: portable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValidationError(f"unsupported serialization type: {type(value).__name__}")


def _canonical_value(value):
    # Type tags prevent payload strings/maps from colliding with clocks, enums, arrays,
    # dataclasses, or special numbers that happen to have the same display encoding.
    if isinstance(value, Enum):
        return ["enum", type(value).__module__, type(value).__qualname__, value.value]
    if isinstance(value, datetime):
        return ["utc", utc(value).isoformat(timespec="microseconds")]
    if isinstance(value, np.ndarray):
        return ["array", value.dtype.str, list(value.shape), _canonical_value(value.tolist())]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if is_dataclass(value):
        return [
            "value",
            type(value).__module__,
            type(value).__qualname__,
            [
                [f.name, _canonical_value(getattr(value, f.name))]
                for f in fields(value)
                if not f.name.startswith("_")
                and not (
                    f.metadata.get("identity_omit_default") and getattr(value, f.name) == f.default
                )
            ],
        ]
    if isinstance(value, Mapping):
        return ["map", [[k, _canonical_value(v)] for k, v in sorted(value.items())]]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_canonical_value(v) for v in value]]
    if isinstance(value, float):
        return ["float", value.hex()]  # exact numeric representation including signed zero
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    raise ValidationError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fingerprint(kind: str, value) -> str:
    encoded = canonical_json({"contract": "ocbf-v1", "kind": kind, "value": value})
    return f"{kind}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
