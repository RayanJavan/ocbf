"""Portable manifest/NPY directories; executable code and native caches are excluded."""

import hashlib
import json
import math
import os
import uuid
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path

import numpy as np

from ocbf._values import freeze
from ocbf.errors import OCBFError, ValidationError
from ocbf.runtime.control import checkpoint, operation

from .bundle import _decode, _encode, value_types

FORMAT = "ocbf-array-bundle-v1"


def _digest(path, control, stage="export.checksum"):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            checkpoint(control, stage)
            digest.update(block)
    return digest.hexdigest()


def write_artifact(path, value, *, types=None, control=None):
    """Atomically publish a new directory, returning its execution assessment.

    An existing destination is never overwritten. Interrupted staging directories have
    no published manifest and remain named .partial for caller inspection/cleanup.
    """
    path = Path(path).resolve()
    with operation("export", control=control) as run:
        if path.exists():
            raise ValidationError("artifact destination already exists", key=str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
        staging.mkdir()
        entries = {}

        def write_array(array):
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise ValidationError("artifact arrays require non-object scalar dtypes")
            checkpoint(
                control, "export.array", allocation_bytes=array.nbytes, completed=len(entries)
            )
            name = f"array-{len(entries):06d}.npy"
            destination = staging / name
            with destination.open("xb") as stream:
                contiguous = np.ascontiguousarray(array)
                np.lib.format.write_array_header_1_0(
                    stream,
                    {
                        "descr": np.lib.format.dtype_to_descr(array.dtype),
                        "fortran_order": False,
                        "shape": array.shape,
                    },
                )
                if contiguous.nbytes:
                    payload = memoryview(contiguous).cast("B")
                    for offset in range(0, len(payload), 1024 * 1024):
                        checkpoint(
                            control, "export.array_chunk", completed=offset, total=len(payload)
                        )
                        stream.write(payload[offset : offset + 1024 * 1024])
            entries[name] = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": array.nbytes,
                "file_bytes": destination.stat().st_size,
                "sha256": _digest(destination, control),
            }
            return name

        encoded = _encode(
            value, value_types() if types is None else dict(types), arrays=write_array
        )
        checkpoint(control, "export.manifest")
        assessment = run.finish(
            arrays=len(entries), array_bytes=sum(e["nbytes"] for e in entries.values())
        )
        manifest = {
            "format": FORMAT,
            "arrays": entries,
            "value": encoded,
            "execution": _encode(assessment, value_types()),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        checkpoint(control, "export.publish")
        # Rename within one parent/filesystem; there is no readable completed artifact
        # until all payloads and their manifest have been written successfully.
        os.rename(staging, path)
        return assessment


def read_artifact(
    path,
    *,
    types=None,
    max_bytes=256 * 1024 * 1024,
    max_manifest_bytes=16 * 1024 * 1024,
    control=None,
):
    """Validate byte budgets, paths, NPY headers and checksums before array allocation."""
    root = Path(path).resolve()
    with operation("import", control=control):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValidationError("artifact read budget must be positive")
        if type(max_manifest_bytes) is not int or max_manifest_bytes < 1:
            raise ValidationError("manifest read budget must be positive")
        manifest_path = root / "manifest.json"
        if manifest_path.resolve().parent != root:
            raise ValidationError("manifest path escapes artifact directory")
        if manifest_path.stat().st_size > max_manifest_bytes:
            raise ValidationError("artifact manifest exceeds read budget")
        try:
            document = json.loads(
                manifest_path.read_text(encoding="utf-8"),
                parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
            )
            if (
                set(document) != {"format", "arrays", "value", "execution"}
                or document["format"] != FORMAT
            ):
                raise ValidationError("unsupported artifact manifest version")
            entries = document["arrays"]
            if not isinstance(entries, dict):
                raise ValidationError("artifact arrays must be a manifest mapping")
            references = []
            registry = value_types() if types is None else dict(types)

            def scan(value, registry):
                if not isinstance(value, dict):
                    if isinstance(value, list):
                        raise ValidationError("untagged artifact sequence")
                    return
                if set(value) == {"npy"}:
                    if not isinstance(value["npy"], str):
                        raise ValidationError("array reference must be a name")
                    references.append(value["npy"])
                elif set(value) == {"map"}:
                    if not isinstance(value["map"], dict):
                        raise ValidationError("invalid artifact mapping")
                    for item in value["map"].values():
                        scan(item, registry)
                elif set(value) == {"sequence"}:
                    if not isinstance(value["sequence"], list):
                        raise ValidationError("invalid artifact sequence")
                    for item in value["sequence"]:
                        scan(item, registry)
                elif set(value) == {"type", "fields"}:
                    cls = registry.get(value["type"])
                    if (
                        cls is None
                        or not is_dataclass(cls)
                        or not isinstance(value["fields"], dict)
                    ):
                        raise ValidationError("unregistered artifact value type")
                    declared = {f.name: f for f in fields(cls) if f.init}
                    required = {
                        name
                        for name, f in declared.items()
                        if f.default is MISSING and f.default_factory is MISSING
                    }
                    if not required <= value["fields"].keys() <= declared.keys():
                        raise ValidationError(
                            "artifact value fields do not match registered schema"
                        )
                    for item in value["fields"].values():
                        scan(item, registry)
                elif set(value) in ({"utc"}, {"float"}, {"enum", "value"}):
                    _decode(value, registry)
                else:
                    raise ValidationError(
                        "unknown artifact encoding; arrays require NPY references"
                    )

            scan(document["value"], registry)
            if len(references) != len(set(references)) or set(references) != set(entries):
                raise ValidationError("each array payload must be referenced exactly once")
            # Execution records contain only ordinary neutral metadata, never arrays.
            scan(document["execution"], value_types())
            if len(references) != len(entries):
                raise ValidationError("execution metadata cannot contain array payloads")
            total = 0
            for name, entry in entries.items():
                checkpoint(control, "import.header", key=name)
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or not name.startswith("array-")
                    or not name.endswith(".npy")
                    or (root / name).resolve().parent != root
                ):
                    raise ValidationError("array path escapes artifact directory")
                if set(entry) != {"dtype", "shape", "nbytes", "file_bytes", "sha256"}:
                    raise ValidationError("invalid array manifest fields")
                shape, dtype = entry["shape"], np.dtype(entry["dtype"])
                if (
                    not isinstance(shape, list)
                    or len(shape) > 32
                    or any(type(n) is not int or n < 0 for n in shape)
                    or dtype.hasobject
                    or dtype.fields is not None
                ):
                    raise ValidationError("invalid array shape/dtype")
                expected = math.prod(shape) * dtype.itemsize
                if type(entry["nbytes"]) is not int or entry["nbytes"] != expected:
                    raise ValidationError("array size disagrees with dtype/shape")
                total += expected
                if total > max_bytes:
                    raise ValidationError("array payloads exceed read budget")
                file = root / name
                if (
                    file.stat().st_size != entry["file_bytes"]
                    or file.stat().st_size > expected + 10000
                ):
                    raise ValidationError("array file size disagrees with manifest")
                with file.open("rb") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        actual_shape, _, actual_dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        actual_shape, _, actual_dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValidationError("unsupported NPY version")
                    if (
                        actual_shape != tuple(shape)
                        or actual_dtype != dtype
                        or stream.tell() + expected != file.stat().st_size
                    ):
                        raise ValidationError("NPY header disagrees with manifest")
                if _digest(file, control, "import.checksum") != entry["sha256"]:
                    raise ValidationError("array checksum mismatch", key=name)

            def load_array(name):
                entry = entries[name]
                checkpoint(control, "import.array", key=name, allocation_bytes=3 * entry["nbytes"])
                return freeze(np.load(root / name, allow_pickle=False))

            _decode(document["execution"], value_types())
            return _decode(document["value"], registry, arrays=load_array)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, OCBFError):
                raise
            raise ValidationError(f"invalid array artifact: {exc}") from exc
