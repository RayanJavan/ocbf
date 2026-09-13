"""Bounded session-local LRU and versioned computational fingerprints."""

import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from ocbf._values import canonical_json, freeze
from ocbf.errors import ValidationError


def artifact_key(*values):
    """Typed streaming digest; array bytes avoid allocating enormous JSON lists.

    This is an execution-artifact identity, never a replacement for scientific IDs.
    """
    digest = sha256(b"ocbf-artifact-v2")

    def add(value):
        scalar_type = type(value)
        if value is None:
            digest.update(b"N")
        elif scalar_type is bool:
            digest.update(b"B1" if value else b"B0")
        elif scalar_type in (str, int, float):
            if scalar_type is str:
                tag, encoded = b"S", value.encode("utf-8")
            elif scalar_type is int:
                tag, encoded = b"I", str(value).encode("ascii")
            else:
                tag, encoded = b"F", value.hex().encode("ascii")
            digest.update(tag + str(len(encoded)).encode("ascii") + b":" + encoded)
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValidationError("object array cannot identify a reusable artifact")
            digest.update(b"array:")
            add((value.dtype.str, value.shape))
            contiguous = np.ascontiguousarray(value)
            if contiguous.nbytes:
                digest.update(memoryview(contiguous).cast("B"))
        elif isinstance(value, Mapping):
            digest.update(b"map[")
            for key, item in sorted(value.items()):
                add(key)
                add(item)
            digest.update(b"]")
        elif isinstance(value, (tuple, list)):
            digest.update(b"sequence[")
            for item in value:
                add(item)
            digest.update(b"]")
        elif is_dataclass(value):
            digest.update(b"dataclass:")
            add((type(value).__module__, type(value).__qualname__))
            add(tuple((f.name, getattr(value, f.name)) for f in fields(value) if f.init))
        else:
            encoded = canonical_json(value).encode()
            digest.update(str(len(encoded)).encode() + b":" + encoded)

    add(values)
    return "artifact:" + digest.hexdigest()


def extension_key(extension):
    """Explicit opt-in for external code; never infer hidden closure/configuration state."""
    try:
        cls = type(extension)
        if cls.__module__.startswith("ocbf.") and is_dataclass(extension):
            return artifact_key(extension)
        key = getattr(extension, "reuse_key", None)
        if key is not None:
            return artifact_key(cls.__module__, cls.__qualname__, key)
    except (ValidationError, TypeError, ValueError):
        pass
    return None


def owned_size(value):
    """Conservative retained-byte charge, including containers and array buffers."""
    if isinstance(value, np.ndarray):
        return sys.getsizeof(value) + value.nbytes
    if isinstance(value, Mapping):
        return 128 + sum(96 + owned_size(k) + owned_size(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(owned_size(v) for v in value)
    if is_dataclass(value):
        return 128 + sum(owned_size(getattr(value, f.name)) for f in fields(value))
    return sys.getsizeof(value)


class MemoryArtifactStore:
    """Single-caller LRU. No globals, implicit disk state, or ownership of returned values."""

    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValidationError("cache byte budget must be a nonnegative integer")
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._slots = {}
        self._bytes = 0
        self._peak = 0
        self._counts = dict.fromkeys(("hits", "misses", "invalidations", "evictions", "refused"), 0)

    @property
    def stats(self):
        return MappingProxyType(
            {
                **self._counts,
                "bytes": self._bytes,
                "peak_bytes": self._peak,
                "entries": len(self._entries),
                "max_bytes": self.max_bytes,
            }
        )

    def get(self, namespace, key, *, slot=None):
        address = (namespace, key)
        entry = self._entries.get(address)
        if entry is None:
            self._counts["misses"] += 1
            if slot is not None and (namespace, slot) in self._slots:
                self._counts["invalidations"] += 1
            return None
        self._entries.move_to_end(address)
        self._counts["hits"] += 1
        return entry[0]

    def _remove(self, address):
        _, charge, slot = self._entries.pop(address)
        self._bytes -= charge
        if slot is not None and self._slots.get((address[0], slot)) == address:
            del self._slots[(address[0], slot)]

    def put(self, namespace, key, value, *, slot=None):
        address = (namespace, key)
        # Refuse large values before copying them into immutable ownership.
        try:
            charge = owned_size(value) + owned_size(address) + owned_size(slot) + 256
        except (RecursionError, TypeError, ValueError):
            self._counts["refused"] += 1
            return False
        if charge > self.max_bytes:
            self._counts["refused"] += 1
            return False
        try:
            owned = freeze(value)
        except ValidationError:
            self._counts["refused"] += 1
            return False
        charge = owned_size(owned) + owned_size(address) + owned_size(slot) + 256
        if charge > self.max_bytes:
            self._counts["refused"] += 1
            return False
        if address in self._entries:
            self._remove(address)
        while self._entries and self._bytes + charge > self.max_bytes:
            self._remove(next(iter(self._entries)))
            self._counts["evictions"] += 1
        self._entries[address] = (owned, charge, slot)
        self._bytes += charge
        self._peak = max(self._peak, self._bytes)
        if slot is not None:
            self._slots[(namespace, slot)] = address
        return True

    def clear(self):
        self._entries.clear()
        self._slots.clear()
        self._bytes = 0


def cached(store, namespace, dependencies, compute, *, slot=None):
    if store is None:
        return compute()
    key = artifact_key(dependencies)
    value = store.get(namespace, key, slot=slot)
    if value is not None:
        return value
    value = compute()
    store.put(namespace, key, value, slot=slot)
    return value
