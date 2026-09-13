"""Caller-owned lifetime; scientific packages receive only narrow ports."""

from ocbf.errors import ValidationError

from .cache import MemoryArtifactStore


class ExecutionSession:
    """Own a bounded in-memory artifact store across related synchronous calls.

    Args:
        max_cache_bytes (int): Required nonnegative retained-cache budget; zero disables
            retention. This bounds accounted cache entries, not total process memory.
        policy (InferencePolicy | None): Default inference policy for facade calls that
            do not supply one explicitly.

    Use as a context manager or call ``close`` explicitly. Closing releases session-held
    references and rejects subsequent use; caller-held results remain valid. A session
    is intended for one synchronous owner and does not provide persistent storage.
    """

    def __init__(self, *, max_cache_bytes, policy=None):
        self._store = MemoryArtifactStore(max_cache_bytes)
        self.policy = policy
        self._closed = False
        self.last_assessment = None

    def store_for(self, reuse=True):
        if self._closed:
            raise ValidationError("execution session is closed")
        return self._store if reuse else None

    def explain_dependencies(self, index):
        """Explain a cache decision: which numeric factors changed since the previous
        model of the same structure. Content-addressed keys drive the actual reuse; this
        is the coordinator-owned explanation surface, holding no lower-package state."""
        self.store_for()
        previous = self._store.get("dependency.previous", index.structure_id)
        self._store.put("dependency.previous", index.structure_id, index.numerics)
        return None if previous is None else index.changed_values(previous)

    @property
    def stats(self):
        """Return current cache accounting, including hit/miss and retained-byte totals."""
        return self._store.stats

    def clear(self):
        """Evict all entries while keeping this open session usable."""
        self.store_for()
        self._store.clear()

    def close(self):
        """Release retained entries and assessment; repeated closure is harmless."""
        self._store.clear()
        self.last_assessment = None
        self._closed = True

    def __enter__(self):
        self.store_for()
        return self

    def __exit__(self, *exc):
        self.close()
