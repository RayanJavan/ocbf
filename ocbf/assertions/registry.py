"""Dense execution storage for stable semantic assertion references.

``AssertionRef`` addresses meanings; ``VariableRegistry`` allocates contiguous indices and
cardinalities for numerical arrays. Indices are local to a registry and must not be used as
persistent evidence identifiers. Freeze a completed registry before numerical execution.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import Enum

import numpy as np

from ocbf.assertions.refs import AssertionRef, Family

NA_STATE = 0
"""Reserved index for an inapplicable discrete attribute. A selected event type can make an attribute outside its domain; this is distinct from an unknown value."""


class VarKind(str, Enum):
    """Representation class of a latent variable."""

    BINARY = "binary"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"

    @property
    def is_discrete(self) -> bool:
        return self is not VarKind.CONTINUOUS


_CONTINUOUS_CARD = 0
"""Sentinel cardinality for continuous variables (they have no state count)."""


class VariableRegistry:
    """Bidirectional map between [`AssertionRef`][ocbf.assertions.refs.AssertionRef] and dense integer ids.

    Build with `add`, then call `freeze` before any inference. Freezing sorts
    ids into contiguous per-family blocks and builds the numpy views; after that the
    registry is read-only.
    """

    __slots__ = (
        "_refs",
        "_index",
        "_kind",
        "_card",
        "_family",
        "_frozen",
        "_family_slices",
        "_kind_arr",
        "_card_arr",
        "_family_arr",
    )

    def __init__(self) -> None:
        self._refs: list[AssertionRef] = []
        self._index: dict[AssertionRef, int] = {}
        self._kind: list[VarKind] = []
        self._card: list[int] = []
        self._family: list[Family] = []
        self._frozen = False
        self._family_slices: dict[Family, slice] = {}
        self._kind_arr: np.ndarray | None = None
        self._card_arr: np.ndarray | None = None
        self._family_arr: np.ndarray | None = None

    # -- construction ---------------------------------------------------------------

    def add(self, ref: AssertionRef, kind: VarKind, cardinality: int | None = None) -> int:
        """Register a latent variable, returning its id.

        Idempotent: re-adding an existing ref returns its existing id, but a *conflicting*
        kind or cardinality is an error rather than a silent overwrite -- that would mean
        two parts of the model disagree about what a variable is.
        """
        if self._frozen:
            raise RuntimeError("registry is frozen; cannot add variables")

        card = self._resolve_cardinality(kind, cardinality)
        existing = self._index.get(ref)
        if existing is not None:
            if self._kind[existing] is not kind or self._card[existing] != card:
                raise ValueError(
                    f"{ref} already registered as {self._kind[existing].value}"
                    f"/{self._card[existing]}, cannot re-register as {kind.value}/{card}"
                )
            return existing

        idx = len(self._refs)
        self._refs.append(ref)
        self._index[ref] = idx
        self._kind.append(kind)
        self._card.append(card)
        self._family.append(ref.family)
        return idx

    @staticmethod
    def _resolve_cardinality(kind: VarKind, cardinality: int | None) -> int:
        if kind is VarKind.CONTINUOUS:
            if cardinality not in (None, _CONTINUOUS_CARD):
                raise ValueError("continuous variables do not take a cardinality")
            return _CONTINUOUS_CARD
        if kind is VarKind.BINARY:
            if cardinality not in (None, 2):
                raise ValueError("binary variables have cardinality 2")
            return 2
        if cardinality is None or cardinality < 2:
            raise ValueError(f"categorical variables need cardinality >= 2, got {cardinality}")
        return cardinality

    def add_binary(self, ref: AssertionRef) -> int:
        return self.add(ref, VarKind.BINARY)

    def add_categorical(self, ref: AssertionRef, cardinality: int) -> int:
        return self.add(ref, VarKind.CATEGORICAL, cardinality)

    def add_continuous(self, ref: AssertionRef) -> int:
        return self.add(ref, VarKind.CONTINUOUS)

    def freeze(self) -> VariableRegistry:
        """Sort into per-family contiguous blocks and build numpy views.

        Returns ``self`` so it can be chained. Idempotent.
        """
        if self._frozen:
            return self

        family_order = list(Family)
        order = sorted(
            range(len(self._refs)),
            key=lambda i: (family_order.index(self._family[i]), self._refs[i]),
        )

        self._refs = [self._refs[i] for i in order]
        self._kind = [self._kind[i] for i in order]
        self._card = [self._card[i] for i in order]
        self._family = [self._family[i] for i in order]
        self._index = {ref: i for i, ref in enumerate(self._refs)}

        self._family_slices = {}
        start = 0
        for i, fam in enumerate(self._family):
            if i > 0 and fam is not self._family[i - 1]:
                self._family_slices[self._family[i - 1]] = slice(start, i)
                start = i
        if self._family:
            self._family_slices[self._family[-1]] = slice(start, len(self._family))

        self._kind_arr = np.array([k.value for k in self._kind], dtype=object)
        self._card_arr = np.asarray(self._card, dtype=np.int64)
        self._family_arr = np.array([f.value for f in self._family], dtype=object)
        self._frozen = True
        return self

    def _require_frozen(self) -> None:
        if not self._frozen:
            raise RuntimeError("registry must be frozen before this operation")

    # -- lookup ---------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._refs)

    def __contains__(self, ref: object) -> bool:
        return ref in self._index

    def __iter__(self) -> Iterator[AssertionRef]:
        return iter(self._refs)

    def index(self, ref: AssertionRef) -> int:
        """Id of ``ref``. Raises ``KeyError`` if unregistered."""
        try:
            return self._index[ref]
        except KeyError:
            raise KeyError(f"{ref} is not registered") from None

    def get(self, ref: AssertionRef, default: int | None = None) -> int | None:
        """Return the execution index of ``ref``, or ``default`` when it is absent. Absence from this registry does not establish that an assertion is false."""
        return self._index.get(ref, default)

    def ref(self, idx: int) -> AssertionRef:
        return self._refs[idx]

    def refs(self, ids: Sequence[int] | np.ndarray) -> list[AssertionRef]:
        return [self._refs[int(i)] for i in ids]

    def kind(self, idx: int) -> VarKind:
        return self._kind[idx]

    def cardinality(self, idx: int) -> int:
        return self._card[idx]

    def family_of(self, idx: int) -> Family:
        return self._family[idx]

    def indices(self, refs: Sequence[AssertionRef]) -> np.ndarray:
        return np.fromiter((self._index[r] for r in refs), dtype=np.int64, count=len(refs))

    # -- vectorised views -----------------------------------------------------------

    @property
    def cardinalities(self) -> np.ndarray:
        """``int64[n]`` state counts; ``0`` for continuous variables."""
        self._require_frozen()
        assert self._card_arr is not None
        return self._card_arr

    def family_slice(self, family: Family) -> slice:
        """Contiguous id range for a family; empty slice if the family is unused."""
        self._require_frozen()
        return self._family_slices.get(family, slice(0, 0))

    def family_ids(self, family: Family) -> np.ndarray:
        s = self.family_slice(family)
        return np.arange(s.start, s.stop, dtype=np.int64)

    def count(self, family: Family) -> int:
        s = self.family_slice(family)
        return s.stop - s.start

    @property
    def discrete_ids(self) -> np.ndarray:
        self._require_frozen()
        assert self._card_arr is not None
        return np.flatnonzero(self._card_arr > 0).astype(np.int64)

    @property
    def continuous_ids(self) -> np.ndarray:
        self._require_frozen()
        assert self._card_arr is not None
        return np.flatnonzero(self._card_arr == _CONTINUOUS_CARD).astype(np.int64)

    @property
    def max_cardinality(self) -> int:
        """Largest discrete domain, i.e. the padded width of a batched belief tensor."""
        self._require_frozen()
        assert self._card_arr is not None
        return int(self._card_arr.max()) if len(self._card_arr) else 0

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def summary(self) -> dict[str, int]:
        """Per-family variable counts -- the first thing to look at after grounding."""
        self._require_frozen()
        return {fam.value: self.count(fam) for fam in Family if self.count(fam)}

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "open"
        return f"VariableRegistry({len(self._refs)} vars, {state})"
