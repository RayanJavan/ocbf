"""Factor banks: batched sum-product for one template at a time."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np

from ocbf.mathx import logsumexp
from ocbf.model.graph import NEG_INF

MAX_DENSE_GROUP = 20
"""Largest cardinality group [`CardinalityBank.dense_factors`][ocbf.model.banks.CardinalityBank.dense_factors] will enumerate.

``2**20`` float64 entries is 8 MB -- already far past any graph an exact oracle can
finish, and small enough that hitting the limit is a bug in the caller rather than a
resource the machine happened not to have.
"""


class UnaryBank:
    """Independent evidence on single variables.

    Source channel likelihoods and structural priors both live here. Their messages do not
    depend on anything incoming, which makes them the natural carriers of **attribution**:
    each unary edge is one named contribution to its variable's posterior log-odds.
    """

    __slots__ = ("name", "_vars", "_pot", "labels", "_padded")

    def __init__(
        self,
        var_ids: np.ndarray,
        log_potentials: np.ndarray,
        *,
        name: str = "unary",
        labels: Sequence[str] | None = None,
    ) -> None:
        if len(var_ids) != len(log_potentials):
            raise ValueError("var_ids and log_potentials must have the same length")
        self.name = name
        self._vars = np.asarray(var_ids, dtype=np.int64)
        self._pot = np.asarray(log_potentials, dtype=np.float64)
        self.labels = tuple(labels) if labels is not None else ()
        self._padded: np.ndarray | None = None

    def edge_vars(self) -> np.ndarray:
        return self._vars

    def factor_to_var(self, incoming: np.ndarray) -> np.ndarray:
        width = incoming.shape[1]
        have = self._pot.shape[1]
        if have == width:
            return self._pot
        if have > width:
            raise ValueError(
                f"unary bank {self.name!r} has width {have} but the graph width is {width}"
            )
        # Pad to the graph width rather than making every caller know it. Cached, since the
        # width is fixed for the lifetime of a graph.
        if self._padded is None or self._padded.shape[1] != width:
            padded = np.full((len(self._vars), width), NEG_INF)
            padded[:, :have] = self._pot
            self._padded = padded
        return self._padded

    def dense_factors(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """One factor per edge -- the stored row, which is already the whole table."""
        for e, v in enumerate(self._vars):
            yield np.array([v], dtype=np.int64), self._pot[e]

    def n_factors(self) -> int:
        return len(self._vars)


class PairwiseBank:
    """Pairwise factors sharing a small set of log-potential tables.

    Edge layout is ``[all A-side edges, all B-side edges]``, so both message directions are
    single batched ``logsumexp`` reductions rather than a Python loop over factors.

    This bank carries the structural coupling that makes the model more than parallel
    per-assertion voting: referential integrity ties a link to its endpoints' existence,
    and the type gate ties a link to its event's latent type. Those edges are how belief
    reaches assertions no source covered -- the mechanism Stage 1 section 4.2 says must
    carry most of the work in a sparse regime.
    """

    __slots__ = ("name", "_a", "_b", "_tables", "_tid", "_ka", "_kb")

    def __init__(
        self,
        a_ids: np.ndarray,
        b_ids: np.ndarray,
        tables: np.ndarray,
        table_id: np.ndarray,
        *,
        name: str = "pairwise",
    ) -> None:
        if not (len(a_ids) == len(b_ids) == len(table_id)):
            raise ValueError("a_ids, b_ids and table_id must have the same length")
        if tables.ndim != 3:
            raise ValueError("tables must be (n_templates, Ka, Kb)")
        self.name = name
        self._a = np.asarray(a_ids, dtype=np.int64)
        self._b = np.asarray(b_ids, dtype=np.int64)
        self._tables = np.asarray(tables, dtype=np.float64)
        self._tid = np.asarray(table_id, dtype=np.int64)
        self._ka = int(tables.shape[1])
        self._kb = int(tables.shape[2])

    def edge_vars(self) -> np.ndarray:
        return np.concatenate([self._a, self._b])

    def factor_to_var(self, incoming: np.ndarray) -> np.ndarray:
        n = len(self._a)
        msg_from_a = incoming[:n, : self._ka]
        msg_from_b = incoming[n:, : self._kb]
        tab = self._tables[self._tid]  # (n, Ka, Kb)

        to_a = logsumexp(tab + msg_from_b[:, None, :], axis=2)
        to_b = logsumexp(tab + msg_from_a[:, :, None], axis=1)

        out = np.full_like(incoming, NEG_INF)
        out[:n, : self._ka] = to_a
        out[n:, : self._kb] = to_b
        return out

    def dense_factors(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """One factor per grounding, sharing the template table it was grounded from."""
        for f in range(len(self._a)):
            pair = np.array([self._a[f], self._b[f]], dtype=np.int64)
            yield pair, self._tables[int(self._tid[f])]

    def n_factors(self) -> int:
        return len(self._a)


class CardinalityBank:
    """Soft counting factors over groups of binary variables.

    Implements ``sum_o R[e,q,o] in [lo, hi]`` from design doc section 3.1. A naive factor
    over ``k`` links is ``2^k``; the forward-backward recursion over the running count is
    ``O(k * C)``, with ``C`` the capped count domain. For the overwhelmingly common
    ``exactly-one`` qualifier ``C`` is 4, so this is effectively linear in ``k``.

    **This is a hard requirement, not an optimisation.** Without it a dense E2O
    neighbourhood is intractable, and the cardinality constraint -- one of the strongest
    structural signals available when evidence is thin -- would have to be dropped.

    Groups are ragged, so they are bucketed by ``(k, lo, hi)`` and each bucket is processed
    fully batched. Vectorising where the data is regular, without pretending it always is.

    Two deliberate approximations, both bounded and both stated:

    * The count domain saturates at ``cap = hi + cap_slack``. Violations beyond the cap are
      penalised as if they were *at* the cap, so gross over-linking is under-penalised. The
      slack keeps the flat region far from the interesting range.
    * The constraint is **soft** (design doc section 3.4). Multiplicity is a domain belief,
      not a definitional rule, and real logs violate their own schemas.
    """

    __slots__ = ("name", "_vars", "_weight", "_buckets", "_n_groups")

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        lo: Sequence[int],
        hi: Sequence[int | None],
        weight: float,
        *,
        cap_slack: int = 2,
        name: str = "cardinality",
    ) -> None:
        if not (len(groups) == len(lo) == len(hi)):
            raise ValueError("groups, lo and hi must have the same length")
        self.name = name
        self._weight = float(weight)
        self._n_groups = len(groups)

        flat: list[int] = []
        starts: list[int] = []
        for g in groups:
            starts.append(len(flat))
            flat.extend(int(v) for v in g)
        self._vars = np.asarray(flat, dtype=np.int64)

        # Bucket by (k, lo, hi): identical shape and identical count potential.
        by_shape: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for g, (grp, l, h) in enumerate(zip(groups, lo, hi, strict=True)):
            by_shape[(len(grp), int(l), -1 if h is None else int(h))].append(g)

        self._buckets: list[tuple[np.ndarray, np.ndarray]] = []
        for (k, l, h), members in by_shape.items():
            if k == 0:
                continue
            cap = (h if h >= 0 else l) + cap_slack
            counts = np.arange(cap + 1)
            below = np.maximum(l - counts, 0)
            above = np.zeros_like(counts) if h < 0 else np.maximum(counts - h, 0)
            phi = -self._weight * (below + above).astype(np.float64)
            if not np.any(phi < 0.0):
                continue  # vacuous constraint (e.g. [0..*]); skip rather than waste messages
            edge_idx = np.array(
                [[starts[g] + i for i in range(k)] for g in members], dtype=np.int64
            )
            self._buckets.append((edge_idx, phi))

    def edge_vars(self) -> np.ndarray:
        return self._vars

    def n_factors(self) -> int:
        return self._n_groups

    def dense_factors(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Each group written out as the full ``2^k`` table the recursion exists to avoid.

        Only an oracle should ever ask for this, and only on a small graph -- hence the
        hard ceiling, which fails loudly rather than trying to allocate the table.

        The potential depends on an assignment only through how many of its links are on,
        so one table per ``(k, lo, hi)`` bucket serves every group in it. Bit order is
        irrelevant for the same reason, which is why the popcount can be computed once
        against a flat index and simply reshaped.
        """
        for edge_idx, phi in self._buckets:
            k = int(edge_idx.shape[1])
            if k > MAX_DENSE_GROUP:
                raise ValueError(
                    f"cardinality bank {self.name!r} has a group of {k} links; writing it "
                    f"out densely needs 2**{k} entries, and the ceiling is "
                    f"2**{MAX_DENSE_GROUP}. Exact oracles are for small graphs."
                )
            flat = np.arange(1 << k)
            counts = np.zeros(1 << k, dtype=np.int64)
            for bit in range(k):
                counts += (flat >> bit) & 1
            table = phi[np.minimum(counts, len(phi) - 1)].reshape((2,) * k)
            for row in edge_idx:
                yield self._vars[row], table

    def factor_to_var(self, incoming: np.ndarray) -> np.ndarray:
        out = np.full_like(incoming, NEG_INF)
        for edge_idx, phi in self._buckets:
            b, k = edge_idx.shape
            c = len(phi)

            msg = incoming[edge_idx.ravel(), :2].reshape(b, k, 2)
            msg = msg - logsumexp(msg, axis=2, keepdims=True)

            fwd = np.full((k + 1, b, c), NEG_INF)
            fwd[0, :, 0] = 0.0
            for i in range(k):
                fwd[i + 1] = _advance(fwd[i], msg[:, i, :])

            bwd = np.full((k + 1, b, c), NEG_INF)
            bwd[k, :, 0] = 0.0
            for i in range(k - 1, -1, -1):
                bwd[i] = _advance(bwd[i + 1], msg[:, i, :])

            combined = _saturated_convolve(fwd[:k], bwd[1:], c)  # (k, b, c)

            shifted = np.minimum(np.arange(c) + 1, c - 1)
            m0 = logsumexp(combined + phi[None, None, :], axis=2)
            m1 = logsumexp(combined + phi[shifted][None, None, :], axis=2)

            pair = np.stack([m0, m1], axis=-1)  # (k, b, 2)
            pair = pair - logsumexp(pair, axis=-1, keepdims=True)
            out[edge_idx.ravel(), :2] = np.transpose(pair, (1, 0, 2)).reshape(b * k, 2)
        return out


def _advance(state: np.ndarray, msg: np.ndarray) -> np.ndarray:
    """One forward/backward step of the running-count recursion.

    ``state`` is ``(batch, cap+1)`` log-mass over counts so far; ``msg`` is ``(batch, 2)``
    the variable's normalised message. The top count absorbs everything above it.
    """
    stay = state + msg[:, 0:1]
    step = np.full_like(state, NEG_INF)
    step[:, 1:] = state[:, :-1] + msg[:, 1:2]
    step[:, -1] = np.logaddexp(step[:, -1], state[:, -1] + msg[:, 1])
    return np.logaddexp(stay, step)


def _saturated_convolve(left: np.ndarray, right: np.ndarray, c: int) -> np.ndarray:
    """``combined[i,b,m] = logsumexp_{a+b'=m} left[i,b,a] + right[i,b,b']``, saturated at ``c-1``.

    This is the "everything except variable ``i``" count distribution, which is exactly what
    variable ``i``'s outgoing message needs.
    """
    k, batch, _ = left.shape
    pairs = left[:, :, :, None] + right[:, :, None, :]  # (k, batch, c, c)
    idx = np.minimum(np.add.outer(np.arange(c), np.arange(c)), c - 1).ravel()

    rows = k * batch
    combined = np.full((rows, c), NEG_INF)
    row_index = np.repeat(np.arange(rows), c * c)
    col_index = np.tile(idx, rows)
    np.logaddexp.at(combined, (row_index, col_index), pairs.reshape(rows, -1).ravel())
    return combined.reshape(k, batch, c)
