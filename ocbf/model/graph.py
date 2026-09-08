"""Factor graph container and the bank protocol.

Everything is in **log space** with a finite ``NEG_INF`` sentinel rather than ``-inf``.
Genuine ``-inf`` is correct mathematically and a disaster numerically: hard factors produce
``-inf`` messages, and the moment two of them meet in a subtraction the result is ``nan``,
which then propagates silently through the entire graph. A large finite floor keeps hard
constraints effectively hard (``exp(-1e30)`` is 0 to any precision that matters) while
leaving every arithmetic operation well-defined.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ocbf.assertions import VariableRegistry

NEG_INF = -1.0e4
"""Finite stand-in for ``-inf`` in log-potentials.

The magnitude is chosen, not arbitrary. It must be large enough that ``exp(NEG_INF)``
underflows to exactly zero in float64 (which happens below about -745), and *small* enough
that adding an ordinary log-potential to it preserves that potential exactly. A -1e30
sentinel fails the second test catastrophically: ``-1e30 + 5.0 == -1e30`` in float64, so
the finite part is annihilated.

This matters because belief propagation subtracts a variable's own outgoing message from
its belief, and with hard factors both terms are near the sentinel. The subtraction is only
meaningful if the finite remainder survived the addition -- at -1e4 it does, and even a
hundred stacked hard constraints leave ~1e-10 absolute precision.
"""


@runtime_checkable
class FactorBank(Protocol):
    """All groundings of one factor template, stored columnar.

    A bank owns ``n_edges`` edges. Edge ``e`` connects one factor to the variable
    ``edge_vars()[e]``. The engine hands the bank the incoming variable-to-factor messages
    for its own edges and receives the outgoing factor-to-variable messages back.
    """

    name: str

    def edge_vars(self) -> np.ndarray:
        """``int64[n_edges]`` -- the variable each edge attaches to."""
        ...

    def factor_to_var(self, incoming: np.ndarray) -> np.ndarray:
        """Sum-product messages.

        ``incoming`` and the return value are both ``float64[n_edges, max_card]`` in log
        space, padded with [`NEG_INF`][ocbf.model.graph.NEG_INF] beyond each variable's cardinality.
        """
        ...

    def n_factors(self) -> int: ...


@runtime_checkable
class DenseFactors(Protocol):
    """A bank that can also write itself out one explicit factor at a time.

    Optional, and deliberately separate from [`FactorBank`][ocbf.model.graph.FactorBank]: the engine never needs it,
    and requiring it would tax every bank for the benefit of the oracles alone. A bank that
    implements it can be checked against exact inference; one that does not is still a
    perfectly good bank, it just cannot be enumerated.
    """

    def dense_factors(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(var_ids, log_table)`` for every grounding of this template.

        ``log_table`` has one axis per entry of ``var_ids``, in that order.
        """
        ...


class FactorGraph:
    """Variables from a registry, plus banks of factors over them."""

    __slots__ = ("registry", "log_prior", "banks", "max_card", "_edge_var", "_offsets")

    def __init__(
        self,
        registry: VariableRegistry,
        log_prior: np.ndarray,
        banks: Sequence[FactorBank],
    ) -> None:
        if not registry.is_frozen:
            raise ValueError("registry must be frozen")
        n = len(registry)
        self.max_card = max(registry.max_cardinality, 1)
        if log_prior.shape != (n, self.max_card):
            raise ValueError(f"log_prior must be {(n, self.max_card)}, got {log_prior.shape}")

        self.registry = registry
        self.log_prior = log_prior
        self.banks = list(banks)

        # One flat edge space across all banks, so the engine does a single scatter-add.
        offsets: list[int] = []
        pieces: list[np.ndarray] = []
        total = 0
        for bank in self.banks:
            ev = bank.edge_vars()
            offsets.append(total)
            pieces.append(ev)
            total += len(ev)
        self._offsets = offsets
        self._edge_var = (
            np.concatenate(pieces).astype(np.int64) if pieces else np.zeros(0, dtype=np.int64)
        )

    @property
    def n_vars(self) -> int:
        return len(self.registry)

    @property
    def n_edges(self) -> int:
        return len(self._edge_var)

    @property
    def edge_var(self) -> np.ndarray:
        return self._edge_var

    def bank_slice(self, i: int) -> slice:
        start = self._offsets[i]
        return slice(start, start + len(self.banks[i].edge_vars()))

    def dense_factors(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Every factor in the graph as an explicit ``(var_ids, log_table)`` pair.

        The deliberate inverse of everything the banks do. A [`CardinalityBank`][ocbf.model.banks.CardinalityBank] exists
        so that a counting constraint over ``k`` links never becomes a ``2^k`` table, and
        this method builds precisely that table. It is here for **oracles** -- exact
        elimination, brute-force enumeration, export to another solver -- which understand
        no other representation and are only ever run on graphs small enough to afford it.
        It is not an inference path, and nothing in the engine calls it.

        Tables are trimmed to each variable's true cardinality, so the
        [`NEG_INF`][ocbf.model.graph.NEG_INF] padding that keeps the batched engine rectangular never reaches the
        consumer. The prior is *not* included: it is a property of the graph rather than of
        any bank, and a caller that wants the full joint adds it from `log_prior`.

        Raises:
            TypeError: if any bank does not implement [`DenseFactors`][ocbf.model.graph.DenseFactors].
        """
        card = self.registry.cardinalities
        for bank in self.banks:
            if not isinstance(bank, DenseFactors):
                raise TypeError(
                    f"bank {bank.name!r} ({type(bank).__name__}) cannot enumerate itself; "
                    "implement dense_factors() to use it with an exact oracle"
                )
            for var_ids, table in bank.dense_factors():
                yield var_ids, _trim_to_cardinality(var_ids, table, card, bank.name)

    @staticmethod
    def padded_log_prior(registry: VariableRegistry, uniform: bool = True) -> np.ndarray:
        """A ``(n_vars, max_card)`` log-prior, ``NEG_INF`` outside each variable's domain.

        Continuous variables get an all-``NEG_INF`` row; the discrete engine never touches
        them and the continuous engine keeps its own representation.
        """
        n = len(registry)
        k = max(registry.max_cardinality, 1)
        out = np.full((n, k), NEG_INF, dtype=np.float64)
        card = registry.cardinalities
        for i in range(n):
            c = int(card[i])
            if c > 0:
                out[i, :c] = -np.log(c) if uniform else 0.0
        return out

    def summary(self) -> dict[str, object]:
        return {
            "variables": self.n_vars,
            "edges": self.n_edges,
            "banks": {b.name: b.n_factors() for b in self.banks},
            "factors": sum(b.n_factors() for b in self.banks),
        }

    def __repr__(self) -> str:
        return (
            f"FactorGraph(vars={self.n_vars}, banks={len(self.banks)}, "
            f"factors={sum(b.n_factors() for b in self.banks)}, edges={self.n_edges})"
        )


def _trim_to_cardinality(
    var_ids: np.ndarray, table: np.ndarray, card: np.ndarray, bank: str
) -> np.ndarray:
    """Cut a bank's padded log-potential table down to the variables' real domains."""
    table = np.asarray(table, dtype=np.float64)
    if table.ndim != len(var_ids):
        raise ValueError(
            f"bank {bank!r} yielded a {table.ndim}-d table for {len(var_ids)} variable(s)"
        )
    cuts = []
    for axis, v in enumerate(var_ids):
        c = int(card[int(v)])
        if table.shape[axis] < c:
            raise ValueError(
                f"bank {bank!r} yielded axis {axis} of width {table.shape[axis]} for "
                f"variable {int(v)}, whose cardinality is {c}"
            )
        cuts.append(slice(0, c))
    return np.ascontiguousarray(table[tuple(cuts)])
