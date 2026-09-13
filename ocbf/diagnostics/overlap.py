"""Overlap-graph diagnostics for static binary source identifiability.

The graph records which sources overlap on assertions. Bipartite or isolated components
cannot resolve the orientation of the binary agreement model without additional
assumptions. The diagnostic concerns that model, not arbitrary observation channels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx
import numpy as np

from ocbf.reliability.moments import BinaryClaimTable
from ocbf.sources import ClaimSet


class Identifiability(str, Enum):
    """Per-source verdict on whether its reliability is data-identified."""

    IDENTIFIED = "identified"
    """In a non-bipartite component of size >= 3: skills are jointly estimable."""

    SIGN_AMBIGUOUS = "sign_ambiguous"
    """In a bipartite component: estimable only up to a global sign flip. The
    better-than-chance prior picks a branch; the data does not."""

    PRIOR_ONLY = "prior_only"
    """Insufficient static-source overlap to estimate reliability. Any assigned quality remains an explicit assumption."""


@dataclass(slots=True)
class OverlapReport:
    """Structure of the source overlap graph and its consequences."""

    n_sources: int
    n_edges: int
    components: list[list[str]]
    verdict: dict[str, Identifiability]
    bipartite_components: list[int]
    min_overlap: int

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def largest_component(self) -> int:
        return max((len(c) for c in self.components), default=0)

    def counts(self) -> dict[str, int]:
        """Number of sources holding each identifiability verdict."""
        out = {v.value: 0 for v in Identifiability}
        for v in self.verdict.values():
            out[v.value] += 1
        return out

    @property
    def is_globally_identifiable(self) -> bool:
        """True when every source sits in a non-bipartite component of size >= 3.

        Almost never true in the sparse regime. That is the point of reporting it.
        """
        return all(v is Identifiability.IDENTIFIED for v in self.verdict.values())

    def summary(self) -> dict[str, object]:
        """Graph shape and verdict counts as a flat mapping."""
        return {
            "sources": self.n_sources,
            "edges": self.n_edges,
            "components": self.n_components,
            "largest_component": self.largest_component,
            "bipartite_components": len(self.bipartite_components),
            "min_overlap": self.min_overlap,
            "globally_identifiable": self.is_globally_identifiable,
            **self.counts(),
        }

    def explain(self) -> str:
        """Formatted verdict summary, including whether the pool is globally identifiable."""
        c = self.counts()
        lines = [
            f"Source overlap graph: {self.n_sources} sources, {self.n_edges} edges, "
            f"{self.n_components} components (largest {self.largest_component}).",
            f"  identified      {c['identified']:6d}  reliability estimable from co-claims",
            f"  sign_ambiguous  {c['sign_ambiguous']:6d}  bipartite component: sign fixed by prior, not data",
            f"  prior_only      {c['prior_only']:6d}  insufficient overlap: reliability is assumed",
        ]
        if not self.is_globally_identifiable:
            lines.append(
                "  -> Not globally identifiable. Reliabilities are comparable across "
                "components only under additional shared assumptions."
            )
        return "\n".join(lines)


def overlap_report(claim_set: ClaimSet, *, min_overlap: int = 2) -> OverlapReport:
    """Build the overlap graph and classify every source.

    ``min_overlap`` is how many co-claimed assertions constitute an edge. Raising it makes
    the graph sparser and the verdict more conservative; a single shared assertion is very
    weak evidence about the relative reliability of two sources.
    """
    table = BinaryClaimTable(claim_set)
    sources = table.sources
    n = len(sources)

    graph = nx.Graph()
    graph.add_nodes_from(sources)
    n_edges = 0
    if n:
        counts = table.overlap_counts()
        np.fill_diagonal(counts, 0)
        rows, cols = np.nonzero(np.triu(counts >= min_overlap))
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            graph.add_edge(sources[i], sources[j], overlap=int(counts[i, j]))
        n_edges = graph.number_of_edges()

    components = [sorted(c) for c in nx.connected_components(graph)] if n else []
    components.sort(key=len, reverse=True)

    verdict: dict[str, Identifiability] = {}
    bipartite: list[int] = []
    for k, comp in enumerate(components):
        sub = graph.subgraph(comp)
        if len(comp) < 3 or sub.number_of_edges() == 0:
            # No triplet, hence no closed-form accuracy and no odd cycle to speak of.
            label = Identifiability.PRIOR_ONLY
        elif nx.is_bipartite(sub):
            # Bipartite <=> no odd cycle <=> the sign of every skill is unresolvable.
            label = Identifiability.SIGN_AMBIGUOUS
            bipartite.append(k)
        else:
            label = Identifiability.IDENTIFIED
        for s in comp:
            verdict[s] = label

    return OverlapReport(
        n_sources=n,
        n_edges=n_edges,
        components=components,
        verdict=verdict,
        bipartite_components=bipartite,
        min_overlap=min_overlap,
    )
