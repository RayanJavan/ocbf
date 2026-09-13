"""Source metadata and static claim utilities.

``Source`` and ``ClaimSet`` support simulation, numerical comparisons and source diagnostics.
The evidence workflow consumes versioned ``EvidenceRecord`` values through the independent
``sources.contracts.EvidenceInterpreter`` protocol. Static claims do not establish receipt
times, observation opportunities or a dependence-corrected likelihood.
"""

from ocbf.sources.base import (
    ChannelFamily,
    CoverageSemantics,
    Source,
    SourceProfile,
    StaticSource,
)
from ocbf.sources.claims import Claim, ClaimSet

__all__ = [
    "ChannelFamily",
    "Claim",
    "ClaimSet",
    "CoverageSemantics",
    "Source",
    "SourceProfile",
    "StaticSource",
]
