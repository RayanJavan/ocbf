"""Explicit parameter values and independent source-estimation utilities.

The evidence workflow resolves fixed parameters through ``reliability.config`` and uses
channel-specific values from ``reliability.channels``. Moment estimators are separate
utilities for declared static-claim assumptions; inference never invokes them implicitly.
"""

from ocbf.reliability.moments import (
    BinaryClaimTable,
    TripletEstimate,
    pairwise_agreement,
    pairwise_channels,
    triplet_accuracies,
)
from ocbf.reliability.params import (
    ContinuousChannel,
    ContinuousChannelTable,
    ReliabilityTable,
    SourceParams,
)

__all__ = [
    "BinaryClaimTable",
    "ContinuousChannel",
    "ContinuousChannelTable",
    "ReliabilityTable",
    "SourceParams",
    "TripletEstimate",
    "pairwise_agreement",
    "pairwise_channels",
    "triplet_accuracies",
]
