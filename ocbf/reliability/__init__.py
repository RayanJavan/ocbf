"""Source reliability: estimating who to trust, without ground truth.

The sparse regime forbids a free parameter per source (Stage 1 section 4.3): a source with
three claims has an accuracy estimate with a standard error near 0.29. Everything here
exists to get statistical strength from somewhere other than the source's own claim count.

* [`ocbf.reliability.moments`][ocbf.reliability.moments] -- label-free moment estimators. The triplet method gives
  per-source accuracy in *closed form* from pairwise agreement rates alone, which is both a
  usable estimator and the initialiser that keeps the outer EM loop out of bad local optima
  (design doc section 6.2). Its continuous sibling,
  [`pairwise_channels`][ocbf.reliability.moments.pairwise_channels], reads a source's bias
  and noise scale off pairwise disagreement in the same label-free way.
* [`ocbf.reliability.params`][ocbf.reliability.params] -- the parameter types the belief
  block consumes: two-sided binary quality, the one-parameter categorical spread model, and
  the per-template continuous channel.
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
