"""Simple static-claim comparison methods.

Voting, two-coin Dawid–Skene and continuous aggregation provide declared numerical
baselines. They consume static claims and do not implement evidence revision, provenance
admission or the canonical process-query workflow.
"""

from ocbf.baselines.aggregate import claim_median, weighted_mean
from ocbf.baselines.dawid_skene import dawid_skene
from ocbf.baselines.vote import majority_vote, weighted_vote

__all__ = [
    "claim_median",
    "dawid_skene",
    "majority_vote",
    "weighted_mean",
    "weighted_vote",
]
