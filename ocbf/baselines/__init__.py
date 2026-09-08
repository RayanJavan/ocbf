"""Baselines. Not decoration -- the bar.

Stage 1 section 3.2 is blunt about this. On real data, sophisticated truth-discovery
methods often barely beat weighted voting, and a 2026 replication in an LLM-evidence
setting found a full fusion stack sharpened nothing over its raw estimator. So design doc
section 8.3 makes weighted vote a **mandatory, always-reported** comparison: a fusion
system that does not clear it is not working, and the only way to know is to run it every
time.

The continuous layer gets the same treatment, for the same reason: fusing timestamps through
a copula has to beat taking the median of what the sources said, or it is decoration. See
[`aggregate`][ocbf.baselines.aggregate].

All baselines return a [`BeliefState`][ocbf.belief.BeliefState], so comparing them against the
full engine is a one-line diff rather than a bespoke harness.
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
