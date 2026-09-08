"""The inference contract: what any engine returns.

Stage 1 decision 8 pins this before any modelling code, because it is the API:

* **calibrated marginals** -- per-assertion posteriors;
* **joint samples** -- coherent whole-log realisations, since downstream process mining
  consumes logs, not marginals;
* **decidability flags** -- an explicit three-valued verdict where the evidence is
  provably insufficient, rather than a confidently-wrong 0.51;
* **attribution vectors** -- the decomposition of each posterior into the contributions
  that produced it.

Baselines and the full engine both return a [`BeliefState`][ocbf.belief.state.BeliefState], which is what makes the
mandatory weighted-vote comparison of design doc section 8.3 a one-line diff.
"""

from ocbf.belief.state import BeliefState, BeliefStateBuilder, Verdict

__all__ = ["BeliefState", "BeliefStateBuilder", "Verdict"]
