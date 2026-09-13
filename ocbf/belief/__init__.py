"""Posterior contracts and marginal numerical views.

``belief.posterior`` declares neutral inference results and posterior capabilities;
``belief.estimates`` describes qualified process results. ``BeliefState`` is a separate
assertion-marginal view used by numerical utilities and comparison methods. Its local
mixtures and message contributions do not constitute joint histories or removal effects.
"""

from ocbf.belief.state import BeliefState, BeliefStateBuilder, Verdict

__all__ = ["BeliefState", "BeliefStateBuilder", "Verdict"]
