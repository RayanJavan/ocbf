"""Sources: the sparse, unreliable, numerous evidence layer.

Design doc section 9.2 makes [`Source`][ocbf.sources.base.Source] *the* extension point. Adding a new kind of
evidence means writing one adapter that declares a scope, a coverage semantics, a channel
family, a cluster, and a feature vector -- and nothing in the core changes.

Two declarations on every adapter are load-bearing rather than metadata:

* [`CoverageSemantics`][ocbf.sources.base.CoverageSemantics] -- what the source's *silence* means. Mandatory, with no
  default, because a wrong default here silently biases everything downstream
  (Stage 1 section 4.5 on informative missingness).
* ``cluster_id`` -- the declared source family. Sources sharing a vendor, upstream model
  or site share a random effect in the reliability GLM, and that shared effect *is* the
  dependency correction (design doc section 5.3). With a thin overlap graph, declared
  families beat a learned dependency graph, because learning dependencies also needs
  overlap.
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
