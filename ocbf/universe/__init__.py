"""Candidate universe construction and grounding.

Stage 1 decision 1 fixes the universe: candidates are enumerated up front and existence is
a latent binary, rather than the universe itself being transdimensional. This package
builds that universe and applies the three prunes of design doc section 2.3.

Pruning is **part of the model, not an optimisation**. A pruned candidate is asserted to
have prior-only belief, so every prune is recorded and reported -- a wrong prune is a
silent ``-inf``.
"""

from ocbf.universe.core import EventCandidate, ObjectRecord, PruneReport, Universe, UniverseBuilder

__all__ = ["EventCandidate", "ObjectRecord", "PruneReport", "Universe", "UniverseBuilder"]
