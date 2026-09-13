"""Candidate construction and immutable semantic context.

A universe enumerates admitted event/object candidates and qualified links. An absent
candidate is outside model support. ``SemanticContext`` snapshots the universe, schema,
constraint declarations and grounding provenance for canonical model construction.
"""

from ocbf.universe.core import EventCandidate, ObjectRecord, PruneReport, Universe, UniverseBuilder

__all__ = ["EventCandidate", "ObjectRecord", "PruneReport", "Universe", "UniverseBuilder"]
