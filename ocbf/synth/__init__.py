"""Explicitly synthetic object-centric processes and corrupted static sources.

Generators retain reference truth for numerical comparisons. They are evaluation utilities;
no generated identity, report or parameter should be treated as observed factory evidence.
"""

from ocbf.synth.corrupt import SourceRegime, simulate_sources
from ocbf.synth.process import GroundTruth, ProcessConfig, order_process_schema, simulate_process

__all__ = [
    "GroundTruth",
    "ProcessConfig",
    "SourceRegime",
    "order_process_schema",
    "simulate_process",
    "simulate_sources",
]
