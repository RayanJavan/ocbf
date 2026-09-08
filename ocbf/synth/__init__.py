"""Synthetic ground truth and source simulation.

Design doc section 9.4 puts this *before* the inference engine in the build order, for two
reasons. Ground truth is the only way to measure calibration, which Stage 1 section 3.2
identifies as where the value actually is. And building the generator first forces the
data model to be honest -- if the schema and universe cannot express a realistic
object-centric process, that shows up here rather than three modules later.
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
