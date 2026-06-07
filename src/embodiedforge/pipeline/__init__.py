"""Pipeline stages — orchestrate adapters into a processing workflow."""

from embodiedforge.pipeline.ingest import ingest_episode
from embodiedforge.pipeline.segment import segment_episode
from embodiedforge.pipeline.annotate_semantic import annotate_semantic
from embodiedforge.pipeline.ground import ground_objects
from embodiedforge.pipeline.segment_mask import segment_masks
from embodiedforge.pipeline.build_heatmap import build_heatmaps
from embodiedforge.pipeline.annotate_3d import annotate_3d
from embodiedforge.pipeline.run_all import run_all

__all__ = [
    "ingest_episode",
    "segment_episode",
    "annotate_semantic",
    "ground_objects",
    "segment_masks",
    "build_heatmaps",
    "annotate_3d",
    "run_all",
]
