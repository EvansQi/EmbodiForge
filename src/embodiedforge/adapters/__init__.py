"""Adapter interfaces and factory for model backends."""

from embodiedforge.adapters.base import (
    SemanticLabelerAdapter,
    GroundingAdapter,
    SegmentationAdapter,
    DepthAdapter,
)
from embodiedforge.adapters.factory import create_adapter

__all__ = [
    "SemanticLabelerAdapter",
    "GroundingAdapter",
    "SegmentationAdapter",
    "DepthAdapter",
    "create_adapter",
]
