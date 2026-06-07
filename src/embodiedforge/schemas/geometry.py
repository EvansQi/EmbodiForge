"""Geometry schemas — bbox, mask, affordance map, 3D keypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates [x1, y1, x2, y2]."""

    x1: float = Field(..., description="Left edge")
    y1: float = Field(..., description="Top edge")
    x2: float = Field(..., description="Right edge")
    y2: float = Field(..., description="Bottom edge")
    label: str = Field(default="", description="Object or part label")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)


class Point2D(BaseModel):
    """A 2D point in pixel coordinates."""

    x: float
    y: float
    label: str = Field(default="", description="Point label, e.g. 'affordance_point'")


class MaskResult(BaseModel):
    """Result of mask generation for a single frame."""

    frame_idx: int
    mask_path: str = Field(
        default="", description="Path to saved binary mask image (uint8, 0/255)"
    )
    mask_area: int = Field(default=0, description="Number of nonzero pixels in mask")
    bbox: Optional[BBox] = None
    backend: str = Field(default="mock")


class AffordanceResult(BaseModel):
    """Affordance point and heatmap for a single keyframe."""

    frame_idx: int
    affordance_point: Optional[Point2D] = None
    heatmap_path: str = Field(
        default="", description="Path to saved affordance heatmap image"
    )
    heatmap_max: float = Field(default=0.0, description="Max value in heatmap")
    backend: str = Field(default="mock")


class Keypoint3D(BaseModel):
    """A single 3D keypoint."""

    x: float
    y: float
    z: float
    label: str = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Geometry3DResult(BaseModel):
    """3D geometry annotation for a keyframe."""

    frame_idx: int
    keypoints_3d: list[Keypoint3D] = Field(default_factory=list)
    # Optional axis representations for insertion tasks
    port_axis: Optional[list[float]] = Field(
        default=None, description="Port axis direction vector [dx, dy, dz]"
    )
    plug_axis: Optional[list[float]] = Field(
        default=None, description="Plug axis direction vector [dx, dy, dz]"
    )
    backend: str = Field(default="mock")
