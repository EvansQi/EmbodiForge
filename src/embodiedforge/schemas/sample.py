"""Export schemas — final training sample and QC summary."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from embodiedforge.schemas.episode import EpisodeMeta
from embodiedforge.schemas.segment import StageSegment
from embodiedforge.schemas.semantic import SemanticAnnotation
from embodiedforge.schemas.geometry import (
    AffordanceResult,
    BBox,
    Geometry3DResult,
    MaskResult,
    Point2D,
)
from embodiedforge.schemas.rl import RLKeyframeSignal, RLSummary


class QCCheck(BaseModel):
    """Result of a single QC check."""

    name: str = Field(..., description="Check name, e.g. 'point_in_mask'")
    passed: bool
    message: str = Field(default="")
    details: dict[str, Any] = Field(default_factory=dict)


class QCSummary(BaseModel):
    """Quality control summary for a training sample."""

    checks: list[QCCheck] = Field(default_factory=list)
    all_passed: bool = Field(default=False)
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall QC score 0-1")

    def compute_score(self) -> float:
        if not self.checks:
            self.score = 0.0
        else:
            self.score = sum(1 for c in self.checks if c.passed) / len(self.checks)
        self.all_passed = all(c.passed for c in self.checks)
        return self.score


class KeyframeAnnotation(BaseModel):
    """Full annotation for a single keyframe — the core training unit."""

    frame_idx: int
    image_path: str = Field(default="", description="Path to the keyframe RGB image")

    # From semantic stage
    skill_id: str = Field(default="")
    stage_label: str = Field(default="")
    target_object: str = Field(default="")
    target_part: str = Field(default="")
    affordance_query: str = Field(default="")
    skill_soft: Optional[dict[str, float]] = None

    # From grounding + mask stage
    object_bbox: Optional[BBox] = None
    part_bbox: Optional[BBox] = None
    object_mask_path: str = Field(default="", description="Path to object mask image")
    affordance_point: Optional[Point2D] = None
    affordance_heatmap_path: str = Field(default="")

    # From 3D stage
    keypoints_3d: list[dict[str, Any]] = Field(default_factory=list)

    # QC
    qc: Optional[QCSummary] = None

    # RL-friendly supervision
    rl: Optional[RLKeyframeSignal] = None


class TrainingSample(BaseModel):
    """A complete training sample — the final export unit."""

    episode_id: str
    task_instruction: str = Field(default="")
    fps: float = Field(default=30.0)

    # Stage info
    stages: list[StageSegment] = Field(default_factory=list)

    # Keyframe-level annotations
    keyframes: list[KeyframeAnnotation] = Field(default_factory=list)

    # Overall QC
    qc: Optional[QCSummary] = None

    # Episode-level RL summary
    rl: Optional[RLSummary] = None

    # Provenance
    pipeline_version: str = Field(default="0.1.0")
    config_hash: str = Field(default="", description="Hash of pipeline config used")


class ExportManifest(BaseModel):
    """Manifest for an exported dataset."""

    samples: list[str] = Field(default_factory=list, description="Paths to sample JSON files")
    total_samples: int = 0
    export_dir: str = Field(default="")
    format: str = Field(default="lerobot")
