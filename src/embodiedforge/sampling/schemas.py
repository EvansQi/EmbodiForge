"""Pydantic schemas for the sampling module.

Defines metadata shapes for:
- Episode summary (lightweight scan record)
- Stratum definition (sampling dimension + bucket)
- Sampling manifest (all episodes + metadata + allocation)
- Ground truth split (selected episodes for annotation)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EpisodeSummary(BaseModel):
    """Lightweight scan record — one per episode found in the dataset."""

    episode_id: str = Field(..., description="Unique episode identifier")
    source_path: str = Field(..., description="Absolute path to the episode directory")
    format_type: str = Field(
        default="standard",
        description="Episode format: 'standard' (meta.json + image dirs) or 'xlsx' (episode_data.xlsx)",
    )

    # Task metadata
    task_type: str = Field(default="unknown", description="Task type label, e.g. 'insert_usb_right', 'pick_place'")
    task_instruction: str = Field(default="", description="Natural language task instruction")

    # Temporal metadata
    num_frames: int = Field(default=0, description="Total number of frames")
    fps: float = Field(default=30.0, description="Recording frame rate")
    duration_seconds: float = Field(default=0.0, description="Episode duration in seconds")

    # Camera metadata
    has_front_rgb: bool = Field(default=False)
    has_wrist_rgb: bool = Field(default=False)
    has_depth: bool = Field(default=False)
    camera_setup: str = Field(
        default="unknown",
        description="Camera configuration label: 'front_only', 'wrist_only', 'both', 'none'",
    )

    # Signal metadata
    has_gripper: bool = Field(default=False, description="Episode contains gripper state signals")
    has_force_torque: bool = Field(default=False, description="Episode contains force/torque signals")
    has_state: bool = Field(default=False, description="Episode contains robot state (joint/pose) signals")

    # Derived metrics (computed during scan)
    trajectory_length: float = Field(
        default=0.0,
        description="Cumulative state displacement (normalized), proxy for motion complexity",
    )
    success: Optional[bool] = Field(
        default=None,
        description="Inferred success: True if gripper close→open detected, None if unknown",
    )
    gripper_close_frame: Optional[int] = Field(
        default=None,
        description="First frame where gripper closes, if detected",
    )
    gripper_open_frame: Optional[int] = Field(
        default=None,
        description="First frame where gripper opens after close, if detected",
    )

    # Metadata bag for extra fields
    extra: dict = Field(default_factory=dict, description="Any extra metadata discovered during scan")

    # File list for traceability
    missing_files: list[str] = Field(
        default_factory=list,
        description="Expected but missing files (empty = complete episode)",
    )

    @property
    def is_complete(self) -> bool:
        return len(self.missing_files) == 0

    @property
    def duration_bucket(self) -> str:
        """Short / medium / long classification for stratified sampling."""
        if self.duration_seconds <= 5:
            return "short"
        if self.duration_seconds <= 20:
            return "medium"
        return "long"


class StratumBucket(BaseModel):
    """A single bucket within a stratified sampling dimension."""

    dimension: str = Field(..., description="Dimension name, e.g. 'task_type', 'duration_bucket'")
    value: str = Field(..., description="Bucket value, e.g. 'insert_usb_right', 'short'")
    count: int = Field(default=0, description="Number of episodes in this bucket")
    episode_ids: list[str] = Field(default_factory=list)


class SamplingManifest(BaseModel):
    """Complete sampling manifest — all scanned episodes + stratification plan.

    This is the primary output artifact for Phase 1.
    """

    dataset_root: str = Field(..., description="Root directory that was scanned")
    scanned_at: datetime = Field(default_factory=datetime.utcnow)
    total_episodes: int = Field(default=0, description="Total episodes discovered")
    complete_episodes: int = Field(default=0, description="Episodes with no missing files")

    # Stratification overview
    strata: list[StratumBucket] = Field(
        default_factory=list,
        description="All strata buckets across all dimensions",
    )
    sampling_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions used for stratified sampling",
    )

    # Per-episode records
    episodes: list[EpisodeSummary] = Field(default_factory=list)

    # Sampling decisions
    target_sample_size: int = Field(default=30, description="Target number of episodes to select")
    selected_ids: list[str] = Field(
        default_factory=list,
        description="Episode IDs selected for the ground truth set",
    )
    sampling_rationale: str = Field(
        default="",
        description="Human-readable explanation of why this sample was selected",
    )


class GroundTruthSplit(BaseModel):
    """The ground truth split — the selected subset of episodes for human annotation."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    source_manifest: str = Field(default="", description="Path to the sampling_manifest.json this was derived from")

    # Selection
    selected_episodes: list[str] = Field(
        default_factory=list,
        description="Episode IDs selected for ground truth annotation",
    )

    # Annotation instructions
    annotation_guidelines: str = Field(
        default="",
        description="Instructions for human annotators: what to look for, how to mark boundaries",
    )

    # Status tracking
    status: str = Field(default="pending_annotation", description="Current status of the ground truth set")
    annotated_count: int = Field(default=0, description="How many episodes have been annotated so far")
    annotators: list[str] = Field(default_factory=list, description="Who annotated which episodes")


# ---------------------------------------------------------------------------
# Batch segmentation / dataset runner schemas
# ---------------------------------------------------------------------------


class EpisodeSegmentResult(BaseModel):
    """Per-episode summary after batch segmentation."""

    episode_id: str
    source_path: str = Field(default="")
    status: str = Field(default="success", description="'success', 'error', or 'skipped'")
    stage_count: int = Field(default=0)
    boundary_count: int = Field(default=0)
    skill_labels: list[str] = Field(default_factory=list)
    num_frames: int = Field(default=0)
    method: str = Field(default="")
    elapsed_seconds: float = Field(default=0.0)
    error_message: str = Field(default="")
    artifacts_dir: str = Field(default="")


class DatasetSegmentIndex(BaseModel):
    """Dataset-level index collecting per-episode batch segmentation results."""

    dataset_root: str = Field(default="")
    config_path: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_episodes: int = Field(default=0)
    success_count: int = Field(default=0)
    error_count: int = Field(default=0)
    episodes: list[EpisodeSegmentResult] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)
