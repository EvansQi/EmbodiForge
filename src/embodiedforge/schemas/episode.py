"""Episode data schemas — raw input format."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class EpisodeMeta(BaseModel):
    """Metadata for a single robot episode."""

    episode_id: str = Field(..., description="Unique episode identifier")
    task_instruction: str = Field(
        ..., description="Natural language task instruction, e.g. 'pick up the red cup'"
    )
    robot_name: str = Field(default="unknown", description="Robot platform name")
    fps: float = Field(default=30.0, description="Recording frame rate")
    total_frames: int = Field(default=0, description="Total number of frames")
    has_depth: bool = Field(default=False, description="Whether depth images are available")
    has_force_torque: bool = Field(default=False, description="Whether F/T data is available")
    camera_intrinsics: Optional[dict] = Field(
        default=None, description="Camera intrinsic parameters (fx, fy, cx, cy)"
    )
    metadata: dict = Field(default_factory=dict, description="Arbitrary extra metadata")


class FrameData(BaseModel):
    """Data for a single frame within an episode.

    Images are stored as file paths (resolved at ingest time).
    Arrays are stored as numpy arrays or None.
    """

    frame_idx: int = Field(..., description="Frame index within episode")
    timestamp: float = Field(default=0.0, description="Timestamp in seconds")

    # Image paths (resolved relative to episode dir)
    front_rgb_path: Optional[str] = Field(default=None, description="Path to front RGB image")
    wrist_rgb_path: Optional[str] = Field(default=None, description="Path to wrist RGB image")
    front_depth_path: Optional[str] = Field(default=None, description="Path to front depth image")

    # Robot state
    state: Optional[list[float]] = Field(
        default=None, description="Robot state vector (e.g. joint positions + gripper)"
    )
    action: Optional[list[float]] = Field(
        default=None, description="Action applied at this frame"
    )
    gripper_state: Optional[float] = Field(
        default=None, description="Gripper opening width or binary state (0=open, 1=closed)"
    )

    # Force/torque
    force_torque: Optional[list[float]] = Field(
        default=None, description="Force/torque sensor reading [fx, fy, fz, tx, ty, tz]"
    )

    model_config = {"arbitrary_types_allowed": True}


class EpisodeData(BaseModel):
    """Complete episode with metadata and all frames."""

    meta: EpisodeMeta
    frames: list[FrameData] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)
