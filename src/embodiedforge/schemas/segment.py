"""Segmentation schemas — stage boundaries and keyframes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StageSegment(BaseModel):
    """A single stage within an episode, identified by temporal boundaries."""

    stage_id: int = Field(..., description="Stage index (0-based)")
    skill_id: str = Field(
        ..., description="Skill identifier, e.g. 'reach', 'grasp', 'lift', 'insert'"
    )
    start_frame: int = Field(..., description="First frame of this stage (inclusive)")
    end_frame: int = Field(..., description="Last frame of this stage (inclusive)")
    keyframe_indices: list[int] = Field(
        default_factory=list,
        description="Important frame indices within this stage (e.g. contact onset, peak force)",
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Segmentation confidence"
    )


class SegmentResult(BaseModel):
    """Result of the temporal segmentation stage."""

    episode_id: str
    stages: list[StageSegment] = Field(default_factory=list)
    method: str = Field(default="gripper_velocity_force", description="Segmentation method used")

    @property
    def total_stages(self) -> int:
        return len(self.stages)

    def get_stage_at_frame(self, frame_idx: int) -> StageSegment | None:
        for s in self.stages:
            if s.start_frame <= frame_idx <= s.end_frame:
                return s
        return None
