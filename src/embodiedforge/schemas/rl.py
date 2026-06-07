"""RL-oriented schemas built from processed robot episodes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewardTerm(BaseModel):
    """One interpretable reward contribution."""

    name: str = Field(..., description="Reward term name")
    value: float = Field(default=0.0, description="Raw reward value before aggregation")
    weight: float = Field(default=1.0, description="Weight applied to this term")
    weighted_value: float = Field(default=0.0, description="Final weighted contribution")


class BinaryClassifierTarget(BaseModel):
    """A binary supervision target useful for success or reward models."""

    name: str = Field(..., description="Classifier target name")
    value: bool = Field(default=False, description="Binary label")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RLKeyframeSignal(BaseModel):
    """RL-friendly signals derived for a single keyframe."""

    frame_idx: int
    stage_id: int = Field(default=0)
    skill_id: str = Field(default="")
    dense_reward: float = Field(default=0.0, description="Aggregated dense reward")
    sparse_reward: float = Field(default=0.0, description="Sparse success-oriented reward")
    advantage_hint: float = Field(
        default=0.0,
        description="Heuristic quality hint that can seed value/reward experiments",
    )
    stage_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    subgoal_completed: bool = Field(default=False)
    success: bool = Field(default=False)
    terminal: bool = Field(default=False)
    contact_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_terms: list[RewardTerm] = Field(default_factory=list)
    classifier_targets: list[BinaryClassifierTarget] = Field(default_factory=list)


class RLSummary(BaseModel):
    """Episode-level RL signal summary."""

    enabled: bool = Field(default=False)
    reward_scheme: str = Field(default="heuristic_v1")
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    success: bool = Field(default=False)
    total_dense_reward: float = Field(default=0.0)
    total_sparse_reward: float = Field(default=0.0)
    mean_contact_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_alignment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    success_criteria: list[str] = Field(default_factory=list)
