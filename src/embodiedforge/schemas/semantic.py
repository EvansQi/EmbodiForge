"""Semantic annotation schemas — target object, part, affordance query."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SemanticAnnotation(BaseModel):
    """Semantic annotation for a single keyframe."""

    frame_idx: int = Field(..., description="Frame index this annotation refers to")
    stage_id: int = Field(..., description="Which stage this frame belongs to")
    skill_id: str = Field(default="", description="Skill identifier for this stage")

    # Semantic labels
    stage_label: str = Field(
        default="", description="Human-readable stage description, e.g. 'reaching toward cup'"
    )
    target_object: str = Field(
        default="", description="Target object name, e.g. 'red cup', 'USB plug'"
    )
    target_part: str = Field(
        default="", description="Target part name, e.g. 'handle', 'connector tip'"
    )
    affordance_query: str = Field(
        default="",
        description="Affordance description, e.g. 'grasp point on cup handle'",
    )

    # Optional skill soft label (probability distribution over skills)
    skill_soft: Optional[dict[str, float]] = Field(
        default=None,
        description="Soft skill label, e.g. {'reach': 0.1, 'grasp': 0.8, 'lift': 0.1}",
    )

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Annotation confidence"
    )


class SemanticResult(BaseModel):
    """Result of the semantic annotation stage."""

    episode_id: str
    annotations: list[SemanticAnnotation] = Field(default_factory=list)
    backend: str = Field(default="mock", description="Which backend produced this")
