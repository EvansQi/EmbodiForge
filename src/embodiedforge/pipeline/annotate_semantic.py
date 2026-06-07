"""Stage 2: Semantic annotation — label keyframes with target object, part, affordance.

Uses the configured SemanticLabelerAdapter (mock or real VLM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from embodiedforge.adapters.factory import create_semantic_adapter
from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.segment import SegmentResult
from embodiedforge.schemas.semantic import SemanticAnnotation, SemanticResult


def annotate_semantic(
    episode: EpisodeData,
    segments: SegmentResult,
    config: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> SemanticResult:
    """Annotate keyframes with semantic labels.

    For each stage, loads the keyframe image and calls the semantic labeler
    to produce target_object, target_part, affordance_query, etc.

    Args:
        episode: The ingested episode data.
        segments: Stage segmentation result.
        config: Semantic annotation config (must include 'backend').
        artifacts_dir: Directory to save intermediate artifacts.

    Returns:
        SemanticResult with annotations per keyframe.
    """
    config = config or {}
    adapter = create_semantic_adapter(config)

    annotations: list[SemanticAnnotation] = []

    for stage in segments.stages:
        for kf_idx in stage.keyframe_indices:
            if kf_idx >= len(episode.frames):
                continue

            frame = episode.frames[kf_idx]

            # Load image
            img_path = frame.front_rgb_path or frame.wrist_rgb_path
            if img_path is None:
                continue
            image = Image.open(img_path).convert("RGB")

            # Call adapter
            annotation = adapter.label(
                image=image,
                task_instruction=episode.meta.task_instruction,
                skill_id=stage.skill_id,
                context={"frame_idx": kf_idx, "stage_id": stage.stage_id},
            )
            # Ensure frame_idx and stage_id are set correctly
            annotation.frame_idx = kf_idx
            annotation.stage_id = stage.stage_id
            annotation.skill_id = stage.skill_id
            annotations.append(annotation)

    result = SemanticResult(
        episode_id=episode.meta.episode_id,
        annotations=annotations,
        backend=config.get("backend", "mock"),
    )

    # Save artifact
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "semantic_annotations.json", "w") as f:
            f.write(result.model_dump_json(indent=2))

    return result
