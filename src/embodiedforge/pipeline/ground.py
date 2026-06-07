"""Stage 3: Visual grounding — localize target object/part via bounding boxes.

Uses the configured GroundingAdapter to produce coarse bounding boxes
from the affordance_query or target_object text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from embodiedforge.adapters.factory import create_grounding_adapter
from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.geometry import BBox
from embodiedforge.schemas.semantic import SemanticResult


def ground_objects(
    episode: EpisodeData,
    semantic: SemanticResult,
    config: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[int, dict[str, BBox]]:
    """Ground semantic annotations to bounding boxes.

    For each annotated keyframe, runs grounding to find:
    - object_bbox: bounding box of the target object
    - part_bbox: bounding box of the target part (if query is different)

    Args:
        episode: The ingested episode data.
        semantic: Semantic annotation result.
        config: Grounding config (must include 'backend').
        artifacts_dir: Directory to save intermediate artifacts.

    Returns:
        Dict mapping frame_idx -> {'object_bbox': BBox, 'part_bbox': BBox}.
    """
    config = config or {}
    adapter = create_grounding_adapter(config)
    confidence_threshold = config.get("confidence_threshold", 0.3)

    results: dict[int, dict[str, BBox]] = {}

    for ann in semantic.annotations:
        frame_idx = ann.frame_idx
        if frame_idx >= len(episode.frames):
            continue

        frame = episode.frames[frame_idx]
        img_path = frame.front_rgb_path or frame.wrist_rgb_path
        if img_path is None:
            continue

        image = Image.open(img_path).convert("RGB")

        # Ground target object
        object_query = ann.target_object
        object_boxes = adapter.ground(image, object_query, confidence_threshold)
        object_bbox = object_boxes[0] if object_boxes else None

        # Ground target part (if different from object)
        part_query = f"{ann.target_object} {ann.target_part}" if ann.target_part else ann.target_object
        part_boxes = adapter.ground(image, part_query, confidence_threshold)
        part_bbox = part_boxes[0] if part_boxes else None

        results[frame_idx] = {
            "object_bbox": object_bbox,
            "part_bbox": part_bbox,
        }

    # Save artifact
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        import json
        artifact = {
            str(k): {
                "object_bbox": v["object_bbox"].model_dump() if v["object_bbox"] else None,
                "part_bbox": v["part_bbox"].model_dump() if v["part_bbox"] else None,
            }
            for k, v in results.items()
        }
        with open(artifacts_dir / "grounding_results.json", "w") as f:
            json.dump(artifact, f, indent=2)

    return results
