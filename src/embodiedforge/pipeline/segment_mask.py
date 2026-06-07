"""Stage 4: Mask generation — produce segmentation masks from bbox/point prompts.

Uses the configured SegmentationAdapter (SAM2 or mock).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.adapters.factory import create_segmentation_adapter
from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.geometry import BBox, MaskResult


def segment_masks(
    episode: EpisodeData,
    grounding_results: dict[int, dict[str, BBox]],
    config: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[int, MaskResult]:
    """Generate object masks for grounded keyframes.

    Args:
        episode: The ingested episode data.
        grounding_results: Output from ground_objects stage.
        config: Segmentation config (must include 'backend').
        artifacts_dir: Directory to save mask images.

    Returns:
        Dict mapping frame_idx -> MaskResult.
    """
    config = config or {}
    adapter = create_segmentation_adapter(config)

    results: dict[int, MaskResult] = {}
    mask_dir = Path(artifacts_dir) / "masks" if artifacts_dir else None
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, boxes in grounding_results.items():
        if frame_idx >= len(episode.frames):
            continue

        frame = episode.frames[frame_idx]
        img_path = frame.front_rgb_path or frame.wrist_rgb_path
        if img_path is None:
            continue

        image = Image.open(img_path).convert("RGB")

        # Use object bbox as prompt
        obj_bbox = boxes.get("object_bbox")
        if obj_bbox is None:
            continue

        # Generate mask
        mask = adapter.segment(
            image=image,
            bboxes=[obj_bbox],
        )

        # Compute mask area
        mask_area = int(np.count_nonzero(mask))

        # Save mask image
        mask_path = ""
        if mask_dir:
            mask_file = mask_dir / f"mask_{frame_idx:06d}.png"
            Image.fromarray(mask).save(str(mask_file))
            mask_path = str(mask_file)

        results[frame_idx] = MaskResult(
            frame_idx=frame_idx,
            mask_path=mask_path,
            mask_area=mask_area,
            bbox=obj_bbox,
            backend=config.get("backend", "mock"),
        )

    # Save artifact
    if artifacts_dir:
        import json
        artifact = {str(k): v.model_dump() for k, v in results.items()}
        with open(Path(artifacts_dir) / "mask_results.json", "w") as f:
            json.dump(artifact, f, indent=2)

    return results
