"""Stage 5: Affordance heatmap — generate affordance maps from point + mask.

Creates a Gaussian heatmap centered on the affordance point,
modulated by the object mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.geometry import AffordanceResult, BBox, MaskResult, Point2D
from embodiedforge.schemas.semantic import SemanticResult


def build_heatmaps(
    episode: EpisodeData,
    semantic: SemanticResult,
    mask_results: dict[int, MaskResult],
    grounding_results: dict[int, dict[str, BBox]],
    config: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[int, AffordanceResult]:
    """Build affordance heatmaps for each keyframe.

    The affordance point is derived from the part bbox center (or object bbox center).
    A 2D Gaussian heatmap is generated and masked by the object segmentation.

    Args:
        episode: The ingested episode data.
        semantic: Semantic annotations.
        mask_results: Mask generation results.
        grounding_results: Grounding results (bboxes).
        config: Heatmap config.
        artifacts_dir: Directory to save heatmap images.

    Returns:
        Dict mapping frame_idx -> AffordanceResult.
    """
    config = config or {}
    sigma_factor = config.get("sigma_factor", 0.15)  # sigma relative to image size

    results: dict[int, AffordanceResult] = {}
    heatmap_dir = Path(artifacts_dir) / "heatmaps" if artifacts_dir else None
    if heatmap_dir:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    for ann in semantic.annotations:
        frame_idx = ann.frame_idx
        if frame_idx not in mask_results:
            continue

        frame = episode.frames[frame_idx]
        img_path = frame.front_rgb_path or frame.wrist_rgb_path
        if img_path is None:
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Determine affordance point from part bbox or object bbox
        boxes = grounding_results.get(frame_idx, {})
        part_bbox = boxes.get("part_bbox")
        obj_bbox = boxes.get("object_bbox")
        target_bbox = part_bbox or obj_bbox

        if target_bbox is None:
            continue

        # Affordance point = center of target bbox
        cx, cy = target_bbox.center
        affordance_point = Point2D(x=cx, y=cy, label="affordance_point")

        # Generate Gaussian heatmap
        sigma_x = w * sigma_factor
        sigma_y = h * sigma_factor
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        heatmap = np.exp(
            -((x_coords - cx) ** 2 / (2 * sigma_x**2) + (y_coords - cy) ** 2 / (2 * sigma_y**2))
        )

        # Modulate by mask
        mask_result = mask_results[frame_idx]
        if mask_result.mask_path:
            mask_img = np.array(Image.open(mask_result.mask_path).convert("L"))
            heatmap = heatmap * (mask_img.astype(np.float32) / 255.0)

        # Normalize to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        # Save heatmap image
        heatmap_path = ""
        if heatmap_dir:
            # Save as colored overlay
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap_uint8, mode="L")
            heatmap_file = heatmap_dir / f"heatmap_{frame_idx:06d}.png"
            heatmap_img.save(str(heatmap_file))
            heatmap_path = str(heatmap_file)

        results[frame_idx] = AffordanceResult(
            frame_idx=frame_idx,
            affordance_point=affordance_point,
            heatmap_path=heatmap_path,
            heatmap_max=float(max_val),
            backend="computed",
        )

    # Save artifact
    if artifacts_dir:
        import json
        artifact = {str(k): v.model_dump() for k, v in results.items()}
        with open(Path(artifacts_dir) / "affordance_results.json", "w") as f:
            json.dump(artifact, f, indent=2)

    return results
