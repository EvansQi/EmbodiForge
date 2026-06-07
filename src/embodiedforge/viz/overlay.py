"""Visualization — generate overlay images showing bboxes, masks, points, heatmaps.

Produces composite visualization images for debugging and inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from embodiedforge.schemas.geometry import AffordanceResult, BBox, MaskResult, Point2D


def generate_overlays(
    episode_dir: str | Path,
    mask_results: dict[int, MaskResult],
    affordance_results: dict[int, AffordanceResult],
    grounding_results: dict[int, dict[str, BBox]],
    artifacts_dir: str | Path | None = None,
) -> dict[int, str]:
    """Generate overlay visualization images for each keyframe.

    Each overlay shows:
    - Original image
    - Object bbox (green)
    - Part bbox (cyan)
    - Object mask (semi-transparent blue)
    - Affordance point (red cross)
    - Affordance heatmap (jet colormap, semi-transparent)

    Args:
        episode_dir: Path to episode directory.
        mask_results: Mask results.
        affordance_results: Affordance results.
        grounding_results: Grounding results (bboxes).
        artifacts_dir: Directory to save overlay images.

    Returns:
        Dict mapping frame_idx -> overlay image path.
    """
    episode_dir = Path(episode_dir)
    viz_dir = Path(artifacts_dir) / "viz" if artifacts_dir else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, str] = {}

    # Collect all frames that have any annotation
    all_frames = set(mask_results.keys()) | set(affordance_results.keys()) | set(grounding_results.keys())

    for frame_idx in sorted(all_frames):
        # Load source image
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = episode_dir / "front_rgb" / f"{frame_idx:06d}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        overlay = img.copy()

        # Draw mask (semi-transparent blue)
        mask_res = mask_results.get(frame_idx)
        if mask_res and mask_res.mask_path and Path(mask_res.mask_path).exists():
            mask_img = cv2.imread(mask_res.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask_bool = mask_img > 0
                blue_layer = np.zeros_like(overlay)
                blue_layer[mask_bool] = [255, 100, 0]  # Blue in BGR
                overlay = cv2.addWeighted(overlay, 0.7, blue_layer, 0.3, 0)

        # Draw object bbox (green)
        ground = grounding_results.get(frame_idx, {})
        obj_bbox = ground.get("object_bbox")
        if obj_bbox:
            cv2.rectangle(
                overlay,
                (int(obj_bbox.x1), int(obj_bbox.y1)),
                (int(obj_bbox.x2), int(obj_bbox.y2)),
                (0, 255, 0), 2,
            )
            cv2.putText(
                overlay, f"obj:{obj_bbox.confidence:.2f}",
                (int(obj_bbox.x1), int(obj_bbox.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        # Draw part bbox (cyan)
        part_bbox = ground.get("part_bbox")
        if part_bbox:
            cv2.rectangle(
                overlay,
                (int(part_bbox.x1), int(part_bbox.y1)),
                (int(part_bbox.x2), int(part_bbox.y2)),
                (255, 255, 0), 2,
            )
            cv2.putText(
                overlay, f"part:{part_bbox.confidence:.2f}",
                (int(part_bbox.x1), int(part_bbox.y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

        # Draw affordance heatmap (jet colormap)
        afford_res = affordance_results.get(frame_idx)
        if afford_res and afford_res.heatmap_path and Path(afford_res.heatmap_path).exists():
            heatmap_gray = cv2.imread(afford_res.heatmap_path, cv2.IMREAD_GRAYSCALE)
            if heatmap_gray is not None:
                heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(overlay, 0.6, heatmap_color, 0.4, 0)

        # Draw affordance point (red cross)
        if afford_res and afford_res.affordance_point:
            pt = afford_res.affordance_point
            cx, cy = int(pt.x), int(pt.y)
            cv2.drawMarker(
                overlay, (cx, cy), (0, 0, 255),
                cv2.MARKER_CROSS, 20, 2,
            )

        # Add text info
        info_text = f"Frame {frame_idx}"
        cv2.putText(
            overlay, info_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        # Save
        if viz_dir:
            out_path = viz_dir / f"overlay_{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), overlay)
            results[frame_idx] = str(out_path)

    return results
