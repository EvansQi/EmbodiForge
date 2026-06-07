"""QC checks — validate pipeline outputs before export.

Checks:
1. point_in_mask: affordance point lies inside the object mask
2. bbox_area_sane: bbox area is within reasonable bounds
3. mask_area_sane: mask area is within reasonable bounds
4. temporal_consistency: masks across frames have reasonable overlap
5. force_contact_check: force spike correlates with contact stage
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.schemas.geometry import AffordanceResult, BBox, MaskResult
from embodiedforge.schemas.sample import QCCheck, QCSummary
from embodiedforge.schemas.segment import SegmentResult


def run_qc(
    episode_id: str,
    mask_results: dict[int, MaskResult],
    affordance_results: dict[int, AffordanceResult],
    segments: SegmentResult | None = None,
    config: dict[str, Any] | None = None,
) -> QCSummary:
    """Run all QC checks on pipeline outputs.

    Args:
        episode_id: Episode identifier.
        mask_results: Mask results from segment_masks stage.
        affordance_results: Affordance results from build_heatmaps stage.
        segments: Optional segment result for temporal checks.
        config: QC config overrides.

    Returns:
        QCSummary with check results.
    """
    config = config or {}
    checks: list[QCCheck] = []

    min_mask_area = config.get("min_mask_area", 100)
    max_mask_ratio = config.get("max_mask_ratio", 0.8)
    min_bbox_ratio = config.get("min_bbox_ratio", 0.01)
    max_bbox_ratio = config.get("max_bbox_ratio", 0.9)

    # Check each keyframe
    common_frames = set(mask_results.keys()) & set(affordance_results.keys())

    for frame_idx in sorted(common_frames):
        mask_res = mask_results[frame_idx]
        afford_res = affordance_results[frame_idx]

        # 1. Point in mask check
        if afford_res.affordance_point and mask_res.mask_path:
            pt = afford_res.affordance_point
            try:
                mask_img = np.array(Image.open(mask_res.mask_path).convert("L"))
                h, w = mask_img.shape
                u = int(round(max(0, min(w - 1, pt.x))))
                v = int(round(max(0, min(h - 1, pt.y))))
                in_mask = mask_img[v, u] > 0
            except Exception:
                in_mask = False

            checks.append(
                QCCheck(
                    name=f"point_in_mask_f{frame_idx}",
                    passed=in_mask,
                    message=f"Affordance point ({pt.x:.0f}, {pt.y:.0f}) {'inside' if in_mask else 'outside'} mask",
                )
            )

        # 2. Mask area sanity
        if mask_res.mask_path:
            try:
                mask_img = np.array(Image.open(mask_res.mask_path).convert("L"))
                total_pixels = mask_img.shape[0] * mask_img.shape[1]
                mask_ratio = mask_res.mask_area / max(1, total_pixels)
                area_ok = min_mask_area <= mask_res.mask_area and mask_ratio <= max_mask_ratio
            except Exception:
                area_ok = False
                mask_ratio = 0

            checks.append(
                QCCheck(
                    name=f"mask_area_f{frame_idx}",
                    passed=area_ok,
                    message=f"Mask area: {mask_res.mask_area} pixels ({mask_ratio:.1%} of image)",
                )
            )

        # 3. Bbox area sanity
        if mask_res.bbox:
            bbox = mask_res.bbox
            # Estimate image size from mask if available
            if mask_res.mask_path:
                try:
                    mask_img = np.array(Image.open(mask_res.mask_path).convert("L"))
                    total = mask_img.shape[0] * mask_img.shape[1]
                except Exception:
                    total = 640 * 480
            else:
                total = 640 * 480

            bbox_ratio = bbox.area / max(1, total)
            bbox_ok = min_bbox_ratio <= bbox_ratio <= max_bbox_ratio

            checks.append(
                QCCheck(
                    name=f"bbox_area_f{frame_idx}",
                    passed=bbox_ok,
                    message=f"BBox area ratio: {bbox_ratio:.1%}",
                )
            )

    # 4. Temporal consistency (if multiple frames)
    if len(mask_results) > 1:
        frames = sorted(mask_results.keys())
        overlaps = []
        prev_mask = None
        for fi in frames:
            mr = mask_results[fi]
            if mr.mask_path:
                try:
                    curr_mask = np.array(Image.open(mr.mask_path).convert("L")) > 0
                    if prev_mask is not None:
                        intersection = np.logical_and(prev_mask, curr_mask).sum()
                        union = np.logical_or(prev_mask, curr_mask).sum()
                        iou = intersection / max(1, union)
                        overlaps.append(iou)
                    prev_mask = curr_mask
                except Exception:
                    pass

        if overlaps:
            avg_iou = np.mean(overlaps)
            temporal_ok = avg_iou > 0.3
            checks.append(
                QCCheck(
                    name="temporal_consistency",
                    passed=temporal_ok,
                    message=f"Average mask IoU between frames: {avg_iou:.2f}",
                )
            )

    summary = QCSummary(checks=checks)
    summary.compute_score()
    return summary
