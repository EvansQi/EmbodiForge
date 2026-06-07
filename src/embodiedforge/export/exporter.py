"""Export stage — assemble final TrainingSample and write to disk.

Output format (LeRobot-like):
    export_dir/
        episode_001/
            sample.json           # Full training sample metadata
            keyframes/
                000042.png        # Keyframe images (copied or linked)
            masks/
                mask_000042.png
            heatmaps/
                heatmap_000042.png
            viz/
                overlay_000042.png
        manifest.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from embodiedforge.export.lerobot_hdf5 import export_lerobot_hdf5
from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.geometry import AffordanceResult, Geometry3DResult, MaskResult, BBox
from embodiedforge.schemas.sample import ExportManifest, KeyframeAnnotation, QCSummary, TrainingSample
from embodiedforge.schemas.segment import SegmentResult
from embodiedforge.schemas.semantic import SemanticResult
from embodiedforge.schemas.rl import RLKeyframeSignal, RLSummary


def export_sample(
    episode: EpisodeData,
    segments: SegmentResult,
    semantic: SemanticResult,
    grounding_results: dict[int, dict[str, BBox]],
    mask_results: dict[int, MaskResult],
    affordance_results: dict[int, AffordanceResult],
    geometry_results: dict[int, Geometry3DResult],
    qc: QCSummary,
    export_dir: str | Path,
    config: dict[str, Any] | None = None,
    rl_signals: dict[int, RLKeyframeSignal] | None = None,
    rl_summary: RLSummary | None = None,
) -> TrainingSample:
    """Assemble and export a complete training sample.

    Args:
        episode: Ingested episode data.
        segments: Stage segmentation result.
        semantic: Semantic annotations.
        grounding_results: Grounding bounding boxes.
        mask_results: Mask generation results.
        affordance_results: Affordance heatmap results.
        geometry_results: 3D geometry results.
        qc: Quality control summary.
        export_dir: Output directory.
        config: Export configuration.

    Returns:
        The assembled TrainingSample.
    """
    config = config or {}
    rl_signals = rl_signals or {}
    export_dir = Path(export_dir)
    ep_dir = export_dir / episode.meta.episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (ep_dir / "keyframes").mkdir(exist_ok=True)
    (ep_dir / "masks").mkdir(exist_ok=True)
    (ep_dir / "heatmaps").mkdir(exist_ok=True)

    # Collect all keyframe indices from semantic annotations
    keyframe_indices = sorted(set(ann.frame_idx for ann in semantic.annotations))

    # Build keyframe annotations
    keyframes: list[KeyframeAnnotation] = []
    for kf_idx in keyframe_indices:
        frame = episode.frames[kf_idx] if kf_idx < len(episode.frames) else None
        img_path = frame.front_rgb_path if frame else None

        # Copy keyframe image
        dst_img = ""
        if img_path and Path(img_path).exists():
            dst = ep_dir / "keyframes" / f"{kf_idx:06d}.png"
            shutil.copy2(img_path, dst)
            dst_img = str(dst)

        # Get semantic annotation
        sem_ann = next((a for a in semantic.annotations if a.frame_idx == kf_idx), None)

        # Get grounding results
        ground = grounding_results.get(kf_idx, {})
        obj_bbox = ground.get("object_bbox")
        part_bbox = ground.get("part_bbox")

        # Get mask
        mask_res = mask_results.get(kf_idx)
        mask_path = mask_res.mask_path if mask_res else ""

        # Get affordance
        afford_res = affordance_results.get(kf_idx)
        afford_point = afford_res.affordance_point if afford_res else None
        heatmap_path = afford_res.heatmap_path if afford_res else ""

        # Get 3D geometry
        geo_res = geometry_results.get(kf_idx)
        kps3d = [kp.model_dump() for kp in geo_res.keypoints_3d] if geo_res else []

        kf = KeyframeAnnotation(
            frame_idx=kf_idx,
            image_path=dst_img,
            skill_id=sem_ann.skill_id if sem_ann else "",
            stage_label=sem_ann.stage_label if sem_ann else "",
            target_object=sem_ann.target_object if sem_ann else "",
            target_part=sem_ann.target_part if sem_ann else "",
            affordance_query=sem_ann.affordance_query if sem_ann else "",
            skill_soft=sem_ann.skill_soft if sem_ann else None,
            object_bbox=obj_bbox,
            part_bbox=part_bbox,
            object_mask_path=mask_path,
            affordance_point=afford_point,
            affordance_heatmap_path=heatmap_path,
            keypoints_3d=kps3d,
            qc=None,  # Per-keyframe QC can be added later
            rl=rl_signals.get(kf_idx),
        )
        keyframes.append(kf)

    # Build training sample
    sample = TrainingSample(
        episode_id=episode.meta.episode_id,
        task_instruction=episode.meta.task_instruction,
        fps=episode.meta.fps,
        stages=segments.stages,
        keyframes=keyframes,
        qc=qc,
        rl=rl_summary,
    )

    # Write sample.json (primary format)
    sample_path = ep_dir / "sample.json"
    with open(sample_path, "w") as f:
        f.write(sample.model_dump_json(indent=2))

    # Also export LeRobot HDF5 format
    try:
        h5_path = export_lerobot_hdf5(sample, "", ep_dir, config.get("lerobot", {}))
        print(f"  -> LeRobot format: {h5_path}")
    except Exception as e:
        print(f"  -> LeRobot HDF5 export skipped ({e})")

    return sample
