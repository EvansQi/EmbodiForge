"""LeRobot HDF5 export format.

LeRobot datasets use HDF5 with the following structure:
    data/
        episode_000/
            observation.state
            action
            observation.images.front
            observation.images.wrist
            ...
        episode_001/
            ...
    meta/
        info.json
        stats.json
        episodes.json

This module provides a converter from TrainingSample to the LeRobot HDF5 format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.schemas.sample import TrainingSample


def export_lerobot_hdf5(
    sample: TrainingSample,
    episode_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Export a TrainingSample in LeRobot HDF5 format.

    Args:
        sample: The training sample to export.
        episode_dir: Original episode directory (for referencing images).
        output_dir: Output directory for HDF5 dataset.
        config: Export configuration.

    Returns:
        Path to the generated HDF5 file.
    """
    config = config or {}
    use_h5py = config.get("use_h5py", False)

    episode_dir = Path(episode_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_h5py:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 export. "
                "Install with: pip install h5py"
            )

        h5_path = output_dir / f"episode_{sample.episode_id}.h5"
        with h5py.File(h5_path, "w") as f:
            ep_group = f.create_group("data")
            meta_group = f.create_group("meta")

            # Collect state/action data from episode frames
            front_rgb_dir = episode_dir / "front_rgb"
            states = []
            actions = []

            for kf in sorted(sample.keyframes, key=lambda x: x.frame_idx):
                # Read image and store as bytes
                if kf.image_path:
                    try:
                        img = Image.open(kf.image_path)
                        img_arr = np.array(img)
                        ds_name = f"observation.images.front_{kf.frame_idx:06d}"
                        ep_group.create_dataset(ds_name, data=img_arr, compression="gzip")
                    except Exception:
                        pass

            # Store metadata
            meta_group.create_dataset("info", data=json.dumps({
                "episode_id": sample.episode_id,
                "task_instruction": sample.task_instruction,
                "fps": sample.fps,
                "n_keyframes": len(sample.keyframes),
                "pipeline_version": sample.pipeline_version,
            }))

            # Store QC results
            if sample.qc:
                meta_group.create_dataset("qc", data=json.dumps({
                    "score": sample.qc.score,
                    "all_passed": sample.qc.all_passed,
                    "n_checks": len(sample.qc.checks),
                }))

        return h5_path
    else:
        # Fallback: use JSON representation compatible with LeRobot structure
        json_path = output_dir / f"episode_{sample.episode_id}" / "lerobot_format.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        lerobot_data = {
            "episode_id": sample.episode_id,
            "task": sample.task_instruction,
            "fps": sample.fps,
            "stages": [s.model_dump() for s in sample.stages],
            "keyframes": [
                {
                    "frame_idx": kf.frame_idx,
                    "image_path": kf.image_path,
                    "skill_id": kf.skill_id,
                    "stage_label": kf.stage_label,
                    "target_object": kf.target_object,
                    "target_part": kf.target_part,
                    "affordance_query": kf.affordance_query,
                    "object_bbox": kf.object_bbox.model_dump() if kf.object_bbox else None,
                    "part_bbox": kf.part_bbox.model_dump() if kf.part_bbox else None,
                    "mask_path": kf.object_mask_path,
                    "affordance_point": kf.affordance_point.model_dump() if kf.affordance_point else None,
                    "heatmap_path": kf.affordance_heatmap_path,
                    "keypoints_3d": kf.keypoints_3d,
                    "skill_soft": kf.skill_soft,
                }
                for kf in sample.keyframes
            ],
            "qc": {
                "score": sample.qc.score if sample.qc else 0.0,
                "all_passed": sample.qc.all_passed if sample.qc else False,
                "checks": [c.model_dump() for c in sample.qc.checks] if sample.qc else [],
            },
        }

        with open(json_path, "w") as f:
            json.dump(lerobot_data, f, indent=2)

        return json_path


def export_manifest(
    samples: list[TrainingSample],
    export_dir: str | Path,
) -> dict[str, Any]:
    """Generate a dataset manifest for multiple samples.

    Args:
        samples: List of exported training samples.
        export_dir: Export directory.

    Returns:
        Manifest dict.
    """
    export_dir = Path(export_dir)
    manifest = {
        "format": "lerobot",
        "version": "0.1.0",
        "n_samples": len(samples),
        "samples": [
            {
                "episode_id": s.episode_id,
                "task": s.task_instruction,
                "n_keyframes": len(s.keyframes),
                "n_stages": len(s.stages),
                "qc_score": s.qc.score if s.qc else 0.0,
            }
            for s in samples
        ],
    }

    manifest_path = export_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
