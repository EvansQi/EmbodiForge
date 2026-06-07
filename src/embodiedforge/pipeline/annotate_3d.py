"""Stage 6: 3D geometry annotation — depth -> 3D keypoints.

Uses the configured DepthAdapter to estimate depth and convert
2D affordance points to 3D camera-frame coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.adapters.factory import create_depth_adapter
from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.geometry import AffordanceResult, Geometry3DResult, Keypoint3D


def annotate_3d(
    episode: EpisodeData,
    affordance_results: dict[int, AffordanceResult],
    config: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[int, Geometry3DResult]:
    """Annotate keyframes with 3D geometry.

    For each keyframe with an affordance point:
    1. Estimate depth map (from sensor or monocular estimation)
    2. Look up depth at the affordance point
    3. Convert to 3D camera-frame coordinates

    Args:
        episode: The ingested episode data.
        affordance_results: Affordance point/heatmap results.
        config: 3D annotation config (must include 'backend').
        artifacts_dir: Directory to save artifacts.

    Returns:
        Dict mapping frame_idx -> Geometry3DResult.
    """
    config = config or {}
    adapter = create_depth_adapter(config)
    intrinsics = episode.meta.camera_intrinsics or {
        "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0
    }

    results: dict[int, Geometry3DResult] = {}

    for frame_idx, affordance in affordance_results.items():
        if affordance.affordance_point is None:
            continue
        if frame_idx >= len(episode.frames):
            continue

        frame = episode.frames[frame_idx]
        img_path = frame.front_rgb_path or frame.wrist_rgb_path
        if img_path is None:
            continue

        image = Image.open(img_path).convert("RGB")

        # Get depth map
        depth_map = adapter.estimate_depth(image, intrinsics)

        # Look up depth at affordance point
        pt = affordance.affordance_point
        u, v = int(round(pt.x)), int(round(pt.y))
        h, w = depth_map.shape
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        depth_val = float(depth_map[v, u])

        # Convert to 3D
        x, y, z = adapter.pixel_to_3d(pt.x, pt.y, depth_val, intrinsics)

        keypoints = [
            Keypoint3D(x=x, y=y, z=z, label="affordance_point_3d", confidence=0.8)
        ]

        results[frame_idx] = Geometry3DResult(
            frame_idx=frame_idx,
            keypoints_3d=keypoints,
            backend=config.get("backend", "mock"),
        )

    # Save artifact
    if artifacts_dir:
        import json
        artifact = {str(k): v.model_dump() for k, v in results.items()}
        with open(Path(artifacts_dir) / "geometry_3d.json", "w") as f:
            json.dump(artifact, f, indent=2)

    return results
