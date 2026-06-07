"""Stage 0: Ingest — load raw episode data from disk into EpisodeData schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.schemas.episode import EpisodeData, EpisodeMeta, FrameData


def ingest_episode(episode_dir: str | Path, config: dict[str, Any] | None = None) -> EpisodeData:
    """Load a raw episode from a directory into the EpisodeData schema.

    Expected directory layout:
        episode_dir/
            front_rgb/
                000000.png
                000001.png
                ...
            wrist_rgb/
                000000.png
                000001.png
                ...
            front_depth/         (optional)
                000000.png
                ...
            meta.json            (episode metadata)
            state.json           (robot state per frame)
            action.json          (actions per frame)

    Args:
        episode_dir: Path to the episode directory.
        config: Optional ingest configuration overrides.

    Returns:
        Populated EpisodeData.
    """
    episode_dir = Path(episode_dir)
    config = config or {}

    # Load metadata
    meta_path = episode_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta_raw = json.load(f)
    else:
        meta_raw = {}

    # Infer total frames from image directory
    front_rgb_dir = episode_dir / "front_rgb"
    wrist_rgb_dir = episode_dir / "wrist_rgb"

    if front_rgb_dir.exists():
        frame_files = sorted(front_rgb_dir.glob("*.png")) + sorted(front_rgb_dir.glob("*.jpg"))
        total_frames = len(frame_files)
    elif wrist_rgb_dir.exists():
        frame_files = sorted(wrist_rgb_dir.glob("*.png")) + sorted(wrist_rgb_dir.glob("*.jpg"))
        total_frames = len(frame_files)
    else:
        raise FileNotFoundError(f"No image directory found in {episode_dir}")

    # Build frame index -> filename mapping
    def _frame_path(directory: Path, idx: int) -> str | None:
        for ext in [".png", ".jpg", ".jpeg"]:
            p = directory / f"{idx:06d}{ext}"
            if p.exists():
                return str(p)
        return None

    meta = EpisodeMeta(
        episode_id=meta_raw.get("episode_id", episode_dir.name),
        task_instruction=meta_raw.get("task_instruction", config.get("task_instruction", "pick up the object")),
        robot_name=meta_raw.get("robot_name", "unknown"),
        fps=meta_raw.get("fps", config.get("fps", 30.0)),
        total_frames=total_frames,
        has_depth=(episode_dir / "front_depth").exists(),
        has_force_torque="force_torque" in meta_raw or (episode_dir / "force_torque.json").exists(),
        camera_intrinsics=meta_raw.get("camera_intrinsics"),
        metadata=meta_raw,
    )

    # Load state/action data
    state_data = _load_json_list(episode_dir / "state.json")
    action_data = _load_json_list(episode_dir / "action.json")
    gripper_data = _load_json_list(episode_dir / "gripper.json")
    ft_data = _load_json_list(episode_dir / "force_torque.json")

    # Build frames
    frames: list[FrameData] = []
    for idx in range(total_frames):
        frame = FrameData(
            frame_idx=idx,
            timestamp=idx / meta.fps,
            front_rgb_path=_frame_path(front_rgb_dir, idx),
            wrist_rgb_path=_frame_path(wrist_rgb_dir, idx),
            front_depth_path=_frame_path(episode_dir / "front_depth", idx) if meta.has_depth else None,
            state=state_data[idx] if state_data and idx < len(state_data) else None,
            action=action_data[idx] if action_data and idx < len(action_data) else None,
            gripper_state=gripper_data[idx] if gripper_data and idx < len(gripper_data) else None,
            force_torque=ft_data[idx] if ft_data and idx < len(ft_data) else None,
        )
        frames.append(frame)

    return EpisodeData(meta=meta, frames=frames)


def _load_json_list(path: Path) -> list | None:
    """Load a JSON file expected to contain a list. Returns None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return None
