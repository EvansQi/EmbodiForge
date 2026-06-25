"""Dataset scanner — walks a directory tree, extracts episode metadata.

Handles two episode formats:
1. Standard:  episode_dir/meta.json + front_rgb/*.png + state.json + gripper.json
2. XLSX:     episode_dir/episode_data.xlsx + session_meta.json + *_rgb_*.jpg
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from embodiedforge.sampling.schemas import EpisodeSummary


# Known task type keywords for heuristic classification
_TASK_KEYWORDS: dict[str, list[str]] = {
    "insert_usb": ["usb", "insert"],
    "pick_place": ["pick up", "pick and place", "pick_place", "grasp"],
    "peg_in_hole": ["peg", "hole", "insert peg"],
    "drawer": ["drawer", "open drawer", "close drawer"],
    "button_press": ["button", "press"],
    "push": ["push", "slide"],
    "pour": ["pour", "fill"],
}


def _infer_task_type(task_instruction: str, extra: dict[str, Any] | None = None) -> str:
    """Heuristic task type classification from instruction text."""
    lower = task_instruction.lower()
    for task_type, keywords in _TASK_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return task_type
    # Check extra metadata
    extra = extra or {}
    if extra.get("task_name"):
        tn = extra["task_name"].lower()
        for task_type, keywords in _TASK_KEYWORDS.items():
            if any(kw in tn for kw in keywords):
                return task_type
    return "unknown"


def _infer_success_from_signals(
    gripper_data: list[float] | None,
    gripper_close_threshold: float = 0.5,
    min_open_after_close_frames: int = 2,
) -> tuple[Optional[bool], Optional[int], Optional[int]]:
    """Infer episode success from gripper close→open pattern.

    A successful manipulation typically involves:
    1. Gripper closes (grasp) → 2. Gripper opens (release)

    Handles both normalized (0-1) and raw servo values (e.g. 39=closed, 151=open).

    Returns (success, close_frame, open_frame).
    """
    if gripper_data is None or len(gripper_data) < 2:
        return None, None, None

    # Auto-detect raw vs normalized gripper values
    max_val = max(gripper_data)
    if max_val > 2.0:
        # Raw servo values: closed ≈ 39-111, open ≈ 149-151
        close_threshold = 120.0  # midpoint
        is_closed = lambda v: v < close_threshold
        is_open = lambda v: v >= close_threshold
    else:
        # Normalized 0-1 values
        is_closed = lambda v: v < gripper_close_threshold
        is_open = lambda v: v >= gripper_close_threshold

    close_frame: Optional[int] = None
    open_frame: Optional[int] = None

    for idx, val in enumerate(gripper_data):
        if close_frame is None and is_closed(val):
            close_frame = idx
        elif close_frame is not None and is_open(val):
            # Count consecutive open frames
            run = 0
            for j in range(idx, len(gripper_data)):
                if is_open(gripper_data[j]):
                    run += 1
                else:
                    break
            if run >= min_open_after_close_frames:
                open_frame = idx
                break

    if close_frame is not None and open_frame is not None:
        return True, close_frame, open_frame
    if close_frame is not None:
        return False, close_frame, None
    return None, None, None


def _compute_trajectory_length(
    state_data: list[list[float]] | None,
    position_dims: int = 3,
) -> float:
    """Compute cumulative displacement of first N dimensions of state."""
    if state_data is None or len(state_data) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(state_data)):
        prev = state_data[i - 1]
        curr = state_data[i]
        dims = min(position_dims, len(prev), len(curr))
        displacement = math.sqrt(sum((curr[d] - prev[d]) ** 2 for d in range(dims)))
        total += displacement
    return total


def _scan_standard_episode(episode_dir: Path) -> EpisodeSummary:
    """Scan a standard-format episode (meta.json + image dirs + signal jsons)."""
    episode_id = episode_dir.name
    missing: list[str] = []

    # meta.json
    meta_path = episode_dir / "meta.json"
    meta_raw: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta_raw = json.load(f)
    else:
        missing.append("meta.json")

    task_instruction = meta_raw.get("task_instruction", "")
    fps = float(meta_raw.get("fps", 30.0))

    # Image directories
    front_rgb_dir = episode_dir / "front_rgb"
    wrist_rgb_dir = episode_dir / "wrist_rgb"
    front_depth_dir = episode_dir / "front_depth"

    has_front = front_rgb_dir.exists() and any(front_rgb_dir.glob("*.png")) or any(front_rgb_dir.glob("*.jpg"))
    has_wrist = wrist_rgb_dir.exists() and any(wrist_rgb_dir.glob("*.png")) or any(wrist_rgb_dir.glob("*.jpg"))
    has_depth = front_depth_dir.exists() and any(front_depth_dir.glob("*.png"))

    if not has_front and not has_wrist:
        missing.append("front_rgb/ or wrist_rgb/ (no images found)")

    # Count frames from image directory
    if has_front:
        frame_files = sorted(front_rgb_dir.glob("*.png")) + sorted(front_rgb_dir.glob("*.jpg"))
        num_frames = len(frame_files)
    elif has_wrist:
        frame_files = sorted(wrist_rgb_dir.glob("*.png")) + sorted(wrist_rgb_dir.glob("*.jpg"))
        num_frames = len(frame_files)
    else:
        num_frames = 0

    # Camera setup label
    if has_front and has_wrist:
        camera_setup = "both"
    elif has_front:
        camera_setup = "front_only"
    elif has_wrist:
        camera_setup = "wrist_only"
    else:
        camera_setup = "none"

    # Signal files
    gripper_path = episode_dir / "gripper.json"
    state_path = episode_dir / "state.json"
    ft_path = episode_dir / "force_torque.json"

    gripper_data: list[float] | None = None
    if gripper_path.exists():
        with open(gripper_path, encoding="utf-8") as f:
            gripper_data = json.load(f)
    else:
        missing.append("gripper.json")

    state_data: list[list[float]] | None = None
    if state_path.exists():
        with open(state_path, encoding="utf-8") as f:
            state_data = json.load(f)
    else:
        missing.append("state.json")

    has_ft = ft_path.exists()

    # Derived metrics
    success, close_frame, open_frame = _infer_success_from_signals(gripper_data)
    trajectory_len = _compute_trajectory_length(state_data)

    task_type = _infer_task_type(task_instruction, meta_raw)

    return EpisodeSummary(
        episode_id=episode_id,
        source_path=str(episode_dir.absolute()),
        format_type="standard",
        task_type=task_type,
        task_instruction=task_instruction,
        num_frames=num_frames,
        fps=fps,
        duration_seconds=num_frames / fps if fps > 0 else 0.0,
        has_front_rgb=has_front,
        has_wrist_rgb=has_wrist,
        has_depth=has_depth,
        camera_setup=camera_setup,
        has_gripper=gripper_data is not None,
        has_force_torque=has_ft,
        has_state=state_data is not None,
        trajectory_length=round(trajectory_len, 4),
        success=success,
        gripper_close_frame=close_frame,
        gripper_open_frame=open_frame,
        extra=meta_raw,
        missing_files=missing,
    )


def _scan_xlsx_episode(episode_dir: Path) -> EpisodeSummary:
    """Scan an xlsx-format episode (episode_data.xlsx + session_meta.json)."""
    episode_id = episode_dir.name
    missing: list[str] = []

    xlsx_path = episode_dir / "episode_data.xlsx"
    if not xlsx_path.exists():
        missing.append("episode_data.xlsx")

    session_meta_path = episode_dir / "session_meta.json"
    session_meta: dict[str, Any] = {}
    if session_meta_path.exists():
        with open(session_meta_path, encoding="utf-8") as f:
            session_meta = json.load(f)
    else:
        missing.append("session_meta.json")

    task_name = session_meta.get("task_name", "")
    task_instruction = session_meta.get("task_description", task_name)

    fps_value = (
        session_meta.get("record_frequency", {}).get("actual_hz")
        or session_meta.get("config", {}).get("target_save_hz")
        or 20.0
    )
    fps = float(fps_value) if fps_value else 20.0

    camera_keys = session_meta.get("camera_keys", ["RealSense_head", "RealSense_wrist"])
    head_prefix = next((k for k in camera_keys if "head" in k.lower()), "RealSense_head")
    wrist_prefix = next((k for k in camera_keys if "wrist" in k.lower()), "RealSense_wrist")

    # Count frames from .jpg images
    head_images = sorted(episode_dir.glob(f"{head_prefix}_rgb_*.jpg"))
    wrist_images = sorted(episode_dir.glob(f"{wrist_prefix}_rgb_*.jpg"))
    has_front = len(head_images) > 0
    has_wrist = len(wrist_images) > 0
    num_frames = max(len(head_images), len(wrist_images))

    if num_frames == 0:
        missing.append(f"{head_prefix}_rgb_*.jpg or {wrist_prefix}_rgb_*.jpg")

    if has_front and has_wrist:
        camera_setup = "both"
    elif has_front:
        camera_setup = "front_only"
    elif has_wrist:
        camera_setup = "wrist_only"
    else:
        camera_setup = "none"

    # Extract signals from xlsx if available
    gripper_data: list[float] | None = None
    state_data: list[list[float]] | None = None
    has_gripper = False
    has_state = False
    has_ft = False

    if xlsx_path.exists():
        try:
            import pandas as pd

            df = pd.read_excel(xlsx_path)
            if "right_arm_gripper" in df.columns:
                gripper_data = [float(v) for v in df["right_arm_gripper"].tolist()]
                has_gripper = True
            state_cols = [f"right_arm_state_{axis}" for axis in range(6)]
            if all(c in df.columns for c in state_cols[:3]):
                state_data = [[float(df.at[i, c]) for c in state_cols if c in df.columns] for i in range(len(df))]
                has_state = True
            if "right_arm_ft_0" in df.columns or "force_torque_0" in df.columns:
                has_ft = True
        except Exception:
            pass

    success, close_frame, open_frame = _infer_success_from_signals(gripper_data)
    trajectory_len = _compute_trajectory_length(state_data)

    task_type = _infer_task_type(task_instruction, session_meta)

    return EpisodeSummary(
        episode_id=episode_id,
        source_path=str(episode_dir.absolute()),
        format_type="xlsx",
        task_type=task_type,
        task_instruction=task_instruction,
        num_frames=num_frames,
        fps=fps,
        duration_seconds=num_frames / fps if fps > 0 else 0.0,
        has_front_rgb=has_front,
        has_wrist_rgb=has_wrist,
        has_depth=False,
        camera_setup=camera_setup,
        has_gripper=has_gripper,
        has_force_torque=has_ft,
        has_state=has_state,
        trajectory_length=round(trajectory_len, 4),
        success=success,
        gripper_close_frame=close_frame,
        gripper_open_frame=open_frame,
        extra=session_meta,
        missing_files=missing,
    )


def _is_episode_directory(path: Path) -> bool:
    """Check if a directory looks like an episode (contains expected files)."""
    # Standard format
    if (path / "meta.json").exists():
        return True
    if (path / "front_rgb").exists() or (path / "wrist_rgb").exists():
        return True
    # XLSX format
    if (path / "episode_data.xlsx").exists():
        return True
    if (path / "session_meta.json").exists():
        return True
    return False


def scan_dataset(
    dataset_root: str | Path,
    max_episodes: int = 0,
    skip_incomplete: bool = False,
) -> list[EpisodeSummary]:
    """Scan a dataset directory tree for episodes.

    Args:
        dataset_root: Root directory containing episode subdirectories.
        max_episodes: Maximum episodes to scan (0 = unlimited). Useful for quick checks.
        skip_incomplete: If True, skip episodes with missing files.

    Returns:
        List of EpisodeSummary objects, one per discovered episode.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    episodes: list[EpisodeSummary] = []
    candidates: list[Path] = []

    # Strategy: look two levels deep for episode directories
    # Level 1: direct subdirectories of dataset_root
    # Level 2: subdirectories of subdirectories (e.g., dataset_root/task_type/episode_001)
    for entry in sorted(dataset_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        if _is_episode_directory(entry):
            candidates.append(entry)
        else:
            # Try one level deeper
            for sub_entry in sorted(entry.iterdir()):
                if sub_entry.is_dir() and _is_episode_directory(sub_entry):
                    candidates.append(sub_entry)

    # Deduplicate while preserving order
    seen = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            unique_candidates.append(c)

    for ep_dir in unique_candidates:
        if max_episodes > 0 and len(episodes) >= max_episodes:
            break

        try:
            summary = scan_single_episode(ep_dir)
            if skip_incomplete and not summary.is_complete:
                continue
            episodes.append(summary)
        except Exception as exc:
            # Don't let one broken episode kill the whole scan
            episodes.append(
                EpisodeSummary(
                    episode_id=ep_dir.name,
                    source_path=str(ep_dir.absolute()),
                    missing_files=[f"scan error: {exc}"],
                )
            )

    return episodes


def scan_single_episode(episode_dir: str | Path) -> EpisodeSummary:
    """Scan a single episode directory and return its summary.

    Auto-detects format (standard vs xlsx).
    """
    episode_dir = Path(episode_dir)
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    # Detect format
    has_xlsx = (episode_dir / "episode_data.xlsx").exists()
    has_meta = (episode_dir / "meta.json").exists() or (episode_dir / "front_rgb").exists()

    if has_xlsx:
        return _scan_xlsx_episode(episode_dir)
    elif has_meta:
        return _scan_standard_episode(episode_dir)
    else:
        return EpisodeSummary(
            episode_id=episode_dir.name,
            source_path=str(episode_dir.absolute()),
            missing_files=["unrecognized format — no meta.json or episode_data.xlsx found"],
        )
