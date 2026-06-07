"""Stage 1: Temporal segmentation — split episode into stages, find keyframes.

Uses gripper state, velocity (from state diffs), and force/torque signals
to detect stage boundaries (e.g. approach, contact, grasp, lift).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.segment import SegmentResult, StageSegment

# Default skill sequence for a pick-and-place task
_DEFAULT_SKILLS = ["reach", "grasp", "lift", "place"]


def segment_episode(
    episode: EpisodeData,
    config: dict[str, Any] | None = None,
    artifacts_dir: str | Path | None = None,
) -> SegmentResult:
    """Segment an episode into temporal stages.

    Detection heuristics:
    1. Gripper state changes -> grasp / release boundaries
    2. Velocity (state diff magnitude) -> motion / stillness transitions
    3. Force spikes -> contact onset

    Args:
        episode: The ingested episode data.
        config: Segmentation config overrides.
        artifacts_dir: Directory to save intermediate artifacts.

    Returns:
        SegmentResult with stage boundaries and keyframes.
    """
    config = config or {}
    skills = config.get("skills", _DEFAULT_SKILLS)
    force_threshold = config.get("force_threshold", 5.0)
    velocity_threshold = config.get("velocity_threshold", 0.01)
    gripper_close_threshold = config.get("gripper_close_threshold", 0.5)

    n = len(episode)
    if n == 0:
        return SegmentResult(episode_id=episode.meta.episode_id, stages=[])

    # Extract signals
    gripper = np.array(
        [f.gripper_state if f.gripper_state is not None else 0.0 for f in episode.frames]
    )
    states = np.array(
        [f.state if f.state is not None else [0.0] * 7 for f in episode.frames]
    )
    forces = np.array(
        [f.force_torque[:3] if f.force_torque and len(f.force_torque) >= 3 else [0, 0, 0]
         for f in episode.frames]
    )
    force_mag = np.linalg.norm(forces, axis=1)

    # Compute velocity (frame-to-frame state change magnitude)
    if states.shape[0] > 1:
        velocity = np.linalg.norm(np.diff(states, axis=0), axis=1)
        velocity = np.concatenate([[0], velocity])
    else:
        velocity = np.zeros(n)

    # Detect boundaries using signal changes
    boundaries: list[int] = [0]

    # Gripper close event (transition from open to closed)
    gripper_closed = gripper < gripper_close_threshold
    gripper_changes = np.diff(gripper_closed.astype(int))
    for idx in np.where(gripper_changes != 0)[0]:
        boundaries.append(int(idx))

    # Force contact onset
    force_contact = force_mag > force_threshold
    force_changes = np.diff(force_contact.astype(int))
    for idx in np.where(force_changes == 1)[0]:
        boundaries.append(int(idx))

    # Velocity transitions (moving -> stopped)
    is_moving = velocity > velocity_threshold
    move_changes = np.diff(is_moving.astype(int))
    for idx in np.where(move_changes == -1)[0]:  # stopped
        boundaries.append(int(idx))

    # Deduplicate and sort, ensure we have at least start and end
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != n - 1:
        boundaries.append(n - 1)

    # Merge boundaries that are too close (< 5 frames apart)
    min_gap = config.get("min_stage_frames", 5)
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= min_gap:
            merged.append(b)
    if merged[-1] != n - 1:
        merged.append(n - 1)

    # Build stage segments
    stages: list[StageSegment] = []
    for i in range(len(merged) - 1):
        start = merged[i]
        end = merged[i + 1] - 1
        if i == len(merged) - 2:  # last stage
            end = n - 1

        # Assign skill based on stage index
        skill_idx = min(i, len(skills) - 1)
        skill_id = skills[skill_idx]

        # Find keyframe: pick frame with peak force in stage, or midpoint
        stage_forces = force_mag[start : end + 1]
        if np.max(stage_forces) > force_threshold:
            kf = start + int(np.argmax(stage_forces))
        else:
            kf = (start + end) // 2

        stages.append(
            StageSegment(
                stage_id=i,
                skill_id=skill_id,
                start_frame=start,
                end_frame=end,
                keyframe_indices=[kf],
                confidence=0.8,
            )
        )

    result = SegmentResult(
        episode_id=episode.meta.episode_id,
        stages=stages,
        method="gripper_velocity_force",
    )

    # Save artifact
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "segments.json", "w") as f:
            f.write(result.model_dump_json(indent=2))

    return result
