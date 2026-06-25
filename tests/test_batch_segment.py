"""Tests for Phase 2: batch segmentation runner."""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    root: Path,
    episode_id: str,
    task_type: str = "insert_usb_right",
    num_frames: int = 18,
) -> Path:
    """Create a minimal standard-format episode with enough signals for segmentation."""
    import pandas as pd

    ep_dir = root / episode_id
    ep_dir.mkdir(parents=True)

    (ep_dir / "front_rgb").mkdir()
    from PIL import Image

    for i in range(num_frames):
        Image.new("RGB", (64, 48)).save(ep_dir / "front_rgb" / f"{i:06d}.png")

    meta = {
        "episode_id": episode_id,
        "task_instruction": f"perform {task_type}",
        "fps": 30.0,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    # Gripper: open → close → open pattern
    gripper = [1.0] * 5 + [0.2] * 8 + [1.0] * (num_frames - 13)
    with open(ep_dir / "gripper.json", "w") as f:
        json.dump(gripper, f)

    # Simple trajectory
    states = [[float(i) * 0.02, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0] for i in range(num_frames)]
    with open(ep_dir / "state.json", "w") as f:
        json.dump(states, f)

    # Force/torque
    ft = [[0.5, 0.5, 0.3, 0.1, 0.1, 0.05] for _ in range(num_frames)]
    with open(ep_dir / "force_torque.json", "w") as f:
        json.dump(ft, f)

    return ep_dir


def _make_xlsx_episode_batch(
    root: Path,
    episode_id: str,
    num_frames: int = 12,
) -> Path:
    """Create a minimal xlsx-format episode for batch testing."""
    import pandas as pd

    ep_dir = root / episode_id
    ep_dir.mkdir(parents=True)

    rows = []
    for idx in range(num_frames):
        if idx < 3:
            gripper = 151.0
            state = [0.20 + idx * 0.01, 0.02, 0.21]
        elif idx < 6:
            gripper = 39.0
            state = [0.24 + idx * 0.005, 0.02, 0.21]
        elif idx < 10:
            gripper = 39.0
            state = [0.27, 0.02, 0.21]
        else:
            gripper = 151.0
            state = [0.31 + (idx - 10) * 0.01, 0.02, 0.21]

        row = {"frame_id": idx, "right_arm_gripper": gripper}
        for axis, val in enumerate(state):
            row[f"right_arm_state_{axis}"] = val
        rows.append(row)

    pd.DataFrame(rows).to_excel(ep_dir / "episode_data.xlsx", index=False)

    from PIL import Image

    for idx in range(num_frames):
        Image.new("RGB", (64, 48)).save(ep_dir / f"RealSense_head_rgb_{idx}.jpg")
        Image.new("RGB", (64, 48)).save(ep_dir / f"RealSense_wrist_rgb_{idx}.jpg")

    session_meta = {
        "task_name": "insert_usb_right",
        "record_frequency": {"actual_hz": 20.0},
        "camera_keys": ["RealSense_head", "RealSense_wrist"],
    }
    with open(ep_dir / "session_meta.json", "w") as f:
        json.dump(session_meta, f)

    return ep_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_batch_processes_all_standard_episodes(tmp_path: Path):
    from embodiedforge.pipeline.batch_segment import run_batch_segment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for eid in ["ep_a", "ep_b", "ep_c"]:
        _make_episode(dataset, eid)

    output = tmp_path / "output"
    index = run_batch_segment(
        dataset_root=dataset,
        output_dir=output,
        config={
            "skills": ["reach", "grasp", "lift", "place"],
            "min_stage_frames": 3,
        },
    )

    assert index.total_episodes == 3
    assert index.success_count == 3
    assert index.error_count == 0

    # Per-episode artifacts
    for eid in ["ep_a", "ep_b", "ep_c"]:
        seg_dir = output / eid / "segments"
        assert (seg_dir / "boundary_candidates.json").exists(), f"Missing candidates for {eid}"
        assert (seg_dir / "segments_draft.json").exists(), f"Missing draft for {eid}"
        assert (seg_dir / "boundary_refine_vlm.json").exists(), f"Missing refine stub for {eid}"

        # Verify draft is valid JSON with expected keys
        with open(seg_dir / "segments_draft.json", encoding="utf-8") as f:
            draft = json.load(f)
        assert set(draft.keys()) == {"episode_id", "source", "method", "status", "stages"}
        assert draft["status"] == "draft"

    # Dataset-level index
    index_path = output / "batch" / "dataset_index.json"
    assert index_path.exists()
    with open(index_path, encoding="utf-8") as f:
        idx_data = json.load(f)
    assert idx_data["total_episodes"] == 3
    assert idx_data["success_count"] == 3
    assert len(idx_data["episodes"]) == 3


def test_batch_mixed_standard_and_xlsx(tmp_path: Path):
    from embodiedforge.pipeline.batch_segment import run_batch_segment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _make_episode(dataset, "std_ep")
    _make_xlsx_episode_batch(dataset, "xlsx_ep")

    output = tmp_path / "output"
    config = {
        "skills": ["reach", "grasp", "lift", "place"],
        "min_stage_frames": 3,
    }

    index = run_batch_segment(dataset_root=dataset, output_dir=output, config=config)

    assert index.total_episodes == 2
    assert index.success_count == 2
    assert index.error_count == 0

    # Both should produce artifacts
    for eid in ["std_ep", "xlsx_ep"]:
        assert (output / eid / "segments" / "segments_draft.json").exists()


def test_batch_error_isolation(tmp_path: Path):
    """One broken episode should not kill the batch."""
    from embodiedforge.pipeline.batch_segment import run_batch_segment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _make_episode(dataset, "good_1")
    # Create a broken episode that has meta.json but no images
    broken_dir = dataset / "broken"
    broken_dir.mkdir()
    with open(broken_dir / "meta.json", "w") as f:
        json.dump({"episode_id": "broken", "task_instruction": "test", "fps": 30.0}, f)
    # No image dirs — will fail during segment_demo but should be caught
    _make_episode(dataset, "good_2")

    output = tmp_path / "output"
    index = run_batch_segment(
        dataset_root=dataset,
        output_dir=output,
        config={"skills": ["reach", "grasp"], "min_stage_frames": 3},
    )

    assert index.total_episodes == 3
    assert index.success_count == 2
    assert index.error_count == 1

    broken_result = [r for r in index.episodes if r.episode_id == "broken"][0]
    assert broken_result.status == "error"

    # Good episodes still produced artifacts
    assert (output / "good_1" / "segments" / "segments_draft.json").exists()
    assert (output / "good_2" / "segments" / "segments_draft.json").exists()


def test_batch_episode_ids_filter(tmp_path: Path):
    from embodiedforge.pipeline.batch_segment import run_batch_segment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for eid in ["ep_a", "ep_b", "ep_c", "ep_d"]:
        _make_episode(dataset, eid)

    output = tmp_path / "output"
    index = run_batch_segment(
        dataset_root=dataset,
        output_dir=output,
        config={"skills": ["reach", "grasp"], "min_stage_frames": 3},
        episode_ids=["ep_a", "ep_c"],
    )

    assert index.total_episodes == 2
    processed = {r.episode_id for r in index.episodes}
    assert processed == {"ep_a", "ep_c"}


def test_batch_max_episodes_limit(tmp_path: Path):
    from embodiedforge.pipeline.batch_segment import run_batch_segment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for eid in ["ep_a", "ep_b", "ep_c", "ep_d", "ep_e"]:
        _make_episode(dataset, eid)

    output = tmp_path / "output"
    index = run_batch_segment(
        dataset_root=dataset,
        output_dir=output,
        config={"skills": ["reach", "grasp"], "min_stage_frames": 3},
        max_episodes=2,
    )

    assert index.total_episodes == 2
    assert index.success_count == 2


def test_dataset_index_schema_valid(tmp_path: Path):
    """The dataset index JSON should conform to DatasetSegmentIndex schema."""
    from embodiedforge.pipeline.batch_segment import run_batch_segment
    from embodiedforge.sampling.schemas import DatasetSegmentIndex

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _make_episode(dataset, "ep_x")

    output = tmp_path / "output"
    index = run_batch_segment(
        dataset_root=dataset,
        output_dir=output,
        config={"skills": ["reach", "grasp"], "min_stage_frames": 3},
    )

    # Roundtrip through JSON
    json_str = index.model_dump_json(indent=2)
    loaded = DatasetSegmentIndex.model_validate_json(json_str)

    assert loaded.total_episodes == index.total_episodes
    assert loaded.success_count == index.success_count
    assert loaded.error_count == index.error_count
    assert len(loaded.episodes) == len(index.episodes)
    assert loaded.summary == index.summary


def test_batch_produces_boundary_refine_stub(tmp_path: Path):
    """boundary_refine_vlm.json should exist with correct format per episode."""
    from embodiedforge.pipeline.batch_segment import run_batch_segment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _make_episode(dataset, "ep_001")

    output = tmp_path / "output"
    run_batch_segment(
        dataset_root=dataset,
        output_dir=output,
        config={"skills": ["reach", "grasp", "lift"], "min_stage_frames": 3},
    )

    refine_path = output / "ep_001" / "segments" / "boundary_refine_vlm.json"
    with open(refine_path, encoding="utf-8") as f:
        refine = json.load(f)

    assert refine["source"] == "vlm_refine"
    assert len(refine["items"]) >= 1
    # Each item should have the expected keys
    for item in refine["items"]:
        assert set(item.keys()) == {
            "candidate_id", "accepted", "refined_frame",
            "from_skill", "to_skill", "confidence", "rationale",
        }


def test_batch_summary_statistics(tmp_path: Path):
    from embodiedforge.pipeline.batch_segment import run_batch_segment

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for eid in ["ep_a", "ep_b"]:
        _make_episode(dataset, eid)

    output = tmp_path / "output"
    index = run_batch_segment(
        dataset_root=dataset,
        output_dir=output,
        config={"skills": ["reach", "grasp"], "min_stage_frames": 3},
    )

    s = index.summary
    assert s["total"] == 2
    assert s["success"] == 2
    assert s["error"] == 0
    assert s["total_stages"] > 0
    assert s["avg_stages_per_episode"] > 0
    assert s["total_elapsed_seconds"] > 0
