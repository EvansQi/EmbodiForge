"""Tests for Phase 1: sampling module — scanner, stratified sampler, manifest/split."""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers — create synthetic datasets for testing
# ---------------------------------------------------------------------------


def _make_standard_episode(
    root: Path,
    episode_id: str,
    task_instruction: str = "pick up the red cup",
    num_frames: int = 30,
    fps: float = 30.0,
    has_wrist: bool = True,
    gripper_close_at: int = 10,
    gripper_open_at: int = 20,
    trajectory_len: float = 1.5,
) -> Path:
    """Create a minimal standard-format episode for scanning."""
    ep_dir = root / episode_id
    ep_dir.mkdir(parents=True)

    (ep_dir / "front_rgb").mkdir()
    for i in range(num_frames):
        from PIL import Image

        Image.new("RGB", (64, 48)).save(ep_dir / "front_rgb" / f"{i:06d}.png")

    if has_wrist:
        (ep_dir / "wrist_rgb").mkdir()
        for i in range(num_frames):
            from PIL import Image

            Image.new("RGB", (64, 48)).save(ep_dir / "wrist_rgb" / f"{i:06d}.png")

    meta = {
        "episode_id": episode_id,
        "task_instruction": task_instruction,
        "fps": fps,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    # Gripper: open → close → open
    gripper = [1.0] * gripper_close_at + [0.2] * (gripper_open_at - gripper_close_at) + [1.0] * (num_frames - gripper_open_at)
    with open(ep_dir / "gripper.json", "w") as f:
        json.dump(gripper, f)

    # State: simple trajectory
    states = [[float(i) * trajectory_len / num_frames, 0.5, 0.5, 0, 0, 0, 0] for i in range(num_frames)]
    with open(ep_dir / "state.json", "w") as f:
        json.dump(states, f)

    return ep_dir


def _make_xlsx_episode(
    root: Path,
    episode_id: str,
    task_name: str = "insert_usb_right",
    num_frames: int = 20,
    fps: float = 20.0,
) -> Path:
    """Create a minimal xlsx-format episode for scanning."""
    import pandas as pd

    ep_dir = root / episode_id
    ep_dir.mkdir(parents=True)

    rows = []
    for idx in range(num_frames):
        if idx < 5:
            gripper = 151.0
            state = [0.20 + idx * 0.01, 0.02, 0.21]
        elif idx < 8:
            gripper = 39.0
            state = [0.25 + idx * 0.005, 0.02, 0.21]
        elif idx < 15:
            gripper = 39.0
            state = [0.27, 0.02, 0.21]
        elif idx < 17:
            gripper = 151.0
            state = [0.27, 0.02, 0.21]
        else:
            gripper = 151.0
            state = [0.31 + (idx - 17) * 0.01, 0.02, 0.21]

        row = {"frame_id": idx, "right_arm_gripper": gripper}
        for axis, val in enumerate(state):
            row[f"right_arm_state_{axis}"] = val
        rows.append(row)

    pd.DataFrame(rows).to_excel(ep_dir / "episode_data.xlsx", index=False)

    # Generate dummy images — scanner counts frames from image files
    from PIL import Image

    for idx in range(num_frames):
        Image.new("RGB", (64, 48)).save(ep_dir / f"RealSense_head_rgb_{idx}.jpg")
        Image.new("RGB", (64, 48)).save(ep_dir / f"RealSense_wrist_rgb_{idx}.jpg")

    session_meta = {
        "task_name": task_name,
        "record_frequency": {"actual_hz": fps},
        "camera_keys": ["RealSense_head", "RealSense_wrist"],
    }
    with open(ep_dir / "session_meta.json", "w") as f:
        json.dump(session_meta, f)

    return ep_dir


# ---------------------------------------------------------------------------
# Scanner tests
# ---------------------------------------------------------------------------


def test_scan_single_standard_episode(tmp_path: Path):
    from embodiedforge.sampling.scanner import scan_single_episode

    _make_standard_episode(tmp_path, "ep_001", task_instruction="pick up the red cup")
    summary = scan_single_episode(tmp_path / "ep_001")

    assert summary.episode_id == "ep_001"
    assert summary.format_type == "standard"
    assert summary.task_type == "pick_place"
    assert summary.num_frames == 30
    assert summary.has_front_rgb is True
    assert summary.has_wrist_rgb is True
    assert summary.camera_setup == "both"
    assert summary.has_gripper is True
    assert summary.has_state is True
    assert summary.success is True
    assert summary.gripper_close_frame == 10
    assert summary.gripper_open_frame == 20
    assert summary.trajectory_length > 0
    assert summary.is_complete is True
    assert len(summary.missing_files) == 0


def test_scan_single_xlsx_episode(tmp_path: Path):
    from embodiedforge.sampling.scanner import scan_single_episode

    _make_xlsx_episode(tmp_path, "usb_001")
    summary = scan_single_episode(tmp_path / "usb_001")

    assert summary.episode_id == "usb_001"
    assert summary.format_type == "xlsx"
    assert summary.task_type == "insert_usb"
    assert summary.num_frames == 20
    assert summary.fps == 20.0
    assert summary.has_gripper is True
    assert summary.has_state is True
    assert summary.success is True  # gripper closes at 5, opens at 15


def test_scan_incomplete_episode(tmp_path: Path):
    from embodiedforge.sampling.scanner import scan_single_episode

    ep_dir = tmp_path / "broken_ep"
    ep_dir.mkdir()
    (ep_dir / "front_rgb").mkdir()
    from PIL import Image

    Image.new("RGB", (64, 48)).save(ep_dir / "front_rgb" / "000000.png")

    summary = scan_single_episode(ep_dir)

    assert summary.is_complete is False
    assert len(summary.missing_files) > 0
    assert "meta.json" in summary.missing_files or "gripper.json" in summary.missing_files


def test_scan_dataset_with_mixed_formats(tmp_path: Path):
    from embodiedforge.sampling.scanner import scan_dataset

    _make_standard_episode(tmp_path, "ep_001", task_instruction="pick up the red cup")
    _make_xlsx_episode(tmp_path, "usb_001", task_name="insert_usb_right")

    episodes = scan_dataset(tmp_path)

    assert len(episodes) == 2
    formats = {ep.format_type for ep in episodes}
    assert "standard" in formats
    assert "xlsx" in formats


def test_infer_success_failure(tmp_path: Path):
    """Episode where gripper only closes but never opens → failure."""
    from embodiedforge.sampling.scanner import scan_single_episode

    _make_standard_episode(tmp_path, "fail_ep", gripper_close_at=5, gripper_open_at=30, num_frames=10)
    # Gripper closes at 5 but never opens (only 10 frames, open_at=30 > num_frames)
    summary = scan_single_episode(tmp_path / "fail_ep")
    assert summary.success is False


def test_task_type_inference():
    from embodiedforge.sampling.scanner import _infer_task_type

    assert _infer_task_type("pick up the red cup from the table") == "pick_place"
    assert _infer_task_type("insert the usb plug into the slot") == "insert_usb"
    assert _infer_task_type("open the drawer and take the block") == "drawer"
    assert _infer_task_type("press the blue button") == "button_press"
    assert _infer_task_type("push the block forward") == "push"
    assert _infer_task_type("something completely new") == "unknown"


# ---------------------------------------------------------------------------
# Stratified sampler tests
# ---------------------------------------------------------------------------


def _make_test_episodes() -> list:
    """Build a diverse set of EpisodeSummary objects for testing."""
    from embodiedforge.sampling.schemas import EpisodeSummary

    episodes = []
    # 3 task types × varied durations/cameras/success
    tasks = ["insert_usb", "pick_place", "drawer"]
    cameras = ["both", "front_only", "wrist_only"]
    durations = [3.0, 10.0, 30.0]  # short, medium, long

    eid = 0
    for task in tasks:
        for cam in cameras:
            for dur in durations:
                success = (eid % 3 != 2)  # mix of success/failure/None
                episodes.append(
                    EpisodeSummary(
                        episode_id=f"{task}_{eid:03d}",
                        source_path=f"/data/{task}_{eid:03d}",
                        task_type=task,
                        task_instruction=f"do {task}",
                        num_frames=int(dur * 30),
                        fps=30.0,
                        duration_seconds=dur,
                        camera_setup=cam,
                        has_gripper=True,
                        has_state=True,
                        trajectory_length=float(eid) * 0.1,
                        success=success if task != "drawer" else None,
                    )
                )
                eid += 1

    return episodes


def test_build_sampling_manifest_proportional_allocation():
    from embodiedforge.sampling.sampler import build_sampling_manifest

    episodes = _make_test_episodes()
    manifest = build_sampling_manifest(
        episodes,
        dataset_root="/data",
        target_sample_size=20,
        min_per_stratum=1,
        seed=42,
    )

    assert manifest.total_episodes == len(episodes)  # 3×3×3 = 27
    # Proportional allocation across 4 dimensions may yield fewer than target
    # due to overlapping episode IDs across strata (each episode appears in 4 strata).
    assert 12 <= len(manifest.selected_ids) <= 20
    assert len(manifest.strata) > 0
    assert len(manifest.sampling_dimensions) == 4
    assert manifest.sampling_rationale != ""

    # Every selected episode should exist
    all_ids = {ep.episode_id for ep in episodes}
    for sid in manifest.selected_ids:
        assert sid in all_ids

    # Coverage: at least one episode per task_type dimension
    task_types_selected = set()
    for ep in episodes:
        if ep.episode_id in manifest.selected_ids:
            task_types_selected.add(ep.task_type)
    assert len(task_types_selected) >= 2  # should cover most task types


def test_build_sampling_manifest_respects_min_per_stratum():
    from embodiedforge.sampling.sampler import build_sampling_manifest

    episodes = _make_test_episodes()
    manifest = build_sampling_manifest(
        episodes,
        dataset_root="/data",
        target_sample_size=20,
        min_per_stratum=2,
        seed=42,
    )

    # With min_per_stratum=2 and 3×3×3=27 strata, some strata will get at least 2
    for stratum in manifest.strata:
        if stratum.count >= 2:
            selected_in_stratum = sum(
                1 for eid in stratum.episode_ids if eid in manifest.selected_ids
            )
            # Strata with ≥2 episodes should have at least min(2, count) selected
            assert selected_in_stratum >= min(2, stratum.count)


def test_build_sampling_manifest_reproducible():
    from embodiedforge.sampling.sampler import build_sampling_manifest

    episodes = _make_test_episodes()

    m1 = build_sampling_manifest(episodes, "/data", target_sample_size=20, seed=42)
    m2 = build_sampling_manifest(episodes, "/data", target_sample_size=20, seed=42)

    assert m1.selected_ids == m2.selected_ids


def test_build_sampling_manifest_different_seed_different_selection():
    from embodiedforge.sampling.sampler import build_sampling_manifest

    episodes = _make_test_episodes()
    m1 = build_sampling_manifest(episodes, "/data", target_sample_size=20, seed=42)
    m2 = build_sampling_manifest(episodes, "/data", target_sample_size=20, seed=123)

    # With 20/27 selected, different seeds likely produce different selections
    # (though not 100% guaranteed, 20 out of 27 is high enough that they should differ)
    assert set(m1.selected_ids) != set(m2.selected_ids)


def test_build_ground_truth_split():
    from embodiedforge.sampling.sampler import build_sampling_manifest, build_ground_truth_split

    episodes = _make_test_episodes()
    manifest = build_sampling_manifest(episodes, "/data", target_sample_size=15, seed=42)
    split = build_ground_truth_split(manifest, manifest_path="/data/sampling_manifest.json")

    assert split.selected_episodes == manifest.selected_ids
    assert split.status == "pending_annotation"
    assert split.source_manifest == "/data/sampling_manifest.json"
    assert "Annotation Guidelines" in split.annotation_guidelines
    assert "stage boundaries" in split.annotation_guidelines.lower()


def test_empty_dataset(tmp_path: Path):
    from embodiedforge.sampling.scanner import scan_dataset
    from embodiedforge.sampling.sampler import build_sampling_manifest

    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    episodes = scan_dataset(empty_dir)
    manifest = build_sampling_manifest(episodes, str(empty_dir), target_sample_size=30)
    assert manifest.total_episodes == 0
    assert len(manifest.selected_ids) == 0


def test_roundtrip_manifest_json():
    """Manifest should survive serialize → deserialize roundtrip."""
    from embodiedforge.sampling.sampler import build_sampling_manifest
    from embodiedforge.sampling.schemas import SamplingManifest

    episodes = _make_test_episodes()
    manifest = build_sampling_manifest(episodes, "/data", target_sample_size=10, seed=42)

    json_str = manifest.model_dump_json(indent=2)
    loaded = SamplingManifest.model_validate_json(json_str)

    assert loaded.total_episodes == manifest.total_episodes
    assert loaded.selected_ids == manifest.selected_ids
    assert len(loaded.episodes) == len(manifest.episodes)
    assert loaded.sampling_rationale == manifest.sampling_rationale
