"""Tests for pipeline stages — uses mock backends."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from embodiedforge.schemas.episode import EpisodeData, EpisodeMeta, FrameData
from embodiedforge.schemas.segment import SegmentResult, StageSegment
from embodiedforge.schemas.semantic import SemanticAnnotation, SemanticResult
from embodiedforge.schemas.geometry import BBox, MaskResult, AffordanceResult, Point2D


def _make_test_episode(tmp_dir: Path, num_frames: int = 10) -> Path:
    """Create a minimal test episode directory."""
    ep_dir = tmp_dir / "test_episode"
    (ep_dir / "front_rgb").mkdir(parents=True)

    # Create simple images
    for i in range(num_frames):
        img = Image.new("RGB", (64, 48), color=(100 + i * 5, 50, 50))
        img.save(ep_dir / "front_rgb" / f"{i:06d}.png")

    # Meta
    meta = {
        "episode_id": "test_ep",
        "task_instruction": "pick up the red cup",
        "fps": 30.0,
    }
    with open(ep_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    # State (simple trajectory)
    states = [[float(i) / num_frames, 0.5, 0.5, 0.0, 0.0, 0.0, float(i > 5) * 0.2] for i in range(num_frames)]
    with open(ep_dir / "state.json", "w") as f:
        json.dump(states, f)

    # Gripper
    gripper = [1.0 if i < 5 else 0.2 for i in range(num_frames)]
    with open(ep_dir / "gripper.json", "w") as f:
        json.dump(gripper, f)

    # Force/torque
    ft = [[0.5, 0.5, 0.3, 0.1, 0.1, 0.05] if i < 5 else [8.0, 4.0, 12.0, 0.2, 0.2, 0.1] for i in range(num_frames)]
    with open(ep_dir / "force_torque.json", "w") as f:
        json.dump(ft, f)

    return ep_dir


def test_ingest():
    from embodiedforge.pipeline.ingest import ingest_episode

    with tempfile.TemporaryDirectory() as tmp:
        ep_dir = _make_test_episode(Path(tmp))
        episode = ingest_episode(ep_dir)

        assert episode.meta.episode_id == "test_ep"
        assert len(episode) == 10
        assert episode.frames[0].front_rgb_path is not None


def test_segment():
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode

    with tempfile.TemporaryDirectory() as tmp:
        ep_dir = _make_test_episode(Path(tmp))
        episode = ingest_episode(ep_dir)
        result = segment_episode(episode)

        assert isinstance(result, SegmentResult)
        assert result.total_stages >= 1
        assert result.episode_id == "test_ep"


def test_semantic_annotation():
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.pipeline.segment import segment_episode
    from embodiedforge.pipeline.annotate_semantic import annotate_semantic

    with tempfile.TemporaryDirectory() as tmp:
        ep_dir = _make_test_episode(Path(tmp))
        episode = ingest_episode(ep_dir)
        segments = segment_episode(episode)
        result = annotate_semantic(episode, segments, {"backend": "mock"})

        assert isinstance(result, SemanticResult)
        assert len(result.annotations) >= 1
        for ann in result.annotations:
            assert ann.target_object != ""


def test_grounding():
    from embodiedforge.pipeline.ingest import ingest_episode
    from embodiedforge.adapters.factory import create_grounding_adapter

    with tempfile.TemporaryDirectory() as tmp:
        ep_dir = _make_test_episode(Path(tmp))
        episode = ingest_episode(ep_dir)

        adapter = create_grounding_adapter({"backend": "mock"})
        with Image.open(episode.frames[0].front_rgb_path) as img:
            boxes = adapter.ground(img, "red cup")

        assert len(boxes) >= 1
        assert boxes[0].width > 0
        assert boxes[0].confidence > 0


def test_segmentation():
    from embodiedforge.adapters.factory import create_segmentation_adapter
    from embodiedforge.schemas.geometry import BBox

    adapter = create_segmentation_adapter({"backend": "mock"})
    img = Image.new("RGB", (64, 48))
    bbox = BBox(x1=10, y1=10, x2=50, y2=40)
    mask = adapter.segment(img, bboxes=[bbox])

    assert mask.shape == (48, 64)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0


def test_qc():
    from embodiedforge.qc.checks import run_qc

    # Create mock mask and affordance results
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Create a mask image
        mask_dir = tmp / "masks"
        mask_dir.mkdir()
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        mask_path = mask_dir / "mask_000000.png"
        Image.fromarray(mask).save(str(mask_path))

        mask_results = {
            0: MaskResult(frame_idx=0, mask_path=str(mask_path), mask_area=int(np.count_nonzero(mask)))
        }

        affordance_results = {
            0: AffordanceResult(
                frame_idx=0,
                affordance_point=Point2D(x=30, y=25),
                heatmap_path="",
                heatmap_max=1.0,
            )
        }

        qc = run_qc("test_ep", mask_results, affordance_results)
        assert len(qc.checks) >= 1
        # The point (30, 25) should be inside the mask (10:40, 10:50)
        point_check = next(c for c in qc.checks if "point_in_mask" in c.name)
        assert point_check.passed
