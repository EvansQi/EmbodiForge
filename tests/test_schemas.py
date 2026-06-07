"""Tests for core schemas."""

from embodiedforge.schemas.episode import EpisodeData, EpisodeMeta, FrameData
from embodiedforge.schemas.segment import SegmentResult, StageSegment
from embodiedforge.schemas.semantic import SemanticAnnotation, SemanticResult
from embodiedforge.schemas.geometry import BBox, Point2D, MaskResult, AffordanceResult
from embodiedforge.schemas.sample import TrainingSample, QCCheck, QCSummary
from embodiedforge.schemas.rl import RLKeyframeSignal, RLSummary


def test_episode_meta():
    meta = EpisodeMeta(episode_id="test_001", task_instruction="pick up cup")
    assert meta.episode_id == "test_001"
    assert meta.fps == 30.0
    assert meta.total_frames == 0


def test_frame_data():
    frame = FrameData(frame_idx=0, gripper_state=0.5)
    assert frame.frame_idx == 0
    assert frame.gripper_state == 0.5


def test_episode_data():
    meta = EpisodeMeta(episode_id="ep1", task_instruction="test")
    frames = [FrameData(frame_idx=i) for i in range(5)]
    ep = EpisodeData(meta=meta, frames=frames)
    assert len(ep) == 5


def test_bbox():
    bbox = BBox(x1=10, y1=20, x2=50, y2=60)
    assert bbox.width == 40
    assert bbox.height == 40
    assert bbox.center == (30, 40)
    assert bbox.area == 1600


def test_segment_result():
    stages = [
        StageSegment(stage_id=0, skill_id="reach", start_frame=0, end_frame=9, keyframe_indices=[5]),
        StageSegment(stage_id=1, skill_id="grasp", start_frame=10, end_frame=19, keyframe_indices=[15]),
    ]
    result = SegmentResult(episode_id="ep1", stages=stages)
    assert result.total_stages == 2
    assert result.get_stage_at_frame(5).skill_id == "reach"
    assert result.get_stage_at_frame(15).skill_id == "grasp"
    assert result.get_stage_at_frame(100) is None


def test_semantic_annotation():
    ann = SemanticAnnotation(
        frame_idx=10,
        stage_id=1,
        skill_id="grasp",
        target_object="red cup",
        target_part="handle",
    )
    assert ann.target_object == "red cup"
    assert ann.confidence == 1.0


def test_qc_summary():
    checks = [
        QCCheck(name="check1", passed=True, message="ok"),
        QCCheck(name="check2", passed=False, message="fail"),
    ]
    qc = QCSummary(checks=checks)
    score = qc.compute_score()
    assert score == 0.5
    assert not qc.all_passed


def test_training_sample():
    sample = TrainingSample(episode_id="ep1", task_instruction="test")
    assert sample.episode_id == "ep1"
    assert len(sample.keyframes) == 0


def test_rl_schemas():
    signal = RLKeyframeSignal(frame_idx=3, dense_reward=0.8, success=True)
    summary = RLSummary(enabled=True, success=True, total_dense_reward=1.2)
    assert signal.success
    assert summary.total_dense_reward == 1.2
