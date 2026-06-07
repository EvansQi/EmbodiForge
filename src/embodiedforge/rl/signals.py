"""Heuristic RL signal generation from processed episodes."""

from __future__ import annotations

import math
from typing import Any

from embodiedforge.schemas.episode import EpisodeData
from embodiedforge.schemas.geometry import AffordanceResult, BBox, Geometry3DResult, MaskResult, Point2D
from embodiedforge.schemas.rl import BinaryClassifierTarget, RLKeyframeSignal, RLSummary, RewardTerm
from embodiedforge.schemas.sample import QCSummary
from embodiedforge.schemas.segment import SegmentResult, StageSegment
from embodiedforge.schemas.semantic import SemanticResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _force_magnitude(force_torque: list[float] | None) -> float:
    if not force_torque or len(force_torque) < 3:
        return 0.0
    fx, fy, fz = force_torque[:3]
    return math.sqrt(fx * fx + fy * fy + fz * fz)


def _contact_confidence(frame, force_threshold: float, gripper_close_threshold: float) -> float:
    force_ratio = _clamp01(_force_magnitude(frame.force_torque) / max(force_threshold, 1e-6))
    gripper = frame.gripper_state if frame.gripper_state is not None else 1.0
    close_ratio = _clamp01((gripper_close_threshold - gripper) / max(gripper_close_threshold, 1e-6))
    return _clamp01(0.7 * force_ratio + 0.3 * close_ratio)


def _point_inside_bbox(point: Point2D | None, bbox: BBox | None) -> bool:
    if point is None or bbox is None:
        return False
    return bbox.x1 <= point.x <= bbox.x2 and bbox.y1 <= point.y <= bbox.y2


def _alignment_score(point: Point2D | None, bbox: BBox | None) -> float:
    if point is None or bbox is None or bbox.width <= 0 or bbox.height <= 0:
        return 0.0
    cx, cy = bbox.center
    dx = (point.x - cx) / max(bbox.width / 2, 1e-6)
    dy = (point.y - cy) / max(bbox.height / 2, 1e-6)
    dist = math.sqrt(dx * dx + dy * dy)
    return _clamp01(1.0 - dist)


def _stage_progress(stage: StageSegment, stage_index: int, total_stages: int, frame_idx: int) -> float:
    if total_stages <= 1:
        coarse = 1.0
    else:
        coarse = stage_index / (total_stages - 1)
    span = max(stage.end_frame - stage.start_frame, 1)
    intra = _clamp01((frame_idx - stage.start_frame) / span)
    return _clamp01(0.7 * coarse + 0.3 * intra)


def _build_reward_terms(
    stage_progress: float,
    alignment_score: float,
    contact_confidence: float,
    qc_score: float,
    subgoal_completed: bool,
    success: bool,
    terminal: bool,
    config: dict[str, Any],
) -> list[RewardTerm]:
    weights = config.get("weights", {})
    term_specs = [
        ("stage_progress", stage_progress, float(weights.get("stage_progress", 0.25))),
        ("alignment", alignment_score, float(weights.get("alignment", 0.25))),
        ("contact", contact_confidence, float(weights.get("contact", 0.2))),
        ("qc", qc_score, float(weights.get("qc", 0.15))),
        ("subgoal", 1.0 if subgoal_completed else 0.0, float(weights.get("subgoal", 0.15))),
        ("success_bonus", 1.0 if success and terminal else 0.0, float(weights.get("success_bonus", 1.0))),
    ]
    return [
        RewardTerm(name=name, value=value, weight=weight, weighted_value=value * weight)
        for name, value, weight in term_specs
    ]


def build_rl_signals(
    episode: EpisodeData,
    segments: SegmentResult,
    semantic: SemanticResult,
    grounding_results: dict[int, dict[str, BBox]],
    mask_results: dict[int, MaskResult],
    affordance_results: dict[int, AffordanceResult],
    geometry_results: dict[int, Geometry3DResult],
    qc: QCSummary | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[int, RLKeyframeSignal], RLSummary]:
    """Build heuristic reward and success signals for RL or reward-model training."""
    config = config or {}
    enabled = bool(config.get("enabled", True))
    reward_scheme = str(config.get("reward_scheme", "heuristic_v1"))
    gamma = float(config.get("gamma", 0.99))
    force_threshold = float(config.get("force_threshold", 5.0))
    gripper_close_threshold = float(config.get("gripper_close_threshold", 0.5))
    success_terminal_skills = set(config.get("success_terminal_skills", ["grasp", "lift", "place", "insert"]))

    summary = RLSummary(
        enabled=enabled,
        reward_scheme=reward_scheme,
        gamma=gamma,
        success=False,
        success_criteria=[
            "qc_passed",
            "final_keyframe_aligned",
            "final_keyframe_subgoal_complete",
        ],
    )
    if not enabled:
        return {}, summary

    stage_lookup = {stage.stage_id: stage for stage in segments.stages}
    stage_index_lookup = {stage.stage_id: idx for idx, stage in enumerate(segments.stages)}
    keyframes = sorted(set(ann.frame_idx for ann in semantic.annotations))
    signals: dict[int, RLKeyframeSignal] = {}

    total_dense = 0.0
    total_sparse = 0.0
    contact_scores: list[float] = []
    alignment_scores: list[float] = []

    for index, ann in enumerate(semantic.annotations):
        frame_idx = ann.frame_idx
        frame = episode.frames[frame_idx]
        stage = stage_lookup.get(ann.stage_id)
        if stage is None:
            stage = segments.get_stage_at_frame(frame_idx) or StageSegment(
                stage_id=ann.stage_id,
                skill_id=ann.skill_id,
                start_frame=frame_idx,
                end_frame=frame_idx,
                keyframe_indices=[frame_idx],
            )

        grounding = grounding_results.get(frame_idx, {})
        affordance = affordance_results.get(frame_idx)
        point = affordance.affordance_point if affordance else None
        bbox = grounding.get("part_bbox") or grounding.get("object_bbox")

        stage_index = stage_index_lookup.get(stage.stage_id, index)
        progress = _stage_progress(stage, stage_index, max(len(segments.stages), 1), frame_idx)
        contact = _contact_confidence(frame, force_threshold, gripper_close_threshold)
        alignment = _alignment_score(point, bbox)
        point_valid = _point_inside_bbox(point, bbox)
        has_mask = frame_idx in mask_results and mask_results[frame_idx].mask_area > 0
        has_3d = frame_idx in geometry_results and bool(geometry_results[frame_idx].keypoints_3d)

        subgoal_completed = point_valid and has_mask and (alignment >= 0.25)
        terminal = frame_idx == keyframes[-1] if keyframes else False
        qc_score = qc.score if qc else 0.0
        success = bool(
            terminal
            and (qc.all_passed if qc else False)
            and subgoal_completed
            and ann.skill_id in success_terminal_skills
        )

        reward_terms = _build_reward_terms(
            stage_progress=progress,
            alignment_score=alignment,
            contact_confidence=contact,
            qc_score=qc_score,
            subgoal_completed=subgoal_completed,
            success=success,
            terminal=terminal,
            config=config,
        )
        dense_reward = sum(term.weighted_value for term in reward_terms)
        sparse_reward = 1.0 if success else 0.0
        advantage_hint = dense_reward + 0.5 * sparse_reward

        classifier_targets = [
            BinaryClassifierTarget(name="success", value=success, confidence=1.0 if terminal else 0.7),
            BinaryClassifierTarget(name="subgoal_completed", value=subgoal_completed, confidence=0.9),
            BinaryClassifierTarget(name="point_in_bbox", value=point_valid, confidence=0.95 if bbox else 0.4),
            BinaryClassifierTarget(name="contact_detected", value=contact >= 0.5, confidence=max(contact, 0.5)),
            BinaryClassifierTarget(name="geometry_available", value=has_3d, confidence=1.0 if has_3d else 0.6),
        ]

        signal = RLKeyframeSignal(
            frame_idx=frame_idx,
            stage_id=ann.stage_id,
            skill_id=ann.skill_id,
            dense_reward=dense_reward,
            sparse_reward=sparse_reward,
            advantage_hint=advantage_hint,
            stage_progress=progress,
            subgoal_completed=subgoal_completed,
            success=success,
            terminal=terminal,
            contact_confidence=contact,
            alignment_score=alignment,
            reward_terms=reward_terms,
            classifier_targets=classifier_targets,
        )
        signals[frame_idx] = signal
        total_dense += dense_reward
        total_sparse += sparse_reward
        contact_scores.append(contact)
        alignment_scores.append(alignment)

    summary.success = any(signal.success for signal in signals.values())
    summary.total_dense_reward = total_dense
    summary.total_sparse_reward = total_sparse
    summary.mean_contact_confidence = sum(contact_scores) / len(contact_scores) if contact_scores else 0.0
    summary.mean_alignment_score = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    return signals, summary
