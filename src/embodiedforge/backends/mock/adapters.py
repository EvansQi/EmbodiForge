"""Mock adapter implementations.

All mock adapters produce realistic synthetic data that matches the real schema.
They are designed so the full pipeline can run end-to-end without any ML model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from embodiedforge.adapters.base import (
    DepthAdapter,
    GroundingAdapter,
    SegmentationAdapter,
    SemanticLabelerAdapter,
)
from embodiedforge.adapters.factory import register_adapter
from embodiedforge.schemas.geometry import BBox, Point2D
from embodiedforge.schemas.semantic import SemanticAnnotation

# ---------------------------------------------------------------------------
# Known skills for the mock semantic labeler
# ---------------------------------------------------------------------------

_SKILL_TEMPLATES: dict[str, dict[str, str]] = {
    "reach": {
        "stage_label": "reaching toward {object}",
        "target_object": "target object",
        "target_part": "center",
        "affordance_query": "approach direction on {object}",
    },
    "grasp": {
        "stage_label": "grasping {object}",
        "target_object": "target object",
        "target_part": "handle",
        "affordance_query": "grasp point on {object}",
    },
    "lift": {
        "stage_label": "lifting {object}",
        "target_object": "target object",
        "target_part": "top",
        "affordance_query": "lift point on {object}",
    },
    "place": {
        "stage_label": "placing {object} on target",
        "target_object": "target object",
        "target_part": "bottom",
        "affordance_query": "release point above target",
    },
    "insert": {
        "stage_label": "inserting {object} into target",
        "target_object": "target object",
        "target_part": "tip",
        "affordance_query": "insertion point on target",
    },
}


def _infer_object_from_instruction(instruction: str) -> str:
    """Simple heuristic to extract object name from instruction."""
    keywords = ["cup", "block", "plug", "can", "bottle", "drawer", "button", "peg"]
    lower = instruction.lower()
    for kw in keywords:
        if kw in lower:
            return kw
    return "target object"


class MockSemanticLabelerAdapter(SemanticLabelerAdapter):
    """Mock semantic labeler — uses heuristics instead of a VLM."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def label(
        self,
        image: Image.Image,
        task_instruction: str,
        skill_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> SemanticAnnotation:
        obj = _infer_object_from_instruction(task_instruction)
        template = _SKILL_TEMPLATES.get(skill_id, _SKILL_TEMPLATES["grasp"])

        # Build soft skill labels
        skill_soft: dict[str, float] = {}
        for s in _SKILL_TEMPLATES:
            skill_soft[s] = 0.8 if s == skill_id else 0.2 / max(1, len(_SKILL_TEMPLATES) - 1)

        return SemanticAnnotation(
            frame_idx=context.get("frame_idx", 0) if context else 0,
            stage_id=context.get("stage_id", 0) if context else 0,
            skill_id=skill_id or "grasp",
            stage_label=template["stage_label"].format(object=obj),
            target_object=obj,
            target_part=template["target_part"],
            affordance_query=template["affordance_query"].format(object=obj),
            skill_soft=skill_soft,
            confidence=0.85,
        )


class MockGroundingAdapter(GroundingAdapter):
    """Mock grounding — returns a plausible bbox near image center."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def ground(
        self,
        image: Image.Image,
        query: str,
        confidence_threshold: float = 0.3,
    ) -> list[BBox]:
        w, h = image.size
        # Place a box roughly in the center, sized ~30% of image
        bw, bh = int(w * 0.3), int(h * 0.3)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2
        return [BBox(x1=x1, y1=y1, x2=x2, y2=y2, label=query, confidence=0.78)]


class MockSegmentationAdapter(SegmentationAdapter):
    """Mock segmentation — draws an ellipse inside the bbox as mask."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def segment(
        self,
        image: Image.Image,
        bboxes: list[BBox] | None = None,
        points: list[Point2D] | None = None,
        point_labels: list[int] | None = None,
    ) -> np.ndarray:
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        if bboxes:
            for bbox in bboxes:
                # Shrink slightly for a more natural mask
                pad = min(bbox.width, bbox.height) * 0.1
                draw.ellipse(
                    [bbox.x1 + pad, bbox.y1 + pad, bbox.x2 - pad, bbox.y2 - pad],
                    fill=255,
                )
        elif points:
            # Draw circles around foreground points
            r = min(w, h) * 0.1
            for pt, lbl in zip(points, point_labels or [1] * len(points)):
                if lbl == 1:
                    draw.ellipse([pt.x - r, pt.y - r, pt.x + r, pt.y + r], fill=255)

        return np.array(mask, dtype=np.uint8)


class MockDepthAdapter(DepthAdapter):
    """Mock depth — produces a simple gradient depth map."""

    def __init__(self, base_depth: float = 0.5, **kwargs: Any) -> None:
        self.base_depth = base_depth

    def estimate_depth(
        self,
        image: Image.Image,
        camera_intrinsics: dict[str, float] | None = None,
    ) -> np.ndarray:
        w, h = image.size
        # Create a depth map that ramps from base_depth to base_depth + 0.3
        y_coords = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        depth = self.base_depth + y_coords * 0.3
        return np.broadcast_to(depth, (h, w)).copy()


# Register all mock adapters
register_adapter("semantic", "mock", MockSemanticLabelerAdapter)
register_adapter("grounding", "mock", MockGroundingAdapter)
register_adapter("segmentation", "mock", MockSegmentationAdapter)
register_adapter("depth", "mock", MockDepthAdapter)
