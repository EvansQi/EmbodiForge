"""Abstract adapter interfaces for model backends.

Every adapter defines the contract between pipeline stages and model implementations.
Pipeline code MUST only depend on these interfaces, never on concrete backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from embodiedforge.schemas.geometry import BBox, Point2D
from embodiedforge.schemas.semantic import SemanticAnnotation


class SemanticLabelerAdapter(ABC):
    """Adapter for vision-language semantic labeling (e.g. Qwen-VL).

    Given an image and context, produce semantic labels:
    stage_label, target_object, target_part, affordance_query.
    """

    @abstractmethod
    def label(
        self,
        image: Image.Image,
        task_instruction: str,
        skill_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> SemanticAnnotation:
        """Produce semantic annotation for a single frame.

        Args:
            image: PIL Image of the keyframe.
            task_instruction: Natural language task instruction.
            skill_id: Current skill/stage identifier.
            context: Optional extra context (e.g. previous annotations).

        Returns:
            SemanticAnnotation with all fields populated.
        """
        ...


class GroundingAdapter(ABC):
    """Adapter for visual grounding (e.g. GroundingDINO).

    Given an image and a text query, produce bounding boxes.
    """

    @abstractmethod
    def ground(
        self,
        image: Image.Image,
        query: str,
        confidence_threshold: float = 0.3,
    ) -> list[BBox]:
        """Ground a text query in the image.

        Args:
            image: PIL Image.
            query: Text description to ground, e.g. "red cup handle".
            confidence_threshold: Minimum confidence for returned boxes.

        Returns:
            List of BBox matching the query, sorted by confidence descending.
        """
        ...


class SegmentationAdapter(ABC):
    """Adapter for object segmentation and mask propagation (e.g. SAM2).

    Given an image and prompts (bbox, points), produce segmentation masks.
    """

    @abstractmethod
    def segment(
        self,
        image: Image.Image,
        bboxes: list[BBox] | None = None,
        points: list[Point2D] | None = None,
        point_labels: list[int] | None = None,
    ) -> np.ndarray:
        """Produce a binary segmentation mask.

        Args:
            image: PIL Image.
            bboxes: Optional bounding box prompts.
            points: Optional point prompts.
            point_labels: 1 for foreground, 0 for background per point.

        Returns:
            Binary mask as uint8 numpy array (H, W), values 0 or 255.
        """
        ...

    def propagate_video(
        self,
        images: list[Image.Image],
        initial_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """Propagate a mask across a video sequence.

        Default implementation: reuse initial mask for all frames.
        Override for temporal propagation (e.g. SAM2 video mode).

        Args:
            images: List of PIL Images for the video.
            initial_mask: Binary mask for the first frame.

        Returns:
            List of binary masks, one per frame.
        """
        return [initial_mask.copy() for _ in images]


class DepthAdapter(ABC):
    """Adapter for depth estimation (e.g. Depth Anything, RealSense)."""

    @abstractmethod
    def estimate_depth(
        self,
        image: Image.Image,
        camera_intrinsics: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Estimate depth map from an RGB image.

        Args:
            image: PIL Image.
            camera_intrinsics: Optional camera params {fx, fy, cx, cy}.

        Returns:
            Depth map as float32 numpy array (H, W) in meters.
        """
        ...

    def pixel_to_3d(
        self,
        u: float,
        v: float,
        depth: float,
        intrinsics: dict[str, float],
    ) -> tuple[float, float, float]:
        """Convert a pixel (u, v) + depth to 3D point in camera frame.

        Args:
            u, v: Pixel coordinates.
            depth: Depth value in meters.
            intrinsics: Camera intrinsic parameters.

        Returns:
            (X, Y, Z) in camera coordinate frame.
        """
        fx = intrinsics.get("fx", 500.0)
        fy = intrinsics.get("fy", 500.0)
        cx = intrinsics.get("cx", 320.0)
        cy = intrinsics.get("cy", 240.0)
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return (x, y, z)
