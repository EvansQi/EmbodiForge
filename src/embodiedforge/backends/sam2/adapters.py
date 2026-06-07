"""SAM2 adapter — placeholder for real implementation.

Example config:
    segmentation:
      backend: sam2_local
      kwargs:
        model_cfg: sam2_hiera_l
        checkpoint: ./checkpoints/sam2_hiera_large.pt
        device: cuda
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.adapters.base import SegmentationAdapter
from embodiedforge.adapters.factory import register_adapter
from embodiedforge.schemas.geometry import BBox, Point2D


class LocalSAM2Adapter(SegmentationAdapter):
    """Local SAM2 inference."""

    def __init__(self, model_cfg: str = "sam2_hiera_l", checkpoint: str = "", device: str = "cuda", **kwargs: Any):
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    def _load_model(self):
        if self._predictor is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
            self._predictor = SAM2ImagePredictor(sam2_model)
        except ImportError as e:
            raise ImportError(
                "SAM2 requires the sam2 package. "
                "Install with: pip install embodiedforge[sam2]"
            ) from e

    def segment(
        self,
        image: Image.Image,
        bboxes: list[BBox] | None = None,
        points: list[Point2D] | None = None,
        point_labels: list[int] | None = None,
    ) -> np.ndarray:
        self._load_model()
        # TODO: Implement actual SAM2 inference
        raise NotImplementedError(
            "LocalSAM2Adapter.segment() is a placeholder. "
            "Implement model inference or use 'mock' backend."
        )

    def propagate_video(
        self,
        images: list[Image.Image],
        initial_mask: np.ndarray,
    ) -> list[np.ndarray]:
        self._load_model()
        # TODO: Implement SAM2 video propagation
        raise NotImplementedError(
            "LocalSAM2Adapter.propagate_video() is a placeholder."
        )


class RemoteSAM2Adapter(SegmentationAdapter):
    """Remote SAM2 via HTTP service."""

    def __init__(self, api_url: str = "http://localhost:8001", **kwargs: Any):
        self.api_url = api_url

    def segment(
        self,
        image: Image.Image,
        bboxes: list[BBox] | None = None,
        points: list[Point2D] | None = None,
        point_labels: list[int] | None = None,
    ) -> np.ndarray:
        # TODO: Implement HTTP call to remote SAM2 service
        raise NotImplementedError(
            "RemoteSAM2Adapter.segment() is a placeholder. "
            "Implement HTTP inference or use 'mock' backend."
        )


register_adapter("segmentation", "sam2_local", LocalSAM2Adapter)
register_adapter("segmentation", "sam2_remote", RemoteSAM2Adapter)
