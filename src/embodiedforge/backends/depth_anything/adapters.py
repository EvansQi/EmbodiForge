"""Depth Anything adapter — placeholder for real implementation.

Example config:
    depth:
      backend: depth_anything_local
      kwargs:
        encoder: vitl
        checkpoint: ./checkpoints/depth_anything_v2_vitl.pth
        device: cuda
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from embodiedforge.adapters.base import DepthAdapter
from embodiedforge.adapters.factory import register_adapter


class LocalDepthAnythingAdapter(DepthAdapter):
    """Local Depth Anything V2 inference."""

    def __init__(self, encoder: str = "vitl", checkpoint: str = "", device: str = "cuda", **kwargs: Any):
        self.encoder = encoder
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            # from depth_anything_v2.dpt import DepthAnythingV2
            # self._model = DepthAnythingV2(encoder=self.encoder, ...)
            # self._model.load_state_dict(torch.load(self.checkpoint))
            pass
        except ImportError as e:
            raise ImportError(
                "Depth Anything requires depth_anything_v2 package. "
                "Install with: pip install embodiedforge[depth]"
            ) from e

    def estimate_depth(
        self,
        image: Image.Image,
        camera_intrinsics: dict[str, float] | None = None,
    ) -> np.ndarray:
        self._load_model()
        # TODO: Implement actual Depth Anything inference
        raise NotImplementedError(
            "LocalDepthAnythingAdapter.estimate_depth() is a placeholder. "
            "Implement model inference or use 'mock' backend."
        )


register_adapter("depth", "depth_anything_local", LocalDepthAnythingAdapter)
