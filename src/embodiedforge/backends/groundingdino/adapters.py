"""GroundingDINO adapter — placeholder for real implementation.

Example config:
    grounding:
      backend: groundingdino_local
      kwargs:
        model_path: IDEA-Research/groundingdino-base
        device: cuda
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from embodiedforge.adapters.base import GroundingAdapter
from embodiedforge.adapters.factory import register_adapter
from embodiedforge.schemas.geometry import BBox


class LocalGroundingDINOAdapter(GroundingAdapter):
    """Local GroundingDINO inference."""

    def __init__(self, model_path: str = "IDEA-Research/groundingdino-base", device: str = "cuda", **kwargs: Any):
        self.model_path = model_path
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from groundingdino.util.inference import load_model as gdino_load

            self._model = gdino_load(
                self.model_path,  # This may need adjustment per actual API
                device=self.device,
            )
        except ImportError as e:
            raise ImportError(
                "GroundingDINO requires the groundingdino package. "
                "Install with: pip install embodiedforge[grounding]"
            ) from e

    def ground(
        self,
        image: Image.Image,
        query: str,
        confidence_threshold: float = 0.3,
    ) -> list[BBox]:
        self._load_model()
        # TODO: Implement actual GroundingDINO inference
        raise NotImplementedError(
            "LocalGroundingDINOAdapter.ground() is a placeholder. "
            "Implement model inference or use 'mock' backend."
        )


register_adapter("grounding", "groundingdino_local", LocalGroundingDINOAdapter)
