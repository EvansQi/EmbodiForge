"""Qwen-VL adapter — Local and Remote implementations.

Supported backends:
- mock: heuristic-based labels (always available, no API needed)
- qwen_vl_local: local HuggingFace transformers inference (needs GPU + torch)
- qwen_vl_remote: remote HTTP API (DashScope, vLLM, or any OpenAI-compatible endpoint)

Adapters provided:
- SemanticLabelerAdapter  (via qwen_vl_local / qwen_vl_remote)
- GroundingAdapter         (via qwen_vl_remote)

Config examples:

    # DashScope (Alibaba Cloud official API)
    semantic:
      backend: qwen_vl_remote
      kwargs:
        provider: dashscope
        api_key: sk-xxxxxxxxxxxxxxxxxx
        model: qwen-vl-max

    # vLLM or any OpenAI-compatible endpoint
    semantic:
      backend: qwen_vl_remote
      kwargs:
        provider: openai_compatible
        api_url: http://gpu-server:8000/v1/chat/completions
        api_key: not-needed
        model: Qwen2.5-VL-7B-Instruct

    # Grounding via the same Qwen-VL API
    grounding:
      backend: qwen_vl_remote
      kwargs:
        provider: dashscope
        model: qwen3-vl-plus
        api_key: sk-xxxxxxxxxxxxxxxxxx

    # Via environment variable
    export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxx
    semantic:
      backend: qwen_vl_remote
      kwargs:
        provider: dashscope
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

from PIL import Image

from embodiedforge.adapters.base import GroundingAdapter, SemanticLabelerAdapter
from embodiedforge.adapters.factory import register_adapter
from embodiedforge.schemas.geometry import BBox
from embodiedforge.schemas.semantic import SemanticAnnotation

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SEMANTIC_PROMPT = """You are a robot manipulation data annotator. Analyze this egocentric robot camera image.

Task instruction: {task_instruction}
Current skill stage: {skill_id}

Identify and return a JSON object with these fields:
- target_object: The name of the object being manipulated (e.g., "red cup", "USB cable", "blue block")
- target_part: The specific part of the object relevant to the current action (e.g., "handle", "rim", "connector tip", "top surface")
- affordance_query: A phrase describing where and how the robot should interact (e.g., "grasp point on the handle", "insertion point on the port")
- stage_label: A brief (5-10 word) description of what the robot is currently doing in this frame
- confidence: A float 0.0-1.0 indicating your confidence in this annotation

Return ONLY the JSON object, no markdown, no extra text. Example:
{{"target_object": "red cup", "target_part": "handle", "affordance_query": "grasp point on cup handle", "stage_label": "robot hand approaching cup handle", "confidence": 0.9}}"""


def _parse_semantic_response(text: str) -> dict[str, Any]:
    """Extract JSON from VLM response text, handling markdown code blocks."""
    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    # Try to find bare JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: use text as stage_label
        return {
            "target_object": "unknown",
            "target_part": "unknown",
            "affordance_query": "interaction point",
            "stage_label": text[:100] if len(text) > 100 else text,
            "confidence": 0.5,
        }

    return {
        "target_object": data.get("target_object", "unknown"),
        "target_part": data.get("target_part", "unknown"),
        "affordance_query": data.get("affordance_query", "interaction point"),
        "stage_label": data.get("stage_label", ""),
        "confidence": float(data.get("confidence", 0.7)),
    }


def _encode_image_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode PIL Image to base64 data URI string."""
    buf = io.BytesIO()
    # Convert RGBA to RGB if needed
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buf, format=format, quality=85)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Grounding prompt template
# ---------------------------------------------------------------------------

_GROUNDING_PROMPT = """You are a visual grounding assistant for robot manipulation.
Locate the object described by the query in this egocentric robot camera image.

Image dimensions: {image_width} x {image_height} pixels.

Query: {query}

Return a JSON object with bounding boxes in PIXEL coordinates. The format is:
{{"bboxes": [{{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "label": "<query text>", "confidence": <float 0-1>}}]}}

Rules:
- x1,y1 is the top-left corner, x2,y2 is the bottom-right corner
- Coordinates MUST be within the image bounds (0 to {image_width} for x, 0 to {image_height} for y)
- Return the ONE best matching bounding box
- confidence should reflect how certain you are the box correctly locates the query

Return ONLY the JSON object, no markdown, no extra text."""


def _parse_grounding_response(text: str, img_w: int, img_h: int) -> list[BBox]:
    """Extract bounding boxes from VLM grounding response."""
    # Try to find JSON in markdown code block first
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()
    # Try to find bare JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    bboxes: list[BBox] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return bboxes

    raw_boxes = data.get("bboxes", [data] if "x1" in data else [])

    for raw in raw_boxes:
        try:
            # Clamp coordinates to image bounds
            x1 = max(0, min(int(raw.get("x1", 0)), img_w))
            y1 = max(0, min(int(raw.get("y1", 0)), img_h))
            x2 = max(0, min(int(raw.get("x2", img_w)), img_w))
            y2 = max(0, min(int(raw.get("y2", img_h)), img_h))

            # Validate: x2 > x1, y2 > y1
            if x2 <= x1 or y2 <= y1:
                continue

            bboxes.append(BBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                label=raw.get("label", ""),
                confidence=float(raw.get("confidence", 0.5)),
            ))
        except (ValueError, TypeError):
            continue

    # Sort by confidence descending
    bboxes.sort(key=lambda b: b.confidence, reverse=True)
    return bboxes


def _infer_object_from_instruction(instruction: str) -> str:
    """Simple heuristic fallback when VLM gives 'unknown'."""
    keywords = ["cup", "block", "plug", "can", "bottle", "drawer", "button", "peg"]
    lower = instruction.lower()
    for kw in keywords:
        if kw in lower:
            return kw
    return "target object"

# ---------------------------------------------------------------------------
# Local adapter
# ---------------------------------------------------------------------------


class LocalQwenVLAdapter(SemanticLabelerAdapter):
    """Local Qwen-VL inference via HuggingFace transformers.

    Requires: pip install embodiedforge[vision]
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda", **kwargs: Any):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the model on first call."""
        if self._model is not None:
            return
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, device_map=self.device
            )
            self._processor = AutoProcessor.from_pretrained(self.model_path)
        except ImportError as e:
            raise ImportError(
                "Qwen-VL requires transformers and torch. "
                "Install with: pip install embodiedforge[vision]"
            ) from e

    def label(
        self,
        image: Image.Image,
        task_instruction: str,
        skill_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> SemanticAnnotation:
        self._load_model()

        prompt = _SEMANTIC_PROMPT.format(
            task_instruction=task_instruction,
            skill_id=skill_id,
        )

        # TODO: Implement actual model inference when transformers is available
        # messages = [{"role": "user", "content": [
        #     {"type": "image", "image": image},
        #     {"type": "text", "text": prompt}
        # ]}]
        # inputs = self._processor(messages, return_tensors="pt").to(self.device)
        # output = self._model.generate(**inputs, max_new_tokens=256)
        # response = self._processor.decode(output[0], skip_special_tokens=True)

        raise NotImplementedError(
            "LocalQwenVLAdapter.label() requires transformers + torch. "
            "Use 'qwen_vl_remote' with an API key for immediate use, "
            "or switch to 'mock' backend for testing."
        )


# ---------------------------------------------------------------------------
# Shared base for Qwen-VL remote API adapters
# ---------------------------------------------------------------------------


class _BaseQwenVLRemote:
    """Shared HTTP API client for Qwen-VL remote adapters (DashScope / OpenAI-compatible).

    Handles API key resolution, message building, and HTTP calls.
    Subclass alongside an adapter ABC to create concrete adapters.
    """

    DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(
        self,
        provider: str = "dashscope",
        api_url: str = "",
        api_key: str = "",
        model: str = "qwen-vl-max",
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: int = 60,
        **kwargs: Any,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Resolve API key
        if provider == "dashscope":
            self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
            self.api_url = api_url or self.DASHSCOPE_URL
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.api_url = api_url or "http://localhost:8000/v1/chat/completions"

    def _call_api(self, messages: list[dict]) -> str:
        """Send request to the API and return response text."""
        import urllib.request
        import urllib.error

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(
                f"Qwen-VL API returned HTTP {e.code}: {error_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Qwen-VL API at {self.api_url}: {e.reason}"
            ) from e

        # Extract content from OpenAI-compatible response
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                f"Unexpected API response format. Expected OpenAI-compatible, got: "
                f"{json.dumps(body, ensure_ascii=False)[:500]}"
            )


# ---------------------------------------------------------------------------
# Remote Semantic Labeler
# ---------------------------------------------------------------------------


class RemoteQwenVLAdapter(_BaseQwenVLRemote, SemanticLabelerAdapter):
    """Remote Qwen-VL semantic labeler via HTTP API.

    Two provider modes:
    1. dashscope: Alibaba Cloud official Qwen-VL API
       - api_key: Your DashScope API key (also reads DASHSCOPE_API_KEY env var)
       - model: qwen-vl-max, qwen-vl-plus, qwen3-vl-plus, etc.
       - api_url: auto-set to DashScope endpoint

    2. openai_compatible: Any OpenAI-compatible endpoint (vLLM, local server, etc.)
       - api_url: Full URL to chat completions endpoint
       - api_key: API key if needed (can be empty for local vLLM)
       - model: Model name as registered on the server
    """

    def _build_messages(self, image: Image.Image, task_instruction: str, skill_id: str) -> list[dict]:
        """Build OpenAI-compatible messages with base64 image and semantic prompt."""
        prompt = _SEMANTIC_PROMPT.format(
            task_instruction=task_instruction,
            skill_id=skill_id,
        )
        data_uri = _encode_image_base64(image)

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def label(
        self,
        image: Image.Image,
        task_instruction: str,
        skill_id: str = "",
        context: dict[str, Any] | None = None,
    ) -> SemanticAnnotation:
        """Call remote Qwen-VL API and parse the result."""
        if not self.api_key and self.provider == "dashscope":
            raise RuntimeError(
                "DashScope API key is required. Set it via one of:\n"
                "  - config: semantic.kwargs.api_key: sk-xxxx\n"
                "  - env: export DASHSCOPE_API_KEY=sk-xxxx"
            )

        messages = self._build_messages(image, task_instruction, skill_id)
        response_text = self._call_api(messages)
        parsed = _parse_semantic_response(response_text)

        # Fallback for 'unknown' target_object
        if parsed["target_object"] == "unknown":
            parsed["target_object"] = _infer_object_from_instruction(task_instruction)

        frame_idx = context.get("frame_idx", 0) if context else 0
        stage_id = context.get("stage_id", 0) if context else 0

        return SemanticAnnotation(
            frame_idx=frame_idx,
            stage_id=stage_id,
            skill_id=skill_id or "unknown",
            stage_label=parsed["stage_label"],
            target_object=parsed["target_object"],
            target_part=parsed["target_part"],
            affordance_query=parsed["affordance_query"],
            confidence=parsed["confidence"],
        )


# ---------------------------------------------------------------------------
# Remote Visual Grounding
# ---------------------------------------------------------------------------


class RemoteQwenVLGroundingAdapter(_BaseQwenVLRemote, GroundingAdapter):
    """Remote Qwen-VL visual grounding via HTTP API.

    Uses the same DashScope / OpenAI-compatible endpoint as the semantic adapter,
    but with a grounding-specific prompt that requests bounding box coordinates.

    Config example:
        grounding:
          backend: qwen_vl_remote
          kwargs:
            provider: dashscope
            model: qwen3-vl-plus
            api_key: sk-xxxxxxxxxxxxxxxxxx
    """

    def ground(
        self,
        image: Image.Image,
        query: str,
        confidence_threshold: float = 0.3,
    ) -> list[BBox]:
        """Ground a text query in the image, returning bounding boxes."""
        if not self.api_key and self.provider == "dashscope":
            raise RuntimeError(
                "DashScope API key is required. Set it via one of:\n"
                "  - config: grounding.kwargs.api_key: sk-xxxx\n"
                "  - env: export DASHSCOPE_API_KEY=sk-xxxx"
            )

        img_w, img_h = image.size

        prompt = _GROUNDING_PROMPT.format(
            image_width=img_w,
            image_height=img_h,
            query=query,
        )
        data_uri = _encode_image_base64(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response_text = self._call_api(messages)
        bboxes = _parse_grounding_response(response_text, img_w, img_h)

        # Filter by confidence threshold
        return [b for b in bboxes if b.confidence >= confidence_threshold]


# Register all adapters
register_adapter("semantic", "qwen_vl_local", LocalQwenVLAdapter)
register_adapter("semantic", "qwen_vl_remote", RemoteQwenVLAdapter)
register_adapter("grounding", "qwen_vl_remote", RemoteQwenVLGroundingAdapter)
