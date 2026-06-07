"""Qwen-VL adapter — Local and Remote implementations.

Supported backends:
- mock: heuristic-based labels (always available, no API needed)
- qwen_vl_local: local HuggingFace transformers inference (needs GPU + torch)
- qwen_vl_remote: remote HTTP API (DashScope, vLLM, or any OpenAI-compatible endpoint)

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

from embodiedforge.adapters.base import SemanticLabelerAdapter
from embodiedforge.adapters.factory import register_adapter
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
# Remote adapter (DASHSCOPE / OpenAI-compatible)
# ---------------------------------------------------------------------------


class RemoteQwenVLAdapter(SemanticLabelerAdapter):
    """Remote Qwen-VL via HTTP API — supports DashScope, vLLM, and OpenAI-compatible endpoints.

    Two provider modes:
    1. dashscope: Alibaba Cloud official Qwen-VL API
       - api_key: Your DashScope API key (also reads DASHSCOPE_API_KEY env var)
       - model: qwen-vl-max, qwen-vl-plus, etc.
       - api_url: auto-set to DashScope endpoint

    2. openai_compatible: Any OpenAI-compatible endpoint (vLLM, local server, etc.)
       - api_url: Full URL to chat completions endpoint
       - api_key: API key if needed (can be empty for local vLLM)
       - model: Model name as registered on the server
    """

    # Default endpoints
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

        if not self.api_key and provider == "dashscope":
            # Don't raise here — let the first call surface the error
            pass

    def _build_messages(self, image: Image.Image, task_instruction: str, skill_id: str) -> list[dict]:
        """Build OpenAI-compatible messages with base64 image."""
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


# Register all adapters
register_adapter("semantic", "qwen_vl_local", LocalQwenVLAdapter)
register_adapter("semantic", "qwen_vl_remote", RemoteQwenVLAdapter)
