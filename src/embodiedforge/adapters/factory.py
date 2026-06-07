"""Adapter factory — creates adapter instances from YAML config.

The factory pattern lets us switch between mock/local/remote backends
without changing any pipeline code. Just change the config YAML.
"""

from __future__ import annotations

from typing import Any

from embodiedforge.adapters.base import (
    DepthAdapter,
    GroundingAdapter,
    SegmentationAdapter,
    SemanticLabelerAdapter,
)

# Registry of all known adapter implementations
_ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(adapter_type: str, backend_name: str, cls: type) -> None:
    """Register an adapter class under 'adapter_type:backend_name'."""
    key = f"{adapter_type}:{backend_name}"
    _ADAPTER_REGISTRY[key] = cls


def _auto_discover() -> None:
    """Import all backend modules to trigger registration."""
    import embodiedforge.backends.mock.adapters  # noqa: F401

    # Real backends are imported only when requested
    try:
        import embodiedforge.backends.qwen_vl.adapters  # noqa: F401
    except ImportError:
        pass
    try:
        import embodiedforge.backends.groundingdino.adapters  # noqa: F401
    except ImportError:
        pass
    try:
        import embodiedforge.backends.sam2.adapters  # noqa: F401
    except ImportError:
        pass
    try:
        import embodiedforge.backends.depth_anything.adapters  # noqa: F401
    except ImportError:
        pass


def create_adapter(adapter_type: str, backend_name: str, **kwargs: Any) -> Any:
    """Create an adapter instance by type and backend name.

    Args:
        adapter_type: One of 'semantic', 'grounding', 'segmentation', 'depth'.
        backend_name: Registered backend name, e.g. 'mock', 'qwen_vl_local', 'sam2_local'.
        **kwargs: Extra configuration passed to the adapter constructor.

    Returns:
        An adapter instance conforming to the appropriate abstract interface.

    Raises:
        ValueError: If the backend is not registered or adapter type is unknown.
    """
    _auto_discover()

    key = f"{adapter_type}:{backend_name}"
    cls = _ADAPTER_REGISTRY.get(key)
    if cls is None:
        available = [k for k in _ADAPTER_REGISTRY if k.startswith(f"{adapter_type}:")]
        raise ValueError(
            f"No adapter registered for '{key}'. "
            f"Available for type '{adapter_type}': {available}"
        )
    return cls(**kwargs)


def create_semantic_adapter(config: dict[str, Any]) -> SemanticLabelerAdapter:
    """Create semantic labeler from pipeline config dict."""
    backend = config.get("backend", "mock")
    return create_adapter("semantic", backend, **config.get("kwargs", {}))


def create_grounding_adapter(config: dict[str, Any]) -> GroundingAdapter:
    """Create grounding adapter from pipeline config dict."""
    backend = config.get("backend", "mock")
    return create_adapter("grounding", backend, **config.get("kwargs", {}))


def create_segmentation_adapter(config: dict[str, Any]) -> SegmentationAdapter:
    """Create segmentation adapter from pipeline config dict."""
    backend = config.get("backend", "mock")
    return create_adapter("segmentation", backend, **config.get("kwargs", {}))


def create_depth_adapter(config: dict[str, Any]) -> DepthAdapter:
    """Create depth adapter from pipeline config dict."""
    backend = config.get("backend", "mock")
    return create_adapter("depth", backend, **config.get("kwargs", {}))
