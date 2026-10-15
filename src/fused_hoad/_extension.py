"""Lazy access to the compiled CUDA extension."""

from types import ModuleType


def load_extension() -> ModuleType:
    """Return the compiled extension or raise an actionable error."""
    try:
        from fused_hoad import _C
    except ImportError as exc:
        raise RuntimeError(
            "The fused_hoad CUDA extension is not built. Install the project on a "
            "CUDA-capable system with `pip install -e . --no-build-isolation`."
        ) from exc
    return _C

