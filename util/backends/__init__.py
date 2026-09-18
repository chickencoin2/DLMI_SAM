"""SAM backend abstraction layer."""

from .base import (
    SamBackend,
    TrackResult,
    DetectResult,
    FramePack,
    BackendError,
    BackendLoadError,
    BackendCapabilityError,
)

__all__ = [
    "SamBackend",
    "TrackResult",
    "DetectResult",
    "FramePack",
    "BackendError",
    "BackendLoadError",
    "BackendCapabilityError",
    "get_backend_manager",
]


def get_backend_manager(app):
    """Lazily import and construct the BackendManager (avoids importing torch at package import time)."""
    from .manager import BackendManager
    return BackendManager(app)
