"""SAM backend abstraction layer.

This package decouples the application from a single model implementation so the
app can switch at runtime between:
  - "hug" : HuggingFace `transformers` SAM3   (HFBackend)
  - "git" : official GitHub `sam3` package, SAM3 checkpoint        (GitBackend)
  - "3.1" : official GitHub `sam3` package, SAM3.1 multiplex ckpt  (GitBackend)

The rest of the app talks to a `SamBackend` instance (via `app.backend`) using a
normalized, framework-agnostic interface (numpy masks / plain ints), and the
`BackendManager` owns the single active backend and handles load/unload/switch.

Heavy implementations (torch, transformers, sam3) are imported lazily inside the
concrete backend modules so importing this package stays cheap.
"""

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
    """Lazily import and construct the BackendManager (avoids importing torch at
    package import time)."""
    from .manager import BackendManager
    return BackendManager(app)
