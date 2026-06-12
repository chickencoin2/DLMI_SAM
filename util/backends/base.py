"""Abstract backend interface + normalized data contracts."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("DLMI_SAM_LABELER.Backends")


def to_pil_rgb(frame):
    """Coerce a frame (PIL.Image or HxWx3 RGB numpy) to a PIL RGB image."""
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    return Image.fromarray(arr).convert("RGB")


# Exceptions
class BackendError(Exception):
    """Base class for backend errors."""


class BackendLoadError(BackendError):
    """Raised when a backend cannot be loaded (package missing, checkpoint download/auth failure, OOM, ...)."""


class BackendCapabilityError(BackendError):
    """Raised when a backend doesn't support a code path; callers should branch on the capability flags instead."""


# Normalized result containers
@dataclass
class TrackResult:
    """Per-object segmentation result for a single frame."""

    obj_ids: List[int] = field(default_factory=list)
    masks: Dict[int, np.ndarray] = field(default_factory=dict)
    scores: Dict[int, float] = field(default_factory=dict)
    confidence: Dict[int, np.ndarray] = field(default_factory=dict)
    frame_idx: Optional[int] = None


@dataclass
class DetectResult:
    """Open-vocabulary / exemplar detection result for a single image."""

    masks: List[np.ndarray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class FramePack:
    """Opaque, backend-specific result of preprocessing one frame."""

    original_size: Tuple[int, int]
    payload: Any = None


# Backend interface
class SamBackend(abc.ABC):
    """Abstract base for all model backends."""

    # Identity / capabilities (override in subclasses)
    key: str = "base"
    label: str = "Base"
    supports_streaming: bool = False
    requires_preloaded_video: bool = False
    supports_dlmi: bool = True

    def __init__(self, app, device, dtype):
        self.app = app
        self.device = device
        self.dtype = dtype
        self._loaded = False

    # ----- lifecycle (abstract) ------------------------------------------- #
    @abc.abstractmethod
    def load(self) -> None:
        """Load all models/processors onto `self.device`."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release all models and free VRAM (del refs + empty cache)."""

    def is_loaded(self) -> bool:
        return self._loaded

    def _todo(self, name: str):
        raise NotImplementedError(f"{type(self).__name__}.{name} not implemented yet")

    # ----- preprocessing --------------------------------------------------- #
    def process_frame(self, frame) -> FramePack:
        """Preprocess one RGB frame (PIL or numpy) into a reusable FramePack."""
        self._todo("process_frame")

    # ----- PVS: tracker session ------------------------------------------- #
    def init_tracker_session(self, frames: Optional[list] = None,
                             video_size: Optional[Tuple[int, int]] = None,
                             num_frames: Optional[int] = None,
                             preserve: bool = False) -> bool:
        """Create a (streaming or preloaded) point/box/mask tracking session."""
        self._todo("init_tracker_session")

    def reset_tracker_session(self) -> None:
        self._todo("reset_tracker_session")

    def tracker_clear_objects(self) -> None:
        self._todo("tracker_clear_objects")

    def tracker_add_prompt(self, frame_pack: FramePack, frame_idx: int, obj_id: int,
                           *, points: Optional[np.ndarray] = None,
                           labels: Optional[np.ndarray] = None,
                           box: Optional[Tuple[float, float, float, float]] = None,
                           mask: Optional[np.ndarray] = None) -> TrackResult:
        """Add a point/box/mask prompt for `obj_id` on `frame_idx` and return the resulting masks for that frame."""
        self._todo("tracker_add_prompt")

    def tracker_add_prompts_batch(self, frame_pack: FramePack, frame_idx: int,
                                  obj_ids: List[int], *,
                                  masks_by_oid: Optional[Dict[int, np.ndarray]] = None,
                                  run_forward: bool = True) -> Optional[TrackResult]:
        """Register multiple mask prompts at once (polygon / DLMI seeding)."""
        last = None
        for oid in obj_ids:
            m = None if masks_by_oid is None else masks_by_oid.get(oid)
            last = self.tracker_add_prompt(frame_pack, frame_idx, oid, mask=m)
        return last

    def tracker_forward_frame(self, frame_pack: FramePack, frame_idx: int,
                              use_existing_frame: bool = False) -> TrackResult:
        """Streaming per-frame inference (push). Valid only if supports_streaming."""
        raise BackendCapabilityError(
            f"{self.key}: tracker_forward_frame not supported (use tracker_propagate)")

    def tracker_propagate(self, start_frame_idx: int, max_frames: int,
                          reverse: bool = False) -> Iterator[TrackResult]:
        """Generator-style propagation (pull). Valid only if not supports_streaming."""
        raise BackendCapabilityError(
            f"{self.key}: tracker_propagate not supported (use tracker_forward_frame)")

    @property
    def tracker_obj_ids(self) -> List[int]:
        """Object ids currently registered in the tracker session (insertion order)."""
        return []

    @property
    def tracker_session_active(self) -> bool:
        return False

    # ----- PCS: text-prompted video --------------------------------------- #
    def init_pcs_session(self, frames: Optional[list] = None,
                         streaming: bool = True) -> bool:
        self._todo("init_pcs_session")

    def pcs_add_text(self, text: str) -> None:
        self._todo("pcs_add_text")

    def pcs_forward_frame(self, frame_pack: FramePack, frame_idx: int) -> TrackResult:
        raise BackendCapabilityError(f"{self.key}: pcs_forward_frame not supported")

    def pcs_detect_frame0(self, text: str, frame_idx: int = 0) -> TrackResult:
        raise BackendCapabilityError(f"{self.key}: pcs_detect_frame0 not supported")

    def pcs_propagate(self, start_frame_idx: int, max_frames: int) -> Iterator[TrackResult]:
        raise BackendCapabilityError(f"{self.key}: pcs_propagate not supported")

    # ----- image segmentation / exemplars --------------------------------- #
    def image_detect(self, frame, *, text: Optional[str] = None,
                     boxes: Optional[List[Tuple[float, float, float, float]]] = None,
                     box_labels: Optional[List[int]] = None,
                     threshold: float = 0.5,
                     mask_threshold: float = 0.0) -> DetectResult:
        """Single-image open-vocabulary (text) and/or exemplar-box detection."""
        self._todo("image_detect")

    # ----- DLMI (Direct Latent Memory Injection) -------------------------- #
    def dlmi_supported(self) -> bool:
        return self.supports_dlmi

    def dlmi_install_injection(self, obj_ids: List[int],
                               masks_by_oid: Dict[int, np.ndarray], *,
                               intensity: float,
                               state: Optional[dict] = None,
                               **logit_kwargs) -> Any:
        """Install a one-shot latent-memory injection hook for the next forward."""
        self._todo("dlmi_install_injection")

    def dlmi_cleanup_injection(self, handle: Any) -> None:
        """Restore the model after a one-shot injection (always call in finally)."""
        self._todo("dlmi_cleanup_injection")

    def dlmi_install_persistent(self, *, preserve: bool, boost: bool) -> None:
        """Install persistent memory hooks: Preserve (keep all conditioning frames) and/or Boost (over-weight conditioning memory)."""
        self._todo("dlmi_install_persistent")

    def dlmi_remove_persistent(self) -> None:
        self._todo("dlmi_remove_persistent")

    def dlmi_mini_propagate(self, frame_n, frame_n1, obj_id_to_mask_label: dict, *,
                            dlmi_enabled: bool, intensity: float,
                            **logit_kwargs) -> Dict[int, np.ndarray]:
        """Two-frame mini session (Cut workflow): seed masks on frame_n, DLMI-propagate to frame_n+1."""
        self._todo("dlmi_mini_propagate")

    # ----- misc ------------------------------------------------------------ #
    def __repr__(self):
        return f"<{type(self).__name__} key={self.key!r} loaded={self._loaded}>"
