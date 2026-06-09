"""HFBackend - HuggingFace `transformers` SAM3 backend (the original path).

Migration "Layer 1" (handle mirroring): `load()` delegates to the app's existing,
known-good `_init_sam3_models()`, which sets `app.pcs_model / tracker_model /
image_model` (+ processors) and `app.model_dtype` directly. Because those handles
live on the app, every existing model call-site keeps working unchanged while the
normalized SamBackend operations are added incrementally in later phases.
"""

from __future__ import annotations

import gc
import logging
from typing import List, Optional

import numpy as np
import torch

from .base import SamBackend, BackendLoadError, DetectResult, to_pil_rgb

logger = logging.getLogger("DLMI_SAM_LABELER.Backends.HF")

# Handles the HF path sets on the app (mirrored bridge for legacy call-sites + SAM2).
_APP_MODEL_HANDLES = (
    "pcs_model", "pcs_processor",
    "tracker_model", "tracker_processor",
    "image_model", "image_processor",
)
_APP_SESSIONS = ("inference_session", "pcs_inference_session", "pcs_streaming_session")


class HFBackend(SamBackend):
    key = "hug"
    label = "HuggingFace"
    supports_streaming = True          # native per-frame streaming session
    requires_preloaded_video = False
    supports_dlmi = True

    # ----- lifecycle ------------------------------------------------------- #
    def load(self) -> None:
        if self._loaded:
            return
        app = self.app
        # Reuse the existing loader; it sets app.* handles + app.model_dtype and
        # logs/handles its own errors (leaving handles None on failure).
        app._init_sam3_models()
        if getattr(app, "tracker_model", None) is None:
            raise BackendLoadError("HuggingFace SAM3 model load failed (see logs).")
        self.dtype = getattr(app, "model_dtype", self.dtype)
        self._loaded = True
        logger.info("HFBackend loaded (handles mirrored on app).")

    def unload(self) -> None:
        app = self.app
        for attr in _APP_MODEL_HANDLES:
            setattr(app, attr, None)
        for attr in _APP_SESSIONS:
            if hasattr(app, attr):
                setattr(app, attr, None)
        self._loaded = False
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("HFBackend unloaded (handles cleared, CUDA cache emptied).")

    # ----- session introspection ------------------------------------------ #
    @property
    def tracker_obj_ids(self) -> List[int]:
        sess = getattr(self.app, "inference_session", None)
        if sess is not None and hasattr(sess, "obj_ids"):
            try:
                return list(sess.obj_ids)
            except Exception:
                return []
        return []

    @property
    def tracker_session_active(self) -> bool:
        return getattr(self.app, "inference_session", None) is not None

    # ----- image segmentation / exemplars (HF) ---------------------------- #
    def image_detect(self, frame, *, text=None, boxes=None, box_labels=None,
                     threshold=0.5, mask_threshold=0.0) -> DetectResult:
        app = self.app
        pil = to_pil_rgb(frame)
        w, h = pil.size
        kwargs = {"images": pil, "return_tensors": "pt"}
        if text:
            kwargs["text"] = text
        if boxes is not None and len(boxes) > 0:
            kwargs["input_boxes"] = [list(boxes)]
            if box_labels is not None:
                kwargs["input_boxes_labels"] = [list(box_labels)]
        inputs = app.image_processor(**kwargs).to(self.device)
        if self.dtype == torch.float32 and hasattr(inputs, "pixel_values"):
            try:
                inputs.pixel_values = inputs.pixel_values.to(dtype=torch.float32)
            except Exception:
                pass
        with torch.inference_mode():
            outputs = app.image_model(**inputs)
        target_sizes = inputs.get("original_sizes")
        target_sizes = target_sizes.tolist() if target_sizes is not None else [[h, w]]
        results = app.image_processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=target_sizes)[0]
        det = DetectResult()
        masks = results.get("masks", []) if isinstance(results, dict) else []
        if masks is not None and len(masks) > 0:
            scores = results.get("scores", [1.0] * len(masks))
            rboxes = results.get("boxes", [None] * len(masks))
            for i in range(len(masks)):
                m = masks[i]
                mnp = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
                det.masks.append(np.squeeze(mnp))
                sc = scores[i] if i < len(scores) else 1.0
                det.scores.append(float(sc.item()) if torch.is_tensor(sc) else float(sc))
                b = rboxes[i] if i < len(rboxes) else None
                if b is not None:
                    bb = b.cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
                    flat = np.asarray(bb).flatten()
                    if flat.size >= 4:
                        det.boxes.append(tuple(float(x) for x in flat[:4]))
        return det

    # NOTE: process_frame / tracker_* / pcs_* / dlmi_* are added in later phases
    # (those call-sites still use the mirrored app.* handles for now).
