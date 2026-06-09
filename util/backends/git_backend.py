"""GitBackend - official GitHub `sam3` package backend (SAM3 and SAM3.1).

Uses the LOW-LEVEL official model API (build_sam3_image_model / build_sam3_video_model
/ build_sam3_multiplex_video_model). One class parametrised by `version`:
  * version="sam3"   -> build_sam3_video_model().tracker          (Sam3TrackerPredictor)
  * version="sam3.1" -> build_sam3_multiplex_video_model()        (demo model IS the tracker)

The image model is loaded eagerly in load(); the (heavier) video model is loaded
lazily on first tracker/pcs use. Image segmentation is implemented here; tracker /
PCS / DLMI are added in later phases.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from typing import List, Optional

import numpy as np
import torch

from .base import SamBackend, BackendLoadError, DetectResult, to_pil_rgb

logger = logging.getLogger("DLMI_SAM_LABELER.Backends.Git")

# checkpoint metadata per version (for logging / sanity)
_CKPT = {"sam3": ("facebook/sam3", "sam3.pt"),
         "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt")}

_SDPA_PATCHED = False


def _patch_decoder_sdpa_allow_math():
    """Let the official tracker run in fp32 as well as its native bf16.

    The official decoder hardcodes ``with sdpa_kernel(SDPBackend.FLASH_ATTENTION)``
    around its attention (sam3/model/decoder.py), and FlashAttention has no fp32
    kernel -> fp32 raises "No available kernel. Aborting execution." We override the
    decoder's ``sdpa_kernel`` name so it ALSO permits the MATH (and efficient)
    backends: under bf16 PyTorch still selects Flash (fast, unchanged behaviour);
    under fp32 it falls back to the MATH kernel. Harmless to the default bf16 path,
    and it is what makes the optional fp32 mode work. HF (transformers) is untouched.
    Applied once, process-wide."""
    global _SDPA_PATCHED
    if _SDPA_PATCHED:
        return
    try:
        import sam3.model.decoder as _dec
        from torch.nn.attention import sdpa_kernel as _real_sdpa, SDPBackend as _B

        def _flex_sdpa(*_args, **_kwargs):
            return _real_sdpa([_B.MATH, _B.EFFICIENT_ATTENTION, _B.FLASH_ATTENTION])

        _dec.sdpa_kernel = _flex_sdpa
        _SDPA_PATCHED = True
        logger.info("Patched sam3 decoder.sdpa_kernel to allow MATH (fp32-capable).")
    except Exception as e:
        logger.warning(f"decoder sdpa patch failed (fp32 mode may error): {e}")


def _np(t):
    """Tensor/array -> numpy, upcasting non-integer tensors to float32 first
    (numpy() rejects bfloat16, which the official models use internally)."""
    if t is None:
        return None
    if torch.is_tensor(t):
        if t.dtype in (torch.bool, torch.uint8, torch.int16, torch.int32, torch.int64):
            return t.detach().cpu().numpy()
        return t.detach().float().cpu().numpy()
    return np.asarray(t)


def _xyxy_px_to_cxcywh_norm(box, width, height):
    """Pixel [x1,y1,x2,y2] -> normalised [cx,cy,w,h] in [0,1] (official box format)."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    bw = abs(x2 - x1) / max(width, 1)
    bh = abs(y2 - y1) / max(height, 1)
    return [cx, cy, bw, bh]


class GitBackend(SamBackend):
    supports_streaming = True       # tracker is driven via single-frame stepping
    requires_preloaded_video = False
    supports_dlmi = True

    def __init__(self, app, device, dtype, version="sam3"):
        super().__init__(app, device, dtype)
        self.version = version
        self.key = "git" if version == "sam3" else "3.1"
        self.label = "GitHub SAM3" if version == "sam3" else "GitHub SAM3.1"
        self._img_model = None
        self._img_proc = None
        self._video = None        # full video model (SAM3) / demo model (SAM3.1)
        self._predictor = None    # multiplex predictor stack (3.1; kept alive)
        self._tracker = None      # the object exposing init_state/propagate/_encode_new_memory
        self._orig_encode = None  # real tracker's original _encode_new_memory (DLMI detect)
        self._is_multiplex = (version == "sam3.1")
        self._device_str = "cuda"

    # ----- lifecycle ------------------------------------------------------- #
    def load(self) -> None:
        if self._loaded:
            return
        try:
            import sam3  # noqa: F401
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as e:
            raise BackendLoadError(f"official 'sam3' package import failed: {e}")

        # Allow the optional fp32 mode (default is the package-native bf16).
        _patch_decoder_sdpa_allow_math()

        self._device_str = "cuda" if (self.device is not None and
                                      getattr(self.device, "type", None) == "cuda") else "cpu"
        repo, ckpt = _CKPT[self.version]
        logger.info(f"GitBackend[{self.version}] loading image model ({repo}/{ckpt})...")
        try:
            self._img_model = build_sam3_image_model(
                device=self._device_str, load_from_HF=True,
                enable_inst_interactivity=True)
            # Precision policy (CLAUDE.md): official checkpoints ship in bf16;
            # upcast to fp32 to match the HF backend (torch_dtype=float32) and
            # avoid reducing precision relative to it.
            self._img_model = self._img_model.float()
            self._img_proc = Sam3Processor(self._img_model, device=self._device_str)
        except Exception as e:
            raise BackendLoadError(f"image model build failed ({self.version}): {e}")
        self._video = None
        self._tracker = None
        self._install_app_wrappers()
        self._loaded = True
        logger.info(f"GitBackend[{self.version}] image model ready (video lazy).")

    def _install_app_wrappers(self):
        """Set HF-API-compatible tracker wrappers on the app so the existing PVS
        streaming call-sites (sam_interaction / propagation_controller) work for git.
        HF-only handles are nulled (git routes image via backend.image_detect)."""
        from . import git_video
        app = self.app
        app.tracker_processor = git_video.GitTrackerProcessor(self)
        app.tracker_model = git_video.GitTrackerModel(self)
        app.inference_session = None
        app.image_model = None
        app.image_processor = None
        app.pcs_model = None
        app.pcs_processor = None

    def _num_frames_hint(self):
        app = self.app
        try:
            cache = getattr(app, "video_frames_cache", None)
            if cache:
                return len(cache)
        except Exception:
            pass
        try:
            import cv2
            cap = getattr(app, "cap", None)
            if cap is not None:
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if n > 0:
                    return n
        except Exception:
            pass
        return 100000

    # ----- precision (option) --------------------------------------------- #
    def _fp32_mode(self) -> bool:
        """True = run the official models in fp32 (HF-parity, via the sdpa MATH
        patch); False (default) = the package-native bf16. Driven by app.git_fp32_var
        so it can be toggled at runtime (the weights stay fp32 masters either way;
        only the autocast compute dtype changes)."""
        v = getattr(self.app, "git_fp32_var", None)
        try:
            return bool(v.get()) if v is not None else False
        except Exception:
            return False

    def _autocast_ctx(self, fp32=None):
        """Autocast context wrapping every official-model forward so the whole git/3.1
        pipeline runs at ONE consistent compute dtype (mixing fp32 weights with a
        leaked bf16 autocast is what caused the 'BFloat16 vs Float' crashes).

        `fp32` overrides the live setting — tracker sessions pass the value captured at
        session creation so a session stays internally consistent even if the user
        toggles the option mid-track (the new precision then applies to the next
        fresh session). image_detect is stateless and reads the live setting."""
        if self._device_str != "cuda":
            return contextlib.nullcontext()
        use_fp32 = self._fp32_mode() if fp32 is None else bool(fp32)
        dtype = torch.float32 if use_fp32 else torch.bfloat16
        return torch.autocast("cuda", dtype=dtype)

    def get_tracker_model(self):
        """The real official tracker object (for DLMI hook installation)."""
        self._ensure_video()
        return self._tracker

    def _ensure_video(self):
        """Lazily build the (heavy) video tracker on first use."""
        if self._tracker is not None:
            return
        if self.version == "sam3.1":
            # sam3.1_multiplex.pt is a FULL-stack checkpoint (keys prefixed
            # tracker.model.* / detector.*) → load via the predictor builder, then
            # extract the low-level multiplex demo tracker model. use_fa3=False keeps
            # us off FlashAttention3 (fp32 path).
            from sam3.model_builder import build_sam3_multiplex_video_predictor
            logger.info("GitBackend[sam3.1] building multiplex video predictor...")
            self._predictor = build_sam3_multiplex_video_predictor(
                use_fa3=False, warm_up=False)
            self._predictor.model.float()
            self._tracker = self._predictor.model.tracker.model  # VideoTrackingMultiplexDemo
            # Wire the shared VL backbone into the demo tracker (mirrors the SAM3
            # `predictor.backbone = detector.backbone` step).
            self._tracker.backbone = self._predictor.model.detector.backbone
        else:
            from sam3.model_builder import build_sam3_video_model
            logger.info("GitBackend[sam3] building video model...")
            self._video = build_sam3_video_model(
                load_from_HF=True, device=self._device_str).float()
            self._tracker = self._video.tracker
            self._tracker.backbone = self._video.detector.backbone
        # Remember the pristine memory-encoder so the video wrapper can detect when
        # a DLMI injection hook has been installed (and then encode memory for it).
        self._orig_encode = self._tracker._encode_new_memory
        logger.info(f"GitBackend[{self.version}] video tracker ready.")

    def unload(self) -> None:
        app = self.app
        for attr in ("tracker_model", "tracker_processor", "image_model",
                     "image_processor", "pcs_model", "pcs_processor",
                     "inference_session", "pcs_inference_session", "pcs_streaming_session"):
            if hasattr(app, attr):
                setattr(app, attr, None)
        for attr in ("_img_model", "_img_proc", "_video", "_tracker", "_predictor"):
            setattr(self, attr, None)
        self._orig_encode = None
        self._loaded = False
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info(f"GitBackend[{self.version}] unloaded.")

    # ----- image segmentation / exemplars (official grounding) ------------ #
    def image_detect(self, frame, *, text=None, boxes=None, box_labels=None,
                     threshold=0.5, mask_threshold=0.0) -> DetectResult:
        if not self._loaded:
            raise BackendLoadError("GitBackend.image_detect before load()")
        proc = self._img_proc
        pil = to_pil_rgb(frame)
        W, H = pil.size

        proc.confidence_threshold = float(threshold)
        has_box = boxes is not None and len(boxes) > 0
        # Run the official image model under the selected compute dtype (bf16 default
        # / fp32 option) so the backbone activations and weights never disagree
        # (the "mat1 and mat2 ... BFloat16 vs Float" failure, facebookresearch/sam3#507).
        with self._autocast_ctx():
            state = proc.set_image(pil)
            if has_box:
                labels = box_labels if box_labels is not None else [True] * len(boxes)
                for box, lbl in zip(boxes, labels):
                    cxcywh = _xyxy_px_to_cxcywh_norm(box, W, H)
                    state = proc.add_geometric_prompt(box=cxcywh, label=bool(lbl), state=state)
            if text:
                state = proc.set_text_prompt(prompt=text, state=state)
        if not has_box and not text:
            return DetectResult()

        det = DetectResult()
        masks = state.get("masks", None)
        if masks is None:
            return det
        if mask_threshold and "masks_logits" in state:
            masks = state["masks_logits"] > float(mask_threshold)
        masks_np = _np(masks)
        scores_np = _np(state.get("scores"))
        boxes_np = _np(state.get("boxes"))
        if masks_np.ndim < 3:
            masks_np = masks_np[None, ...]
        for i in range(masks_np.shape[0]):
            det.masks.append(np.squeeze(masks_np[i]))
            if scores_np is not None and i < len(scores_np):
                det.scores.append(float(scores_np[i]))
            else:
                det.scores.append(1.0)
            if boxes_np is not None and i < boxes_np.shape[0]:
                det.boxes.append(tuple(float(x) for x in np.asarray(boxes_np[i]).flatten()[:4]))
        return det

    # ----- session introspection (filled in P8/P9) ------------------------ #
    @property
    def tracker_obj_ids(self) -> List[int]:
        return []

    @property
    def tracker_session_active(self) -> bool:
        return False

    # tracker_* / pcs_* / dlmi_* implemented in later phases (P8-P10).
