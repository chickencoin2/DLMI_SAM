"""DLMI memory-encoder injection - fresh, self-contained implementation."""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from . import dlmi_core

logger = logging.getLogger("DLMI_SAM_LABELER.Backends.DLMIInject")


def collect_dlmi_settings(app) -> dict:
    """Read all DLMI tk options into the kwargs for build_injection_queue; single source of truth for every injection call-site."""
    def _get(name, default):
        var = getattr(app, name, None)
        if var is None:
            return default
        try:
            return var.get()
        except Exception:
            return default

    settings = {
        "intensity": float(_get("dlmi_alpha_var", dlmi_core.DEFAULT_INTENSITY)),
        "bg_confidence": (float(_get("dlmi_bg_conf_value_var", 0.0))
                          if _get("dlmi_bg_conf_enabled_var", False) else None),
        "boundary_soft": bool(_get("dlmi_boundary_soft_enabled_var", False)),
        "boundary_soft_inside": bool(_get("dlmi_boundary_soft_inside_var", True)),
        "boundary_soft_outside": bool(_get("dlmi_boundary_soft_outside_var", True)),
        "boundary_gradient": bool(_get("dlmi_boundary_soft_gradient_var", False)),
        "boundary_width_pct": float(_get("dlmi_boundary_soft_width_var", 1.0)),
        # FINAL confidence (%) the user wants at the mask boundary; the logit conversion happens inside dlmi_core.
        "boundary_conf_pct": float(_get("dlmi_boundary_soft_conf_var",
                                        dlmi_core.DEFAULT_BOUNDARY_CONF_PCT)),
    }
    return settings


def build_injection_queue(obj_ids: List[int], masks_by_oid: Dict[int, np.ndarray], *,
                          intensity: float, device, **logit_kwargs) -> List[torch.Tensor]:
    """Pre-compute one logit tensor per object id (in `obj_ids` order) on `device`."""
    queue: List[torch.Tensor] = []
    for oid in obj_ids:
        mask = masks_by_oid[oid]
        logit_np = dlmi_core.compute_logit_map(mask, intensity=intensity,
                                               **logit_kwargs)
        queue.append(torch.from_numpy(logit_np).to(device))
    return queue


def make_injection_hook(queue: List[torch.Tensor], original_encode, *,
                        multiplex: bool = False, log_prefix: str = "",
                        state: Optional[dict] = None):
    """Return a closure replacing `<model>._encode_new_memory`."""
    st = state if state is not None else {}
    st.setdefault("idx", 0)

    def hook(*args, **kwargs):
        phr = kwargs.get("pred_masks_high_res", None)
        if phr is None or not torch.is_tensor(phr):
            return original_encode(*args, **kwargs)

        ndim = phr.ndim
        if ndim == 4:                       # [N, 1, H, W]
            n_obj = phr.shape[0]
        elif ndim == 5:                     # [1, N, 1, H, W]
            n_obj = phr.shape[1]
        else:
            return original_encode(*args, **kwargs)
        tgt_h, tgt_w = phr.shape[-2], phr.shape[-1]

        start = st["idx"]
        tensors = []
        for i in range(n_obj):
            qi = start + i
            if qi >= len(queue):
                logger.warning(f"DLMI {log_prefix}: injection queue exhausted at idx={qi}")
                break
            t = queue[qi]
            if t.dim() == 2:
                t = t.view(1, 1, t.shape[0], t.shape[1])
            elif t.dim() == 3:
                t = t.unsqueeze(0)
            if (t.shape[-2], t.shape[-1]) != (tgt_h, tgt_w):
                t = torch.nn.functional.interpolate(
                    t.float(), size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
            tensors.append(t.to(device=phr.device, dtype=phr.dtype))

        if tensors and len(tensors) == n_obj:
            injected = torch.cat(tensors, dim=0)          # [N, 1, H, W]
            if ndim == 5:
                injected = injected.unsqueeze(0)          # [1, N, 1, H, W]
            if injected.shape == phr.shape:
                kwargs["pred_masks_high_res"] = injected
            else:
                try:
                    kwargs["pred_masks_high_res"] = injected.view_as(phr)
                except Exception as e:
                    logger.error(
                        f"DLMI {log_prefix}: shape mismatch {tuple(injected.shape)} "
                        f"vs {tuple(phr.shape)}: {e}")
            st["idx"] += n_obj

            # Auto-detect the multiplex memory encoder (it receives a `multiplex_state` kwarg).
            if multiplex or ("multiplex_state" in kwargs):
                injected_idxs = list(range(n_obj))
                co = kwargs.get("conditioning_objects", None)
                if co is None:
                    kwargs["conditioning_objects"] = injected_idxs
                else:
                    kwargs["conditioning_objects"] = sorted(set(list(co)) | set(injected_idxs))

        # Force sigmoid path (not binarisation) in _encode_new_memory.
        kwargs["is_mask_from_pts"] = False
        return original_encode(*args, **kwargs)

    return hook


def install_injection(model, queue: List[torch.Tensor], *, multiplex: bool = False,
                      log_prefix: str = "", state: Optional[dict] = None):
    """Monkey-patch `model._encode_new_memory` with an injection hook."""
    original = model._encode_new_memory
    model._encode_new_memory = make_injection_hook(
        queue, original, multiplex=multiplex, log_prefix=log_prefix, state=state)
    return original


def restore_injection(model, original) -> None:
    """Undo `install_injection` (always call in a finally block)."""
    if original is not None:
        try:
            model._encode_new_memory = original
        except Exception as e:
            logger.error(f"DLMI restore failed: {e}")


# Alias matching the legacy util.dlmi_hooks API name.
create_injection_hook = make_injection_hook


@contextlib.contextmanager
def injection(model, queue: List[torch.Tensor], *, multiplex: bool = False,
              log_prefix: str = "", state: Optional[dict] = None):
    """Context manager: install the injection hook, yield, always restore."""
    original = install_injection(model, queue, multiplex=multiplex,
                                 log_prefix=log_prefix, state=state)
    try:
        yield
    finally:
        restore_injection(model, original)
