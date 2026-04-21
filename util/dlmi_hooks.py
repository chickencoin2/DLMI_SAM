"""Shared DLMI (Direct Latent Memory Injection) helpers.

Responsibility split:
  - `compute_dlmi_logits` lives in `customutil.py` and converts a binary
    mask into a float logit map (Fixed / Gradient modes).
  - This module provides the small adapter layer that wraps those logit
    tensors into the `_encode_new_memory` hook closure used in THREE
    places (app.inject_low_level_mask_prompt, app._dlmi_mini_propagate_n_to_n1,
    propagation_controller._setup_dlmi_injection_hook). The hook logic
    was ~70% duplicated across those sites; centralising it here.

Two public helpers:
  - `create_injection_hook(queue, original_encode)` - returns a closure
    suitable for replacing `tracker_model._encode_new_memory`. The closure
    consumes `queue` (a list of per-object logit torch tensors) in order,
    then calls back into `original_encode`.
  - `dlmi_hook(tracker_model, queue)` - context manager that installs
    the hook on `tracker_model._encode_new_memory`, yields, and
    unconditionally restores the original method in `finally`.

The per-frame variant in `propagation_controller._setup_dlmi_injection_hook`
keeps its custom install/uninstall flow (app.dlmi_hook_active flag, persistent
Preserve/Boost hooks) but uses `create_injection_hook` to avoid duplicating
the kwargs-mutation closure.
"""
import contextlib
import logging
import torch

import numpy as np

from .customutil import compute_dlmi_logits

logger = logging.getLogger("DLMI_SAM_LABELER.DLMIHooks")


def build_injection_queue(obj_ids, masks_by_oid, intensity, mode, falloff, device):
    """Pre-compute a list of logit torch tensors (on `device`), one per
    `obj_ids` entry, from the given {oid: mask_ndarray} dict. Order of the
    returned list matches the order of `obj_ids`.

    - obj_ids: ordered list of object ids.
    - masks_by_oid: dict like {oid: np.ndarray[bool H,W]}.
    - intensity: DLMI alpha (typical default 10.0).
    - mode: 'Fixed' | 'Gradient'.
    - falloff: pixel falloff for Gradient mode.
    - device: torch.device destination.
    """
    queue = []
    for oid in obj_ids:
        mask = masks_by_oid[oid]
        logits_np = compute_dlmi_logits(mask, mode=mode, intensity=intensity, falloff=falloff)
        queue.append(torch.from_numpy(logits_np).to(device))
    return queue


def create_injection_hook(injection_queue, original_encode, log_prefix="", state=None):
    """Return a closure that replaces `tracker_model._encode_new_memory`.

    The closure consumes one tensor per current-frame object from
    `injection_queue` (in index order), interpolates it to the required
    spatial size, concatenates, and substitutes into
    `kwargs['pred_masks_high_res']`. Also forces
    `kwargs['is_mask_from_pts'] = False` so the downstream encoder uses the
    sigmoid path (not binarisation). Finally, calls `original_encode(**kwargs)`.

    A module-level `logger.warning` is emitted if the queue is exhausted
    before the batch is filled.

    Parameters
    ----------
    injection_queue : list[torch.Tensor]
        Per-object logit tensors; consumed sequentially.
    original_encode : callable
        The replaced method, called after kwargs mutation.
    log_prefix : str
        Optional prefix used in debug logs (e.g. "mini-prop" or "low-level").
    state : dict | None
        Optional externally-provided dict. The closure writes progress into
        `state['idx']` so callers can inspect how many queue items were
        consumed after the forward pass. If omitted, internal state is used.
    """
    injection_state = state if state is not None else {}
    injection_state.setdefault("idx", 0)

    def injection_hook(**kwargs):
        if 'pred_masks_high_res' not in kwargs:
            return original_encode(**kwargs)

        input_tensor = kwargs['pred_masks_high_res']

        if input_tensor.ndim == 4:
            curr_num_obj = input_tensor.shape[0]
            curr_h, curr_w = input_tensor.shape[-2:]
        elif input_tensor.ndim == 5:
            curr_num_obj = input_tensor.shape[1]
            curr_h, curr_w = input_tensor.shape[-2:]
        else:
            return original_encode(**kwargs)

        target_tensors = []
        start_idx = injection_state["idx"]

        for i in range(curr_num_obj):
            q_idx = start_idx + i
            if q_idx < len(injection_queue):
                logit_t = injection_queue[q_idx]
                if logit_t.dim() == 2:
                    logit_t = logit_t.view(1, 1, logit_t.shape[0], logit_t.shape[1])
                elif logit_t.dim() == 3:
                    logit_t = logit_t.unsqueeze(0)
                if (logit_t.shape[-2], logit_t.shape[-1]) != (curr_h, curr_w):
                    logit_t = torch.nn.functional.interpolate(
                        logit_t.float(), size=(curr_h, curr_w),
                        mode='bilinear', align_corners=False
                    )
                target_tensors.append(logit_t.to(input_tensor.dtype))
            else:
                logger.warning(f"  DLMI {log_prefix}: injection queue exhausted (idx={q_idx})")

        if target_tensors and len(target_tensors) == curr_num_obj:
            injected_batch = torch.cat(target_tensors, dim=0)

            if input_tensor.ndim == 4:
                if injected_batch.shape == input_tensor.shape:
                    kwargs['pred_masks_high_res'] = injected_batch
                else:
                    logger.error(
                        f"  DLMI {log_prefix}: 4D shape mismatch "
                        f"{injected_batch.shape} vs {input_tensor.shape}"
                    )
            elif input_tensor.ndim == 5:
                injected_batch = injected_batch.unsqueeze(0)
                if injected_batch.shape == input_tensor.shape:
                    kwargs['pred_masks_high_res'] = injected_batch
                else:
                    try:
                        kwargs['pred_masks_high_res'] = injected_batch.view_as(input_tensor)
                    except Exception as e:
                        logger.error(f"  DLMI {log_prefix}: view_as failed: {e}")

            injection_state["idx"] += curr_num_obj

        # Force sigmoid path instead of binarisation in _encode_new_memory
        kwargs['is_mask_from_pts'] = False
        return original_encode(**kwargs)

    return injection_hook


@contextlib.contextmanager
def dlmi_hook(tracker_model, injection_queue, log_prefix=""):
    """Context manager: install a DLMI injection hook on
    `tracker_model._encode_new_memory`, yield, and ALWAYS restore the
    original method in the finally block (even if the body raises).

    Usage:
        with dlmi_hooks.dlmi_hook(self.tracker_model, queue, "low-level"):
            outputs = self.tracker_model(inference_session=..., frame=...)
    """
    original_encode = tracker_model._encode_new_memory
    tracker_model._encode_new_memory = create_injection_hook(
        injection_queue, original_encode, log_prefix=log_prefix
    )
    try:
        yield
    finally:
        tracker_model._encode_new_memory = original_encode
