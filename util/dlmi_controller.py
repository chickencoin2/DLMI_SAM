"""
DLMI (Deep Layer Mask Injection) / Low-level API functions extracted from app.py.

Each method becomes a module-level function with ``app`` as its first parameter.
"""

import numpy as np
import torch
import logging
from tkinter import messagebox
from PIL import Image
import cv2

from .customutil import compute_dlmi_logits

logger = logging.getLogger("DLMI_SAM_LABELER.DLMIController")


def prepare_dlmi_mid_propagation(app):
    """Prepare DLMI injection to apply on the next frame when propagation resumes."""
    if not app.propagation_paused:
        messagebox.showwarning("Notice", "Propagation is not paused.\nPause propagation first, then inject.", parent=app.root)
        return

    if not app.low_level_api_enabled_var.get():
        messagebox.showwarning("Notice", "Low-level API (DLMI) is not enabled.\nEnable it in Advanced settings.", parent=app.root)
        return

    if not app.tracked_objects:
        messagebox.showwarning("Notice", "No objects to inject.\nModify or add objects first.", parent=app.root)
        return

    masks_to_inject = {}
    processed_sam_ids = set()

    for group_id, member_sam_ids in app.object_groups.items():
        merged_mask = app.get_group_merged_mask(group_id)
        if merged_mask is None:
            continue
        representative_id = min(member_sam_ids)
        first_obj_data = app.tracked_objects.get(representative_id, {})
        label = first_obj_data.get('custom_label', app.default_object_label_var.get())
        masks_to_inject[representative_id] = {
            'mask': merged_mask.astype(np.uint8),
            'label': label,
        }
        processed_sam_ids.update(member_sam_ids)

    for obj_id, obj_data in app.tracked_objects.items():
        if obj_id in processed_sam_ids:
            continue
        mask = obj_data.get('last_mask')
        if mask is None or not mask.any():
            continue
        masks_to_inject[obj_id] = {
            'mask': mask.astype(np.uint8),
            'label': obj_data.get('custom_label', app.default_object_label_var.get()),
        }

    if not masks_to_inject:
        messagebox.showwarning("Notice", "No valid masks to inject.", parent=app.root)
        return

    app.dlmi_pending_masks = masks_to_inject
    app.dlmi_pending_injection = True
    app.update_status(f"DLMI injection prepared ({len(masks_to_inject)} objects). Click 'Resume' to apply.")
    logger.info(f"DLMI mid-propagation injection prepared: {len(masks_to_inject)} objects at paused frame {app.propagation_current_frame_idx}")


def inject_low_level_mask_prompt(app):
    import torch.nn.functional as F

    # If paused during propagation, use mid-propagation injection
    if app.propagation_paused:
        prepare_dlmi_mid_propagation(app)
        return

    if not app.low_level_api_enabled_var.get():
        messagebox.showwarning("Notice", "Low-level API is not enabled.\nEnable it in advanced settings.", parent=app.root)
        return

    if not app.tracked_objects:
        messagebox.showwarning("Notice", "No masks to inject.\nDetect objects first.", parent=app.root)
        return

    if app.tracker_model is None:
        messagebox.showerror("Error", "SAM3 Tracker model not loaded.", parent=app.root)
        return

    if not hasattr(app.tracker_model, '_encode_new_memory'):
        logger.error("tracker_model does not have _encode_new_memory method.")
        messagebox.showerror("Error", "SAM3 model does not support Low-level API.\n_encode_new_memory method not found.", parent=app.root)
        return

    original_encode = None
    try:
        logger.info("Low-level API: Starting mask injection...")

        masks_to_inject = {}
        processed_sam_ids = set()

        for group_id, member_sam_ids in app.object_groups.items():
            merged_mask = app.get_group_merged_mask(group_id)
            if merged_mask is None:
                continue

            representative_id = min(member_sam_ids)
            first_obj_data = app.tracked_objects.get(representative_id, {})
            label = first_obj_data.get('custom_label', app.default_object_label_var.get())

            masks_to_inject[representative_id] = {
                'mask': merged_mask.astype(np.uint8),
                'label': label,
                'is_group': True,
                'group_id': group_id,
                'member_count': len(member_sam_ids)
            }
            processed_sam_ids.update(member_sam_ids)
            logger.info(f"Group {group_id}: {len(member_sam_ids)} objects -> merged to object ID {representative_id}")

        for obj_id, obj_data in app.tracked_objects.items():
            if obj_id in processed_sam_ids:
                continue
            mask = obj_data.get('last_mask')
            if mask is None or not mask.any():
                continue
            masks_to_inject[obj_id] = {
                'mask': mask.astype(np.uint8),
                'label': obj_data.get('custom_label', app.default_object_label_var.get()),
                'is_group': False
            }

        if not masks_to_inject:
            messagebox.showwarning("Notice", "No valid masks to inject.", parent=app.root)
            return

        logger.info("Initializing SAM3 session...")
        app.inference_session = None
        if not app._init_inference_session():
            messagebox.showerror("Error", "SAM3 session initialization failed.", parent=app.root)
            return

        old_tracked_objects = app.tracked_objects.copy()
        app.tracked_objects.clear()
        app.object_groups.clear()
        app.sam_id_to_group.clear()
        app.next_group_id = 1
        app.suppressed_sam_ids.clear()
        logger.info("Low data injection: suppressed_sam_ids cleared")
        if hasattr(app, 'object_prompt_history'):
            app.object_prompt_history.clear()
            logger.info("Low data injection: object_prompt_history cleared")

        frame_idx = 0
        dtype = app.model_dtype
        injected_count = 0

        if app.current_cv_frame is None:
            messagebox.showerror("Error", "No current frame.", parent=app.root)
            return

        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        inputs = app.tracker_processor(images=frame_pil, device=app.device, return_tensors="pt")

        frame_tensor = inputs.pixel_values[0]
        if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
            frame_tensor = frame_tensor.to(dtype=torch.float32)

        obj_ids_list = list(masks_to_inject.keys())
        input_masks_list = [masks_to_inject[oid]['mask'] for oid in obj_ids_list]

        logger.info(f"Passing {len(obj_ids_list)} object prompts via input_masks...")
        app.tracker_processor.add_inputs_to_inference_session(
            inference_session=app.inference_session,
            frame_idx=frame_idx,
            obj_ids=list(obj_ids_list),
            input_masks=input_masks_list,
            original_size=inputs.original_sizes[0],
        )

        injection_queue = []
        dlmi_mode = app.dlmi_boundary_mode_var.get()
        dlmi_intensity = app.dlmi_alpha_var.get()
        dlmi_falloff = app.dlmi_gradient_falloff_var.get()

        logger.info(f"Preparing DLMI logits: {len(obj_ids_list)} objects, "
                    f"mode={dlmi_mode}, intensity={dlmi_intensity}, falloff={dlmi_falloff}")

        for obj_id in obj_ids_list:
            mask = masks_to_inject[obj_id]['mask']
            logits_np = compute_dlmi_logits(mask, mode=dlmi_mode,
                                            intensity=dlmi_intensity, falloff=dlmi_falloff)
            logit_tensor = torch.from_numpy(logits_np).to(app.device)
            injection_queue.append(logit_tensor)
            logger.debug(f"  Object {obj_id}: logit range=[{logits_np.min():.2f}, {logits_np.max():.2f}]")

        logger.info(f"Injection queue ready: {len(injection_queue)} logit maps")

        original_encode = app.tracker_model._encode_new_memory
        injection_state = {"idx": 0}

        def injection_hook(**kwargs):
            if 'pred_masks_high_res' not in kwargs:
                return original_encode(**kwargs)

            input_tensor = kwargs['pred_masks_high_res']
            logger.debug(f"  Hook called! Input Shape: {input_tensor.shape}")

            if input_tensor.ndim == 4:
                curr_num_obj = input_tensor.shape[0]
                curr_h, curr_w = input_tensor.shape[-2:]
            elif input_tensor.ndim == 5:
                curr_num_obj = input_tensor.shape[1]
                curr_h, curr_w = input_tensor.shape[-2:]
            else:
                logger.warning(f"  Unexpected tensor dimension: {input_tensor.ndim}")
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
                            logit_t.float(), size=(curr_h, curr_w), mode='bilinear',
                            align_corners=False
                        )

                    target_tensors.append(logit_t.to(input_tensor.dtype))
                else:
                    logger.warning(f"  Injection queue exhausted (idx={q_idx}). Keeping original.")

            if target_tensors and len(target_tensors) == curr_num_obj:
                injected_batch = torch.cat(target_tensors, dim=0)

                if input_tensor.ndim == 4:
                    if injected_batch.shape == input_tensor.shape:
                        kwargs['pred_masks_high_res'] = injected_batch
                        logger.info(f"  >> 4D tensor replaced ({len(target_tensors)} items)")
                    else:
                        logger.error(f"  4D dimension mismatch: {injected_batch.shape} vs {input_tensor.shape}")
                elif input_tensor.ndim == 5:
                    injected_batch = injected_batch.unsqueeze(0)
                    if injected_batch.shape == input_tensor.shape:
                        kwargs['pred_masks_high_res'] = injected_batch
                        logger.info(f"  >> 5D tensor replaced ({len(target_tensors)} items)")
                    else:
                        try:
                            kwargs['pred_masks_high_res'] = injected_batch.view_as(input_tensor)
                            logger.info(f"  >> 5D view transform complete")
                        except Exception as e:
                            logger.error(f"  5D dimension matching failed: {e}")

                injection_state["idx"] += curr_num_obj

            # Force sigmoid path instead of binarization in _encode_new_memory
            kwargs['is_mask_from_pts'] = False
            return original_encode(**kwargs)

        app.tracker_model._encode_new_memory = injection_hook

        logger.info("Forward pass starting (Hook replaces masks)")
        with torch.no_grad():
            try:
                outputs = app.tracker_model(
                    inference_session=app.inference_session,
                    frame=frame_tensor,
                )
                logger.info(f"Forward pass complete. Injected masks: {injection_state['idx']}")
            except Exception as e:
                logger.exception(f"Forward pass error: {e}")
                app.tracker_model._encode_new_memory = original_encode
                messagebox.showerror("Error", f"Forward pass failed: {e}", parent=app.root)
                return

        app.tracker_model._encode_new_memory = original_encode

        # Install persistent DLMI hooks (Preserve + Boost) after forward pass
        _install_dlmi_persistent_hooks(app)

        if injection_state["idx"] == 0:
            messagebox.showwarning("Notice", "No masks injected. Hook was not called.", parent=app.root)
            return

        injected_count = injection_state["idx"]

        for obj_id, obj_info in masks_to_inject.items():
            mask = obj_info['mask']
            label = obj_info['label']
            app.tracked_objects[obj_id] = {
                'custom_label': label,
                'last_mask': mask.astype(bool),
                'is_injected_group': obj_info.get('is_group', False)
            }

        if injected_count > 0:
            app.low_level_mask_injected = True
            app.is_tracking_ever_started = True
            app.update_status(f"Low-level API: {injected_count} objects injected to new SAM3 session. Start propagation.")
            logger.info(f"Low-level API: {injected_count} objects injected (groups merged into single objects)")

            if app.current_cv_frame is not None:
                app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
        else:
            messagebox.showwarning("Notice", "No masks injected.", parent=app.root)

        if app.view and hasattr(app.view, 'update_low_data_inject_button_state'):
            app.view.update_low_data_inject_button_state()

    except Exception as e:
        logger.exception(f"Low-level API mask injection failed: {e}")
        messagebox.showerror("Error", f"Error during mask injection:\n{e}", parent=app.root)
        if 'original_encode' in dir() and original_encode is not None:
            try:
                app.tracker_model._encode_new_memory = original_encode
            except:
                pass


def _install_dlmi_persistent_hooks(app):
    """Install persistent DLMI hooks for Preserve Memory and Boost Conditioning.
    These hooks persist across propagation frames (not cleaned up per-frame)."""
    # --- Preserve: set max_cond_frame_num = -1 to keep ALL conditioning frames ---
    if app.dlmi_preserve_memory_var.get():
        if not hasattr(app, '_dlmi_original_max_cond_frame_num'):
            app._dlmi_original_max_cond_frame_num = app.tracker_model.config.max_cond_frame_num
        app.tracker_model.config.max_cond_frame_num = -1
        logger.info(f"DLMI Preserve: max_cond_frame_num set to -1 "
                    f"(was {app._dlmi_original_max_cond_frame_num})")

    # --- Boost: hook _gather_memory_frame_outputs to triple conditioning entries ---
    if app.dlmi_boost_cond_var.get():
        if not hasattr(app, '_dlmi_original_gather'):
            original_gather = app.tracker_model._gather_memory_frame_outputs
            app._dlmi_original_gather = original_gather
            app_ref = app

            def boosted_gather(inference_session, obj_idx, frame_idx,
                               track_in_reverse_time=False):
                result = original_gather(
                    inference_session, obj_idx, frame_idx,
                    track_in_reverse_time=track_in_reverse_time
                )
                # Check at runtime so user can toggle on/off
                if not app_ref.dlmi_boost_cond_var.get():
                    return result

                # Conditioning entries have offset=0; duplicate them 3x
                cond_entries = [(off, data) for off, data in result if off == 0 and data is not None]
                non_cond_entries = [(off, data) for off, data in result if off != 0]
                # Triple the conditioning entries
                boosted = cond_entries * 3 + non_cond_entries
                logger.debug(f"DLMI Boost: {len(cond_entries)} cond entries -> "
                             f"{len(cond_entries)*3} (total {len(boosted)} memories)")
                return boosted

            app.tracker_model._gather_memory_frame_outputs = boosted_gather
            logger.info("DLMI Boost: _gather_memory_frame_outputs hooked (3x conditioning)")


def _remove_dlmi_persistent_hooks(app):
    """Remove persistent DLMI hooks (called on session reset/cleanup)."""
    if not hasattr(app, 'tracker_model') or app.tracker_model is None:
        return

    # Restore max_cond_frame_num
    if hasattr(app, '_dlmi_original_max_cond_frame_num'):
        app.tracker_model.config.max_cond_frame_num = app._dlmi_original_max_cond_frame_num
        logger.info(f"DLMI Preserve: max_cond_frame_num restored to "
                    f"{app._dlmi_original_max_cond_frame_num}")
        del app._dlmi_original_max_cond_frame_num

    # Restore _gather_memory_frame_outputs
    if hasattr(app, '_dlmi_original_gather'):
        app.tracker_model._gather_memory_frame_outputs = app._dlmi_original_gather
        logger.info("DLMI Boost: _gather_memory_frame_outputs restored to original")
        del app._dlmi_original_gather
