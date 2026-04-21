import cv2
import numpy as np
import torch
import logging
import tkinter.messagebox as messagebox
from PIL import Image

from .customutil import process_sam_mask, get_bbox_from_mask
from . import dlmi_hooks

logger = logging.getLogger("DLMI_SAM_LABELER.PropagationController")


def _wait_if_paused(app):
    """Block thread while paused. Returns False if stop was requested during wait.
    Sets app._resume_needs_cap_seek = True if pause was actually engaged."""
    was_paused = not app.propagation_pause_event.is_set()
    app.propagation_pause_event.wait()
    if was_paused:
        app._resume_needs_cap_seek = True
    return not app.propagation_stop_requested


def _setup_dlmi_injection_hook(app, frame_bgr, frame_idx):
    """Set up DLMI injection before a frame's forward pass.

    Registers ALL injection objects with the inference session via a
    SINGLE call to add_inputs_to_inference_session (both new and existing).
    This is critical because each call to add_inputs_to_inference_session
    REPLACES obj_with_new_inputs (not extends), so separate calls would
    cause the first batch to lose their has_new_inputs flag.

    Installs _encode_new_memory hook to:
    1. Replace pred_masks_high_res with DLMI logits (Fixed or Gradient)
    2. Set is_mask_from_pts=False to use sigmoid path instead of binarization
    """
    masks_to_inject = getattr(app, 'dlmi_pending_masks', {})
    if not masks_to_inject:
        app.dlmi_pending_injection = False
        return

    if app.inference_session is None:
        logger.error("DLMI mid-propagation: No inference session available")
        app.dlmi_pending_injection = False
        return

    obj_ids = list(masks_to_inject.keys())

    # Identify new vs existing objects for logging
    existing_obj_ids = set()
    if hasattr(app.inference_session, 'obj_ids'):
        existing_obj_ids = set(app.inference_session.obj_ids)

    new_obj_ids = [oid for oid in obj_ids if oid not in existing_obj_ids]
    existing_injection_ids = [oid for oid in obj_ids if oid in existing_obj_ids]

    # Prepare frame data for add_inputs_to_inference_session
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    inputs = app.tracker_processor(images=frame_pil, device=app.device, return_tensors="pt")

    # SINGLE call with ALL objects (new + existing) to avoid obj_with_new_inputs overwrite bug
    all_masks = [masks_to_inject[oid]['mask'] for oid in obj_ids]

    # Calculate the model's NEXT frame_idx
    model_frame_idx = len(app.inference_session.processed_frames) if app.inference_session.processed_frames else 0

    app.tracker_processor.add_inputs_to_inference_session(
        inference_session=app.inference_session,
        frame_idx=model_frame_idx,
        obj_ids=obj_ids,
        input_masks=all_masks,
        original_size=inputs.original_sizes[0],
    )

    # Register new objects in app.tracked_objects
    for oid in new_obj_ids:
        if oid not in app.tracked_objects:
            app.tracked_objects[oid] = {
                'custom_label': masks_to_inject[oid].get('label', app.default_object_label_var.get()),
                'last_mask': masks_to_inject[oid]['mask'].astype(bool),
            }

    logger.info(f"DLMI mid-propagation: {len(obj_ids)} objects registered in single call "
                f"({len(new_obj_ids)} new, {len(existing_injection_ids)} existing) at frame {frame_idx}")

    # Pre-compute DLMI logits and install the injection hook via shared helper.
    dlmi_mode = app.dlmi_boundary_mode_var.get()
    dlmi_intensity = app.dlmi_alpha_var.get()
    dlmi_falloff = app.dlmi_gradient_falloff_var.get()

    injection_queue = dlmi_hooks.build_injection_queue(
        obj_ids=obj_ids,
        masks_by_oid={oid: masks_to_inject[oid]['mask'] for oid in obj_ids},
        intensity=dlmi_intensity,
        mode=dlmi_mode,
        falloff=dlmi_falloff,
        device=app.device,
    )

    logger.info(f"DLMI mid-propagation: logit maps computed (mode={dlmi_mode}, "
                f"intensity={dlmi_intensity}, falloff={dlmi_falloff})")

    # Manual install (not context-manager) because restore happens per-frame via
    # _cleanup_dlmi_hook in the propagation loop, not inside this function's scope.
    original_encode = app.tracker_model._encode_new_memory
    app._dlmi_original_encode = original_encode
    app.tracker_model._encode_new_memory = dlmi_hooks.create_injection_hook(
        injection_queue, original_encode, log_prefix="mid-prop"
    )
    app.dlmi_hook_active = True
    logger.info(f"DLMI mid-propagation: _encode_new_memory hook installed for frame {frame_idx}")

    # Install persistent hooks (Preserve + Boost) via app method
    if hasattr(app, '_install_dlmi_persistent_hooks'):
        app._install_dlmi_persistent_hooks()


def _cleanup_dlmi_hook(app):
    """Clean up DLMI state after frame processing. Restores original _encode_new_memory."""
    if getattr(app, 'dlmi_hook_active', False):
        if hasattr(app, '_dlmi_original_encode') and app._dlmi_original_encode is not None:
            app.tracker_model._encode_new_memory = app._dlmi_original_encode
            app._dlmi_original_encode = None
            logger.info("DLMI mid-propagation: _encode_new_memory restored to original")
        app.dlmi_hook_active = False
        app.dlmi_pending_injection = False
        app.dlmi_pending_masks = {}
        logger.info("DLMI mid-propagation: state cleaned up after frame processing")


def propagate_pvs_mode(app, start_frame, actual_total_frames):
    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for relative_idx in range(actual_total_frames):
        frame_idx = relative_idx
        if app.propagation_stop_requested:
            logger.info(f"Propagation stop requested. Processed up to frame {frame_idx}.")
            break

        # Pause check
        app.propagation_current_frame_idx = frame_idx
        if not _wait_if_paused(app):
            logger.info(f"Propagation stopped during pause at frame {frame_idx}.")
            break

        # Restore video position after pause (user may have used review slider)
        if getattr(app, '_resume_needs_cap_seek', False):
            app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + relative_idx)
            app._resume_needs_cap_seek = False

        ret, frame_bgr = app.cap.read()
        if not ret:
            logger.info(f"Video end reached at frame {frame_idx}.")
            break

        # DLMI injection check (PVS mode only)
        if getattr(app, 'dlmi_pending_injection', False):
            _setup_dlmi_injection_hook(app, frame_bgr, frame_idx)

        try:
            sam_masks_display, sam_masks_labeling = app._perform_sam_tracking_for_frame(frame_bgr, frame_idx)
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"GPU memory insufficient (OOM)!\nPropagation stopped at frame {frame_idx}.\n\n{str(e)[:200]}"
            logger.error(f"PVS propagation OOM: {e}")
            app.propagation_stop_requested = True
            _cleanup_dlmi_hook(app)
            app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error - GPU Memory Insufficient", msg, parent=app.root))
            break
        except Exception as e:
            error_msg = f"Error during SAM3 tracking!\nPropagation stopped at frame {frame_idx}.\n\nError: {str(e)[:200]}"
            logger.error(f"PVS propagation error: {e}")
            app.propagation_stop_requested = True
            _cleanup_dlmi_hook(app)
            app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error", msg, parent=app.root))
            break

        # Cleanup DLMI hook after forward pass
        _cleanup_dlmi_hook(app)

        if hasattr(app, '_tracking_fatal_error') and app._tracking_fatal_error:
            error_msg = app._tracking_fatal_error
            app._tracking_fatal_error = None
            logger.error(f"PVS propagation tracking failure detected: {error_msg}")
            app.propagation_stop_requested = True
            app.root.after(0, lambda msg=error_msg: messagebox.showerror(
                "Tracking Failure",
                f"SAM3 tracking failed, stopping propagation.\n\n{msg}\n\nProcessed up to frame {frame_idx}.",
                parent=app.root
            ))
            break

        app.propagated_results[frame_idx] = {
            'frame': frame_bgr.copy(),
            'masks': sam_masks_labeling.copy()
        }

        progress = int((relative_idx + 1) / actual_total_frames * 100)
        app.propagation_progress = progress
        actual_frame_num = start_frame + relative_idx
        app.root.after(0, app.view.update_propagate_progress, progress, f"Processing: {relative_idx + 1}/{actual_total_frames} (frame {actual_frame_num})")

        app.current_cv_frame = frame_bgr.copy()
        app.root.after(0, app._display_cv_frame_on_view, frame_bgr, sam_masks_display)
        app.root.after(0, app.view.review_frame_slider.set, frame_idx)
        app.root.after(0, app.view.update_review_frame_info, frame_idx, actual_total_frames - 1)


def propagate_pcs_mode(app, start_frame, actual_total_frames):
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model or processor not available.")
        return

    if app.pcs_streaming_session is None:
        logger.error("PCS streaming session not available. Please perform detection first.")
        return

    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        with torch.inference_mode():
            for relative_idx in range(actual_total_frames):
                frame_idx = relative_idx
                actual_video_frame = start_frame + relative_idx

                if app.propagation_stop_requested:
                    logger.info(f"PCS propagation stop requested. Processed up to frame {frame_idx}.")
                    break

                # Pause check
                app.propagation_current_frame_idx = frame_idx
                if not _wait_if_paused(app):
                    logger.info(f"PCS propagation stopped during pause at frame {frame_idx}.")
                    break

                if getattr(app, '_resume_needs_cap_seek', False):
                    app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + relative_idx)
                    app._resume_needs_cap_seek = False

                ret, frame_bgr = app.cap.read()
                if not ret:
                    logger.info(f"PCS propagation: Video end reached at frame {actual_video_frame}.")
                    break

                try:
                    sam_masks_display, sam_masks_labeling = app._perform_pcs_streaming_tracking(
                        frame_bgr, frame_idx
                    )

                    if hasattr(app, '_tracking_fatal_error') and app._tracking_fatal_error:
                        error_msg = app._tracking_fatal_error
                        app._tracking_fatal_error = None
                        logger.error(f"PCS propagation fatal error: {error_msg}")
                        app.propagation_stop_requested = True
                        app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error", msg, parent=app.root))
                        break

                except torch.cuda.OutOfMemoryError as e:
                    error_msg = f"GPU memory insufficient (OOM)!\nPropagation stopped at frame {actual_video_frame}.\n\n{str(e)[:200]}"
                    logger.error(f"PCS propagation OOM: {e}")
                    app.propagation_stop_requested = True
                    app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error - GPU Memory Insufficient", msg, parent=app.root))
                    break
                except Exception as e:
                    error_msg = f"Error during PCS tracking!\nPropagation stopped at frame {actual_video_frame}.\n\nError: {str(e)[:200]}"
                    logger.error(f"PCS propagation error: {e}")
                    app.propagation_stop_requested = True
                    app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error", msg, parent=app.root))
                    break

                app.propagated_results[frame_idx] = {
                    'frame': frame_bgr.copy(),
                    'masks': sam_masks_labeling.copy()
                }

                progress = int((relative_idx + 1) / actual_total_frames * 100)
                app.propagation_progress = progress
                app.root.after(0, app.view.update_propagate_progress, progress, f"PCS processing: {relative_idx + 1}/{actual_total_frames} (frame {actual_video_frame})")

                app.current_cv_frame = frame_bgr.copy()
                app.root.after(0, app._display_cv_frame_on_view, frame_bgr, sam_masks_display)
                app.root.after(0, app.view.review_frame_slider.set, frame_idx)
                app.root.after(0, app.view.update_review_frame_info, frame_idx, actual_total_frames - 1)

    except Exception as e:
        logger.exception(f"Error during PCS propagation: {e}")


def propagate_pcs_image_mode(app, start_frame, actual_total_frames):
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model or processor not available.")
        return

    text_prompt = app.pcs_text_prompt_var.get().strip()
    if not text_prompt:
        logger.error("PCS(per-image) mode: No text prompt provided.")
        return

    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        with torch.inference_mode():
            for relative_idx in range(actual_total_frames):
                frame_idx = relative_idx
                actual_video_frame = start_frame + relative_idx

                if app.propagation_stop_requested:
                    logger.info(f"PCS(per-image) propagation stop requested. Processed up to frame {frame_idx}.")
                    break

                # Pause check
                app.propagation_current_frame_idx = frame_idx
                if not _wait_if_paused(app):
                    logger.info(f"PCS(per-image) propagation stopped during pause at frame {frame_idx}.")
                    break

                if getattr(app, '_resume_needs_cap_seek', False):
                    app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + relative_idx)
                    app._resume_needs_cap_seek = False

                ret, frame_bgr = app.cap.read()
                if not ret:
                    logger.info(f"PCS(per-image) propagation: Video end reached at frame {actual_video_frame}.")
                    break

                try:
                    sam_masks_display, sam_masks_labeling = app._perform_pcs_single_image_detection(
                        frame_bgr, text_prompt, frame_idx
                    )
                except torch.cuda.OutOfMemoryError as e:
                    error_msg = f"GPU memory insufficient (OOM)!\nStopped at frame {actual_video_frame}.\n\n{str(e)[:200]}"
                    logger.error(f"PCS(per-image) propagation OOM: {e}")
                    app.propagation_stop_requested = True
                    app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error - GPU Memory Insufficient", msg, parent=app.root))
                    break
                except Exception as e:
                    error_msg = f"Error during PCS(per-image) processing!\nStopped at frame {actual_video_frame}.\n\nError: {str(e)[:200]}"
                    logger.error(f"PCS(per-image) propagation error: {e}")
                    app.propagation_stop_requested = True
                    app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error", msg, parent=app.root))
                    break

                if sam_masks_labeling:
                    app.propagated_results[frame_idx] = {
                        'frame': frame_bgr.copy(),
                        'masks': sam_masks_labeling.copy()
                    }

                progress = int((relative_idx + 1) / actual_total_frames * 100)
                app.propagation_progress = progress
                app.root.after(0, app.view.update_propagate_progress, progress,
                               f"PCS(per-image) processing: {relative_idx + 1}/{actual_total_frames}")

                app.current_cv_frame = frame_bgr.copy()
                app.root.after(0, app._display_cv_frame_on_view, frame_bgr, sam_masks_display)
                app.root.after(0, app.view.review_frame_slider.set, frame_idx)
                app.root.after(0, app.view.update_review_frame_info, frame_idx, actual_total_frames - 1)

    except Exception as e:
        logger.exception(f"Error during PCS(per-image) propagation: {e}")


def propagate_pvs_chunk_mode(app, start_frame, actual_total_frames):
    app.chunk_processing = True
    chunk_start = start_frame
    total_processed = 0

    while chunk_start < start_frame + actual_total_frames:
        if app.propagation_stop_requested:
            logger.info("PVS(chunk) mode: Propagation stopped by user")
            break

        if not _wait_if_paused(app):
            logger.info("PVS(chunk) mode: Propagation stopped during pause")
            break

        remaining_frames = (start_frame + actual_total_frames) - chunk_start

        try:
            chunk_processed = _propagate_pvs_chunk_segment(
                app, chunk_start, remaining_frames, total_processed, actual_total_frames
            )

            if chunk_processed > 0:
                total_processed += chunk_processed
                chunk_start += chunk_processed

                if chunk_start >= start_frame + actual_total_frames:
                    logger.info("PVS(chunk) mode: Full propagation complete")
                    break
            else:
                logger.warning("PVS(chunk) mode: No frames processed, stopping")
                break

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"PVS(chunk) OOM at frame {chunk_start}: {e}")
            torch.cuda.empty_cache()

            if app.propagated_results:
                last_frame_idx = max(app.propagated_results.keys())
                logger.info(f"PVS(chunk) mode: OOM occurred, attempting auto-restart from frame {last_frame_idx}")

                if _auto_reprompt_from_mask(app, last_frame_idx):
                    chunk_start = last_frame_idx + 1
                    total_processed = last_frame_idx + 1
                    continue
                else:
                    logger.error("PVS(chunk) mode: Auto-reprompt failed")
                    break
            else:
                logger.error("PVS(chunk) mode: No saved results, stopping due to OOM")
                app.root.after(0, lambda: messagebox.showerror(
                    "Propagation Error",
                    "Propagation stopped due to GPU memory insufficient.\nFailed from first frame, cannot recover.",
                    parent=app.root
                ))
                break

        except Exception as e:
            logger.exception(f"PVS(chunk) mode error: {e}")
            app.root.after(0, lambda msg=str(e)[:200]: messagebox.showerror(
                "Propagation Error", f"PVS(chunk) mode error:\n{msg}", parent=app.root
            ))
            break

    app.chunk_processing = False
    logger.info(f"PVS(chunk) mode complete: {total_processed} frames processed")


def _propagate_pvs_chunk_segment(app, chunk_start, remaining_frames, processed_so_far, total_frames):
    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)

    processed_in_chunk = 0

    for i in range(remaining_frames):
        if app.propagation_stop_requested:
            break

        # Pause check
        frame_idx = processed_so_far + i
        app.propagation_current_frame_idx = frame_idx
        if not _wait_if_paused(app):
            break

        if getattr(app, '_resume_needs_cap_seek', False):
            app.cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start + i)
            app._resume_needs_cap_seek = False

        ret, frame_bgr = app.cap.read()
        if not ret:
            break

        # DLMI injection check (PVS chunk mode)
        if getattr(app, 'dlmi_pending_injection', False):
            _setup_dlmi_injection_hook(app, frame_bgr, frame_idx)

        sam_masks_display, sam_masks_labeling = app._perform_sam_tracking_for_frame(
            frame_bgr, frame_idx
        )

        # Cleanup DLMI hook
        _cleanup_dlmi_hook(app)

        if hasattr(app, '_tracking_fatal_error') and app._tracking_fatal_error:
            error_msg = app._tracking_fatal_error
            app._tracking_fatal_error = None
            logger.warning(f"PVS(chunk) tracking failure at frame {frame_idx}: {error_msg}")
            raise Exception(f"Tracking failure: {error_msg}")

        app.propagated_results[frame_idx] = {
            'frame': frame_bgr.copy(),
            'masks': sam_masks_labeling.copy()
        }

        processed_in_chunk += 1

        progress = int((frame_idx + 1) / total_frames * 100)
        app.root.after(0, app.view.update_propagate_progress, progress,
                       f"PVS(chunk) processing: {frame_idx + 1}/{total_frames}")

        app.current_cv_frame = frame_bgr.copy()
        app.root.after(0, app._display_cv_frame_on_view, frame_bgr, sam_masks_display)
        app.root.after(0, app.view.review_frame_slider.set, frame_idx)

    return processed_in_chunk


def _auto_reprompt_from_mask(app, last_frame_idx):
    if last_frame_idx not in app.propagated_results:
        return False

    result = app.propagated_results[last_frame_idx]
    masks = result['masks']
    frame_bgr = result['frame']

    if not masks:
        return False

    try:
        app.inference_session = None
        app.tracked_objects.clear()
        app.is_tracking_ever_started = False

        if not app._init_inference_session():
            return False

        threshold = app.chunk_error_threshold_var.get()

        for obj_id, obj_data in masks.items():
            mask = obj_data.get('last_mask')
            if mask is None or not mask.any():
                continue

            bbox = get_bbox_from_mask(mask)
            if bbox is None:
                continue

            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

            app._handle_sam_prompt_wrapper(
                'point',
                np.array([[cx, cy]]),
                1,
                obj_id,
                None,
                obj_data.get('custom_label', app.default_object_label_var.get())
            )

        logger.info(f"Auto-reprompt complete: {len(masks)} objects")
        return True

    except Exception as e:
        logger.exception(f"Auto-reprompt error: {e}")
        return False


def propagate_sam2_mode(app, start_frame, actual_total_frames):
    if app.sam2_model is None or app.sam2_processor is None:
        logger.error("SAM2 model or processor not available.")
        return

    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_masks = {}
    if app.sam2_masks:
        prev_masks = {obj_id: mask.copy() for obj_id, mask in app.sam2_masks.items()}
    elif app.tracked_objects:
        for obj_id, obj_data in app.tracked_objects.items():
            if 'last_mask' in obj_data and obj_data['last_mask'] is not None:
                prev_masks[obj_id] = obj_data['last_mask'].copy()

    if not prev_masks:
        logger.error("SAM2 tracking: No initial masks available.")
        return

    try:
        with torch.inference_mode():
            for relative_idx in range(actual_total_frames):
                frame_idx = relative_idx
                actual_video_frame = start_frame + relative_idx

                if app.propagation_stop_requested:
                    logger.info(f"SAM2 propagation stop requested. Processed up to frame {frame_idx}.")
                    break

                # Pause check
                app.propagation_current_frame_idx = frame_idx
                if not _wait_if_paused(app):
                    logger.info(f"SAM2 propagation stopped during pause at frame {frame_idx}.")
                    break

                if getattr(app, '_resume_needs_cap_seek', False):
                    app.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + relative_idx)
                    app._resume_needs_cap_seek = False

                ret, frame_bgr = app.cap.read()
                if not ret:
                    logger.info(f"SAM2 propagation: Video end reached at frame {actual_video_frame}.")
                    break

                try:
                    sam_masks_display = {}
                    sam_masks_labeling = {}
                    frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                    for obj_id, prev_mask in prev_masks.items():
                        y_coords, x_coords = np.where(prev_mask > 0)
                        if len(x_coords) == 0:
                            continue

                        cx = int(np.mean(x_coords))
                        cy = int(np.mean(y_coords))

                        input_points = [[[[cx, cy]]]]
                        input_labels = [[[1]]]

                        inputs = app.sam2_processor(
                            images=frame_pil,
                            input_points=input_points,
                            input_labels=input_labels,
                            return_tensors="pt"
                        ).to(app.device)

                        mask_tensor = torch.from_numpy(prev_mask).unsqueeze(0).unsqueeze(0).float().to(app.device)

                        outputs = app.sam2_model(
                            **inputs,
                            input_masks=mask_tensor,
                            multimask_output=False
                        )

                        pred_mask = outputs.pred_masks[0, 0, 0].cpu().numpy()
                        pred_mask = (pred_mask > 0.5).astype(np.uint8)

                        prev_masks[obj_id] = pred_mask.copy()
                        sam_masks_display[obj_id] = pred_mask
                        sam_masks_labeling[obj_id] = {
                            'last_mask': pred_mask.copy(),
                            'custom_label': app.tracked_objects.get(obj_id, {}).get('custom_label', ''),
                            'points_for_reprompt': [(cx, cy)],
                            'initial_bbox_prompt': None,
                        }

                except torch.cuda.OutOfMemoryError as e:
                    error_msg = f"GPU memory insufficient (OOM)!\nPropagation stopped at frame {actual_video_frame}.\n\n{str(e)[:200]}"
                    logger.error(f"SAM2 propagation OOM: {e}")
                    app.propagation_stop_requested = True
                    app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error - GPU Memory Insufficient", msg, parent=app.root))
                    break
                except Exception as e:
                    error_msg = f"Error during SAM2 tracking!\nPropagation stopped at frame {actual_video_frame}.\n\nError: {str(e)[:200]}"
                    logger.error(f"SAM2 propagation error: {e}")
                    app.propagation_stop_requested = True
                    app.root.after(0, lambda msg=error_msg: messagebox.showerror("Propagation Error", msg, parent=app.root))
                    break

                app.propagated_results[frame_idx] = {
                    'frame': frame_bgr.copy(),
                    'masks': sam_masks_labeling.copy()
                }

                progress = int((relative_idx + 1) / actual_total_frames * 100)
                app.propagation_progress = progress
                app.root.after(0, app.view.update_propagate_progress, progress, f"SAM2 processing: {relative_idx + 1}/{actual_total_frames} (frame {actual_video_frame})")

                app.current_cv_frame = frame_bgr.copy()
                app.root.after(0, app._display_cv_frame_on_view, frame_bgr, sam_masks_display)
                app.root.after(0, app.view.review_frame_slider.set, frame_idx)
                app.root.after(0, app.view.update_review_frame_info, frame_idx, actual_total_frames - 1)

    except Exception as e:
        logger.exception(f"Error during SAM2 propagation: {e}")
