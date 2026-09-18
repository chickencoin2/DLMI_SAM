"""PCS (Prompted Class Segmentation) — SAM3 text-prompt and exemplar-box driven object detection + streaming tracking."""
import logging

import cv2
import numpy as np
import torch
from PIL import Image
from tkinter import messagebox

from .customutil import process_sam_mask

logger = logging.getLogger("DLMI_SAM_LABELER.PCSController")

# Global object-id stride per text phrase in multi-class PCS.
PCS_MULTI_ID_STRIDE = 1000


def split_text_prompts(text):
    """'tomato, paprika' -> ['tomato', 'paprika']."""
    if not text:
        return []
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _mask_logit_thr(app):
    """User-adjustable binarisation threshold in logit space; 0.0 (=50%) when the app doesn't provide one."""
    fn = getattr(app, '_mask_logit_threshold', None)
    if fn is None:
        return 0.0
    try:
        return float(fn())
    except Exception:
        return 0.0


def _store_pcs_confidence_map(app, obj_id, conf_2d):
    """Keep the raw (pre-threshold) PCS logit map so the Object-conf-threshold slider can re-binarise this object's mask live."""
    if conf_2d is None or getattr(conf_2d, 'ndim', 0) != 2:
        return
    if not hasattr(app, 'current_confidence_masks'):
        app.current_confidence_masks = {}
    app.current_confidence_masks[obj_id] = conf_2d.astype(np.float32, copy=True)


def init_pcs_streaming_session(app, text_prompt):
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model not initialized.")
        return False
    try:
        model_dtype = torch.float32
        app.pcs_streaming_session = app.pcs_processor.init_video_session(
            inference_device=app.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=model_dtype,
        )
        app.pcs_streaming_session = app.pcs_processor.add_text_prompt(
            inference_session=app.pcs_streaming_session,
            text=text_prompt,
        )
        logger.info(f"PCS streaming session initialized (text: '{text_prompt}')")
        return True
    except Exception as e:
        logger.exception(f"PCS streaming session init failed: {e}")
        app.pcs_streaming_session = None
        return False


def _ingest_pcs_frame_outputs(app, processed_outputs, display, labeling,
                              id_offset=0, default_label=None):
    """Merge one session's frame outputs into display/labeling dicts, remapping ids by id_offset."""
    object_ids = processed_outputs.get("object_ids", [])
    masks = processed_outputs.get("masks", [])
    scores = processed_outputs.get("scores", [])
    logit_thr = _mask_logit_thr(app)

    for obj_id_tensor, mask, score in zip(object_ids, masks, scores):
        local_id = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(obj_id_tensor)
        obj_id = id_offset + local_id
        if obj_id in app.suppressed_sam_ids:
            logger.debug(f"PCS tracking: object {obj_id} deleted. Ignoring.")
            continue

        mask_2d = mask.float().cpu().numpy()
        if mask_2d.ndim == 3:
            mask_2d = mask_2d.squeeze()
        _store_pcs_confidence_map(app, obj_id, mask_2d)
        mask_2d = (mask_2d > logit_thr).astype(np.float32)

        display[obj_id] = mask_2d

        if obj_id in app.tracked_objects:
            app.tracked_objects[obj_id]["last_mask"] = mask_2d
            labeling[obj_id] = {
                'last_mask': mask_2d.copy(),
                'custom_label': app.tracked_objects[obj_id].get('custom_label', ''),
                'points_for_reprompt': list(app.tracked_objects[obj_id].get('points_for_reprompt', [])),
                'initial_bbox_prompt': app.tracked_objects[obj_id].get('initial_bbox_prompt'),
            }
        else:
            obj_data = {
                "last_mask": mask_2d,
                "custom_label": default_label if default_label is not None
                                else app.default_object_label_var.get(),
                "points_for_reprompt": [],
                "initial_bbox_prompt": None,
            }
            app.tracked_objects[obj_id] = obj_data
            labeling[obj_id] = {
                'last_mask': mask_2d.copy(),
                'custom_label': obj_data.get('custom_label', ''),
                'points_for_reprompt': list(obj_data.get('points_for_reprompt', [])),
                'initial_bbox_prompt': obj_data.get('initial_bbox_prompt'),
            }
            if obj_id not in app.object_colors:
                app._get_object_color(obj_id)


def perform_pcs_streaming_tracking(app, frame_bgr, frame_num):
    logger.debug(f"PCS streaming tracking start. frame: {frame_num}")
    current_sam_masks_for_display = {}
    current_sam_masks_for_labeling = {}

    multi_sessions = getattr(app, 'pcs_multi_streaming', None)

    if app.pcs_model is None or (app.pcs_streaming_session is None and not multi_sessions):
        logger.debug(f"Frame {frame_num}: PCS streaming session not ready.")
        return current_sam_masks_for_display, current_sam_masks_for_labeling

    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        inputs = app.pcs_processor(images=frame_pil, device=app.device, return_tensors="pt")

        if multi_sessions:
            # Multi-class: one streaming session per text phrase; outputs are merged with per-phrase id offsets and per-phrase labels.
            for entry in multi_sessions:
                with torch.inference_mode():
                    model_outputs = app.pcs_model(
                        inference_session=entry['session'],
                        frame=inputs.pixel_values[0],
                        reverse=False,
                    )
                processed_outputs = app.pcs_processor.postprocess_outputs(
                    entry['session'],
                    model_outputs,
                    original_sizes=inputs.original_sizes,
                )
                _ingest_pcs_frame_outputs(
                    app, processed_outputs,
                    current_sam_masks_for_display, current_sam_masks_for_labeling,
                    id_offset=entry['offset'], default_label=entry['text'],
                )
        else:
            with torch.inference_mode():
                model_outputs = app.pcs_model(
                    inference_session=app.pcs_streaming_session,
                    frame=inputs.pixel_values[0],
                    reverse=False,
                )

            processed_outputs = app.pcs_processor.postprocess_outputs(
                app.pcs_streaming_session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )
            _ingest_pcs_frame_outputs(
                app, processed_outputs,
                current_sam_masks_for_display, current_sam_masks_for_labeling,
            )

        logger.debug(f"PCS streaming tracking complete. frame: {frame_num}, "
                     f"objects: {len(current_sam_masks_for_display)}")
    except Exception as e:
        logger.error(f"PCS streaming tracking error (frame {frame_num}): {e}")
        app._tracking_fatal_error = f"PCS tracking error (frame {frame_num}): {str(e)[:100]}"

    return current_sam_masks_for_display, current_sam_masks_for_labeling


def init_pcs_session_with_single_frame(app):
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model not initialized.")
        return False
    if app.current_cv_frame is None:
        logger.error("No current frame.")
        return False
    try:
        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        app.video_frames_cache = [frame_pil]

        model_dtype = torch.float32
        app.pcs_inference_session = app.pcs_processor.init_video_session(
            video=app.video_frames_cache,
            inference_device=app.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=model_dtype,
        )
        logger.info("PCS inference session initialized (single frame)")
        return True
    except Exception as e:
        logger.exception(f"PCS inference session init failed: {e}")
        app.pcs_inference_session = None
        return False


def init_pcs_multi_streaming_sessions(app, phrases):
    """Create one streaming session per text phrase for multi-class tracking."""
    app.pcs_multi_streaming = []
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model not initialized.")
        return False
    ok_all = True
    for pi, phrase in enumerate(phrases):
        try:
            sess = app.pcs_processor.init_video_session(
                inference_device=app.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.float32,
            )
            sess = app.pcs_processor.add_text_prompt(
                inference_session=sess,
                text=phrase,
            )
            app.pcs_multi_streaming.append({
                'text': phrase,
                'session': sess,
                'offset': pi * PCS_MULTI_ID_STRIDE,
            })
            logger.info(f"PCS multi streaming session ready: '{phrase}' "
                        f"(id offset {pi * PCS_MULTI_ID_STRIDE})")
        except Exception as e:
            logger.exception(f"PCS multi streaming session init failed for '{phrase}': {e}")
            ok_all = False
    return bool(app.pcs_multi_streaming) and ok_all


def detect_objects_with_pcs_multi(app, phrases, frame_idx=0):
    """Multi-class PCS: one detection session per phrase, merged with per-phrase id offsets/labels, then one streaming session each."""
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model not initialized.")
        return False

    app.pcs_multi_streaming = []
    detected_objects = {}

    for pi, phrase in enumerate(phrases):
        offset = pi * PCS_MULTI_ID_STRIDE
        if not init_pcs_session_with_single_frame(app):
            logger.error(f"PCS multi: session init failed for '{phrase}'.")
            continue
        try:
            app.pcs_inference_session = app.pcs_processor.add_text_prompt(
                inference_session=app.pcs_inference_session,
                text=phrase,
            )
            logger.info(f"PCS multi text prompt added: '{phrase}'")

            with torch.inference_mode():
                for model_outputs in app.pcs_model.propagate_in_video_iterator(
                    inference_session=app.pcs_inference_session,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=1,
                ):
                    processed_outputs = app.pcs_processor.postprocess_outputs(
                        app.pcs_inference_session, model_outputs,
                    )
                    if model_outputs.frame_idx != frame_idx:
                        continue
                    object_ids = processed_outputs["object_ids"]
                    masks = processed_outputs["masks"]
                    scores = processed_outputs["scores"]
                    logger.info(f"PCS multi detection: '{phrase}' - {len(object_ids)} objects")

                    for obj_id_tensor, mask, score in zip(object_ids, masks, scores):
                        local_id = int(obj_id_tensor.item())
                        obj_id = offset + local_id
                        mask_2d = mask.float().cpu().numpy()
                        if mask_2d.ndim == 3:
                            mask_2d = mask_2d.squeeze()
                        _store_pcs_confidence_map(app, obj_id, mask_2d)
                        mask_2d = (mask_2d > _mask_logit_thr(app)).astype(np.float32)
                        score_val = float(score.item()) if hasattr(score, 'item') else float(score)

                        app.tracked_objects[obj_id] = {
                            "last_mask": mask_2d,
                            "custom_label": phrase,
                            "points_for_reprompt": [],
                            "initial_bbox_prompt": None,
                            "pcs_score": score_val,
                        }
                        if obj_id not in app.object_colors:
                            app._get_object_color(obj_id)
                        detected_objects[obj_id] = mask_2d
                        logger.info(f"PCS multi object {obj_id} ('{phrase}') added "
                                    f"(confidence: {score_val:.3f})")
        except Exception as e:
            logger.exception(f"PCS multi detection error for '{phrase}': {e}")

    if not detected_objects:
        logger.warning(f"PCS multi: No objects found for {phrases}.")
        return False

    app.next_obj_id_to_propose = max(detected_objects.keys()) + 1
    app._update_obj_id_info_label()
    if not init_pcs_multi_streaming_sessions(app, phrases):
        logger.warning("PCS multi streaming session init incomplete. Tracking may be limited.")
    else:
        logger.info(f"PCS multi detection complete. {len(detected_objects)} objects "
                    f"across {len(phrases)} classes. Streaming tracking ready.")
    return True


def detect_objects_with_pcs(app, text_prompt, frame_idx=0):
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model not initialized.")
        return False

    # Comma-separated prompt = multi-class detection (one phrase per class).
    phrases = split_text_prompts(text_prompt)
    if len(phrases) >= 2:
        return detect_objects_with_pcs_multi(app, phrases, frame_idx=frame_idx)
    app.pcs_multi_streaming = []

    if not init_pcs_session_with_single_frame(app):
        return False

    try:
        app.pcs_inference_session = app.pcs_processor.add_text_prompt(
            inference_session=app.pcs_inference_session,
            text=text_prompt,
        )
        logger.info(f"PCS text prompt added: '{text_prompt}'")

        detected_objects = {}
        with torch.inference_mode():
            for model_outputs in app.pcs_model.propagate_in_video_iterator(
                inference_session=app.pcs_inference_session,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            ):
                processed_outputs = app.pcs_processor.postprocess_outputs(
                    app.pcs_inference_session, model_outputs,
                )
                current_frame_idx = model_outputs.frame_idx
                if current_frame_idx == frame_idx:
                    object_ids = processed_outputs["object_ids"]
                    masks = processed_outputs["masks"]
                    scores = processed_outputs["scores"]

                    num_objects = len(object_ids)
                    logger.info(f"PCS detection complete: '{text_prompt}' - {num_objects} objects found")

                    if num_objects > 0:
                        for i, (obj_id_tensor, mask, score) in enumerate(zip(object_ids, masks, scores)):
                            obj_id = int(obj_id_tensor.item())
                            mask_2d = mask.float().cpu().numpy()
                            if mask_2d.ndim == 3:
                                mask_2d = mask_2d.squeeze()
                            _store_pcs_confidence_map(app, obj_id, mask_2d)
                            mask_2d = (mask_2d > _mask_logit_thr(app)).astype(np.float32)
                            score_val = float(score.item()) if hasattr(score, 'item') else float(score)

                            app.tracked_objects[obj_id] = {
                                "last_mask": mask_2d,
                                "custom_label": text_prompt or app.default_object_label_var.get(),
                                "points_for_reprompt": [],
                                "initial_bbox_prompt": None,
                                "pcs_score": score_val,
                            }
                            if obj_id not in app.object_colors:
                                app._get_object_color(obj_id)
                            detected_objects[obj_id] = mask_2d
                            logger.info(f"PCS object {obj_id} added (confidence: {score_val:.3f})")

        if detected_objects:
            app.next_obj_id_to_propose = max(detected_objects.keys()) + 1
            app._update_obj_id_info_label()
            if not init_pcs_streaming_session(app, text_prompt):
                logger.warning("PCS streaming session init failed. Tracking may be limited.")
            else:
                logger.info(f"PCS detection complete. {len(detected_objects)} objects registered. Streaming tracking ready.")
            return True
        else:
            logger.warning(f"PCS: No objects found for '{text_prompt}'.")
            return False
    except Exception as e:
        logger.exception(f"PCS object detection error: {e}")
        return False


def execute_pcs_with_exemplars(app):
    if app.backend is None or not app.backend.is_loaded():
        logger.error("Model backend not initialized.")
        app.update_status("Error: model backend not loaded.")
        return
    if app.current_cv_frame is None:
        logger.error("No current frame.")
        return

    text_prompt = app.pcs_text_prompt_var.get().strip() or None

    if not app.pcs_exemplar_boxes:
        logger.warning("No PCS exemplar boxes.")
        app.update_status("No exemplar boxes. Drag to add boxes.")
        return

    logger.info(f"PCS exemplar detection start. text='{text_prompt}', boxes={len(app.pcs_exemplar_boxes)}")

    try:
        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        det = app.backend.image_detect(
            frame_pil,
            text=text_prompt,
            boxes=app.pcs_exemplar_boxes,
            box_labels=app.pcs_exemplar_labels,
            threshold=app.pcs_detection_threshold_var.get(),
            mask_threshold=app.pcs_mask_threshold_var.get(),
        )

        if det.masks:
            masks = det.masks
            scores = det.scores

            app.tracked_objects.clear()

            for i, (mask_np, score_val) in enumerate(zip(masks, scores)):
                obj_id = i
                mask_np = np.squeeze(np.asarray(mask_np))
                mask_np = (mask_np > 0.5).astype(np.float32)
                score_val = float(score_val)

                app.tracked_objects[obj_id] = {
                    "last_mask": mask_np,
                    "custom_label": text_prompt or app.default_object_label_var.get(),
                    "points_for_reprompt": [],
                    "initial_bbox_prompt": None,
                    "pcs_score": score_val,
                }
                if obj_id not in app.object_colors:
                    app._get_object_color(obj_id)
                logger.info(f"PCS exemplar detection: object {obj_id} added (confidence: {score_val:.3f})")

            app.next_obj_id_to_propose = len(app.tracked_objects)
            app._update_obj_id_info_label()

            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
            app.update_status(f"PCS exemplar detection complete: {len(app.tracked_objects)} objects")
            logger.info(f"PCS exemplar detection complete: {len(app.tracked_objects)} objects found")
        else:
            app.update_status("PCS exemplar detection: No matching objects")
            logger.warning("PCS exemplar detection: No objects found.")
    except Exception as e:
        logger.exception(f"PCS exemplar detection error: {e}")
        app.update_status(f"PCS exemplar detection error: {e}")


def execute_pcs_detection(app):
    if app.pcs_model is None or app.current_cv_frame is None:
        messagebox.showwarning("Info", "PCS model not loaded or no frame available.", parent=app.root)
        return

    text_prompt = app.pcs_text_prompt_var.get().strip()
    if not text_prompt:
        messagebox.showwarning("Info", "Please enter a text prompt.", parent=app.root)
        return

    app.update_status(f"PCS detection: '{text_prompt}'... (loading video frames)")
    app.root.update()

    if detect_objects_with_pcs(app, text_prompt, frame_idx=0):
        num_objects = len(app.tracked_objects)
        app.update_status(f"PCS detection complete: '{text_prompt}' - {num_objects} objects added")
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    else:
        app.update_status(f"PCS detection complete: No objects for '{text_prompt}'")
        messagebox.showinfo("Info", f"No objects found for '{text_prompt}'.", parent=app.root)


def perform_pcs_single_image_detection(app, frame_bgr, text_prompt, frame_idx):
    if app.backend is None or not app.backend.is_loaded():
        logger.error("PCS(per-image) mode: backend not loaded.")
        return {}, {}

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Comma-separated prompt = detect each phrase independently and merge.
    phrases = split_text_prompts(text_prompt)
    is_multi = len(phrases) >= 2
    if not phrases:
        phrases = [text_prompt]

    masks_for_display = {}
    masks_for_labeling = {}
    next_obj_id = 1

    for phrase in phrases:
        det = app.backend.image_detect(
            frame_pil, text=phrase,
            threshold=app.pcs_detection_threshold_var.get(),
            mask_threshold=app.pcs_mask_threshold_var.get(),
        )

        if not det.masks:
            continue
        for mask_np0 in det.masks:
            mask_np = np.squeeze(np.asarray(mask_np0)).astype(np.uint8)

            pil_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            proc_mask = process_sam_mask(
                mask_np, pil_size,
                apply_closing=app.sam_apply_closing_var.get(),
                closing_kernel_size=app.sam_closing_kernel_size_var.get()
            )

            if proc_mask is not None and proc_mask.any():
                obj_id = next_obj_id
                next_obj_id += 1
                masks_for_display[obj_id] = proc_mask
                masks_for_labeling[obj_id] = {
                    'last_mask': proc_mask,
                    # Multi-class: each detection carries its own phrase as label; single prompt keeps the legacy default label.
                    'custom_label': phrase if is_multi else app.default_object_label_var.get()
                }
                if obj_id not in app.object_colors:
                    app._get_object_color(obj_id)

    return masks_for_display, masks_for_labeling


def apply_loaded_labels_for_pcs(app, loaded_objects):
    if not loaded_objects:
        return

    label_set = set()
    for obj in loaded_objects:
        label = obj.get('label', '')
        if label:
            label_set.add(label)

    if not label_set:
        messagebox.showinfo("Info", "No object labels found in loaded labels.", parent=app.root)
        return

    prompt_text = ", ".join(sorted(label_set))
    app.pcs_text_prompt_var.set(prompt_text)

    response = messagebox.askyesno(
        "PCS Mode Label Apply",
        f"Proceed with PCS detection using these object names?\n\n"
        f"Object names: {prompt_text}\n\n"
        f"Yes: Execute 'Cut here' and start PCS detection\n"
        f"No: Only set text prompt",
        parent=app.root
    )

    if response:
        app.propagated_results = {}
        app.tracked_objects.clear()
        app.next_obj_id_to_propose = 1
        app.is_tracking_ever_started = False

        if app.inference_session is not None:
            try:
                app.inference_session.reset_inference_session()
            except Exception:
                pass
            app.inference_session = None

        app._update_obj_id_info_label()
        execute_pcs_detection(app)
    else:
        messagebox.showinfo(
            "Info",
            f"Text prompt has been set.\n"
            f"'{prompt_text}'\n\n"
            f"Press 'Detect' button to start PCS detection.",
            parent=app.root
        )
