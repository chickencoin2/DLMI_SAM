import numpy as np
import torch
import cv2
import threading
import logging
from tkinter import messagebox
from PIL import Image

from .customutil import process_sam_mask

logger = logging.getLogger("DLMI_SAM_LABELER.PCSController")


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


def _perform_pcs_streaming_tracking(app, frame_bgr, frame_num):
    logger.debug(f"PCS streaming tracking start. frame: {frame_num}")
    current_sam_masks_for_display = {}
    current_sam_masks_for_labeling = {}

    if app.pcs_model is None or app.pcs_streaming_session is None:
        logger.debug(f"Frame {frame_num}: PCS streaming session not ready.")
        return current_sam_masks_for_display, current_sam_masks_for_labeling

    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        inputs = app.pcs_processor(images=frame_pil, device=app.device, return_tensors="pt")

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

        object_ids = processed_outputs.get("object_ids", [])
        masks = processed_outputs.get("masks", [])
        scores = processed_outputs.get("scores", [])

        target_h, target_w = frame_bgr.shape[:2]

        for obj_id_tensor, mask, score in zip(object_ids, masks, scores):
            obj_id = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(obj_id_tensor)

            if obj_id in app.suppressed_sam_ids:
                logger.debug(f"PCS tracking: object {obj_id} deleted. Ignoring.")
                continue

            mask_2d = mask.float().cpu().numpy()
            if mask_2d.ndim == 3:
                mask_2d = mask_2d.squeeze()
            mask_2d = (mask_2d > 0.0).astype(np.float32)

            current_sam_masks_for_display[obj_id] = mask_2d

            if obj_id in app.tracked_objects:
                app.tracked_objects[obj_id]["last_mask"] = mask_2d
                current_sam_masks_for_labeling[obj_id] = {
                    'last_mask': mask_2d.copy(),
                    'custom_label': app.tracked_objects[obj_id].get('custom_label', ''),
                    'points_for_reprompt': list(app.tracked_objects[obj_id].get('points_for_reprompt', [])),
                    'initial_bbox_prompt': app.tracked_objects[obj_id].get('initial_bbox_prompt'),
                }
            else:
                obj_data = {
                    "last_mask": mask_2d,
                    "custom_label": app.default_object_label_var.get(),
                    "points_for_reprompt": [],
                    "initial_bbox_prompt": None,
                }
                app.tracked_objects[obj_id] = obj_data
                current_sam_masks_for_labeling[obj_id] = {
                    'last_mask': mask_2d.copy(),
                    'custom_label': obj_data.get('custom_label', ''),
                    'points_for_reprompt': list(obj_data.get('points_for_reprompt', [])),
                    'initial_bbox_prompt': obj_data.get('initial_bbox_prompt'),
                }

                if obj_id not in app.object_colors:
                    app._get_object_color(obj_id)

        logger.debug(f"PCS streaming tracking complete. frame: {frame_num}, objects: {len(current_sam_masks_for_display)}")

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


def detect_objects_with_pcs(app, text_prompt, frame_idx=0):
    if app.pcs_model is None or app.pcs_processor is None:
        logger.error("PCS model not initialized.")
        return False

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
                    app.pcs_inference_session,
                    model_outputs,
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
                            mask_2d = (mask_2d > 0.0).astype(np.float32)

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

            text_prompt_for_streaming = text_prompt
            if not init_pcs_streaming_session(app, text_prompt_for_streaming):
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


def _execute_pcs_with_exemplars(app):
    if app.image_model is None or app.image_processor is None:
        logger.error("Sam3 Image model not initialized.")
        app.update_status("Error: Sam3 Image model not loaded.")
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
    logger.info(f"  - Boxes: {app.pcs_exemplar_boxes}")
    logger.info(f"  - Labels: {app.pcs_exemplar_labels}")

    try:
        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_h, frame_w = app.current_cv_frame.shape[:2]

        input_boxes = [app.pcs_exemplar_boxes]
        input_boxes_labels = [app.pcs_exemplar_labels]

        processor_kwargs = {
            "images": frame_pil,
            "input_boxes": input_boxes,
            "input_boxes_labels": input_boxes_labels,
            "return_tensors": "pt"
        }
        if text_prompt:
            processor_kwargs["text"] = text_prompt

        inputs = app.image_processor(**processor_kwargs).to(app.device)

        if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
            if hasattr(inputs, 'pixel_values'):
                inputs.pixel_values = inputs.pixel_values.to(dtype=torch.float32)

        with torch.inference_mode():
            outputs = app.image_model(**inputs)

        target_sizes = inputs.get("original_sizes")
        if target_sizes is not None:
            target_sizes = target_sizes.tolist()
        else:
            target_sizes = [[frame_h, frame_w]]

        results = app.image_processor.post_process_instance_segmentation(
            outputs,
            threshold=app.pcs_detection_threshold_var.get(),
            mask_threshold=app.pcs_mask_threshold_var.get(),
            target_sizes=target_sizes
        )[0]

        if 'masks' in results and len(results['masks']) > 0:
            masks = results['masks']
            scores = results.get('scores', [1.0] * len(masks))

            app.tracked_objects.clear()

            for i, (mask, score) in enumerate(zip(masks, scores)):
                obj_id = i
                mask_np = mask.cpu().numpy()
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()
                mask_np = (mask_np > 0.5).astype(np.float32)

                score_val = float(score.item()) if hasattr(score, 'item') else float(score)

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


def _perform_pcs_single_image_detection(app, frame_bgr, text_prompt, frame_idx):
    if app.image_model is None or app.image_processor is None:
        logger.error("PCS(per-image) mode: Image model not loaded.")
        return {}, {}

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    inputs = app.image_processor(
        images=frame_pil,
        text=text_prompt,
        return_tensors="pt"
    ).to(app.device)

    if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
        if hasattr(inputs, 'pixel_values'):
            inputs.pixel_values = inputs.pixel_values.to(dtype=torch.float32)

    with torch.inference_mode():
        outputs = app.image_model(**inputs)

    target_sizes = inputs.get("original_sizes")
    if target_sizes is not None:
        target_sizes = target_sizes.tolist()
    else:
        target_sizes = [[frame_bgr.shape[0], frame_bgr.shape[1]]]

    results = app.image_processor.post_process_instance_segmentation(
        outputs,
        threshold=app.pcs_detection_threshold_var.get(),
        mask_threshold=app.pcs_mask_threshold_var.get(),
        target_sizes=target_sizes
    )[0]

    masks_for_display = {}
    masks_for_labeling = {}

    if 'masks' in results and len(results['masks']) > 0:
        masks = results['masks']
        scores = results.get('scores', [1.0] * len(masks))

        for obj_idx, mask_tensor in enumerate(masks):
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

            pil_size = (frame_bgr.shape[1], frame_bgr.shape[0])
            proc_mask = process_sam_mask(
                mask_np, pil_size,
                apply_closing=app.sam_apply_closing_var.get(),
                closing_kernel_size=app.sam_closing_kernel_size_var.get()
            )

            if proc_mask is not None and proc_mask.any():
                obj_id = obj_idx + 1
                masks_for_display[obj_id] = proc_mask
                masks_for_labeling[obj_id] = {
                    'last_mask': proc_mask,
                    'custom_label': app.default_object_label_var.get()
                }

                if obj_id not in app.object_colors:
                    app._get_object_color(obj_id)

    return masks_for_display, masks_for_labeling


def _run_pcs_review_mode_async(app):
    from util.autolabel_workflow import run_pcs_review_mode

    def progress_callback(current, total):
        progress = int(current / total * 100)
        app.root.after(0, app.view.update_propagate_progress, progress,
                        f"Review mode: {current}/{total} frames analyzing...")

    def run_review():
        app.root.after(0, app.update_status, "PCS review mode running...")
        run_pcs_review_mode(app, progress_callback)
        app.root.after(0, app.update_status, "PCS review mode complete")
        app.root.after(0, app.view.update_propagate_progress, 100, "Review mode complete")

    thread = threading.Thread(target=run_review, daemon=True)
    thread.start()
