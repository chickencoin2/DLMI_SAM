import numpy as np
import torch
import contextlib
import logging
import tkinter as tk
import cv2
from PIL import Image

from .customutil import process_sam_mask, get_bbox_from_mask, get_hashable_obj_id, is_bbox_on_edge

logger = logging.getLogger("DLMI_SAM_LABELER.SAMInteraction")


def prepare_and_conditionally_reset_sam3(app, target_existing_obj_id, proposed_obj_id_for_new):
    logger.debug(f"SAM3 preparation and reset review started. target: {target_existing_obj_id}, new_prop: {proposed_obj_id_for_new}")
    needs_reset = False

    if app.is_tracking_ever_started:
        needs_reset = True
        logger.info("Manual prompt detected during tracking. Starting SAM3 session reset procedure.")

    if needs_reset:
        logger.info("SAM3 inference session reset and all object re-prompt started.")
        snapshot_of_tracked_objects = {
            obj_id: data.copy() for obj_id, data in app.tracked_objects.items()
            if obj_id not in app.suppressed_sam_ids
        }

        try:
            if app.inference_session is not None:
                app.inference_session.reset_inference_session()
                logger.info("inference_session.reset_inference_session() call completed.")
        except Exception as e:
            logger.warning(f"inference_session reset error: {e}")

        app.is_tracking_ever_started = False
        app.last_active_tracked_sam_ids = set()

        if app.current_cv_frame is None:
            logger.error("No current_cv_frame for session initialization after SAM3 reset.")
            tk.messagebox.showerror("Error", "No current frame information during reset.", parent=app.root)
            return False

        if app.inference_session is None:
            if not app._init_inference_session():
                logger.error("SAM3 inference session initialization failed")
                return False

        temp_reprompted_objects = {}
        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        inputs = app.tracker_processor(images=frame_pil, device=app.device, return_tensors="pt")

        for obj_id, data in snapshot_of_tracked_objects.items():
            latest_mask = data.get("last_mask")
            if latest_mask is None or not latest_mask.any():
                logger.warning(f"ObjID {obj_id} skipped due to no latest mask for re-prompting.")
                continue

            reprompt_bbox = get_bbox_from_mask(latest_mask)
            if reprompt_bbox is None:
                logger.warning(f"ObjID {obj_id} skipped due to unable to generate valid BBox from latest mask.")
                continue

            M = cv2.moments(latest_mask.astype(np.uint8))
            if M["m00"] > 0:
                center_x = float(M["m10"] / M["m00"])
                center_y = float(M["m01"] / M["m00"])
            else:
                center_x = (reprompt_bbox[0] + reprompt_bbox[2]) / 2
                center_y = (reprompt_bbox[1] + reprompt_bbox[3]) / 2

            try:
                app.tracker_processor.add_inputs_to_inference_session(
                    inference_session=app.inference_session,
                    frame_idx=0,
                    obj_ids=obj_id,
                    input_points=[[[[center_x, center_y]]]],
                    input_labels=[[[1]]],
                    original_size=inputs.original_sizes[0],
                )

                frame_tensor = inputs.pixel_values[0]
                if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
                    frame_tensor = frame_tensor.to(dtype=torch.float32)

                with torch.inference_mode():
                    outputs = app.tracker_model(
                        inference_session=app.inference_session,
                        frame=frame_tensor,
                    )

                masks = app.tracker_processor.post_process_masks(
                    [outputs.pred_masks],
                    original_sizes=[[app.inference_session.video_height, app.inference_session.video_width]],
                    binarize=False
                )[0]

                obj_ids_in_session = list(app.inference_session.obj_ids) if hasattr(app.inference_session, 'obj_ids') else [obj_id]
                if obj_id in obj_ids_in_session:
                    idx = obj_ids_in_session.index(obj_id)
                    if idx < masks.shape[0]:
                        mask_tensor = masks[idx]
                        mask_np = mask_tensor.float().cpu().numpy()
                        pil_size = app.current_frame_pil_rgb_original.size if app.current_frame_pil_rgb_original else (app.current_cv_frame.shape[1], app.current_cv_frame.shape[0])
                        proc_mask = process_sam_mask(mask_np, pil_size)
                        if proc_mask is not None:
                            data["last_mask"] = proc_mask
                            temp_reprompted_objects[obj_id] = data
                            logger.info(f"ObjID {obj_id} SAM3 re-prompt success.")
                        else:
                            logger.warning(f"ObjID {obj_id} re-prompt mask processing failed.")
                    else:
                        logger.warning(f"ObjID {obj_id}: mask index out of range.")
                else:
                    logger.warning(f"ObjID {obj_id} is not in the session.")

            except Exception as e:
                logger.exception(f"Error during re-prompt for ObjID {obj_id}: {e}")

        app.tracked_objects.clear()
        app.tracked_objects.update(temp_reprompted_objects)
        logger.info(f"tracked_objects after SAM3 reset and re-prompt: {list(app.tracked_objects.keys())}")

        current_max_id = max(list(app.tracked_objects.keys()) + [0]) if app.tracked_objects else 0
        app.next_obj_id_to_propose = max(app.next_obj_id_to_propose, current_max_id + 1)
        app._update_obj_id_info_label()
        app.just_reset_sam = True
        return True

    elif app.inference_session is None:
        if app.current_cv_frame is None:
            logger.error("No current_cv_frame for first frame loading. Prompt aborted.")
            tk.messagebox.showerror("Error", "No current frame information during first frame loading.", parent=app.root)
            return False

        if not app._init_inference_session():
            logger.error("SAM3 inference session initialization failed")
            return False

        app.is_tracking_ever_started = False
        app.just_reset_sam = False
        logger.info("SAM3 inference session initialized (first prompt).")

    return True


def prepare_and_conditionally_reset_sam(app, target_existing_obj_id, proposed_obj_id_for_new):
    return prepare_and_conditionally_reset_sam3(app, target_existing_obj_id, proposed_obj_id_for_new)


def build_sam3_prompt_args(app, prompt_type, coords, label, target_existing_obj_id, proposed_obj_id_for_new, custom_label):
    logger.debug(f"SAM3 prompt argument build. type: {prompt_type}, target: {target_existing_obj_id}, new_prop: {proposed_obj_id_for_new}")

    final_obj_id_for_sam_call = None
    obj_data_for_update = {}

    prompt_data = {
        "frame_idx": 0,
        "input_points": None,
        "input_labels": None,
        "input_boxes": None,
    }

    if target_existing_obj_id is not None:
        final_obj_id_for_sam_call = target_existing_obj_id
        obj_data_for_update = app.tracked_objects.get(final_obj_id_for_sam_call, {}).copy()

        if prompt_type == 'bbox':
            obj_data_for_update["points_for_reprompt"] = []
            obj_data_for_update["initial_bbox_prompt"] = coords.copy()
            prompt_data["input_boxes"] = [[[float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]]]
            logger.info(f"ObjID {final_obj_id_for_sam_call}: reassigned/modified with BBox prompt.")

        elif prompt_type == 'point':
            if "points_for_reprompt" not in obj_data_for_update:
                obj_data_for_update["points_for_reprompt"] = []
            obj_data_for_update["points_for_reprompt"].append((coords.copy(), label))

            all_points = []
            all_labels = []
            for pt, lbl in obj_data_for_update["points_for_reprompt"]:
                if hasattr(pt, 'flatten'):
                    pt_flat = pt.flatten()
                else:
                    pt_flat = np.array(pt).flatten()
                if len(pt_flat) >= 2:
                    all_points.append([float(pt_flat[0]), float(pt_flat[1])])
                    all_labels.append(int(lbl))

            if all_points:
                prompt_data["input_points"] = [[all_points]]
                prompt_data["input_labels"] = [[all_labels]]

        if "custom_label" not in obj_data_for_update or not obj_data_for_update["custom_label"]:
            obj_data_for_update["custom_label"] = custom_label if custom_label else app.default_object_label_var.get()

    elif proposed_obj_id_for_new is not None:
        final_obj_id_for_sam_call = proposed_obj_id_for_new
        if final_obj_id_for_sam_call in app.tracked_objects and not app.just_reset_sam:
            logger.warning(f"Proposed new object ID {final_obj_id_for_sam_call} already exists. Using next_obj_id_to_propose.")
            final_obj_id_for_sam_call = app.next_obj_id_to_propose

        obj_data_for_update = {
            "points_for_reprompt": [],
            "custom_label": custom_label if custom_label else app.default_object_label_var.get(),
        }

        if prompt_type == 'bbox':
            prompt_data["input_boxes"] = [[[float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]]]
            obj_data_for_update["initial_bbox_prompt"] = coords.copy()

        elif prompt_type == 'point':
            pt_flat = coords.flatten() if hasattr(coords, 'flatten') else np.array(coords).flatten()
            prompt_data["input_points"] = [[[[float(pt_flat[0]), float(pt_flat[1])]]]]
            prompt_data["input_labels"] = [[[int(label)]]]
            obj_data_for_update["points_for_reprompt"].append((coords.copy(), label))
    else:
        logger.error("SAM3 prompt target ID not specified.")
        return None, None, None

    prompt_data["obj_id"] = final_obj_id_for_sam_call
    return final_obj_id_for_sam_call, prompt_data, obj_data_for_update


def call_sam3_and_update_state(app, final_obj_id, prompt_data, obj_data_to_update_with_mask, prompt_type, target_existing_obj_id):
    logger.debug(f"SAM3 call and state update. obj_id: {final_obj_id}, type: {prompt_type}")

    log_data = {k: v for k, v in prompt_data.items() if v is not None and k not in ['frame_idx', 'obj_id']}
    logger.info(f"SAM3 add_inputs_to_inference_session call. Request ID: {final_obj_id}, Data: {log_data}")

    if hasattr(app, 'object_prompt_history'):
        if final_obj_id not in app.object_prompt_history:
            app.object_prompt_history[final_obj_id] = {
                'boxes': [], 'positive_points': [], 'negative_points': [],
                'mask_contours': [], 'exemplar_positive': [], 'exemplar_negative': []
            }
        history = app.object_prompt_history[final_obj_id]
        if prompt_data.get('input_boxes') is not None:
            for box_batch in prompt_data['input_boxes']:
                for box in box_batch:
                    history['boxes'].append(list(box))
        if prompt_data.get('input_points') is not None and prompt_data.get('input_labels') is not None:
            points = prompt_data['input_points']
            labels = prompt_data['input_labels']
            for pt_batch, lbl_batch in zip(points, labels):
                for pt, lbl in zip(pt_batch[0] if len(pt_batch) > 0 else [], lbl_batch[0] if len(lbl_batch) > 0 else []):
                    if lbl == 1:
                        history['positive_points'].append(list(pt))
                    else:
                        history['negative_points'].append(list(pt))
        if prompt_data.get('input_masks') is not None:
            for mask_batch in prompt_data['input_masks']:
                for mask in mask_batch:
                    if hasattr(mask, 'numpy'):
                        mask = mask.numpy()
                    mask_uint8 = (mask > 0).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        history['mask_contours'].append(contour.tolist())

    processed_sam_id_actual = None

    try:
        if app.current_cv_frame is None:
            logger.error("No current_cv_frame.")
            return

        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        inputs = app.tracker_processor(images=frame_pil, device=app.device, return_tensors="pt")

        frame_idx = prompt_data.get("frame_idx", 0)

        input_points = None
        input_labels = None

        if prompt_data.get("input_points") is not None:
            input_points = prompt_data["input_points"]
            input_labels = prompt_data["input_labels"]

        if prompt_data.get("input_boxes") is not None:
            box = prompt_data["input_boxes"][0][0]
            center_x = float((box[0] + box[2]) / 2)
            center_y = float((box[1] + box[3]) / 2)
            input_points = [[[[center_x, center_y]]]]
            input_labels = [[[1]]]

        app.tracker_processor.add_inputs_to_inference_session(
            inference_session=app.inference_session,
            frame_idx=frame_idx,
            obj_ids=final_obj_id,
            input_points=input_points,
            input_labels=input_labels,
            original_size=inputs.original_sizes[0],
        )
        logger.info(f"SAM3 prompt added to session. obj_id: {final_obj_id}")

        frame_tensor = inputs.pixel_values[0]
        if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
            frame_tensor = frame_tensor.to(dtype=torch.float32)

        with torch.inference_mode():
            session_num_frames = getattr(app.inference_session, 'num_frames', None) or 0

            if session_num_frames > 0:
                logger.debug(f"Streaming mode: referencing existing frame (num_frames={session_num_frames}, frame_idx={frame_idx})")
                outputs = app.tracker_model(
                    inference_session=app.inference_session,
                    frame_idx=frame_idx,
                )
            else:
                logger.debug(f"Streaming mode: processing first frame")
                outputs = app.tracker_model(
                    inference_session=app.inference_session,
                    frame=frame_tensor,
                )

        pil_size_for_mask = app.current_frame_pil_rgb_original.size if app.current_frame_pil_rgb_original else (app.current_cv_frame.shape[1], app.current_cv_frame.shape[0])

        masks = app.tracker_processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=[list(pil_size_for_mask[::-1])],
            binarize=False
        )[0]

        obj_ids_in_session = list(app.inference_session.obj_ids) if hasattr(app.inference_session, 'obj_ids') else [final_obj_id]

        if final_obj_id in obj_ids_in_session:
            idx = obj_ids_in_session.index(final_obj_id)
            if idx < masks.shape[0]:
                mask_tensor = masks[idx]
                mask_np = mask_tensor.float().cpu().numpy()

                apply_closing = app.sam_apply_closing_var.get()
                closing_kernel = app.sam_closing_kernel_size_var.get()
                proc_mask = process_sam_mask(
                    mask_np,
                    pil_size_for_mask,
                    apply_closing=apply_closing,
                    closing_kernel_size=closing_kernel
                )

                if proc_mask is not None:
                    processed_sam_id_actual = final_obj_id
                    obj_data_to_update_with_mask["last_mask"] = proc_mask

                    if ("initial_bbox_prompt" not in obj_data_to_update_with_mask or obj_data_to_update_with_mask["initial_bbox_prompt"] is None) and \
                       prompt_type == 'point' and target_existing_obj_id is None:
                        obj_data_to_update_with_mask["initial_bbox_prompt"] = get_bbox_from_mask(
                            proc_mask, app.erosion_kernel_size.get(), app.erosion_iterations.get(), app.min_bbox_area_for_reprompt.get()
                        )

                    app.tracked_objects[processed_sam_id_actual] = obj_data_to_update_with_mask
                    logger.info(f"SAM3 ObjID {processed_sam_id_actual} information updated/added. tracked_objects: {list(app.tracked_objects.keys())}")

                    if processed_sam_id_actual not in app.object_colors:
                        app._get_object_color(processed_sam_id_actual)

                    correction_completed_or_reassigned = False

                    if hasattr(app, 'interaction_correction_pending') and app.interaction_correction_pending == processed_sam_id_actual:
                        logger.info(f"Object {processed_sam_id_actual} auto-correction completed.")
                        if hasattr(app, 'problematic_objects_flagged') and processed_sam_id_actual in app.problematic_objects_flagged:
                            del app.problematic_objects_flagged[processed_sam_id_actual]
                        app.interaction_correction_pending = None
                        app.selected_object_sam_id = None
                        app.root.after(0, app.update_status, f"Object {processed_sam_id_actual} auto-correction completed.")
                        correction_completed_or_reassigned = True

                    if hasattr(app, 'reassign_bbox_mode_active_sam_id') and app.reassign_bbox_mode_active_sam_id == processed_sam_id_actual:
                        logger.info(f"Object {processed_sam_id_actual} BBox reassignment completed.")
                        app.reassign_bbox_mode_active_sam_id = None
                        app.root.after(0, app.update_status, f"Object {processed_sam_id_actual} BBox reassignment completed.")
                        correction_completed_or_reassigned = True

                    if correction_completed_or_reassigned:
                        app.autolabel_active = False
                        app.playback_paused = True
                        app.root.after(0, app._update_ui_for_autolabel_state, False)
                        app.root.after(0, app._update_obj_id_info_label)

                    if target_existing_obj_id is None:
                        current_max_id = max(list(app.tracked_objects.keys()) + [0]) if app.tracked_objects else 0
                        app.next_obj_id_to_propose = max(current_max_id + 1, app.next_obj_id_to_propose, processed_sam_id_actual + 1)
                        app._update_obj_id_info_label()
                else:
                    logger.warning(f"SAM3 ObjID {final_obj_id} mask processing failed.")
            else:
                logger.warning(f"SAM3 ObjID {final_obj_id}: mask index out of range.")
        else:
            logger.warning(f"SAM3 ObjID {final_obj_id} is not in the session.")

    except Exception as e:
        logger.exception(f"Error during SAM3 prompt processing: {e}")

    if not processed_sam_id_actual:
        logger.warning(f"SAM3 prompt application failed. Request ID: {final_obj_id}")

        if hasattr(app, 'interaction_correction_pending') and app.interaction_correction_pending == final_obj_id:
            logger.error(f"Object {final_obj_id} auto-correction failed.")
            app.root.after(0, lambda: tk.messagebox.showerror("Auto-correction Failed", f"Failed to auto-correct object {final_obj_id}.", parent=app.root))
            if hasattr(app, 'problematic_objects_flagged'):
                app.problematic_objects_flagged[final_obj_id] = 'pending_user_interaction'
            app.interaction_correction_pending = None

        if hasattr(app, 'reassign_bbox_mode_active_sam_id') and app.reassign_bbox_mode_active_sam_id == final_obj_id:
            logger.error(f"Object {final_obj_id} BBox reassignment failed.")
            app.root.after(0, lambda: tk.messagebox.showerror("BBox Reassignment Failed", f"Failed to reassign BBox for object {final_obj_id}.", parent=app.root))
            app.reassign_bbox_mode_active_sam_id = None

        app.selected_object_sam_id = None
        app.autolabel_active = False
        app.playback_paused = True
        app.root.after(0, app._update_ui_for_autolabel_state, False)
        app.root.after(0, app._update_obj_id_info_label)


def _handle_incremental_add(app, prompt_type, coords, label, proposed_obj_id_for_new, custom_label):
    logger.debug(f"SAM3 incremental add started. type: {prompt_type}, new_prop: {proposed_obj_id_for_new}")

    try:
        final_obj_id = proposed_obj_id_for_new
        if final_obj_id in app.tracked_objects:
            logger.warning(f"Proposed new object ID {final_obj_id} already exists. Using next_obj_id_to_propose.")
            final_obj_id = app.next_obj_id_to_propose

        if app.current_cv_frame is None:
            logger.error("No current_cv_frame.")
            return

        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        inputs = app.tracker_processor(images=frame_pil, device=app.device, return_tensors="pt")

        if prompt_type == 'bbox':
            center_x = (float(coords[0]) + float(coords[2])) / 2
            center_y = (float(coords[1]) + float(coords[3])) / 2
            input_points = [[[[center_x, center_y]]]]
            input_labels = [[[1]]]
        elif prompt_type == 'point':
            pt_flat = coords.flatten() if hasattr(coords, 'flatten') else np.array(coords).flatten()
            input_points = [[[[float(pt_flat[0]), float(pt_flat[1])]]]]
            input_labels = [[[int(label)]]]
        else:
            return

        app.tracker_processor.add_inputs_to_inference_session(
            inference_session=app.inference_session,
            frame_idx=app.current_frame_idx_conceptual,
            obj_ids=final_obj_id,
            input_points=input_points,
            input_labels=input_labels,
            original_size=inputs.original_sizes[0],
        )

        frame_tensor = inputs.pixel_values[0]
        if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
            frame_tensor = frame_tensor.to(dtype=torch.float32)

        with torch.inference_mode():
            outputs = app.tracker_model(
                inference_session=app.inference_session,
                frame=frame_tensor,
            )

        pil_size = app.current_frame_pil_rgb_original.size if app.current_frame_pil_rgb_original else (app.current_cv_frame.shape[1], app.current_cv_frame.shape[0])
        masks = app.tracker_processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=[list(pil_size[::-1])],
            binarize=False
        )[0]

        obj_ids_in_session = list(app.inference_session.obj_ids) if hasattr(app.inference_session, 'obj_ids') else [final_obj_id]
        if final_obj_id in obj_ids_in_session:
            idx = obj_ids_in_session.index(final_obj_id)
            if idx < masks.shape[0]:
                mask_tensor = masks[idx]
                mask_np = mask_tensor.float().cpu().numpy()
                proc_mask = process_sam_mask(
                    mask_np,
                    pil_size,
                    apply_closing=app.sam_apply_closing_var.get(),
                    closing_kernel_size=app.sam_closing_kernel_size_var.get()
                )

                if proc_mask is not None:
                    new_obj_data = {
                        "last_mask": proc_mask,
                        "initial_bbox_prompt": coords.copy() if prompt_type == 'bbox' else get_bbox_from_mask(proc_mask),
                        "points_for_reprompt": [(coords.copy(), label)] if prompt_type == 'point' else [],
                        "custom_label": custom_label if custom_label else app.default_object_label_var.get(),
                    }
                    app.tracked_objects[final_obj_id] = new_obj_data

                    if final_obj_id not in app.object_colors:
                        app._get_object_color(final_obj_id)

                    current_max_id = max(list(app.tracked_objects.keys()) + [0])
                    app.next_obj_id_to_propose = max(current_max_id + 1, app.next_obj_id_to_propose)
                    app._update_obj_id_info_label()
                    logger.info(f"SAM3 incremental add completed. New object ID: {final_obj_id}")
                else:
                    logger.error("SAM3 incremental add: mask processing failed.")
            else:
                logger.error(f"SAM3 incremental add: mask index out of range.")
        else:
            logger.error(f"SAM3 incremental add: object ID {final_obj_id} is not in the session.")

    except Exception as e:
        logger.exception("Error during SAM3 incremental add:")
        app.update_status("Error: incremental add failed.")


def handle_sam_prompt(app, prompt_type, coords, label=None, proposed_obj_id_for_new=None, target_existing_obj_id=None, custom_label=None):
    logger.info(f"SAM3 prompt processing started. type: {prompt_type}, target: {target_existing_obj_id}, new_prop: {proposed_obj_id_for_new}")

    if app.tracker_model is None:
        tk.messagebox.showerror("Error", "SAM3 Tracker model not initialized.", parent=app.root)
        return
    if app.current_cv_frame is None:
        logger.error("No current_cv_frame during SAM3 prompt processing.")
        return

    app.sam_operation_in_progress = True
    try:
        current_autocast_context = app.autocast_context if hasattr(app, 'autocast_context') else contextlib.nullcontext()
        with torch.inference_mode(), current_autocast_context:

            adjust_method = app.new_object_method_var.get() if hasattr(app, 'new_object_method_var') else 'reset'
            is_manual_prompt_while_tracking = app.is_tracking_ever_started

            if adjust_method == 'incremental' and is_manual_prompt_while_tracking and proposed_obj_id_for_new is not None:
                logger.info("Using SAM3 incremental add mode")
                _handle_incremental_add(app, prompt_type, coords, label, proposed_obj_id_for_new, custom_label)
                if app.current_cv_frame is not None:
                    app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
                app.sam_operation_in_progress = False
                return

            should_proceed = prepare_and_conditionally_reset_sam3(app, target_existing_obj_id, proposed_obj_id_for_new)

            if not should_proceed:
                logger.info("Received signal to stop processing during SAM3 preparation stage.")
                if app.current_cv_frame is not None:
                    app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
                app.sam_operation_in_progress = False
                return

            final_obj_id, prompt_data, obj_data_for_update = build_sam3_prompt_args(
                app, prompt_type, coords, label, target_existing_obj_id, proposed_obj_id_for_new, custom_label
            )

            if final_obj_id is None:
                logger.error("SAM3 prompt argument build failed.")
                app.sam_operation_in_progress = False
                return

            call_sam3_and_update_state(app, final_obj_id, prompt_data, obj_data_for_update, prompt_type, target_existing_obj_id)

            if app.current_cv_frame is not None:
                app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())

    except Exception as e:
        logger.exception("SAM3 handle_sam_prompt error:")
    finally:
        app.sam_operation_in_progress = False
        app._update_obj_id_info_label()
