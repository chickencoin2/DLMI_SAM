import threading
import logging

import torch
import cv2
import numpy as np
from PIL import Image
from tkinter import messagebox

try:
    from transformers import Sam2Model, Sam2Processor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

logger = logging.getLogger("DLMI_SAM_LABELER.SAM2Controller")


def load_sam2_model_async(app):
    if not SAM2_AVAILABLE:
        messagebox.showerror("SAM2 Error", "SAM2 library is not installed.")
        app.view.update_sam2_loading_complete(success=False)
        return
    if app.sam2_loading_in_progress:
        logger.warning("SAM2 model loading already in progress")
        return
    app.sam2_loading_in_progress = True
    threading.Thread(target=lambda: _load_sam2_model_thread(app), daemon=True).start()


def _load_sam2_model_thread(app):
    try:
        logger.info(f"SAM2 model loading started: {app.sam2_model_id}")
        app.sam2_processor = Sam2Processor.from_pretrained(app.sam2_model_id)
        app.sam2_model = Sam2Model.from_pretrained(
            app.sam2_model_id, torch_dtype=torch.float32
        ).to(app.device).eval()
        logger.info("SAM2 model loading complete")
        app.root.after(0, lambda: _on_sam2_load_complete(app, True))
    except Exception as e:
        logger.exception(f"SAM2 model loading failed: {e}")
        app.root.after(0, lambda: _on_sam2_load_complete(app, False, str(e)))


def _on_sam2_load_complete(app, success: bool, error_msg: str = None):
    app.sam2_loading_in_progress = False
    if success:
        app.view.update_sam2_loading_complete(success=True)
        app.update_status("SAM2 model loaded.")
        _transfer_sam3_masks_to_sam2(app)
    else:
        app.view.update_sam2_loading_complete(success=False)
        app.update_status(f"SAM2 loading error: {error_msg}")
        messagebox.showerror("SAM2 Error", f"SAM2 model loading failed:\n{error_msg}")


def unload_sam2_model(app):
    if app.sam2_model is not None:
        del app.sam2_model
        app.sam2_model = None
    if app.sam2_processor is not None:
        del app.sam2_processor
        app.sam2_processor = None
    app.sam2_masks.clear()
    app.sam2_prompt_points.clear()
    app.sam2_image_embeddings.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SAM2 model unloaded")
    app.update_status("SAM2 deactivated.")


def _transfer_sam3_masks_to_sam2(app):
    frame_idx = app.review_current_frame
    current_masks = {}
    if frame_idx in app.propagated_results:
        result = app.propagated_results[frame_idx]
        masks = result.get('masks', {})
        current_masks = {obj_id: data['last_mask'] for obj_id, data in masks.items() if 'last_mask' in data and data['last_mask'] is not None}
    logger.info(f"SAM2 activation - Transferring {len(current_masks)} masks from frame {frame_idx} to SAM2")
    app.tracked_objects.clear()
    if app.inference_session is not None:
        app.inference_session = None
    app.is_tracking_ever_started = False
    app.sam2_masks.clear()
    app.sam2_prompt_points.clear()
    if not current_masks:
        logger.info("SAM2 activation complete - No masks to transfer")
        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, {})
        return
    frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    image_inputs = app.sam2_processor(images=frame_pil, return_tensors="pt").to(app.device)
    original_sizes = image_inputs["original_sizes"]
    with torch.no_grad():
        with app.autocast_context:
            dummy_outputs = app.sam2_model(pixel_values=image_inputs["pixel_values"])
            image_embeddings = dummy_outputs.image_embeddings
    for obj_id, mask in current_masks.items():
        if not isinstance(mask, np.ndarray) or np.sum(mask > 0) == 0:
            continue
        try:
            mask_float = mask.astype(np.float32)
            mask_resized = cv2.resize(mask_float, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).to(app.device)
            with torch.no_grad():
                with app.autocast_context:
                    outputs = app.sam2_model(
                        input_masks=mask_tensor,
                        image_embeddings=image_embeddings,
                        multimask_output=False,
                    )
            pred_masks = app.sam2_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                original_sizes
            )[0]
            pred_mask = pred_masks[0, 0].numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            app.sam2_masks[obj_id] = pred_mask
            app.tracked_objects[obj_id] = {
                'last_mask': pred_mask.copy(),
                'custom_label': '',
                'points_for_reprompt': [],
                'initial_bbox_prompt': None,
            }
            logger.debug(f"SAM2 mask predict complete: obj_id={obj_id}")
        except Exception as e:
            logger.warning(f"SAM2 mask predict failed (obj_id={obj_id}): {e}")
            app.sam2_masks[obj_id] = mask.copy()
            app.tracked_objects[obj_id] = {
                'last_mask': mask.copy(),
                'custom_label': '',
                'points_for_reprompt': [],
                'initial_bbox_prompt': None,
            }
    if app.sam2_masks:
        app.next_obj_id_to_propose = max(app.sam2_masks.keys()) + 1
    logger.info(f"SAM2 activation complete - {len(app.sam2_masks)} masks predicted")
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def transfer_sam2_masks_to_sam3_and_unload(app):
    current_masks = app._get_current_masks_for_display()
    logger.info(f"SAM2 deactivation - Transferring {len(current_masks)} displayed masks to SAM3")
    app.sam2_masks.clear()
    app.sam2_prompt_points.clear()
    unload_sam2_model(app)
    app.tracked_objects.clear()
    if app.inference_session is not None:
        app.inference_session = None
    app.is_tracking_ever_started = False
    for obj_id, mask in current_masks.items():
        if isinstance(mask, np.ndarray):
            app.tracked_objects[obj_id] = {
                'last_mask': mask.copy(),
                'custom_label': '',
                'points_for_reprompt': [],
                'initial_bbox_prompt': None,
            }
    if app.tracked_objects:
        app.next_obj_id_to_propose = max(app.tracked_objects.keys()) + 1
    if current_masks:
        try:
            app._reinit_sam3_session_with_masks()
        except Exception as e:
            logger.exception(f"SAM3 session re-initialization failed: {e}")
    logger.info(f"SAM2 -> SAM3 transition complete - {len(current_masks)} objects")
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def _handle_sam2_prompt(app, prompt_type, coords, label=None,
                        proposed_obj_id_for_new=None, target_existing_obj_id=None,
                        custom_label=None):
    if app.sam2_model is None or app.sam2_processor is None:
        logger.error("SAM2 model not loaded.")
        return
    if app.current_frame_pil_rgb_original is None:
        logger.error("No current frame image.")
        return
    try:
        is_refinement = target_existing_obj_id is not None
        if is_refinement:
            obj_id = target_existing_obj_id
            logger.info(f"SAM2 object refinement mode: obj_id={obj_id}")
        else:
            obj_id = app.next_obj_id_to_propose
            app.next_obj_id_to_propose += 1
            logger.info(f"SAM2 new object creation: obj_id={obj_id}")
        logger.info(f"SAM2 prompt processing: type={prompt_type}, obj_id={obj_id}, coords={coords}")
        current_mask = app.sam2_masks.get(obj_id)
        input_points = None
        input_labels = None
        input_boxes = None
        if isinstance(coords, np.ndarray):
            coords_list = coords.flatten().tolist()
        elif coords is not None:
            coords_list = list(coords) if hasattr(coords, '__iter__') else [coords]
        else:
            coords_list = None
        if prompt_type in ("box", "bbox") and coords_list is not None:
            input_boxes = [[coords_list]]
            app.sam2_prompt_points[obj_id] = {'points': [], 'labels': []}
        elif prompt_type in ("point", "incremental_point") and coords_list is not None:
            new_point = (float(coords_list[0]), float(coords_list[1]))
            new_label = 1 if label else 0
            if obj_id not in app.sam2_prompt_points:
                app.sam2_prompt_points[obj_id] = {'points': [], 'labels': []}
            app.sam2_prompt_points[obj_id]['points'].append(new_point)
            app.sam2_prompt_points[obj_id]['labels'].append(new_label)
            all_points = app.sam2_prompt_points[obj_id]['points']
            all_labels = app.sam2_prompt_points[obj_id]['labels']
            points_list = [[list(p) for p in all_points]]
            labels_list = [all_labels]
            input_points = [points_list]
            input_labels = [labels_list]
            logger.info(f"SAM2 accumulated points: obj_id={obj_id}, total {len(all_points)} points (pos: {sum(all_labels)}, neg: {len(all_labels)-sum(all_labels)})")
        processor_kwargs = {
            "images": app.current_frame_pil_rgb_original,
            "return_tensors": "pt"
        }
        if input_points is not None:
            processor_kwargs["input_points"] = input_points
            processor_kwargs["input_labels"] = input_labels
        if input_boxes is not None:
            processor_kwargs["input_boxes"] = input_boxes
        inputs = app.sam2_processor(**processor_kwargs).to(app.device)
        with torch.no_grad():
            with app.autocast_context:
                if current_mask is not None:
                    mask_tensor = torch.from_numpy(current_mask).unsqueeze(0).unsqueeze(0).float().to(app.device)
                    outputs = app.sam2_model(**inputs, input_masks=mask_tensor, multimask_output=False)
                else:
                    outputs = app.sam2_model(**inputs, multimask_output=True)
        pred_masks = outputs.pred_masks.cpu()
        iou_scores = outputs.iou_scores.cpu() if hasattr(outputs, 'iou_scores') else None
        if iou_scores is not None and len(pred_masks.shape) > 2 and pred_masks.shape[2] > 1:
            best_idx = torch.argmax(iou_scores.squeeze())
            best_mask = pred_masks[0, 0, best_idx].numpy()
        else:
            best_mask = pred_masks[0, 0, 0].numpy()
        orig_h, orig_w = app.current_cv_frame.shape[:2]
        if best_mask.shape != (orig_h, orig_w):
            best_mask = cv2.resize(best_mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        best_mask = (best_mask > 0.5).astype(np.uint8)
        app.sam2_masks[obj_id] = best_mask
        if obj_id not in app.tracked_objects:
            app.tracked_objects[obj_id] = {
                'last_mask': best_mask.copy(),
                'custom_label': custom_label or '',
                'points_for_reprompt': [tuple(coords_list[:2])] if coords_list else [],
                'initial_bbox_prompt': coords_list if prompt_type in ("box", "bbox") else None,
            }
        else:
            app.tracked_objects[obj_id]['last_mask'] = best_mask.copy()
            if coords_list:
                if 'points_for_reprompt' not in app.tracked_objects[obj_id]:
                    app.tracked_objects[obj_id]['points_for_reprompt'] = []
                app.tracked_objects[obj_id]['points_for_reprompt'].append(tuple(coords_list[:2]))
        logger.info(f"SAM2 mask generation complete: obj_id={obj_id}, mask_shape={best_mask.shape}")
        app.update_status(f"SAM2 object {obj_id} segmentation complete")
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
        app._update_obj_id_info_label()
    except Exception as e:
        logger.exception(f"SAM2 prompt processing error: {e}")
        app.update_status(f"SAM2 error: {e}")


def refine_mask_with_sam2(app, obj_id: int, input_points=None, input_labels=None, input_boxes=None):
    if app.sam2_model is None or app.sam2_processor is None:
        logger.error("SAM2 model not loaded.")
        return None
    if app.current_frame_pil_rgb_original is None:
        logger.error("No current frame image.")
        return None
    try:
        current_mask = app.sam2_masks.get(obj_id)
        inputs = app.sam2_processor(
            images=app.current_frame_pil_rgb_original,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(app.device)
        with torch.no_grad():
            with app.autocast_context:
                if current_mask is not None:
                    mask_tensor = torch.from_numpy(current_mask).unsqueeze(0).unsqueeze(0).float().to(app.device)
                    outputs = app.sam2_model(**inputs, input_masks=mask_tensor, multimask_output=False)
                else:
                    outputs = app.sam2_model(**inputs, multimask_output=True)
        pred_masks = outputs.pred_masks.cpu()
        iou_scores = outputs.iou_scores.cpu() if hasattr(outputs, 'iou_scores') else None
        if iou_scores is not None and pred_masks.shape[2] > 1:
            best_idx = torch.argmax(iou_scores.squeeze())
            best_mask = pred_masks[0, 0, best_idx].numpy()
        else:
            best_mask = pred_masks[0, 0, 0].numpy()
        best_mask = (best_mask > 0.5).astype(np.uint8)
        app.sam2_masks[obj_id] = best_mask
        logger.info(f"SAM2 mask refinement complete: obj_id={obj_id}")
        return best_mask
    except Exception as e:
        logger.exception(f"SAM2 mask refinement error: {e}")
        return None
