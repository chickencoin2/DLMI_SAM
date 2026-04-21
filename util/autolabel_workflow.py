import base64
import io
import os
import json
import cv2
from PIL import Image
import torch
import numpy as np
import logging
import tkinter as tk

from .customutil import get_bbox_from_mask, process_sam_mask, is_bbox_on_edge, merge_contours_into_single_polygon

logger = logging.getLogger("DLMI_SAM_LABELER.AutoLabelWorkflow")


def _resolve_save_paths(app, frame_idx, label_subdir, label_ext, image_ext="jpg",
                        include_image=True):
    """Compute (save_dir, base_filename, label_filepath, image_filepath) for a
    frame save operation. Centralises the video_name + custom_path + batch
    subfolder + filename-template + overwrite/rename-counter logic that used
    to live inline in save_yolo_format and save_yolo_pose_format.

    - label_subdir: 'labels' / 'pose_labels' / '' (=> save_dir root for labelme).
    - label_ext:    'txt' / 'json'.
    - include_image: when True, also computes image_filepath inside
                     `<save_dir>/images/` and ensures that directory exists
                     (used by YOLO seg/pose save). When False, the returned
                     image_filepath is None.
    The function honours app.overwrite_policy == 'overwrite' (returns first
    candidate path) or 'rename' (increments `_vN` suffix until no collision).
    """
    video_name = "video"
    if isinstance(app.video_source_path, str):
        video_name = os.path.splitext(os.path.basename(app.video_source_path))[0]

    save_dir = app.AUTOLABEL_FOLDER_val
    final_base_filename = f"{video_name}_frame"

    if app.use_custom_save_path_var.get():
        base_dir = app.custom_save_dir_var.get()
        folder_template = app.custom_folder_name_var.get()
        file_template = app.custom_file_name_var.get()

        if app.batch_processing_mode_var.get() and app.batch_save_option_var.get() == "subfolder":
            folder_name = folder_template.format(video_name=video_name)
            save_dir = os.path.join(base_dir, folder_name)
        else:
            save_dir = base_dir

        if app.batch_processing_mode_var.get() and app.batch_filename_option_var.get() == "video_name":
            final_base_filename = f"{video_name}_frame"
        else:
            final_base_filename = file_template.format(video_name=video_name)

    labels_dir = os.path.join(save_dir, label_subdir) if label_subdir else save_dir
    os.makedirs(labels_dir, exist_ok=True)

    images_dir = None
    if include_image:
        images_dir = os.path.join(save_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    overwrite_policy = getattr(app, 'overwrite_policy', 'rename')
    counter = 0
    while True:
        suffix = f"_v{counter}" if counter > 0 else ""
        stem = f"{final_base_filename}_{frame_idx:05d}{suffix}"
        label_filepath = os.path.join(labels_dir, f"{stem}.{label_ext}")
        image_filepath = os.path.join(images_dir, f"{stem}.{image_ext}") if images_dir else None

        if overwrite_policy == "overwrite":
            break
        collision = os.path.exists(label_filepath) or (image_filepath and os.path.exists(image_filepath))
        if not collision:
            break
        if overwrite_policy == "rename":
            counter += 1
        else:
            logger.warning(f"Unexpected overwrite_policy={overwrite_policy!r}; using {label_filepath}")
            break

    return save_dir, final_base_filename, label_filepath, image_filepath


def save_yolo_format(app, frame_pil_image_rgb, frame_idx, masks_data_for_frame, base_filename_prefix):
    logger.debug(f"YOLO format save attempt. frame_idx: {frame_idx}")
    if not masks_data_for_frame: return

    h, w = frame_pil_image_rgb.height, frame_pil_image_rgb.width

    processed_masks = {}
    processed_group_ids = set()

    for sam_id, obj_data_original in masks_data_for_frame.items():
        group_id = app.sam_id_to_group.get(sam_id) if hasattr(app, 'sam_id_to_group') else None

        if group_id is not None:
            if group_id in processed_group_ids:
                continue
            processed_group_ids.add(group_id)

            merged_mask = app.get_group_merged_mask(group_id, masks_data_for_frame) if hasattr(app, 'get_group_merged_mask') else None
            if merged_mask is None:
                continue

            representative_id = min(app.object_groups.get(group_id, {sam_id})) if hasattr(app, 'object_groups') else sam_id
            rep_obj_data = app.tracked_objects.get(representative_id, {}) if hasattr(app, 'tracked_objects') else {}
            label = rep_obj_data.get('custom_label', app.default_object_label_var.get())

            processed_masks[representative_id] = {
                'last_mask': merged_mask,
                'custom_label': label
            }
            logger.debug(f"YOLO group {group_id} -> merged into object {representative_id}")
        else:
            processed_masks[sam_id] = obj_data_original.copy()

    masks_data_for_frame = processed_masks

    save_dir, final_base_filename, label_filepath, image_filepath = _resolve_save_paths(
        app, frame_idx, label_subdir="labels", label_ext="txt", include_image=True
    )

    try:
        frame_pil_image_rgb.save(image_filepath, "JPEG", quality=95)
    except Exception as e:
        logger.error(f"Image save failed {image_filepath}: {e}")
        return

    labeling_mode = app.labeling_mode_var.get()
    ignore_edge = app.ignore_edge_labels_var.get()
    edge_margin = app.edge_margin_var.get()

    class_name_to_idx = {name: idx for idx, name in enumerate(app.yolo_class_names_for_save)} if app.yolo_class_names_for_save else {}
    default_label = app.default_object_label_var.get()

    label_lines = []

    for sam_id, obj_data_original in masks_data_for_frame.items():
        if sam_id in app.suppressed_sam_ids:
            logger.debug(f"YOLO save skipped: SAM ID {sam_id} is in suppression list.")
            continue

        if sam_id in app.problematic_objects_flagged and \
           app.problematic_objects_flagged[sam_id] in ['pending_user_interaction', 'user_correcting', 'highlight_problem']:
            logger.info(f"YOLO save skipped: SAM ID {sam_id} is pending user correction/confirmation.")
            continue

        obj_data = obj_data_original.copy()
        mask = obj_data.get("last_mask")
        if mask is None or not mask.any(): continue

        if ignore_edge:
            bbox = get_bbox_from_mask(mask, min_bbox_area_val=1)
            if is_bbox_on_edge(bbox, (h, w), edge_margin):
                logger.debug(f"YOLO save skipped: SAM ID {sam_id} is an edge object.")
                continue

        filter_small_obj = getattr(app, 'filter_small_objects_var', None)
        if filter_small_obj and filter_small_obj.get():
            threshold_ratio = app.small_object_threshold_var.get()
            min_bbox_width = w * threshold_ratio
            bbox = get_bbox_from_mask(mask, min_bbox_area_val=1)
            if bbox is not None:
                bbox_width = bbox[2] - bbox[0]
                if bbox_width < min_bbox_width:
                    logger.debug(f"YOLO save skipped: SAM ID {sam_id} (width {bbox_width:.1f} < threshold {min_bbox_width:.1f})")
                    continue

        obj_label = obj_data.get("custom_label", default_label)
        class_idx = class_name_to_idx.get(obj_label)
        if class_idx is None:
            class_idx = 0
            logger.warning(f"Object {sam_id} label '{obj_label}' not in class list, saving as first class.")

        if labeling_mode == "Bounding Box":
            bbox = get_bbox_from_mask(mask, app.erosion_kernel_size.get(),
                                     app.erosion_iterations.get(), min_bbox_area_val=1)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                center_x = max(0.0, min(1.0, center_x))
                center_y = max(0.0, min(1.0, center_y))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                label_lines.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

        elif labeling_mode == "Instance":
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filter_small_contours = getattr(app, 'filter_small_contours_var', None)
            if filter_small_contours and filter_small_contours.get():
                threshold_ratio = app.small_contour_threshold_var.get()
                base_type = app.small_contour_base_var.get()

                if base_type == "image":
                    base_area = h * w
                else:
                    base_area = np.sum(mask)

                min_contour_area = base_area * threshold_ratio

                filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
                if len(filtered_contours) < len(contours):
                    logger.debug(f"Small contour filtering: {len(contours)} -> {len(filtered_contours)} (ObjID {sam_id})")
                contours = filtered_contours

            merged_polygon_contour = merge_contours_into_single_polygon(contours, min_area=10)

            if merged_polygon_contour is not None and len(merged_polygon_contour) >= 3:
                points = merged_polygon_contour.reshape(-1, 2).astype(np.float64)
                points[:, 0] /= w
                points[:, 1] /= h
                np.clip(points, 0.0, 1.0, out=points)
                normalized_points = points.flatten()

                points_str = ' '.join([f"{p:.6f}" for p in normalized_points])
                label_lines.append(f"{class_idx} {points_str}")

        elif labeling_mode == "Semantic":
            logger.warning(f"YOLO format does not support Semantic Segmentation. Object {sam_id} will be saved as BBox.")
            bbox = get_bbox_from_mask(mask, app.erosion_kernel_size.get(),
                                     app.erosion_iterations.get(), min_bbox_area_val=1)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                center_x = ((x1 + x2) / 2) / w
                center_y = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                center_x = max(0.0, min(1.0, center_x))
                center_y = max(0.0, min(1.0, center_y))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                label_lines.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    if label_lines:
        try:
            with open(label_filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(label_lines))
            logger.debug(f"YOLO label saved: {label_filepath}")
        except Exception as e:
            logger.error(f"YOLO label file save failed {label_filepath}: {e}")


def save_labelme_json(app, frame_pil_image_rgb, frame_idx, masks_data_for_frame, base_filename_prefix, is_both_mode=False):
    logger.debug(f"LabelMe JSON save attempt. frame_idx: {frame_idx}, both_mode: {is_both_mode}")
    if not masks_data_for_frame: return
    h, w = frame_pil_image_rgb.height, frame_pil_image_rgb.width

    processed_masks = {}
    processed_group_ids = set()

    for sam_id, obj_data_original in masks_data_for_frame.items():
        group_id = app.sam_id_to_group.get(sam_id) if hasattr(app, 'sam_id_to_group') else None

        if group_id is not None:
            if group_id in processed_group_ids:
                continue
            processed_group_ids.add(group_id)

            merged_mask = app.get_group_merged_mask(group_id, masks_data_for_frame) if hasattr(app, 'get_group_merged_mask') else None
            if merged_mask is None:
                continue

            representative_id = min(app.object_groups.get(group_id, {sam_id})) if hasattr(app, 'object_groups') else sam_id
            rep_obj_data = app.tracked_objects.get(representative_id, {}) if hasattr(app, 'tracked_objects') else {}
            label = rep_obj_data.get('custom_label', app.default_object_label_var.get())

            processed_masks[representative_id] = {
                'last_mask': merged_mask,
                'custom_label': label,
                'is_group': True
            }
            logger.debug(f"Group {group_id} -> merged into object {representative_id}")
        else:
            processed_masks[sam_id] = obj_data_original.copy()

    masks_data_for_frame = processed_masks

    actual_frame_idx = frame_idx

    video_name = "video"
    if isinstance(app.video_source_path, str):
        video_name = os.path.splitext(os.path.basename(app.video_source_path))[0]

    save_dir = app.AUTOLABEL_FOLDER_val
    final_base_filename = f"{video_name}_frame"

    if app.use_custom_save_path_var.get():
        base_dir = app.custom_save_dir_var.get()
        folder_template = app.custom_folder_name_var.get()
        file_template = app.custom_file_name_var.get()

        if app.batch_processing_mode_var.get() and app.batch_save_option_var.get() == "subfolder":
            folder_name = folder_template.format(video_name=video_name)
            save_dir = os.path.join(base_dir, folder_name)
        else:
            save_dir = base_dir

        if app.batch_processing_mode_var.get() and app.batch_filename_option_var.get() == "video_name":
            final_base_filename = f"{video_name}_frame"
        else:
            final_base_filename = file_template.format(video_name=video_name)

    if is_both_mode:
        save_dir = os.path.join(save_dir, "labelme")
        logger.debug(f"Both mode: LabelMe saves to {save_dir}")

    os.makedirs(save_dir, exist_ok=True)

    current_overwrite_policy = app.overwrite_policy

    counter = 0
    while True:
        suffix = f"_v{counter}" if counter > 0 else ""
        image_filename = f"{final_base_filename}_{actual_frame_idx:05d}{suffix}.jpg"
        json_filename = f"{final_base_filename}_{actual_frame_idx:05d}{suffix}.json"
        image_filepath = os.path.join(save_dir, image_filename)
        json_filepath = os.path.join(save_dir, json_filename)

        if current_overwrite_policy == "overwrite":
            break
        if not os.path.exists(image_filepath) and not os.path.exists(json_filepath):
            break
        if current_overwrite_policy == "rename":
            counter += 1
        else:
            logger.warning(f"Unexpected situation during filename duplication: {image_filename}, current policy: {current_overwrite_policy}")
            break

    try: frame_pil_image_rgb.save(image_filepath, "JPEG", quality=95)
    except Exception as e: logger.error(f"Image save failed {image_filepath}: {e}"); return

    labelme_data = {
        "version": app.LABELME_VERSION_val, "flags": {}, "shapes": [],
        "imagePath": image_filename, "imageData": None, "imageHeight": h, "imageWidth": w
    }
    labeling_mode = app.labeling_mode_var.get()
    ignore_edge = app.ignore_edge_labels_var.get()
    edge_margin = app.edge_margin_var.get()

    for sam_id, obj_data_original in masks_data_for_frame.items():
        if sam_id in app.suppressed_sam_ids:
            logger.debug(f"LabelMe save skipped: SAM ID {sam_id} is in suppression list.")
            continue

        if sam_id in app.problematic_objects_flagged and \
           app.problematic_objects_flagged[sam_id] in ['pending_user_interaction', 'user_correcting', 'highlight_problem']:
            logger.info(f"LabelMe save skipped: SAM ID {sam_id} is pending user correction/confirmation.")
            continue

        obj_data = obj_data_original.copy()
        mask = obj_data.get("last_mask")
        if mask is None or not mask.any(): continue

        if ignore_edge:
            bbox = get_bbox_from_mask(mask, min_bbox_area_val=1)
            if is_bbox_on_edge(bbox, (h, w), edge_margin):
                logger.debug(f"LabelMe save skipped: SAM ID {sam_id} is an edge object.")
                continue

        filter_small_obj = getattr(app, 'filter_small_objects_var', None)
        if filter_small_obj and filter_small_obj.get():
            threshold_ratio = app.small_object_threshold_var.get()
            min_bbox_width = w * threshold_ratio
            bbox = get_bbox_from_mask(mask, min_bbox_area_val=1)
            if bbox is not None:
                bbox_width = bbox[2] - bbox[0]
                if bbox_width < min_bbox_width:
                    logger.debug(f"LabelMe save skipped: SAM ID {sam_id} (width {bbox_width:.1f} < threshold {min_bbox_width:.1f})")
                    continue

        obj_label_raw = obj_data.get("custom_label", app.default_object_label_var.get())
        if not obj_label_raw or obj_label_raw.strip() == "" or obj_label_raw.lower() == "object":
            obj_label = f"Object_{sam_id}"
        else: obj_label = obj_label_raw

        shape_base = {
            "label": obj_label,
            "points": None,
            "group_id": None,
            "description": "",
            "shape_type": None,
            "flags": {}
        }

        if labeling_mode == "Bounding Box":
            bbox = get_bbox_from_mask(mask, app.erosion_kernel_size.get(), app.erosion_iterations.get(), min_bbox_area_val=1)
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox.tolist()
                shape = shape_base.copy(); shape["points"] = [[xmin, ymin], [xmax, ymax]]; shape["shape_type"] = "rectangle"
                labelme_data["shapes"].append(shape)
        elif labeling_mode == "Instance":
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filter_small_contours = getattr(app, 'filter_small_contours_var', None)
            if filter_small_contours and filter_small_contours.get():
                threshold_ratio = app.small_contour_threshold_var.get()
                base_type = app.small_contour_base_var.get()

                if base_type == "image":
                    base_area = h * w
                else:
                    base_area = np.sum(mask)

                min_contour_area = base_area * threshold_ratio

                filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
                if len(filtered_contours) < len(contours):
                    logger.debug(f"LabelMe small contour filtering: {len(contours)} -> {len(filtered_contours)} (ObjID {sam_id})")
                contours = filtered_contours

            merged_polygon_contour = merge_contours_into_single_polygon(contours, min_area=10)

            if merged_polygon_contour is not None and len(merged_polygon_contour) >= 3:
                points = merged_polygon_contour.reshape(-1, 2).tolist()
                shape_poly = shape_base.copy()
                shape_poly["points"] = points
                shape_poly["shape_type"] = "polygon"
                labelme_data["shapes"].append(shape_poly)
        elif labeling_mode == "Semantic":
            bbox = get_bbox_from_mask(mask, min_bbox_area_val=1)
            if bbox is not None:
                xmin, ymin, xmax, ymax = map(int, bbox)

                full_mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
                cropped_mask_pil = full_mask_pil.crop((xmin, ymin, xmax, ymax))

                buffer = io.BytesIO()
                cropped_mask_pil.save(buffer, format="PNG")
                mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                shape = shape_base.copy()
                shape["points"] = [bbox.tolist()[:2], bbox.tolist()[2:]]
                shape["shape_type"] = "mask"
                shape["mask"] = mask_base64
                labelme_data["shapes"].append(shape)

    if labelme_data["shapes"]:
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"LabelMe JSON saved: {json_filepath}")
        except Exception as e:
            logger.error(f"LabelMe JSON file save failed {json_filepath}: {e}")


def run_pcs_review_mode(app, progress_callback=None):
    from PIL import Image

    review_enabled_var = getattr(app, 'review_mode_enabled_var', None)
    if not review_enabled_var or not review_enabled_var.get():
        return

    positive_var = getattr(app, 'review_positive_prompts_var', None)
    negative_var = getattr(app, 'review_negative_prompts_var', None)

    positive_prompts = positive_var.get().strip() if positive_var else ""
    negative_prompts = negative_var.get().strip() if negative_var else ""

    if not positive_prompts and not negative_prompts:
        logger.info("Review mode: positive/negative prompts are empty.")
        return

    overlap_var = getattr(app, 'review_overlap_threshold_var', None)
    overlap_threshold = overlap_var.get() if overlap_var else 0.8

    consecutive_var = getattr(app, 'review_consecutive_frames_var', None)
    consecutive_threshold = consecutive_var.get() if consecutive_var else 3

    if not hasattr(app, 'propagated_results') or not app.propagated_results:
        logger.info("Review mode: No propagation results.")
        return

    logger.info(f"Review mode start: positive='{positive_prompts}', negative='{negative_prompts}'")
    logger.info(f"Review mode settings: overlap threshold={overlap_threshold}, consecutive frames threshold={consecutive_threshold}")

    if not hasattr(app, 'image_model') or app.image_model is None:
        logger.error("Review mode: SAM3 image model (image_model) not loaded.")
        return

    if not hasattr(app, 'image_processor') or app.image_processor is None:
        logger.error("Review mode: SAM3 image processor (image_processor) not loaded.")
        return

    positive_list = [p.strip() for p in positive_prompts.split(',') if p.strip()]
    negative_list = [n.strip() for n in negative_prompts.split(',') if n.strip()]

    logger.info(f"Review mode prompts - Positive ({len(positive_list)}): {positive_list}")
    logger.info(f"Review mode prompts - Negative ({len(negative_list)}): {negative_list}")

    negative_streak = {}

    sorted_frames = sorted(app.propagated_results.keys())
    total_frames = len(sorted_frames)

    for idx, frame_num in enumerate(sorted_frames):
        if progress_callback:
            progress_callback(idx + 1, total_frames)

        frame_data = app.propagated_results.get(frame_num, {})
        if not frame_data:
            continue

        frame_cv = frame_data.get('frame')
        if frame_cv is None:
            try:
                cap = getattr(app, 'cap', None)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame_cv = cap.read()
                    if not ret:
                        logger.warning(f"Review mode: Frame {frame_num} read failed")
                        continue
                else:
                    logger.warning("Review mode: Video capture object not available.")
                    continue
            except Exception as e:
                logger.warning(f"Review mode: Frame {frame_num} load failed: {e}")
                continue

        positive_masks = {}
        negative_masks = {}

        try:
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_h, frame_w = frame_cv.shape[:2]

            with torch.inference_mode():
                for i, prompt_text in enumerate(positive_list):
                    try:
                        inputs = app.image_processor(
                            images=frame_pil,
                            text=prompt_text,
                            return_tensors="pt"
                        ).to(app.device)

                        if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
                            if hasattr(inputs, 'pixel_values'):
                                inputs.pixel_values = inputs.pixel_values.to(dtype=torch.float32)

                        outputs = app.image_model(**inputs)

                        target_sizes = inputs.get("original_sizes")
                        if target_sizes is not None:
                            target_sizes = target_sizes.tolist()
                        else:
                            target_sizes = [[frame_h, frame_w]]

                        results = app.image_processor.post_process_instance_segmentation(
                            outputs,
                            threshold=getattr(app, 'pcs_detection_threshold_var', type('obj', (), {'get': lambda self: 0.5})()).get(),
                            mask_threshold=getattr(app, 'pcs_mask_threshold_var', type('obj', (), {'get': lambda self: 0.0})()).get(),
                            target_sizes=target_sizes
                        )[0]

                        if 'masks' in results and len(results['masks']) > 0:
                            masks_list = results['masks']
                            combined_mask = None
                            for mask_tensor in masks_list:
                                mask_np = mask_tensor.cpu().numpy().astype(bool)
                                if combined_mask is None:
                                    combined_mask = mask_np
                                else:
                                    combined_mask = combined_mask | mask_np
                            if combined_mask is not None:
                                positive_masks[f"pos_{i}_{prompt_text}"] = combined_mask
                                logger.debug(f"Review mode: positive '{prompt_text}' - {len(masks_list)} masks combined")
                    except Exception as e:
                        logger.debug(f"Review mode: positive prompt '{prompt_text}' detection failed: {e}")

                for i, prompt_text in enumerate(negative_list):
                    try:
                        inputs = app.image_processor(
                            images=frame_pil,
                            text=prompt_text,
                            return_tensors="pt"
                        ).to(app.device)

                        if hasattr(app, 'model_dtype') and app.model_dtype == torch.float32:
                            if hasattr(inputs, 'pixel_values'):
                                inputs.pixel_values = inputs.pixel_values.to(dtype=torch.float32)

                        outputs = app.image_model(**inputs)

                        target_sizes = inputs.get("original_sizes")
                        if target_sizes is not None:
                            target_sizes = target_sizes.tolist()
                        else:
                            target_sizes = [[frame_h, frame_w]]

                        results = app.image_processor.post_process_instance_segmentation(
                            outputs,
                            threshold=getattr(app, 'pcs_detection_threshold_var', type('obj', (), {'get': lambda self: 0.5})()).get(),
                            mask_threshold=getattr(app, 'pcs_mask_threshold_var', type('obj', (), {'get': lambda self: 0.0})()).get(),
                            target_sizes=target_sizes
                        )[0]

                        if 'masks' in results and len(results['masks']) > 0:
                            masks_list = results['masks']
                            combined_mask = None
                            for mask_tensor in masks_list:
                                mask_np = mask_tensor.cpu().numpy().astype(bool)
                                if combined_mask is None:
                                    combined_mask = mask_np
                                else:
                                    combined_mask = combined_mask | mask_np
                            if combined_mask is not None:
                                negative_masks[f"neg_{i}_{prompt_text}"] = combined_mask
                                logger.debug(f"Review mode: negative '{prompt_text}' - {len(masks_list)} masks combined")
                    except Exception as e:
                        logger.debug(f"Review mode: negative prompt '{prompt_text}' detection failed: {e}")

        except Exception as e:
            logger.warning(f"Review mode: Frame {frame_num} PCS detection failed: {e}")
            continue

        combined_positive_mask = None
        for pos_key, pos_mask in positive_masks.items():
            if combined_positive_mask is None:
                combined_positive_mask = pos_mask.copy()
            else:
                if pos_mask.shape != combined_positive_mask.shape:
                    if pos_mask.ndim == 2 and combined_positive_mask.ndim == 2:
                        pos_mask = cv2.resize(pos_mask.astype(np.uint8),
                                             (combined_positive_mask.shape[1], combined_positive_mask.shape[0]),
                                             interpolation=cv2.INTER_NEAREST).astype(bool)
                combined_positive_mask = combined_positive_mask | pos_mask

        combined_negative_mask = None
        for neg_key, neg_mask in negative_masks.items():
            if combined_negative_mask is None:
                combined_negative_mask = neg_mask.copy()
            else:
                if neg_mask.shape != combined_negative_mask.shape:
                    if neg_mask.ndim == 2 and combined_negative_mask.ndim == 2:
                        neg_mask = cv2.resize(neg_mask.astype(np.uint8),
                                             (combined_negative_mask.shape[1], combined_negative_mask.shape[0]),
                                             interpolation=cv2.INTER_NEAREST).astype(bool)
                combined_negative_mask = combined_negative_mask | neg_mask

        masks_data = frame_data.get('masks', {})
        for sam_id, obj_data in masks_data.items():
            tracked_mask = obj_data.get("last_mask")
            if tracked_mask is None or not tracked_mask.any():
                continue

            if tracked_mask.dtype != bool:
                tracked_mask = tracked_mask.astype(bool) if tracked_mask.dtype == np.uint8 else (tracked_mask > 0.5).astype(bool)

            if sam_id in app.suppressed_sam_ids:
                continue

            tracked_pixels = tracked_mask.sum()
            if tracked_pixels == 0:
                continue

            overlaps_positive = False
            positive_overlap_ratio = 0.0
            if combined_positive_mask is not None:
                if combined_positive_mask.shape != tracked_mask.shape:
                    if combined_positive_mask.ndim == 2 and tracked_mask.ndim == 2:
                        combined_positive_resized = cv2.resize(combined_positive_mask.astype(np.uint8),
                                                               (tracked_mask.shape[1], tracked_mask.shape[0]),
                                                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        combined_positive_resized = None
                else:
                    combined_positive_resized = combined_positive_mask

                if combined_positive_resized is not None:
                    intersection = np.sum(tracked_mask & combined_positive_resized)
                    positive_overlap_ratio = intersection / tracked_pixels
                    if positive_overlap_ratio >= overlap_threshold:
                        overlaps_positive = True

            overlaps_negative = False
            negative_overlap_ratio = 0.0
            if combined_negative_mask is not None:
                if combined_negative_mask.shape != tracked_mask.shape:
                    if combined_negative_mask.ndim == 2 and tracked_mask.ndim == 2:
                        combined_negative_resized = cv2.resize(combined_negative_mask.astype(np.uint8),
                                                               (tracked_mask.shape[1], tracked_mask.shape[0]),
                                                               interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        combined_negative_resized = None
                else:
                    combined_negative_resized = combined_negative_mask

                if combined_negative_resized is not None:
                    intersection = np.sum(tracked_mask & combined_negative_resized)
                    negative_overlap_ratio = intersection / tracked_pixels
                    if negative_overlap_ratio >= overlap_threshold:
                        overlaps_negative = True

            if overlaps_positive:
                if sam_id in negative_streak:
                    negative_streak[sam_id] = []
                logger.debug(f"[Review mode] Frame {frame_num}, Object {sam_id}: positive overlap {positive_overlap_ratio*100:.1f}% -> normal")
            elif overlaps_negative:
                if sam_id not in negative_streak:
                    negative_streak[sam_id] = []
                negative_streak[sam_id].append(frame_num)
                logger.debug(f"[Review mode] Frame {frame_num}, Object {sam_id}: negative overlap {negative_overlap_ratio*100:.1f}% -> error candidate")

                if len(negative_streak[sam_id]) >= consecutive_threshold:
                    for error_frame in negative_streak[sam_id]:
                        logger.warning(f"[Review mode] Frame {error_frame}, Object {sam_id}: negative overlap {negative_overlap_ratio*100:.1f}% (consecutive {len(negative_streak[sam_id])} frames)")

            else:
                if sam_id in negative_streak:
                    negative_streak[sam_id] = []

    logger.info(f"Review mode complete: {total_frames} frames inspected")


def parse_labelme_json(json_path, frame_width, frame_height):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects = []
    shapes = data.get('shapes', [])

    json_width = data.get('imageWidth', 0)
    json_height = data.get('imageHeight', 0)

    scale_x = frame_width / json_width if json_width > 0 else 1.0
    scale_y = frame_height / json_height if json_height > 0 else 1.0

    for shape in shapes:
        label = shape.get('label', 'object')
        shape_type = shape.get('shape_type', '')
        points = shape.get('points', [])

        if not points:
            continue

        obj = {'label': label, 'bbox': None, 'polygon': None}

        if shape_type == 'rectangle':
            if len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                obj['bbox'] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

        elif shape_type == 'polygon':
            scaled_points = [[p[0] * scale_x, p[1] * scale_y] for p in points]
            obj['polygon'] = scaled_points

            xs = [p[0] for p in scaled_points]
            ys = [p[1] for p in scaled_points]
            obj['bbox'] = [min(xs), min(ys), max(xs), max(ys)]

        if obj['bbox'] is not None:
            objects.append(obj)

    return objects


def parse_yolo_txt(txt_path, frame_width, frame_height):
    objects = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = parts[0]
        values = [float(v) for v in parts[1:]]

        obj = {'label': str(class_id), 'bbox': None, 'polygon': None}

        if len(values) == 4:
            x_center, y_center, w, h = values

            x_center *= frame_width
            y_center *= frame_height
            w *= frame_width
            h *= frame_height

            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            obj['bbox'] = [x1, y1, x2, y2]

        elif len(values) >= 6 and len(values) % 2 == 0:
            polygon = []
            for i in range(0, len(values), 2):
                x = values[i] * frame_width
                y = values[i + 1] * frame_height
                polygon.append([x, y])

            obj['polygon'] = polygon

            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            obj['bbox'] = [min(xs), min(ys), max(xs), max(ys)]

        if obj['bbox'] is not None:
            objects.append(obj)

    return objects


def save_yolo_pose_format(app, frame_pil_image_rgb, frame_idx, masks_data_for_frame, base_filename_prefix,
                          separate_subdir=None):
    """Save YOLO-pose format labels. Each line:
        class_id cx cy w h x1 y1 v1 x2 y2 v2 ... xN yN vN
    where all coords are normalized [0,1] and v in {0,1}.

    Only objects with non-empty 'pose_points' are emitted. If the object also
    has 'last_mask', the bbox is derived from that mask; else the bbox is the
    bounding rectangle of the keypoints themselves.

    When `separate_subdir` is provided (e.g., 'pose_labels'), writes into a
    sibling folder under the YOLO dataset root. Otherwise writes alongside
    the normal labels/.
    """
    if not masks_data_for_frame:
        return
    has_any_pose = any(
        isinstance(d, dict) and d.get('pose_points') for d in masks_data_for_frame.values()
    )
    if not has_any_pose:
        return

    h, w = frame_pil_image_rgb.height, frame_pil_image_rgb.width

    labels_subdir = separate_subdir or "labels"
    # Pose labels share their YOLO dataset root with seg labels but land in a
    # different subdir. We don't re-emit the image file; the seg save already
    # wrote it. include_image=False skips the image collision check and dir.
    _save_dir, _base, label_filepath, _img = _resolve_save_paths(
        app, frame_idx, label_subdir=labels_subdir, label_ext="txt", include_image=False
    )

    class_name_to_idx = {name: idx for idx, name in enumerate(app.yolo_class_names_for_save)} if app.yolo_class_names_for_save else {}
    default_label = app.default_object_label_var.get()

    lines = []
    seen_groups = set()
    for sam_id, obj_data in masks_data_for_frame.items():
        if not isinstance(obj_data, dict):
            continue

        group_id = app.sam_id_to_group.get(sam_id) if hasattr(app, 'sam_id_to_group') else None
        if group_id is not None:
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            members = app.object_groups.get(group_id, {sam_id}) if hasattr(app, 'object_groups') else {sam_id}
            pose_pts = []
            merged_mask = None
            rep_label = None
            for m_id in members:
                m_data = masks_data_for_frame.get(m_id)
                if not isinstance(m_data, dict):
                    continue
                if rep_label is None:
                    rep_label = m_data.get('custom_label')
                if not pose_pts and m_data.get('pose_points'):
                    pose_pts = m_data.get('pose_points')
                mm = m_data.get('last_mask')
                if mm is not None and mm.any():
                    if merged_mask is None:
                        merged_mask = mm.astype(bool).copy()
                    else:
                        merged_mask = merged_mask | mm.astype(bool)
            if not pose_pts:
                continue
            obj_label = rep_label or default_label
            obj_mask = merged_mask
        else:
            pose_pts = obj_data.get('pose_points')
            if not pose_pts:
                continue
            obj_label = obj_data.get('custom_label', default_label)
            obj_mask = obj_data.get('last_mask')

        if obj_mask is not None and obj_mask.any():
            bb = get_bbox_from_mask(obj_mask, min_bbox_area_val=1)
            if bb is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
        else:
            xs = [p['x'] for p in pose_pts if p.get('visibility', 1) > 0]
            ys = [p['y'] for p in pose_pts if p.get('visibility', 1) > 0]
            if not xs or not ys:
                continue
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            pad = 4
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cx = max(0.0, min(1.0, cx)); cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw)); bh = max(0.0, min(1.0, bh))

        class_idx = class_name_to_idx.get(obj_label, 0)

        parts = [str(class_idx), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for p in pose_pts:
            nx = max(0.0, min(1.0, p['x'] / w))
            ny = max(0.0, min(1.0, p['y'] / h))
            v = int(p.get('visibility', 2))
            if v < 0:
                v = 0
            elif v > 2:
                v = 2
            parts.extend([f"{nx:.6f}", f"{ny:.6f}", str(v)])
        lines.append(" ".join(parts))

    if not lines:
        return
    try:
        with open(label_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        logger.debug(f"YOLO pose saved: {label_filepath}")
    except Exception as e:
        logger.error(f"YOLO pose save failed {label_filepath}: {e}")


def save_frame_dispatch(app, frame_pil, frame_idx, masks, video_name, pose_subdir=None):
    """Dispatch save for one frame across all requested output formats.

    Consolidates the three-way dispatch block (YOLO seg + YOLO pose + labelme)
    that used to be repeated at 3 call sites in app.py (cut_and_repropagate,
    _cut_save_frames_0_to_n, _save_labels_thread). The dispatch honours
    `app.save_format_var.get()` which is 'yolo' | 'labelme' | 'both'. Pose
    labels are always written to a separate subdir (default 'pose_labels')
    so they never collide with YOLO-seg label files.
    """
    fmt = app.save_format_var.get()
    if fmt in ("yolo", "both"):
        save_yolo_format(app, frame_pil, frame_idx, masks, video_name)
        save_yolo_pose_format(
            app, frame_pil, frame_idx, masks, video_name,
            separate_subdir=(pose_subdir or 'pose_labels')
        )
    if fmt in ("labelme", "both"):
        save_labelme_json(
            app, frame_pil, frame_idx, masks, video_name,
            is_both_mode=(fmt == "both")
        )
