"""
Display / rendering functions extracted from app.py.

Each former method is now a module-level function whose first parameter is
``app`` (the application instance that was previously ``self``).
"""

import math
import numpy as np
import cv2
import logging
from PIL import Image, ImageDraw, ImageFont, ImageTk

from .customutil import get_bbox_from_mask, draw_star_marker

logger = logging.getLogger("DLMI_SAM_LABELER.FrameRenderer")

# Fallback alpha value (mirrors the module-level constant in app.py)
ALPHA_NORMAL = 153


def _display_cv_frame_on_view(app, frame_bgr, masks_to_overlay=None, yolo_bboxes_to_draw=None):
    if frame_bgr is None: return
    try:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        app.current_frame_pil_rgb_original = Image.fromarray(frame_rgb)

        img_arr = np.array(app.current_frame_pil_rgb_original.convert("RGBA"))
        h, w = img_arr.shape[:2]

        bbox_cache = {}
        special_focus_objs = []

        if masks_to_overlay:
            erosion_k = app.erosion_kernel_size.get()
            erosion_i = app.erosion_iterations.get()
            base_alpha = app.mask_alpha_var.get() if hasattr(app, 'mask_alpha_var') else ALPHA_NORMAL

            group_bbox_cache = {}
            processed_groups = set()
            for obj_id, mask_array_bool in masks_to_overlay.items():
                if mask_array_bool is None: continue

                if mask_array_bool.dtype != bool:
                    mask_array_bool = mask_array_bool > 0.5
                if mask_array_bool.shape[0] != h or mask_array_bool.shape[1] != w:
                    continue

                bbox_cache[obj_id] = get_bbox_from_mask(mask_array_bool, erosion_k, erosion_i, 1)
                group_id = app.sam_id_to_group.get(obj_id)
                if group_id is not None:
                    first_member = min(app.object_groups.get(group_id, {obj_id}))
                    rgb_color = app._get_object_color(first_member)
                    current_bbox = bbox_cache[obj_id]
                    if current_bbox is not None:
                        if group_id not in group_bbox_cache:
                            group_bbox_cache[group_id] = list(current_bbox)
                        else:
                            x1, y1, x2, y2 = group_bbox_cache[group_id]
                            nx1, ny1, nx2, ny2 = current_bbox
                            group_bbox_cache[group_id] = [min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2)]
                else:
                    rgb_color = app._get_object_color(obj_id)

                alpha = base_alpha
                is_multi_selected = obj_id in app.selected_objects_sam_ids

                is_group_selected = False
                if group_id is not None:
                    group_members = app.object_groups.get(group_id, set())
                    is_group_selected = any(
                        m in app.selected_objects_sam_ids or m == app.selected_object_sam_id
                        for m in group_members
                    )

                is_special = (is_multi_selected or is_group_selected or
                              obj_id == app.selected_object_sam_id or
                              obj_id == app.interaction_correction_pending or
                              obj_id == app.problematic_highlight_active_sam_id or
                              obj_id == app.reassign_bbox_mode_active_sam_id)

                if is_special:
                    alpha = min(255, alpha + 50)
                    special_focus_objs.append(obj_id)
                if obj_id == app.problematic_highlight_active_sam_id:
                    alpha = max(30, alpha - 50)
                elif obj_id == app.interaction_correction_pending or obj_id == app.reassign_bbox_mode_active_sam_id:
                    alpha = max(20, alpha - 80)

                alpha_ratio = alpha / 255.0
                inv_alpha = 1.0 - alpha_ratio
                color_arr = np.array(rgb_color, dtype=np.float32)
                img_arr[mask_array_bool, :3] = (
                    img_arr[mask_array_bool, :3] * inv_alpha + color_arr * alpha_ratio
                ).astype(np.uint8)

        pil_image_to_draw_on = Image.fromarray(img_arr, "RGBA")
        draw_final = ImageDraw.Draw(pil_image_to_draw_on)

        drawn_group_ids = set()
        for obj_id in special_focus_objs:
            group_id = app.sam_id_to_group.get(obj_id)

            if group_id is not None:
                if group_id in drawn_group_ids:
                    continue
                drawn_group_ids.add(group_id)
                bbox = group_bbox_cache.get(group_id) if 'group_bbox_cache' in dir() else bbox_cache.get(obj_id)
            else:
                bbox = bbox_cache.get(obj_id)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x1 < x2 and y1 < y2:
                    draw_final.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

        if masks_to_overlay:
            labeled_groups = set()
            for obj_id in masks_to_overlay.keys():
                group_id = app.sam_id_to_group.get(obj_id)

                if group_id is not None:
                    if group_id in labeled_groups:
                        continue
                    labeled_groups.add(group_id)
                    first_member = min(app.object_groups.get(group_id, {obj_id}))
                    obj_data = app.tracked_objects.get(first_member, {})
                    member_count = len(app.object_groups.get(group_id, set()))
                    display_label = f"Group-{group_id} ({member_count}): {obj_data.get('custom_label', 'object')}"
                    bbox_for_label = group_bbox_cache.get(group_id) if 'group_bbox_cache' in dir() else bbox_cache.get(obj_id)
                else:
                    obj_data = app.tracked_objects.get(obj_id, {})
                    display_label = obj_data.get("custom_label", f"Obj-{obj_id}")
                    if obj_id == app.reassign_bbox_mode_active_sam_id: display_label += " (BBox Reassign)"
                    elif obj_id == app.problematic_highlight_active_sam_id: display_label += " (Check Required!)"
                    elif obj_id == app.interaction_correction_pending: display_label += " (Auto Correction...)"
                    bbox_for_label = bbox_cache.get(obj_id)

                if bbox_for_label is not None:
                    x1, y1, x2, y2 = bbox_for_label
                    diagonal = math.sqrt(w**2 + h**2)
                    font_size_percent = app.label_font_size_percent_var.get()
                    dynamic_font_size = max(8, int(diagonal * font_size_percent / 100))
                    dynamic_font = None
                    font_paths = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                        "arial.ttf",
                        "/System/Library/Fonts/Helvetica.ttc"
                    ]
                    for font_path in font_paths:
                        try:
                            dynamic_font = ImageFont.truetype(font_path, dynamic_font_size)
                            break
                        except:
                            continue
                    if dynamic_font is None:
                        dynamic_font = ImageFont.load_default()
                    text_pos = (int(x1 + 5), int(y1 - dynamic_font_size - 5 if y1 > dynamic_font_size + 5 else y1 + 5))
                    try: draw_final.text(text_pos, display_label, fill="yellow", font=dynamic_font)
                    except Exception: pass
                    if obj_id in (app.problematic_highlight_active_sam_id, app.interaction_correction_pending, app.reassign_bbox_mode_active_sam_id):
                        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        marker_base_size = min(w, h) / 80
                        marker_color = "orange" if obj_id in (app.interaction_correction_pending, app.reassign_bbox_mode_active_sam_id) else "red"
                        marker_size = marker_base_size * 0.7 if marker_color == "orange" else marker_base_size
                        draw_star_marker(draw_final, center_x, center_y, marker_size, color=marker_color)

        if getattr(app, 'polygon_mode_active', False):
            polygon_color = (0, 255, 255)
            diagonal = math.sqrt(w**2 + h**2)
            point_size_percent = app.polygon_point_size_percent_var.get()
            point_radius = max(2, int(diagonal * point_size_percent / 100))

            if hasattr(app, 'polygon_points') and app.polygon_points:
                for i, (px, py) in enumerate(app.polygon_points):
                    draw_final.ellipse(
                        [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                        fill="cyan", outline="white"
                    )
                    draw_final.text((px + point_radius + 2, py - point_radius), str(i + 1), fill="yellow")

                if len(app.polygon_points) >= 2:
                    for i in range(len(app.polygon_points) - 1):
                        p1 = app.polygon_points[i]
                        p2 = app.polygon_points[i + 1]
                        draw_final.line([p1[0], p1[1], p2[0], p2[1]], fill="cyan", width=2)

            if hasattr(app, 'polygon_objects') and app.polygon_objects:
                for obj_idx, poly_obj in enumerate(app.polygon_objects):
                    points = poly_obj.get('points', [])
                    if len(points) >= 3:
                        flat_points = [(p[0], p[1]) for p in points]
                        draw_final.polygon(flat_points, outline="lime", width=2)
                        first_pt = points[0]
                        draw_final.text((first_pt[0] + 5, first_pt[1] - 15), f"Poly-{obj_idx + 1}", fill="lime")

        if (hasattr(app, 'show_prompt_visualization_var') and app.show_prompt_visualization_var.get()
            and hasattr(app, 'object_prompt_history') and app.object_prompt_history):
            diagonal = math.sqrt(w**2 + h**2)
            point_size_percent = app.polygon_point_size_percent_var.get()
            line_width = max(1, int(diagonal * point_size_percent / 100))
            point_radius = max(2, int(diagonal * point_size_percent / 100))

            show_per_object = hasattr(app, 'show_prompt_per_object_var') and app.show_prompt_per_object_var.get()
            selected_id = getattr(app, 'selected_object_sam_id', None)

            for obj_id, prompts in app.object_prompt_history.items():
                if show_per_object and selected_id is not None and obj_id != selected_id:
                    continue
                for box in prompts.get('boxes', []):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    draw_final.rectangle([x1, y1, x2, y2], outline="black", width=line_width)

                for pt in prompts.get('positive_points', []):
                    px, py = int(pt[0]), int(pt[1])
                    draw_final.ellipse(
                        [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                        fill="lime", outline="white"
                    )

                for pt in prompts.get('negative_points', []):
                    px, py = int(pt[0]), int(pt[1])
                    draw_final.ellipse(
                        [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                        fill="red", outline="white"
                    )

                for contour in prompts.get('mask_contours', []):
                    if len(contour) >= 2:
                        flat_pts = [(int(p[0][0]), int(p[0][1])) for p in contour if len(p) > 0]
                        if len(flat_pts) >= 2:
                            draw_final.line(flat_pts + [flat_pts[0]], fill="lime", width=line_width)

                for box in prompts.get('exemplar_positive', []):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    draw_final.rectangle([x1, y1, x2, y2], outline="lime", width=max(1, line_width // 2))

                for box in prompts.get('exemplar_negative', []):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    draw_final.rectangle([x1, y1, x2, y2], outline="red", width=max(1, line_width // 2))

        if yolo_bboxes_to_draw:
            for yolo_id, data in yolo_bboxes_to_draw.items():
                if data["bbox"] is None: continue
                x1, y1, x2, y2 = data["bbox"]
                label = f"Y-{yolo_id}: {data['class_name']}"
                draw_final.rectangle([x1, y1, x2, y2], outline="cyan", width=1)
                draw_final.text((x1, y1 - 10 if y1 > 10 else y1 + 2), label, fill="cyan", font=app.label_font)

        if (hasattr(app, 'show_object_border_var') and app.show_object_border_var.get() and masks_to_overlay):
            border_color = (0, 0, 0, 255)
            for obj_id, mask_arr in masks_to_overlay.items():
                if mask_arr is None or not mask_arr.any(): continue
                mask_uint8 = (mask_arr.astype(np.uint8) if mask_arr.dtype == bool else (mask_arr > 0.5).astype(np.uint8)) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        points = [tuple(pt[0]) for pt in contour]
                        if len(points) >= 3:
                            draw_final.polygon(points, outline=border_color, width=2)

        if app.view: app.view.display_image(pil_image_to_draw_on)
    except Exception as e: logger.exception(f"_display_cv_frame_on_view error: {e}")


def _get_current_masks_for_display(app):
    current_keys = list(app.tracked_objects.keys())
    masks = {}

    filter_enabled = app.filter_small_objects_var.get()
    threshold_ratio = app.small_object_threshold_var.get()

    min_bbox_width = 0
    if filter_enabled and app.current_cv_frame is not None:
        frame_width = app.current_cv_frame.shape[1]
        min_bbox_width = frame_width * threshold_ratio

    for obj_id in current_keys:
        data = app.tracked_objects.get(obj_id)
        if data and "last_mask" in data and data["last_mask"] is not None:
            mask = data["last_mask"]

            if filter_enabled and min_bbox_width > 0:
                bbox = get_bbox_from_mask(mask, min_bbox_area_val=1)
                if bbox is not None:
                    bbox_width = bbox[2] - bbox[0]
                    if bbox_width < min_bbox_width:
                        logger.debug(f"Small object filter: ObjID {obj_id} (width {bbox_width:.1f} < threshold {min_bbox_width:.1f})")
                        continue

            masks[obj_id] = mask
    return masks
