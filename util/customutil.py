import numpy as np
import cv2
import logging
import torch
from PIL import Image

logger = logging.getLogger("DLMI_SAM_LABELER.CustomUtil")

def rgb_to_tkinter_hex(rgb_tuple):
    return f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"

def get_bbox_from_mask(mask_hw_bool,
                       erosion_kernel_size_val=0,
                       erosion_iterations_val=0,
                       min_bbox_area_val=1):
    if mask_hw_bool is None or not mask_hw_bool.any():
        return None

    eroded_mask_uint8 = mask_hw_bool.astype(np.uint8)
    if erosion_kernel_size_val > 0 and erosion_iterations_val > 0:
        kernel = np.ones((erosion_kernel_size_val, erosion_kernel_size_val), np.uint8)
        eroded_mask_uint8 = cv2.erode(eroded_mask_uint8, kernel, iterations=erosion_iterations_val)

    if not eroded_mask_uint8.any():
        logger.debug("Mask disappeared after erosion operation.")
        return None

    rows = np.any(eroded_mask_uint8, axis=1)
    cols = np.any(eroded_mask_uint8, axis=0)

    if not rows.any() or not cols.any():
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    if width * height < min_bbox_area_val:
        logger.debug(f"BBox area ({width*height}) is too small, ignored (threshold: {min_bbox_area_val}).")
        return None

    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

def calculate_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea <= 0 or boxBArea <= 0 :
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def get_stabilized_bbox(history_bboxes, current_bbox_xyxy, stabilize_bbox_history_size):
    if current_bbox_xyxy is None:
        return None

    history_bboxes.append(current_bbox_xyxy.copy())
    if len(history_bboxes) > stabilize_bbox_history_size:
        history_bboxes.pop(0)

    if not history_bboxes:
        return current_bbox_xyxy.copy()

    valid_boxes = [box for box in history_bboxes if box is not None and len(box) == 4]
    if not valid_boxes:
        return current_bbox_xyxy.copy()

    widths = [box[2] - box[0] for box in valid_boxes]
    heights = [box[3] - box[1] for box in valid_boxes]

    if not widths or not heights:
        return current_bbox_xyxy.copy()

    median_width = np.median(widths)
    median_height = np.median(heights)

    center_x = (current_bbox_xyxy[0] + current_bbox_xyxy[2]) / 2
    center_y = (current_bbox_xyxy[1] + current_bbox_xyxy[3]) / 2

    stable_x1 = center_x - median_width / 2
    stable_y1 = center_y - median_height / 2
    stable_x2 = center_x + median_width / 2
    stable_y2 = center_y + median_height / 2

    return np.array([stable_x1, stable_y1, stable_x2, stable_y2], dtype=np.float32)

def process_sam_mask(mask_from_sam_np, target_pil_size_wh,
                     apply_closing=False, closing_kernel_size=3):
    if not isinstance(mask_from_sam_np, np.ndarray):
        return None

    squeezed_mask = np.squeeze(mask_from_sam_np)
    if squeezed_mask.ndim != 2:
        if squeezed_mask.ndim > 2 and squeezed_mask.shape[0] > 0:
            squeezed_mask = squeezed_mask[0]
        else:
            return None

    if squeezed_mask.ndim != 2:
        return None

    if squeezed_mask.dtype != bool:
        squeezed_mask = squeezed_mask > 0.0

    target_w, target_h = target_pil_size_wh

    if squeezed_mask.shape[0] != target_h or squeezed_mask.shape[1] != target_w:
        try:
            squeezed_mask_uint8 = (squeezed_mask.astype(np.uint8) * 255)
            resized_mask_uint8 = cv2.resize(squeezed_mask_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            squeezed_mask = resized_mask_uint8 > 0
        except Exception as e:
            logger.warning(f"Mask resizing failed: {e}")
            return None

    if apply_closing and closing_kernel_size > 0:
        kernel_size_odd = closing_kernel_size if closing_kernel_size % 2 != 0 else closing_kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_odd, kernel_size_odd))
        mask_uint8_for_closing = squeezed_mask.astype(np.uint8) * 255
        closed_mask_uint8 = cv2.morphologyEx(mask_uint8_for_closing, cv2.MORPH_CLOSE, kernel)
        squeezed_mask = closed_mask_uint8 > 0

    return squeezed_mask

def draw_star_marker(draw_context, center_x, center_y, size, color="yellow", outline_color="black", outline_width=1):
    points = []
    for i in range(5):
        angle_rad = np.pi / 2 - (2 * np.pi * i / 5)
        outer_x = center_x + size * np.cos(angle_rad)
        outer_y = center_y - size * np.sin(angle_rad)
        points.append((outer_x, outer_y))
        angle_rad_inner = angle_rad - np.pi / 5
        inner_x = center_x + (size / 2.5) * np.cos(angle_rad_inner)
        inner_y = center_y - (size / 2.5) * np.sin(angle_rad_inner)
        points.append((inner_x, inner_y))

    if outline_width > 0 and outline_color:
        draw_context.polygon(points, fill=color, outline=outline_color, width=outline_width)
    else:
        draw_context.polygon(points, fill=color)

def get_hashable_obj_id(obj_id_from_sam):
    if isinstance(obj_id_from_sam, torch.Tensor):
        return int(obj_id_from_sam.item())
    if isinstance(obj_id_from_sam, (int, float, np.integer, np.floating)):
        return int(obj_id_from_sam)
    logger.warning(f"Unexpected type for obj_id_from_sam: {type(obj_id_from_sam)}. Using hash of string.")
    return hash(str(obj_id_from_sam)) % 100000

def is_bbox_on_edge(bbox, image_shape, margin):
    if bbox is None:
        return False

    img_h, img_w = image_shape[:2]
    xmin, ymin, xmax, ymax = bbox

    if (xmin <= margin) or (ymin <= margin) or \
       (xmax >= img_w - margin) or (ymax >= img_h - margin):
        return True

    return False

def merge_contours_into_single_polygon(contours, min_area=10):
    from scipy.spatial.distance import cdist

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None
    if len(valid_contours) == 1:
        return valid_contours[0]

    valid_contours.sort(key=cv2.contourArea, reverse=True)

    merged_contour = valid_contours[0]

    remaining_contours = valid_contours[1:]
    while remaining_contours:
        merged_pts_2d = merged_contour.reshape(-1, 2)

        closest_dist = float('inf')
        closest_contour_idx = -1
        closest_idx_merged = -1
        closest_idx_to_merge = -1

        for i, contour_to_check in enumerate(remaining_contours):
            check_pts_2d = contour_to_check.reshape(-1, 2)

            dist_matrix = cdist(merged_pts_2d, check_pts_2d, 'euclidean')

            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            min_dist = dist_matrix[min_idx]

            if min_dist < closest_dist:
                closest_dist = min_dist
                closest_contour_idx = i
                closest_idx_merged = min_idx[0]
                closest_idx_to_merge = min_idx[1]

        if closest_contour_idx == -1:
            break

        contour_to_merge = remaining_contours.pop(closest_contour_idx)

        idx_merged = closest_idx_merged
        idx_to_merge = closest_idx_to_merge

        merged_part1 = merged_contour[:idx_merged + 1]
        merged_part2 = merged_contour[idx_merged:]

        reordered_to_merge = np.roll(contour_to_merge, -idx_to_merge, axis=0)

        bridge_end_point = contour_to_merge[idx_to_merge : idx_to_merge + 1]

        merged_contour = np.concatenate([
            merged_part1,
            bridge_end_point,
            reordered_to_merge,
            bridge_end_point,
            merged_part2,
        ], axis=0)

    return merged_contour

def compute_dlmi_logits(mask_np, mode="Fixed", intensity=10.0, falloff=20):
    """Convert a binary mask to DLMI logits for memory encoding.

    Args:
        mask_np: Binary mask (H, W), values 0/1 or boolean.
        mode: "Fixed" for uniform logits, "Gradient" for distance-based continuous logits.
        intensity: Max logit magnitude. Range [-intensity, +intensity].
        falloff: Pixels from boundary to reach full intensity (Gradient mode only).

    Returns:
        np.ndarray (H, W) float32, logit values in [-intensity, +intensity].
    """
    if mode == "Fixed":
        return (mask_np.astype(np.float32) * 2.0 * intensity) - intensity

    mask_uint8 = (mask_np > 0).astype(np.uint8)
    dist_inside = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(1 - mask_uint8, cv2.DIST_L2, 5)

    falloff = max(falloff, 1)
    logits = np.zeros_like(dist_inside, dtype=np.float32)
    inside = mask_uint8 > 0
    outside = ~inside
    logits[inside] = np.clip(dist_inside[inside] / falloff, 0, 1) * intensity
    logits[outside] = -np.clip(dist_outside[outside] / falloff, 0, 1) * intensity
    return logits
