import numpy as np
import cv2
import logging
from tkinter import messagebox

logger = logging.getLogger("DLMI_SAM_LABELER.PolygonController")


def toggle_polygon_mode(app):
    app.polygon_mode_active = not app.polygon_mode_active
    if app.polygon_mode_active:
        app.polygon_points = []
        app.update_status("Polygon mode: Left click to add points. Click 'Complete Object' when done.")
        logger.info("Polygon add mode activated")
    else:
        app.polygon_points = []
        app.update_status("Polygon mode deactivated")
        logger.info("Polygon add mode deactivated")

    if app.view and hasattr(app.view, 'update_polygon_mode_ui'):
        app.view.update_polygon_mode_ui(app.polygon_mode_active)

    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def add_polygon_point(app, x, y):
    if not app.polygon_mode_active:
        return False

    app.polygon_points.append((int(x), int(y)))
    logger.debug(f"Polygon point added: ({x}, {y}), total {len(app.polygon_points)}")
    app.update_status(f"{len(app.polygon_points)} polygon points entered. Add more or click 'Complete Object'.")

    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())

    return True


def undo_last_polygon_point(app):
    if not app.polygon_mode_active or not app.polygon_points:
        return

    removed = app.polygon_points.pop()
    logger.debug(f"Polygon point removed: {removed}, remaining {len(app.polygon_points)}")
    app.update_status(f"Point removed. Remaining: {len(app.polygon_points)}")

    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def complete_polygon_object(app):
    if not app.polygon_mode_active:
        messagebox.showwarning("Notice", "Polygon mode is not active.", parent=app.root)
        return

    if len(app.polygon_points) < 3:
        messagebox.showwarning("Notice", "Polygon requires at least 3 points.", parent=app.root)
        return

    if app.current_cv_frame is None:
        messagebox.showwarning("Notice", "No current frame.", parent=app.root)
        return

    try:
        h, w = app.current_cv_frame.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(app.polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        mask_bool = mask > 0

        if not mask_bool.any():
            messagebox.showwarning("Notice", "Generated mask is empty.", parent=app.root)
            return

        new_obj_id = app.next_obj_id_to_propose
        app.next_obj_id_to_propose += 1

        label = app.default_object_label_var.get()
        app.tracked_objects[new_obj_id] = {
            'custom_label': label,
            'last_mask': mask_bool,
            'is_polygon_object': True,
            'polygon_points': app.polygon_points.copy()
        }

        app.polygon_objects.append({
            'obj_id': new_obj_id,
            'points': app.polygon_points.copy(),
            'mask': mask_bool,
            'label': label
        })

        logger.info(f"Polygon object created: ID={new_obj_id}, points={len(app.polygon_points)}, label={label}")
        app.update_status(f"Polygon object {new_obj_id} created. Input to SAM3 or continue adding polygons.")

        app.polygon_points = []

        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())

        app._update_obj_id_info_label()

    except Exception as e:
        logger.exception(f"Polygon object creation failed: {e}")
        messagebox.showerror("Error", f"Error creating polygon object:\n{e}", parent=app.root)


def input_polygon_to_sam3(app):
    if app.app_state == "PAUSED":
        app.update_status("Paused: Cannot input to SAM3 session directly. Use DLMI injection instead.")
        return

    polygon_objs = [obj_id for obj_id, data in app.tracked_objects.items()
                   if data.get('is_polygon_object', False)]

    if not polygon_objs:
        messagebox.showwarning("Notice", "No polygon objects to input to SAM3.\nDraw polygons first and click 'Complete Object'.", parent=app.root)
        return

    if app.tracker_model is None or app.tracker_processor is None:
        messagebox.showerror("Error", "SAM3 model not loaded.", parent=app.root)
        return

    if app.inference_session is None:
        if not app._init_inference_session():
            messagebox.showerror("Error", "SAM3 session initialization failed.", parent=app.root)
            return

    app.suppressed_sam_ids.clear()
    logger.info("Polygon to SAM3 input: suppressed_sam_ids cleared")

    try:
        frame_idx = 0

        obj_ids_list = []
        input_masks_list = []

        for obj_id in polygon_objs:
            obj_data = app.tracked_objects.get(obj_id, {})
            mask = obj_data.get('last_mask')
            if mask is not None and mask.any():
                obj_ids_list.append(obj_id)
                input_masks_list.append(mask.astype(np.uint8))

        if not obj_ids_list:
            messagebox.showwarning("Notice", "No valid masks.", parent=app.root)
            return

        logger.info(f"Inputting {len(obj_ids_list)} polygon objects to SAM3...")
        app.tracker_processor.add_inputs_to_inference_session(
            inference_session=app.inference_session,
            frame_idx=frame_idx,
            obj_ids=obj_ids_list,
            input_masks=input_masks_list,
        )

        for obj_id in polygon_objs:
            if obj_id in app.tracked_objects:
                app.tracked_objects[obj_id]['is_polygon_object'] = False
                app.tracked_objects[obj_id]['is_sam3_object'] = True

        app.is_tracking_ever_started = True
        app.update_status(f"{len(obj_ids_list)} polygon objects input to SAM3. Start propagation.")
        logger.info(f"Polygon objects input to SAM3 complete: {obj_ids_list}")

        app.polygon_mode_active = False
        app.polygon_points = []
        if app.view and hasattr(app.view, 'update_polygon_mode_ui'):
            app.view.update_polygon_mode_ui(False)

    except Exception as e:
        logger.exception(f"SAM3 polygon input failed: {e}")
        messagebox.showerror("Error", f"Error during SAM3 input:\n{e}", parent=app.root)


def cancel_polygon_mode(app):
    app.polygon_mode_active = False
    app.polygon_points = []
    app.update_status("Polygon mode cancelled")
    logger.info("Polygon mode cancelled")

    if app.view and hasattr(app.view, 'update_polygon_mode_ui'):
        app.view.update_polygon_mode_ui(False)

    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def _auto_disable_polygon_mode(app):
    """Auto-disable polygon mode if active. Called before propagation start/resume."""
    if app.polygon_mode_active:
        app.polygon_mode_active = False
        app.polygon_points = []
        if app.view and hasattr(app.view, 'update_polygon_mode_ui'):
            app.view.update_polygon_mode_ui(False)
        logger.info("Polygon mode auto-disabled before propagation")
