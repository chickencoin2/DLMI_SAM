import numpy as np
import logging
import tkinter.messagebox as messagebox

logger = logging.getLogger("DLMI_SAM_LABELER.InputHandlers")


def on_ctrl_press(app, event=None):
    if not app.is_ctrl_pressed:
        app.is_ctrl_pressed = True
        logger.debug("Ctrl pressed")
        update_interaction_status_and_label(app)


def on_spacebar_press(app, event=None):
    pass


def on_ctrl_release(app, event=None):
    if app.is_ctrl_pressed:
        app.is_ctrl_pressed = False
        logger.debug("Ctrl released")
        update_interaction_status_and_label(app)


def on_shift_press(app, event=None):
    if not app.is_shift_pressed:
        app.is_shift_pressed = True
        logger.debug("Shift pressed")
        update_interaction_status_and_label(app)


def on_shift_release(app, event=None):
    if app.is_shift_pressed:
        app.is_shift_pressed = False
        logger.debug("Shift released")
        update_interaction_status_and_label(app)


def on_alt_press(app, event=None):
    if not app.is_alt_pressed:
        app.is_alt_pressed = True
        logger.debug("Alt pressed")
        update_interaction_status_and_label(app)


def on_alt_release(app, event=None):
    if app.is_alt_pressed:
        app.is_alt_pressed = False
        logger.debug("Alt released")
        update_interaction_status_and_label(app)


def update_interaction_status_and_label(app):
    if not app.playback_paused:
        return

    status_msg = ""
    if app.reassign_bbox_mode_active_sam_id is not None:
        status_msg = f"Reassigning BBox for object {app.reassign_bbox_mode_active_sam_id}. Please draw a BBox."
    elif app.problematic_highlight_active_sam_id is not None:
        status_msg = f"Checking issues with object {app.problematic_highlight_active_sam_id}."
    elif app.interaction_correction_pending is not None:
        status_msg = f"Auto-correcting object {app.interaction_correction_pending}. Please draw a BBox."
    elif app.is_ctrl_pressed and app.is_shift_pressed:
        status_msg = "Object deletion mode (Ctrl+Shift+left-click to delete object)."
    elif app.is_ctrl_pressed and len(app.selected_objects_sam_ids) >= 2:
        status_msg = f"Multi-select: {len(app.selected_objects_sam_ids)} objects ({sorted(app.selected_objects_sam_ids)}). Can merge with 'Merge Objects'."
    elif app.is_ctrl_pressed and app.selected_object_sam_id is not None:
        status_msg = f"Object {app.selected_object_sam_id} selected. Add points (Ctrl+wheel/right-click) or click button."
    elif app.is_ctrl_pressed:
        status_msg = "Object selection mode (Ctrl+left-click to select object, multi-select available)."
    else:
        status_msg = "Paused. Draw a BBox or Ctrl+left-click to select object."

    app.update_status(status_msg)
    app._update_obj_id_info_label()
    app._update_ui_for_autolabel_state(False)


def is_any_special_mode_active(app):
    return app.reassign_bbox_mode_active_sam_id is not None or \
           app.problematic_highlight_active_sam_id is not None or \
           app.interaction_correction_pending is not None


def get_object_id_at_coords(app, img_x, img_y):
    if not app.tracked_objects or app.current_cv_frame is None:
        return None
    obj_ids_to_check = list(app.tracked_objects.keys())
    sorted_obj_ids = sorted(obj_ids_to_check, reverse=True)
    for obj_id in sorted_obj_ids:
        data = app.tracked_objects.get(obj_id)
        if data and "last_mask" in data and data["last_mask"] is not None:
            mask = data["last_mask"]
            if 0 <= img_y < mask.shape[0] and 0 <= img_x < mask.shape[1] and mask[img_y, img_x]:
                return obj_id
    return None


def on_left_mouse_press(app, event):
    if is_any_special_mode_active(app) and app.reassign_bbox_mode_active_sam_id is None and app.interaction_correction_pending is None:
        app.update_status(f"Checking object {app.problematic_highlight_active_sam_id or '?'}. Mouse input is disabled.")
        return

    if not app.playback_paused:
        app.update_status("Prompt/selection only available when paused.")
        return
    if app.current_cv_frame is None:
        logger.error("Original frame image not found.")
        return

    # During propagation pause: restrict interactions
    if getattr(app, 'app_state', '') == "PAUSED":
        if app.is_ctrl_pressed:
            pass  # Always allow Ctrl+click (selection) and Ctrl+Shift+click (delete)
        elif getattr(app, 'polygon_mode_active', False):
            pass  # Allow polygon point placement during pause
        else:
            app.update_status("Paused: Use polygon mode for mask creation, or Ctrl+click for selection.")
            return

    if getattr(app, 'polygon_mode_active', False):
        img_x, img_y = app._canvas_to_image_coords(event.x, event.y)
        if app.add_polygon_point(img_x, img_y):
            return

    pose_add_on = bool(getattr(app, 'pose_add_mode_var', None) and app.pose_add_mode_var.get())
    if pose_add_on and not app.is_ctrl_pressed and not app.is_alt_pressed:
        img_x, img_y = app._canvas_to_image_coords(event.x, event.y)
        app.add_pose_point_at(img_x, img_y)
        app.bbox_start_canvas_coords = None
        if app.view:
            app.view.delete_temp_bbox()
        return

    if app.is_shift_pressed and not app.is_ctrl_pressed and not app.is_alt_pressed:
        img_x, img_y = app._canvas_to_image_coords(event.x, event.y)
        hit = None
        if getattr(app, '_pose_ui', None) is not None:
            try:
                hit = app._pose_ui.hit_test_pose_point(app, img_x, img_y)
            except Exception as _hit_err:
                logger.debug(f"pose hit_test failed: {_hit_err}")
        if hit is not None:
            app.toggle_pose_point_selection(hit[0], hit[1])
        else:
            app.clear_pose_selection()
        app.bbox_start_canvas_coords = None
        if app.view:
            app.view.delete_temp_bbox()
        return

    if app.is_ctrl_pressed and app.is_shift_pressed:
        on_ctrl_shift_left_click(app, event)
        return

    app.bbox_start_canvas_coords = (event.x, event.y)

    if app.is_ctrl_pressed:
        if app.reassign_bbox_mode_active_sam_id is not None or app.interaction_correction_pending is not None:
            app.bbox_start_canvas_coords = None
            return

        img_x, img_y = app._canvas_to_image_coords(event.x, event.y)
        clicked_obj_id = get_object_id_at_coords(app, img_x, img_y)

        if clicked_obj_id is not None:
            if clicked_obj_id in app.selected_objects_sam_ids:
                app.selected_objects_sam_ids.discard(clicked_obj_id)
                logger.info(f"Object {clicked_obj_id} removed from multi-select. Current selection: {app.selected_objects_sam_ids}")
                if app.selected_object_sam_id == clicked_obj_id:
                    app.selected_object_sam_id = None
            else:
                app.selected_objects_sam_ids.add(clicked_obj_id)
                app.selected_object_sam_id = clicked_obj_id
                logger.info(f"Object {clicked_obj_id} added to multi-select. Current selection: {app.selected_objects_sam_ids}")
        else:
            if app.selected_objects_sam_ids:
                logger.info(f"All objects deselected (empty space). Previous selection: {app.selected_objects_sam_ids}")
            app.selected_objects_sam_ids.clear()
            app.selected_object_sam_id = None

        if app.view and hasattr(app.view, 'btn_merge_objects'):
            if len(app.selected_objects_sam_ids) >= 2:
                app.view.btn_merge_objects.config(state='normal')
            else:
                app.view.btn_merge_objects.config(state='disabled')

        if app.view and app.view.notebook and app.view.obj_control_tab:
            app.view.notebook.select(app.view.obj_control_tab)

        update_interaction_status_and_label(app)
        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
        app.bbox_start_canvas_coords = None
        if app.view:
            app.view.delete_temp_bbox()
        return

    if app.view:
        app.view.delete_temp_bbox()
    if app.view:
        app.view.draw_temp_bbox(event.x, event.y, event.x, event.y)


def on_left_mouse_drag(app, event):
    if is_any_special_mode_active(app) and app.reassign_bbox_mode_active_sam_id is None and app.interaction_correction_pending is None:
        return
    if app.is_ctrl_pressed and not (app.reassign_bbox_mode_active_sam_id or app.interaction_correction_pending):
        return
    if app.bbox_start_canvas_coords is None:
        return
    if not app.view or app.view.temp_bbox_on_canvas_id is None:
        return
    start_x, start_y = app.bbox_start_canvas_coords
    if app.view:
        app.view.draw_temp_bbox(start_x, start_y, event.x, event.y)


def on_left_mouse_release(app, event):
    if is_any_special_mode_active(app) and app.reassign_bbox_mode_active_sam_id is None and app.interaction_correction_pending is None:
        return

    if app.is_ctrl_pressed and not app.is_shift_pressed and not (app.reassign_bbox_mode_active_sam_id or app.interaction_correction_pending):
        app.bbox_start_canvas_coords = None
        return

    if app.bbox_start_canvas_coords is None:
        return

    temp_bbox_coords = app.view.get_temp_bbox_coords() if app.view else None
    if temp_bbox_coords:
        x0_c, y0_c, x1_c, y1_c = temp_bbox_coords
        if app.view:
            app.view.delete_temp_bbox()
        app.bbox_start_canvas_coords = None

        if abs(x0_c - x1_c) < 5 or abs(y0_c - y1_c) < 5:
            logger.info("BBox too small (canvas)")
            return
        if app.current_frame_pil_rgb_original is None:
            logger.error("_on_left_mouse_release: current_frame_pil_rgb_original is None")
            return

        img_s_x, img_s_y = app._canvas_to_image_coords(min(x0_c, x1_c), min(y0_c, y1_c))
        img_e_x, img_e_y = app._canvas_to_image_coords(max(x0_c, x1_c), max(y0_c, y1_c))

        if img_e_x - img_s_x < 5 or img_e_y - img_s_y < 5:
            logger.info("BBox too small (image)")
            return

        bbox_np = np.array([img_s_x, img_s_y, img_e_x, img_e_y], dtype=np.float32)

        if app.reassign_bbox_mode_active_sam_id is not None:
            logger.info(f"Object {app.reassign_bbox_mode_active_sam_id} BBox reassignment: {bbox_np.tolist()}")
            app._handle_sam_prompt_wrapper(prompt_type='bbox', coords=bbox_np,
                                           target_existing_obj_id=app.reassign_bbox_mode_active_sam_id)
        elif app.interaction_correction_pending is not None:
            logger.info(f"Object {app.interaction_correction_pending} auto-correction BBox: {bbox_np.tolist()}")
            app._handle_sam_prompt_wrapper(prompt_type='bbox', coords=bbox_np,
                                           target_existing_obj_id=app.interaction_correction_pending)
        elif app.is_alt_pressed or (event.state & 0x20000):
            actual_alt_pressed = bool(event.state & 0x20000) or bool(event.state & 0x0008)

            if not actual_alt_pressed and app.is_alt_pressed:
                logger.info("Alt key state mismatch detected: is_alt_pressed=True but actual key is not pressed. Resetting state.")
                app.is_alt_pressed = False
                app._handle_sam_prompt_wrapper(prompt_type='bbox', coords=bbox_np,
                                               proposed_obj_id_for_new=app.next_obj_id_to_propose,
                                               custom_label=app.default_object_label_var.get())
                return

            current_prompt_mode = app.prompt_mode_var.get()
            if current_prompt_mode in ("PCS", "PCS_IMAGE"):
                logger.info(f"PCS mode: adding negative exemplar box. bbox={bbox_np.tolist()}")
                app.pcs_exemplar_boxes.append([float(img_s_x), float(img_s_y), float(img_e_x), float(img_e_y)])
                app.pcs_exemplar_labels.append(0)
                app._execute_pcs_with_exemplars()
                return

            logger.info(f"Alt+left-click-drag: negative box region. bbox={bbox_np.tolist()}")
            center_x = (img_s_x + img_e_x) / 2
            center_y = (img_s_y + img_e_y) / 2

            if app.selected_object_sam_id is not None:
                point_coord = np.array([[center_x, center_y]], dtype=np.float32)
                app._handle_sam_prompt_wrapper(
                    prompt_type='point', coords=point_coord, label=0,
                    target_existing_obj_id=app.selected_object_sam_id
                )
            else:
                point_coord = np.array([[center_x, center_y]], dtype=np.float32)
                app._handle_sam_prompt_wrapper(
                    prompt_type='point', coords=point_coord, label=0,
                    proposed_obj_id_for_new=app.next_obj_id_to_propose,
                    custom_label=app.default_object_label_var.get()
                )
        else:
            current_prompt_mode = app.prompt_mode_var.get()
            if current_prompt_mode in ("PCS", "PCS_IMAGE"):
                logger.info(f"PCS mode: adding positive exemplar box. bbox={bbox_np.tolist()}")
                app.pcs_exemplar_boxes.append([float(img_s_x), float(img_s_y), float(img_e_x), float(img_e_y)])
                app.pcs_exemplar_labels.append(1)
                app._execute_pcs_with_exemplars()
                return

            app._handle_sam_prompt_wrapper(prompt_type='bbox', coords=bbox_np,
                                           proposed_obj_id_for_new=app.next_obj_id_to_propose,
                                           custom_label=app.default_object_label_var.get())
    else:
        app.bbox_start_canvas_coords = None


def on_ctrl_shift_left_click(app, event):
    if not app.playback_paused or not app.is_ctrl_pressed or not app.is_shift_pressed:
        return
    if is_any_special_mode_active(app):
        logger.info("Cannot delete object during special mode (correction/reassign/highlight).")
        return

    img_x, img_y = app._canvas_to_image_coords(event.x, event.y)
    clicked_obj_id = get_object_id_at_coords(app, img_x, img_y)

    if clicked_obj_id is not None:
        if messagebox.askyesno("Confirm Object Deletion", f"Are you sure you want to delete object ID {clicked_obj_id}?", parent=app.root):
            app._delete_object_by_id(clicked_obj_id, "Manual deletion (Ctrl+Shift+Click)")
    else:
        app.update_status("No object selected to delete (Ctrl+Shift+Click attempted).")


def on_ctrl_middle_click_for_point(app, event):
    if is_any_special_mode_active(app):
        return
    if getattr(app, 'app_state', '') == "PAUSED":
        app.update_status("Paused: Cannot add points. Resume propagation first.")
        return
    if not app.is_ctrl_pressed:
        return
    handle_ctrl_point_click_event(app, event, label=1)


def on_ctrl_right_click_for_point(app, event):
    if is_any_special_mode_active(app):
        return
    if getattr(app, 'app_state', '') == "PAUSED":
        app.update_status("Paused: Cannot add points. Resume propagation first.")
        return
    if not app.is_ctrl_pressed:
        return
    handle_ctrl_point_click_event(app, event, label=0)


def on_right_mouse_press(app, event):
    if not app.playback_paused:
        return
    if getattr(app, 'app_state', '') == "PAUSED":
        return
    if app.current_cv_frame is None:
        return
    if is_any_special_mode_active(app):
        return

    if app.is_shift_pressed and not app.is_ctrl_pressed and not app.is_alt_pressed:
        img_x, img_y = app._canvas_to_image_coords(event.x, event.y)
        if hasattr(app, 'select_pose_chain_at'):
            app.select_pose_chain_at(img_x, img_y)
        app.right_bbox_start_canvas_coords = None
        if app.view:
            app.view.delete_temp_bbox()
        return

    if app.is_alt_pressed:
        app.right_bbox_start_canvas_coords = (event.x, event.y)
        if app.view:
            app.view.delete_temp_bbox()
            app.view.draw_temp_bbox(event.x, event.y, event.x, event.y)
    else:
        app.right_bbox_start_canvas_coords = None


def on_right_mouse_drag(app, event):
    if not app.is_alt_pressed:
        return
    if not hasattr(app, 'right_bbox_start_canvas_coords') or app.right_bbox_start_canvas_coords is None:
        return
    if not app.view:
        return

    start_x, start_y = app.right_bbox_start_canvas_coords
    app.view.draw_temp_bbox(start_x, start_y, event.x, event.y)


def on_right_mouse_release(app, event):
    if not hasattr(app, 'right_bbox_start_canvas_coords') or app.right_bbox_start_canvas_coords is None:
        return

    if not app.is_alt_pressed:
        app.right_bbox_start_canvas_coords = None
        if app.view:
            app.view.delete_temp_bbox()
        return

    temp_bbox_coords = app.view.get_temp_bbox_coords() if app.view else None
    if temp_bbox_coords:
        x0_c, y0_c, x1_c, y1_c = temp_bbox_coords
        if app.view:
            app.view.delete_temp_bbox()
        app.right_bbox_start_canvas_coords = None

        if abs(x0_c - x1_c) < 5 or abs(y0_c - y1_c) < 5:
            logger.info("Alt+right-click BBox too small")
            return

        if app.current_frame_pil_rgb_original is None:
            logger.error("_on_right_mouse_release: current_frame_pil_rgb_original is None")
            return

        img_s_x, img_s_y = app._canvas_to_image_coords(min(x0_c, x1_c), min(y0_c, y1_c))
        img_e_x, img_e_y = app._canvas_to_image_coords(max(x0_c, x1_c), max(y0_c, y1_c))

        if img_e_x - img_s_x < 5 or img_e_y - img_s_y < 5:
            logger.info("Alt+right-click BBox too small (image)")
            return

        bbox_np = np.array([img_s_x, img_s_y, img_e_x, img_e_y], dtype=np.float32)
        logger.info(f"Alt+right-click-drag: positive box prompt. bbox={bbox_np.tolist()}")

        if app.selected_object_sam_id is not None:
            app._handle_sam_prompt_wrapper(
                prompt_type='bbox', coords=bbox_np,
                target_existing_obj_id=app.selected_object_sam_id
            )
        else:
            app._handle_sam_prompt_wrapper(
                prompt_type='bbox', coords=bbox_np,
                proposed_obj_id_for_new=app.next_obj_id_to_propose,
                custom_label=app.default_object_label_var.get()
            )
    else:
        app.right_bbox_start_canvas_coords = None


def emergency_stop(app):
    if messagebox.askokcancel("Emergency Stop Confirmation",
                              "Stop all auto-labeling tasks immediately and delete all objects?",
                              parent=app.root):
        logger.warning("Emergency stop triggered! Stopping all tasks and resetting.")
        app.autolabel_active = False
        app.playback_paused = True

        app.interaction_correction_pending = None
        app.problematic_highlight_active_sam_id = None
        app.reassign_bbox_mode_active_sam_id = None
        app.sam_operation_in_progress = False

        app.clear_all_tracked_objects()

        app.update_status("Emergency stop completed. All states have been reset.")
        app._update_ui_for_autolabel_state(False)
        update_interaction_status_and_label(app)


def handle_ctrl_point_click_event(app, event, label):
    if not app.playback_paused:
        app.update_status("Adding points only available when paused.")
        return
    if getattr(app, 'app_state', '') == "PAUSED":
        app.update_status("Paused: Cannot add points. Resume propagation first.")
        return
    if app.current_cv_frame is None:
        return

    img_x, img_y = app._canvas_to_image_coords(event.x, event.y)
    point_coord_for_sam = np.array([[img_x, img_y]], dtype=np.float32)

    if app.selected_object_sam_id is None:
        logger.info(f"Attempting to create new object with point. label={label}, coords=({img_x:.1f}, {img_y:.1f})")
        app._handle_sam_prompt_wrapper(
            prompt_type='point', coords=point_coord_for_sam, label=label,
            proposed_obj_id_for_new=app.next_obj_id_to_propose,
            custom_label=app.default_object_label_var.get()
        )
    else:
        logger.info(f"Attempting to add point (label:{label}) to selected object {app.selected_object_sam_id}.")
        app._handle_sam_prompt_wrapper(
            prompt_type='point', coords=point_coord_for_sam, label=label,
            target_existing_obj_id=app.selected_object_sam_id
        )


def on_canvas_resize(app, event):
    if app._resize_job_id:
        app.root.after_cancel(app._resize_job_id)
    app._resize_job_id = app.root.after(100, app._perform_resize)
