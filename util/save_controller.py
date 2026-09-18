"""Save controller: "Confirm Labels" dialog, threaded frame save, completion callback."""
import copy
import os
import logging
import threading

import cv2
import yaml
from PIL import Image
from tkinter import messagebox
import shutil

from .autolabel_workflow import save_frame_dispatch

logger = logging.getLogger("DLMI_SAM_LABELER.SaveController")

_save_dispatch_lock = threading.Lock()


def _refresh_current_frame_save_button(app):
    view = getattr(app, 'view', None)
    if view is None or not hasattr(view, 'update_current_frame_save_button_state'):
        return
    try:
        view.update_current_frame_save_button_state()
    except Exception:
        pass


def _ensure_yolo_dataset_ready(app, fmt, has_any_pose, interactive=True):
    """Create/verify the YOLO dataset structure when the requested formats need it. Returns False to abort."""
    needs_yolo_dataset = fmt in ["yolo", "both"] or has_any_pose
    if not needs_yolo_dataset:
        return True

    # Dataset root: dedicated pose root when configured with labelme-only seg, else the regular save dir.
    use_pose_root = bool(getattr(app, 'use_custom_pose_save_path_var', None) and
                         app.use_custom_pose_save_path_var.get())
    if use_pose_root and fmt == "labelme":
        save_dir = app.custom_pose_save_dir_var.get()
    else:
        save_dir = app._get_save_directory()

    if not interactive:
        return _prepare_yolo_dataset_silently(app, save_dir)

    check_result = app._check_existing_yolo_dataset(save_dir)
    if check_result is None:
        return False
    elif check_result == "new_setup":
        if not app._prompt_yolo_class_info():
            return False

        if os.path.exists(save_dir):
            yaml_path = os.path.join(save_dir, "data.yaml")
            images_dir = os.path.join(save_dir, "images")
            labels_dir = os.path.join(save_dir, "labels")
            if not (os.path.exists(yaml_path) and os.path.exists(images_dir) and os.path.exists(labels_dir)):
                folder_response = messagebox.askyesnocancel(
                    "Existing Folder Found",
                    f"Folder '{save_dir}' already exists.\n\n"
                    f"Yes: Delete folder contents and create YOLO structure\n"
                    f"No: Keep existing contents and add YOLO structure\n"
                    f"Cancel: Abort operation",
                    parent=app.root
                )
                if folder_response is None:
                    return False
                elif folder_response:
                    try:
                        shutil.rmtree(save_dir)
                        logger.info(f"Existing folder deleted: {save_dir}")
                    except Exception as e:
                        logger.error(f"Folder deletion failed: {e}")
                        messagebox.showerror("Error", f"Folder deletion failed:\n{e}", parent=app.root)
                        return False

        if not app._init_yolo_dataset_structure(save_dir):
            messagebox.showerror("Error", "Failed to create YOLO dataset structure.", parent=app.root)
            return False

    return True


def _prepare_yolo_dataset_silently(app, save_dir):
    if getattr(app, 'yolo_dataset_initialized', False):
        return True

    yaml_path = os.path.join(save_dir, "data.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as handle:
                yaml_data = yaml.safe_load(handle) or {}
            app.yolo_nc = yaml_data.get('nc', 0)
            app.yolo_class_names_for_save = yaml_data.get('names', []) or []
            app.yolo_dataset_initialized = True
            return True
        except Exception as e:
            logger.error(f"data.yaml read failed, rebuilding structure: {e}")

    if not app.yolo_class_names_for_save:
        app.yolo_class_names_for_save = [app.default_object_label_var.get() or "object"]
    app.yolo_nc = len(app.yolo_class_names_for_save)
    if not app._init_yolo_dataset_structure(save_dir):
        messagebox.showerror("Error", "Failed to create YOLO dataset structure.", parent=app.root)
        return False
    return True


def _complete_pending_input_for_current_save(app):
    pending = app._pending_manual_input_modes()
    if not pending:
        return True

    if ("paint" in pending and app.paint_stroke_active
            and app.paint_negative_stroke_mask is not None
            and app.paint_negative_stroke_mask.any()):
        messagebox.showwarning(
            "Notice",
            "Finish the active paint stroke before saving this frame.",
            parent=app.root,
        )
        return False

    if "polygon" in pending and len(app.polygon_points) < 3:
        messagebox.showwarning(
            "Notice",
            "The unfinished polygon needs at least 3 points.\n"
            "Complete or cancel it before saving this frame.",
            parent=app.root,
        )
        return False

    names = " and ".join(mode.capitalize() for mode in pending)
    if not messagebox.askyesno(
            "Complete Unfinished Input",
            f"{names} input has not been completed.\n\n"
            "Complete it and include it in this frame save?",
            icon=messagebox.WARNING,
            parent=app.root):
        return False

    if "paint" in pending:
        app.complete_paint_object()
    if "polygon" in pending:
        app.complete_polygon_object()
    return not app._pending_manual_input_modes()


def _displayed_annotations(app):
    overlay = dict(getattr(app, 'displayed_overlay_masks', None) or {})
    tracked = dict(getattr(app, 'tracked_objects', None) or {})

    frame_masks = {}
    result = (getattr(app, 'propagated_results', None) or {}).get(
        getattr(app, 'displayed_frame_idx', 0))
    if isinstance(result, dict):
        frame_masks = result.get('masks') or {}

    pose_visible = bool(getattr(app, 'displayed_pose_visible', True))
    obj_ids = list(overlay.keys())
    if pose_visible:
        for obj_id, data in tracked.items():
            if obj_id not in overlay and isinstance(data, dict) and data.get('pose_points'):
                obj_ids.append(obj_id)

    snapshot = {}
    for obj_id in obj_ids:
        source = tracked.get(obj_id)
        if not isinstance(source, dict):
            source = frame_masks.get(obj_id)
        if not isinstance(source, dict):
            source = {}
        entry = copy.deepcopy(source)

        mask = overlay.get(obj_id)
        if mask is not None:
            entry['last_mask'] = mask.copy()
        else:
            entry.pop('last_mask', None)
        if not pose_visible:
            entry.pop('pose_points', None)
            entry.pop('pose_edges', None)

        has_mask = entry.get('last_mask') is not None and entry['last_mask'].any()
        if not has_mask and not entry.get('pose_points'):
            continue
        snapshot[obj_id] = entry
    return snapshot


def _no_annotation_message(app):
    if (app._is_pre_propagate_phase()
            and app.label_anchor_frame_idx is not None
            and getattr(app, 'displayed_frame_idx', 0) != app.label_anchor_frame_idx):
        anchor = app.label_anchor_frame_idx + getattr(app, 'cut_start_frame', 0)
        return (f"Nothing is drawn on the displayed frame.\n"
                f"Labels made before propagation live on frame {anchor}; "
                f"go back there to save them.")
    return "Nothing is drawn on the displayed frame.\nDetect objects or create polygon/paint masks first."


def _finish_current_frame_save(app, actual_frame_num, error=None):
    app.current_frame_save_in_progress = False
    _refresh_current_frame_save_button(app)
    if error is not None:
        app.update_status(f"Error during current-frame save: {error}")
        messagebox.showerror(
            "Error", f"Error saving current frame:\n{error}", parent=app.root)
        return

    app.update_status(f"Current frame {actual_frame_num} saved.")
    messagebox.showinfo(
        "Complete",
        f"Image and labels for frame {actual_frame_num} have been saved.",
        parent=app.root,
    )


def save_current_frame_labels(app):
    if getattr(app, 'current_frame_save_in_progress', False):
        messagebox.showinfo(
            "Busy", "The previous frame save is still running.", parent=app.root)
        return

    if getattr(app, 'displayed_frame_bgr', None) is None and app.current_cv_frame is None:
        messagebox.showwarning("Notice", "No frame loaded.", parent=app.root)
        return

    if not _complete_pending_input_for_current_save(app):
        return

    frame_source = getattr(app, 'displayed_frame_bgr', None)
    if frame_source is None:
        frame_source = app.current_cv_frame
    frame_bgr = frame_source.copy()
    masks_data = _displayed_annotations(app)
    if not masks_data:
        messagebox.showwarning("Notice", _no_annotation_message(app), parent=app.root)
        return

    is_image = getattr(app, 'is_image_source', False)
    actual_frame_num = 0 if is_image else (
        getattr(app, 'cut_start_frame', 0)
        + getattr(app, 'displayed_frame_idx', 0)
    )
    source_name = (
        os.path.basename(app.video_source_path)
        if isinstance(app.video_source_path, str) else "camera"
    )
    target_text = "this image" if is_image else f"frame {actual_frame_num}"
    response = messagebox.askyesno(
        "Confirm Labeling",
        f"Do you want to save labels for {target_text}?\n\n"
        f"{source_name} \u2014 {len(masks_data)} object(s)",
        parent=app.root
    )
    if not response:
        return

    fmt = app.save_format_var.get()
    has_pose = any(isinstance(d, dict) and d.get('pose_points') for d in masks_data.values())
    if not _ensure_yolo_dataset_ready(app, fmt, has_pose, interactive=False):
        return

    app.current_frame_save_in_progress = True
    _refresh_current_frame_save_button(app)
    app.update_status(f"Saving frame {actual_frame_num}...")

    save_error = None
    try:
        pose_subdir = app._pose_labels_subdir()
        base_name = "frame"
        if app.video_source_path and isinstance(app.video_source_path, str):
            base_name = os.path.splitext(os.path.basename(app.video_source_path))[0]
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        with _save_dispatch_lock:
            save_frame_dispatch(
                app, frame_pil, actual_frame_num, masks_data, base_name,
                pose_subdir=pose_subdir, always_json=True)
    except Exception as e:
        logger.exception("Error during current-frame label saving:")
        save_error = e
    finally:
        app.current_frame_save_in_progress = False
        _refresh_current_frame_save_button(app)

    _finish_current_frame_save(app, actual_frame_num, save_error)


def _confirm_and_save_image_labels(app):
    """Image source: save the current frame's annotations directly — no propagation required."""
    return save_current_frame_labels(app)


def confirm_and_save_labels(app):
    if getattr(app, 'is_image_source', False):
        _confirm_and_save_image_labels(app)
        return

    if not app.propagated_results:
        messagebox.showwarning("Notice", "No propagation results to save.", parent=app.root)
        return

    cut_offset = getattr(app, 'cut_start_frame', 0)
    frame_indices = sorted(app.propagated_results.keys())
    actual_start = cut_offset + min(frame_indices) if frame_indices else cut_offset
    actual_end = cut_offset + max(frame_indices) if frame_indices else cut_offset

    response = messagebox.askyesno(
        "Confirm Labeling",
        f"Do you want to save labels for {len(app.propagated_results)} frames?\n\n"
        f"Original video frame range: {actual_start} ~ {actual_end}",
        parent=app.root
    )
    if not response:
        return

    fmt = app.save_format_var.get()

    # Pose YOLO save runs whenever any frame has pose data, even when seg fmt is "labelme" (segment as LabelMe JSON, pose as YOLO-pose).
    has_any_pose_overall = False
    try:
        for _frame_idx, _result in app.propagated_results.items():
            if _frame_idx in app.discarded_frames:
                continue
            _masks = _result.get('masks') if _result else None
            if not _masks:
                continue
            for _oid, _odata in _masks.items():
                if isinstance(_odata, dict) and _odata.get('pose_points'):
                    has_any_pose_overall = True
                    break
            if has_any_pose_overall:
                break
    except Exception:
        has_any_pose_overall = False

    if not _ensure_yolo_dataset_ready(app, fmt, has_any_pose_overall):
        return

    app.app_state = "LABELING"
    app.update_status("Saving labels...")

    save_thread = threading.Thread(target=lambda: _save_labels_thread(app), daemon=True)
    save_thread.start()


def _save_labels_thread(app):
    try:
        pose_subdir = app._pose_labels_subdir()

        frame_base_name = "frame"
        if app.video_source_path and isinstance(app.video_source_path, str):
            frame_base_name = os.path.splitext(os.path.basename(app.video_source_path))[0]

        cut_offset = getattr(app, 'cut_start_frame', 0)

        frames_to_save = {
            frame_idx: result
            for frame_idx, result in app.propagated_results.items()
            if frame_idx not in app.discarded_frames
        }
        total_frames = len(frames_to_save)
        skipped_count = len(app.discarded_frames)
        if skipped_count > 0:
            logger.info(f"Label saving: {skipped_count} frames excluded by discard marking")

        saved_count = 0
        # Throttle progress callbacks to ~50 over the run so the Tk loop isn't flooded.
        progress_step = max(1, total_frames // 50) if total_frames > 0 else 1
        last_reported_progress = -1
        for i, (frame_idx, result) in enumerate(sorted(frames_to_save.items())):
            frame_bgr = result['frame']
            masks_data = result['masks']
            if not masks_data:
                continue

            actual_frame_num = cut_offset + frame_idx
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            with _save_dispatch_lock:
                save_frame_dispatch(app, frame_pil, actual_frame_num, masks_data, frame_base_name, pose_subdir=pose_subdir)
            saved_count += 1

            done = i + 1
            if done == total_frames or done - last_reported_progress >= progress_step:
                last_reported_progress = done
                progress = int(done / total_frames * 100) if total_frames > 0 else 100
                app.root.after(0, app.view.update_propagate_progress, progress,
                               f"Saving: {done}/{total_frames} (frame {actual_frame_num})")

        app.discarded_frames.clear()
        app.root.after(0, app.view.update_discarded_frames_display, set())
        app.root.after(0, lambda n=saved_count: on_save_finished(app, n))

    except Exception as e:
        logger.exception("Error during label saving:")
        app.root.after(0, app.update_status, f"Error during saving: {e}")


def on_save_finished(app, total_frames):
    app.app_state = "IDLE"
    app.view.update_propagate_progress(100, f"Save complete: {total_frames} frames")
    app.update_status(f"Label saving complete! {total_frames} frames saved.")
    if hasattr(app, 'object_prompt_history'):
        app.object_prompt_history.clear()
        logger.info("Label saving complete: object_prompt_history cleared")
    messagebox.showinfo("Complete", f"Labels for {total_frames} frames have been saved.", parent=app.root)
