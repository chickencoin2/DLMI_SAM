"""Save controller: "Confirm Labels" dialog, threaded frame save, completion callback."""
import os
import logging
import threading

import cv2
from PIL import Image
from tkinter import messagebox
import shutil

from .autolabel_workflow import save_frame_dispatch

logger = logging.getLogger("DLMI_SAM_LABELER.SaveController")


def _ensure_yolo_dataset_ready(app, fmt, has_any_pose):
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


def _confirm_and_save_image_labels(app):
    """Image source: save the current frame's annotations directly — no propagation required."""
    if app.current_cv_frame is None:
        messagebox.showwarning("Notice", "No image loaded.", parent=app.root)
        return

    masks_data = {
        obj_id: data for obj_id, data in app.tracked_objects.items()
        if data and data.get('last_mask') is not None and data['last_mask'].any()
    }
    if not masks_data:
        messagebox.showwarning(
            "Notice",
            "No annotations to save.\nDetect objects or create polygon/paint masks first.",
            parent=app.root
        )
        return

    image_name = os.path.basename(app.video_source_path) if isinstance(app.video_source_path, str) else "image"
    response = messagebox.askyesno(
        "Confirm Labeling",
        f"Do you want to save labels for this image?\n\n"
        f"{image_name} — {len(masks_data)} object(s)",
        parent=app.root
    )
    if not response:
        return

    fmt = app.save_format_var.get()
    has_pose = any(isinstance(d, dict) and d.get('pose_points') for d in masks_data.values())
    if not _ensure_yolo_dataset_ready(app, fmt, has_pose):
        return

    app.app_state = "LABELING"
    app.update_status("Saving labels...")

    frame_bgr = app.current_cv_frame.copy()

    def _save_image_thread():
        try:
            pose_subdir = app._pose_labels_subdir()
            base_name = "frame"
            if app.video_source_path and isinstance(app.video_source_path, str):
                base_name = os.path.splitext(os.path.basename(app.video_source_path))[0]
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            save_frame_dispatch(app, frame_pil, 0, masks_data, base_name, pose_subdir=pose_subdir)
            app.root.after(0, lambda: on_save_finished(app, 1))
        except Exception as e:
            logger.exception("Error during image label saving:")
            app.root.after(0, app.update_status, f"Error during saving: {e}")

    threading.Thread(target=_save_image_thread, daemon=True).start()


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
