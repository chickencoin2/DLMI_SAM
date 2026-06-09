"""Save controller: "Confirm Labels" dialog, threaded frame save, completion
callback. Thin orchestration layer over `autolabel_workflow.save_frame_dispatch`
(which itself is the single save entry point added in Phase 3)."""
import os
import logging
import threading

import cv2
from PIL import Image
from tkinter import messagebox
import shutil

from .autolabel_workflow import save_frame_dispatch

logger = logging.getLogger("DLMI_SAM_LABELER.SaveController")


def confirm_and_save_labels(app):
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

    # Pose YOLO save runs whenever any frame has pose data, even when seg fmt
    # is "labelme" (segment as LabelMe JSON, pose as YOLO-pose). In that case
    # we still need a valid YOLO class list and data.yaml so the pose .txt
    # files have meaningful class indices and the resulting folder is a
    # trainable YOLO-pose dataset.
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

    needs_yolo_dataset = fmt in ["yolo", "both"] or has_any_pose_overall

    if needs_yolo_dataset:
        # Pick the dataset root: when the user has configured a dedicated
        # pose root AND seg fmt is labelme-only, the YOLO dataset (= pose
        # dataset) belongs under that pose root. Otherwise it lives under
        # the regular save dir.
        use_pose_root = bool(getattr(app, 'use_custom_pose_save_path_var', None) and
                             app.use_custom_pose_save_path_var.get())
        if use_pose_root and fmt == "labelme":
            save_dir = app.custom_pose_save_dir_var.get()
        else:
            save_dir = app._get_save_directory()

        check_result = app._check_existing_yolo_dataset(save_dir)
        if check_result is None:
            return
        elif check_result == "new_setup":
            if not app._prompt_yolo_class_info():
                return

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
                        return
                    elif folder_response:
                        try:
                            shutil.rmtree(save_dir)
                            logger.info(f"Existing folder deleted: {save_dir}")
                        except Exception as e:
                            logger.error(f"Folder deletion failed: {e}")
                            messagebox.showerror("Error", f"Folder deletion failed:\n{e}", parent=app.root)
                            return

            if not app._init_yolo_dataset_structure(save_dir):
                messagebox.showerror("Error", "Failed to create YOLO dataset structure.", parent=app.root)
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
        # Throttle progress callbacks: at most ~50 idle-callbacks across the
        # whole save run (every ~2% of progress) so the Tk main loop isn't
        # flooded by per-frame `after(0, ...)` posts when saving many frames.
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
