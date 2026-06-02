"""Cut workflow: cut_and_repropagate, cut_and_dlmi_propagate, cut_and_load_labels.

Every public function takes `app` (the SAM3AutolabelApp instance) as its first
argument so this module is a pure "business logic" layer over app state. The
`app.py` class keeps thin wrappers that forward to these functions; external
callers (gui_view button commands etc.) continue to invoke the wrappers
through the instance, so signatures are preserved.

All DLMI injection uses `dlmi_hooks.create_injection_hook` (Phase 2) for
consistency. All save paths go through `autolabel_workflow.save_frame_dispatch`
(Phase 3).
"""
import os
import logging
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
import torch
from PIL import Image

from . import dlmi_hooks
from .autolabel_workflow import save_frame_dispatch

logger = logging.getLogger("DLMI_SAM_LABELER.CutWorkflow")

POSE_LABELS_SUBDIR = "pose_labels"


def pose_labels_subdir(app):
    """Pose labels always go to a dedicated subdir so they don't collide with
    YOLO-seg label files. The advanced `merge_save_format` setting is advisory."""
    return POSE_LABELS_SUBDIR


def save_frames_0_to_n(app, slider_idx, save_labels, current_offset, inclusive=True):
    """Save propagated frames 0..slider_idx (default inclusive). When
    `inclusive=False`, saves 0..slider_idx-1 instead — used by Cut+DLMI and
    Cut+Load, which intentionally do NOT save frame N because frame N is the
    starting frame of the new sub-video and its labels will be re-emitted
    from there. Returns number of saved frames.

    Drives the propagate progress bar in the same style as the threaded
    Confirm-Labels save (throttled to ~50 updates across the run). Cut runs
    on the main Tk thread, so each progress update is paired with
    `update_idletasks()` to flush the redraw and let the user see progress
    while saving.
    """
    if not save_labels:
        return 0
    saved_count = 0
    upper = slider_idx if inclusive else slider_idx - 1
    frames_to_save = sorted(f for f in app.propagated_results.keys() if f <= upper)
    total_frames = len(frames_to_save)
    pose_subdir = pose_labels_subdir(app)
    video_name = (
        os.path.splitext(os.path.basename(app.video_source_path))[0]
        if isinstance(app.video_source_path, str) else "video"
    )

    progress_step = max(1, total_frames // 50) if total_frames > 0 else 1
    last_reported = -1

    if total_frames > 0:
        try:
            app.view.update_propagate_progress(0, f"Cut save: 0/{total_frames}")
            app.root.update_idletasks()
        except Exception:
            pass

    for i, frame_idx in enumerate(frames_to_save):
        result = app.propagated_results.get(frame_idx)
        if not result:
            continue
        frame_bgr = result.get('frame')
        masks = result.get('masks')
        if frame_bgr is None or not masks:
            continue
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        actual_save_frame = current_offset + frame_idx
        save_frame_dispatch(app, frame_pil, actual_save_frame, masks, video_name, pose_subdir=pose_subdir)
        saved_count += 1

        done = i + 1
        if done == total_frames or done - last_reported >= progress_step:
            last_reported = done
            try:
                pct = int(done / total_frames * 100) if total_frames > 0 else 100
                app.view.update_propagate_progress(
                    pct, f"Cut save: {done}/{total_frames} (frame {actual_save_frame})"
                )
                app.root.update_idletasks()
            except Exception:
                pass

    if total_frames > 0:
        try:
            app.view.update_propagate_progress(
                100, f"Cut save complete: {saved_count}/{total_frames}"
            )
            app.root.update_idletasks()
        except Exception:
            pass
    return saved_count


def reset_state_and_seek(app, next_start_frame):
    """Shared cut state reset + video seek + session reinit. Returns success bool."""
    app.propagated_results = {}
    app.tracked_objects.clear()
    app.next_obj_id_to_propose = 1
    app._update_obj_id_info_label()
    app.label_anchor_frame_idx = None
    app.is_tracking_ever_started = False
    app.inference_session = None
    app.pcs_inference_session = None
    app.pcs_streaming_session = None
    app.video_frames_cache = []
    app.discarded_frames.clear()
    if hasattr(app.view, 'update_discarded_frames_display'):
        app.view.update_discarded_frames_display(set())
    if hasattr(app, 'object_prompt_history'):
        app.object_prompt_history.clear()

    app.cut_start_frame = next_start_frame

    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, next_start_frame)
        ret, frame_bgr = app.cap.read()
        if ret:
            app.current_cv_frame = frame_bgr
            app.current_frame_idx_conceptual = 0
            app._display_cv_frame_on_view(frame_bgr, {})
            if not app._init_inference_session():
                messagebox.showerror("Error", "Failed to initialize new session", parent=app.root)
                return False
        else:
            return False

    remaining_frames = app.video_total_frames - next_start_frame
    max_slider_frame = remaining_frames - 1 if remaining_frames > 0 else 0
    app.view.update_review_slider_range(max_slider_frame)
    app.view.review_frame_slider.set(0)
    app.review_current_frame = 0
    app.view.update_review_frame_info(next_start_frame, max_slider_frame)
    app.view.btn_start_propagate.config(state=tk.NORMAL)
    app.view.enable_review_controls(False)
    app.app_state = "IDLE"
    app.info_video_total_frames_var.set(
        f"{remaining_frames} (original {next_start_frame}~{app.video_total_frames-1})"
    )
    return True


def dlmi_mini_propagate_n_to_n1(app, frame_n_bgr, frame_n_plus_1_bgr, obj_id_to_mask_label):
    """Run a 2-frame mini inference session from frame n to n+1.

    Apply saved n-masks as prompts at frame 0, optionally with DLMI injection,
    propagate to frame 1, return {obj_id: mask_ndarray_bool} for frame n+1.

    obj_id_to_mask_label: dict[int, {'mask': np.ndarray[bool], 'label': str}]
    """
    result_masks = {}
    original_encode = None
    try:
        frame_n_rgb = cv2.cvtColor(frame_n_bgr, cv2.COLOR_BGR2RGB)
        frame_n1_rgb = cv2.cvtColor(frame_n_plus_1_bgr, cv2.COLOR_BGR2RGB)
        pil_n = Image.fromarray(frame_n_rgb)
        pil_n1 = Image.fromarray(frame_n1_rgb)

        model_dtype = getattr(app, 'model_dtype', torch.float32)
        mini_session = app.tracker_processor.init_video_session(
            video=[pil_n, pil_n1],
            inference_device=app.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=model_dtype,
        )

        obj_ids_list = list(obj_id_to_mask_label.keys())
        if not obj_ids_list:
            return {}
        input_masks_list = [obj_id_to_mask_label[oid]['mask'] for oid in obj_ids_list]

        inputs_n = app.tracker_processor(images=pil_n, device=app.device, return_tensors="pt")
        inputs_n1 = app.tracker_processor(images=pil_n1, device=app.device, return_tensors="pt")

        app.tracker_processor.add_inputs_to_inference_session(
            inference_session=mini_session,
            frame_idx=0,
            obj_ids=list(obj_ids_list),
            input_masks=input_masks_list,
            original_size=inputs_n.original_sizes[0],
        )

        dlmi_enabled = bool(app.low_level_api_enabled_var.get()) if hasattr(app, 'low_level_api_enabled_var') else False
        original_encode = app.tracker_model._encode_new_memory

        if dlmi_enabled:
            dlmi_mode = app.dlmi_boundary_mode_var.get()
            dlmi_intensity = app.dlmi_alpha_var.get()
            dlmi_falloff = app.dlmi_gradient_falloff_var.get()
            injection_queue = dlmi_hooks.build_injection_queue(
                obj_ids=obj_ids_list,
                masks_by_oid={oid: obj_id_to_mask_label[oid]['mask'] for oid in obj_ids_list},
                intensity=dlmi_intensity, mode=dlmi_mode, falloff=dlmi_falloff,
                device=app.device,
            )
            app.tracker_model._encode_new_memory = dlmi_hooks.create_injection_hook(
                injection_queue, original_encode, log_prefix="mini-prop"
            )

        # Frame 0 forward (memory encoding possibly with injection)
        frame_n_tensor = inputs_n.pixel_values[0]
        if model_dtype == torch.float32 and frame_n_tensor.dtype != torch.float32:
            frame_n_tensor = frame_n_tensor.to(dtype=torch.float32)
        with torch.inference_mode():
            _ = app.tracker_model(inference_session=mini_session, frame=frame_n_tensor)

        # Restore hook before frame 1 forward pass
        if original_encode is not None:
            app.tracker_model._encode_new_memory = original_encode
            original_encode = None

        # Frame 1 forward (propagation via memory)
        frame_n1_tensor = inputs_n1.pixel_values[0]
        if model_dtype == torch.float32 and frame_n1_tensor.dtype != torch.float32:
            frame_n1_tensor = frame_n1_tensor.to(dtype=torch.float32)
        with torch.inference_mode():
            n1_outputs = app.tracker_model(inference_session=mini_session, frame=frame_n1_tensor)

        processed = app.tracker_processor.post_process_masks(
            [n1_outputs.pred_masks], original_sizes=inputs_n1.original_sizes, binarize=False
        )[0]

        target_h, target_w = frame_n_plus_1_bgr.shape[:2]
        tracked_ids = list(mini_session.obj_ids) if hasattr(mini_session, 'obj_ids') else obj_ids_list
        for i, oid in enumerate(tracked_ids):
            if i >= processed.shape[0]:
                continue
            mask_np = processed[i].cpu().numpy()
            mask_bin = np.squeeze(mask_np)
            if mask_bin.ndim > 2:
                mask_bin = mask_bin[0]
            if mask_bin.shape != (target_h, target_w):
                mask_bin = cv2.resize(
                    mask_bin.astype(np.float32), (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
            result_masks[int(oid)] = (mask_bin > 0.5)

        try:
            del mini_session
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logger.exception(f"Mini-video DLMI propagation failed: {e}")
        if original_encode is not None:
            try:
                app.tracker_model._encode_new_memory = original_encode
            except Exception:
                pass
        return {}

    return result_masks


def seed_session_with_masks(app, oid_to_mask):
    """Seed the current `inference_session` at frame 0 with the given obj_id→mask
    dict, and run a forward pass so memory is established. Used after Cut+DLMI to
    bootstrap the fresh session with the propagated masks as initial prompts."""
    if app.inference_session is None or app.current_cv_frame is None:
        return
    try:
        frame_rgb = cv2.cvtColor(app.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        inputs = app.tracker_processor(images=frame_pil, device=app.device, return_tensors="pt")

        oid_list = [int(o) for o in oid_to_mask.keys()]
        mask_list = [oid_to_mask[o].astype(bool) if hasattr(oid_to_mask[o], 'astype') else oid_to_mask[o]
                     for o in oid_list]
        if not oid_list:
            return

        app.tracker_processor.add_inputs_to_inference_session(
            inference_session=app.inference_session,
            frame_idx=0,
            obj_ids=oid_list,
            input_masks=mask_list,
            original_size=inputs.original_sizes[0],
        )

        model_dtype = getattr(app, 'model_dtype', torch.float32)
        frame_tensor = inputs.pixel_values[0]
        if model_dtype == torch.float32 and frame_tensor.dtype != torch.float32:
            frame_tensor = frame_tensor.to(dtype=torch.float32)
        with torch.inference_mode():
            _ = app.tracker_model(inference_session=app.inference_session, frame=frame_tensor)
        app.is_tracking_ever_started = True
        logger.info(f"Seeded new session with {len(oid_list)} propagated masks at frame 0.")
    except Exception as e:
        logger.exception(f"seed_session_with_masks failed: {e}")


def _yolo_dataset_prechecks(app):
    """Shared guard used by cut_and_{repropagate,dlmi_propagate,load_labels} when
    save_format is YOLO/both, OR when seg fmt is labelme but pose data exists
    (then pose still gets saved as YOLO-pose, which needs a proper YOLO
    dataset root with class names + data.yaml).
    Returns (ok: bool). False means user cancelled or init failed."""
    fmt = app.save_format_var.get()

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

    if fmt not in ["yolo", "both"] and not has_any_pose_overall:
        return True

    use_pose_root = bool(getattr(app, 'use_custom_pose_save_path_var', None) and
                         app.use_custom_pose_save_path_var.get())
    if use_pose_root and fmt == "labelme":
        save_dir = app.custom_pose_save_dir_var.get()
    else:
        save_dir = app._get_save_directory()

    check_result = app._check_existing_yolo_dataset(save_dir)
    if check_result is None:
        return False
    if check_result == "new_setup":
        if not app._prompt_yolo_class_info():
            return False
        if not app._init_yolo_dataset_structure(save_dir):
            messagebox.showerror("Error", "Failed to create YOLO dataset structure.", parent=app.root)
            return False
    return True


def cut_and_repropagate(app):
    """Classic Cut Here: save 0..n, reset state, resume from n+1 with empty tracked_objects."""
    slider_idx = app.review_current_frame
    if slider_idx < 0:
        messagebox.showwarning("Info", "Please select a valid frame.", parent=app.root)
        return

    current_offset = getattr(app, 'cut_start_frame', 0)
    absolute_cut_frame = current_offset + slider_idx
    next_start_frame = absolute_cut_frame + 1

    frames_to_save = [f for f in app.propagated_results.keys() if f <= slider_idx]

    if next_start_frame >= app.video_total_frames:
        messagebox.showwarning("Info", "This is the last frame. No next video to cut.\nUse normal label save.", parent=app.root)
        return

    response = messagebox.askyesnocancel(
        "Cut Here",
        f"Save labels for frames 0~{slider_idx} ({len(frames_to_save)} frames),\n"
        f"and treat original video frames {next_start_frame} onwards as new video.\n\n"
        f"(Original frame numbers: {current_offset}~{absolute_cut_frame} saved, start from {next_start_frame})\n\n"
        f"Yes: Save labels then cut\n"
        f"No: Cut without saving labels\n"
        f"Cancel: Abort operation",
        parent=app.root
    )
    if response is None:
        return
    save_labels = response

    if save_labels and not _yolo_dataset_prechecks(app):
        return

    saved_count = 0
    if save_labels:
        saved_count = save_frames_0_to_n(app, slider_idx, True, current_offset)
        logger.info(f"Cut: Saved {saved_count} frames for slider index 0~{slider_idx}. "
                    f"(Original frames {current_offset}~{absolute_cut_frame})")
    else:
        logger.info("Cut: Label saving skipped (user choice)")

    if not reset_state_and_seek(app, next_start_frame):
        return
    app.view.update_propagate_progress(
        0, f"Cut complete (new video from original frame {next_start_frame})"
    )

    remaining_frames = app.video_total_frames - next_start_frame
    if save_labels:
        app.update_status(
            f"Cut complete. {saved_count} frames saved (0~{absolute_cut_frame}).\n"
            f"Starting from original frame {next_start_frame} ({remaining_frames} frames remaining). "
            f"Add objects and start propagation."
        )
        messagebox.showinfo(
            "Cut Complete",
            f"Labels for {saved_count} frames (slider index 0~{slider_idx-1}) have been saved.\n"
            f"(Original video frames {current_offset}~{absolute_cut_frame-1})\n\n"
            f"From original frame {absolute_cut_frame}, it will be treated as a new video.\n"
            f"Add new objects and click 'Start Propagation' to proceed.",
            parent=app.root
        )
    else:
        app.update_status(
            f"Cut complete. (Label saving skipped)\n"
            f"Starting from original frame {next_start_frame} ({remaining_frames} frames remaining). "
            f"Add objects and start propagation."
        )
        messagebox.showinfo(
            "Cut Complete",
            f"Cut performed without saving labels.\n\n"
            f"From original frame {absolute_cut_frame}, it will be treated as a new video.\n"
            f"Add new objects and click 'Start Propagation' to proceed.",
            parent=app.root
        )


def cut_and_dlmi_propagate(app):
    """Cut at frame N (slider position): save 0..N-1, then start the new
    sub-video AT original frame N (one frame before a plain Cut Here would).
    The propagated masks already computed at frame N are reused as-is and
    DLMI-injected into a fresh inference session anchored at frame N — no
    mini-video propagation is needed because the masks the user already
    confirmed at N are exactly what should seed N going forward.
    """
    slider_idx = app.review_current_frame
    if slider_idx < 0:
        messagebox.showwarning("Info", "Please select a valid frame.", parent=app.root)
        return
    if slider_idx not in app.propagated_results:
        messagebox.showwarning("Info", f"Frame {slider_idx} has no propagated result.", parent=app.root)
        return

    current_offset = getattr(app, 'cut_start_frame', 0)
    absolute_cut_frame = current_offset + slider_idx
    next_start_frame = absolute_cut_frame  # start the new sub-video at frame N itself

    if next_start_frame >= app.video_total_frames:
        messagebox.showwarning("Info", "This is the last frame. Cannot cut+DLMI.", parent=app.root)
        return
    if slider_idx == 0:
        messagebox.showwarning(
            "Info",
            "Cut+DLMI requires at least one previously-labelled frame before the cut "
            "point (frame N-1). Cannot cut at the first frame.",
            parent=app.root
        )
        return

    frame_n_result = app.propagated_results[slider_idx]
    frame_n_bgr = frame_n_result['frame'].copy()
    frame_n_masks = {}
    for oid, odata in frame_n_result['masks'].items():
        m = odata.get('last_mask')
        if m is None or not m.any():
            continue
        frame_n_masks[int(oid)] = {
            'mask': m.astype(bool),
            'label': odata.get('custom_label', f'Object_{int(oid)}'),
        }
    if not frame_n_masks:
        messagebox.showwarning("Info", "No valid masks at current frame.", parent=app.root)
        return

    frames_to_save = [f for f in app.propagated_results.keys() if f <= slider_idx - 1]
    response = messagebox.askyesnocancel(
        "Cut + DLMI",
        f"Save labels for frames 0~{slider_idx - 1} ({len(frames_to_save)} frames),\n"
        f"then restart the new sub-video at frame {slider_idx} reusing its masks via DLMI?\n\n"
        f"Yes: Save then cut+DLMI\n"
        f"No: Cut+DLMI without saving labels\n"
        f"Cancel: Abort",
        parent=app.root
    )
    if response is None:
        return
    save_labels = response

    if save_labels and not _yolo_dataset_prechecks(app):
        return

    # Save 0..N-1 only — frame N is intentionally NOT saved here because it
    # becomes the starting frame of the new sub-video.
    saved_count = save_frames_0_to_n(app, slider_idx, save_labels, current_offset, inclusive=False)

    if not reset_state_and_seek(app, next_start_frame):
        return

    if frame_n_bgr is not None:
        app.current_cv_frame = frame_n_bgr

    # The new sub-video's frame 0 IS the original frame N. Use its already
    # computed propagated masks directly — no mini-video propagation, no
    # accuracy loss vs. the frame the user just reviewed.
    propagated_n1 = {oid: data['mask'] for oid, data in frame_n_masks.items()}

    if propagated_n1:
        synthetic_masks = {}
        for oid, mask in propagated_n1.items():
            if mask is None or not mask.any():
                continue
            label = frame_n_masks.get(oid, {}).get('label', f'Object_{oid}')
            mask_bool = mask.astype(bool)
            app.tracked_objects[oid] = {
                'custom_label': label,
                'last_mask': mask_bool,
                'is_polygon_object': False,
            }
            synthetic_masks[oid] = {
                'last_mask': mask_bool,
                'custom_label': label,
            }
            app.next_obj_id_to_propose = max(app.next_obj_id_to_propose, int(oid) + 1)
        app._update_obj_id_info_label()

        # Stamp a synthetic propagated_results[0] so any redraw triggered
        # by slider events (review_frame_slider.set(0) inside
        # reset_state_and_seek fires _on_review_slider_change →
        # on_review_frame_change(0)) finds this frame in propagated_results
        # and shows the masks instead of silently clearing them via the
        # empty-results fallback path.
        if synthetic_masks and app.current_cv_frame is not None:
            app.propagated_results[0] = {
                'frame': app.current_cv_frame.copy(),
                'masks': synthetic_masks,
            }

        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
        # Seed the fresh inference session at frame 0 (= original frame N)
        # with these masks, optionally with DLMI injection on the memory
        # encode path (controlled inside seed_session_with_masks via the
        # standard injection flag).
        seed_session_with_masks(app, propagated_n1)
        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())

    app.update_status(
        f"Cut+DLMI complete. {saved_count} frames saved (0~{absolute_cut_frame - 1}). "
        f"{len(propagated_n1)} objects reused at frame {next_start_frame}."
    )
    app.view.update_propagate_progress(
        0, f"Cut+DLMI complete (new video from original frame {next_start_frame})"
    )
    messagebox.showinfo(
        "Cut + DLMI Complete",
        f"{saved_count} frames saved (0~{absolute_cut_frame - 1}).\n"
        f"{len(propagated_n1)} objects reused at frame {next_start_frame} via DLMI.\n\n"
        f"Review the masks and click 'Start Propagation' to continue.",
        parent=app.root
    )

    # Final defensive redraw after the modal dialog closes.
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def cut_and_load_labels(app):
    """Cut at current frame, then open a file dialog to load labels for frame n+1."""
    slider_idx = app.review_current_frame
    if slider_idx < 0:
        messagebox.showwarning("Info", "Please select a valid frame.", parent=app.root)
        return
    if slider_idx not in app.propagated_results:
        messagebox.showwarning("Info", f"Frame {slider_idx} has no propagated result.", parent=app.root)
        return

    current_offset = getattr(app, 'cut_start_frame', 0)
    absolute_cut_frame = current_offset + slider_idx
    next_start_frame = absolute_cut_frame  # start the new sub-video at frame N itself

    if next_start_frame >= app.video_total_frames:
        messagebox.showwarning("Info", "This is the last frame. Cannot cut+load.", parent=app.root)
        return
    if slider_idx == 0:
        messagebox.showwarning(
            "Info",
            "Cut+Load requires at least one previously-labelled frame before the cut "
            "point (frame N-1). Cannot cut at the first frame.",
            parent=app.root
        )
        return

    # Capture frame N's existing propagated masks BEFORE reset wipes them.
    frame_n_result = app.propagated_results[slider_idx]
    frame_n_objects = []  # list of {'label', 'bbox', 'polygon'} like load_label_file's parsers emit
    for oid, odata in frame_n_result.get('masks', {}).items():
        m = odata.get('last_mask')
        if m is None or not m.any():
            continue
        label = odata.get('custom_label', f'Object_{int(oid)}')
        # Derive bbox + polygon contour from the existing mask so
        # _apply_loaded_labels_as_prompts (PVS) and
        # _apply_loaded_labels_as_polygon_masks (low-level API) both have what
        # they need. This mirrors the structure produced by
        # parse_labelme_json / parse_yolo_txt.
        ys, xs = np.where(m)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        polygon = None
        try:
            mask_uint8 = m.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 3:
                    polygon = [[int(p[0][0]), int(p[0][1])] for p in largest]
        except Exception:
            polygon = None
        obj_entry = {
            'label': label,
            'bbox': [x1, y1, x2, y2],
        }
        if polygon is not None:
            obj_entry['polygon'] = polygon
        frame_n_objects.append(obj_entry)

    if not frame_n_objects:
        messagebox.showwarning("Info", "No valid masks at current frame.", parent=app.root)
        return

    frames_to_save = [f for f in app.propagated_results.keys() if f <= slider_idx - 1]
    response = messagebox.askyesnocancel(
        "Cut + Load Labels",
        f"Save labels for frames 0~{slider_idx - 1} ({len(frames_to_save)} frames),\n"
        f"then restart the new sub-video at frame {slider_idx} reusing its masks via the\n"
        f"same path as the 'Load Labels' button?\n\n"
        f"Yes: Save then cut+load\n"
        f"No: Cut+load without saving labels\n"
        f"Cancel: Abort",
        parent=app.root
    )
    if response is None:
        return
    save_labels = response

    if save_labels and not _yolo_dataset_prechecks(app):
        return

    # Save 0..N-1 only — frame N is intentionally NOT saved here because it
    # becomes the starting frame of the new sub-video and will be re-emitted
    # from there.
    saved_count = save_frames_0_to_n(app, slider_idx, save_labels, current_offset, inclusive=False)

    if not reset_state_and_seek(app, next_start_frame):
        return

    # Feed the frame-N masks straight through the same path the Load Labels
    # button uses (PVS bbox prompts, PCS exemplars, or low-level polygon-mask
    # injection — picked by current prompt_mode_var and low_level_api_var).
    app._apply_loaded_objects(frame_n_objects, source_desc=f"cut+load@frame{slider_idx}")

    # Mirror tracked_objects into propagated_results[0] so any redraw
    # triggered by slider/notebook events finds the masks instead of falling
    # through to the empty-results fallback path that wipes the canvas.
    if app.tracked_objects and app.current_cv_frame is not None:
        synthetic_masks = {}
        for oid, data in app.tracked_objects.items():
            mask = data.get('last_mask') if data else None
            if mask is None or not mask.any():
                continue
            synthetic_masks[oid] = {
                'last_mask': mask,
                'custom_label': data.get('custom_label', f'Object_{oid}'),
            }
        if synthetic_masks:
            app.propagated_results[0] = {
                'frame': app.current_cv_frame.copy(),
                'masks': synthetic_masks,
            }
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())

    app.update_status(
        f"Cut+Load complete. {saved_count} frames saved (0~{absolute_cut_frame - 1}). "
        f"{len(frame_n_objects)} objects reused at frame {next_start_frame}."
    )
    app.view.update_propagate_progress(
        0, f"Cut+Load complete (new video from original frame {next_start_frame})"
    )


def repropagate_all(app):
    """Full re-propagation from frame 0, keeping currently-defined objects."""
    response = messagebox.askyesno(
        "Confirm Full Re-propagation",
        "Do you want to delete all results and re-propagate from the beginning?\n\n"
        "Currently defined objects will be maintained.",
        parent=app.root
    )
    if not response:
        return

    app.propagated_results = {}
    app.cut_point_frame = None

    if app.cap:
        app.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame_bgr = app.cap.read()
        if ret:
            app.current_cv_frame = frame_bgr
            app._display_cv_frame_on_view(frame_bgr, app._get_current_masks_for_display())

    app.view.enable_review_controls(False)
    app.app_state = "IDLE"
    app.view.update_propagate_progress(0, "Ready for re-propagation")
    app.update_status("Re-propagation ready. Click 'Start Propagation' to begin.")
