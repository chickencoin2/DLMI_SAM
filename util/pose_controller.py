"""Pose business logic: per-object pose mutations, auto-match class assignment,
TAPNext++/YOLO pose integration, tracker lifecycle, and TAPNext checkpoint
auto-download.

This module is the "controller" layer for everything pose. The `pose_ui` module
owns UI widgets (settings dialog, canvas rendering, hit-test) and
`pose_tracker` owns the model wrappers. `pose_controller` glues them together
into app-level operations that the main `SAM3AutolabelApp` exposes via thin
method wrappers.

All public functions take `app` (the app instance) as the first parameter.
"""
import os
import logging
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict

import cv2
import numpy as np

from . import ui_dialogs

logger = logging.getLogger("DLMI_SAM_LABELER.PoseController")

TAPNEXTPP_DEFAULT_URL = "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt"


def _track_bidirectional(tracker, frames_rgb, query_entries):
    """Run TAPNext-style tracking with per-query anchor frames, producing
    tracks in BOTH time directions (forward AND backward from each anchor).

    TAPNext++ is trained as next-token prediction so a single forward pass
    only tracks from a query's anchor-frame towards the end of the clip;
    earlier frames would be filled with the anchor coordinate. To get proper
    bidirectional coverage we run the tracker twice per unique anchor t:
      - Forward  on frames[t:]   with the query at t=0 of that slice
      - Backward on reversed(frames[:t+1]) with the query at t=0 of that slice
    Results are merged into a single [T, N, 2] track array.

    Query entries are expected as (oid, kpt_idx, t_anchor, x, y). The function
    is agnostic to oid/kpt_idx — it only uses t_anchor + (x, y).
    """
    T = len(frames_rgb)
    N = len(query_entries)
    if N == 0:
        return np.zeros((T, 0, 2), dtype=np.float32), np.zeros((T, 0), dtype=np.bool_)

    tracks = np.zeros((T, N, 2), dtype=np.float32)
    vis = np.zeros((T, N), dtype=np.bool_)

    # Initialise with anchor coords so any frames never written by either
    # pass still have a sensible (static) position instead of zero.
    for qi, (_oid, _k, t_anchor, x, y) in enumerate(query_entries):
        tracks[:, qi] = [float(x), float(y)]

    by_t = {}
    for qi, (_oid, _k, t_anchor, x, y) in enumerate(query_entries):
        by_t.setdefault(int(t_anchor), []).append((qi, float(x), float(y)))

    logger.info(f"_track_bidirectional: T={T} frames, N={N} queries, "
                f"anchor_frames={sorted(by_t.keys())}")

    for t_anchor, qlist in by_t.items():
        if not (0 <= t_anchor < T):
            continue
        q_indices = [q[0] for q in qlist]
        xy = np.array([[q[1], q[2]] for q in qlist], dtype=np.float32)

        # Forward from t_anchor
        if t_anchor < T:
            fwd_frames = frames_rgb[t_anchor:]
            if len(fwd_frames) >= 1:
                q_fwd = np.column_stack([
                    np.zeros((len(qlist),), dtype=np.float32), xy
                ])  # [N, 3] = (t=0, x, y)
                logger.info(f"  forward pass: t_anchor={t_anchor}, "
                            f"frames[{t_anchor}:{T}]={len(fwd_frames)} frames, "
                            f"queries={len(qlist)}")
                tr_fwd, vis_fwd = tracker.track(fwd_frames, q_fwd)
                # tr_fwd has shape [len(fwd_frames), len(qlist), 2]
                # Debug: log a representative query's track across frames so we
                # can tell if the model output actually moves or stays static.
                if len(qlist) > 0 and tr_fwd.shape[0] >= 2:
                    sample_qi = 0
                    first_pos = tr_fwd[0, sample_qi]
                    mid_pos = tr_fwd[tr_fwd.shape[0] // 2, sample_qi]
                    last_pos = tr_fwd[-1, sample_qi]
                    orig_xy = xy[sample_qi]
                    logger.info(
                        f"  forward result (q{q_indices[sample_qi]}): "
                        f"input=({orig_xy[0]:.0f},{orig_xy[1]:.0f}) "
                        f"→ t=anchor:({first_pos[0]:.0f},{first_pos[1]:.0f}) "
                        f"t=mid:({mid_pos[0]:.0f},{mid_pos[1]:.0f}) "
                        f"t=end:({last_pos[0]:.0f},{last_pos[1]:.0f})"
                    )
                for i, qi in enumerate(q_indices):
                    for ti_rel in range(tr_fwd.shape[0]):
                        global_t = t_anchor + ti_rel
                        if 0 <= global_t < T:
                            tracks[global_t, qi] = tr_fwd[ti_rel, i]
                            vis[global_t, qi] = vis_fwd[ti_rel, i]

        # Backward from t_anchor: run on reversed frames[:t_anchor+1]
        if t_anchor > 0:
            bwd_frames = frames_rgb[:t_anchor + 1][::-1]
            q_bwd = np.column_stack([
                np.zeros((len(qlist),), dtype=np.float32), xy
            ])
            logger.info(f"  backward pass: t_anchor={t_anchor}, "
                        f"reversed frames[0:{t_anchor+1}]={len(bwd_frames)} frames, "
                        f"queries={len(qlist)}")
            tr_bwd, vis_bwd = tracker.track(bwd_frames, q_bwd)
            # tr_bwd[0] is at t_anchor; tr_bwd[i] is at t_anchor - i
            for i, qi in enumerate(q_indices):
                for ti_rel in range(tr_bwd.shape[0]):
                    global_t = t_anchor - ti_rel
                    if 0 <= global_t < T:
                        # Backward overrides forward at t_anchor (same value anyway)
                        tracks[global_t, qi] = tr_bwd[ti_rel, i]
                        vis[global_t, qi] = vis_bwd[ti_rel, i]

    return tracks, vis


# ---- UI / schema helpers ---------------------------------------------------

def default_pose_class_name(app):
    try:
        schema = (app.pose_config or {}).get("schema") or {}
        if schema:
            return next(iter(schema.keys()))
    except Exception:
        pass
    return "person"


def open_pose_settings(app):
    """Open the TAPNext++ pose settings dialog."""
    if app._pose_ui is None:
        messagebox.showerror("Error", "pose_ui module not available.", parent=app.root)
        return
    try:
        app._pose_ui.open_pose_settings_dialog(app)
        refresh_pose_class_menu(app)
    except Exception as e:
        logger.exception(f"open_pose_settings failed: {e}")
        messagebox.showerror("Error", f"Pose settings dialog failed:\n{e}", parent=app.root)


def refresh_pose_class_menu(app):
    """Rebuild the Class OptionMenu with the current pose_config schema."""
    if not app.view or not hasattr(app.view, 'pose_class_menu'):
        return
    try:
        schema = (app.pose_config or {}).get('schema') or {}
        classes = list(schema.keys()) or ['(no schema)']
        menu = app.view.pose_class_menu['menu']
        menu.delete(0, 'end')
        for cname in classes:
            menu.add_command(label=cname, command=lambda v=cname: app.view.pose_class_var.set(v))
    except (tk.TclError, AttributeError):
        pass


def update_pose_class_display(app):
    """Sync the class dropdown text with the currently-selected object's pose_class."""
    if not app.view or not hasattr(app.view, 'pose_class_var'):
        return
    try:
        oid = app.selected_object_sam_id
        if oid is not None and oid in app.tracked_objects:
            cls = app.tracked_objects[oid].get('pose_class', default_pose_class_name(app))
            current = app.view.pose_class_var.get()
            if current != cls:
                app._suppress_pose_class_trace = True
                try:
                    app.view.pose_class_var.set(cls or '')
                finally:
                    app._suppress_pose_class_trace = False
            app.view.btn_pose_delete_obj.config(
                state=('normal' if app.tracked_objects[oid].get('pose_points') else 'disabled')
            )
        else:
            app.view.btn_pose_delete_obj.config(state='disabled')
    except (tk.TclError, AttributeError):
        pass


def on_pose_class_selected(app):
    """Apply the chosen class to the currently-selected object. Also updates
    pose_edges to the schema default if they were empty."""
    if getattr(app, '_suppress_pose_class_trace', False):
        return
    if not app.view:
        return
    oid = app.selected_object_sam_id
    if oid is None or oid not in app.tracked_objects:
        return
    try:
        new_cls = app.view.pose_class_var.get()
    except (tk.TclError, AttributeError):
        return
    if not new_cls or new_cls == '(no schema)':
        return
    obj = app.tracked_objects[oid]
    obj['pose_class'] = new_cls
    obj['_automatch_done'] = True
    schema = (app.pose_config or {}).get('schema', {}).get(new_cls, {})
    if schema and not obj.get('pose_edges'):
        obj['pose_edges'] = [list(e) for e in schema.get('skeleton', [])]
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    app.update_status(f"Object {oid} pose class set to '{new_cls}' (manual)")


# ---- Pose point CRUD -------------------------------------------------------

def add_pose_point_at(app, img_x, img_y):
    """Add a pose keypoint at image coords to the currently selected
    tracked_object. If no object is selected, create a new pose-only object.
    If Chain mode is ON and a previous point exists, auto-connect."""
    target_oid = app.selected_object_sam_id
    if target_oid is None or target_oid not in app.tracked_objects:
        target_oid = app.next_obj_id_to_propose
        app.next_obj_id_to_propose += 1
        label = app.default_object_label_var.get() or f"Pose_{target_oid}"
        app.tracked_objects[target_oid] = {
            'custom_label': label,
            'pose_points': [],
            'pose_edges': [],
            'pose_class': default_pose_class_name(app),
        }
        app.selected_object_sam_id = target_oid
        app._update_obj_id_info_label()

    obj = app.tracked_objects.get(target_oid)
    if obj is None:
        return
    obj.setdefault('pose_points', [])
    obj.setdefault('pose_edges', [])
    obj.setdefault('pose_class', default_pose_class_name(app))

    new_kpt_idx = len(obj['pose_points'])
    # `cur_slider` is the frame the user is VISUALLY looking at right now
    # (relative to current propagation session). We anchor the pose to that
    # frame unconditionally — whether app_state is REVIEWING (post-propagate
    # review), IDLE (before propagate), or anything else. Storing frame_idx=0
    # regardless of actual view frame was the root cause of "labelled at
    # frame 0 but user was at frame N" bug.
    cur_slider = int(getattr(app, 'review_current_frame', 0) or 0)
    in_propagated = cur_slider in app.propagated_results
    new_point = {
        'x': int(img_x), 'y': int(img_y),
        'visibility': 2, 'kpt_idx': new_kpt_idx,
        'user_prompt': True,
        'frame_idx': cur_slider,
    }
    obj['pose_points'].append(new_point)

    if app.pose_chain_mode_var.get() and new_kpt_idx > 0:
        edge = [new_kpt_idx - 1, new_kpt_idx]
        if edge not in obj['pose_edges']:
            obj['pose_edges'].append(edge)

    # Mirror into per-frame propagated_results so TAPNext can pick it up as a
    # query at this specific frame (mid-video prompt).
    if in_propagated:
        frame_slot = app.propagated_results[cur_slider].setdefault('masks', {}).setdefault(target_oid, {})
        frame_pts = frame_slot.setdefault('pose_points', [])
        while len(frame_pts) <= new_kpt_idx:
            frame_pts.append({'x': 0, 'y': 0, 'visibility': 0, 'kpt_idx': len(frame_pts)})
        frame_pts[new_kpt_idx] = dict(new_point)
        if 'pose_edges' not in frame_slot and obj.get('pose_edges'):
            frame_slot['pose_edges'] = [list(e) for e in obj['pose_edges']]
        elif app.pose_chain_mode_var.get() and new_kpt_idx > 0:
            fe = frame_slot.setdefault('pose_edges', [])
            if [new_kpt_idx - 1, new_kpt_idx] not in fe:
                fe.append([new_kpt_idx - 1, new_kpt_idx])
        if obj.get('custom_label') and 'custom_label' not in frame_slot:
            frame_slot['custom_label'] = obj['custom_label']
        if obj.get('pose_class') and 'pose_class' not in frame_slot:
            frame_slot['pose_class'] = obj['pose_class']

    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    logger.info(f"Pose point added: obj={target_oid} kpt={new_kpt_idx} at "
                f"({img_x},{img_y}) frame_idx={new_point['frame_idx']} (user_prompt)")


def new_pose_object(app):
    """Create a new empty pose-only tracked_object and make it the current
    target for subsequent Add Pose clicks."""
    new_oid = app.next_obj_id_to_propose
    app.next_obj_id_to_propose += 1
    label_base = (app.default_object_label_var.get() or "Pose").strip() or "Pose"
    app.tracked_objects[new_oid] = {
        'custom_label': f"{label_base}_{new_oid}",
        'pose_points': [],
        'pose_edges': [],
        'pose_class': default_pose_class_name(app),
    }
    app.selected_object_sam_id = new_oid
    app.selected_objects_sam_ids = {new_oid}
    clear_pose_selection(app)
    app._update_obj_id_info_label()
    app.update_status(f"New pose object created (id={new_oid}). Click canvas in Add Pose mode to add keypoints.")


def toggle_pose_point_selection(app, obj_id, kpt_idx):
    key = (int(obj_id), int(kpt_idx))
    if key in app.selected_pose_points:
        app.selected_pose_points.discard(key)
    else:
        app.selected_pose_points.add(key)
    update_pose_action_button_states(app)
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def clear_pose_selection(app):
    if not app.selected_pose_points:
        update_pose_action_button_states(app)
        return
    app.selected_pose_points.clear()
    update_pose_action_button_states(app)
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def update_pose_action_button_states(app):
    if not app.view:
        return
    n_sel = len(app.selected_pose_points)
    try:
        app.view.btn_pose_connect.config(state=('normal' if n_sel >= 2 else 'disabled'))
        app.view.btn_pose_delete.config(state=('normal' if n_sel >= 1 else 'disabled'))
    except (tk.TclError, AttributeError):
        pass
    try:
        if n_sel == 1:
            oid, kidx = next(iter(app.selected_pose_points))
            app.view.pose_idx_var.set(str(kidx))
            app.view.pose_idx_entry.config(state=tk.NORMAL)
            app.view.btn_pose_set_idx.config(state=tk.NORMAL)
        else:
            app.view.pose_idx_var.set("")
            app.view.pose_idx_entry.config(state=tk.DISABLED)
            app.view.btn_pose_set_idx.config(state=tk.DISABLED)
    except (tk.TclError, AttributeError):
        pass
    try:
        app.view.btn_pose_toggle_vis.config(state=('normal' if n_sel >= 1 else 'disabled'))
    except (tk.TclError, AttributeError):
        pass
    update_pose_class_display(app)


def delete_selected_object_pose(app):
    """Remove ALL pose data (points, edges) from the currently selected object."""
    oid = app.selected_object_sam_id
    if oid is None or oid not in app.tracked_objects:
        messagebox.showinfo("Pose", "No object selected. Ctrl+click an object first.", parent=app.root)
        return
    obj = app.tracked_objects[oid]
    if not obj.get('pose_points'):
        messagebox.showinfo("Pose", f"Object {oid} has no pose data.", parent=app.root)
        return
    n = len(obj.get('pose_points', []))
    if not messagebox.askyesno("Confirm",
                               f"Delete ALL {n} pose points from object {oid}?",
                               parent=app.root):
        return
    obj.pop('pose_points', None)
    obj.pop('pose_edges', None)
    app.selected_pose_points = {(o, k) for (o, k) in app.selected_pose_points if o != oid}
    update_pose_action_button_states(app)
    update_pose_class_display(app)
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    app.update_status(f"Removed pose from object {oid} ({n} points).")


def select_pose_chain_at(app, img_x, img_y):
    """Shift+right-click handler: BFS the connected component of the pose
    point nearest (img_x, img_y) and select all reachable keypoints."""
    if app._pose_ui is None:
        return
    hit = app._pose_ui.hit_test_pose_point(app, img_x, img_y)
    if hit is None:
        return
    oid, kidx = hit
    obj = app.tracked_objects.get(oid)
    if not obj:
        return
    edges = obj.get('pose_edges', []) or []
    adjacency = defaultdict(set)
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            a, b = int(edge[0]), int(edge[1])
            adjacency[a].add(b)
            adjacency[b].add(a)
    visited = {kidx}
    stack = [kidx]
    while stack:
        cur = stack.pop()
        for nb in adjacency.get(cur, ()):
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    app.selected_pose_points = {(int(oid), int(k)) for k in visited}
    update_pose_action_button_states(app)
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    app.update_status(f"Selected chain: {len(visited)} point(s) on object {oid}.")


def toggle_selected_pose_visibility(app):
    """Cycle visibility of every selected pose point between 2 (visible) and
    1 (occluded). v=0 (not labeled) is reserved; use delete instead."""
    if not app.selected_pose_points:
        return
    changed = 0
    for (oid, kidx) in list(app.selected_pose_points):
        obj = app.tracked_objects.get(oid)
        if not obj:
            continue
        pts = obj.get('pose_points', [])
        if kidx < 0 or kidx >= len(pts):
            continue
        cur = int(pts[kidx].get('visibility', 2))
        pts[kidx]['visibility'] = 1 if cur >= 2 else 2
        changed += 1
    if changed > 0:
        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
        app.update_status(f"Toggled visibility on {changed} pose point(s). (2=visible, 1=occluded)")


def reassign_selected_pose_idx(app):
    """Swap the kpt_idx of the single-selected pose point with another index."""
    if len(app.selected_pose_points) != 1:
        return
    oid, kidx = next(iter(app.selected_pose_points))
    obj = app.tracked_objects.get(oid)
    if obj is None:
        return
    pts = obj.get('pose_points', [])
    if kidx < 0 or kidx >= len(pts):
        return
    try:
        new_idx = int(app.view.pose_idx_var.get())
    except (ValueError, AttributeError):
        messagebox.showwarning("Pose", "Enter a valid integer index.", parent=app.root)
        return
    if new_idx < 0 or new_idx >= len(pts):
        messagebox.showwarning("Pose", f"Index must be in [0, {len(pts)-1}].", parent=app.root)
        return
    if new_idx == kidx:
        return
    pts[kidx], pts[new_idx] = pts[new_idx], pts[kidx]
    for i, p in enumerate(pts):
        p['kpt_idx'] = i
    new_edges = []
    for edge in obj.get('pose_edges', []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        a, b = int(edge[0]), int(edge[1])
        if a == kidx:
            a = new_idx
        elif a == new_idx:
            a = kidx
        if b == kidx:
            b = new_idx
        elif b == new_idx:
            b = kidx
        new_edges.append([min(a, b), max(a, b)])
    obj['pose_edges'] = new_edges
    app.selected_pose_points = {(oid, new_idx)}
    update_pose_action_button_states(app)
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    app.update_status(f"Pose point renumbered: obj={oid} {kidx} \u2194 {new_idx}")


def connect_selected_pose_points(app):
    """Toggle edges between consecutive (by kpt_idx) selected points within each object."""
    if len(app.selected_pose_points) < 2:
        return
    by_obj = defaultdict(list)
    for oid, kidx in app.selected_pose_points:
        by_obj[oid].append(kidx)
    for oid, kidxs in by_obj.items():
        kidxs = sorted(set(kidxs))
        obj = app.tracked_objects.get(oid)
        if obj is None or len(kidxs) < 2:
            continue
        if 'pose_edges' not in obj:
            obj['pose_edges'] = []
        for i in range(len(kidxs) - 1):
            a, b = kidxs[i], kidxs[i + 1]
            edge = [min(a, b), max(a, b)]
            if edge in obj['pose_edges']:
                obj['pose_edges'].remove(edge)
            else:
                obj['pose_edges'].append(edge)
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


def delete_selected_pose_points(app):
    if not app.selected_pose_points:
        return
    by_obj = defaultdict(set)
    for oid, kidx in app.selected_pose_points:
        by_obj[oid].add(int(kidx))
    for oid, remove_idx_set in by_obj.items():
        obj = app.tracked_objects.get(oid)
        if obj is None:
            continue
        pts = obj.get('pose_points', [])
        edges = obj.get('pose_edges', [])
        keep = [i for i in range(len(pts)) if i not in remove_idx_set]
        remap = {old: new for new, old in enumerate(keep)}
        new_pts = [pts[i] for i in keep]
        for new_i, p in enumerate(new_pts):
            p['kpt_idx'] = new_i
        new_edges = []
        for edge in edges:
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                a, b = int(edge[0]), int(edge[1])
                if a in remap and b in remap:
                    new_edges.append([remap[a], remap[b]])
        obj['pose_points'] = new_pts
        obj['pose_edges'] = new_edges
    app.selected_pose_points.clear()
    update_pose_action_button_states(app)
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())


# ---- Seeds snapshot (pre-propagation) --------------------------------------

def snapshot_pose_queries_and_hide(app):
    """Capture the current per-object pose_points/pose_edges as the initial
    TAPNext++ query seeds, then remove them from `tracked_objects` so they
    are not rendered during SAM3 propagation. They are restored frame-by-frame
    after TAPNext post-process populates `propagated_results`."""
    seeds = {}
    for oid, obj in list(app.tracked_objects.items()):
        pts = obj.get('pose_points')
        if not pts:
            continue
        seeds[int(oid)] = {
            'points': [dict(p) for p in pts],
            'edges': [list(e) for e in obj.get('pose_edges', [])],
            'pose_class': obj.get('pose_class'),
            'custom_label': obj.get('custom_label'),
        }
        obj.pop('pose_points', None)
        obj.pop('pose_edges', None)
    app._pose_query_seeds = seeds
    app.selected_pose_points.clear()
    if seeds and app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
        logger.info(f"Pose queries snapshotted ({len(seeds)} objects); display cleared until TAPNext runs.")


# ---- Auto-match ------------------------------------------------------------

def automatch_classify_pose_object(app, oid, force=False):
    """Assign a pose_class to object `oid` by matching against
    pose_config.schema. Runs exactly once per object unless force=True."""
    obj = app.tracked_objects.get(oid)
    if not obj:
        return False
    pose_pts = obj.get('pose_points')
    if not pose_pts:
        return False
    if not force and obj.get('_automatch_done'):
        return False

    schema = (app.pose_config or {}).get('schema') or {}
    if not schema:
        return False

    overlaps = []
    for sid, sobj in app.tracked_objects.items():
        if int(sid) == int(oid):
            continue
        mask = sobj.get('last_mask')
        if mask is None or not hasattr(mask, 'any') or not mask.any():
            continue
        H, W = mask.shape[:2]
        inside = 0
        for p in pose_pts:
            x = int(p.get('x', 0)); y = int(p.get('y', 0))
            if 0 <= x < W and 0 <= y < H and bool(mask[y, x]):
                inside += 1
        if inside > 0:
            overlaps.append({
                'id': int(sid),
                'label': sobj.get('custom_label', '') or '',
                'inside': inside,
                'area': int(mask.sum()),
            })
    overlap_labels = {o['label'] for o in overlaps if o['label']}
    count = len(pose_pts)

    chosen_class = None
    chosen_reason = None

    for cname, cdef in schema.items():
        spans = [s for s in (cdef.get('span_objects') or []) if s]
        if spans and set(spans) <= overlap_labels:
            chosen_class = cname
            chosen_reason = f"span_objects={spans}"
            break

    if chosen_class is None:
        count_matches = [c for c, d in schema.items()
                         if len(d.get('keypoints', [])) == count]
        if len(count_matches) == 1:
            chosen_class = count_matches[0]
            chosen_reason = f"count={count}"
        elif len(count_matches) > 1:
            for c in count_matches:
                spans = set(schema[c].get('span_objects', []) or [])
                if spans and spans <= overlap_labels:
                    chosen_class = c
                    chosen_reason = "count+span"
                    break
            if chosen_class is None:
                chosen_class = count_matches[0]
                chosen_reason = f"count={count} (first of {len(count_matches)})"

    if chosen_class is None and overlaps:
        for o in sorted(overlaps, key=lambda x: -x['area']):
            if o['label'] in schema:
                chosen_class = o['label']
                chosen_reason = f"overlap-largest='{o['label']}'"
                break

    if chosen_class is None:
        return False

    obj['pose_class'] = chosen_class
    sch = schema.get(chosen_class, {})
    if not obj.get('pose_edges'):
        obj['pose_edges'] = [list(e) for e in sch.get('skeleton', [])]
    obj['_automatch_done'] = True
    logger.info(f"Auto-match: obj {oid} -> class '{chosen_class}' ({chosen_reason})")
    return True


def automatch_all_new_pose_objects(app):
    """Walk tracked_objects and classify any pose-bearing object that has not
    yet been auto-matched. Called when the user exits Add Pose mode."""
    if not getattr(app, 'pose_automatch_var', None) or not app.pose_automatch_var.get():
        return 0
    schema = (app.pose_config or {}).get('schema') or {}
    if not schema:
        return 0
    classified = 0
    for oid in list(app.tracked_objects.keys()):
        if automatch_classify_pose_object(app, oid, force=False):
            classified += 1
    if classified > 0:
        update_pose_class_display(app)
        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    return classified


def try_automatch_pose_to_segments(app, min_ratio=0.7):
    """When `pose_automatch_var` is ON, merge pose-only objects into segment
    objects when ≥min_ratio of their visible points fall inside the segment
    mask. Returns the number of merges."""
    if not getattr(app, 'pose_automatch_var', None) or not app.pose_automatch_var.get():
        return 0
    pose_only_oids = []
    seg_oids = []
    for oid, obj in app.tracked_objects.items():
        has_pose = bool(obj.get('pose_points'))
        mask = obj.get('last_mask')
        has_seg = mask is not None and hasattr(mask, 'any') and mask.any()
        if has_pose and not has_seg:
            pose_only_oids.append(oid)
        elif has_seg:
            seg_oids.append(oid)
    if not pose_only_oids or not seg_oids:
        return 0
    merges = 0
    for po_id in list(pose_only_oids):
        po = app.tracked_objects.get(po_id)
        if po is None:
            continue
        pts = po.get('pose_points', [])
        if not pts:
            continue
        best_seg = None
        best_ratio = 0.0
        for sid in seg_oids:
            seg = app.tracked_objects.get(sid)
            if seg is None:
                continue
            mask = seg.get('last_mask')
            if mask is None or not mask.any():
                continue
            H, W = mask.shape[:2]
            inside = 0
            total = 0
            for p in pts:
                if p.get('visibility', 1) <= 0:
                    continue
                total += 1
                x = int(p['x']); y = int(p['y'])
                if 0 <= x < W and 0 <= y < H and bool(mask[y, x]):
                    inside += 1
            if total == 0:
                continue
            ratio = inside / float(total)
            if ratio > best_ratio and ratio >= min_ratio:
                best_ratio = ratio
                best_seg = sid
        if best_seg is not None:
            tgt = app.tracked_objects[best_seg]
            tgt['pose_points'] = list(pts)
            tgt['pose_edges'] = list(po.get('pose_edges', []))
            if po.get('pose_class'):
                tgt['pose_class'] = po['pose_class']
            try:
                del app.tracked_objects[po_id]
            except KeyError:
                pass
            app.selected_pose_points = {(o, k) for (o, k) in app.selected_pose_points if o != po_id}
            if app.selected_object_sam_id == po_id:
                app.selected_object_sam_id = best_seg
            if po_id in app.selected_objects_sam_ids:
                app.selected_objects_sam_ids.discard(po_id)
                app.selected_objects_sam_ids.add(best_seg)
            merges += 1
            logger.info(f"Auto-match: pose obj {po_id} \u2192 seg obj {best_seg} (ratio={best_ratio:.2f})")
    if merges > 0:
        app._update_obj_id_info_label()
        update_pose_action_button_states(app)
        if app.current_cv_frame is not None:
            app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    return merges


# ---- Tracker lifecycle (TAPNext++ / YOLO pose) -----------------------------

def default_pose_models_dir(app):
    """Return the on-disk directory where the TAPNext++ checkpoint lives by default."""
    sam3_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py")))
    return os.path.join(sam3_dir, "util", "pose_models")


def ensure_tapnext_ckpt(app, interactive=True):
    """Resolve a usable TAPNext++ checkpoint path. If the config path is empty
    or missing, download the default checkpoint into util/pose_models/
    and persist the path to pose_config.json."""
    cfg = app.pose_config or {}
    ckpt = (cfg.get("tapnext_ckpt") or "").strip()
    if ckpt and os.path.exists(ckpt):
        return ckpt

    models_dir = default_pose_models_dir(app)
    try:
        os.makedirs(models_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create pose_models dir: {e}")
        return None
    default_ckpt = os.path.join(models_dir, "tapnextpp_ckpt.pt")

    if os.path.exists(default_ckpt):
        cfg["tapnext_ckpt"] = default_ckpt
        app.pose_config = cfg
        if app._pose_ui is not None:
            app._pose_ui.save_pose_config(app.pose_config_path, cfg)
        return default_ckpt

    url = TAPNEXTPP_DEFAULT_URL
    if interactive:
        proceed = messagebox.askyesno(
            "Download TAPNext++ Checkpoint",
            "TAPNext++ checkpoint (~2.5 GB) is not found locally.\n\n"
            f"Source:\n{url}\n\n"
            f"Destination:\n{default_ckpt}\n\n"
            "Download now? (May take a few minutes depending on network.)",
            parent=app.root
        )
        if not proceed:
            app.update_status("TAPNext++ download cancelled by user.")
            return None
    logger.info(f"Downloading TAPNext++ checkpoint: {url} \u2192 {default_ckpt}")

    dlg = ui_dialogs.open_download_dialog(app, url, default_ckpt) if interactive else None
    tmp_path = default_ckpt + ".part"

    def _report(block_num, block_size, total_size):
        if dlg is not None and dlg.get("cancel_flag", {}).get("flag"):
            raise RuntimeError("user-cancelled")
        try:
            if total_size and total_size > 0:
                downloaded = min(block_num * block_size, total_size)
                pct = downloaded * 100 / total_size
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                line = f"{pct:.1f}% \u2014 {mb:.1f} / {total_mb:.1f} MB"
                if dlg is not None:
                    dlg["bar"]['value'] = pct
                    dlg["status"].config(text=line)
                    dlg["window"].update_idletasks()
                app.update_status(f"Downloading TAPNext++: {line}")
            else:
                downloaded = block_num * block_size
                mb = downloaded / (1024 * 1024)
                line = f"{mb:.1f} MB"
                if dlg is not None:
                    dlg["status"].config(text=f"Downloaded: {line}")
                    dlg["window"].update_idletasks()
                app.update_status(f"Downloading TAPNext++: {line}")
        except Exception:
            pass

    try:
        import urllib.request
        urllib.request.urlretrieve(url, tmp_path, reporthook=_report)
        os.replace(tmp_path, default_ckpt)
        cfg["tapnext_ckpt"] = default_ckpt
        app.pose_config = cfg
        if app._pose_ui is not None:
            app._pose_ui.save_pose_config(app.pose_config_path, cfg)
        if dlg is not None:
            dlg["window"].destroy()
        if interactive:
            app.update_status(f"TAPNext++ checkpoint ready: {default_ckpt}")
        logger.info(f"TAPNext++ checkpoint downloaded to {default_ckpt}")
        return default_ckpt
    except Exception as e:
        logger.exception(f"TAPNext++ ckpt download failed: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        if dlg is not None:
            try:
                dlg["window"].destroy()
            except Exception:
                pass
        if interactive:
            app.update_status(f"TAPNext++ ckpt download failed: {e}")
            messagebox.showerror("Download Failed",
                                 f"TAPNext++ download failed:\n{e}", parent=app.root)
        return None


def offer_pose_fallback(app, reason=""):
    """Offer the user an optical-flow fallback when TAPNext++ is unusable."""
    from .pose_tracker import OpticalFlowPoseTracker
    proceed = messagebox.askyesno(
        "Pose Tracker",
        f"{reason}\n\nUse OpenCV optical-flow fallback instead?\n"
        "(Lower accuracy than TAPNext++, but runs without extra dependencies.)",
        parent=app.root
    )
    if not proceed:
        return None
    app.pose_tracker = OpticalFlowPoseTracker()
    app.update_status("Pose tracker: using OpenCV optical-flow fallback.")
    return app.pose_tracker


def get_pose_tracker(app):
    cfg = app.pose_config or {}
    device = cfg.get("device", "cuda")
    ckpt = ensure_tapnext_ckpt(app, interactive=True)
    if not ckpt:
        return offer_pose_fallback(app, reason="TAPNext++ checkpoint could not be prepared.")
    if app.pose_tracker is not None and getattr(app.pose_tracker, 'ckpt_path', None) == ckpt:
        return app.pose_tracker
    from .pose_tracker import TAPNextPPTracker, LibraryMissingError

    dlg = ui_dialogs.open_loading_dialog(
        app,
        "Preparing TAPNext++",
        "Installing dependencies and loading the model into GPU memory.\n"
        "First run can take a minute or two."
    )

    def _status_cb(msg):
        try:
            if dlg:
                dlg["status"].config(text=msg)
                dlg["window"].update_idletasks()
            app.update_status(msg)
            app.root.update_idletasks()
        except Exception:
            pass

    try:
        app.pose_tracker = TAPNextPPTracker(ckpt_path=ckpt, device=device, auto_install_cb=_status_cb)
        dlg["close"]()
        return app.pose_tracker
    except LibraryMissingError as lib_err:
        dlg["close"]()
        logger.warning(f"TAPNext++ library missing and auto-install failed: {lib_err}")
        return offer_pose_fallback(app, reason=str(lib_err))
    except Exception as e:
        dlg["close"]()
        logger.exception(f"TAPNext++ load failed: {e}")
        return offer_pose_fallback(app, reason=f"Load failed: {e}")


def get_yolo_pose_detector(app):
    cfg = app.pose_config or {}
    weights = cfg.get("yolo_pose_ckpt", "").strip()
    device = cfg.get("device", "cuda")
    if not weights:
        messagebox.showwarning(
            "Pose Config",
            "YOLO pose weights path is not set. Open pose settings (\u2699) first.",
            parent=app.root
        )
        return None
    if app.yolo_pose_detector is not None and getattr(app.yolo_pose_detector, 'weights', None) == weights:
        return app.yolo_pose_detector
    try:
        from .pose_tracker import YOLOPoseDetector
        app.yolo_pose_detector = YOLOPoseDetector(weights=weights, device=device)
        return app.yolo_pose_detector
    except Exception as e:
        logger.exception(f"YOLO pose load failed: {e}")
        messagebox.showerror("Pose Detector", f"Failed to load YOLO pose:\n{e}", parent=app.root)
        return None


# ---- Execution -------------------------------------------------------------

def run_yolo_pose_detect(app):
    """Run YOLO pose detection on the current frame and populate pose_points."""
    if app.current_cv_frame is None:
        messagebox.showwarning("Info", "No current frame.", parent=app.root)
        return
    det = get_yolo_pose_detector(app)
    if det is None:
        return
    try:
        detections = det.detect_pose(app.current_cv_frame)
    except Exception as e:
        logger.exception(f"YOLO pose detect failed: {e}")
        messagebox.showerror("Pose", f"Detection failed:\n{e}", parent=app.root)
        return
    if not detections:
        messagebox.showinfo("Pose", "No pose detected.", parent=app.root)
        return

    default_class = default_pose_class_name(app)
    schema = (app.pose_config or {}).get("schema", {}).get(default_class, {})
    default_edges = [list(e) for e in schema.get("skeleton", [])]

    added = 0
    for d in detections:
        new_oid = app.next_obj_id_to_propose
        app.next_obj_id_to_propose += 1
        kpts = d.get("keypoints", [])
        pose_pts = []
        for i, kp in enumerate(kpts):
            x, y = float(kp[0]), float(kp[1])
            v = int(kp[2]) if len(kp) > 2 else 1
            pose_pts.append({'x': int(x), 'y': int(y), 'visibility': 1 if v > 0 else 0, 'kpt_idx': i})
        app.tracked_objects[new_oid] = {
            'custom_label': d.get('class_name', default_class),
            'pose_points': pose_pts,
            'pose_edges': default_edges,
            'pose_class': default_class,
        }
        added += 1
    app._update_obj_id_info_label()
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
    app.update_status(f"YOLO pose: detected {added} objects.")


def run_tapnext_post_process(app):
    """Propagate pose_points across all propagated_results frames using
    TAPNext++. Uses `app._pose_query_seeds` (captured at propagation start) if
    present, else reads live pose_points from tracked_objects as query seeds."""
    if not app.propagated_results:
        messagebox.showwarning("Info", "No propagated frames to process. Run 'Start Propagate' first.", parent=app.root)
        return
    tracker = get_pose_tracker(app)
    if tracker is None:
        return

    frame_idx_sorted = sorted(app.propagated_results.keys())
    if not frame_idx_sorted:
        messagebox.showwarning("Info", "No propagated frames.", parent=app.root)
        return
    frame_to_t = {fi: ti for ti, fi in enumerate(frame_idx_sorted)}

    query_entries = []  # (oid, kpt_idx, t_rel, x, y)
    # `seen_pose_key` is per-point: a user-labelled pose at (oid, kpt_idx)
    # exists EXACTLY ONCE in the query list. We look at mid-frame prompts
    # FIRST (since they are the user's most-recent input; a scrolled-to-N
    # click at the same point wins over the original frame-0 seed). Only
    # pose keys not covered by a mid-frame prompt fall back to the seed.
    seen_pose_key = set()  # (oid, kpt_idx) that already have a query
    user_vis_map = {}
    edge_map = {}
    label_map = {}
    class_map = {}

    # 1) Mid-frame user prompts first — any (oid, kpt_idx) anchored at a
    #    non-zero frame wins over the stale seed at frame 0.
    mid_frame_count = 0
    for fi in frame_idx_sorted:
        frame_data = app.propagated_results[fi].get('masks', {})
        t_rel = frame_to_t[fi]
        for oid, obj_data in frame_data.items():
            if not isinstance(obj_data, dict):
                continue
            for p in obj_data.get('pose_points', []) or []:
                if not p.get('user_prompt'):
                    continue
                oid_i = int(oid); k = int(p.get('kpt_idx', 0))
                pose_key = (oid_i, k)
                if pose_key in seen_pose_key:
                    continue
                seen_pose_key.add(pose_key)
                query_entries.append((oid_i, k, t_rel, float(p['x']), float(p['y'])))
                user_vis_map[pose_key] = int(p.get('visibility', 2))
                if t_rel != 0:
                    mid_frame_count += 1
            if obj_data.get('pose_edges'):
                edge_map.setdefault(int(oid), [list(e) for e in obj_data['pose_edges']])
            if obj_data.get('pose_class'):
                class_map.setdefault(int(oid), obj_data['pose_class'])
            if obj_data.get('custom_label'):
                label_map.setdefault(int(oid), obj_data['custom_label'])

    # 2) Pre-propagate seeds for any pose keys not already covered. Each seed
    #    point carries its own `frame_idx` (set by add_pose_point_at from the
    #    user's current slider frame at click time); use that as the TAPNext
    #    anchor so mid-video seeds anchor correctly. Fall back to 0 if missing
    #    or out-of-range.
    T_max = max(frame_idx_sorted) if frame_idx_sorted else 0
    seeds = getattr(app, '_pose_query_seeds', None)
    if seeds:
        for oid, seed in seeds.items():
            oid_i = int(oid)
            for p in seed.get('points', []):
                k = int(p.get('kpt_idx', 0))
                pose_key = (oid_i, k)
                if pose_key in seen_pose_key:
                    continue
                raw_fi = int(p.get('frame_idx', 0))
                t_seed = frame_to_t.get(raw_fi, 0 if raw_fi <= T_max else T_max)
                seen_pose_key.add(pose_key)
                query_entries.append((oid_i, k, t_seed, float(p['x']), float(p['y'])))
                user_vis_map[pose_key] = int(p.get('visibility', 2))
            edge_map.setdefault(oid_i, [list(e) for e in seed.get('edges', [])])
            if seed.get('pose_class'):
                class_map.setdefault(oid_i, seed['pose_class'])
            if seed.get('custom_label'):
                label_map.setdefault(oid_i, seed['custom_label'])

    if mid_frame_count > 0:
        logger.info(f"TAPNext queries: {mid_frame_count} mid-frame prompt(s) take precedence over seeds.")

    if query_entries:
        logger.info(
            f"TAPNext query entries (oid, kpt, t_anchor, x, y): "
            + ", ".join(f"({q[0]},{q[1]},{q[2]},{q[3]:.0f},{q[4]:.0f})" for q in query_entries)
        )

    # 3) Fallback: live tracked_objects (legacy behaviour) anchored at the
    #    currently viewed slider frame.
    if not query_entries:
        for oid, obj in app.tracked_objects.items():
            pts = obj.get('pose_points')
            if not pts:
                continue
            oid_i = int(oid)
            cur_t = frame_to_t.get(getattr(app, 'review_current_frame', 0), 0)
            for p in pts:
                k = int(p.get('kpt_idx', 0))
                pose_key = (oid_i, k)
                if pose_key in seen_pose_key:
                    continue
                seen_pose_key.add(pose_key)
                query_entries.append((oid_i, k, cur_t, float(p['x']), float(p['y'])))
                user_vis_map[pose_key] = int(p.get('visibility', 2))
            edge_map.setdefault(oid_i, [list(e) for e in obj.get('pose_edges', [])])
            if obj.get('pose_class'):
                class_map.setdefault(oid_i, obj['pose_class'])
            if obj.get('custom_label'):
                label_map.setdefault(oid_i, obj['custom_label'])

    if not query_entries:
        messagebox.showwarning("Info", "No pose points to track.", parent=app.root)
        return

    frames_rgb = []
    for fi in frame_idx_sorted:
        fb = app.propagated_results[fi].get('frame')
        if fb is None:
            continue
        frames_rgb.append(cv2.cvtColor(fb, cv2.COLOR_BGR2RGB))
    if not frames_rgb:
        messagebox.showwarning("Info", "Propagated frames have no image data.", parent=app.root)
        return

    app.update_status(
        f"TAPNext++: tracking {len(query_entries)} points across "
        f"{len(frames_rgb)} frames (bidirectional)..."
    )
    app.root.update_idletasks()
    try:
        tracks, vis = _track_bidirectional(tracker, frames_rgb, query_entries)
    except Exception as e:
        logger.exception(f"TAPNext++ track failed: {e}")
        messagebox.showerror("TAPNext++", f"Tracking failed:\n{e}", parent=app.root)
        return

    for ti, fi in enumerate(frame_idx_sorted):
        if ti >= tracks.shape[0]:
            continue
        frame_masks = app.propagated_results[fi].setdefault('masks', {})
        for qi, (oid, kpt_idx, t_anchor, _, _) in enumerate(query_entries):
            if qi >= tracks.shape[1]:
                continue
            x, y = float(tracks[ti, qi, 0]), float(tracks[ti, qi, 1])
            model_visible = bool(vis[ti, qi]) if (vis is not None
                                                   and ti < vis.shape[0] and qi < vis.shape[1]) else True
            if ti == t_anchor:
                v_out = int(user_vis_map.get((oid, kpt_idx), 2))
            else:
                v_out = 2 if model_visible else 1
            obj_slot = frame_masks.setdefault(oid, {})
            pts_slot = obj_slot.setdefault('pose_points', [])
            while len(pts_slot) <= kpt_idx:
                pts_slot.append({'x': 0, 'y': 0, 'visibility': 0, 'kpt_idx': len(pts_slot)})
            out_pt = {'x': int(x), 'y': int(y), 'visibility': v_out, 'kpt_idx': kpt_idx}
            if ti == t_anchor:
                out_pt['user_prompt'] = True
                out_pt['frame_idx'] = int(fi)
            pts_slot[kpt_idx] = out_pt
            if 'pose_edges' not in obj_slot:
                obj_slot['pose_edges'] = [list(e) for e in edge_map.get(oid, [])]
            if 'pose_class' not in obj_slot and oid in class_map:
                obj_slot['pose_class'] = class_map[oid]
            if 'custom_label' not in obj_slot and oid in label_map:
                obj_slot['custom_label'] = label_map[oid]

    app.update_status(
        f"TAPNext++: done. {len(query_entries)} points tracked across {len(frames_rgb)} frames."
    )
    cur_slider = getattr(app, 'review_current_frame', 0)
    if cur_slider in app.propagated_results:
        for _oid, _data in app.propagated_results[cur_slider].get('masks', {}).items():
            _pts = _data.get('pose_points')
            if _pts is None:
                continue
            if _oid not in app.tracked_objects:
                app.tracked_objects[_oid] = {}
            app.tracked_objects[_oid]['pose_points'] = _pts
            if 'pose_edges' in _data:
                app.tracked_objects[_oid]['pose_edges'] = _data['pose_edges']
    if app.current_cv_frame is not None:
        app._display_cv_frame_on_view(app.current_cv_frame, app._get_current_masks_for_display())
