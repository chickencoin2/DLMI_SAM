"""Pose estimation UI utilities for the SAM3 autolabel app.

Provides:
  - default COCO-17 pose schema
  - pose_config.json load/save
  - open_pose_settings_dialog: Toplevel to edit models, run mode, per-class schema
  - hit_test_pose_point / draw_pose_overlay: canvas helpers consumed by
    gui_view / input_handlers when pose points are rendered or selected.

tracked_objects[obj_id] is extended with three OPTIONAL fields:
  - "pose_points": List[{"x": int, "y": int, "visibility": int, "kpt_idx": int}]
  - "pose_edges":  List[[int, int]]
  - "pose_class":  str  (key into pose_config["schema"])
All access uses .get(...) so existing objects without these fields keep working.
"""
import os
import json
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

logger = logging.getLogger("DLMI_SAM_LABELER.PoseUI")


COCO17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6), (11, 12),
]


def default_pose_schema():
    return {
        "person": {
            "keypoints": list(COCO17_KEYPOINTS),
            "skeleton": [list(e) for e in COCO17_SKELETON],
            "span_objects": [],
        }
    }


def _default_config():
    return {
        "tapnext_ckpt": "",
        "yolo_pose_ckpt": "",
        "run_mode": "post",  # "post" | "sync" | "both"
        "device": "cuda",
        "merge_save_format": "combined",  # "combined" | "separate"
        "schema": default_pose_schema(),
    }


def load_pose_config(path):
    if not os.path.exists(path):
        return _default_config()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = _default_config()
        merged.update({k: v for k, v in data.items() if v is not None})
        if "schema" not in data or not data["schema"]:
            merged["schema"] = default_pose_schema()
        return merged
    except Exception as e:
        logger.warning(f"Failed to load pose config {path}: {e}")
        return _default_config()


def save_pose_config(path, data):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Pose config saved: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save pose config {path}: {e}")
        return False


def _browse_file(var, ftypes, parent=None):
    path = filedialog.askopenfilename(title="Select file", filetypes=ftypes, parent=parent)
    if path:
        var.set(path)


def open_pose_settings_dialog(app):
    """Open modal pose settings editor. Persists to app.pose_config_path on Save."""
    win = tk.Toplevel(app.root)
    win.title("TAPNext++ Pose Settings")
    win.transient(app.root)
    try:
        win.grab_set()
    except tk.TclError:
        pass
    win.resizable(True, True)

    cfg = app.pose_config if getattr(app, 'pose_config', None) else load_pose_config(app.pose_config_path)

    mdl_frame = tk.LabelFrame(win, text="Models")
    mdl_frame.pack(fill=tk.X, padx=10, pady=5)

    tk.Label(mdl_frame, text="TAPNext++ ckpt:").grid(row=0, column=0, sticky='w', padx=5, pady=3)
    tap_var = tk.StringVar(value=cfg.get("tapnext_ckpt", ""))
    tk.Entry(mdl_frame, textvariable=tap_var, width=50).grid(row=0, column=1, padx=5, pady=3)
    tk.Button(mdl_frame, text="...", width=3,
              command=lambda: _browse_file(tap_var, [("PyTorch", "*.pt *.pth"), ("All", "*.*")], win)
              ).grid(row=0, column=2, padx=3)

    tk.Label(mdl_frame, text="YOLO pose ckpt:").grid(row=1, column=0, sticky='w', padx=5, pady=3)
    yolo_var = tk.StringVar(value=cfg.get("yolo_pose_ckpt", ""))
    tk.Entry(mdl_frame, textvariable=yolo_var, width=50).grid(row=1, column=1, padx=5, pady=3)
    tk.Button(mdl_frame, text="...", width=3,
              command=lambda: _browse_file(yolo_var, [("PyTorch", "*.pt *.pth"), ("All", "*.*")], win)
              ).grid(row=1, column=2, padx=3)

    run_frame = tk.LabelFrame(win, text="Execution")
    run_frame.pack(fill=tk.X, padx=10, pady=5)

    run_var = tk.StringVar(value="post")
    tk.Label(run_frame,
             text="TAPNext++ runs automatically AFTER SAM3 propagate when its toggle is ON\n"
                  "(TAPNext is non-causal \u2014 a 'sync-during-propagate' mode would be slow and\n"
                  "semantically equivalent, so only post-process is supported).",
             fg='gray40', font=('TkDefaultFont', 8), justify=tk.LEFT
             ).grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=3)

    device_var = tk.StringVar(value=cfg.get("device", "cuda"))
    tk.Label(run_frame, text="Device:").grid(row=1, column=0, sticky='w', padx=5, pady=3)
    tk.Radiobutton(run_frame, text="cuda", variable=device_var, value="cuda").grid(row=1, column=1, sticky='w')
    tk.Radiobutton(run_frame, text="cpu", variable=device_var, value="cpu").grid(row=1, column=2, sticky='w')

    merge_var = tk.StringVar(value=cfg.get("merge_save_format", "combined"))
    tk.Label(run_frame, text="Merge save:").grid(row=2, column=0, sticky='w', padx=5, pady=3)
    tk.Radiobutton(run_frame, text="Combined (one YOLO-pose line)",
                   variable=merge_var, value="combined").grid(row=2, column=1, columnspan=2, sticky='w')
    tk.Radiobutton(run_frame, text="Separate (seg + pose)",
                   variable=merge_var, value="separate").grid(row=2, column=3, sticky='w')

    schema_frame = tk.LabelFrame(win, text="Per-class Pose Schema")
    schema_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    schema_data = {k: {"keypoints": list(v.get("keypoints", [])),
                       "skeleton": [list(e) for e in v.get("skeleton", [])],
                       "span_objects": list(v.get("span_objects", []))}
                   for k, v in cfg.get("schema", default_pose_schema()).items()}

    class_list_frame = tk.Frame(schema_frame)
    class_list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    tk.Label(class_list_frame, text="Classes:").pack(anchor='w')
    class_listbox = tk.Listbox(class_list_frame, width=18, height=10, exportselection=False)
    class_listbox.pack()
    for cname in schema_data.keys():
        class_listbox.insert(tk.END, cname)

    class_btn_frame = tk.Frame(class_list_frame)
    class_btn_frame.pack(fill=tk.X, pady=2)

    detail_frame = tk.Frame(schema_frame)
    detail_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    tk.Label(detail_frame, text="Keypoints (one per line, ordered):").pack(anchor='w')
    kpts_text = tk.Text(detail_frame, width=34, height=12)
    kpts_text.pack(fill=tk.X)

    tk.Label(detail_frame, text="Skeleton edges (i,j per line):").pack(anchor='w')
    skel_text = tk.Text(detail_frame, width=34, height=6)
    skel_text.pack(fill=tk.X)

    tk.Label(detail_frame,
             text="Span objects (comma-separated segment label names this pose\n"
                  "crosses; whitespace around commas ignored). Example:\n"
                  "  head, torso, arms\n"
                  "Used by Auto-match: if a pose's points overlap segments with\n"
                  "ALL these labels, this class wins.",
             justify=tk.LEFT).pack(anchor='w', pady=(4, 0))
    span_text = tk.Text(detail_frame, width=34, height=3)
    span_text.pack(fill=tk.X)

    current_class = {"name": None}

    def commit_current_class():
        cname = current_class.get("name")
        if not cname:
            return
        kpts = [l.strip() for l in kpts_text.get("1.0", tk.END).splitlines() if l.strip()]
        skel = []
        for line in skel_text.get("1.0", tk.END).splitlines():
            line = line.strip().replace(";", ",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    skel.append([int(parts[0]), int(parts[1])])
                except ValueError:
                    pass
        raw = span_text.get("1.0", tk.END).strip()
        spans = []
        if raw:
            for token in raw.replace("\n", ",").replace(";", ",").split(","):
                t = token.strip()
                if t:
                    spans.append(t)
        schema_data[cname] = {"keypoints": kpts, "skeleton": skel, "span_objects": spans}

    def on_class_select(_event=None):
        commit_current_class()
        sel = class_listbox.curselection()
        if not sel:
            current_class["name"] = None
            return
        cname = class_listbox.get(sel[0])
        current_class["name"] = cname
        cdef = schema_data.get(cname, {})
        kpts = cdef.get("keypoints", [])
        skel = cdef.get("skeleton", [])
        spans = cdef.get("span_objects", [])
        kpts_text.delete("1.0", tk.END)
        kpts_text.insert(tk.END, "\n".join(kpts))
        skel_text.delete("1.0", tk.END)
        skel_text.insert(tk.END, "\n".join(f"{a},{b}" for a, b in skel))
        span_text.delete("1.0", tk.END)
        span_text.insert(tk.END, ", ".join(spans))

    class_listbox.bind("<<ListboxSelect>>", on_class_select)

    def add_class():
        commit_current_class()
        new_name = simpledialog.askstring("Add Class", "Class name:", parent=win)
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name or new_name in schema_data:
            return
        schema_data[new_name] = {"keypoints": list(COCO17_KEYPOINTS),
                                 "skeleton": [list(e) for e in COCO17_SKELETON],
                                 "span_objects": []}
        class_listbox.insert(tk.END, new_name)
        class_listbox.selection_clear(0, tk.END)
        class_listbox.select_set(tk.END)
        on_class_select()

    def del_class():
        sel = class_listbox.curselection()
        if not sel:
            return
        cname = class_listbox.get(sel[0])
        if cname in schema_data:
            del schema_data[cname]
        class_listbox.delete(sel[0])
        kpts_text.delete("1.0", tk.END)
        skel_text.delete("1.0", tk.END)
        span_text.delete("1.0", tk.END)
        current_class["name"] = None

    tk.Button(class_btn_frame, text="+ Add", command=add_class).pack(side=tk.LEFT, padx=1)
    tk.Button(class_btn_frame, text="- Del", command=del_class).pack(side=tk.LEFT, padx=1)

    if class_listbox.size() > 0:
        class_listbox.select_set(0)
        on_class_select()

    btn_bar = tk.Frame(win)
    btn_bar.pack(fill=tk.X, pady=8)

    def do_save():
        commit_current_class()
        new_cfg = {
            "tapnext_ckpt": tap_var.get().strip(),
            "yolo_pose_ckpt": yolo_var.get().strip(),
            "run_mode": run_var.get(),
            "device": device_var.get(),
            "merge_save_format": merge_var.get(),
            "schema": schema_data,
        }
        if save_pose_config(app.pose_config_path, new_cfg):
            app.pose_config = new_cfg
            messagebox.showinfo("Saved", "Pose configuration saved.", parent=win)
            win.destroy()
        else:
            messagebox.showerror("Error", "Failed to save configuration.", parent=win)

    tk.Button(btn_bar, text="Save", command=do_save, bg="#c8e6c9", width=10).pack(side=tk.RIGHT, padx=8)
    tk.Button(btn_bar, text="Cancel", command=win.destroy, width=10).pack(side=tk.RIGHT)


def _img_to_canvas(app, x, y):
    sx = getattr(app, 'scale_x', 1.0) or 1.0
    sy = getattr(app, 'scale_y', 1.0) or 1.0
    ox = getattr(app, 'offset_x', 0)
    oy = getattr(app, 'offset_y', 0)
    return x / sx + ox, y / sy + oy


def draw_pose_overlay_on_image(pil_draw, app, obj_id_color_pair_iter, selected_pose_set=None):
    """Deprecated-ish: reserved for drawing on the display PIL image directly."""
    pass


def render_pose_on_canvas(canvas, app, selected_pose_set=None):
    """Draw pose points and edges on the tkinter canvas using create_* items.
    Selected points are highlighted with a bright bold marker that stands out
    from the object color: larger radius, yellow fill, thick white outline, and
    a red crosshair, so selection is unambiguous."""
    try:
        canvas.delete("pose_overlay")
    except tk.TclError:
        return
    base_radius = 5
    sel_radius = 9
    for oid, obj in list(app.tracked_objects.items()):
        pts = obj.get("pose_points")
        if not pts:
            continue
        edges = obj.get("pose_edges", [])
        rgb = app.object_colors.get(oid, (255, 255, 0))
        color = '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        gray = '#808080'
        for (a, b) in edges:
            if a < len(pts) and b < len(pts):
                xa, ya = _img_to_canvas(app, pts[a]['x'], pts[a]['y'])
                xb, yb = _img_to_canvas(app, pts[b]['x'], pts[b]['y'])
                canvas.create_line(xa, ya, xb, yb, fill=color, width=2, tags="pose_overlay")
        for i, p in enumerate(pts):
            cx, cy = _img_to_canvas(app, p['x'], p['y'])
            is_selected = bool(selected_pose_set and (oid, i) in selected_pose_set)
            vis = p.get('visibility', 1)
            if is_selected:
                # Outer dark halo for contrast against any background
                canvas.create_oval(cx - sel_radius - 3, cy - sel_radius - 3,
                                   cx + sel_radius + 3, cy + sel_radius + 3,
                                   outline='black', width=2, tags="pose_overlay")
                # Bright yellow fill + thick white outline
                canvas.create_oval(cx - sel_radius, cy - sel_radius,
                                   cx + sel_radius, cy + sel_radius,
                                   fill='#fff200', outline='#ffffff', width=3, tags="pose_overlay")
                # Red crosshair in the middle so it's unmistakable
                canvas.create_line(cx - sel_radius - 2, cy, cx + sel_radius + 2, cy,
                                   fill='#e53935', width=2, tags="pose_overlay")
                canvas.create_line(cx, cy - sel_radius - 2, cx, cy + sel_radius + 2,
                                   fill='#e53935', width=2, tags="pose_overlay")
                # Small kpt index number label next to it
                canvas.create_text(cx + sel_radius + 6, cy - sel_radius - 6,
                                   text=str(i), fill='#e53935', anchor='nw',
                                   font=('TkDefaultFont', 9, 'bold'), tags="pose_overlay")
            else:
                if vis >= 2:
                    canvas.create_oval(cx - base_radius, cy - base_radius,
                                       cx + base_radius, cy + base_radius,
                                       fill=color, outline=color, width=1, tags="pose_overlay")
                elif vis == 1:
                    canvas.create_oval(cx - base_radius - 1, cy - base_radius - 1,
                                       cx + base_radius + 1, cy + base_radius + 1,
                                       outline=color, width=2, dash=(2, 2), fill='', tags="pose_overlay")
                    canvas.create_oval(cx - 1, cy - 1, cx + 1, cy + 1,
                                       fill=color, outline=color, tags="pose_overlay")
                else:
                    canvas.create_oval(cx - base_radius, cy - base_radius,
                                       cx + base_radius, cy + base_radius,
                                       fill=gray, outline=gray, width=1, tags="pose_overlay")


def hit_test_pose_point(app, img_x, img_y, radius=None):
    """Return (obj_id, kpt_idx) of the pose point nearest to (img_x, img_y).
    When `radius` is None, computes a zoom-aware image-space radius so the
    on-screen hit tolerance stays around ~22 canvas pixels regardless of zoom."""
    if radius is None:
        sx = getattr(app, 'scale_x', 1.0) or 1.0
        sy = getattr(app, 'scale_y', 1.0) or 1.0
        radius = max(18.0, 22.0 * max(sx, sy))
    best = None
    best_dist = radius * radius + 1
    for oid, obj in app.tracked_objects.items():
        pts = obj.get("pose_points")
        if not pts:
            continue
        for i, p in enumerate(pts):
            dx = p['x'] - img_x
            dy = p['y'] - img_y
            d2 = dx * dx + dy * dy
            if d2 <= radius * radius and d2 < best_dist:
                best = (int(oid), int(i))
                best_dist = d2
    return best
