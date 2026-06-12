#!/usr/bin/env python
"""Headless CLI auto-labeler for the SAM3 backends (hug / git / 3.1)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.customutil import merge_contours_into_single_polygon  # noqa: E402
from util.backends import dlmi_inject  # noqa: E402
from util.backends.dlmi_core import logit_to_confidence, confidence_to_logit  # noqa: E402

LABELME_VERSION = "5.10.1"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_PALETTE = [(255, 64, 64), (64, 200, 64), (64, 96, 255), (240, 200, 40),
            (220, 64, 220), (64, 220, 220), (255, 140, 0), (150, 80, 255)]


# Backend loading (headless)
class _HeadlessApp:
    """Minimal stand-in for the Tk app that the backends read attributes from."""
    def __init__(self, device):
        self.device = device
        self.model_dtype = torch.float32
        self.video_frames_cache = []
        # HF handles (filled for the hug backend)
        self.image_model = None
        self.image_processor = None
        self.tracker_model = None
        self.tracker_processor = None
        self.pcs_model = None
        self.pcs_processor = None
        self.inference_session = None


def load_backend(name, device):
    """Construct + load a backend headlessly. Returns (backend, app)."""
    from util.backends.git_backend import GitBackend
    from util.backends.hf_backend import HFBackend
    app = _HeadlessApp(device)
    if name in ("git", "3.1"):
        version = "sam3" if name == "git" else "sam3.1"
        backend = GitBackend(app, device, torch.float32, version=version)
        backend.key = name
        backend.load()
    elif name == "hug":
        from transformers import Sam3Model, Sam3Processor
        app.image_model = Sam3Model.from_pretrained(
            "facebook/sam3", torch_dtype=torch.float32).to(device).eval()
        app.image_processor = Sam3Processor.from_pretrained("facebook/sam3")
        backend = HFBackend(app, device, torch.float32)
        backend._loaded = True
    else:
        raise ValueError(f"unknown backend {name!r} (expected hug|git|3.1)")
    app.backend = backend
    return backend, app


# Frame iteration
def iter_frames(input_path, every=1, max_frames=None):
    """Yield (frame_index, frame_bgr) from a video file or an image directory."""
    if os.path.isdir(input_path):
        paths = []
        for p in sorted(os.listdir(input_path)):
            if p.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(input_path, p))
        count = 0
        for i, p in enumerate(paths):
            if i % every != 0:
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            yield i, img
            count += 1
            if max_frames and count >= max_frames:
                return
    else:
        cap = cv2.VideoCapture(input_path)
        idx = 0
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every == 0:
                yield idx, frame
                count += 1
                if max_frames and count >= max_frames:
                    break
            idx += 1
        cap.release()


def count_frames(input_path, every=1, max_frames=None):
    if os.path.isdir(input_path):
        n = len([p for p in os.listdir(input_path) if p.lower().endswith(IMG_EXTS)])
    else:
        cap = cv2.VideoCapture(input_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    n = (n + every - 1) // every
    return min(n, max_frames) if max_frames else n


def load_all_frames(input_path, every=1, max_frames=None):
    return [(i, f) for i, f in iter_frames(input_path, every, max_frames)]


# Mask -> geometry / output
def mask_to_polygon(mask_bool, min_area=40):
    """Largest external contour(s) -> a single LabelMe polygon (list of [x,y])."""
    m = (np.squeeze(np.asarray(mask_bool)) > 0).astype(np.uint8)
    if m.sum() < min_area:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    merged = merge_contours_into_single_polygon(contours, min_area=min_area)
    if merged is None or len(merged) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in merged]


def mask_to_bbox(mask_bool):
    m = (np.squeeze(np.asarray(mask_bool)) > 0)
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def write_labelme(out_dir, frame_name, frame_bgr, shapes):
    h, w = frame_bgr.shape[:2]
    img_path = os.path.join(out_dir, frame_name + ".jpg")
    cv2.imwrite(img_path, frame_bgr)
    data = {
        "version": LABELME_VERSION, "flags": {},
        "shapes": shapes,
        "imagePath": frame_name + ".jpg", "imageData": None,
        "imageHeight": h, "imageWidth": w,
    }
    with open(os.path.join(out_dir, frame_name + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_shape(label, polygon, group_id=None):
    return {"label": label, "points": polygon, "group_id": group_id,
            "description": "", "shape_type": "polygon", "flags": {}, "mask": None}


def write_yolo_label(labels_dir, frame_name, frame_bgr, items, class_to_id):
    """items: list of (label, mask_bool). YOLO-seg: cls x1 y1 ... normalised."""
    h, w = frame_bgr.shape[:2]
    lines = []
    for label, mask in items:
        poly = mask_to_polygon(mask)
        if poly is None:
            continue
        cid = class_to_id[label]
        coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in poly)
        lines.append(f"{cid} {coords}")
    with open(os.path.join(labels_dir, frame_name + ".txt"), "w") as f:
        f.write("\n".join(lines))


def viz_overlay(frame_bgr, items, out_path):
    ov = frame_bgr.copy()
    for i, (label, mask) in enumerate(items):
        m = (np.squeeze(np.asarray(mask)) > 0)
        if m.shape != ov.shape[:2]:
            continue
        color = _PALETTE[i % len(_PALETTE)][::-1]  # RGB->BGR
        ov[m] = (0.5 * ov[m] + 0.5 * np.array(color)).astype(np.uint8)
        bb = mask_to_bbox(m)
        if bb:
            cv2.rectangle(ov, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)
            cv2.putText(ov, label, (int(bb[0]), max(0, int(bb[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imwrite(out_path, ov)


# Modes
def run_detect(backend, frames, prompts, threshold, base_name, out_dir,
               fmt, viz, class_to_id):
    """Per-frame open-vocabulary detection."""
    viz_dir = os.path.join(out_dir, "viz")
    labels_dir = os.path.join(out_dir, "labels")
    if viz:
        os.makedirs(viz_dir, exist_ok=True)
    if fmt in ("yolo", "both"):
        os.makedirs(labels_dir, exist_ok=True)

    total_objs = 0
    for fi, frame_bgr in frames:
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        shapes, items = [], []
        for prompt in prompts:
            det = backend.image_detect(frame_pil, text=prompt, threshold=threshold)
            for mask in det.masks:
                poly = mask_to_polygon(mask)
                if poly is None:
                    continue
                shapes.append(make_shape(prompt, poly))
                items.append((prompt, mask))
        frame_name = f"{base_name}_{fi:06d}"
        write_labelme(out_dir, frame_name, frame_bgr, shapes)
        if fmt in ("yolo", "both"):
            write_yolo_label(labels_dir, frame_name, frame_bgr, items, class_to_id)
        if viz:
            viz_overlay(frame_bgr, items, os.path.join(viz_dir, frame_name + ".png"))
        total_objs += len(shapes)
        print(f"  frame {fi}: {len(shapes)} objects")
    return total_objs


def dlmi_kwargs_from_args(args):
    """argparse Namespace -> compute_logit_map keyword dict (CLI counterpart of the GUI's collect_dlmi_settings)."""
    return {
        "intensity": float(args.dlmi_alpha),
        "bg_confidence": (float(args.dlmi_bg_conf)
                          if args.dlmi_bg_conf and float(args.dlmi_bg_conf) > 0 else None),
        "boundary_soft": bool(args.dlmi_boundary_soft),
        "boundary_soft_inside": args.dlmi_boundary_sides in ("both", "inside"),
        "boundary_soft_outside": args.dlmi_boundary_sides in ("both", "outside"),
        "boundary_gradient": bool(args.dlmi_boundary_gradient),
        "boundary_width_pct": float(args.dlmi_boundary_width_pct),
        "boundary_conf_pct": float(args.dlmi_boundary_conf),
    }


def print_dlmi_confidence_summary(kw):
    """Print the DLMI settings as final confidences (alpha stays, per the paper)."""
    a = kw["intensity"]
    obj_c = logit_to_confidence(a)
    bg_c = logit_to_confidence(-a)
    line = (f"[dlmi] alpha={a:g} -> object confidence {obj_c:.3f}%, "
            f"background {bg_c:.3f}%")
    if kw["bg_confidence"]:
        line += f" (floored to {max(bg_c, kw['bg_confidence']):.3f}% by --dlmi-bg-conf)"
    print(line)
    if kw["boundary_soft"]:
        c = kw["boundary_conf_pct"]
        sides = []
        if kw["boundary_soft_inside"]:
            sides.append("inside")
        if kw["boundary_soft_outside"]:
            sides.append("outside")
        print(f"[dlmi] boundary softening: band {kw['boundary_width_pct']:g}% of width/side "
              f"({'+'.join(sides) if sides else 'no side selected'}) held at {c:g}% confidence"
              f"{', blending back with distance (gradient)' if kw['boundary_gradient'] else ''}")


def run_track(backend, frames, prompts, threshold, base_name, out_dir,
              fmt, viz, class_to_id, seed="point", dlmi_kw=None,
              mask_conf=50.0):
    """Detect objects on frame 0, then propagate with consistent ids (git only)."""
    mask_logit_thr = confidence_to_logit(min(max(float(mask_conf), 1.0), 99.0))
    if not getattr(backend, "supports_streaming", False) or backend.key not in ("git", "3.1"):
        print(f"[warn] track mode requires the 'git' or '3.1' backend; "
              f"got '{backend.key}'. Falling back to detect mode.")
        return run_detect(backend, frames, prompts, threshold, base_name, out_dir,
                          fmt, viz, class_to_id)

    viz_dir = os.path.join(out_dir, "viz")
    labels_dir = os.path.join(out_dir, "labels")
    if viz:
        os.makedirs(viz_dir, exist_ok=True)
    if fmt in ("yolo", "both"):
        os.makedirs(labels_dir, exist_ok=True)

    app = backend.app
    app.video_frames_cache = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                              for _, f in frames]
    proc, model = app.tracker_processor, app.tracker_model
    sess = proc.init_video_session(inference_device=str(app.device),
                                   processing_device="cpu",
                                   video_storage_device="cpu", dtype=torch.float32)

    # ---- seed objects from frame 0 detections ----
    f0_bgr = frames[0][1]
    f0_pil = Image.fromarray(cv2.cvtColor(f0_bgr, cv2.COLOR_BGR2RGB))
    obj_labels = {}        # obj_id -> label
    obj_masks = {}         # obj_id -> uint8 mask (dlmi seeding)
    next_id = 1
    inp0 = proc(images=f0_pil, device=str(app.device), return_tensors="pt")
    for prompt in prompts:
        det = backend.image_detect(f0_pil, text=prompt, threshold=threshold)
        for mask in det.masks:
            m = np.squeeze(np.asarray(mask)) > 0
            bb = mask_to_bbox(m)
            if bb is None:
                continue
            if seed == "dlmi":
                obj_masks[next_id] = m.astype(np.uint8)
            else:
                cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
                proc.add_inputs_to_inference_session(
                    sess, frame_idx=0, obj_ids=next_id,
                    input_points=[[[[cx, cy]]]], input_labels=[[[1]]],
                    original_size=inp0.original_sizes[0])
            obj_labels[next_id] = prompt
            next_id += 1
    if not obj_labels:
        print("[warn] no objects detected on frame 0; nothing to track.")
        return 0

    if seed == "dlmi":
        kw = dict(dlmi_kw or {})
        print_dlmi_confidence_summary(kw)
        oid_list = sorted(obj_masks.keys())
        # Register all masks in ONE call (separate calls overwrite obj_with_new_inputs), then substitute the memory-encoder input.
        proc.add_inputs_to_inference_session(
            sess, frame_idx=0, obj_ids=oid_list,
            input_masks=[obj_masks[o] for o in oid_list],
            original_size=inp0.original_sizes[0])
        queue = dlmi_inject.build_injection_queue(
            obj_ids=oid_list, masks_by_oid=obj_masks,
            device=app.device, **kw)
        state = {"idx": 0}
        original_encode = model._encode_new_memory
        model._encode_new_memory = dlmi_inject.create_injection_hook(
            queue, original_encode, log_prefix="cli", state=state)
        try:
            out0 = model(inference_session=sess, frame=inp0.pixel_values[0])
        finally:
            model._encode_new_memory = original_encode
        print(f"  seeded {len(obj_labels)} objects on frame 0 via DLMI "
              f"(injected {state['idx']} logit maps): {sorted(obj_labels.items())}")
        if state["idx"] == 0:
            print("[warn] DLMI hook did not fire; falling back to plain mask seeding result.")
    else:
        out0 = model(inference_session=sess, frame=inp0.pixel_values[0])
        print(f"  seeded {len(obj_labels)} objects on frame 0: {sorted(obj_labels.items())}")

    total_objs = 0

    def emit(fi, frame_bgr, masks_t, obj_ids):
        nonlocal total_objs
        shapes, items = [], []
        for i, oid in enumerate(obj_ids):
            if i >= masks_t.shape[0]:
                break
            mask = masks_t[i].cpu().numpy() > mask_logit_thr
            poly = mask_to_polygon(mask)
            if poly is None:
                continue
            label = obj_labels.get(oid, "object")
            shapes.append(make_shape(label, poly, group_id=int(oid)))
            items.append((f"{label}#{oid}", mask))
        frame_name = f"{base_name}_{fi:06d}"
        write_labelme(out_dir, frame_name, frame_bgr, shapes)
        if fmt in ("yolo", "both"):
            write_yolo_label(labels_dir, frame_name, frame_bgr,
                             [(obj_labels.get(o, "object"), m) for (o, (_, m)) in
                              zip(obj_ids, [(None, it[1]) for it in items])] if False else
                             [(lbl.split('#')[0], m) for lbl, m in items], class_to_id)
        if viz:
            viz_overlay(frame_bgr, items, os.path.join(viz_dir, frame_name + ".png"))
        total_objs += len(shapes)
        print(f"  frame {fi}: {len(shapes)} objects tracked")

    masks0 = proc.post_process_masks([out0.pred_masks],
                                     original_sizes=inp0.original_sizes, binarize=False)[0]
    emit(frames[0][0], f0_bgr, masks0, list(sess.obj_ids))

    for fi, frame_bgr in frames[1:]:
        fpil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inp = proc(images=fpil, device=str(app.device), return_tensors="pt")
        out = model(inference_session=sess, frame=inp.pixel_values[0])
        masks = proc.post_process_masks([out.pred_masks],
                                        original_sizes=inp.original_sizes, binarize=False)[0]
        emit(fi, frame_bgr, masks, list(sess.obj_ids))
    return total_objs


# main
def main():
    ap = argparse.ArgumentParser(
        description="Headless SAM3 open-vocabulary auto-labeler (hug/git/3.1).")
    ap.add_argument("--input", required=True, help="video file or image directory")
    ap.add_argument("--text", required=True,
                    help="comma-separated open-vocabulary prompts, e.g. 'tomato,leaf'")
    ap.add_argument("--backend", default="git", choices=["hug", "git", "3.1"])
    ap.add_argument("--mode", default="detect", choices=["detect", "track"])
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5, help="detection confidence")
    ap.add_argument("--format", default="labelme", choices=["labelme", "yolo", "both"])
    ap.add_argument("--viz", action="store_true", help="also save visualization PNGs")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # ---- track seeding / DLMI options ----
    ap.add_argument("--seed", default="point", choices=["point", "dlmi"],
                    help="track mode frame-0 seeding: bbox-centre point (legacy) or "
                         "exact-mask DLMI latent injection")
    ap.add_argument("--dlmi-alpha", type=float, default=10.0,
                    help="DLMI logit intensity (alpha; 10 = 99.995%% object confidence)")
    ap.add_argument("--dlmi-bg-conf", type=float, default=0.0,
                    help="background confidence in %% (e.g. 5 keeps 5%% foreground "
                         "belief on the background; 0 = off, must stay <50)")
    ap.add_argument("--dlmi-boundary-soft", action="store_true",
                    help="set a band around each mask boundary to --dlmi-boundary-conf")
    ap.add_argument("--dlmi-boundary-sides", default="both",
                    choices=["both", "inside", "outside"],
                    help="which side(s) of the boundary band to soften")
    ap.add_argument("--dlmi-boundary-gradient", action="store_true",
                    help="blend from boundary conf back to the region value with distance")
    ap.add_argument("--dlmi-boundary-width-pct", type=float, default=1.0,
                    help="boundary band width per side, as %% of the image WIDTH")
    ap.add_argument("--dlmi-boundary-conf", type=float, default=50.0,
                    help="FINAL confidence in %% applied at the mask boundary "
                         "(50 = fully uncertain); the logit is derived internally")
    ap.add_argument("--mask-conf", type=float, default=50.0,
                    help="tracking mask binarisation threshold as confidence %% "
                         "(a pixel counts as object above this; default 50 = legacy)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; using CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    prompts = [t.strip() for t in args.text.split(",") if t.strip()]
    class_to_id = {p: i for i, p in enumerate(prompts)}
    base_name = os.path.splitext(os.path.basename(args.input.rstrip("/")))[0]
    out_dir = args.out or (os.path.abspath(args.input.rstrip("/")) + "_autolabel")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"[autolabel] backend={args.backend} mode={args.mode} prompts={prompts}")
    print(f"[autolabel] input={args.input} out={out_dir}")
    backend, app = load_backend(args.backend, device)
    print(f"[autolabel] backend loaded in {time.time() - t0:.1f}s")

    frames = load_all_frames(args.input, args.every, args.max_frames)
    if not frames:
        print("[error] no frames found.")
        sys.exit(1)
    print(f"[autolabel] processing {len(frames)} frames "
          f"({frames[0][1].shape[1]}x{frames[0][1].shape[0]})")

    t1 = time.time()
    if args.mode == "track":
        total = run_track(backend, frames, prompts, args.threshold, base_name,
                          out_dir, args.format, args.viz, class_to_id,
                          seed=args.seed, dlmi_kw=dlmi_kwargs_from_args(args),
                          mask_conf=args.mask_conf)
    else:
        total = run_detect(backend, frames, prompts, args.threshold, base_name,
                           out_dir, args.format, args.viz, class_to_id)

    if args.format in ("yolo", "both"):
        names = "\n".join(f"  {i}: {p}" for p, i in
                          sorted(class_to_id.items(), key=lambda kv: kv[1]))
        with open(os.path.join(out_dir, "data.yaml"), "w") as f:
            f.write(f"path: {out_dir}\ntrain: images\nval: images\n"
                    f"nc: {len(class_to_id)}\nnames:\n{names}\n")

    dt = time.time() - t1
    print(f"\n[autolabel] DONE: {total} objects across {len(frames)} frames "
          f"in {dt:.1f}s ({len(frames) / max(dt, 1e-6):.2f} fps)")
    print(f"[autolabel] labels written to: {out_dir}")


if __name__ == "__main__":
    main()
