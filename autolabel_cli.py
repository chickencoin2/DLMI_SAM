#!/usr/bin/env python
"""Headless CLI auto-labeler for the SAM3 backends (hug / git / 3.1).

Runs open-vocabulary (text-prompted) auto-labeling on a video or image folder
WITHOUT the Tk GUI, so it can be driven programmatically (e.g. by Claude Code).

Modes
-----
  detect : independent open-vocabulary detection on every processed frame
           (works on all backends: hug / git / 3.1).
  track  : detect objects on the first frame, then propagate them across the
           video with consistent object ids (git SAM3 only; uses the official
           low-level video tracker).

Outputs (per processed frame), under --out:
  * LabelMe JSON  (<name>_NNNNNN.json) + the frame image (<name>_NNNNNN.jpg)
  * optional YOLO-seg labels (--format yolo|both) + data.yaml
  * optional visualization PNGs (--viz)
and prints per-frame counts and a final summary.

Examples
--------
  python autolabel_cli.py --input ../20251015_132820.mp4 --text "tomato,leaf" \
      --backend git --mode track --out ./out_tomato --viz
  python autolabel_cli.py --input ./frames_dir --text "person" \
      --backend hug --mode detect --out ./out --every 5
"""

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

LABELME_VERSION = "5.10.1"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_PALETTE = [(255, 64, 64), (64, 200, 64), (64, 96, 255), (240, 200, 40),
            (220, 64, 220), (64, 220, 220), (255, 140, 0), (150, 80, 255)]


# --------------------------------------------------------------------------- #
# Backend loading (headless)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Frame iteration
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Mask -> geometry / output
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
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


def run_track(backend, frames, prompts, threshold, base_name, out_dir,
              fmt, viz, class_to_id):
    """Detect objects on frame 0, then propagate with consistent ids (git only)."""
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
    next_id = 1
    inp0 = proc(images=f0_pil, device=str(app.device), return_tensors="pt")
    for prompt in prompts:
        det = backend.image_detect(f0_pil, text=prompt, threshold=threshold)
        for mask in det.masks:
            bb = mask_to_bbox(mask)
            if bb is None:
                continue
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
    out0 = model(inference_session=sess, frame=inp0.pixel_values[0])
    print(f"  seeded {len(obj_labels)} objects on frame 0: {sorted(obj_labels.items())}")

    total_objs = 0

    def emit(fi, frame_bgr, masks_t, obj_ids):
        nonlocal total_objs
        shapes, items = [], []
        for i, oid in enumerate(obj_ids):
            if i >= masks_t.shape[0]:
                break
            mask = masks_t[i].cpu().numpy() > 0.0
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


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
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
                          out_dir, args.format, args.viz, class_to_id)
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
