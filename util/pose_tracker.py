"""TAPNext++ and YOLO-pose wrappers for the SAM3 autolabel app.

Design:
- `TAPNextPPTracker` uses the PyTorch port shipped in
  `google-deepmind/tapnet`. If the tapnet package isn't installed, this wrapper
  attempts a one-shot `pip install git+https://...` before retrying the import.
- If tapnet still can't be loaded (no network, build failure, etc.), callers
  can use `OpticalFlowPoseTracker` as a graceful fallback using OpenCV's
  Lucas-Kanade sparse optical flow. It's lower-quality but dependency-free.
- `YOLOPoseDetector` wraps ultralytics YOLO-pose for initial keypoint
  detection on a single frame.

The tracker `track(frames_rgb, query_points_xy_t0)` contract is shared:
  - frames_rgb: list of np.uint8 [H, W, 3] RGB frames (len >= 1)
  - query_points_xy_t0: np.float32 [N, 2] pixel coords at t=0
Returns (tracks[T, N, 2] pixel xy, visibilities[T, N] bool).
"""
import os
import sys
import logging
import subprocess
import numpy as np
import cv2

logger = logging.getLogger("DLMI_SAM_LABELER.PoseTracker")


class LibraryMissingError(RuntimeError):
    """Raised when tapnet package is not available and cannot be installed."""
    pass


class TAPNextPPTracker:
    """PyTorch wrapper for TAPNext++ following the official colab demo pattern:
        from tapnet.tapnext.tapnext_torch import TAPNext
        model = TAPNext(image_size=(256, 256))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict({k.replace('tapnext.', ''): v for k, v in ckpt['state_dict'].items()})
        pred_tracks, track_logits, visible_logits, tracking_state = model(
            video=video, query_points=query_points
        )
    Video frames are resized to 256x256 internally; track outputs are rescaled
    back to original pixel coordinates.
    """
    MODEL_IMAGE_SIZE = (256, 256)
    TAPNET_GIT_URL = "git+https://github.com/google-deepmind/tapnet.git"

    def __init__(self, ckpt_path: str, device: str = "cuda", auto_install_cb=None):
        """
        ckpt_path: path to tapnextpp_ckpt.pt
        device: 'cuda' | 'cpu'
        auto_install_cb: optional callable(stage_str) invoked with progress
                         messages during auto-install ("installing...", etc.)
        """
        self.ckpt_path = ckpt_path
        self.device_str = device
        self.auto_install_cb = auto_install_cb
        self.model = None
        self._torch = None
        self.device = None
        self._load()

    def _log(self, msg):
        logger.info(msg)
        if self.auto_install_cb:
            try:
                self.auto_install_cb(msg)
            except Exception:
                pass

    def _try_import_tapnext(self):
        from tapnet.tapnext.tapnext_torch import TAPNext  # noqa: F401
        return True

    EXTRA_DEPS = ["einops", "mediapy", "chex", "flax", "optax"]

    def _pip_install(self, spec_list, quiet=True):
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.append("--quiet")
        cmd.extend(spec_list)
        logger.info(f"Running pip: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=900
        )
        if proc.returncode != 0:
            logger.error(f"pip install failed rc={proc.returncode} stderr={proc.stderr[:1500]}")
            return False, proc.stderr
        return True, proc.stdout

    def _attempt_pip_install(self):
        self._log("Installing tapnet + dependencies (this may take a minute)...")
        ok, err = self._pip_install([self.TAPNET_GIT_URL])
        if not ok:
            self._log(f"tapnet pip install failed: {err[:200]}")
            return False
        self._log("tapnet installed. Ensuring runtime dependencies (einops, etc.)...")
        ok2, err2 = self._pip_install(self.EXTRA_DEPS)
        if not ok2:
            self._log(f"Runtime deps install warning: {err2[:200]}")
        return True

    def _install_missing_module(self, mod_name):
        self._log(f"Missing runtime module '{mod_name}' \u2014 installing...")
        ok, err = self._pip_install([mod_name])
        if not ok:
            logger.error(f"pip install {mod_name} failed: {err[:500]}")
        return ok

    def _load(self):
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            raise RuntimeError(f"TAPNext++ checkpoint not found: {self.ckpt_path}")
        try:
            import torch as _torch
        except ImportError as e:
            raise RuntimeError(f"PyTorch not available: {e}")
        self._torch = _torch

        # Import attempt with up to 3 passes: first import, after pip install
        # tapnet, and a third pass that picks off any missing-module errors one
        # by one (einops, mediapy, chex, etc.) by pip-installing them.
        last_err = None
        for attempt in range(4):
            try:
                self._try_import_tapnext()
                last_err = None
                break
            except ImportError as e:
                last_err = e
                msg = str(e)
                if attempt == 0 and "tapnet" in msg:
                    if not self._attempt_pip_install():
                        raise LibraryMissingError(
                            "tapnet auto-install failed. Install manually:\n"
                            "  pip install git+https://github.com/google-deepmind/tapnet.git einops mediapy chex flax optax"
                        )
                    continue
                # missing-module style: "No module named 'X'"
                import re
                m = re.search(r"No module named ['\"]([\w\.]+)['\"]", msg)
                if m:
                    mod = m.group(1).split(".")[0]
                    if self._install_missing_module(mod):
                        continue
                raise LibraryMissingError(f"tapnet import error after retries: {e}")
        if last_err is not None:
            raise LibraryMissingError(f"tapnet still not importable: {last_err}")

        from tapnet.tapnext.tapnext_torch import TAPNext

        device = _torch.device(self.device_str if (_torch.cuda.is_available() or self.device_str == "cpu") else "cpu")
        self.device = device

        self._log(f"Loading TAPNext++ weights on {device}...")
        model = TAPNext(image_size=self.MODEL_IMAGE_SIZE)
        ckpt = _torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            remapped = {}
            for k, v in sd.items():
                nk = k.replace("tapnext.", "") if k.startswith("tapnext.") else k
                remapped[nk] = v
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            if missing:
                logger.warning(f"TAPNext++ load: {len(missing)} missing keys (first: {list(missing)[:3]})")
            if unexpected:
                logger.warning(f"TAPNext++ load: {len(unexpected)} unexpected keys (first: {list(unexpected)[:3]})")
        else:
            try:
                model.load_state_dict(ckpt, strict=False)
            except Exception:
                raise RuntimeError("Checkpoint format not recognized (no 'state_dict' key).")
        model.to(device).eval()
        self.model = model
        self._log("TAPNext++ ready.")

    def track(self, frames_rgb, query_points):
        """Track query points through a video.

        Args:
            frames_rgb: list of np.uint8 [H, W, 3] RGB frames.
            query_points: np.float32 array of shape either
                - [N, 2] with (x, y) in pixel coords; all queries anchored at t=0, OR
                - [N, 3] with (t_frame_index, x, y); each query anchored at its
                  own frame. t_frame_index is the index into frames_rgb.
        Returns:
            (tracks [T, N, 2] pixel xy, visibilities [T, N] bool)
        """
        if self.model is None:
            raise RuntimeError("TAPNext++ not initialized.")
        if not frames_rgb:
            raise ValueError("frames_rgb is empty.")
        _torch = self._torch
        device = self.device
        T = len(frames_rgb)
        H, W = frames_rgb[0].shape[:2]

        q_arr = np.asarray(query_points, dtype=np.float32)
        if q_arr.ndim != 2 or q_arr.shape[1] not in (2, 3):
            raise ValueError(f"query_points must be [N,2] or [N,3], got {q_arr.shape}")
        N = q_arr.shape[0]
        if q_arr.shape[1] == 2:
            t_vals = np.zeros((N,), dtype=np.float32)
            xy_pixel = q_arr
        else:
            t_vals = np.clip(q_arr[:, 0], 0, T - 1).astype(np.float32)
            xy_pixel = q_arr[:, 1:3]

        rH, rW = self.MODEL_IMAGE_SIZE
        resized = np.stack([cv2.resize(f, (rW, rH), interpolation=cv2.INTER_LINEAR) for f in frames_rgb], axis=0)
        video_np = resized.astype(np.float32) / 255.0  # [T, H, W, 3] channels-last
        # TAPNext++ expects [B, T, H, W, C] with C-last
        video = _torch.from_numpy(video_np).unsqueeze(0).to(device)  # [1, T, H, W, 3]

        # Convert pixel (x, y) to resized-frame pixel space
        xy_resized = xy_pixel.copy()
        xy_resized[:, 0] *= (rW / max(W, 1))
        xy_resized[:, 1] *= (rH / max(H, 1))
        # Demo uses (t, y, x) pixel coordinates in the resized frame space
        q = np.zeros((N, 3), dtype=np.float32)
        q[:, 0] = t_vals
        q[:, 1] = xy_resized[:, 1]
        q[:, 2] = xy_resized[:, 0]
        query_points = _torch.from_numpy(q).unsqueeze(0).to(device)  # [1, N, 3]

        try:
            with _torch.inference_mode():
                with _torch.amp.autocast(device_type="cuda", dtype=_torch.float16,
                                          enabled=(device.type == "cuda")):
                    out = self.model(video=video, query_points=query_points)
        except Exception as e:
            logger.exception(f"TAPNext++ forward failed on [B,T,H,W,C] layout: {e}")
            # Fallback: try [B, T, C, H, W] layout
            try:
                video2 = _torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)
                with _torch.inference_mode():
                    with _torch.amp.autocast(device_type="cuda", dtype=_torch.float16,
                                              enabled=(device.type == "cuda")):
                        out = self.model(video=video2, query_points=query_points)
            except Exception as e2:
                logger.exception(f"TAPNext++ forward also failed on [B,T,C,H,W] layout: {e2}")
                raise e

        pred_tracks = None
        visible_logits = None
        if isinstance(out, (tuple, list)):
            if len(out) >= 3:
                pred_tracks = out[0]
                visible_logits = out[2]
            elif len(out) >= 1:
                pred_tracks = out[0]
        elif isinstance(out, dict):
            pred_tracks = out.get("tracks", out.get("pred_tracks"))
            visible_logits = out.get("visible_logits", out.get("visibles"))
        if pred_tracks is None:
            raise RuntimeError("TAPNext++ output did not contain tracks.")

        tracks_np = pred_tracks.detach().to(_torch.float32).cpu().numpy()
        # Squeeze batch dim if present
        if tracks_np.ndim == 4 and tracks_np.shape[0] == 1:
            tracks_np = tracks_np[0]  # [T, N, 2]
        elif tracks_np.ndim == 4 and tracks_np.shape[1] == 1:
            tracks_np = tracks_np[:, 0]
        if tracks_np.shape[-1] != 2:
            raise RuntimeError(f"Unexpected tracks shape: {tracks_np.shape}")
        # TAPNext tracks come out as (y, x) in the resized-frame pixel space,
        # matching the (t, y, x) order of the query points.
        tracks_xy_resized = tracks_np[..., ::-1].copy()  # -> (x, y)
        tracks_xy = tracks_xy_resized.copy()
        tracks_xy[..., 0] *= (W / max(rW, 1))
        tracks_xy[..., 1] *= (H / max(rH, 1))

        if visible_logits is not None:
            vis_np = visible_logits.detach().to(_torch.float32).cpu().numpy()
            # Common TAPNext output: [B, T, N, 1] -> squeeze batch and last dim
            while vis_np.ndim > 2 and vis_np.shape[0] == 1:
                vis_np = vis_np[0]
            if vis_np.ndim == 3 and vis_np.shape[-1] == 1:
                vis_np = vis_np[..., 0]
            if vis_np.ndim == 2 and vis_np.shape != tracks_xy.shape[:2]:
                if vis_np.shape == (tracks_xy.shape[1], tracks_xy.shape[0]):
                    vis_np = vis_np.T
            vis_bool = vis_np > 0
        else:
            vis_bool = np.ones(tracks_xy.shape[:2], dtype=np.bool_)

        if tracks_xy.shape[0] != T:
            if tracks_xy.shape[0] == 1:
                tracks_xy = np.repeat(tracks_xy, T, axis=0)
                if vis_bool.shape[0] == 1:
                    vis_bool = np.repeat(vis_bool, T, axis=0)
        return tracks_xy.astype(np.float32), vis_bool.astype(np.bool_)


class OpticalFlowPoseTracker:
    """OpenCV Lucas-Kanade sparse optical flow fallback. Supports per-query t
    via bidirectional tracking from each anchor frame (forward + backward)."""
    def __init__(self, ckpt_path=None, device=None, auto_install_cb=None):
        self.ckpt_path = ckpt_path
        self.auto_install_cb = auto_install_cb

    def track(self, frames_rgb, query_points):
        if not frames_rgb:
            raise ValueError("frames_rgb is empty.")
        T = len(frames_rgb)
        q_arr = np.asarray(query_points, dtype=np.float32)
        if q_arr.ndim != 2 or q_arr.shape[1] not in (2, 3):
            raise ValueError(f"query_points must be [N,2] or [N,3]")
        N = q_arr.shape[0]
        if q_arr.shape[1] == 2:
            t_anchors = np.zeros((N,), dtype=np.int32)
            xy = q_arr
        else:
            t_anchors = np.clip(q_arr[:, 0], 0, T - 1).astype(np.int32)
            xy = q_arr[:, 1:3]

        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
        tracks = np.zeros((T, N, 2), dtype=np.float32)
        vis = np.zeros((T, N), dtype=np.bool_)
        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        for qi in range(N):
            ti_anchor = int(t_anchors[qi])
            tracks[ti_anchor, qi] = xy[qi]
            vis[ti_anchor, qi] = True
            # Forward
            cur = xy[qi].reshape(1, 1, 2).astype(np.float32)
            for ti in range(ti_anchor + 1, T):
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(grays[ti - 1], grays[ti], cur, None, **lk_params)
                if nxt is None:
                    tracks[ti, qi] = cur.reshape(-1, 2)[0]
                    vis[ti, qi] = False
                    continue
                tracks[ti, qi] = nxt.reshape(-1, 2)[0]
                vis[ti, qi] = bool(status.reshape(-1)[0]) if status is not None else True
                cur = nxt
            # Backward
            cur = xy[qi].reshape(1, 1, 2).astype(np.float32)
            for ti in range(ti_anchor - 1, -1, -1):
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(grays[ti + 1], grays[ti], cur, None, **lk_params)
                if nxt is None:
                    tracks[ti, qi] = cur.reshape(-1, 2)[0]
                    vis[ti, qi] = False
                    continue
                tracks[ti, qi] = nxt.reshape(-1, 2)[0]
                vis[ti, qi] = bool(status.reshape(-1)[0]) if status is not None else True
                cur = nxt
        return tracks, vis


class YOLOPoseDetector:
    """Ultralytics YOLO-pose wrapper. Returns list of dicts per detection:
        {'class_id', 'class_name', 'confidence', 'bbox':[x1,y1,x2,y2],
         'keypoints':[[x,y,v], ...]}"""
    def __init__(self, weights: str, device: str = "cuda"):
        self.weights = weights
        self.device = device
        self.model = None
        self._load()

    def _load(self):
        if not self.weights or not os.path.exists(self.weights):
            raise RuntimeError(f"YOLO-pose weights not found: {self.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(f"ultralytics not installed: {e}")
        self.model = YOLO(self.weights)

    def detect_pose(self, frame_bgr):
        if self.model is None:
            raise RuntimeError("YOLOPoseDetector is not initialized.")
        try:
            results = self.model.predict(frame_bgr, verbose=False, device=self.device)
        except Exception as e:
            logger.warning(f"YOLO-pose predict failed on given device; retrying on cpu: {e}")
            results = self.model.predict(frame_bgr, verbose=False, device="cpu")
        out = []
        if not results:
            return out
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        kpts = getattr(r0, "keypoints", None)
        names = getattr(r0, "names", {}) or getattr(self.model, "names", {}) or {}
        if boxes is None or kpts is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls, dtype=int)
        kpts_data = None
        if hasattr(kpts, "data"):
            kpts_data = kpts.data.cpu().numpy() if hasattr(kpts.data, "cpu") else np.asarray(kpts.data)
        for i in range(len(xyxy)):
            entry = {
                "class_id": int(cls[i]) if i < len(cls) else 0,
                "class_name": names.get(int(cls[i]), str(int(cls[i]))) if names else str(int(cls[i])),
                "confidence": float(confs[i]) if i < len(confs) else 0.0,
                "bbox": [float(v) for v in xyxy[i].tolist()],
                "keypoints": [],
            }
            if kpts_data is not None and i < len(kpts_data):
                kps = kpts_data[i]
                for kp in kps:
                    if len(kp) >= 3:
                        entry["keypoints"].append([float(kp[0]), float(kp[1]), float(kp[2])])
                    elif len(kp) == 2:
                        entry["keypoints"].append([float(kp[0]), float(kp[1]), 1.0])
            out.append(entry)
        return out
