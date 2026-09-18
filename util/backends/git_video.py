"""HF-API-compatible wrappers around the official sam3 low-level video tracker."""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

logger = logging.getLogger("DLMI_SAM_LABELER.Backends.GitVideo")

IMAGE_SIZE = 1008
_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _deep_flatten_points(nested):
    """app input_points = [[ [[x,y],...] ]] -> list[[x,y],...]; labels similar."""
    arr = np.asarray(nested, dtype=np.float32)
    arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else arr.reshape(-1)
    return arr


class _Inputs:
    """Mimics the HF processor BatchFeature subset the app reads."""
    def __init__(self, pixel_values, original_sizes):
        self.pixel_values = pixel_values          # [tensor [3,1008,1008]]
        self.original_sizes = original_sizes        # [(H, W)]

    def get(self, key, default=None):
        if key == "original_sizes":
            return self.original_sizes
        if key == "pixel_values":
            return self.pixel_values
        return default


class GitTrackerSession:
    """Mimics the HF inference_session object."""
    def __init__(self, tracker, device_str, total_frames):
        self._tracker = tracker
        self._device_str = device_str
        self._state = None
        self.obj_ids = []
        self.num_frames = 0                 # processed frame count (HF semantics)
        self.video_height = None
        self.video_width = None
        self._pending = []                  # queued prompts awaiting the frame tensor
        self._total_frames = int(total_frames) if total_frames else 100000
        self._preflight_done = False
        self._last_pred = None              # last video-res pred_masks [N,1,H,W]
        # Accumulated per-object prompts + cached cond-frame features, used to rebuild the multiplex (3.1) state on each prompt (its refine path is broken).
        self._obj_prompts = OrderedDict()
        self._frame0_feat = None            # (img, backbone_out) for the cond frame
        self._frame0_idx = 0
        self._fp32 = False                  # compute precision captured at creation

    def reset_inference_session(self):
        self._state = None
        self.obj_ids = []
        self.num_frames = 0
        self.video_height = None
        self.video_width = None
        self._pending = []
        self._preflight_done = False
        self._last_pred = None
        self._obj_prompts = OrderedDict()
        self._frame0_feat = None
        self._frame0_idx = 0

    @property
    def processed_frames(self):
        """HF-session compat: code reads len(session.processed_frames) as the next streaming frame index."""
        return list(range(self.num_frames))


class _ModelOutput:
    def __init__(self, pred_masks):
        self.pred_masks = pred_masks


class GitTrackerProcessor:
    def __init__(self, backend):
        self.backend = backend

    def __call__(self, images, device=None, return_tensors="pt"):
        pil = images.convert("RGB") if isinstance(images, Image.Image) \
            else Image.fromarray(np.asarray(images)).convert("RGB")
        W, H = pil.size
        t = _TRANSFORM(pil).to(self.backend._device_str)
        return _Inputs([t], [(H, W)])

    def init_video_session(self, inference_device=None, processing_device=None,
                           video_storage_device=None, dtype=None, video=None):
        self.backend._ensure_video()
        total = None
        if video is not None:
            try:
                total = len(video)
            except Exception:
                total = None
        if total is None:
            total = self.backend._num_frames_hint()
        sess = GitTrackerSession(self.backend._tracker, self.backend._device_str, total)
        # Capture the compute precision now so this whole session stays consistent even if the user toggles the fp32 option mid-track.
        sess._fp32 = self.backend._fp32_mode()
        return sess

    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids,
                                        input_points=None, input_labels=None,
                                        input_boxes=None, input_masks=None,
                                        original_size=None):
        sess = inference_session
        if original_size is not None:
            os_ = list(original_size)
            sess.video_height, sess.video_width = int(os_[0]), int(os_[1])
        # Expand a batched mask prompt into one pending entry per object so every object gets registered.
        oid_list = [int(o) for o in obj_ids] if isinstance(obj_ids, (list, tuple)) \
            else [int(obj_ids)]
        entries = []
        if input_masks is not None and len(oid_list) > 1:
            masks_seq = input_masks if isinstance(input_masks, (list, tuple)) \
                else [input_masks]
            for i, oid in enumerate(oid_list):
                entries.append({
                    "frame_idx": int(frame_idx), "obj_id": oid,
                    "points": None, "labels": None, "boxes": None,
                    "masks": masks_seq[i] if i < len(masks_seq) else masks_seq[-1]})
        else:
            entries.append({
                "frame_idx": int(frame_idx), "obj_id": oid_list[0],
                "points": input_points, "labels": input_labels,
                "boxes": input_boxes, "masks": input_masks})
        for e in entries:
            sess._pending.append(e)
            # Accumulate the latest prompt per object (insertion-ordered) so the multiplex path can rebuild the full conditioning frame.
            oid = e["obj_id"]
            acc = sess._obj_prompts.get(oid)
            if acc is None:
                acc = {"frame_idx": e["frame_idx"], "obj_id": oid,
                       "points": None, "labels": None, "boxes": None, "masks": None}
                sess._obj_prompts[oid] = acc
            if e["masks"] is not None:
                acc["masks"] = e["masks"]
                acc["points"] = acc["labels"] = acc["boxes"] = None
            if e["points"] is not None:
                acc["points"] = e["points"]
                acc["labels"] = e["labels"]
                acc["masks"] = None
            if e["boxes"] is not None:
                acc["boxes"] = e["boxes"]
                acc["masks"] = None
            acc["frame_idx"] = e["frame_idx"]

    def post_process_masks(self, pred_masks_list, original_sizes=None, binarize=False):
        pred = pred_masks_list[0]
        if pred is None:
            return [pred]
        m = pred
        if m.dim() == 3:
            m = m.unsqueeze(1)                     # [N,1,h,w]
        if original_sizes:
            H, W = int(original_sizes[0][0]), int(original_sizes[0][1])
            if (m.shape[-2], m.shape[-1]) != (H, W):
                m = torch.nn.functional.interpolate(m.float(), size=(H, W),
                                                    mode="bilinear", align_corners=False)
        m = m[:, 0]                                # [N,H,W]
        if binarize:
            m = m > 0.0
        return [m]


class GitTrackerModel:
    """Mimics Sam3TrackerVideoModel.__call__ + exposes _encode_new_memory for DLMI."""
    def __init__(self, backend):
        self.backend = backend

    # DLMI hook target proxies to the real official tracker.
    @property
    def _encode_new_memory(self):
        return self.backend._tracker._encode_new_memory

    @_encode_new_memory.setter
    def _encode_new_memory(self, fn):
        # Track whether a DLMI hook is active; bound methods can't be compared with `is`, so compare __func__ against the class method.
        tracker = self.backend._tracker
        try:
            orig_func = type(tracker)._encode_new_memory
            self.backend._dlmi_injection_active = (
                getattr(fn, "__func__", fn) is not orig_func)
        except Exception:
            self.backend._dlmi_injection_active = True
        tracker._encode_new_memory = fn

    def _ensure_state(self, sess):
        if sess._state is None:
            tracker = self.backend._tracker
            H = sess.video_height or IMAGE_SIZE
            W = sess.video_width or IMAGE_SIZE
            sess._state = tracker.init_state(
                video_height=H, video_width=W, num_frames=sess._total_frames,
                cached_features={})
        return sess._state

    def _compute_frame_features(self, frame_tensor):
        """Run the backbone on one frame -> (img, backbone_out), reusable across multiplex state rebuilds."""
        tracker = self.backend._tracker
        img = frame_tensor
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.float().to(self.backend._device_str)
        # The whole forward already runs inside the backend autocast context (see __call__), so no per-op autocast here.
        if self.backend._is_multiplex:
            from sam3.model.data_misc import NestedTensor
            # Only propagation+interactive features are needed; need_sam3_out breaks the demo's feature-clone loop.
            backbone_out = tracker.forward_image(
                NestedTensor(tensors=img, mask=None),
                need_sam3_out=False, need_interactive_out=True,
                need_propagation_out=True)
        else:
            backbone_out = tracker.forward_image(img)
        return (img, backbone_out)

    def _cache_frame(self, sess, fidx, frame_tensor):
        sess._state["cached_features"] = {fidx: self._compute_frame_features(frame_tensor)}

    @torch.inference_mode()
    def __call__(self, inference_session, frame=None, frame_idx=None):
        # Run the entire forward at ONE consistent compute dtype (bf16 default / fp32 option).
        with self.backend._autocast_ctx(getattr(inference_session, "_fp32", None)):
            return self._run(inference_session, frame, frame_idx)

    def _run(self, inference_session, frame=None, frame_idx=None):
        sess = inference_session
        tracker = self.backend._tracker
        state = self._ensure_state(sess)

        if sess._pending:
            # ---- prompt frame: apply queued prompts, return their masks ----
            if self.backend._is_multiplex:
                # SAM3.1 multiplex: rebuild the conditioning frame from accumulated prompts (refine path is broken); SAM3 stays incremental.
                return self._multiplex_prompt(sess, frame)
            fidx = sess._pending[0]["frame_idx"]
            if frame is not None:
                self._cache_frame(sess, fidx, frame)
            video_res = None
            W = sess.video_width or IMAGE_SIZE
            H = sess.video_height or IMAGE_SIZE
            mux = self.backend._is_multiplex
            for p in sess._pending:
                oid = p["obj_id"]
                if p["masks"] is not None:
                    mask = self._extract_mask(p["masks"])          # [H,W] on device
                    if mux:
                        _, _, _, video_res = tracker.add_new_masks(
                            state, fidx, [oid], mask.unsqueeze(0))  # [1,H,W]
                    else:
                        _, _, _, video_res = tracker.add_new_mask(state, fidx, oid, mask)
                else:
                    pts, lbls, box = self._extract_points_box(p, W, H)
                    if mux:
                        # multiplex demo has add_new_points (no box arg); convert a box to its centre point (the app already does this upstream).
                        if pts is None and box is not None:
                            cx = float((box[0] + box[2]) / 2.0)
                            cy = float((box[1] + box[3]) / 2.0)
                            pts = torch.tensor([[cx, cy]], dtype=torch.float32)
                            lbls = torch.tensor([1], dtype=torch.int32)
                        _, _, _, video_res = tracker.add_new_points(
                            state, fidx, oid, points=pts, labels=lbls,
                            clear_old_points=True, rel_coordinates=True)
                    else:
                        _, _, _, video_res = tracker.add_new_points_or_box(
                            state, fidx, oid, points=pts, labels=lbls, box=box,
                            clear_old_points=True, rel_coordinates=True)
            sess.obj_ids = list(state["obj_ids"])
            sess._pending = []
            sess.num_frames = max(sess.num_frames, fidx + 1)
            sess._preflight_done = False
            hooked = bool(getattr(self.backend, "_dlmi_injection_active", False))
            # Preflight when a DLMI hook is installed (fires with the injected logits) or on multiplex (consolidates all objects).
            if mux or hooked:
                tracker.propagate_in_video_preflight(state)
                sess._preflight_done = True
                cond = state["output_dict"]["cond_frame_outputs"].get(fidx)
                if cond is not None and cond.get("pred_masks") is not None:
                    _, video_res = tracker._get_orig_video_res_output(state, cond["pred_masks"])
            sess._last_pred = video_res
            return _ModelOutput(video_res)

        # Tracking frame: preflight BEFORE caching the new frame (the LRU feature cache holds only one frame).
        if not sess._preflight_done:
            tracker.propagate_in_video_preflight(state)
            sess._preflight_done = True

        if frame is not None:
            fidx = sess.num_frames
            self._cache_frame(sess, fidx, frame)
        else:
            fidx = frame_idx if frame_idx is not None else max(sess.num_frames - 1, 0)

        batch_size = tracker._get_obj_num(state)
        current_out, pred_masks = tracker._run_single_frame_inference(
            inference_state=state, output_dict=state["output_dict"], frame_idx=fidx,
            batch_size=batch_size, is_init_cond_frame=False, point_inputs=None,
            mask_inputs=None, reverse=False, run_mem_encoder=True)
        if self.backend._is_multiplex:
            import copy
            current_out["local_obj_id_to_idx"] = copy.deepcopy(state["obj_id_to_idx"])
        state["output_dict"]["non_cond_frame_outputs"][fidx] = current_out
        tracker._add_output_per_object(state, fidx, current_out, "non_cond_frame_outputs")
        state["frames_already_tracked"][fidx] = {"reverse": False}
        _, video_res = tracker._get_orig_video_res_output(state, pred_masks)
        sess.num_frames = max(sess.num_frames, fidx + 1)
        sess._last_pred = video_res
        return _ModelOutput(video_res)

    def _multiplex_prompt(self, sess, frame):
        """SAM3.1 multiplex prompt-frame handler."""
        tracker = self.backend._tracker
        fidx = sess._pending[0]["frame_idx"]
        sess._frame0_idx = fidx
        if frame is not None:
            sess._frame0_feat = self._compute_frame_features(frame)
        if sess._frame0_feat is None:
            raise RuntimeError("multiplex prompt: conditioning frame features missing")
        sess._pending = []

        H = sess.video_height or IMAGE_SIZE
        W = sess.video_width or IMAGE_SIZE
        state = tracker.init_state(
            video_height=H, video_width=W, num_frames=sess._total_frames,
            cached_features={fidx: sess._frame0_feat})
        sess._state = state

        # mask prompts -> one batched add_new_masks; point/box prompts -> per object
        mask_oids, mask_tensors, point_items = [], [], []
        for oid, pr in sess._obj_prompts.items():
            if pr.get("masks") is not None:
                mask_oids.append(oid)
                mask_tensors.append(self._extract_mask(pr["masks"]))
            elif pr.get("points") is not None or pr.get("boxes") is not None:
                point_items.append((oid, pr))

        video_res = None
        if mask_oids:
            stacked = torch.stack(mask_tensors, dim=0)             # [N,H,W]
            _, _, _, video_res = tracker.add_new_masks(state, fidx, mask_oids, stacked)
        for oid, pr in point_items:
            pts, lbls, box = self._extract_points_box(pr, W, H)
            if pts is None and box is not None:
                cx = float((box[0] + box[2]) / 2.0)
                cy = float((box[1] + box[3]) / 2.0)
                pts = torch.tensor([[cx, cy]], dtype=torch.float32)
                lbls = torch.tensor([1], dtype=torch.int32)
            _, _, _, video_res = tracker.add_new_points(
                state, fidx, oid, points=pts, labels=lbls,
                clear_old_points=True, rel_coordinates=True)

        sess.obj_ids = list(state["obj_ids"])
        sess.num_frames = max(sess.num_frames, fidx + 1)
        tracker.propagate_in_video_preflight(state)
        sess._preflight_done = True
        cond = state["output_dict"]["cond_frame_outputs"].get(fidx)
        if cond is not None and cond.get("pred_masks") is not None:
            _, video_res = tracker._get_orig_video_res_output(state, cond["pred_masks"])
        sess._last_pred = video_res
        return _ModelOutput(video_res)

    # ----- prompt extraction (pixel -> normalised) ----- #
    def _extract_points_box(self, p, W, H):
        pts = lbls = box = None
        if p["boxes"] is not None:
            b = np.asarray(p["boxes"], dtype=np.float32).reshape(-1)[:4]
            box = torch.tensor([b[0] / W, b[1] / H, b[2] / W, b[3] / H],
                               dtype=torch.float32)
        if p["points"] is not None:
            arr = _deep_flatten_points(p["points"])      # [K,2] pixel
            if arr.size:
                arr = arr.reshape(-1, 2).astype(np.float32)
                norm = np.stack([arr[:, 0] / W, arr[:, 1] / H], axis=1)
                pts = torch.tensor(norm, dtype=torch.float32)
                if p["labels"] is not None:
                    lbls = torch.tensor(
                        np.asarray(p["labels"]).reshape(-1).astype(np.int32),
                        dtype=torch.int32)
        return pts, lbls, box

    def _extract_mask(self, masks):
        arr = masks
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        arr = np.squeeze(arr)
        while arr.ndim > 2:
            arr = arr[0]
        return torch.from_numpy((arr > 0).astype(np.float32)).to(self.backend._device_str)
