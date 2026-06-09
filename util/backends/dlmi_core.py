"""DLMI logit-map computation - fresh, self-contained implementation.

This is a NEW implementation written for the backend layer; it intentionally does
NOT import or call the legacy `util.dlmi_hooks` / `util.customutil.compute_dlmi_logits`.

DLMI = Direct Latent Memory Injection. A binary object mask is converted into a
float "latent logit" map that is injected into the tracker's memory encoder
(see `dlmi_inject`). The encoder then runs `sigmoid(logits)` on it, so the map
expresses a soft foreground field rather than a hard binary mask.

Modes:
  * "Fixed"    - uniform: +intensity inside the mask, -intensity outside.
  * "Gradient" - signed distance ramp: magnitude grows from ~0 at the boundary
                 to +/-intensity over `falloff` pixels (smoother, more stable
                 memory near object edges).

Output: float32 (H, W) clipped to [-intensity, +intensity].
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("DLMI_SAM_LABELER.Backends.DLMICore")

DEFAULT_INTENSITY = 10.0
DEFAULT_FALLOFF = 20
VALID_MODES = ("Fixed", "Gradient")


def _to_binary_2d(mask) -> np.ndarray:
    """Coerce an arbitrary mask (bool / uint8 / float, possibly with singleton
    dims) into a 2D uint8 {0,1} array."""
    arr = np.squeeze(np.asarray(mask))
    if arr.ndim != 2:
        raise ValueError(
            f"DLMI mask must be 2D after squeeze, got shape {np.asarray(mask).shape}")
    return (arr > 0).astype(np.uint8)


def compute_logit_map(mask, mode: str = "Fixed",
                      intensity: float = DEFAULT_INTENSITY,
                      falloff: int = DEFAULT_FALLOFF) -> np.ndarray:
    """Convert a binary mask (H, W) into a float32 logit map in
    [-intensity, +intensity]."""
    binm = _to_binary_2d(mask)
    intensity = float(intensity)

    if mode == "Fixed":
        # inside (1) -> +intensity, outside (0) -> -intensity
        return (binm.astype(np.float32) * 2.0 - 1.0) * intensity

    if mode == "Gradient":
        falloff = max(int(falloff), 1)
        inside = binm > 0
        # Signed distance to the boundary, normalised by falloff and scaled.
        dist_in = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform((1 - binm).astype(np.uint8), cv2.DIST_L2, 5)
        out = np.empty(binm.shape, dtype=np.float32)
        out[inside] = np.clip(dist_in[inside] / falloff, 0.0, 1.0) * intensity
        out[~inside] = -np.clip(dist_out[~inside] / falloff, 0.0, 1.0) * intensity
        return out

    raise ValueError(f"Unknown DLMI mode {mode!r}; expected one of {VALID_MODES}")
