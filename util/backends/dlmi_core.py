"""DLMI logit maps: uniform ±alpha base from a binary mask, plus confidence-% options (background floor, boundary softening) converted to logits here."""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger("DLMI_SAM_LABELER.Backends.DLMICore")

DEFAULT_INTENSITY = 10.0
DEFAULT_BOUNDARY_CONF_PCT = 50.0

# Background floor must stay below 50% foreground, so the percentage is clamped before the logit conversion.
MAX_BG_CONFIDENCE_PCT = 49.9


def confidence_to_logit(confidence_pct: float) -> float:
    """Foreground confidence in percent (0..100) -> sigmoid logit."""
    p = min(max(float(confidence_pct) / 100.0, 1e-7), 1.0 - 1e-7)
    return math.log(p / (1.0 - p))


def logit_to_confidence(logit: float) -> float:
    """Sigmoid logit -> foreground confidence in percent (0..100)."""
    return 100.0 / (1.0 + math.exp(-float(logit)))


def _to_binary_2d(mask) -> np.ndarray:
    """Coerce an arbitrary mask (bool / uint8 / float, possibly with singleton dims) into a 2D uint8 {0,1} array."""
    arr = np.squeeze(np.asarray(mask))
    if arr.ndim != 2:
        raise ValueError(
            f"DLMI mask must be 2D after squeeze, got shape {np.asarray(mask).shape}")
    return (arr > 0).astype(np.uint8)


def compute_logit_map(mask,
                      intensity: float = DEFAULT_INTENSITY,
                      *,
                      bg_confidence: float = None,
                      boundary_soft: bool = False,
                      boundary_soft_inside: bool = True,
                      boundary_soft_outside: bool = True,
                      boundary_gradient: bool = False,
                      boundary_width_pct: float = 1.0,
                      boundary_conf_pct: float = DEFAULT_BOUNDARY_CONF_PCT) -> np.ndarray:
    """Convert a binary mask (H, W) into a float32 logit map: +intensity inside, -intensity outside (the paper's alpha)."""
    binm = _to_binary_2d(mask)
    intensity = float(intensity)
    inside = binm > 0

    # Base map: uniform +intensity inside, -intensity outside.
    out = (binm.astype(np.float32) * 2.0 - 1.0) * intensity

    # ---- background confidence floor (clamp background logits upward) ----
    if bg_confidence is not None and float(bg_confidence) > 0.0:
        bg_pct = min(float(bg_confidence), MAX_BG_CONFIDENCE_PCT)
        if bg_pct != float(bg_confidence):
            logger.warning(f"DLMI bg_confidence {bg_confidence}% clamped to "
                           f"{bg_pct}% (must stay below 50%).")
        bg_logit = np.float32(confidence_to_logit(bg_pct))
        out[~inside] = np.maximum(out[~inside], bg_logit)

    # ---- boundary softening: hold/blend the band at boundary_conf_pct ----
    if boundary_soft and (boundary_soft_inside or boundary_soft_outside):
        width = binm.shape[1]
        band_px = float(boundary_width_pct) / 100.0 * width
        if band_px >= 1.0:
            b_logit = np.float32(confidence_to_logit(boundary_conf_pct))
            dist_in = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
            dist_out = cv2.distanceTransform((1 - binm).astype(np.uint8), cv2.DIST_L2, 5)
            # distanceTransform gives >=1 for pixels adjacent to the boundary, so t=0 (pure boundary confidence) lands on the first pixel row.
            for enabled, dist, region in (
                    (boundary_soft_inside, dist_in, inside),
                    (boundary_soft_outside, dist_out, ~inside)):
                if not enabled:
                    continue
                band = region & (dist <= band_px)
                if not band.any():
                    continue
                if boundary_gradient:
                    # boundary_conf at the edge, blending back (linearly in logit space) to the region's own value at the band end.
                    t = np.clip((dist[band] - 1.0) / band_px, 0.0, 1.0).astype(np.float32)
                    out[band] = b_logit * (1.0 - t) + out[band] * t
                else:
                    out[band] = b_logit

    return out.astype(np.float32, copy=False)
