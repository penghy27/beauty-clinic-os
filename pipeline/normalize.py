"""Colour and exposure normalisation for cross-visit consistency.

Skin colour metrics (redness, evenness) are meaningless across visits if the
lighting or device colour cast changes. Two anchors, both classical:

* **Chromatic gains from the background** (pixels outside the face box).
  The background carries the ambient light cast and — unlike the face —
  is not the signal being measured: gray-world computed on face pixels
  would force the average skin tone to neutral and erase redness itself.
  A clinic room is also constant between visits, so the same reference
  yields the same correction each time.
* **Luminance anchored on the face.** After the cast is removed, all
  channels are scaled so the mean face brightness lands on a fixed target,
  making L*-based metrics and blob thresholds exposure-invariant.
"""
from __future__ import annotations

import numpy as np

FACE_LUMINANCE_TARGET = 128.0

# Below this fraction of background pixels the gray-world estimate is too
# thin to trust and the whole frame is used instead.
_MIN_BG_FRACTION = 0.05


def white_balance(
    image_bgr: np.ndarray,
    face_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    """Background-anchored gray-world + face luminance anchor.

    Without a face box this degrades to plain full-frame gray-world with no
    luminance anchoring (the v1 behaviour). Returns the corrected image and
    the gains applied (shown in the UI as the correction made).
    """
    img = image_bgr.astype(np.float32)
    h, w = img.shape[:2]

    reference = img.reshape(-1, 3)
    method = "gray_world"
    if face_box is not None:
        x, y, bw, bh = face_box
        mask = np.ones((h, w), dtype=bool)
        mask[max(y, 0):min(y + bh, h), max(x, 0):min(x + bw, w)] = False
        background = img[mask]
        if background.shape[0] >= _MIN_BG_FRACTION * h * w:
            reference = background
            method = "gray_world_bg"

    means = reference.mean(axis=0)  # B, G, R
    gray = float(means.mean())
    gains = gray / np.clip(means, 1e-6, None)

    lum_gain = 1.0
    if face_box is not None:
        x, y, bw, bh = face_box
        face = img[max(y, 0):min(y + bh, h), max(x, 0):min(x + bw, w)]
        if face.size:
            face_mean = float((face * gains).mean())
            lum_gain = FACE_LUMINANCE_TARGET / max(face_mean, 1e-6)
            method += "+face_lum"

    out = np.clip(img * gains * lum_gain, 0, 255).astype(np.uint8)

    cast = _describe_cast(means)
    info = {
        "method": method,
        "gain_b": round(float(gains[0]), 3),
        "gain_g": round(float(gains[1]), 3),
        "gain_r": round(float(gains[2]), 3),
        "gain_luminance": round(lum_gain, 3),
        "original_cast": cast,
    }
    return out, info


def _describe_cast(means_bgr: np.ndarray) -> str:
    """Plain-language description of the original colour cast."""
    b, g, r = means_bgr
    if r > g and r > b:
        return "偏暖（紅）"
    if b > g and b > r:
        return "偏冷（藍）"
    return "中性"
