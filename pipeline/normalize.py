"""Colour normalisation — white balance for cross-visit consistency.

Skin colour metrics (redness, evenness) are meaningless across visits if the
lighting / device colour cast changes. Gray-world white balance removes most
of that cast so two visits become comparable.
"""
from __future__ import annotations

import cv2
import numpy as np


def white_balance(image_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Gray-world white balance.

    Assumes the average of a scene is neutral gray; rescales each channel so
    its mean matches the global mean. Returns the corrected image and the
    per-channel gains applied (shown in the UI as the correction made).
    """
    img = image_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)  # B, G, R
    gray = float(means.mean())
    gains = gray / np.clip(means, 1e-6, None)
    out = np.clip(img * gains, 0, 255).astype(np.uint8)

    cast = _describe_cast(means)
    info = {
        "method": "gray_world",
        "gain_b": round(float(gains[0]), 3),
        "gain_g": round(float(gains[1]), 3),
        "gain_r": round(float(gains[2]), 3),
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
