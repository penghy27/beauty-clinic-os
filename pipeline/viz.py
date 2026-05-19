"""Visual overlays for the consultation view.

OpenCV's putText cannot render Chinese, so overlays use colour-coded shapes
only; the Streamlit UI supplies the Mandarin legend.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipeline.regions import RegionPatch

# region -> BGR colour
_REGION_COLORS = {
    "forehead": (0, 200, 255),
    "left_cheek": (0, 220, 120),
    "right_cheek": (255, 180, 0),
    "nose": (200, 120, 255),
    "chin": (120, 200, 255),
    "under_eye": (255, 120, 160),
}


def draw_regions(image_bgr: np.ndarray,
                 regions: dict[str, RegionPatch]) -> np.ndarray:
    """Draw region ellipse outlines on a copy of the image."""
    canvas = image_bgr.copy()
    for name, patch in regions.items():
        color = _REGION_COLORS.get(name, (255, 255, 255))
        cv2.ellipse(
            canvas,
            (int(patch.center[0]), int(patch.center[1])),
            (int(patch.rx), int(patch.ry)),
            0, 0, 360, color, 2,
        )
    return canvas


def region_color_hex(name: str) -> str:
    """BGR colour of a region as an HTML hex string (for the legend)."""
    b, g, r = _REGION_COLORS.get(name, (255, 255, 255))
    return f"#{r:02x}{g:02x}{b:02x}"
