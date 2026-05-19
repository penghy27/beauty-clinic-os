"""Facial region patches anchored to stable landmarks.

Each region is an ellipse centred on the centroid of a fixed set of
face-mesh landmarks, sized relative to the face box. Because the anchors and
the scaling are identical every visit, the same skin area is sampled each
time regardless of how large the face appears in frame — this is what makes
longitudinal comparison valid.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.landmarks import LandmarkResult

# region -> (landmark indices for centre, x-radius factor, y-radius factor)
_REGIONS = {
    "forehead":    ([151, 9, 107, 336], 0.20, 0.11),
    "left_cheek":  ([50, 205, 117],     0.11, 0.14),
    "right_cheek": ([280, 425, 346],    0.11, 0.14),
    "nose":        ([4, 5, 195],        0.08, 0.13),
    "chin":        ([152, 175, 199],    0.13, 0.09),
    "under_eye":   ([230, 450, 119, 348], 0.26, 0.05),
}

REGION_LABELS_ZH = {
    "forehead": "額頭",
    "left_cheek": "左頰",
    "right_cheek": "右頰",
    "nose": "鼻部 T 區",
    "chin": "下巴",
    "under_eye": "眼周",
}

REGION_ORDER = list(_REGIONS.keys())


@dataclass
class RegionPatch:
    name: str
    center: tuple[float, float]
    rx: float
    ry: float

    def mask(self, image_shape: tuple[int, int]) -> np.ndarray:
        """Binary mask (uint8 0/255) for this region."""
        h, w = image_shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            m,
            (int(self.center[0]), int(self.center[1])),
            (int(self.rx), int(self.ry)),
            0, 0, 360, 255, -1,
        )
        return m


def build_regions(lm: LandmarkResult) -> dict[str, RegionPatch]:
    """Build the fixed region patches for one detected face."""
    face_w = max(lm.face_box[2], 1)
    patches: dict[str, RegionPatch] = {}
    for name, (idx, fx, fy) in _REGIONS.items():
        center = lm.landmarks_px[idx].mean(axis=0)
        rx = max(fx * face_w, 5.0)
        ry = max(fy * face_w, 5.0)
        patches[name] = RegionPatch(name, (float(center[0]), float(center[1])),
                                    rx, ry)
    return patches
