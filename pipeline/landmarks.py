"""Face detection, landmarks and head-pose estimation (MediaPipe Face Mesh).

This is the entry point of the imaging pipeline: every photo must yield a
single detected face before any skin metric is computed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

_FACE_MESH = None


def _face_mesh():
    """Lazily build a reusable static-image Face Mesh."""
    global _FACE_MESH
    if _FACE_MESH is None:
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _FACE_MESH


# Face-mesh landmark indices used for head-pose PnP.
_POSE_IDX = [1, 152, 33, 263, 61, 291]
# Matching canonical 3D face model (generic head, arbitrary mm units).
_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),          # 1   nose tip
        (0.0, -330.0, -65.0),     # 152 chin
        (-225.0, 170.0, -135.0),  # 33  eye outer corner
        (225.0, 170.0, -135.0),   # 263 eye outer corner
        (-150.0, -150.0, -125.0), # 61  mouth corner
        (150.0, -150.0, -125.0),  # 291 mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class LandmarkResult:
    landmarks_px: np.ndarray          # (N, 2) pixel coordinates
    image_shape: tuple[int, int]      # (height, width)
    face_box: tuple[int, int, int, int]  # x, y, w, h
    face_ratio: float                 # face-box height / image height
    yaw: float                        # degrees, 0 = frontal
    pitch: float                      # degrees, 0 = frontal
    roll: float                       # degrees, 0 = level


def detect(image_bgr: np.ndarray) -> LandmarkResult | None:
    """Detect one face. Returns None when no face is found."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _face_mesh().process(rgb)
    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

    xs, ys = pts[:, 0], pts[:, 1]
    x0, y0 = float(xs.min()), float(ys.min())
    bw, bh = float(xs.max() - x0), float(ys.max() - y0)
    face_box = (int(x0), int(y0), int(round(bw)), int(round(bh)))
    face_ratio = bh / h if h else 0.0

    yaw, pitch, roll = _estimate_pose(pts, (h, w))
    return LandmarkResult(pts, (h, w), face_box, face_ratio, yaw, pitch, roll)


def crop_face(image_bgr: np.ndarray, result: LandmarkResult,
              margin: float = 0.25) -> np.ndarray:
    """Crop a padded square-ish region around the detected face box."""
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = result.face_box
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + bw + mx, w), min(y + bh + my, h)
    return image_bgr[y0:y1, x0:x1].copy()


def _estimate_pose(pts: np.ndarray,
                   image_shape: tuple[int, int]) -> tuple[float, float, float]:
    h, w = image_shape
    image_pts = np.array([pts[i] for i in _POSE_IDX], dtype=np.float64)
    focal = float(w)
    cam = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_3D, image_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
        roll = 0.0
    return _fold(yaw), _fold(pitch), _fold(roll)


def _fold(angle: float) -> float:
    """Fold an angle into [-90, 90] so a frontal face reads near 0."""
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle = 180.0 - angle
    elif angle < -90.0:
        angle = -180.0 - angle
    return round(angle, 1)
