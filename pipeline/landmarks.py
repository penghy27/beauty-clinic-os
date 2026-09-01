"""Face detection, landmarks and head-pose estimation.

Uses the MediaPipe Face Landmarker (Tasks API): 478 landmarks plus a facial
transformation matrix from which head pose is read directly. This is the
entry point of the imaging pipeline — every photo must yield a single
detected face before any skin metric is computed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

_LANDMARKER = None


def _landmarker():
    """Lazily build a reusable single-image Face Landmarker."""
    global _LANDMARKER
    if _LANDMARKER is None:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER


@dataclass
class LandmarkResult:
    landmarks_px: np.ndarray          # (N, 2) pixel coordinates
    image_shape: tuple[int, int]      # (height, width)
    face_box: tuple[int, int, int, int]  # x, y, w, h
    face_ratio: float                 # face-box height / image height
    yaw: float                        # degrees, 0 = frontal
    pitch: float                      # degrees, 0 = frontal
    roll: float                       # degrees, 0 = level
    native_face_width: float = 0.0    # face-box width at capture resolution

    def scaled(self, factor: float,
               image_shape: tuple[int, int]) -> LandmarkResult:
        """This result mapped onto a resized copy of the same image.

        Ratios and angles are scale-invariant; native_face_width keeps the
        capture resolution so the quality gate can still see it.
        """
        x, y, w, h = self.face_box
        return LandmarkResult(
            landmarks_px=self.landmarks_px * factor,
            image_shape=image_shape,
            face_box=(int(x * factor), int(y * factor),
                      int(round(w * factor)), int(round(h * factor))),
            face_ratio=self.face_ratio,
            yaw=self.yaw, pitch=self.pitch, roll=self.roll,
            native_face_width=self.native_face_width,
        )


def detect(image_bgr: np.ndarray) -> LandmarkResult | None:
    """Detect one face. Returns None when no face is found."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

    xs, ys = pts[:, 0], pts[:, 1]
    x0, y0 = float(xs.min()), float(ys.min())
    bw, bh = float(xs.max() - x0), float(ys.max() - y0)
    face_box = (int(x0), int(y0), int(round(bw)), int(round(bh)))
    face_ratio = bh / h if h else 0.0

    yaw = pitch = roll = 0.0
    if result.facial_transformation_matrixes:
        yaw, pitch, roll = _pose_from_matrix(
            result.facial_transformation_matrixes[0]
        )
    return LandmarkResult(pts, (h, w), face_box, face_ratio, yaw, pitch, roll,
                          native_face_width=float(face_box[2]))


def crop_face(image_bgr: np.ndarray, result: LandmarkResult,
              margin: float = 0.25) -> np.ndarray:
    """Crop a padded region around the detected face box."""
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = result.face_box
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + bw + mx, w), min(y + bh + my, h)
    return image_bgr[y0:y1, x0:x1].copy()


def _pose_from_matrix(matrix) -> tuple[float, float, float]:
    """Extract yaw/pitch/roll (degrees) from a 4x4 transformation matrix."""
    m = np.array(matrix, dtype=np.float64).reshape(4, 4)
    r = m[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
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
