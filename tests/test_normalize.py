"""Colour/exposure normalisation tests."""
from __future__ import annotations

import numpy as np

from pipeline.normalize import FACE_LUMINANCE_TARGET, white_balance
from tests.conftest import make_image


def _channel_means(img):
    return img.reshape(-1, 3).mean(axis=0)


def test_full_frame_gray_world_neutralises_cast():
    img = make_image(bgr=(110, 130, 150), noise=4)
    out, info = white_balance(img)
    b, g, r = _channel_means(out)
    assert info["method"] == "gray_world"
    assert info["gain_luminance"] == 1.0
    assert max(b, g, r) - min(b, g, r) < 3.0
    assert info["original_cast"] == "偏暖（紅）"


def test_background_anchored_wb_preserves_face_colour():
    # Neutral background, reddish "face", warm global cast on top.
    img = np.full((200, 200, 3), (128, 128, 128), dtype=np.float32)
    face_box = (60, 60, 80, 80)
    img[60:140, 60:140] = (105, 125, 165)  # skin-like: R above B
    cast = img * np.array([0.9, 1.0, 1.1])  # warm light
    img8 = np.clip(cast, 0, 255).astype(np.uint8)

    out, info = white_balance(img8, face_box)
    assert info["method"] == "gray_world_bg+face_lum"

    face = out[60:140, 60:140].reshape(-1, 3).mean(axis=0)
    # Exposure anchored on the face...
    assert abs(face.mean() - FACE_LUMINANCE_TARGET) < 3.0
    # ...while the face keeps its red-over-blue signal (not forced gray).
    assert face[2] - face[0] > 30.0
    # The cast estimate came from the background, which ends up neutral.
    bg = out[:40].reshape(-1, 3).mean(axis=0)
    assert max(bg) - min(bg) < 5.0


def test_face_filling_frame_falls_back_to_full_frame():
    img = make_image(bgr=(120, 130, 140), noise=3)
    out, info = white_balance(img, (0, 0, 200, 200))
    assert info["method"] == "gray_world+face_lum"
    assert abs(_channel_means(out).mean() - FACE_LUMINANCE_TARGET) < 3.0
