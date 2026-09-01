"""Quality-gate tests on synthetic degradations (no MediaPipe needed)."""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import quality
from tests.conftest import fake_landmarks, make_image


def _sharp_image():
    # 700px wide so the default fake face box (60% of width) clears the
    # capture-resolution check.
    return make_image(h=700, w=700, bgr=(140, 140, 140), noise=30, seed=5)


def _status(report, name):
    return next(c.status for c in report.checks if c.name == name)


def test_good_capture_passes():
    img = _sharp_image()
    report = quality.assess(img, fake_landmarks(img))
    assert report.passed
    assert report.score == 100.0
    assert all(c.status == "pass" for c in report.checks)


def test_blur_fails_sharpness():
    img = cv2.GaussianBlur(_sharp_image(), (0, 0), 6)
    report = quality.assess(img, fake_landmarks(img))
    assert _status(report, "清晰度") == "fail"
    assert not report.passed


def test_dark_image_fails_exposure():
    img = (make_image(h=400, w=400, bgr=(40, 40, 40), noise=30, seed=5))
    report = quality.assess(img, fake_landmarks(img))
    assert _status(report, "曝光") == "fail"


def test_side_lighting_fails_symmetry():
    img = _sharp_image()
    half = img.shape[1] // 2
    img[:, :half] = (img[:, :half] * 0.4).astype(np.uint8)
    report = quality.assess(img, fake_landmarks(img))
    assert _status(report, "光線均勻") == "fail"


def test_turned_head_fails_pose():
    img = _sharp_image()
    report = quality.assess(img, fake_landmarks(img, yaw=25.0))
    assert _status(report, "正面角度") == "fail"


def test_small_face_fails_face_size():
    img = _sharp_image()
    report = quality.assess(img, fake_landmarks(img, face_ratio=0.15))
    assert _status(report, "臉部大小") == "fail"


def test_low_capture_resolution_flagged():
    img = _sharp_image()
    report = quality.assess(img, fake_landmarks(img, native_face_width=190))
    assert _status(report, "解析度") == "fail"

    report = quality.assess(img, fake_landmarks(img, native_face_width=300))
    assert _status(report, "解析度") == "warn"


def test_score_penalties_and_pass_rule():
    img = _sharp_image()
    warn_report = quality.assess(
        img, fake_landmarks(img, native_face_width=300))
    assert warn_report.score == 100.0 - quality.WARN_PENALTY
    assert warn_report.passed  # one warn stays above the pass line

    fail_report = quality.assess(
        img, fake_landmarks(img, native_face_width=190))
    assert fail_report.score == 100.0 - quality.FAIL_PENALTY
    assert not fail_report.passed  # any fail rejects regardless of score


def test_report_serialises():
    img = _sharp_image()
    d = quality.assess(img, fake_landmarks(img)).to_dict()
    assert d["passed"] is True
    assert len(d["checks"]) == 6
