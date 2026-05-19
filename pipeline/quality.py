"""Capture quality gate — the photo-consistency depth area.

Standardised intake means a photo only becomes a clinical record if it is
sharp, frontal, well-framed and evenly lit. Each check is explainable and
the consultant sees exactly why a photo was rejected, so re-shoots are fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.landmarks import LandmarkResult

# --- thresholds (tuned for typical phone selfies) ---
BLUR_FAIL, BLUR_WARN = 55.0, 110.0          # variance of Laplacian
FACE_RATIO_FAIL, FACE_RATIO_WARN = 0.22, 0.38
POSE_FAIL, POSE_WARN = 20.0, 12.0           # yaw / pitch, degrees
ROLL_WARN = 12.0
DARK_FAIL, DARK_WARN = 55.0, 85.0           # mean face brightness (0-255)
BRIGHT_WARN, BRIGHT_FAIL = 205.0, 230.0
SYMMETRY_WARN, SYMMETRY_FAIL = 22.0, 42.0   # left/right brightness gap
CLIP_WARN = 0.08                            # fraction of blown-out pixels

PASS_SCORE = 60.0
FAIL_PENALTY, WARN_PENALTY = 26.0, 9.0


@dataclass
class QualityCheck:
    name: str       # zh label
    status: str     # "pass" | "warn" | "fail"
    detail: str
    value: float

    @property
    def icon(self) -> str:
        return {"pass": "✅", "warn": "⚠️", "fail": "❌"}[self.status]


@dataclass
class QualityReport:
    score: float                  # 0-100 Capture Quality Score
    passed: bool
    checks: list[QualityCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "status": c.status,
                 "detail": c.detail, "value": round(c.value, 2)}
                for c in self.checks
            ],
        }


def assess(image_bgr: np.ndarray, lm: LandmarkResult) -> QualityReport:
    """Score a photo's capture quality given its detected landmarks."""
    checks = [
        _check_sharpness(image_bgr, lm),
        _check_face_size(lm),
        _check_pose(lm),
        _check_exposure(image_bgr, lm),
        _check_lighting_symmetry(image_bgr, lm),
    ]
    penalty = sum(
        FAIL_PENALTY if c.status == "fail"
        else WARN_PENALTY if c.status == "warn"
        else 0.0
        for c in checks
    )
    score = round(max(0.0, 100.0 - penalty), 1)
    passed = score >= PASS_SCORE and not any(c.status == "fail" for c in checks)
    return QualityReport(score=score, passed=passed, checks=checks)


def _face_gray(image_bgr: np.ndarray, lm: LandmarkResult) -> np.ndarray:
    x, y, w, h = lm.face_box
    H, W = image_bgr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = image_bgr
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _check_sharpness(image_bgr: np.ndarray, lm: LandmarkResult) -> QualityCheck:
    gray = _face_gray(image_bgr, lm)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if var < BLUR_FAIL:
        return QualityCheck("清晰度", "fail",
                            f"照片模糊（銳利度 {var:.0f}），請重拍。", var)
    if var < BLUR_WARN:
        return QualityCheck("清晰度", "warn",
                            f"略微模糊（銳利度 {var:.0f}），建議重拍。", var)
    return QualityCheck("清晰度", "pass", f"清晰（銳利度 {var:.0f}）。", var)


def _check_face_size(lm: LandmarkResult) -> QualityCheck:
    r = lm.face_ratio
    if r < FACE_RATIO_FAIL:
        return QualityCheck("臉部大小", "fail",
                            f"臉部太小（占畫面 {r*100:.0f}%），請靠近。", r)
    if r < FACE_RATIO_WARN:
        return QualityCheck("臉部大小", "warn",
                            f"臉部偏小（占畫面 {r*100:.0f}%），建議靠近。", r)
    return QualityCheck("臉部大小", "pass",
                        f"構圖適中（占畫面 {r*100:.0f}%）。", r)


def _check_pose(lm: LandmarkResult) -> QualityCheck:
    off = max(abs(lm.yaw), abs(lm.pitch))
    detail = f"yaw {lm.yaw:+.0f}° / pitch {lm.pitch:+.0f}° / roll {lm.roll:+.0f}°"
    if off > POSE_FAIL:
        return QualityCheck("正面角度", "fail", f"臉未正對鏡頭（{detail}）。", off)
    if off > POSE_WARN or abs(lm.roll) > ROLL_WARN:
        return QualityCheck("正面角度", "warn", f"角度略偏（{detail}）。", off)
    return QualityCheck("正面角度", "pass", f"正面良好（{detail}）。", off)


def _check_exposure(image_bgr: np.ndarray, lm: LandmarkResult) -> QualityCheck:
    gray = _face_gray(image_bgr, lm)
    mean = float(gray.mean())
    clip = float((gray > 250).mean())
    if mean < DARK_FAIL:
        return QualityCheck("曝光", "fail",
                            f"臉部過暗（亮度 {mean:.0f}），請加強打光。", mean)
    if mean > BRIGHT_FAIL or clip > CLIP_WARN * 2:
        return QualityCheck("曝光", "fail",
                            f"臉部過曝（亮度 {mean:.0f}），請降低光線。", mean)
    if mean < DARK_WARN or mean > BRIGHT_WARN or clip > CLIP_WARN:
        return QualityCheck("曝光", "warn",
                            f"曝光略偏（亮度 {mean:.0f}）。", mean)
    return QualityCheck("曝光", "pass", f"曝光適中（亮度 {mean:.0f}）。", mean)


def _check_lighting_symmetry(image_bgr: np.ndarray,
                             lm: LandmarkResult) -> QualityCheck:
    gray = _face_gray(image_bgr, lm)
    mid = gray.shape[1] // 2
    left = float(gray[:, :mid].mean())
    right = float(gray[:, mid:].mean())
    gap = abs(left - right)
    if gap > SYMMETRY_FAIL:
        return QualityCheck("光線均勻", "fail",
                            f"明顯側光（左右亮度差 {gap:.0f}）。", gap)
    if gap > SYMMETRY_WARN:
        return QualityCheck("光線均勻", "warn",
                            f"光線略不均（左右亮度差 {gap:.0f}）。", gap)
    return QualityCheck("光線均勻", "pass",
                        f"光線均勻（左右亮度差 {gap:.0f}）。", gap)
