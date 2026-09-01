"""Skin metric extraction — turns a normalised photo into a Skin Profile.

Four classical-CV metrics per region. Every metric has a documented formula
and a 0-100 sub-score (100 = best). Nothing is a black box: each number is
reproducible from the pixels.

All metrics assume the canonical face scale enforced by the pipeline
(``process.CANONICAL_FACE_WIDTH``), so kernel sizes and blob areas cover the
same physical skin patch no matter what resolution the photo arrived at.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.regions import RegionPatch

METRICS = ["redness", "evenness", "texture", "spots"]

METRIC_LABELS_ZH = {
    "redness": "泛紅",
    "evenness": "膚色均勻",
    "texture": "肌膚細緻度",
    "spots": "斑點 / 瑕疵",
}

# (good_value, bad_value) — linear map to a 0-100 sub-score, clamped.
# Tuned at the canonical face scale on the reference photo set.
RANGES = {
    "redness": (7.0, 30.0),    # CIELAB a* offset; higher = redder = worse
    "evenness": (5.0, 30.0),   # CV of L* (%); higher = patchier = worse
    "texture": (2.0, 22.0),    # band-pass Laplacian variance; higher = rougher
    "spots": (0.0, 14.0),      # blemish blob count; higher = worse
}

# Between-visit noise band per metric, in sub-score points: the largest
# drift observed on the sensitivity harness (scripts/repeatability.py)
# under capture variation the quality gate accepts without warning, plus
# a small margin. A cross-visit delta below this band is measurement
# noise, not change.
NOISE_BAND = {
    "redness": 4.0,
    "evenness": 3.0,
    "texture": 6.0,
    "spots": 5.0,
}

# Gaussian pre-blur for the texture metric: measures mid-frequency surface
# structure instead of pixel-level gradients, which mostly encode capture
# sharpness/resolution rather than skin.
_TEXTURE_SIGMA = 1.0

# Black-hat structuring element and minimum blob area at canonical scale
# (equivalent to the v1 15px/6px values at the reference photos' native
# resolution).
_SPOT_KERNEL = 9
_SPOT_MIN_AREA = 2


@dataclass
class ProfileEntry:
    region: str
    metric: str
    value: float
    sub_score: float


def analyze(normalized_bgr: np.ndarray,
            regions: dict[str, RegionPatch],
            valid_mask: np.ndarray | None = None) -> list[ProfileEntry]:
    """Compute every (region, metric) pair for a normalised photo.

    ``valid_mask`` marks pixels that were not saturated in the capture;
    the redness metric skips saturated pixels because a clipped channel
    has already destroyed the hue information there.
    """
    lab = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2GRAY)
    l_chan = lab[:, :, 0].astype(np.float64)
    a_chan = lab[:, :, 1].astype(np.float64)
    lap = cv2.Laplacian(
        cv2.GaussianBlur(gray, (0, 0), _TEXTURE_SIGMA), cv2.CV_64F
    )
    blackhat = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (_SPOT_KERNEL, _SPOT_KERNEL)),
    )

    entries: list[ProfileEntry] = []
    for name, patch in regions.items():
        sel = patch.mask(normalized_bgr.shape) > 0
        if sel.sum() < 25:
            continue
        # Saturated pixels corrupt hue, so redness skips them. Evenness
        # keeps the full region: it measures spread, and truncating the
        # bright tail would bias exactly what it measures.
        redness_sel = sel
        if valid_mask is not None:
            unsaturated = sel & valid_mask
            if unsaturated.sum() >= 25:
                redness_sel = unsaturated
        values = {
            "redness": _redness(a_chan, redness_sel),
            "evenness": _evenness(l_chan, sel),
            "texture": _texture(lap, gray, sel),
            "spots": _spots(blackhat, sel),
        }
        for metric, value in values.items():
            entries.append(
                ProfileEntry(name, metric, round(value, 2),
                             _sub_score(metric, value))
            )
    return entries


def _redness(a_chan: np.ndarray, sel: np.ndarray) -> float:
    """Mean CIELAB a* offset from neutral (128)."""
    return float(a_chan[sel].mean() - 128.0)


def _evenness(l_chan: np.ndarray, sel: np.ndarray) -> float:
    """Coefficient of variation of L* (%), lower = more even.

    Dividing by the region's own mean makes the number robust to exposure
    and to lighting geometry (a shadowed region scales std and mean alike).
    """
    region = l_chan[sel]
    return float(100.0 * region.std() / max(region.mean(), 1e-6))


def _texture(lap: np.ndarray, gray: np.ndarray, sel: np.ndarray) -> float:
    """Band-pass Laplacian variance, contrast-normalised to the region.

    The Gaussian pre-blur suppresses the pixel-level frequencies dominated
    by capture sharpness; dividing by the squared relative brightness of
    the region cancels the contrast scaling that brightness applies to
    every gradient.
    """
    rel_brightness = float(gray[sel].mean()) / 128.0
    return float(lap[sel].var() / max(rel_brightness**2, 1e-6))


def _spots(blackhat: np.ndarray, sel: np.ndarray) -> float:
    """Count dark blemish blobs via black-hat morphology."""
    masked = np.where(sel, blackhat, 0).astype(np.uint8)
    _, thresh = cv2.threshold(masked, 20, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    return float(sum(1 for i in range(1, n)
                     if stats[i, cv2.CC_STAT_AREA] >= _SPOT_MIN_AREA))


def _sub_score(metric: str, value: float) -> float:
    """Map a raw metric value to a 0-100 sub-score (100 = best)."""
    good, bad = RANGES[metric]
    score = 100.0 * (bad - value) / (bad - good)
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def overall_score(entries: list[ProfileEntry]) -> float:
    """Mean sub-score across all entries — the Skin Health Index."""
    if not entries:
        return 0.0
    return round(sum(e.sub_score for e in entries) / len(entries), 1)


def by_metric(entries: list[ProfileEntry]) -> dict[str, float]:
    """Average sub-score per metric across all regions."""
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for e in entries:
        out[e.metric].append(e.sub_score)
    return {m: round(sum(v) / len(v), 1) for m, v in out.items() if v}
