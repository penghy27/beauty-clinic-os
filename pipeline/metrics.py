"""Skin metric extraction — turns a normalised photo into a Skin Profile.

Four classical-CV metrics per region. Every metric has a documented formula
and a 0-100 sub-score (100 = best). Nothing is a black box: each number is
reproducible from the pixels.
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
_RANGES = {
    "redness": (8.0, 34.0),    # CIELAB a* offset; higher = redder = worse
    "evenness": (6.0, 30.0),   # std of L*; higher = patchier = worse
    "texture": (15.0, 260.0),  # Laplacian variance; higher = rougher = worse
    "spots": (0.0, 14.0),      # blemish blob count; higher = worse
}


@dataclass
class ProfileEntry:
    region: str
    metric: str
    value: float
    sub_score: float


def analyze(normalized_bgr: np.ndarray,
            regions: dict[str, RegionPatch]) -> list[ProfileEntry]:
    """Compute every (region, metric) pair for a normalised photo."""
    lab = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2GRAY)
    l_chan = lab[:, :, 0].astype(np.float64)
    a_chan = lab[:, :, 1].astype(np.float64)

    entries: list[ProfileEntry] = []
    for name, patch in regions.items():
        sel = patch.mask(normalized_bgr.shape) > 0
        if sel.sum() < 25:
            continue
        values = {
            "redness": _redness(a_chan, sel),
            "evenness": _evenness(l_chan, sel),
            "texture": _texture(gray, sel),
            "spots": _spots(gray, sel),
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
    """Std of CIELAB L* — tone uniformity (lower is more even)."""
    return float(l_chan[sel].std())


def _texture(gray: np.ndarray, sel: np.ndarray) -> float:
    """Laplacian variance inside the region — surface roughness proxy."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[sel].var())


def _spots(gray: np.ndarray, sel: np.ndarray) -> float:
    """Count dark blemish blobs via black-hat morphology."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    masked = np.where(sel, blackhat, 0).astype(np.uint8)
    _, thresh = cv2.threshold(masked, 20, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
    return float(sum(1 for i in range(1, n)
                     if stats[i, cv2.CC_STAT_AREA] >= 6))


def _sub_score(metric: str, value: float) -> float:
    """Map a raw metric value to a 0-100 sub-score (100 = best)."""
    good, bad = _RANGES[metric]
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
