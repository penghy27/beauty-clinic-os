"""Imaging pipeline orchestrator.

One call runs the full chain: detect -> canonical scale -> quality gate ->
colour/exposure normalisation -> region patches -> skin metrics. The UI only
ever talks to process_photo().

Canonical scale is the backbone of cross-visit comparability: after face
detection the image is resized so the face box is a fixed width, so texture
kernels and blob-area thresholds sample the same physical skin patch whether
the photo came from a phone camera or a low-resolution webcam.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline import landmarks, metrics, normalize, quality, regions
from pipeline.landmarks import LandmarkResult
from pipeline.metrics import ProfileEntry
from pipeline.quality import QualityReport
from pipeline.regions import RegionPatch

# Every metric is computed with the face box scaled to this width (px).
CANONICAL_FACE_WIDTH = 384.0

# Upper bound on enlargement: captures needing more than this are too
# low-resolution to fake detail into, and the quality gate's resolution
# check rejects them anyway.
_MAX_UPSCALE = 2.5


@dataclass
class PipelineResult:
    ok: bool
    message: str
    landmark: LandmarkResult | None = None
    quality: QualityReport | None = None
    normalized_bgr: np.ndarray | None = None
    wb_info: dict = field(default_factory=dict)
    regions: dict[str, RegionPatch] = field(default_factory=dict)
    profile: list[ProfileEntry] = field(default_factory=list)
    overall: float = 0.0


def process_photo(image_bgr: np.ndarray) -> PipelineResult:
    """Run the imaging pipeline on a single BGR image."""
    lm = landmarks.detect(image_bgr)
    if lm is None:
        return PipelineResult(
            ok=False,
            message="未偵測到人臉，請重新上傳清楚的正面照。",
        )

    canonical, lm = _to_canonical(image_bgr, lm)
    q = quality.assess(canonical, lm)
    # Saturated capture pixels carry no colour information; flag them
    # before normalisation rescales them below the detection threshold.
    valid_mask = canonical.max(axis=2) < 250
    normalized, wb_info = normalize.white_balance(canonical, lm.face_box)
    region_patches = regions.build_regions(lm)
    profile = metrics.analyze(normalized, region_patches, valid_mask)
    overall = metrics.overall_score(profile)

    return PipelineResult(
        ok=True,
        message="分析完成。",
        landmark=lm,
        quality=q,
        normalized_bgr=normalized,
        wb_info=wb_info,
        regions=region_patches,
        profile=profile,
        overall=overall,
    )


def _to_canonical(
    image_bgr: np.ndarray, lm: LandmarkResult
) -> tuple[np.ndarray, LandmarkResult]:
    """Resize the image so the face box has the canonical width."""
    factor = CANONICAL_FACE_WIDTH / max(lm.face_box[2], 1)
    factor = min(factor, _MAX_UPSCALE)
    if abs(factor - 1.0) < 0.01:
        return image_bgr, lm

    h, w = image_bgr.shape[:2]
    size = (max(int(round(w * factor)), 1), max(int(round(h * factor)), 1))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, size, interpolation=interp)
    return resized, lm.scaled(factor, resized.shape[:2])
