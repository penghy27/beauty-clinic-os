"""Imaging pipeline orchestrator.

One call runs the full chain: detect -> quality gate -> white balance ->
region patches -> skin metrics. The UI only ever talks to process_photo().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline import landmarks, metrics, normalize, quality, regions
from pipeline.landmarks import LandmarkResult
from pipeline.metrics import ProfileEntry
from pipeline.quality import QualityReport
from pipeline.regions import RegionPatch


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

    q = quality.assess(image_bgr, lm)
    normalized, wb_info = normalize.white_balance(image_bgr)
    region_patches = regions.build_regions(lm)
    profile = metrics.analyze(normalized, region_patches)
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
