"""Full-pipeline regression tests on the committed sample photos.

The golden fixture pins the Skin Profile each photo produces; any change
to the imaging chain that shifts the numbers turns these tests red, so
method drift is always an explicit decision (regenerate the fixture with
``python tests/make_golden.py`` and bump METHOD_VERSION when the change
is intentional).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pytest

from pipeline.metrics import NOISE_BAND, by_metric
from pipeline.process import process_photo

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden_profiles.json"

SUB_SCORE_TOLERANCE = 3.0  # same-platform landmark jitter allowance


def _photos():
    return sorted((ROOT / "sample_photos").glob("hero_v*.jpg"))


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="golden fixture generated on macOS; OpenCV/MediaPipe builds on other "
    "platforms shift count-based scores (e.g. spots) beyond the jitter allowance",
)
@pytest.mark.parametrize("path", _photos(), ids=lambda p: p.name)
def test_profile_matches_golden(path, golden):
    result = process_photo(cv2.imread(str(path)))
    assert result.ok
    expected = golden[path.name]

    assert result.quality.passed == expected["quality_passed"]
    assert abs(result.overall - expected["overall"]) <= SUB_SCORE_TOLERANCE

    got = {f"{e.region}.{e.metric}": e.sub_score for e in result.profile}
    assert set(got) == set(expected["profile"])
    for key, sub in expected["profile"].items():
        assert abs(got[key] - sub) <= SUB_SCORE_TOLERANCE, key


def test_downsampled_capture_stays_in_noise_band():
    """The depth-area claim itself: a 1280px re-capture of the same face
    must score within each metric's noise band of the full-resolution run."""
    img = cv2.imread(str(_photos()[0]))
    h, w = img.shape[:2]
    f = 1280 / max(h, w)
    small = cv2.resize(img, (int(w * f), int(h * f)),
                       interpolation=cv2.INTER_AREA)

    full = by_metric(process_photo(img).profile)
    down = by_metric(process_photo(small).profile)
    for metric, band in NOISE_BAND.items():
        assert abs(full[metric] - down[metric]) <= band, metric


def test_no_face_is_rejected():
    import numpy as np
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = process_photo(blank)
    assert not result.ok
    assert "人臉" in result.message
