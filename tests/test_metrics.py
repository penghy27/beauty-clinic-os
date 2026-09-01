"""Behavioural tests for the skin metrics: each one must move in the
documented direction when the corresponding skin property degrades."""
from __future__ import annotations

import cv2
import numpy as np

from pipeline import metrics
from pipeline.metrics import ProfileEntry, analyze, by_metric, overall_score
from tests.conftest import center_region, make_image


def _value(entries, metric):
    return next(e.value for e in entries if e.metric == metric)


def _sub(entries, metric):
    return next(e.sub_score for e in entries if e.metric == metric)


def test_redness_increases_with_red_shift():
    base = make_image(noise=3)
    redder = base.copy()
    redder[:, :, 2] = np.clip(redder[:, :, 2].astype(int) + 35, 0, 255)
    regions = center_region()
    a = analyze(base, regions)
    b = analyze(redder, regions)
    assert _value(b, "redness") > _value(a, "redness")
    assert _sub(b, "redness") < _sub(a, "redness")


def test_evenness_worsens_with_patchy_tone():
    base = make_image(noise=2)
    patchy = base.copy()
    patchy[60:140, 60:100] = (110, 115, 125)  # darker patch inside region
    regions = center_region()
    assert (_value(analyze(patchy, regions), "evenness")
            > _value(analyze(base, regions), "evenness"))


def test_evenness_is_brightness_invariant():
    base = make_image(noise=15, seed=3)
    darker = np.clip(base.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    regions = center_region()
    v_base = _value(analyze(base, regions), "evenness")
    v_dark = _value(analyze(darker, regions), "evenness")
    assert abs(v_base - v_dark) / v_base < 0.10


def test_texture_increases_with_roughness():
    smooth = make_image(noise=2)
    rough = make_image(noise=25, seed=11)
    regions = center_region()
    assert (_value(analyze(rough, regions), "texture")
            > _value(analyze(smooth, regions), "texture"))


def test_spots_counts_dark_blobs():
    plain = make_image(noise=2)
    dotted = plain.copy()
    for cx in (80, 100, 120):
        cv2.circle(dotted, (cx, 100), 3, (60, 60, 60), -1)
    regions = center_region()
    assert (_value(analyze(dotted, regions), "spots")
            >= _value(analyze(plain, regions), "spots") + 3)


def test_redness_skips_saturated_pixels():
    base = make_image(bgr=(120, 130, 180), noise=2)
    blown = base.copy()
    blown[85:115, 85:115] = (255, 255, 255)  # saturated square in region
    regions = center_region()
    valid = blown.max(axis=2) < 250

    with_mask = _value(analyze(blown, regions, valid_mask=valid), "redness")
    without = _value(analyze(blown, regions), "redness")
    reference = _value(analyze(base, regions), "redness")
    assert abs(with_mask - reference) < abs(without - reference)


def test_sub_score_clamps_to_range():
    good, bad = metrics.RANGES["redness"]
    assert metrics._sub_score("redness", good - 10) == 100.0
    assert metrics._sub_score("redness", bad + 10) == 0.0
    mid = (good + bad) / 2
    assert abs(metrics._sub_score("redness", mid) - 50.0) < 0.1


def test_overall_and_by_metric():
    entries = [
        ProfileEntry("a", "redness", 10.0, 80.0),
        ProfileEntry("b", "redness", 12.0, 60.0),
        ProfileEntry("a", "texture", 5.0, 40.0),
    ]
    assert overall_score(entries) == 60.0
    assert by_metric(entries) == {"redness": 70.0, "texture": 40.0}
    assert overall_score([]) == 0.0


def test_noise_band_covers_all_metrics():
    assert set(metrics.NOISE_BAND) == set(metrics.METRICS)
    assert all(v > 0 for v in metrics.NOISE_BAND.values())
