"""Suggestion-engine tests: rule firing, explanations, trend notes,
serialisation and the LLM fallback contract."""
from __future__ import annotations

from engine import suggest as engine
from pipeline.metrics import NOISE_BAND, ProfileEntry


def profile(**scores: float) -> list[ProfileEntry]:
    """Two regions per metric at the given sub-scores."""
    entries = []
    for metric, sub in scores.items():
        for region in ("left_cheek", "forehead"):
            entries.append(ProfileEntry(region, metric, 0.0, sub))
    return entries


def test_low_metric_fires_most_severe_rule_only():
    out = engine.suggest(profile(redness=30, evenness=90,
                                 texture=90, spots=90))
    assert [s.metric for s in out] == ["redness"]
    assert out[0].rule_id == "redness_significant"


def test_moderate_metric_fires_moderate_tier():
    out = engine.suggest(profile(redness=50, evenness=90,
                                 texture=90, spots=90))
    assert [s.rule_id for s in out] == ["redness_moderate"]


def test_healthy_profile_yields_no_suggestions():
    assert engine.suggest(profile(redness=90, evenness=90,
                                  texture=90, spots=90)) == []


def test_reason_names_worst_region_and_score():
    entries = profile(redness=90, evenness=90, texture=90, spots=90)
    entries += [ProfileEntry("chin", "redness", 0.0, 20.0)]  # drags avg to 66.7
    entries += [ProfileEntry("chin", "spots", 0.0, 10.0)]    # avg 63.3 — above 62
    entries[0] = ProfileEntry("left_cheek", "redness", 0.0, 30.0)  # avg 46.7
    out = engine.suggest(entries)
    assert len(out) == 1
    s = out[0]
    assert s.metric == "redness"
    assert "泛紅" in s.reason and "下巴" in s.reason  # worst region named
    assert f"{s.sub_score:.0f}" in s.reason


def test_priority_orders_suggestions():
    out = engine.suggest(profile(redness=30, evenness=90,
                                 texture=50, spots=90))
    assert [s.rule_id for s in out] == ["redness_significant",
                                        "texture_moderate"]


def test_trend_notes_use_noise_band():
    band = NOISE_BAND["redness"]
    curr = profile(redness=50, evenness=90, texture=90, spots=90)

    improved = engine.suggest(curr, profile(redness=50 - band - 1))
    assert "進步" in improved[0].trend

    flat = engine.suggest(curr, profile(redness=50 - band + 1))
    assert "持平" in flat[0].trend

    worse = engine.suggest(curr, profile(redness=50 + band + 1))
    assert "退步" in worse[0].trend

    first_visit = engine.suggest(curr)
    assert first_visit[0].trend == ""


def test_positive_notes_report_improvements():
    prev = profile(redness=50, evenness=60, texture=70, spots=80)
    curr = profile(redness=50 + NOISE_BAND["redness"] + 1,
                   evenness=60, texture=70, spots=80)
    notes = engine.positive_notes(curr, prev)
    assert len(notes) == 1 and "泛紅" in notes[0]


def test_json_round_trip():
    out = engine.suggest(profile(redness=30, evenness=50,
                                 texture=90, spots=90),
                         profile(redness=20, evenness=55))
    again = engine.from_json(engine.to_json(out))
    assert again == out
    assert engine.from_json("") == []


def test_humanize_failure_keeps_template(monkeypatch):
    from engine import humanize

    def boom(suggestions):
        raise RuntimeError("api down")

    monkeypatch.setattr(humanize, "refine", boom)
    out = engine.suggest(profile(redness=30, evenness=90,
                                 texture=90, spots=90))
    assert len(out) == 1          # suggestion still produced...
    assert out[0].reason_humanized == ""  # ...with the template fallback


def test_humanize_success_fills_field(monkeypatch):
    from engine import humanize
    monkeypatch.setattr(
        humanize, "refine",
        lambda suggestions: {s.rule_id: "自然文案" for s in suggestions},
    )
    out = engine.suggest(profile(redness=30, evenness=90,
                                 texture=90, spots=90))
    assert out[0].reason_humanized == "自然文案"
    assert out[0].reason  # template stays available for audit
