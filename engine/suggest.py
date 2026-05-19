"""Treatment suggestion engine — explainable, editable, outcome-aware.

* Explainable: every suggestion names the exact metric, region and score
  that triggered it, so it never feels like an upsell.
* Editable: suggestions serialise to JSON; staff edit the copy in the DB.
* Outcome-aware: when a previous visit exists, each suggestion carries a
  trend note comparing this visit to the last — this closes the loop.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from pipeline.metrics import METRIC_LABELS_ZH, ProfileEntry, by_metric
from pipeline.regions import REGION_LABELS_ZH

RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "treatments.yaml"

IMPROVED_DELTA = 5.0   # sub-score points counted as a real improvement


@dataclass
class Suggestion:
    rule_id: str
    metric: str
    metric_label: str
    region: str
    region_label: str
    treatment: str
    detail: str
    reason: str          # data-grounded explanation
    caution: str
    priority: int
    sub_score: float     # triggering metric average
    trend: str = ""      # outcome-aware note (closed loop); "" on first visit


def _load_rules() -> list[dict]:
    with open(RULES_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["rules"]


def suggest(current: list[ProfileEntry],
            previous: list[ProfileEntry] | None = None) -> list[Suggestion]:
    """Generate ranked treatment suggestions from a Skin Profile.

    When `previous` is supplied, suggestions become outcome-aware.
    """
    rules = _load_rules()
    curr_metric = by_metric(current)
    prev_metric = by_metric(previous) if previous else {}
    worst = _worst_region_per_metric(current)

    suggestions: list[Suggestion] = []
    fired: set[str] = set()
    for rule in sorted(rules, key=lambda r: r["max_sub_score"]):
        metric = rule["metric"]
        if metric in fired:
            continue
        avg = curr_metric.get(metric)
        if avg is None or avg > rule["max_sub_score"]:
            continue
        fired.add(metric)

        region, region_score = worst.get(metric, ("", 0.0))
        reason = (
            f"{METRIC_LABELS_ZH[metric]}平均膚質分 {avg:.0f}，"
            f"最弱區為{REGION_LABELS_ZH.get(region, region)}（{region_score:.0f} 分）"
        )
        suggestions.append(
            Suggestion(
                rule_id=rule["id"],
                metric=metric,
                metric_label=METRIC_LABELS_ZH[metric],
                region=region,
                region_label=REGION_LABELS_ZH.get(region, region),
                treatment=rule["treatment"],
                detail=rule["detail"],
                reason=reason,
                caution=rule["caution"],
                priority=int(rule["priority"]),
                sub_score=round(avg, 1),
                trend=_trend_note(metric, avg, prev_metric.get(metric)),
            )
        )

    suggestions.sort(key=lambda s: (s.priority, s.sub_score))
    return suggestions


def _worst_region_per_metric(
    entries: list[ProfileEntry],
) -> dict[str, tuple[str, float]]:
    worst: dict[str, tuple[str, float]] = {}
    for e in entries:
        cur = worst.get(e.metric)
        if cur is None or e.sub_score < cur[1]:
            worst[e.metric] = (e.region, e.sub_score)
    return worst


def _trend_note(metric: str, curr_avg: float,
                prev_avg: float | None) -> str:
    """Outcome-aware comparison vs the previous visit."""
    if prev_avg is None:
        return ""
    label = METRIC_LABELS_ZH[metric]
    delta = curr_avg - prev_avg
    span = f"（{prev_avg:.0f} → {curr_avg:.0f}）"
    if delta >= IMPROVED_DELTA:
        return f"{label}較上次進步 {delta:+.0f} 分{span}，療程有效，建議延續同方案。"
    if delta <= -IMPROVED_DELTA:
        return f"{label}較上次退步 {delta:+.0f} 分{span}，改善有限，建議調整療程強度或方式。"
    return f"{label}較上次持平{span}，建議維持並持續追蹤。"


def positive_notes(current: list[ProfileEntry],
                   previous: list[ProfileEntry]) -> list[str]:
    """Metrics that improved notably — retention-facing encouragement."""
    curr_metric = by_metric(current)
    prev_metric = by_metric(previous)
    notes: list[str] = []
    for metric, curr_avg in curr_metric.items():
        prev_avg = prev_metric.get(metric)
        if prev_avg is None:
            continue
        delta = curr_avg - prev_avg
        if delta >= IMPROVED_DELTA:
            notes.append(
                f"{METRIC_LABELS_ZH[metric]}進步 {delta:+.0f} 分"
                f"（{prev_avg:.0f} → {curr_avg:.0f}）"
            )
    return notes


def to_json(suggestions: list[Suggestion]) -> str:
    return json.dumps([asdict(s) for s in suggestions], ensure_ascii=False)


def from_json(raw: str) -> list[Suggestion]:
    if not raw:
        return []
    return [Suggestion(**d) for d in json.loads(raw)]
