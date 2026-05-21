"""Seed demo data so the prototype always has a story to show.

Seeded Skin Profiles are plausible hand-set numbers (no photos needed), so a
cold restart on an ephemeral host still demonstrates the CRM, longitudinal
comparison and outcome-aware suggestions. Live photo uploads exercise the
real imaging pipeline on top of this.
"""
from __future__ import annotations

import random
import shutil
from datetime import date, datetime

from db.models import (
    METHOD_VERSION,
    PHOTO_DIR,
    PROJECT_ROOT,
    Customer,
    Package,
    Photo,
    Recommendation,
    SkinProfile,
    Visit,
    get_session,
    init_db,
)
from engine import suggest as engine
from pipeline.metrics import METRICS, ProfileEntry
from pipeline.regions import REGION_ORDER

# inverse of pipeline.metrics._RANGES — derive a plausible raw value
_RANGES = {
    "redness": (8.0, 34.0),
    "evenness": (6.0, 30.0),
    "texture": (15.0, 260.0),
    "spots": (0.0, 14.0),
}


def _raw_value(metric: str, sub_score: float) -> float:
    good, bad = _RANGES[metric]
    return round(bad - sub_score / 100.0 * (bad - good), 2)


def _make_profile(base: dict[str, float], seed: int) -> list[ProfileEntry]:
    """Build 6 regions x 4 metrics of ProfileEntry around per-metric bases."""
    rnd = random.Random(seed)
    entries: list[ProfileEntry] = []
    for region in REGION_ORDER:
        for metric in METRICS:
            sub = base[metric] + rnd.uniform(-8.0, 8.0)
            sub = round(max(6.0, min(97.0, sub)), 1)
            entries.append(
                ProfileEntry(region, metric, _raw_value(metric, sub), sub)
            )
    return entries


def _copy_sample_photo(src_name: str, label: str) -> tuple[str, str] | None:
    """Copy sample_photos/<src_name> into PHOTO_DIR as raw + norm files.

    Returns (raw_path, normalized_path), or None if the source is missing,
    so the seed degrades gracefully and the visit just shows "無照片紀錄".
    """
    source = PROJECT_ROOT / "sample_photos" / src_name
    if not source.exists():
        return None
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    raw = PHOTO_DIR / f"{label}_raw.jpg"
    norm = PHOTO_DIR / f"{label}_norm.jpg"
    shutil.copyfile(source, raw)
    shutil.copyfile(source, norm)
    return (str(raw), str(norm))


def _persist_visit(session, visit: Visit, entries: list[ProfileEntry],
                   quality_score: float,
                   previous: list[ProfileEntry] | None,
                   photo_paths: tuple[str, str] | None = None) -> None:
    """Attach a photo, skin profile and recommendation to a visit."""
    raw_path, norm_path = photo_paths if photo_paths else ("", "")
    photo = Photo(
        view="front",
        raw_path=raw_path,
        normalized_path=norm_path,
        quality_score=quality_score,
        quality_json="{}",
        wb_method="gray_world",
    )
    visit.photos.append(photo)
    for e in entries:
        photo.profiles.append(
            SkinProfile(
                region=e.region, metric=e.metric, value=e.value,
                sub_score=e.sub_score, method_version=METHOD_VERSION,
            )
        )
    suggestions = engine.suggest(entries, previous)
    visit.recommendation = Recommendation(
        generated_json=engine.to_json(suggestions),
        edited_json=engine.to_json(suggestions),
        status="approved" if previous else "draft",
    )


def seed_if_empty() -> None:
    """Populate demo data only when the database is empty. Idempotent."""
    init_db()
    session = get_session()
    if session.query(Customer).count() > 0:
        session.close()
        return

    # --- hero customer: 3 visits, visible improvement, closes the loop ---
    hero = Customer(
        name="陳怡君", sex="女", birth_year=1990, phone="0912-345-678",
        skin_note="混合性肌膚，主訴兩頰泛紅與毛孔粗大。",
        consent_signed_at=datetime(2026, 2, 20, 10, 0),
    )
    hero_pkg = Package(
        name="6 堂淨膚雷射療程", total_sessions=6, sessions_used=3,
        price=36000, is_paid=True, purchased_at=date(2026, 2, 20),
    )
    hero.packages.append(hero_pkg)

    # Monotonic improvement, but texture stays moderate so the final visit
    # still produces an outcome-aware suggestion ("療程有效，建議延續").
    hero_bases = [
        {"redness": 40, "evenness": 48, "texture": 39, "spots": 52},
        {"redness": 55, "evenness": 60, "texture": 49, "spots": 64},
        {"redness": 70, "evenness": 73, "texture": 58, "spots": 77},
    ]
    hero_dates = [date(2026, 2, 20), date(2026, 3, 25), date(2026, 4, 29)]
    hero_revisits = [date(2026, 3, 25), date(2026, 4, 29), date(2026, 5, 26)]
    prev_entries: list[ProfileEntry] | None = None
    for i, (base, vdate, rdate) in enumerate(
        zip(hero_bases, hero_dates, hero_revisits)
    ):
        visit = Visit(
            visit_date=vdate, consultant="諮詢師 A",
            next_revisit_date=rdate,
            notes=f"第 {i + 1} 次淨膚雷射療程。",
        )
        hero.visits.append(visit)
        entries = _make_profile(base, seed=100 + i)
        photo_paths = _copy_sample_photo(
            f"hero_v{i + 1}.jpg", f"hero_v{i + 1}"
        )
        _persist_visit(session, visit, entries, 88.0 - i * 2,
                       prev_entries, photo_paths)
        prev_entries = entries

    # --- other customers: populate the CRM list and the overdue filter ---
    others = [
        dict(
            name="林佳穎", sex="女", birth_year=1985, phone="0922-111-222",
            note="敏感性肌膚，季節交替易泛紅。",
            pkg=("4 堂舒緩護理", 4, 1, 18000, True),
            visits=[(date(2026, 3, 10), date(2026, 4, 10), 45)],
        ),
        dict(
            name="王思婷", sex="女", birth_year=1996, phone="0933-222-333",
            note="出油旺盛，T 字部位毛孔明顯。",
            pkg=("8 堂毛孔緊緻療程", 8, 2, 52000, False),
            visits=[(date(2026, 4, 2), date(2026, 5, 2), 55),
                    (date(2026, 5, 2), date(2026, 6, 5), 60)],
        ),
        dict(
            name="張雅文", sex="女", birth_year=1992, phone="0944-333-444",
            note="初次諮詢，主訴膚色不均與暗沉。",
            pkg=None,
            visits=[(date(2026, 5, 15), date(2026, 6, 15), 50)],
        ),
        dict(
            name="黃美玲", sex="女", birth_year=1979, phone="0955-444-555",
            note="淡斑療程已完成一個療程。",
            pkg=("6 堂淨白雷射", 6, 6, 42000, True),
            visits=[(date(2026, 3, 1), date(2026, 4, 1), 62),
                    (date(2026, 4, 1), date(2026, 5, 1), 68)],
        ),
    ]
    for idx, c in enumerate(others):
        cust = Customer(
            name=c["name"], sex=c["sex"], birth_year=c["birth_year"],
            phone=c["phone"], skin_note=c["note"],
            consent_signed_at=datetime(2026, 3, 1, 9, 0),
        )
        pkg = None
        if c["pkg"]:
            name, total, used, price, paid = c["pkg"]
            pkg = Package(
                name=name, total_sessions=total, sessions_used=used,
                price=price, is_paid=paid, purchased_at=date(2026, 3, 1),
            )
            cust.packages.append(pkg)
        prev = None
        for j, (vdate, rdate, base_score) in enumerate(c["visits"]):
            visit = Visit(
                visit_date=vdate, consultant="諮詢師 B",
                next_revisit_date=rdate, notes=f"第 {j + 1} 次就診。",
            )
            cust.visits.append(visit)
            base = {m: base_score + (j * 5) for m in METRICS}
            entries = _make_profile(base, seed=200 + idx * 10 + j)
            _persist_visit(session, visit, entries, base_score + 25, prev)
            prev = entries
        session.add(cust)

    session.add(hero)
    session.commit()
    session.close()
