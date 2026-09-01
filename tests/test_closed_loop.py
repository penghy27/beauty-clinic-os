"""Closed-loop integration test on a throwaway database: seed -> visits ->
profiles -> recommendations -> outcome-aware trend -> overdue filter."""
from __future__ import annotations

import json
from datetime import date, timedelta


def test_seed_builds_the_closed_loop(tmp_db):
    from db.models import METHOD_VERSION, Customer
    from db.seed import seed_if_empty

    seed_if_empty()
    session = tmp_db.get_session()
    try:
        customers = session.query(Customer).all()
        assert len(customers) >= 5

        hero = next(c for c in customers if len(c.visits) == 3)

        # every visit carries a photo, a profile and a recommendation
        for visit in hero.visits:
            photo = visit.photos[0]
            assert photo.profiles, "visit without a Skin Profile"
            assert all(p.method_version == METHOD_VERSION
                       for p in photo.profiles)
            assert visit.recommendation is not None
            assert json.loads(visit.recommendation.generated_json)

        # outcome awareness: later visits carry a trend note
        last_gen = json.loads(hero.visits[-1].recommendation.generated_json)
        assert any(s["trend"] for s in last_gen)
        first_gen = json.loads(hero.visits[0].recommendation.generated_json)
        assert all(s["trend"] == "" for s in first_gen)

        # package tally replaces the paper book
        pkg = hero.packages[0]
        assert pkg.sessions_left == pkg.total_sessions - pkg.sessions_used

        # seeding is idempotent
        seed_if_empty()
        assert session.query(Customer).count() == len(customers)
    finally:
        session.close()


def test_overdue_filter(tmp_db):
    from db.models import Customer, Visit, init_db
    from ui.common import is_overdue

    init_db()
    session = tmp_db.get_session()
    try:
        overdue = Customer(name="逾期客")
        overdue.visits.append(Visit(
            visit_date=date.today() - timedelta(days=60),
            next_revisit_date=date.today() - timedelta(days=10),
        ))
        fresh = Customer(name="正常客")
        fresh.visits.append(Visit(
            visit_date=date.today() - timedelta(days=10),
            next_revisit_date=date.today() + timedelta(days=20),
        ))
        session.add_all([overdue, fresh])
        session.commit()

        assert is_overdue(overdue)
        assert not is_overdue(fresh)
    finally:
        session.close()


def test_method_version_guard(tmp_db):
    """Profiles from an older method version must not be trend-compared."""
    from db.models import Customer, Photo, SkinProfile, Visit, init_db
    from ui.common import visit_method_version
    from ui.consultation import _previous_entries

    init_db()
    session = tmp_db.get_session()
    try:
        cust = Customer(name="舊資料客")
        visit = Visit(visit_date=date.today() - timedelta(days=30))
        photo = Photo(view="front")
        photo.profiles.append(SkinProfile(
            region="forehead", metric="redness",
            value=12.0, sub_score=55.0, method_version="v1",
        ))
        visit.photos.append(photo)
        cust.visits.append(visit)
        session.add(cust)
        session.commit()

        assert visit_method_version(visit) == "v1"
        assert _previous_entries(cust) is None  # old scale: start fresh
    finally:
        session.close()
