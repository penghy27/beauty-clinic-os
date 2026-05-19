"""SQLite data model for Beauty Clinic OS.

Six tables that carry the closed loop:
customers -> packages / visits -> photos -> skin_profiles -> recommendations.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from sqlalchemy import ForeignKey, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHOTO_DIR = DATA_DIR / "photos"
DB_PATH = DATA_DIR / "clinic.db"

METHOD_VERSION = "v1"


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    sex: Mapped[str] = mapped_column(default="")
    birth_year: Mapped[int | None] = mapped_column(default=None)
    phone: Mapped[str] = mapped_column(default="")
    skin_note: Mapped[str] = mapped_column(Text, default="")
    consent_signed_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    packages: Mapped[list[Package]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    visits: Mapped[list[Visit]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        order_by="Visit.visit_date",
    )


class Package(Base):
    """A pre-paid treatment package — replaces the paper session tally."""

    __tablename__ = "packages"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    name: Mapped[str]
    total_sessions: Mapped[int]
    sessions_used: Mapped[int] = mapped_column(default=0)
    price: Mapped[float] = mapped_column(default=0.0)
    is_paid: Mapped[bool] = mapped_column(default=False)
    purchased_at: Mapped[date] = mapped_column(default=date.today)

    customer: Mapped[Customer] = relationship(back_populates="packages")

    @property
    def sessions_left(self) -> int:
        return max(self.total_sessions - self.sessions_used, 0)


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    package_id: Mapped[int | None] = mapped_column(
        ForeignKey("packages.id"), default=None
    )
    visit_date: Mapped[date] = mapped_column(default=date.today)
    consultant: Mapped[str] = mapped_column(default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    next_revisit_date: Mapped[date | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    customer: Mapped[Customer] = relationship(back_populates="visits")
    photos: Mapped[list[Photo]] = relationship(
        back_populates="visit", cascade="all, delete-orphan"
    )
    recommendation: Mapped[Recommendation | None] = relationship(
        back_populates="visit", cascade="all, delete-orphan", uselist=False
    )


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"))
    view: Mapped[str] = mapped_column(default="front")
    raw_path: Mapped[str] = mapped_column(default="")
    normalized_path: Mapped[str] = mapped_column(default="")
    quality_score: Mapped[float] = mapped_column(default=0.0)
    quality_json: Mapped[str] = mapped_column(Text, default="{}")
    wb_method: Mapped[str] = mapped_column(default="")
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    visit: Mapped[Visit] = relationship(back_populates="photos")
    profiles: Mapped[list[SkinProfile]] = relationship(
        back_populates="photo", cascade="all, delete-orphan"
    )


class SkinProfile(Base):
    """One quantified metric for one facial region, in long/tidy format.

    Long format keeps cross-visit diffing and charting trivial — filter by
    (region, metric) and the values line up across visits automatically.
    """

    __tablename__ = "skin_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"))
    region: Mapped[str]
    metric: Mapped[str]
    value: Mapped[float]
    sub_score: Mapped[float]
    method_version: Mapped[str] = mapped_column(default=METHOD_VERSION)

    photo: Mapped[Photo] = relationship(back_populates="profiles")


class Recommendation(Base):
    """Treatment suggestions for a visit.

    generated_json is what the rules engine produced; edited_json is what the
    consultant approved. Keeping both visible shows the human-in-the-loop edit.
    """

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"))
    generated_json: Mapped[str] = mapped_column(Text, default="[]")
    edited_json: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(default="draft")  # draft | approved
    consultant_notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now, onupdate=datetime.now
    )

    visit: Mapped[Visit] = relationship(back_populates="recommendation")


_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine():
    global _engine
    if _engine is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), future=True)
    return _SessionFactory()


def init_db() -> None:
    """Create tables if missing. Idempotent."""
    Base.metadata.create_all(get_engine())
