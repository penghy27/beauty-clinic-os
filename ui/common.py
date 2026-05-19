"""Shared UI helpers — image conversion, scoring, navigation, DB access."""
from __future__ import annotations

from datetime import date

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from db.models import PHOTO_DIR, Customer, get_session
from pipeline.metrics import ProfileEntry, overall_score


def get_db():
    """A fresh SQLAlchemy session for the current Streamlit run."""
    return get_session()


def nav(page: str, customer_id: int | None = None) -> None:
    """Navigation callback — set the active page (and optional customer)."""
    st.session_state.page = page
    if customer_id is not None:
        st.session_state.customer_id = customer_id


# --- image helpers ---

def read_upload(uploaded) -> np.ndarray:
    """Streamlit UploadedFile -> BGR ndarray."""
    image = Image.open(uploaded).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_photo(image_bgr: np.ndarray, filename: str) -> str:
    """Persist an image under data/photos and return its path."""
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = PHOTO_DIR / filename
    cv2.imwrite(str(path), image_bgr)
    return str(path)


# --- scoring helpers ---

def score_hex(score: float) -> str:
    if score >= 70:
        return "#1a9850"
    if score >= 50:
        return "#e8a317"
    return "#d73027"


def score_tag(score: float, suffix: str = "") -> str:
    """An inline coloured score badge (HTML)."""
    return (
        f"<span style='background:{score_hex(score)};color:#fff;"
        f"padding:2px 10px;border-radius:11px;font-weight:600'>"
        f"{score:.0f}{suffix}</span>"
    )


# --- visit / profile helpers ---

def visit_entries(visit) -> list[ProfileEntry]:
    """Skin Profile of a visit as ProfileEntry list (first photo)."""
    if not visit.photos:
        return []
    return [
        ProfileEntry(p.region, p.metric, p.value, p.sub_score)
        for p in visit.photos[0].profiles
    ]


def visit_overall(visit) -> float:
    return overall_score(visit_entries(visit))


def latest_visit(customer):
    return customer.visits[-1] if customer.visits else None


def is_overdue(customer) -> bool:
    """True when the customer's last booked revisit date has passed."""
    lv = latest_visit(customer)
    return bool(lv and lv.next_revisit_date and lv.next_revisit_date < date.today())


def customer_age(customer) -> str:
    if not customer.birth_year:
        return "—"
    return str(date.today().year - customer.birth_year)


def pick_customer(db, label: str = "顧客"):
    """Customer selectbox that returns a session-bound Customer.

    Selecting by id (never by ORM object) avoids DetachedInstanceError when
    Streamlit restores widget state across reruns.
    """
    customers = db.query(Customer).order_by(Customer.name).all()
    if not customers:
        return None
    names = {c.id: c.name for c in customers}
    ids = list(names)
    preset = st.session_state.get("customer_id")
    index = ids.index(preset) if preset in ids else 0
    chosen = st.selectbox(label, ids, index=index,
                          format_func=lambda i: names[i])
    return db.get(Customer, chosen)
