"""Longitudinal comparison and the customer-facing progress report.

Pain point 3: customers never knew what improved. Comparison uses identical
regions and metrics across visits, with a tolerance band so noise is not
mistaken for change. The progress report turns that into something the
customer can see — the retention hook.
"""
from __future__ import annotations

import os

import cv2
import pandas as pd
import streamlit as st

from db.models import Customer
from engine.suggest import positive_notes
from pipeline.metrics import METRIC_LABELS_ZH, METRICS, by_metric
from ui.common import (
    get_db,
    pick_customer,
    score_tag,
    to_rgb,
    visit_entries,
    visit_overall,
)

TOL = 4.0  # sub-score tolerance — smaller swings are reported as 持平


def _pick_customer(db) -> Customer | None:
    customer = pick_customer(db)
    if customer is None:
        st.info("尚無顧客資料。")
    return customer


def _history_df(customer) -> pd.DataFrame:
    rows, index = [], []
    for i, v in enumerate(customer.visits, 1):
        bm = by_metric(visit_entries(v))
        row = {"膚質總分": visit_overall(v)}
        for m in METRICS:
            if m in bm:
                row[METRIC_LABELS_ZH[m]] = bm[m]
        rows.append(row)
        index.append(f"第 {i} 次")
    return pd.DataFrame(rows, index=index)


# --- clinician-facing comparison ---

def render_compare() -> None:
    st.header("療程比較")
    db = get_db()
    customer = _pick_customer(db)
    if customer is None:
        return
    if len(customer.visits) < 2:
        st.info("此顧客就診未滿兩次，尚無法進行縱向比對。")
        return

    visits = customer.visits
    labels = [f"第 {i + 1} 次　{v.visit_date}" for i, v in enumerate(visits)]
    c1, c2 = st.columns(2)
    ia = c1.selectbox("對照前", range(len(visits)),
                      format_func=lambda i: labels[i], index=0)
    ib = c2.selectbox("對照後", range(len(visits)),
                      format_func=lambda i: labels[i], index=len(visits) - 1)
    va, vb = visits[ia], visits[ib]

    st.caption(
        f"拍照品質：對照前 {va.photos[0].quality_score:.0f}　/　"
        f"對照後 {vb.photos[0].quality_score:.0f}　"
        "— 品質分數相近時，比對結果最可靠。"
        if va.photos and vb.photos else "部分就診無拍照品質紀錄。"
    )

    _before_after(va, vb)
    _delta_table(va, vb)

    if len(visits) >= 3:
        st.subheader("膚質趨勢")
        st.line_chart(_history_df(customer))


def _before_after(va, vb) -> None:
    st.subheader("前後對照")
    c1, c2 = st.columns(2)
    for col, v, tag in ((c1, va, "對照前"), (c2, vb, "對照後")):
        path = v.photos[0].normalized_path if v.photos else ""
        if path and os.path.exists(path):
            img = cv2.imread(path)
            col.image(to_rgb(img), caption=f"{tag}　{v.visit_date}",
                      use_container_width=True)
        else:
            col.info(f"{tag}（{v.visit_date}）無照片紀錄")


def _delta_table(va, vb) -> None:
    st.subheader("指標變化")
    a = by_metric(visit_entries(va))
    b = by_metric(visit_entries(vb))
    rows = [_delta_row("膚質總分", visit_overall(va), visit_overall(vb))]
    for m in METRICS:
        if m in a and m in b:
            rows.append(_delta_row(METRIC_LABELS_ZH[m], a[m], b[m]))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"判定容差為 ±{TOL:.0f} 分，小於此範圍的變化視為持平（排除量測雜訊）。")


def _delta_row(label: str, before: float, after: float) -> dict:
    delta = round(after - before, 1)
    if delta >= TOL:
        verdict = "改善"
    elif delta <= -TOL:
        verdict = "退步"
    else:
        verdict = "持平"
    return {"指標": label, "對照前": before, "對照後": after,
            "變化": f"{delta:+.1f}", "判定": verdict}


# --- customer-facing progress report ---

def render_progress_report() -> None:
    st.header("膚質進步報告")
    db = get_db()
    customer = _pick_customer(db)
    if customer is None:
        return
    if not customer.visits:
        st.info("此顧客尚無就診紀錄。")
        return

    st.markdown(f"### {customer.name} 的膚質追蹤")
    first, last = customer.visits[0], customer.visits[-1]
    first_score, last_score = visit_overall(first), visit_overall(last)

    if len(customer.visits) >= 2:
        delta = last_score - first_score
        m1, m2, m3 = st.columns(3)
        m1.metric("初次膚質總分", f"{first_score:.0f}")
        m2.metric("最新膚質總分", f"{last_score:.0f}", f"{delta:+.1f}")
        m3.metric("追蹤次數", f"{len(customer.visits)} 次")
        if delta >= TOL:
            st.success(
                f"從 {first.visit_date} 到 {last.visit_date}，"
                f"您的膚質總分進步了 {delta:+.0f} 分，療程成效良好！"
            )
        elif delta <= -TOL:
            st.warning("近期膚質分數略有下降，建議與諮詢師討論調整療程方向。")
        else:
            st.info("膚質維持穩定，建議持續療程與居家保養。")

        notes = positive_notes(visit_entries(last), visit_entries(first))
        if notes:
            st.markdown("**具體進步項目**")
            for n in notes:
                st.markdown(f"- ✅ {n}")

        st.subheader("膚質趨勢")
        st.line_chart(_history_df(customer))
    else:
        st.info("這是第一次膚質紀錄，下次回訪後即可看到變化趨勢。")
        st.markdown(f"目前膚質總分 {score_tag(last_score)}",
                    unsafe_allow_html=True)

    st.subheader("最新各項膚質分數")
    bm = by_metric(visit_entries(last))
    cols = st.columns(len(bm))
    for col, (metric, score) in zip(cols, bm.items()):
        col.metric(METRIC_LABELS_ZH[metric], f"{score:.0f}")
