"""Longitudinal comparison and the customer-facing progress report.

Pain point 3: customers never knew what improved. Comparison uses identical
regions and metrics across visits, with a tolerance band so noise is not
mistaken for change. The progress report turns that into something the
customer can see — the retention hook.
"""
from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from db.models import Customer
from engine.suggest import positive_notes
from pipeline.metrics import METRIC_LABELS_ZH, METRICS, NOISE_BAND, by_metric
from ui.common import (
    get_db,
    photo_file,
    pick_customer,
    score_tag,
    to_rgb,
    visit_entries,
    visit_method_version,
    visit_overall,
)

# The Skin Health Index averages every (region, metric) sub-score, so its
# noise band is the average of the per-metric bands.
_OVERALL_BAND = sum(NOISE_BAND.values()) / len(NOISE_BAND)

_BAND_CAPTION = (
    "容差帶依重測數據按指標訂定（"
    + "、".join(f"{METRIC_LABELS_ZH[m]} ±{NOISE_BAND[m]:.0f} 分" for m in METRICS)
    + f"；總分 ±{_OVERALL_BAND:.0f} 分），小於容差的變化視為量測雜訊。"
)


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

    comparable = visit_method_version(va) == visit_method_version(vb)
    if not comparable:
        st.warning(
            "兩次就診的量測方法版本不同"
            f"（{visit_method_version(va) or '無'} / "
            f"{visit_method_version(vb) or '無'}），"
            "分數尺度不一致，以下數字僅供參考，不做改善/退步判定。"
        )

    _before_after(va, vb)
    _delta_table(va, vb, comparable)

    if len(visits) >= 3:
        st.subheader("膚質趨勢")
        st.line_chart(_history_df(customer))


def _before_after(va, vb) -> None:
    st.subheader("前後對照")
    c1, c2 = st.columns(2)
    for col, v, tag in ((c1, va, "對照前"), (c2, vb, "對照後")):
        path = photo_file(v.photos[0].normalized_path if v.photos else "")
        if path is not None:
            img = cv2.imread(str(path))
            col.image(to_rgb(img), caption=f"{tag}　{v.visit_date}",
                      use_container_width=True)
        else:
            col.info(f"{tag}（{v.visit_date}）無照片紀錄")


def _delta_table(va, vb, comparable: bool = True) -> None:
    st.subheader("指標變化")
    a = by_metric(visit_entries(va))
    b = by_metric(visit_entries(vb))
    rows = [_delta_row("膚質總分", visit_overall(va), visit_overall(vb),
                       _OVERALL_BAND, comparable)]
    for m in METRICS:
        if m in a and m in b:
            rows.append(_delta_row(METRIC_LABELS_ZH[m], a[m], b[m],
                                   NOISE_BAND[m], comparable))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(_BAND_CAPTION)


def _delta_row(label: str, before: float, after: float,
               band: float, comparable: bool = True) -> dict:
    delta = round(after - before, 1)
    if not comparable:
        verdict = "無法判定（版本不同）"
    elif delta >= band:
        verdict = "改善"
    elif delta <= -band:
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
        comparable = (visit_method_version(first)
                      == visit_method_version(last))
        if not comparable:
            st.info(
                "追蹤期間量測方法已更新，初次與最新分數尺度不同，"
                "成效判定將以新方法的後續紀錄為準。"
            )
        elif delta >= _OVERALL_BAND:
            st.success(
                f"從 {first.visit_date} 到 {last.visit_date}，"
                f"您的膚質總分進步了 {delta:+.0f} 分，療程成效良好！"
            )
        elif delta <= -_OVERALL_BAND:
            st.warning("近期膚質分數略有下降，建議與諮詢師討論調整療程方向。")
        else:
            st.info("膚質維持穩定，建議持續療程與居家保養。")

        if comparable:
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
    for col, (metric, score) in zip(cols, bm.items(), strict=True):
        col.metric(METRIC_LABELS_ZH[metric], f"{score:.0f}")
