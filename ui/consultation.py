"""New-consultation flow and consultation view.

This is the answer to pain point 1: a standardised, scientific first-line
consultation. The consultant and customer look at the same screen — every
treatment suggestion is tied to a measured number, so it never feels like an
upsell, and a junior consultant runs the same flow as a senior one.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from db.models import Photo, Recommendation, SkinProfile, Visit
from engine import suggest as engine
from pipeline.metrics import METRIC_LABELS_ZH, by_metric
from pipeline.process import process_photo
from pipeline.regions import REGION_LABELS_ZH
from pipeline import viz
from ui.common import (
    get_db,
    nav,
    pick_customer,
    read_upload,
    save_photo,
    score_hex,
    score_tag,
    to_rgb,
    visit_entries,
)


def render_new_consultation() -> None:
    st.header("新增諮詢")
    db = get_db()
    customer = pick_customer(db)
    if customer is None:
        st.info("請先到「顧客管理」建立顧客。")
        return

    uploaded = st.file_uploader("上傳顧客正面臉部照片", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("上傳一張正面臉部照片即可開始標準化膚質諮詢。")
        return

    result = _run_pipeline(uploaded)
    if not result.ok:
        st.error(result.message)
        return

    _render_quality_gate(result)
    _render_skin_analysis(result)
    _render_consultation_plan(db, customer, result)


def _run_pipeline(uploaded):
    """Run the imaging pipeline once per uploaded file (cached in session)."""
    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("pipe_key") != file_key:
        image_bgr = read_upload(uploaded)
        with st.spinner("影像分析中…"):
            result = process_photo(image_bgr)
        st.session_state.pipe_key = file_key
        st.session_state.pipe_result = result
        st.session_state.pipe_image = image_bgr
    return st.session_state.pipe_result


def _render_quality_gate(result) -> None:
    st.subheader("① 拍照品質")
    q = result.quality
    head, body = st.columns([1, 3])
    with head:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:13px;color:#666'>Capture Quality</div>"
            f"<div style='font-size:46px;font-weight:700;color:{score_hex(q.score)}'>"
            f"{q.score:.0f}</div></div>",
            unsafe_allow_html=True,
        )
    with body:
        for c in q.checks:
            st.markdown(f"{c.icon} **{c.name}** — {c.detail}")
    if not q.passed:
        st.warning("照片品質未達標準，建議依上述提示重拍，以確保跨次比對的一致性。")
    else:
        st.success("照片品質良好，可作為標準化的膚質紀錄。")


def _render_skin_analysis(result) -> None:
    st.subheader("② 膚質分析")
    raw = st.session_state.pipe_image
    overlay = viz.draw_regions(result.normalized_bgr, result.regions)

    c1, c2, c3 = st.columns(3)
    c1.image(to_rgb(raw), caption="原始照片", use_container_width=True)
    c2.image(to_rgb(result.normalized_bgr),
             caption=f"色彩校正後（{result.wb_info.get('original_cast', '')}）",
             use_container_width=True)
    c3.image(to_rgb(overlay), caption="量測區域", use_container_width=True)
    st.caption(
        "色彩校正採 gray-world white balance，移除光線/裝置色偏；"
        "量測區域錨定臉部 landmark，每次就診取樣一致。"
    )

    st.markdown(f"**膚質總分** {score_tag(result.overall)}　"
                f"（六區域 × 四指標的平均，100 為最佳）",
                unsafe_allow_html=True)
    metric_scores = by_metric(result.profile)
    cols = st.columns(len(metric_scores))
    for col, (metric, score) in zip(cols, metric_scores.items()):
        col.metric(METRIC_LABELS_ZH[metric], f"{score:.0f}")

    with st.expander("各區域明細"):
        st.dataframe(_profile_table(result.profile), use_container_width=True)


def _profile_table(profile) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for e in profile:
        rows.setdefault(REGION_LABELS_ZH.get(e.region, e.region), {})[
            METRIC_LABELS_ZH[e.metric]
        ] = e.sub_score
    return pd.DataFrame(rows).T


def _render_consultation_plan(db, customer, result) -> None:
    st.subheader("③ 療程建議")
    previous = _previous_entries(customer)
    suggestions = engine.suggest(result.profile, previous)

    if previous:
        st.caption("已對照上次就診數據，建議含療程成效追蹤。")
    if not suggestions:
        st.success("目前各項膚質指標良好，無須額外療程，建議維持保養與定期追蹤。")

    for s in suggestions:
        with st.container(border=True):
            st.markdown(f"**{s.treatment}**　:gray[優先度 {s.priority}]")
            st.markdown(s.detail)
            st.markdown(f"📊 **依據**：{s.reason}")
            if s.trend:
                st.markdown(f"🔁 **療程追蹤**：{s.trend}")
            st.caption(f"注意事項：{s.caution}")

    st.markdown("**療程計畫（諮詢師可調整）**")
    st.caption("可直接修改、刪除或新增項目；下方表格即為存入顧客紀錄的版本。")
    editor_key = f"plan_{st.session_state.get('pipe_key')}_{customer.id}"
    edited = st.data_editor(
        _plan_dataframe(suggestions),
        num_rows="dynamic",
        use_container_width=True,
        key=editor_key,
        column_config={
            "採用": st.column_config.CheckboxColumn(default=True),
        },
    )

    note = st.text_area("諮詢師備註", key=f"note_{editor_key}")
    c1, c2 = st.columns(2)
    consultant = c1.text_input("諮詢師", value="諮詢師 A")
    next_revisit = c2.date_input("建議下次回訪日",
                                 value=date.today() + timedelta(days=35))

    package = None
    if customer.packages:
        package = st.selectbox(
            "本次使用療程套組", customer.packages,
            format_func=lambda p: f"{p.name}（剩 {p.sessions_left} 堂）",
        )

    if st.button("儲存此次諮詢", type="primary"):
        _save(db, customer, result, suggestions, edited, note,
              consultant, next_revisit, package)


def _plan_dataframe(suggestions) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "採用": True,
            "療程": s.treatment,
            "說明": s.detail,
            "依據（數據）": s.reason,
            "療程追蹤": s.trend,
            "注意事項": s.caution,
        }
        for s in suggestions
    ])


def _previous_entries(customer):
    if not customer.visits:
        return None
    return visit_entries(customer.visits[-1]) or None


def _save(db, customer, result, suggestions, edited_df, note,
          consultant, next_revisit, package) -> None:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    raw_path = save_photo(st.session_state.pipe_image,
                          f"c{customer.id}_{stamp}_raw.jpg")
    norm_path = save_photo(result.normalized_bgr,
                           f"c{customer.id}_{stamp}_norm.jpg")

    visit = Visit(
        customer_id=customer.id, visit_date=date.today(),
        consultant=consultant, notes=note,
        next_revisit_date=next_revisit,
        package_id=package.id if package else None,
    )
    photo = Photo(
        view="front", raw_path=raw_path, normalized_path=norm_path,
        quality_score=result.quality.score,
        quality_json=json.dumps(result.quality.to_dict(), ensure_ascii=False),
        wb_method=result.wb_info.get("method", ""),
    )
    visit.photos.append(photo)
    for e in result.profile:
        photo.profiles.append(SkinProfile(
            region=e.region, metric=e.metric,
            value=e.value, sub_score=e.sub_score,
        ))

    adopted = [r for r in edited_df.to_dict("records") if r.get("採用", True)]
    visit.recommendation = Recommendation(
        generated_json=engine.to_json(suggestions),
        edited_json=json.dumps(adopted, ensure_ascii=False),
        status="approved",
        consultant_notes=note,
    )
    customer.visits.append(visit)
    if package:
        package.sessions_used += 1
    db.add(visit)
    db.commit()

    for key in ("pipe_key", "pipe_result", "pipe_image"):
        st.session_state.pop(key, None)
    st.success(f"已儲存「{customer.name}」本次諮詢紀錄。")
    st.button("查看顧客紀錄", on_click=nav, args=("customer_detail", customer.id))
