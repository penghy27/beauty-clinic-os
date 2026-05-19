"""CRM pages — customer list and customer detail.

Replaces the clinic's paper records: who is due for a revisit, which package
they bought, how many sessions are left, whether they have paid.
"""
from __future__ import annotations

from datetime import date, datetime

import streamlit as st

from db.models import Customer
from ui.common import (
    customer_age,
    get_db,
    is_overdue,
    latest_visit,
    nav,
    score_tag,
    visit_overall,
)


def _package_line(customer) -> str:
    if not customer.packages:
        return "尚無療程套組"
    p = customer.packages[0]
    paid = "已付款" if p.is_paid else "⚠ 未付款"
    return f"{p.name}　{p.sessions_used}/{p.total_sessions} 堂（剩 {p.sessions_left}）　{paid}"


def render_customer_list() -> None:
    st.header("顧客管理")
    db = get_db()
    customers = db.query(Customer).order_by(Customer.name).all()
    overdue = [c for c in customers if is_overdue(c)]

    if overdue:
        st.warning(f"有 {len(overdue)} 位顧客回訪逾期，建議今日聯繫安排回診。")

    col_a, col_b = st.columns([3, 1])
    col_a.caption(f"共 {len(customers)} 位顧客")
    only_overdue = col_b.toggle("只看回訪逾期")

    with st.expander("新增顧客"):
        _new_customer_form(db)

    shown = overdue if only_overdue else customers
    if not shown:
        st.info("沒有符合條件的顧客。")
        return

    for c in shown:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 4, 1.4])
            with c1:
                st.markdown(f"**{c.name}**　{c.sex}・{customer_age(c)} 歲")
                st.caption(c.phone or "—")
            with c2:
                st.markdown(_package_line(c))
                lv = latest_visit(c)
                if lv and lv.next_revisit_date:
                    if is_overdue(c):
                        st.markdown(
                            f":red[下次回訪 {lv.next_revisit_date}　⚠ 已逾期]"
                        )
                    else:
                        st.caption(f"下次回訪 {lv.next_revisit_date}")
                else:
                    st.caption("尚未安排回訪")
            with c3:
                st.button("查看", key=f"view_{c.id}", use_container_width=True,
                          on_click=nav, args=("customer_detail", c.id))


def _new_customer_form(db) -> None:
    with st.form("new_customer", clear_on_submit=True):
        name = st.text_input("姓名")
        c1, c2, c3 = st.columns(3)
        sex = c1.selectbox("性別", ["女", "男", "其他"])
        birth_year = c2.number_input("出生年", 1940, 2015, 1990)
        phone = c3.text_input("電話")
        skin_note = st.text_area("膚況備註", placeholder="例：混合性肌膚，主訴兩頰泛紅")
        consent = st.checkbox("顧客已簽署拍照與資料使用同意書")
        submitted = st.form_submit_button("建立顧客", type="primary")
    if submitted:
        if not name.strip():
            st.error("請輸入姓名。")
            return
        db.add(Customer(
            name=name.strip(), sex=sex, birth_year=int(birth_year),
            phone=phone.strip(), skin_note=skin_note.strip(),
            consent_signed_at=datetime.now() if consent else None,
        ))
        db.commit()
        st.success(f"已建立顧客「{name.strip()}」。")
        st.rerun()


def render_customer_detail() -> None:
    customer_id = st.session_state.get("customer_id")
    db = get_db()
    customer = db.get(Customer, customer_id) if customer_id else None
    if customer is None:
        st.info("找不到顧客資料。")
        st.button("← 回顧客列表", on_click=nav, args=("customers",))
        return

    st.button("← 回顧客列表", on_click=nav, args=("customers",))
    st.header(f"{customer.name}")
    st.caption(
        f"{customer.sex}・{customer_age(customer)} 歲・{customer.phone or '無電話'}"
    )
    if customer.skin_note:
        st.markdown(f"**膚況備註**：{customer.skin_note}")
    if customer.consent_signed_at:
        st.caption(f"已簽署同意書：{customer.consent_signed_at:%Y-%m-%d}")
    else:
        st.caption(":red[尚未簽署拍照同意書]")

    a, b, c = st.columns(3)
    a.button("＋ 新增諮詢", use_container_width=True, type="primary",
             on_click=nav, args=("new_consultation", customer.id))
    b.button("療程比較", use_container_width=True,
             on_click=nav, args=("compare", customer.id))
    c.button("膚質進步報告", use_container_width=True,
             on_click=nav, args=("progress", customer.id))

    st.subheader("療程套組")
    if not customer.packages:
        st.caption("尚無療程套組。")
    for p in customer.packages:
        with st.container(border=True):
            st.markdown(f"**{p.name}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("已使用堂數", f"{p.sessions_used} / {p.total_sessions}")
            m2.metric("剩餘堂數", p.sessions_left)
            m3.metric("付款狀態", "已付款" if p.is_paid else "未付款")
            st.progress(min(p.sessions_used / max(p.total_sessions, 1), 1.0))

    st.subheader("就診時間軸")
    if not customer.visits:
        st.caption("尚無就診紀錄。")
    for i, v in enumerate(reversed(customer.visits), 1):
        idx = len(customer.visits) - i + 1
        with st.container(border=True):
            head, badge = st.columns([4, 1])
            head.markdown(
                f"**第 {idx} 次就診**　{v.visit_date}　諮詢師：{v.consultant or '—'}"
            )
            badge.markdown(
                f"膚質 {score_tag(visit_overall(v))}", unsafe_allow_html=True
            )
            if v.notes:
                st.caption(v.notes)
            if v.next_revisit_date:
                overdue = v.next_revisit_date < date.today()
                tag = "　⚠ 已逾期" if overdue else ""
                st.caption(f"下次回訪：{v.next_revisit_date}{tag}")
