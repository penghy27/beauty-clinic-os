"""Beauty Clinic OS — AI-native consultation + CRM for aesthetic clinics.

A Streamlit prototype built for the AI Fund "Beauty Clinic OS" challenge.
Anchored on three pain points heard from clinic owners:
  1. inconsistent, sales-driven first-line consultations
  2. paper-based CRM (revisits, packages, payments)
  3. customers with no record of what actually improved

Run: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from db.seed import seed_if_empty
from ui import compare, consultation, customers
from ui.common import nav

st.set_page_config(page_title="Beauty Clinic OS", page_icon="🪞",
                   layout="wide")

@st.cache_resource
def _bootstrap() -> bool:
    """Create tables and seed demo data once per server process.

    Running this inside cache_resource makes it execute exactly once,
    which avoids a cold-start race where concurrent script runs each
    inspect the empty database and then both issue CREATE TABLE.
    """
    seed_if_empty()
    return True


_bootstrap()
st.session_state.setdefault("page", "customers")

with st.sidebar:
    st.title("🪞 Beauty Clinic OS")
    st.caption("醫美診所 AI 諮詢 + CRM")
    st.divider()
    st.button("顧客管理", use_container_width=True,
              on_click=nav, args=("customers",))
    st.button("新增諮詢", use_container_width=True,
              on_click=nav, args=("new_consultation",))
    st.button("療程比較", use_container_width=True,
              on_click=nav, args=("compare",))
    st.button("膚質進步報告", use_container_width=True,
              on_click=nav, args=("progress",))
    st.divider()
    st.caption("標準化諮詢 → 量化膚質 → 療程建議 → CRM → 回訪追蹤 → 進步報告")

_PAGES = {
    "customers": customers.render_customer_list,
    "customer_detail": customers.render_customer_detail,
    "new_consultation": consultation.render_new_consultation,
    "compare": compare.render_compare,
    "progress": compare.render_progress_report,
}
_PAGES.get(st.session_state.page, customers.render_customer_list)()
