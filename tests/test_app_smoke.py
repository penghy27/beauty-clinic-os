"""Streamlit AppTest smoke tests: every page renders without an exception."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["customers", "customer_detail", "new_consultation",
         "compare", "progress"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file("app.py", default_timeout=30)
    at.session_state["page"] = page
    if page == "customer_detail":
        at.session_state["customer_id"] = 1
    at.run()
    assert not at.exception
