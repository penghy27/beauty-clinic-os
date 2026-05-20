"""Optional LLM layer that polishes Suggestion.reason text only.

Decision logic lives in `engine.suggest`; this module never changes which
treatments are recommended or the numbers behind them — it only rewrites
the template reason into more natural Mandarin. Any failure (missing
API key, missing package, timeout, quota, parse error, session cap)
returns None, and the caller falls back to the template text.

Inputs sent to the model are numeric only: metric label, region label,
sub-score, the template reason, the treatment name and the trend note.
No images, names or medical history leave the host.
"""
from __future__ import annotations

import json
import logging

import streamlit as st

_MODEL = "claude-haiku-4-5"
_MAX_CALLS_PER_SESSION = 5
_SESSION_KEY = "_humanize_calls"

_SYSTEM = (
    "你是台灣醫美診所的諮詢助手。"
    "請把每條機械化的膚質分析依據，改寫成自然、專業、不帶推銷感的繁體中文"
    "（台灣用語）。"
    "規則：決策與療程名稱不可改；分數、區域名、增減趨勢數字保留；"
    "每則不超過 60 字；不加表情符號。"
    "輸出一個純 JSON 物件，鍵為 rule_id、值為改寫後的依據文字，"
    "不要加 code fence、不要加任何解釋。"
)

log = logging.getLogger(__name__)


def refine(suggestions) -> dict[str, str] | None:
    """Return {rule_id: humanized_reason} or None on any failure."""
    if not suggestions:
        return None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return None
    if st.session_state.get(_SESSION_KEY, 0) >= _MAX_CALLS_PER_SESSION:
        return None
    return _refine_cached(_payload(suggestions))


def _payload(suggestions) -> str:
    items = [
        {
            "rule_id": s.rule_id,
            "metric": s.metric_label,
            "region": s.region_label,
            "sub_score": s.sub_score,
            "template_reason": s.reason,
            "treatment": s.treatment,
            "trend": s.trend,
        }
        for s in suggestions
    ]
    return json.dumps(items, ensure_ascii=False, sort_keys=True)


@st.cache_data(show_spinner="正在自動潤飾建議文字…")
def _refine_cached(payload: str) -> dict[str, str] | None:
    try:
        import anthropic

        api_key = st.secrets.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=800,
            system=_SYSTEM,
            messages=[{"role": "user", "content": payload}],
            timeout=20.0,
        )
        text = resp.content[0].text if resp.content else ""
        data = json.loads(_strip_code_fence(text))
        if not isinstance(data, dict):
            return None
        st.session_state[_SESSION_KEY] = (
            st.session_state.get(_SESSION_KEY, 0) + 1
        )
        return {k: v for k, v in data.items() if isinstance(v, str) and v}
    except Exception as exc:
        log.warning("humanize refine failed: %s", exc)
        return None


def _strip_code_fence(text: str) -> str:
    """Tolerate ```json fences if the model slips."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    inner = text.strip("`").strip()
    if inner.lower().startswith("json"):
        inner = inner[4:].strip()
    return inner
