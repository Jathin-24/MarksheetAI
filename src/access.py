"""Shared access-code gate (used by EVERY page, so no page can be opened by URL without the code)."""
from __future__ import annotations

import hmac
import os

import streamlit as st


def require_access_code() -> None:
    """If APP_PASSWORD is set (env var or Streamlit secret) nothing renders until it is entered."""
    pw = os.environ.get("APP_PASSWORD")
    if not pw:
        try:
            pw = st.secrets.get("APP_PASSWORD")
        except Exception:  # no secrets file (local run)
            pw = None
    if not pw or st.session_state.get("authed"):
        return
    st.title("ðŸ“ Marksheet Evaluator")
    entered = st.text_input("Access code", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(pw).encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.error("Wrong access code.")
    st.stop()
