"""MFU Health Agent — Streamlit entry point."""

import streamlit as st

st.set_page_config(
    page_title="МФУ Индекс Здоровья",
    page_icon="🏥",
    layout="wide",
)

st.title("МФУ Индекс Здоровья")
st.caption("ИИ-агент мониторинга парка многофункциональных устройств")

from state.session import (  # noqa: E402
    get_active_llm_endpoint,
    get_current_factor_store,
    get_current_report,
)

endpoint = get_active_llm_endpoint()
fs = get_current_factor_store()
report = get_current_report()

col1, col2, col3 = st.columns(3)
col1.metric("LLM Endpoint", endpoint)
col2.metric("FactorStore", "загружен" if fs else "—")
col3.metric("Отчёт", "готов" if report else "—")
