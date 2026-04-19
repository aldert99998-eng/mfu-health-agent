"""Страница 3 — Чат с ИИ-агентом."""

from __future__ import annotations

import streamlit as st
import yaml

from config.loader import CONFIGS_DIR
from state.session import (
    append_chat_message,
    get_active_llm_endpoint,
    get_chat_history,
    get_current_factor_store,
    get_current_report,
    set_active_llm_endpoint,
)
from state.singletons import get_agent, invalidate_llm_singletons

st.header("Чат с агентом")

# ── Block 1: Endpoint selector ───────────────────────────────────────────────

_TRACES_KEY = "mfu_chat_traces"

if _TRACES_KEY not in st.session_state:
    st.session_state[_TRACES_KEY] = {}


def _list_endpoints() -> list[str]:
    path = CONFIGS_DIR / "llm_endpoints.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        endpoints = data.get("endpoints", data)
        if isinstance(endpoints, dict):
            return list(endpoints.keys())
    return ["default"]


endpoints = _list_endpoints()
current_endpoint = get_active_llm_endpoint()

with st.sidebar:
    st.subheader("LLM Endpoint")
    selected_endpoint = st.selectbox(
        "Endpoint",
        endpoints,
        index=endpoints.index(current_endpoint) if current_endpoint in endpoints else 0,
        key="llm_endpoint_select",
    )
    if selected_endpoint != current_endpoint:
        set_active_llm_endpoint(selected_endpoint)
        invalidate_llm_singletons()
        st.rerun()

# ── Block 2: Chat history display ────────────────────────────────────────────

history = get_chat_history()
traces: dict[int, str] = st.session_state[_TRACES_KEY]

for idx, msg in enumerate(history):
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        if msg.role == "assistant" and idx in traces:
            with st.expander("Трассировка", expanded=False):
                st.code(traces[idx], language="json")

# ── Block 3: Chat input ─────────────────────────────────────────────────────

user_input = st.chat_input("Задайте вопрос агенту…")

if user_input:
    append_chat_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Агент думает…"):
            from data_io.models import ChatContext

            report = get_current_report()
            fs = get_current_factor_store()

            chat_history_dicts = [
                {"role": m.role, "content": m.content} for m in get_chat_history()
            ]

            context = ChatContext(
                current_report=report,
                conversation_history=chat_history_dicts,
                factor_store=fs,
            )

            agent = get_agent(get_active_llm_endpoint())
            answer, trace = agent.run_chat(user_input, context)

        st.markdown(answer)

        trace_json = trace.to_json()
        with st.expander("Трассировка", expanded=False):
            st.code(trace_json, language="json")

    assistant_idx = len(get_chat_history())
    append_chat_message("assistant", answer)
    st.session_state[_TRACES_KEY][assistant_idx] = trace_json
    st.rerun()
