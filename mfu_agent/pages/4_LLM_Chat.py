"""Страница 3 — Чат с ИИ-агентом."""

from __future__ import annotations

import logging

import streamlit as st

from state.session import (
    append_chat_message,
    claim_bg_results_if_any,
    get_active_llm_endpoint,
    get_active_weights_profile,
    get_chat_history,
    get_current_factor_store,
    get_current_report,
    get_mass_error_analyses,
)
from state.singletons import get_agent, get_hybrid_searcher, get_llm_client
from ui.endpoint_selector import render_endpoint_selector

logger = logging.getLogger(__name__)

st.header("Чат с агентом")

claim_bg_results_if_any()

# ── Block 1: Endpoint selector ───────────────────────────────────────────────

_TRACES_KEY = "mfu_chat_traces"

if _TRACES_KEY not in st.session_state:
    st.session_state[_TRACES_KEY] = {}

render_endpoint_selector(location="sidebar", key_suffix="chat")

with st.sidebar:
    if st.button("🧹 Очистить чат", help="Сбросить историю диалога"):
        st.session_state.pop("mfu_chat_history", None)
        st.session_state[_TRACES_KEY] = {}
        st.rerun()

# ── Block 2: Guardrail — отчёт должен быть сформирован ─────────────────────

_report = get_current_report()
_fs = get_current_factor_store()

if _report is None or _fs is None:
    st.warning(
        "Отчёт ещё не сформирован. Загрузите данные на странице "
        "**«Загрузка данных»** и дождитесь завершения анализа — "
        "после этого чат сможет отвечать по вашему парку."
    )
    st.stop()

# ── Block 3: Chat history display ────────────────────────────────────────────

history = get_chat_history()
traces: dict[int, str] = st.session_state[_TRACES_KEY]

for idx, msg in enumerate(history):
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        if msg.role == "assistant" and idx in traces:
            with st.expander("Трассировка", expanded=False):
                st.code(traces[idx], language="json")

# ── Block 4: Chat input ─────────────────────────────────────────────────────

user_input = st.chat_input("Задайте вопрос агенту…")

if user_input:
    append_chat_message("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Агент думает…"):
            from agent.tools.impl import ToolDependencies, register_all_tools
            from agent.tools.registry import ToolRegistry
            from data_io.models import ChatContext

            endpoint_name = get_active_llm_endpoint()

            try:
                _searcher = get_hybrid_searcher()
            except Exception as exc:
                logger.warning("RAG searcher unavailable: %s", exc)
                _searcher = None

            deps = ToolDependencies(
                factor_store=_fs,
                weights=get_active_weights_profile(),
                searcher=_searcher,
                llm_client=get_llm_client(endpoint_name),
                current_report=_report,
                mass_error_analyses=get_mass_error_analyses(),
            )
            registry = ToolRegistry()
            register_all_tools(registry, deps)

            agent = get_agent(endpoint_name)
            agent.set_tools(registry)

            chat_history_dicts = [
                {"role": m.role, "content": m.content} for m in get_chat_history()
            ]

            context = ChatContext(
                current_report=_report,
                conversation_history=chat_history_dicts,
                factor_store=_fs,
            )

            answer, trace = agent.run_chat(user_input, context)

        st.markdown(answer)

        rag_hits = getattr(trace, "rag_hits", None) or []
        if rag_hits:
            with st.expander(f"🔎 Источники ({len(rag_hits)})", expanded=False):
                for hit in rag_hits:
                    src = hit.get("source", "")
                    doc_id = hit.get("document_id", "")
                    score = float(hit.get("score", 0.0))
                    text = str(hit.get("text", ""))[:400]
                    st.markdown(f"**[{src}: {doc_id}]** (score={score:.2f})")
                    st.caption(text)

        trace_json = trace.to_json()
        with st.expander("Трассировка", expanded=False):
            st.code(trace_json, language="json")

    assistant_idx = len(get_chat_history())
    append_chat_message("assistant", answer)
    st.session_state[_TRACES_KEY][assistant_idx] = trace_json
    st.rerun()
