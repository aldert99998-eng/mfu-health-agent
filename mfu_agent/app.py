"""MFU Health Agent — Streamlit entry point.

Uses st.navigation to build the sidebar explicitly so the entry-file
doesn't appear as a separate 'app' item.
"""

# Load .env (optional dependency; if missing we degrade gracefully).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover — python-dotenv is a declared dep
    pass

import streamlit as st

st.set_page_config(
    page_title="МФУ Индекс Здоровья",
    page_icon="🏥",
    layout="wide",
)

from state.session import get_current_report  # noqa: E402

_has_report = get_current_report() is not None

pages = [
    st.Page(
        "pages/1_Загрузка_данных.py",
        title="Загрузка данных",
        icon="📤",
        default=not _has_report,
    ),
    st.Page(
        "pages/2_Dashboard.py",
        title="Dashboard",
        icon="📊",
        default=_has_report,
    ),
    st.Page("pages/3_Weights.py", title="Веса", icon="⚖️"),
    st.Page("pages/4_LLM_Chat.py", title="Чат с агентом", icon="💬"),
    st.Page("pages/5_Error_Codes.py", title="Справочник ошибок", icon="🔧"),
    st.Page("pages/5_RAG_Admin.py", title="RAG админка", icon="📚"),
]

pg = st.navigation(pages)
pg.run()
