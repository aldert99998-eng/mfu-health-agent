"""Session state helpers — Phase 7.1.

Typed wrappers over ``st.session_state`` for all shared
mutable state in the Streamlit app.  Keys are string constants
defined at the top of this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from data_io.factor_store import FactorStore
    from data_io.models import HealthResult, Report, WeightsProfile

# ── Session-state keys ───────────────────────────────────────────────────────

_KEY_FACTOR_STORE = "mfu_factor_store"
_KEY_HEALTH_RESULTS = "mfu_health_results"
_KEY_REPORT = "mfu_report"
_KEY_WEIGHTS_PROFILE = "mfu_weights_profile"
_KEY_LLM_ENDPOINT = "mfu_llm_endpoint"
_KEY_CHAT_HISTORY = "mfu_chat_history"


# ── Chat message ─────────────────────────────────────────────────────────────


@dataclass
class Message:
    """Single chat message stored in session history."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# ── FactorStore ──────────────────────────────────────────────────────────────


def get_current_factor_store() -> FactorStore | None:
    """Return the active FactorStore or ``None`` if not loaded."""
    return st.session_state.get(_KEY_FACTOR_STORE)


def set_current_factor_store(fs: FactorStore) -> None:
    """Set the active FactorStore (typically after ingestion)."""
    st.session_state[_KEY_FACTOR_STORE] = fs


# ── HealthResults ────────────────────────────────────────────────────────────


def get_current_health_results() -> list[HealthResult]:
    """Return the list of per-device health results (may be empty)."""
    return st.session_state.get(_KEY_HEALTH_RESULTS, [])  # type: ignore[no-any-return]


def set_current_health_results(results: list[HealthResult]) -> None:
    """Store calculated health results."""
    st.session_state[_KEY_HEALTH_RESULTS] = list(results)


# ── Report ───────────────────────────────────────────────────────────────────


def get_current_report() -> Report | None:
    """Return the latest generated Report or ``None``."""
    return st.session_state.get(_KEY_REPORT)


def set_current_report(report: Report) -> None:
    """Store the generated Report."""
    st.session_state[_KEY_REPORT] = report


# ── WeightsProfile ───────────────────────────────────────────────────────────


def get_active_weights_profile() -> WeightsProfile | None:
    """Return the active weights profile or ``None`` if not set."""
    return st.session_state.get(_KEY_WEIGHTS_PROFILE)


def set_active_weights_profile(profile: WeightsProfile) -> None:
    """Set the active weights profile."""
    st.session_state[_KEY_WEIGHTS_PROFILE] = profile


# ── LLM Endpoint ─────────────────────────────────────────────────────────────

_DEFAULT_ENDPOINT = "default"


def get_active_llm_endpoint() -> str:
    """Return the name of the active LLM endpoint."""
    return st.session_state.get(_KEY_LLM_ENDPOINT, _DEFAULT_ENDPOINT)  # type: ignore[no-any-return]


def set_active_llm_endpoint(name: str) -> None:
    """Switch the active LLM endpoint.

    Call ``singletons.invalidate_llm_singletons()`` after this
    to rebuild LLMClient and Agent with the new endpoint.
    """
    st.session_state[_KEY_LLM_ENDPOINT] = name


# ── Chat History ─────────────────────────────────────────────────────────────


def get_chat_history() -> list[Message]:
    """Return the current chat message history."""
    return st.session_state.get(_KEY_CHAT_HISTORY, [])  # type: ignore[no-any-return]


def append_chat_message(role: str, content: str) -> None:
    """Append a message to the chat history."""
    if _KEY_CHAT_HISTORY not in st.session_state:
        st.session_state[_KEY_CHAT_HISTORY] = []
    st.session_state[_KEY_CHAT_HISTORY].append(Message(role=role, content=content))


# ── Reset ────────────────────────────────────────────────────────────────────


def clear_all() -> None:
    """Remove all MFU-related keys from session state."""
    for key in (
        _KEY_FACTOR_STORE,
        _KEY_HEALTH_RESULTS,
        _KEY_REPORT,
        _KEY_WEIGHTS_PROFILE,
        _KEY_LLM_ENDPOINT,
        _KEY_CHAT_HISTORY,
    ):
        st.session_state.pop(key, None)
