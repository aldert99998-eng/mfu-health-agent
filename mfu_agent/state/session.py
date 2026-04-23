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
    from data_io.models import (
        DeepDeviceAnalysis,
        HealthResult,
        MassErrorAnalysis,
        Report,
        WeightsProfile,
    )

# ── Session-state keys ───────────────────────────────────────────────────────

_KEY_FACTOR_STORE = "mfu_factor_store"
_KEY_HEALTH_RESULTS = "mfu_health_results"
_KEY_BASELINE_RESULTS = "mfu_baseline_health_results"
_KEY_REPORT = "mfu_report"
_KEY_WEIGHTS_PROFILE = "mfu_weights_profile"
_KEY_LLM_ENDPOINT = "mfu_llm_endpoint"
_KEY_CHAT_HISTORY = "mfu_chat_history"
_KEY_RAW_FACTORS = "mfu_raw_factors"
_KEY_DEEP_DEVICE_ANALYSES = "mfu_deep_device_analyses"
_KEY_MASS_ERROR_ANALYSES = "mfu_mass_error_analyses"
_KEY_CLAIMED_BG_RUN = "mfu_claimed_bg_run_id"


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


def get_baseline_health_results() -> list[HealthResult]:
    """Return the immutable snapshot captured after the initial batch analysis.

    Used by the Weights page to show a stable "before" against which the
    recalculated "after" results are compared.
    """
    return st.session_state.get(_KEY_BASELINE_RESULTS, [])  # type: ignore[no-any-return]


def set_baseline_health_results(results: list[HealthResult]) -> None:
    """Store the baseline snapshot (typically right after initial batch)."""
    st.session_state[_KEY_BASELINE_RESULTS] = list(results)


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


def get_raw_factors() -> dict[str, list[dict]]:
    """Return per-device raw factor dicts saved after analysis."""
    return st.session_state.get(_KEY_RAW_FACTORS, {})  # type: ignore[no-any-return]


def set_raw_factors(factors: dict[str, list[dict]]) -> None:
    """Store per-device raw factor dicts for weight recalculation."""
    st.session_state[_KEY_RAW_FACTORS] = factors


# ── Deep LLM Analyses ────────────────────────────────────────────────────────


def get_deep_device_analyses() -> dict[str, DeepDeviceAnalysis]:
    """Return per-device deep LLM analyses (may be empty)."""
    return st.session_state.get(_KEY_DEEP_DEVICE_ANALYSES, {})  # type: ignore[no-any-return]


def set_deep_device_analyses(data: dict[str, DeepDeviceAnalysis]) -> None:
    """Store deep LLM analyses keyed by device_id."""
    st.session_state[_KEY_DEEP_DEVICE_ANALYSES] = dict(data)


def get_mass_error_analyses() -> dict[str, MassErrorAnalysis]:
    """Return LLM analyses of mass error codes (may be empty)."""
    return st.session_state.get(_KEY_MASS_ERROR_ANALYSES, {})  # type: ignore[no-any-return]


def set_mass_error_analyses(data: dict[str, MassErrorAnalysis]) -> None:
    """Store mass-error LLM analyses keyed by error_code."""
    st.session_state[_KEY_MASS_ERROR_ANALYSES] = dict(data)


def claim_bg_results_if_any() -> bool:
    """Promote background-analysis results into this session's state.

    The ingestion worker stores finished results in module-level
    `state._analysis_bg`. Until this session has consumed them, any page
    calling this helper will pull them into `st.session_state`. Safe and
    idempotent per session — subsequent calls after claiming are no-ops.
    """
    from state import _analysis_bg as _bg

    if _bg.thread is None or _bg.thread.is_alive():
        return False
    if _bg.progress.get("status") != "complete":
        return False

    run_id = id(_bg.thread)
    if st.session_state.get(_KEY_CLAIMED_BG_RUN) == run_id:
        return False

    report_obj = _bg.progress.get("_result_report")
    fs_obj = _bg.progress.get("_result_fs")
    health_obj = _bg.progress.get("_result_health")
    raw_factors_obj = _bg.progress.get("_result_raw_factors")

    # Пустой health_obj == [] тоже валидный результат (все устройства отфильтрованы).
    # Поэтому проверяем именно отсутствие ключа (None), а не falsy-значение.
    if report_obj is None or fs_obj is None or health_obj is None:
        return False

    set_current_factor_store(fs_obj)
    set_current_health_results(health_obj)
    set_baseline_health_results(health_obj)
    set_current_report(report_obj)
    if raw_factors_obj:
        set_raw_factors(raw_factors_obj)

    st.session_state[_KEY_CLAIMED_BG_RUN] = run_id
    return True


def clear_all() -> None:
    """Remove all MFU-related keys from session state."""
    for key in (
        _KEY_FACTOR_STORE,
        _KEY_HEALTH_RESULTS,
        _KEY_BASELINE_RESULTS,
        _KEY_REPORT,
        _KEY_WEIGHTS_PROFILE,
        _KEY_LLM_ENDPOINT,
        _KEY_CHAT_HISTORY,
        _KEY_RAW_FACTORS,
        _KEY_DEEP_DEVICE_ANALYSES,
        _KEY_MASS_ERROR_ANALYSES,
    ):
        st.session_state.pop(key, None)


def clear_derived_state() -> None:
    """Drop everything tied to a prior analysis run.

    Keeps the user's active weights profile, LLM endpoint selection, and
    in-progress weights draft. Clears report, factor store, health results,
    baseline, raw factors, deep and mass-error analyses, chat history,
    chat traces, and the bg-claim marker.
    """
    keys_to_clear = {
        _KEY_FACTOR_STORE,
        _KEY_HEALTH_RESULTS,
        _KEY_BASELINE_RESULTS,
        _KEY_REPORT,
        _KEY_CHAT_HISTORY,
        _KEY_RAW_FACTORS,
        _KEY_DEEP_DEVICE_ANALYSES,
        _KEY_MASS_ERROR_ANALYSES,
        _KEY_CLAIMED_BG_RUN,
        "mfu_chat_traces",  # per-page state in 4_LLM_Chat.py
    }
    for key in keys_to_clear:
        st.session_state.pop(key, None)
