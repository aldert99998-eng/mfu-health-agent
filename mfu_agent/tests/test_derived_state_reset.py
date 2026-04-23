"""Upload must atomically drop state derived from the prior analysis.

Covers the P0 bug where deep-device analyses, mass-error analyses,
baseline results, chat history, and chat traces survived a new file
upload — confusing users who saw stale artefacts on top of the new run.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def stub_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the `streamlit` module with a dict-backed session_state double.

    `state.session.clear_derived_state` calls `st.session_state.pop`; a plain
    dict supports that while letting tests assert on contents directly.
    """
    fake_state: dict = {}
    fake_st = types.ModuleType("streamlit")
    fake_st.session_state = fake_state  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    # Force re-import of state.session so it picks up the fake.
    sys.modules.pop("state.session", None)
    return fake_state


def test_clear_derived_state_drops_run_artefacts(stub_streamlit: dict) -> None:
    from state.session import clear_derived_state

    fake_state = stub_streamlit
    fake_state.update({
        "mfu_factor_store": MagicMock(),
        "mfu_health_results": [MagicMock()],
        "mfu_baseline_health_results": [MagicMock()],
        "mfu_report": MagicMock(),
        "mfu_chat_history": [{"role": "user", "content": "old"}],
        "mfu_raw_factors": {"DEV001": []},
        "mfu_deep_device_analyses": {"DEV001": MagicMock()},
        "mfu_mass_error_analyses": {"75-530-00": MagicMock()},
        "mfu_claimed_bg_run_id": 123456,
        "mfu_chat_traces": {0: "trace"},
    })

    clear_derived_state()

    for key in (
        "mfu_factor_store",
        "mfu_health_results",
        "mfu_baseline_health_results",
        "mfu_report",
        "mfu_chat_history",
        "mfu_raw_factors",
        "mfu_deep_device_analyses",
        "mfu_mass_error_analyses",
        "mfu_claimed_bg_run_id",
        "mfu_chat_traces",
    ):
        assert key not in fake_state, f"{key} survived clear_derived_state"


def test_clear_derived_state_preserves_user_preferences(stub_streamlit: dict) -> None:
    from state.session import clear_derived_state

    fake_state = stub_streamlit
    fake_state.update({
        "mfu_weights_profile": MagicMock(),
        "mfu_llm_endpoint": "local-qwen",
        "weights_profile_draft": MagicMock(),
        "mfu_report": MagicMock(),
    })

    clear_derived_state()

    assert "mfu_weights_profile" in fake_state, "active weights profile must survive"
    assert "mfu_llm_endpoint" in fake_state, "LLM endpoint selection must survive"
    assert "weights_profile_draft" in fake_state, "in-progress slider state must survive"
    assert "mfu_report" not in fake_state, "derived report must be cleared"


def test_claim_bg_results_accepts_empty_health_list(stub_streamlit: dict) -> None:
    """Пустой health_obj == [] — валидный результат (все устройства отфильтрованы).

    Регрессия: раньше truthy-check `if not (report and fs and health)` отклонял
    пустые списки, из-за чего UI показывал устаревший кэш вместо нового прогона.
    """
    import state._analysis_bg as bg
    from state.session import claim_bg_results_if_any

    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = False
    bg.thread = fake_thread
    bg.progress = {
        "status": "complete",
        "_result_report": MagicMock(),
        "_result_fs": MagicMock(),
        "_result_health": [],
    }

    try:
        claimed = claim_bg_results_if_any()
    finally:
        bg.thread = None
        bg.progress = {}

    assert claimed is True, "Пустой health должен быть промоутнут в session_state"
    assert stub_streamlit.get("mfu_health_results") == [], (
        "health_results должен быть установлен именно в пустой список"
    )
