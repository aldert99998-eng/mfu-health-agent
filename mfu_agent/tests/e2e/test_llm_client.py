"""E2E verification for LLMClient (Phase 5.1 — LLM client checks).

Checks from the playbook:
1. Ping on a running endpoint → True, latency < 500 ms.
2. detect_capabilities correctly determines strategy for a known model.
3. Fake endpoint (unreachable) → ConnectionError with a clear message.

Requires: running OpenAI-compatible LLM on localhost:8000.
Run:  python -m pytest tests/e2e/test_llm_client.py -v -s
"""

from __future__ import annotations

import pytest
from openai import OpenAI

from llm.client import (
    STRATEGY_GUIDED_JSON,
    STRATEGY_NATIVE,
    STRATEGY_REACT,
    LLMClient,
    LLMConnectionError,
    LLMResponse,
)

# ── Availability check ────────────────────────────────────────────────────


def _llm_available() -> tuple[bool, str]:
    try:
        c = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="test",
            timeout=5,
        )
        resp = c.models.list()
        if resp.data:
            return True, resp.data[0].id
    except Exception:
        pass
    return False, ""


_LLM_OK, _MODEL_ID = _llm_available()

requires_llm = pytest.mark.skipif(
    not _LLM_OK,
    reason="No LLM endpoint running on localhost:8000",
)


def _make_config(**overrides):
    from config.loader import LLMEndpointConfig

    defaults = {
        "url": "http://localhost:8000/v1",
        "api_key": "test",
        "model": _MODEL_ID or "test-model",
        "timeout_seconds": 30.0,
    }
    defaults.update(overrides)
    return LLMEndpointConfig(**defaults)


# ── Tests ─────────────────────────────────────────────────────────────────


@requires_llm
class TestLLMClientE2E:
    """Phase 5.1 LLM client verification suite."""

    def test_ping_returns_true_and_low_latency(self) -> None:
        """Ping on running endpoint → ok=True, latency < 500 ms."""
        client = LLMClient(_make_config())
        result = client.ping()

        assert result.ok is True, f"Ping failed: {result.error}"
        assert result.latency_ms < 500, (
            f"Latency too high: {result.latency_ms} ms"
        )
        assert result.latency_ms > 0

    def test_detect_capabilities_returns_valid_strategy(self) -> None:
        """detect_capabilities returns one of the three valid strategies."""
        client = LLMClient(_make_config())
        strategy = client.detect_capabilities()

        assert strategy in {STRATEGY_NATIVE, STRATEGY_GUIDED_JSON, STRATEGY_REACT}, (
            f"Unknown strategy: {strategy}"
        )
        assert client.tool_strategy == strategy
        assert client._config.tool_strategy == strategy

    def test_detect_capabilities_is_cached(self) -> None:
        """Second call returns cached result without re-probing."""
        client = LLMClient(_make_config())
        s1 = client.detect_capabilities()
        s2 = client.detect_capabilities()
        assert s1 == s2

    def test_detect_capabilities_skips_if_preset(self) -> None:
        """If tool_strategy is preset in config, detection is skipped."""
        client = LLMClient(_make_config(tool_strategy="react"))
        assert client.tool_strategy == "react"

    def test_generate_returns_llm_response(self) -> None:
        """Basic generate call returns a valid LLMResponse."""
        client = LLMClient(_make_config())
        resp = client.generate(
            messages=[{"role": "user", "content": "Скажи «тест». Одно слово."}],
        )

        assert isinstance(resp, LLMResponse)
        assert len(resp.content) > 0
        assert resp.finish_reason != ""
        assert resp.usage.total_tokens > 0


class TestLLMClientFakeEndpoint:
    """Fake endpoint tests — no real LLM needed."""

    def test_ping_fake_endpoint_returns_false(self) -> None:
        """Ping on unreachable endpoint → ok=False with error message."""
        cfg = _make_config(
            url="http://127.0.0.1:19999/v1",
            timeout_seconds=2.0,
        )
        client = LLMClient(cfg)
        result = client.ping()

        assert result.ok is False
        assert len(result.error) > 0

    def test_generate_fake_endpoint_raises_connection_error(self) -> None:
        """Generate on unreachable endpoint → LLMConnectionError."""
        cfg = _make_config(
            url="http://127.0.0.1:19999/v1",
            timeout_seconds=2.0,
        )
        client = LLMClient(cfg)
        client._tool_strategy = "react"

        with pytest.raises(LLMConnectionError) as exc_info:
            client.generate(
                messages=[{"role": "user", "content": "test"}],
            )

        msg = str(exc_info.value)
        assert "127.0.0.1:19999" in msg or "подключиться" in msg.lower() or "connect" in msg.lower()

    def test_detect_capabilities_fake_endpoint_falls_to_react(self) -> None:
        """Unreachable endpoint → fallback to react strategy."""
        cfg = _make_config(
            url="http://127.0.0.1:19999/v1",
            timeout_seconds=2.0,
        )
        client = LLMClient(cfg)
        strategy = client.detect_capabilities()

        assert strategy == STRATEGY_REACT
