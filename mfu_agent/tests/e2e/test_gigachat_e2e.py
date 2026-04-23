"""E2E verification for GigaChat endpoint (Phase PR 2 — multi-provider).

Skipped automatically unless ``GIGACHAT_AUTH_KEY`` env var is set with a
real base64 credential.  Mirrors the shape of ``tests/e2e/test_llm_client.py``.

Run locally with:
    GIGACHAT_AUTH_KEY=<your-base64> .venv/bin/pytest \\
        tests/e2e/test_gigachat_e2e.py -v -s
"""

from __future__ import annotations

import os

import pytest

from config.loader import LLMEndpointConfig
from llm.client import LLMClient, LLMResponse

# ── Availability check ────────────────────────────────────────────────────

requires_gigachat = pytest.mark.skipif(
    not os.environ.get("GIGACHAT_AUTH_KEY"),
    reason="GIGACHAT_AUTH_KEY env var not set",
)


def _make_gigachat_config(model: str = "GigaChat") -> LLMEndpointConfig:
    """Build an LLMEndpointConfig matching configs/llm_endpoints.yaml::gigachat."""
    return LLMEndpointConfig.model_validate(
        {
            "url": "https://gigachat.devices.sberbank.ru/api/v1",
            "api_key": "",
            "model": model,
            "timeout_seconds": 120.0,
            "max_retries_network": 2,
            "max_retries_invalid": 2,
            "display_name": "GigaChat (e2e)",
            "auth": {
                "type": "gigachat_oauth",
                "scope": os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
                "auth_key_env": "GIGACHAT_AUTH_KEY",
                "verify_tls": False,  # dev — see docs for prod ca_bundle
            },
        }
    )


# ── Tests ─────────────────────────────────────────────────────────────────


@requires_gigachat
class TestGigaChatE2E:
    """End-to-end verification against real Sber GigaChat API."""

    def test_ping_succeeds(self) -> None:
        client = LLMClient(_make_gigachat_config())
        result = client.ping()
        assert result.ok is True, f"GigaChat ping failed: {result.error}"
        assert result.latency_ms < 5000, (
            f"GigaChat ping too slow: {result.latency_ms} ms"
        )

    def test_detect_capabilities_is_native_or_react(self) -> None:
        """GigaChat должен поддерживать либо native tool_calls, либо react fallback."""
        client = LLMClient(_make_gigachat_config())
        strategy = client.detect_capabilities()
        assert strategy in {"native", "guided_json", "react"}

    def test_generate_simple_prompt_returns_response(self) -> None:
        client = LLMClient(_make_gigachat_config())
        resp = client.generate(
            [{"role": "user", "content": "Напиши одно русское слово: привет."}]
        )
        assert isinstance(resp, LLMResponse)
        assert resp.content, "empty content from GigaChat"

    def test_token_is_refreshed_on_first_use(self) -> None:
        """After construction, the first generate() call triggers OAuth refresh."""
        client = LLMClient(_make_gigachat_config())
        # Before any request: token cache is empty
        tp = client._token_provider
        assert tp._token is None  # type: ignore[attr-defined]

        client.ping()
        assert tp._token is not None  # type: ignore[attr-defined]
        assert tp._expires_at > 0  # type: ignore[attr-defined]
