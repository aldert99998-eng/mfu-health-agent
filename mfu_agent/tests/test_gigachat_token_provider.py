"""Unit tests for GigaChatTokenProvider (PR 2).

These tests use monkeypatch on ``httpx.Client`` so they never hit the real
Sber OAuth endpoint — CI-safe, network-free.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from config.loader import GigaChatAuthConfig, LLMEndpointConfig
from llm.auth import (
    GigaChatTokenProvider,
    LLMConfigError,
    StaticTokenProvider,
    build_token_provider,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _mock_oauth_response(
    token: str = "tok-abc123",
    expires_in: float = 1800,
) -> MagicMock:
    """Simulate httpx.Client().post(...) → Response."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "access_token": token,
        "expires_at": int((time.time() + expires_in) * 1000),  # ms epoch
    }
    return resp


def _patch_httpx_client_with(mock_resp: MagicMock):
    """Return a patcher that replaces httpx.Client with one returning mock_resp."""
    mock_client_cm = MagicMock()
    mock_client_cm.__enter__.return_value.post.return_value = mock_resp
    mock_client_cm.__exit__.return_value = None
    return patch.object(httpx, "Client", return_value=mock_client_cm)


# ── factory ──────────────────────────────────────────────────────────────


class TestBuildTokenProvider:
    def test_no_auth_returns_static(self):
        cfg = LLMEndpointConfig(api_key="local-key")
        tp = build_token_provider(cfg)
        assert isinstance(tp, StaticTokenProvider)
        assert tp.get_token() == "local-key"

    def test_gigachat_auth_returns_oauth_provider(self, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "base64credentialshere")
        cfg = LLMEndpointConfig.model_validate(
            {
                "url": "https://gigachat.devices.sberbank.ru/api/v1",
                "model": "GigaChat",
                "auth": {"type": "gigachat_oauth"},
            }
        )
        tp = build_token_provider(cfg)
        assert isinstance(tp, GigaChatTokenProvider)

    def test_unknown_auth_type_raises(self):
        # We can't construct LLMEndpointConfig with a bogus auth type (Literal
        # rejects it), so build_token_provider only sees valid types in
        # practice.  This test exercises defensive coding — bypass pydantic:
        class _FakeAuth:
            type = "unknown_provider"

        class _FakeCfg:
            auth = _FakeAuth()
            api_key = ""
            timeout_seconds = 30.0

        with pytest.raises(LLMConfigError, match="Unknown auth.type"):
            build_token_provider(_FakeCfg())  # type: ignore[arg-type]


# ── GigaChatTokenProvider ────────────────────────────────────────────────


class TestGigaChatTokenProvider:
    @pytest.fixture
    def auth_cfg(self) -> GigaChatAuthConfig:
        return GigaChatAuthConfig(type="gigachat_oauth")

    def test_missing_env_var_raises(self, auth_cfg, monkeypatch):
        monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
        with pytest.raises(LLMConfigError, match="auth_key missing"):
            GigaChatTokenProvider(auth_cfg)

    def test_empty_env_var_raises(self, auth_cfg, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "   ")
        with pytest.raises(LLMConfigError, match="auth_key missing"):
            GigaChatTokenProvider(auth_cfg)

    def test_first_call_triggers_refresh(self, auth_cfg, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        with _patch_httpx_client_with(_mock_oauth_response("tok-1")):
            p = GigaChatTokenProvider(auth_cfg)
            assert p.get_token() == "tok-1"

    def test_second_call_uses_cache(self, auth_cfg, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        resp = _mock_oauth_response("tok-cached", expires_in=1800)
        with _patch_httpx_client_with(resp) as m:
            p = GigaChatTokenProvider(auth_cfg)
            a = p.get_token()
            b = p.get_token()
            c = p.get_token()
            assert a == b == c == "tok-cached"
            # Client constructed only once (first refresh)
            assert m.call_count == 1

    def test_expired_token_triggers_refresh(self, auth_cfg, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        # Token expires in 30 seconds — less than safety margin (120s) → immediate refresh next call.
        resp1 = _mock_oauth_response("tok-old", expires_in=30)
        resp2 = _mock_oauth_response("tok-new", expires_in=1800)
        with patch.object(httpx, "Client") as m_client:
            cm1 = MagicMock()
            cm1.__enter__.return_value.post.return_value = resp1
            cm2 = MagicMock()
            cm2.__enter__.return_value.post.return_value = resp2
            m_client.side_effect = [cm1, cm2]

            p = GigaChatTokenProvider(auth_cfg)
            assert p.get_token() == "tok-old"
            # next call: safety margin kicks in, must refresh
            assert p.get_token() == "tok-new"
            assert m_client.call_count == 2

    def test_invalidate_forces_refresh(self, auth_cfg, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        resp1 = _mock_oauth_response("tok-A", expires_in=1800)
        resp2 = _mock_oauth_response("tok-B", expires_in=1800)
        with patch.object(httpx, "Client") as m_client:
            cm1 = MagicMock()
            cm1.__enter__.return_value.post.return_value = resp1
            cm2 = MagicMock()
            cm2.__enter__.return_value.post.return_value = resp2
            m_client.side_effect = [cm1, cm2]

            p = GigaChatTokenProvider(auth_cfg)
            assert p.get_token() == "tok-A"
            p.invalidate()
            assert p.get_token() == "tok-B"
            assert m_client.call_count == 2

    def test_concurrent_first_calls_refresh_once(self, auth_cfg, monkeypatch):
        """10 threads racing on first get_token() must produce exactly ONE refresh."""
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        resp = _mock_oauth_response("tok-race", expires_in=1800)
        refresh_calls: list[float] = []

        def _slow_post(*args, **kwargs):
            # Simulate 50ms OAuth round-trip — widens the race window.
            refresh_calls.append(time.time())
            time.sleep(0.05)
            return resp

        cm = MagicMock()
        cm.__enter__.return_value.post.side_effect = _slow_post
        cm.__exit__.return_value = None

        with patch.object(httpx, "Client", return_value=cm):
            p = GigaChatTokenProvider(auth_cfg)
            results = []

            def _worker():
                results.append(p.get_token())

            threads = [threading.Thread(target=_worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert results == ["tok-race"] * 10
            assert len(refresh_calls) == 1, (
                f"expected exactly 1 refresh, got {len(refresh_calls)}"
            )

    def test_oauth_failure_raises_config_error(self, auth_cfg, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        cm = MagicMock()
        cm.__enter__.return_value.post.side_effect = httpx.HTTPError("network down")
        cm.__exit__.return_value = None
        with patch.object(httpx, "Client", return_value=cm):
            p = GigaChatTokenProvider(auth_cfg)
            with pytest.raises(LLMConfigError, match="OAuth request failed"):
                p.get_token()

    def test_missing_access_token_in_body_raises(self, auth_cfg, monkeypatch):
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"wrong_field": "whatever"}
        with _patch_httpx_client_with(resp):
            p = GigaChatTokenProvider(auth_cfg)
            with pytest.raises(LLMConfigError, match="no access_token"):
                p.get_token()

    def test_expires_in_seconds_fallback(self, auth_cfg, monkeypatch):
        """If server returns only expires_in (not expires_at ms), we handle it."""
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "dummy-key")
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"access_token": "tok-exp-in", "expires_in": 1200}
        with _patch_httpx_client_with(resp):
            p = GigaChatTokenProvider(auth_cfg)
            assert p.get_token() == "tok-exp-in"
            # Roughly 1200s − now, minus some delta
            assert p._expires_at > time.time() + 1000

    def test_auth_headers_and_scope_sent_correctly(self, auth_cfg, monkeypatch):
        """Verify Basic auth, RqUID UUID, and scope body field."""
        monkeypatch.setenv("GIGACHAT_AUTH_KEY", "mykeyvalue==")
        resp = _mock_oauth_response()

        cm = MagicMock()
        cm.__enter__.return_value.post.return_value = resp
        cm.__exit__.return_value = None

        with patch.object(httpx, "Client", return_value=cm):
            p = GigaChatTokenProvider(auth_cfg)
            p.get_token()

            call_args = cm.__enter__.return_value.post.call_args
            # positional arg: URL
            assert call_args.args[0] == auth_cfg.oauth_url

            # kwargs
            headers = call_args.kwargs["headers"]
            data = call_args.kwargs["data"]

            assert headers["Authorization"] == "Basic mykeyvalue=="
            assert "RqUID" in headers
            # RqUID is a UUID
            import uuid as _uuid

            _uuid.UUID(headers["RqUID"])  # raises if not a UUID

            assert data == {"scope": "GIGACHAT_API_PERS"}


# ── StaticTokenProvider ──────────────────────────────────────────────────


class TestStaticTokenProvider:
    def test_echoes_api_key(self):
        p = StaticTokenProvider("my-secret-xyz")
        assert p.get_token() == "my-secret-xyz"
        assert p.get_token() == "my-secret-xyz"

    def test_invalidate_is_noop(self):
        p = StaticTokenProvider("x")
        p.invalidate()  # must not raise
        assert p.get_token() == "x"
