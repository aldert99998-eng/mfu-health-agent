"""Unit tests for LLMEndpointConfig + GigaChatAuthConfig (PR 1)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from config.loader import CONFIGS_DIR, GigaChatAuthConfig, LLMEndpointConfig


class TestLLMEndpointConfigLocal:
    """Local-endpoint path: auth=None, api_key carries the secret placeholder."""

    def test_defaults_have_no_auth(self):
        cfg = LLMEndpointConfig()
        assert cfg.auth is None
        assert cfg.display_name == ""
        assert cfg.api_key == "dummy-for-local"

    def test_local_endpoint_from_dict(self):
        cfg = LLMEndpointConfig.model_validate(
            {
                "url": "http://localhost:8000/v1",
                "api_key": "dummy-for-local",
                "model": "nvidia_NVIDIA-Nemotron-Nano-12B-v2-Q8_0.gguf",
                "timeout_seconds": 180.0,
                "max_retries_network": 3,
                "max_retries_invalid": 2,
                "display_name": "Локальная",
            }
        )
        assert cfg.auth is None
        assert cfg.display_name == "Локальная"


class TestLLMEndpointConfigGigaChat:
    """GigaChat-endpoint path: auth has type=gigachat_oauth."""

    def test_minimal_gigachat_from_dict(self):
        cfg = LLMEndpointConfig.model_validate(
            {
                "url": "https://gigachat.devices.sberbank.ru/api/v1",
                "api_key": "",
                "model": "GigaChat",
                "auth": {"type": "gigachat_oauth"},
            }
        )
        assert isinstance(cfg.auth, GigaChatAuthConfig)
        # defaults applied
        assert cfg.auth.scope == "GIGACHAT_API_PERS"
        assert cfg.auth.auth_key_env == "GIGACHAT_AUTH_KEY"
        assert cfg.auth.verify_tls is False
        assert cfg.auth.ca_bundle is None
        assert (
            cfg.auth.oauth_url
            == "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        )

    def test_gigachat_with_corp_scope(self):
        cfg = LLMEndpointConfig.model_validate(
            {
                "url": "https://gigachat.devices.sberbank.ru/api/v1",
                "model": "GigaChat-Pro",
                "auth": {
                    "type": "gigachat_oauth",
                    "scope": "GIGACHAT_API_CORP",
                    "verify_tls": True,
                    "ca_bundle": "/etc/ssl/russian_ca.pem",
                },
            }
        )
        assert cfg.auth is not None
        assert cfg.auth.scope == "GIGACHAT_API_CORP"
        assert cfg.auth.verify_tls is True
        assert cfg.auth.ca_bundle == "/etc/ssl/russian_ca.pem"


class TestLLMEndpointConfigInvalid:
    def test_invalid_auth_type_rejected(self):
        with pytest.raises(ValidationError):
            LLMEndpointConfig.model_validate(
                {
                    "url": "http://x/v1",
                    "api_key": "",
                    "model": "m",
                    "auth": {"type": "bogus_provider"},
                }
            )

    def test_missing_auth_type_rejected(self):
        # type is Literal, no default → must be provided
        with pytest.raises(ValidationError):
            LLMEndpointConfig.model_validate(
                {
                    "url": "http://x/v1",
                    "api_key": "",
                    "model": "m",
                    "auth": {"scope": "GIGACHAT_API_PERS"},
                }
            )


class TestLLMEndpointConfigYAML:
    """The shipped YAML must parse cleanly for every endpoint entry."""

    def test_llm_endpoints_yaml_parses(self):
        path = CONFIGS_DIR / "llm_endpoints.yaml"
        assert path.exists(), f"YAML missing: {path}"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        endpoints = data.get("endpoints") or {}
        assert endpoints, "expected at least one endpoint in configs/llm_endpoints.yaml"
        for name, ep in endpoints.items():
            cfg = LLMEndpointConfig.model_validate(ep)
            # sanity: url+model populated, auth is a GigaChatAuthConfig iff set
            assert cfg.url, f"endpoint '{name}' has empty url"
            assert cfg.model, f"endpoint '{name}' has empty model"
            if cfg.auth is not None:
                assert isinstance(cfg.auth, GigaChatAuthConfig)

    def test_default_and_gigachat_both_present(self):
        path = CONFIGS_DIR / "llm_endpoints.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        endpoints = data.get("endpoints") or {}
        assert "default" in endpoints
        assert "gigachat" in endpoints

    def test_gigachat_yaml_has_oauth_auth(self):
        path = CONFIGS_DIR / "llm_endpoints.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        gc = LLMEndpointConfig.model_validate(data["endpoints"]["gigachat"])
        assert gc.auth is not None
        assert gc.auth.type == "gigachat_oauth"
        assert gc.auth.auth_key_env == "GIGACHAT_AUTH_KEY"
