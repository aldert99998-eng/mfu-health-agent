"""OAuth / static-key token providers for LLMClient.

The LLMClient is OpenAI-compatible and was originally wired for one concrete
endpoint (local llama-server). To add a second provider whose `Authorization`
header rotates on a schedule (GigaChat issues a fresh `access_token` every
30 minutes via OAuth), we inject a small `TokenProvider` whose value is read
by an httpx event hook on every outgoing request.

Providers:
    * StaticTokenProvider  — echoes the configured api_key (local path; no-op).
    * GigaChatTokenProvider — requests a fresh Bearer via Sber OAuth, caches
      it until `safety_margin_s` before expiry, thread-safe.

Why not inherit a full provider abstraction?  The three strategies (native /
guided_json / react) in LLMClient are provider-agnostic.  The ONLY real
difference for GigaChat is the auth header, and — separately — the TLS trust
store.  Both are solved by the tiny abstraction in this module and the one
in ``llm/http.py``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from config.loader import GigaChatAuthConfig, LLMEndpointConfig

logger = logging.getLogger(__name__)


class LLMConfigError(Exception):
    """Raised when endpoint auth configuration is invalid or incomplete."""


class TokenProvider(ABC):
    """Interface: give me the current Bearer token value."""

    @abstractmethod
    def get_token(self) -> str:
        """Return a valid token; refresh transparently if needed."""

    def invalidate(self) -> None:
        """Force the next get_token() to refresh. No-op for static providers."""


class StaticTokenProvider(TokenProvider):
    """For local/dev endpoints — the configured api_key is the token."""

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    def get_token(self) -> str:
        return self._key


class GigaChatTokenProvider(TokenProvider):
    """OAuth2 client-credentials flow with automatic refresh.

    Thread-safe: concurrent `get_token()` calls will trigger at most one
    refresh (`_lock` serializes the `_refresh_locked` path).  Preemptive
    refresh: if the remaining TTL is less than `safety_margin_s`, we refresh
    before handing out the token.

    The caller is expected to pair this with a 401-invalidation in
    LLMClient._call_api — if the server rejects even a "fresh" token (clock
    skew), we drop the cache and retry once.
    """

    _SAFETY_MARGIN_S = 120.0
    _DEFAULT_TTL_S = 1800.0  # GigaChat tokens live ~30 min

    def __init__(
        self,
        auth_cfg: GigaChatAuthConfig,
        *,
        oauth_timeout_s: float = 10.0,
    ) -> None:
        self._cfg = auth_cfg
        self._oauth_timeout_s = oauth_timeout_s
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

        auth_key = os.environ.get(auth_cfg.auth_key_env, "").strip()
        if not auth_key:
            raise LLMConfigError(
                f"GigaChat auth_key missing: set env var {auth_cfg.auth_key_env} "
                f"(see mfu_agent/.env.example)"
            )
        self._auth_key = auth_key

        # TLS trust store for the OAuth endpoint itself.
        if auth_cfg.ca_bundle:
            self._verify: bool | str = auth_cfg.ca_bundle
        elif auth_cfg.verify_tls:
            self._verify = True
        else:
            logger.warning(
                "GigaChat: TLS verification DISABLED for %s. "
                "OK for dev, NOT for production. "
                "Set ca_bundle in configs/llm_endpoints.yaml for prod.",
                auth_cfg.oauth_url,
            )
            self._verify = False

    # ── public API ────────────────────────────────────────────────────────

    def get_token(self) -> str:
        with self._lock:
            if (
                self._token is None
                or time.time() + self._SAFETY_MARGIN_S >= self._expires_at
            ):
                self._refresh_locked()
            assert self._token is not None
            return self._token

    def invalidate(self) -> None:
        with self._lock:
            self._token = None
            self._expires_at = 0.0
            logger.debug("GigaChat OAuth: token invalidated, next call will refresh")

    # ── internals ─────────────────────────────────────────────────────────

    def _refresh_locked(self) -> None:
        """Caller must hold self._lock.  Obtains a fresh access_token."""
        rq_uid = str(uuid.uuid4())
        headers = {
            "Authorization": f"Basic {self._auth_key}",
            "RqUID": rq_uid,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        data = {"scope": self._cfg.scope}
        logger.info(
            "GigaChat OAuth: refreshing (scope=%s)", self._cfg.scope,
        )
        try:
            with httpx.Client(verify=self._verify, timeout=self._oauth_timeout_s) as c:
                resp = c.post(self._cfg.oauth_url, headers=headers, data=data)
                resp.raise_for_status()
                body = resp.json()
        except httpx.HTTPError as exc:
            raise LLMConfigError(
                f"GigaChat OAuth request failed ({self._cfg.oauth_url}): {exc}"
            ) from exc

        token = body.get("access_token")
        if not token or not isinstance(token, str):
            raise LLMConfigError(
                f"GigaChat OAuth returned no access_token in body: keys={list(body.keys())}"
            )

        # GigaChat returns `expires_at` as epoch MILLISECONDS.  Some older
        # variants return seconds-TTL under `expires_in`.  Handle both.
        expires_at_ms = body.get("expires_at")
        if isinstance(expires_at_ms, int | float) and expires_at_ms > 1e11:
            expires_at_s = float(expires_at_ms) / 1000.0
        elif isinstance(body.get("expires_in"), int | float):
            expires_at_s = time.time() + float(body["expires_in"])
        else:
            expires_at_s = time.time() + self._DEFAULT_TTL_S

        self._token = token
        self._expires_at = expires_at_s
        ttl = int(self._expires_at - time.time())
        logger.info("GigaChat OAuth: refreshed, valid ~%d s", ttl)


# ── factory ──────────────────────────────────────────────────────────────


def build_token_provider(config: LLMEndpointConfig) -> TokenProvider:
    """Pick the right provider based on ``config.auth``."""
    if config.auth is None:
        return StaticTokenProvider(config.api_key)
    if config.auth.type == "gigachat_oauth":
        return GigaChatTokenProvider(
            config.auth,
            oauth_timeout_s=min(30.0, config.timeout_seconds),
        )
    raise LLMConfigError(f"Unknown auth.type: {config.auth.type!r}")
