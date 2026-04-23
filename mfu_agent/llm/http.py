"""HTTP client wiring for LLMClient.

Builds the ``httpx.Client`` that the OpenAI SDK will use for every request.
For endpoints with OAuth auth (e.g. GigaChat) the outgoing request's
``Authorization`` header is overwritten with a fresh Bearer from the
TokenProvider on every single request — guaranteeing we never send a stale
token even as it rotates every 30 min.

For local endpoints (StaticTokenProvider) the OpenAI SDK's own
``api_key``-based Bearer is left untouched (the hook is a no-op).

We also centralize the TLS trust-store decision here.  For the OAuth URL
itself (the one that mints tokens) the trust decision lives inside
``GigaChatTokenProvider`` — this module only handles the `/chat/completions`
endpoint trust.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from .auth import StaticTokenProvider, TokenProvider

if TYPE_CHECKING:
    from config.loader import LLMEndpointConfig


def _resolve_verify(config: LLMEndpointConfig) -> bool | str:
    """Trust-store verdict for the main chat endpoint (not OAuth URL)."""
    if config.auth is None:
        return True
    if config.auth.ca_bundle:
        return config.auth.ca_bundle
    return bool(config.auth.verify_tls)


def build_http_client(
    config: LLMEndpointConfig,
    token_provider: TokenProvider,
) -> httpx.Client:
    """Return an ``httpx.Client`` wired for the given endpoint.

    Notes
    -----
    * For a ``StaticTokenProvider`` the hook is a pure no-op and the OpenAI
      SDK's default Bearer-from-api_key flow is preserved byte-for-byte.
    * For GigaChat the hook calls ``get_token()`` which is cached — so it
      does NOT hit the OAuth endpoint on every request, only when close to
      expiry.
    """

    def _auth_hook(request: httpx.Request) -> None:
        if isinstance(token_provider, StaticTokenProvider):
            return
        request.headers["Authorization"] = f"Bearer {token_provider.get_token()}"

    return httpx.Client(
        verify=_resolve_verify(config),
        timeout=config.timeout_seconds,
        event_hooks={"request": [_auth_hook]},
    )
