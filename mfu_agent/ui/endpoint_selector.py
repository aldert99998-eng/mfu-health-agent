"""Shared LLM-endpoint selector + ping-status badge.

Usage::

    from ui.endpoint_selector import render_endpoint_selector

    active = render_endpoint_selector(location="main", key_suffix="upload")

The component:
1. Lists every endpoint defined in ``configs/llm_endpoints.yaml``.
2. Renders a ``st.selectbox`` (either in sidebar or main column) using
   ``display_name`` when set.
3. On change: calls ``set_active_llm_endpoint`` + ``invalidate_llm_singletons``
   + ``st.rerun`` — the next rerun picks up the new client everywhere.
4. Shows a ping badge (🟢 доступно · N ms / 🔴 недоступно: <err> / ⚪ не
   проверено) with a short-lived cache in ``st.session_state`` (5 min TTL)
   so free-pings don't hammer the endpoint on every Streamlit rerun.
5. When the endpoint has ``auth.verify_tls=false`` and no ``ca_bundle``,
   warns the user that TLS is disabled.

Returned: the currently active endpoint name.
"""

from __future__ import annotations

import time
from typing import Literal

import streamlit as st
import yaml

from config.loader import CONFIGS_DIR, LLMEndpointConfig
from state.session import get_active_llm_endpoint, set_active_llm_endpoint
from state.singletons import get_llm_client, invalidate_llm_singletons

_PING_CACHE_KEY = "_mfu_endpoint_ping_cache"
_PING_TTL_S = 300.0


# ── helpers ──────────────────────────────────────────────────────────────


def _load_endpoints() -> dict[str, LLMEndpointConfig]:
    """Parse llm_endpoints.yaml → {name: LLMEndpointConfig}. Falls back to default."""
    path = CONFIGS_DIR / "llm_endpoints.yaml"
    if not path.exists():
        return {"default": LLMEndpointConfig()}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {"default": LLMEndpointConfig()}
    raw = data.get("endpoints", data)
    if not isinstance(raw, dict):
        return {"default": LLMEndpointConfig()}
    out: dict[str, LLMEndpointConfig] = {}
    for name, cfg_data in raw.items():
        try:
            out[name] = LLMEndpointConfig.model_validate(cfg_data)
        except Exception:  # noqa: BLE001 — skip malformed entries
            continue
    return out or {"default": LLMEndpointConfig()}


def _cached_ping(
    endpoint_name: str, force: bool = False
) -> tuple[bool, float, str]:
    """Return (ok, latency_ms, error) for ``endpoint_name`` using the session cache."""
    cache: dict = st.session_state.setdefault(_PING_CACHE_KEY, {})
    entry = cache.get(endpoint_name)
    if (
        not force
        and entry is not None
        and time.time() - entry["at"] < _PING_TTL_S
    ):
        return entry["ok"], entry["lat"], entry["err"]
    try:
        client = get_llm_client(endpoint_name)
        # Streamlit may hand back a cached LLMClient instance built before
        # list_available_models existed — drop the singleton and rebuild once
        # so the next rerun sees a fresh object.
        if not hasattr(client, "list_available_models"):
            invalidate_llm_singletons()
            client = get_llm_client(endpoint_name)
        r = client.ping()
        list_fn = getattr(client, "list_available_models", None)
        models = list_fn() if (r.ok and callable(list_fn)) else []
        entry = {
            "ok": bool(r.ok),
            "lat": float(r.latency_ms),
            "err": r.error,
            "models": models,
            "at": time.time(),
        }
    except Exception as exc:  # noqa: BLE001 — surface as UI state
        entry = {
            "ok": False,
            "lat": 0.0,
            "err": str(exc)[:200],
            "models": [],
            "at": time.time(),
        }
    cache[endpoint_name] = entry
    return entry["ok"], entry["lat"], entry["err"]


def _cached_available_models(endpoint_name: str) -> list[str]:
    """Read models discovered during the last ping (empty if never pinged or offline)."""
    cache: dict = st.session_state.get(_PING_CACHE_KEY, {})
    entry = cache.get(endpoint_name) or {}
    return list(entry.get("models") or [])


def get_cached_ping_ok(endpoint_name: str) -> bool:
    """Read-only helper for callers (e.g. pages that gate a Start button)."""
    cache: dict = st.session_state.get(_PING_CACHE_KEY, {})
    entry = cache.get(endpoint_name)
    return bool(entry and entry.get("ok"))


# ── main component ───────────────────────────────────────────────────────


def render_endpoint_selector(
    location: Literal["sidebar", "main"] = "sidebar",
    *,
    key_suffix: str = "",
    show_status: bool = True,
) -> str:
    endpoints = _load_endpoints()
    names = list(endpoints.keys())
    current = get_active_llm_endpoint()
    if current not in names:
        current = names[0]

    def _label(n: str) -> str:
        dn = endpoints[n].display_name
        return f"{dn}" if dn else n

    # ``st`` itself is not a context manager — use ``st.container()`` for main.
    container = st.sidebar if location == "sidebar" else st.container()
    with container:
        if location == "main":
            st.markdown("#### 🧠 LLM-провайдер")
            st.caption(
                "Выберите провайдера ДО загрузки файла — все запросы "
                "классификации и анализа пойдут через него."
            )
        else:
            st.subheader("LLM Endpoint")

        selected = st.selectbox(
            "Провайдер" if location == "main" else "Endpoint",
            names,
            index=names.index(current),
            format_func=_label,
            key=f"llm_endpoint_select_{key_suffix}",
        )

        # Prefer live answer from /v1/models — reflects what's actually
        # loaded on the server, not what the YAML hint says.
        # On the main page (upload) we hide the exact model — users pick
        # the provider, details live in the sidebar.
        if location != "main":
            live_models = _cached_available_models(selected)
            shown_model = (
                ", ".join(live_models) if live_models else endpoints[selected].model
            )
            if shown_model:
                suffix = "" if live_models else " (из конфига)"
                st.caption(f"Модель: `{shown_model}`{suffix}")

        if selected != current:
            set_active_llm_endpoint(selected)
            invalidate_llm_singletons()
            # drop ping cache for the new endpoint so a fresh ping is shown
            cache = st.session_state.get(_PING_CACHE_KEY, {})
            cache.pop(selected, None)
            st.rerun()

        if show_status:
            ok, lat, err = _cached_ping(selected)
            cols = st.columns([3, 1])
            with cols[0]:
                if ok:
                    st.caption(f"🟢 доступно · {lat:.0f} ms")
                elif err:
                    st.caption(f"🔴 недоступно: {err[:80]}")
                else:
                    st.caption("⚪ не проверено")
            with cols[1]:
                if st.button(
                    "Проверить",
                    key=f"ping_btn_{key_suffix}",
                    use_container_width=True,
                ):
                    _cached_ping(selected, force=True)
                    st.rerun()

            cfg = endpoints[selected]
            if (
                cfg.auth is not None
                and not cfg.auth.verify_tls
                and not cfg.auth.ca_bundle
            ):
                st.warning(
                    "⚠️ TLS-проверка отключена для этого эндпоинта. "
                    "Для production укажите `ca_bundle` в "
                    "`configs/llm_endpoints.yaml`.",
                    icon="🔓",
                )

        _render_local_model_switcher(
            selected_endpoint=selected,
            endpoint_cfg=endpoints[selected],
            key_suffix=key_suffix,
        )

    return selected


def _analysis_is_running() -> bool:
    try:
        from state import _analysis_bg as _bg
    except Exception:  # noqa: BLE001 — optional during tests
        return False
    return _bg.thread is not None and _bg.thread.is_alive()


def _render_local_model_switcher(
    *,
    selected_endpoint: str,
    endpoint_cfg: LLMEndpointConfig,
    key_suffix: str,
) -> None:
    """Show a gguf-model picker for the local llama-server endpoint."""
    from ui.local_model_manager import (
        LLAMA_LOG_PATH,
        MODELS_DIR,
        detect_running_server,
        list_gguf_models,
        switch_model,
    )

    running = detect_running_server()
    if running is None:
        return

    models = list_gguf_models()
    if len(models) < 2:
        return

    current_path = running.model_path
    with st.expander("🔀 Переключить локальную модель", expanded=False):
        st.caption(
            f"Файлы .gguf в `{MODELS_DIR}`. Переключение перезапустит "
            f"llama-server (PID {running.pid}) с теми же параметрами."
        )

        options = [p.name for p in models]
        try:
            current_idx = options.index(current_path.name)
        except ValueError:
            current_idx = 0

        chosen_name = st.selectbox(
            "Модель",
            options,
            index=current_idx,
            key=f"local_model_select_{key_suffix}",
        )
        chosen_path = next(p for p in models if p.name == chosen_name)
        size_gb = chosen_path.stat().st_size / (1024 ** 3)
        st.caption(f"Размер файла: {size_gb:.1f} GB")

        disabled_reason: str | None = None
        if chosen_path.resolve() == current_path.resolve():
            disabled_reason = "уже загружена"
        elif _analysis_is_running():
            disabled_reason = "идёт анализ — дождитесь завершения"

        cols = st.columns([3, 1])
        with cols[0]:
            if disabled_reason:
                st.caption(f"⏸ Переключение недоступно: {disabled_reason}")
        with cols[1]:
            clicked = st.button(
                "Применить",
                key=f"local_model_switch_{key_suffix}",
                type="primary",
                use_container_width=True,
                disabled=disabled_reason is not None,
            )

        if clicked:
            with st.spinner(f"Загрузка {chosen_name} на GPU… (до 2 минут)"):
                ok, msg = switch_model(chosen_path, endpoint_cfg.url)
            if ok:
                invalidate_llm_singletons()
                st.session_state.pop(_PING_CACHE_KEY, None)
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                st.caption(f"Полный лог llama-server: `{LLAMA_LOG_PATH}`")
