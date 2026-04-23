"""Drop ClassifyErrorSeverityTool._cache so next classify reloads severity."""

from __future__ import annotations


def invalidate_severity_caches() -> None:
    # No-op outside Streamlit — the singleton is UI-scoped, CLI/tests have no registry.
    try:
        from state.singletons import get_tool_registry
        registry = get_tool_registry()
    except Exception:
        return

    tool = registry._tools.get("classify_error_severity")
    cache = getattr(tool, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()
