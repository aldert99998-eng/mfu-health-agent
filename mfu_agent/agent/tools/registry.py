"""Tool registry — Track B, Level 1.

Manages registration, schema exposure, and execution of agent tools.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── ToolResult ───────────────────────────────────────────────────────────────


class ToolResult(BaseModel):
    """Result returned by every tool execution."""

    success: bool
    data: Any = None
    error: str | None = None


# ── Tool protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class Tool(Protocol):
    """Interface that every agent tool must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any]) -> ToolResult: ...


# ── ToolRegistry ─────────────────────────────────────────────────────────────


class ToolRegistryError(Exception):
    """Raised on registry-level problems (unknown tool, duplicate, etc.)."""


class ToolRegistry:
    """Central registry for all agent tools.

    Provides schema listing for LLM tool descriptions and
    validated execution dispatch.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolRegistryError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("Tool зарегистрирован: %s", tool.name)

    def get_schema(self, name: str) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolRegistryError(f"Unknown tool: {name}")
        return tool.schema

    def get_all_schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {name}. Available: {', '.join(sorted(self._tools))}",
            )
        try:
            return tool.execute(args)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
