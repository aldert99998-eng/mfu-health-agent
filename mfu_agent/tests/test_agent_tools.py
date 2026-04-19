"""Phase 5.2 verification — agent tools unit tests.

Checks:
1. Each of the 9 tools — unit test with mocked dependencies.
2. JSON schemas of all 9 tools validate via jsonschema.
3. execute of unknown tool → ToolRegistryError in result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import jsonschema
import pytest

from agent.tools.impl import (
    CalculateHealthIndexTool,
    ClassifyErrorSeverityTool,
    CountErrorRepetitionsTool,
    FindSimilarDevicesTool,
    GetDeviceEventsTool,
    GetDeviceHistoryTool,
    GetDeviceResourcesTool,
    GetFleetStatisticsTool,
    SearchServiceDocsTool,
    ToolDependencies,
    register_all_tools,
)
from agent.tools.registry import ToolRegistry

# ── Fake models for mocking ─────────────────────────────────────────────────


@dataclass
class FakeEvent:
    device_id: str = "DEV001"
    timestamp: datetime = datetime(2026, 4, 10, tzinfo=UTC)
    error_code: str | None = "SC542"
    error_description: str | None = "Fuser overheat"
    model: str | None = "MX-3071"
    vendor: str | None = "Sharp"
    location: str | None = "Office-A"
    status: str | None = "active"


@dataclass
class FakeResource:
    device_id: str = "DEV001"
    timestamp: datetime = datetime(2026, 4, 10, tzinfo=UTC)
    toner_level: int | None = 45
    drum_level: int | None = 60
    fuser_level: int | None = 30
    mileage: int | None = 125000
    service_interval: int | None = 50000


@dataclass
class FakeMetadata:
    device_id: str = "DEV001"
    model: str | None = "MX-3071"
    vendor: str | None = "Sharp"
    location: str | None = "Office-A"
    critical_function: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class FakeSearchResult:
    chunk_id: str = "chunk_1"
    document_id: str = "doc_1"
    text: str = "Fuser error SC542 requires immediate replacement."
    score: float = 0.91
    dense_score: float = 0.88
    sparse_score: float = 0.85
    payload: dict[str, Any] = field(default_factory=lambda: {
        "vendor": "Sharp", "model": "MX-3071", "doc_type": "procedure",
    })


@dataclass
class FakeHealthResult:
    device_id: str = "DEV001"
    health_index: int = 72
    confidence: float = 0.85
    zone: _FakeZone = None
    confidence_zone: _FakeConfZone = None
    factor_contributions: list = field(default_factory=list)
    confidence_reasons: list[str] = field(default_factory=list)
    calculated_at: datetime = datetime(2026, 4, 10, tzinfo=UTC)

    def __post_init__(self) -> None:
        if self.zone is None:
            self.zone = _FakeZone("yellow")
        if self.confidence_zone is None:
            self.confidence_zone = _FakeConfZone("high")


class _FakeZone:
    def __init__(self, v: str) -> None:
        self.value = v


class _FakeConfZone:
    def __init__(self, v: str) -> None:
        self.value = v


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_factor_store():
    fs = MagicMock()
    fs.get_events.return_value = [FakeEvent(), FakeEvent(error_code="SC320")]
    fs.get_resources.return_value = FakeResource()
    fs.count_repetitions.return_value = 3
    fs.list_devices.return_value = ["DEV001", "DEV002", "DEV003"]
    fs.get_device_metadata.side_effect = lambda did: FakeMetadata(
        device_id=did,
        model="MX-3071" if did != "DEV003" else "MX-4071",
        location="Office-A" if did != "DEV003" else "Office-B",
    )
    return fs


def _make_weights():
    from data_io.models import WeightsProfile
    return WeightsProfile(profile_name="test")


def _make_deps(**overrides) -> ToolDependencies:
    kwargs: dict[str, Any] = {
        "factor_store": _make_factor_store(),
        "weights": _make_weights(),
        "searcher": None,
        "llm_client": None,
        "health_cache": {},
    }
    kwargs.update(overrides)
    return ToolDependencies(**kwargs)


@pytest.fixture
def deps():
    return _make_deps()


@pytest.fixture
def registry_with_tools(deps):
    reg = ToolRegistry()
    register_all_tools(reg, deps)
    return reg


# ── 1. Schema validation — all 9 schemas via jsonschema ─────────────────────

TOOL_NAMES = [
    "search_service_docs",
    "classify_error_severity",
    "get_device_events",
    "get_device_resources",
    "count_error_repetitions",
    "calculate_health_index",
    "get_fleet_statistics",
    "find_similar_devices",
    "get_device_history",
    "get_learned_patterns",
]


class TestSchemaValidation:
    """JSON schemas of all 10 tools validate via jsonschema."""

    def test_registry_has_all_10_tools(self, registry_with_tools) -> None:
        assert len(registry_with_tools) == 10
        for name in TOOL_NAMES:
            assert name in registry_with_tools

    @pytest.mark.parametrize("tool_name", TOOL_NAMES)
    def test_schema_is_valid_openai_function(self, registry_with_tools, tool_name) -> None:
        schema = registry_with_tools.get_schema(tool_name)
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == tool_name
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params["type"] == "object"

    @pytest.mark.parametrize("tool_name", TOOL_NAMES)
    def test_schema_parameters_are_valid_json_schema(self, registry_with_tools, tool_name) -> None:
        schema = registry_with_tools.get_schema(tool_name)
        params_schema = schema["function"]["parameters"]
        jsonschema.Draft7Validator.check_schema(params_schema)


# ── 2. Unknown tool → error ──────────────────────────────────────────────────


class TestUnknownTool:
    def test_execute_unknown_tool_returns_error(self, registry_with_tools) -> None:
        result = registry_with_tools.execute("nonexistent_tool", {})
        assert result.success is False
        assert "Unknown tool" in result.error
        assert "nonexistent_tool" not in registry_with_tools

    def test_get_schema_unknown_tool_raises(self, registry_with_tools) -> None:
        from agent.tools.registry import ToolRegistryError
        with pytest.raises(ToolRegistryError, match="Unknown tool"):
            registry_with_tools.get_schema("nonexistent_tool")


# ── 3. Individual tool unit tests with mocked deps ──────────────────────────


class TestSearchServiceDocsTool:
    def test_no_searcher_returns_error(self, deps) -> None:
        tool = SearchServiceDocsTool(deps)
        result = tool.execute({"query": "fuser error"})
        assert result.success is False
        assert "not available" in result.error

    def test_with_searcher_returns_hits(self) -> None:
        searcher = MagicMock()
        searcher.search.return_value = [FakeSearchResult()]
        deps = _make_deps(searcher=searcher)
        tool = SearchServiceDocsTool(deps)
        result = tool.execute({"query": "SC542", "top_k": 3})
        assert result.success is True
        assert result.data["total"] == 1
        assert result.data["hits"][0]["chunk_id"] == "chunk_1"
        searcher.search.assert_called_once()

    def test_invalid_args(self, deps) -> None:
        tool = SearchServiceDocsTool(deps)
        result = tool.execute({})
        assert result.success is False
        assert "Invalid args" in result.error


class TestClassifyErrorSeverityTool:
    def test_no_searcher_returns_fallback(self, deps) -> None:
        tool = ClassifyErrorSeverityTool(deps)
        result = tool.execute({"error_code": "SC542"})
        assert result.success is True
        assert result.data["severity"] == "Medium"
        assert result.data["confidence"] == 0.3

    def test_cache_works(self, deps) -> None:
        tool = ClassifyErrorSeverityTool(deps)
        r1 = tool.execute({"error_code": "SC542"})
        r2 = tool.execute({"error_code": "SC542"})
        assert r1.data == r2.data

    def test_with_rag_but_no_llm_returns_fallback(self) -> None:
        searcher = MagicMock()
        searcher.search.return_value = [FakeSearchResult()]
        deps = _make_deps(searcher=searcher, llm_client=None)
        tool = ClassifyErrorSeverityTool(deps)
        result = tool.execute({"error_code": "SC542"})
        assert result.success is True
        assert result.data["severity"] == "Medium"
        assert "LLM client not available" in result.data["reasoning"]

    def test_invalid_args(self, deps) -> None:
        tool = ClassifyErrorSeverityTool(deps)
        result = tool.execute({})
        assert result.success is False


class TestGetDeviceEventsTool:
    def test_returns_events(self, deps) -> None:
        tool = GetDeviceEventsTool(deps)
        result = tool.execute({"device_id": "DEV001"})
        assert result.success is True
        assert result.data["total"] == 2
        assert result.data["events"][0]["error_code"] == "SC542"
        deps.factor_store.get_events.assert_called_once_with("DEV001", window_days=30)

    def test_custom_window(self, deps) -> None:
        tool = GetDeviceEventsTool(deps)
        tool.execute({"device_id": "DEV001", "window_days": 7})
        deps.factor_store.get_events.assert_called_with("DEV001", window_days=7)

    def test_invalid_args(self, deps) -> None:
        tool = GetDeviceEventsTool(deps)
        result = tool.execute({})
        assert result.success is False


class TestGetDeviceResourcesTool:
    def test_returns_resources(self, deps) -> None:
        tool = GetDeviceResourcesTool(deps)
        result = tool.execute({"device_id": "DEV001"})
        assert result.success is True
        assert result.data["toner_level"] == 45
        assert result.data["mileage"] == 125000

    def test_no_resources(self, deps) -> None:
        deps.factor_store.get_resources.return_value = None
        tool = GetDeviceResourcesTool(deps)
        result = tool.execute({"device_id": "DEV999"})
        assert result.success is True
        assert result.data is None


class TestCountErrorRepetitionsTool:
    def test_returns_count(self, deps) -> None:
        tool = CountErrorRepetitionsTool(deps)
        result = tool.execute({"device_id": "DEV001", "error_code": "SC542"})
        assert result.success is True
        assert result.data["count"] == 3

    def test_invalid_args(self, deps) -> None:
        tool = CountErrorRepetitionsTool(deps)
        result = tool.execute({"device_id": "DEV001"})
        assert result.success is False


class TestCalculateHealthIndexTool:
    def test_calculates_index(self, deps) -> None:
        tool = CalculateHealthIndexTool(deps)
        result = tool.execute({
            "device_id": "DEV001",
            "factors": [
                {
                    "error_code": "SC542",
                    "severity_level": "High",
                    "n_repetitions": 2,
                    "event_timestamp": "2026-04-10T00:00:00+00:00",
                },
            ],
        })
        assert result.success is True
        assert "health_index" in result.data
        assert 1 <= result.data["health_index"] <= 100
        assert result.data["device_id"] == "DEV001"
        assert "DEV001" in deps.health_cache

    def test_empty_factors(self, deps) -> None:
        tool = CalculateHealthIndexTool(deps)
        result = tool.execute({"device_id": "DEV001", "factors": []})
        assert result.success is True
        assert result.data["health_index"] == 100


class TestGetFleetStatisticsTool:
    def test_empty_cache(self, deps) -> None:
        tool = GetFleetStatisticsTool(deps)
        result = tool.execute({})
        assert result.success is True
        assert result.data["total_devices"] == 0

    def test_with_cached_data(self, deps) -> None:
        deps.health_cache["DEV001"] = [FakeHealthResult(device_id="DEV001")]
        deps.health_cache["DEV002"] = [FakeHealthResult(device_id="DEV002", health_index=90)]
        tool = GetFleetStatisticsTool(deps)
        result = tool.execute({})
        assert result.success is True
        assert result.data["total_devices"] == 2


class TestFindSimilarDevicesTool:
    def test_finds_similar_by_errors(self, deps) -> None:
        deps.factor_store.get_events.side_effect = lambda did, **kw: {
            "DEV001": [FakeEvent(device_id="DEV001", error_code="SC542")],
            "DEV002": [FakeEvent(device_id="DEV002", error_code="SC542")],
            "DEV003": [FakeEvent(device_id="DEV003", error_code="XX999")],
        }.get(did, [])
        tool = FindSimilarDevicesTool(deps)
        result = tool.execute({"device_id": "DEV001", "similarity_dim": "errors"})
        assert result.success is True
        similar = result.data["similar"]
        assert any(d["device_id"] == "DEV002" for d in similar)

    def test_no_data_for_target(self, deps) -> None:
        deps.factor_store.get_events.return_value = []
        deps.factor_store.get_device_metadata.return_value = None
        tool = FindSimilarDevicesTool(deps)
        result = tool.execute({"device_id": "MISSING"})
        assert result.success is True
        assert result.data["similar"] == []

    def test_invalid_similarity_dim(self, deps) -> None:
        tool = FindSimilarDevicesTool(deps)
        result = tool.execute({"device_id": "DEV001", "similarity_dim": "invalid"})
        assert result.success is False


class TestGetDeviceHistoryTool:
    def test_empty_history(self, deps) -> None:
        tool = GetDeviceHistoryTool(deps)
        result = tool.execute({"device_id": "DEV001"})
        assert result.success is True
        assert result.data["history"] == []
        assert result.data["total"] == 0

    def test_with_history(self, deps) -> None:
        deps.health_cache["DEV001"] = [FakeHealthResult()]
        tool = GetDeviceHistoryTool(deps)
        result = tool.execute({"device_id": "DEV001"})
        assert result.success is True
        assert result.data["total"] == 1
        assert result.data["history"][0]["health_index"] == 72

    def test_limit(self, deps) -> None:
        deps.health_cache["DEV001"] = [FakeHealthResult(health_index=i + 50) for i in range(20)]
        tool = GetDeviceHistoryTool(deps)
        result = tool.execute({"device_id": "DEV001", "limit": 5})
        assert result.data["total"] == 5


# ── 4. register_all_tools ───────────────────────────────────────────────────


class TestRegisterAllTools:
    def test_registers_exactly_10(self, deps) -> None:
        reg = ToolRegistry()
        register_all_tools(reg, deps)
        assert len(reg) == 10
        assert sorted(reg.list_tools()) == sorted(TOOL_NAMES)

    def test_duplicate_registration_raises(self, deps) -> None:
        from agent.tools.registry import ToolRegistryError
        reg = ToolRegistry()
        register_all_tools(reg, deps)
        with pytest.raises(ToolRegistryError, match="already registered"):
            register_all_tools(reg, deps)
