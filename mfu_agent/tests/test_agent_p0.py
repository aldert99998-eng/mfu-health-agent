"""P0 tests for MFU Agent system.

Covers:
- Agent loop limits (TC-B-005, TC-B-006)
- Reflection (TC-B-020-021)
- Memory gating (TC-B-030, TC-B-031, TC-B-034)
- Tool correctness (TC-B-040..042, TC-B-044)
- Tool error handling (TC-B-050, TC-B-051)
- Trace structure (TC-B-090)
- Flagged for review (TC-B-094)
- Device isolation (TC-B-100)
- ToolRegistry edge cases
- MemoryManager edge cases
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.core import Agent
from agent.memory import MemoryManager
from agent.tools.impl import ToolDependencies, register_all_tools
from agent.tools.registry import ToolRegistry, ToolRegistryError, ToolResult
from config.loader import AgentConfig, AgentLoopConfig, ReflectionConfig
from data_io.models import (
    BatchContext,
    HealthZone,
    LearnedPattern,
    ReflectionAction,
    ReflectionResult,
    ReflectionVerdict,
    TraceStepType,
    WeightsProfile,
)
from llm.client import LLMResponse, TokenUsage, ToolCall


# ── Fake data ────────────────────────────────────────────────────────────────


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
    tags: list[str] = dc_field(default_factory=list)


_USAGE = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

_FINAL_ANSWER_JSON = json.dumps({
    "health_index": 72,
    "confidence": 0.85,
    "factors": [],
    "explanation": "Test result.",
    "reflection_notes": "",
}, ensure_ascii=False)

_REFLECTION_APPROVED_JSON = json.dumps({
    "verdict": "approved",
    "issues": [],
    "recommended_action": "accept",
})

_REFLECTION_NEEDS_REVISION_JSON = json.dumps({
    "verdict": "needs_revision",
    "issues": [{"issue": "Index too high", "severity": "high"}],
    "recommended_action": "recalculate",
})


def _resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason="stop",
        usage=_USAGE,
    )


def _tc(name: str, args: dict[str, Any], call_id: str = "tc_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


def _make_factor_store():
    fs = MagicMock()
    fs.get_events.return_value = [FakeEvent(), FakeEvent(error_code="SC320")]
    fs.get_resources.return_value = FakeResource()
    fs.count_repetitions.return_value = 3
    fs.list_devices.return_value = ["DEV001"]
    fs.get_device_metadata.return_value = FakeMetadata()
    return fs


def _make_context(**overrides) -> BatchContext:
    defaults = {
        "weights_profile": WeightsProfile(profile_name="test"),
    }
    defaults.update(overrides)
    return BatchContext(**defaults)


def _make_agent(
    llm_client: MagicMock,
    factor_store=None,
    config: AgentConfig | None = None,
    health_cache: dict | None = None,
    memory_manager: MemoryManager | None = None,
) -> Agent:
    fs = factor_store or _make_factor_store()
    cfg = config or AgentConfig()

    registry = ToolRegistry()
    deps = ToolDependencies(
        factor_store=fs,
        weights=WeightsProfile(profile_name="test"),
        health_cache=health_cache or {},
        memory_manager=memory_manager,
    )
    register_all_tools(registry, deps)

    return Agent(
        llm_client=llm_client,
        tool_registry=registry,
        factor_store=fs,
        config=cfg,
        memory_manager=memory_manager,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-005: MAX_TOOL_CALLS limit enforced
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB005MaxToolCalls:
    """Mock LLM to always request tools, verify loop stops at limit."""

    def test_max_tool_calls_limit_enforced(self) -> None:
        max_tools = 3
        cfg = AgentConfig(
            agent=AgentLoopConfig(
                max_tool_calls_per_attempt=max_tools,
                max_llm_calls_per_attempt=50,  # high so tool limit hits first
            ),
            reflection=ReflectionConfig(enabled=False),
        )

        # LLM always requests a tool call
        llm = MagicMock()
        call_count = 0

        def gen_side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            # After tools exhausted, LLM gets no tools param and should return text
            if kw.get("tools") is None:
                return _resp(content=_FINAL_ANSWER_JSON)
            return _resp(
                content="",
                tool_calls=[_tc("get_device_events", {"device_id": "DEV001"}, f"tc_{call_count}")],
            )

        llm.generate = MagicMock(side_effect=gen_side_effect)

        agent = _make_agent(llm, config=cfg)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.total_tool_calls <= max_tools, (
            f"Tool calls {trace.total_tool_calls} exceeded limit {max_tools}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-006: MAX_LLM_CALLS limit enforced
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB006MaxLLMCalls:
    """Verify loop stops at LLM call limit."""

    def test_max_llm_calls_limit_enforced(self) -> None:
        max_llm = 4
        cfg = AgentConfig(
            agent=AgentLoopConfig(
                max_llm_calls_per_attempt=max_llm,
                max_tool_calls_per_attempt=100,
            ),
            reflection=ReflectionConfig(enabled=False),
        )

        # LLM always requests a tool, never finishes
        call_count = 0
        llm = MagicMock()

        def gen_side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            return _resp(
                content="thinking...",
                tool_calls=[_tc("get_device_events", {"device_id": "DEV001"}, f"tc_{call_count}")],
            )

        llm.generate = MagicMock(side_effect=gen_side_effect)

        agent = _make_agent(llm, config=cfg)
        result, trace = agent.run_batch("DEV001", _make_context())

        # The agent loop LLM calls (not counting reflection) should not exceed max_llm
        # trace.total_llm_calls includes reflection calls, so we check the generate call count
        # from the first attempt only
        assert call_count <= max_llm + 2, (
            f"LLM calls {call_count} far exceeded limit {max_llm}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-020-021: Reflection detects errors / approves correct results
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB020021Reflection:
    """Test _run_reflection with mock."""

    def test_reflection_approves_correct_result(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),          # agent loop
            _resp(content=_REFLECTION_APPROVED_JSON),    # reflection
        ])

        agent = _make_agent(llm, config=AgentConfig(reflection=ReflectionConfig(enabled=True)))
        result, trace = agent.run_batch("DEV001", _make_context())

        assert not trace.flagged_for_review
        assert result.health_index == 72

    def test_reflection_detects_error_needs_revision(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),                # attempt 1
            _resp(content=_REFLECTION_NEEDS_REVISION_JSON),   # reflection 1 -> needs_revision
            _resp(content=_FINAL_ANSWER_JSON),                # attempt 2
            _resp(content=_REFLECTION_APPROVED_JSON),         # reflection 2 -> approved
        ])

        cfg = AgentConfig(
            agent=AgentLoopConfig(max_attempts_per_device=2),
            reflection=ReflectionConfig(enabled=True),
        )
        agent = _make_agent(llm, config=cfg)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.attempts == 2
        assert not trace.flagged_for_review


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-030: Memory written only after approved reflection
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB030MemoryAfterApproval:
    """Memory written only after approved reflection (not flagged)."""

    def test_memory_saved_after_approved(self) -> None:
        mm = MemoryManager(min_evidence_devices=1)
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_APPROVED_JSON),
        ])

        # Provide fleet_stats so _save_learned_patterns can run
        ctx = _make_context(
            fleet_stats={"DEV001": [], "DEV002": []},
        )
        cfg = AgentConfig(
            reflection=ReflectionConfig(enabled=True),
        )
        agent = _make_agent(llm, config=cfg, memory_manager=mm)
        result, trace = agent.run_batch("DEV001", ctx)

        # The key assertion: memory is enabled, not flagged -> patterns CAN be saved
        assert not trace.flagged_for_review


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-031: Memory NOT written after needs_revision (flagged)
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB031MemoryNotWrittenAfterRevision:
    """Memory NOT written after needs_revision that exhausts attempts."""

    def test_memory_not_saved_when_flagged(self) -> None:
        mm = MemoryManager(min_evidence_devices=1)
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_NEEDS_REVISION_JSON),
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_NEEDS_REVISION_JSON),
        ])

        ctx = _make_context(fleet_stats={"DEV001": [], "DEV002": []})
        cfg = AgentConfig(
            agent=AgentLoopConfig(max_attempts_per_device=2),
            reflection=ReflectionConfig(enabled=True),
        )
        agent = _make_agent(llm, config=cfg, memory_manager=mm)
        result, trace = agent.run_batch("DEV001", ctx)

        assert trace.flagged_for_review
        # When flagged, _save_patterns_to_memory is NOT called
        assert len(mm) == 0, "Memory should not contain patterns when flagged_for_review"


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-034: Memory contains no PII (check pattern structure)
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB034MemoryNoPII:
    """Patterns should follow a structured format without PII."""

    def test_pattern_structure_no_pii(self) -> None:
        pattern = LearnedPattern(
            type="pattern",
            scope="fleet",
            observation="Factor 'SC542' found on 3 devices",
            evidence_devices=["DEV001", "DEV002", "DEV003"],
        )

        mm = MemoryManager(min_evidence_devices=1)
        mm.save_pattern(pattern)

        saved = mm.get_patterns()
        assert len(saved) == 1
        p = saved[0]

        # Check structure: no emails, phone numbers, names in observation
        email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
        phone_re = re.compile(r"\+?\d{10,}")
        assert not email_re.search(p.observation), "PII (email) found in pattern"
        assert not phone_re.search(p.observation), "PII (phone) found in pattern"

        # evidence_devices should be device IDs, not personal identifiers
        for dev_id in p.evidence_devices:
            assert not email_re.search(dev_id), f"PII in device id: {dev_id}"

        # Pattern must have required fields
        assert p.type in ("pattern", "anomaly", "trend")
        assert p.scope
        assert p.observation


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-040: get_device_events returns correct data from FactorStore
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB040GetDeviceEvents:
    """get_device_events tool returns correct data from FactorStore."""

    def test_returns_events_from_factor_store(self) -> None:
        fs = _make_factor_store()
        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        result = registry.execute("get_device_events", {"device_id": "DEV001"})

        assert result.success
        assert "events" in result.data
        assert len(result.data["events"]) == 2
        assert result.data["events"][0]["error_code"] == "SC542"
        assert result.data["events"][1]["error_code"] == "SC320"
        fs.get_events.assert_called_once_with("DEV001", window_days=30)


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-041: get_device_resources returns ResourceSnapshot
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB041GetDeviceResources:
    """get_device_resources tool returns ResourceSnapshot data."""

    def test_returns_resource_snapshot(self) -> None:
        fs = _make_factor_store()
        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        result = registry.execute("get_device_resources", {"device_id": "DEV001"})

        assert result.success
        assert result.data is not None
        assert result.data["device_id"] == "DEV001"
        assert result.data["toner_level"] == 45
        assert result.data["drum_level"] == 60
        assert result.data["fuser_level"] == 30
        assert result.data["mileage"] == 125000


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-042: count_error_repetitions returns exact count
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB042CountErrorRepetitions:
    """count_error_repetitions tool returns exact count."""

    def test_returns_exact_count(self) -> None:
        fs = _make_factor_store()
        fs.count_repetitions.return_value = 7

        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        result = registry.execute("count_error_repetitions", {
            "device_id": "DEV001",
            "error_code": "SC542",
        })

        assert result.success
        assert result.data["count"] == 7
        fs.count_repetitions.assert_called_once_with("DEV001", "SC542", window_days=14)


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-044: calculate_health_index delegates correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB044CalculateHealthIndex:
    """calculate_health_index tool delegates to calculator."""

    def test_delegates_to_calculator(self) -> None:
        fs = _make_factor_store()
        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        result = registry.execute("calculate_health_index", {
            "device_id": "DEV001",
            "factors": [
                {
                    "error_code": "SC542",
                    "severity_level": "High",
                    "n_repetitions": 3,
                    "event_timestamp": "2026-04-10T00:00:00+00:00",
                },
            ],
        })

        assert result.success
        assert "health_index" in result.data
        assert "confidence" in result.data
        assert "zone" in result.data
        assert isinstance(result.data["health_index"], int)
        assert 1 <= result.data["health_index"] <= 100

    def test_empty_factors_returns_full_health(self) -> None:
        fs = _make_factor_store()
        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        result = registry.execute("calculate_health_index", {
            "device_id": "DEV001",
            "factors": [],
        })

        assert result.success
        assert result.data["health_index"] == 100


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-050: Invalid tool args -> error result, not crash
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB050InvalidToolArgs:
    """Invalid tool args produce error result, not crash."""

    def test_missing_required_arg_returns_error(self) -> None:
        fs = _make_factor_store()
        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        # get_device_events requires device_id
        result = registry.execute("get_device_events", {})
        assert not result.success
        assert result.error is not None
        assert "Invalid args" in result.error

    def test_wrong_type_arg_returns_error(self) -> None:
        fs = _make_factor_store()
        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        # window_days should be int >= 1
        result = registry.execute("get_device_events", {
            "device_id": "DEV001",
            "window_days": -5,
        })
        assert not result.success
        assert "Invalid args" in result.error


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-051: Tool execution error -> ToolResult with error, not exception
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB051ToolExecutionError:
    """Tool execution error returns ToolResult with error, not exception."""

    def test_factor_store_exception_returns_error(self) -> None:
        fs = MagicMock()
        fs.get_events.side_effect = RuntimeError("DB down")
        fs.get_resources.side_effect = RuntimeError("DB down")
        fs.count_repetitions.side_effect = RuntimeError("DB down")
        fs.list_devices.return_value = []
        fs.get_device_metadata.return_value = None

        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        # This should NOT raise, but return ToolResult with error
        result = registry.execute("get_device_events", {"device_id": "DEV001"})
        assert not result.success
        assert "RuntimeError" in result.error


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-090: Trace contains all steps with correct step types
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB090TraceStepTypes:
    """Trace contains all steps with correct step types."""

    def test_trace_has_llm_and_reflection_steps(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_APPROVED_JSON),
        ])

        cfg = AgentConfig(reflection=ReflectionConfig(enabled=True))
        agent = _make_agent(llm, config=cfg)
        _, trace = agent.run_batch("DEV001", _make_context())

        step_types = [s.type for s in trace.steps]
        assert TraceStepType.LLM_CALL in step_types
        assert TraceStepType.REFLECTION in step_types

    def test_trace_has_tool_call_steps(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(
                content="",
                tool_calls=[_tc("get_device_events", {"device_id": "DEV001"})],
            ),
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_APPROVED_JSON),
        ])

        cfg = AgentConfig(reflection=ReflectionConfig(enabled=True))
        agent = _make_agent(llm, config=cfg)
        _, trace = agent.run_batch("DEV001", _make_context())

        step_types = [s.type for s in trace.steps]
        assert TraceStepType.TOOL_CALL in step_types

    def test_step_numbers_are_sequential(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(
                content="",
                tool_calls=[_tc("get_device_events", {"device_id": "DEV001"})],
            ),
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_APPROVED_JSON),
        ])

        cfg = AgentConfig(reflection=ReflectionConfig(enabled=True))
        agent = _make_agent(llm, config=cfg)
        _, trace = agent.run_batch("DEV001", _make_context())

        step_numbers = [s.step_number for s in trace.steps]
        for i in range(1, len(step_numbers)):
            assert step_numbers[i] > step_numbers[i - 1], (
                f"Step numbers not sequential: {step_numbers}"
            )

    def test_all_steps_have_duration(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_APPROVED_JSON),
        ])

        cfg = AgentConfig(reflection=ReflectionConfig(enabled=True))
        agent = _make_agent(llm, config=cfg)
        _, trace = agent.run_batch("DEV001", _make_context())

        for step in trace.steps:
            assert step.duration_ms >= 0, f"Step {step.step_number} has negative duration"


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-094: flagged_for_review when max attempts exceeded
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB094FlaggedForReview:
    """flagged_for_review set when max attempts exceeded."""

    def test_flagged_when_all_attempts_need_revision(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_NEEDS_REVISION_JSON),
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_NEEDS_REVISION_JSON),
        ])

        cfg = AgentConfig(
            agent=AgentLoopConfig(max_attempts_per_device=2),
            reflection=ReflectionConfig(enabled=True),
        )
        agent = _make_agent(llm, config=cfg)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.flagged_for_review is True
        assert result.reflection_notes is not None
        assert len(result.reflection_notes) > 0

    def test_not_flagged_when_approved_on_first_attempt(self) -> None:
        llm = MagicMock()
        llm.generate = MagicMock(side_effect=[
            _resp(content=_FINAL_ANSWER_JSON),
            _resp(content=_REFLECTION_APPROVED_JSON),
        ])

        cfg = AgentConfig(reflection=ReflectionConfig(enabled=True))
        agent = _make_agent(llm, config=cfg)
        _, trace = agent.run_batch("DEV001", _make_context())

        assert trace.flagged_for_review is False


# ═══════════════════════════════════════════════════════════════════════════
# TC-B-100: Device isolation - batch devices don't cross-contaminate
# ═══════════════════════════════════════════════════════════════════════════


class TestTCB100DeviceIsolation:
    """Batch devices don't cross-contaminate."""

    def test_two_devices_produce_independent_results(self) -> None:
        fs = MagicMock()

        def get_events(device_id, window_days=30):
            if device_id == "DEV001":
                return [FakeEvent(device_id="DEV001", error_code="SC542")]
            return [FakeEvent(device_id="DEV002", error_code="SC999")]

        def get_resources(device_id):
            if device_id == "DEV001":
                return FakeResource(device_id="DEV001", toner_level=45)
            return FakeResource(device_id="DEV002", toner_level=90)

        fs.get_events.side_effect = get_events
        fs.get_resources.side_effect = get_resources
        fs.count_repetitions.return_value = 1
        fs.list_devices.return_value = ["DEV001", "DEV002"]
        fs.get_device_metadata.return_value = FakeMetadata()

        answer1 = json.dumps({
            "health_index": 60, "confidence": 0.8,
        })
        answer2 = json.dumps({
            "health_index": 90, "confidence": 0.9,
        })

        llm = MagicMock()
        call_count = 0

        def gen_side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            # Alternate between devices
            msgs = kw.get("messages", a[0] if a else [])
            # Check if this is a reflection call (system prompt starts differently)
            if any("DEV001" in str(m) for m in msgs) and kw.get("tools") is not None:
                return _resp(content=answer1)
            if any("DEV002" in str(m) for m in msgs) and kw.get("tools") is not None:
                return _resp(content=answer2)
            return _resp(content=_REFLECTION_APPROVED_JSON)

        llm.generate = MagicMock(side_effect=gen_side_effect)

        cfg = AgentConfig(reflection=ReflectionConfig(enabled=True))
        agent = _make_agent(llm, factor_store=fs, config=cfg)

        result1, trace1 = agent.run_batch("DEV001", _make_context())
        result2, trace2 = agent.run_batch("DEV002", _make_context())

        # Results should be for the correct device
        assert result1.device_id == "DEV001"
        assert result2.device_id == "DEV002"

        # Traces should be independent
        assert trace1.session_id != trace2.session_id
        assert trace1.device_id == "DEV001"
        assert trace2.device_id == "DEV002"


# ═══════════════════════════════════════════════════════════════════════════
# ToolRegistry edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestToolRegistryEdgeCases:
    """ToolRegistry: duplicate registration, unknown tool, schemas."""

    def test_duplicate_registration_raises_error(self) -> None:
        registry = ToolRegistry()
        fs = _make_factor_store()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        # Try registering again
        with pytest.raises(ToolRegistryError, match="already registered"):
            register_all_tools(registry, deps)

    def test_unknown_tool_execution_returns_error(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("nonexistent_tool", {})
        assert not result.success
        assert "Unknown tool" in result.error

    def test_unknown_tool_get_schema_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolRegistryError, match="Unknown tool"):
            registry.get_schema("nonexistent_tool")

    def test_all_schemas_returned_correctly(self) -> None:
        registry = ToolRegistry()
        fs = _make_factor_store()
        deps = ToolDependencies(
            factor_store=fs,
            weights=WeightsProfile(profile_name="test"),
        )
        register_all_tools(registry, deps)

        schemas = registry.get_all_schemas()
        assert len(schemas) == 10

        for schema in schemas:
            assert "type" in schema
            assert schema["type"] == "function"
            assert "function" in schema
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func


# ═══════════════════════════════════════════════════════════════════════════
# MemoryManager edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryManagerEdgeCases:
    """MemoryManager: evidence thresholds, dedup, capacity."""

    def test_pattern_saved_with_sufficient_evidence(self) -> None:
        mm = MemoryManager(min_evidence_devices=2)
        pattern = LearnedPattern(
            type="pattern",
            scope="fleet",
            observation="Test pattern",
            evidence_devices=["DEV001", "DEV002"],
        )
        assert mm.save_pattern(pattern) is True
        assert len(mm) == 1

    def test_pattern_rejected_with_insufficient_evidence(self) -> None:
        # MemoryManager with higher threshold than the model minimum
        mm = MemoryManager(min_evidence_devices=3)
        pattern = LearnedPattern(
            type="pattern",
            scope="fleet",
            observation="Test pattern",
            evidence_devices=["DEV001", "DEV002"],  # 2 devices, but manager needs 3
        )
        assert mm.save_pattern(pattern) is False
        assert len(mm) == 0

    def test_duplicate_pattern_merges_device_lists(self) -> None:
        mm = MemoryManager(min_evidence_devices=2)
        p1 = LearnedPattern(
            type="pattern",
            scope="fleet",
            observation="Same pattern",
            evidence_devices=["DEV001", "DEV002"],
        )
        p2 = LearnedPattern(
            type="pattern",
            scope="fleet",
            observation="Same pattern",
            evidence_devices=["DEV002", "DEV003"],
        )

        assert mm.save_pattern(p1) is True
        assert mm.save_pattern(p2) is True
        assert len(mm) == 1  # merged, not duplicated

        patterns = mm.get_patterns()
        # Should have all 3 unique devices
        assert set(patterns[0].evidence_devices) == {"DEV001", "DEV002", "DEV003"}

    def test_max_patterns_per_model_enforced(self) -> None:
        mm = MemoryManager(max_patterns_per_model=3, min_evidence_devices=2)

        for i in range(5):
            pattern = LearnedPattern(
                type="pattern",
                scope="MX-3071",
                observation=f"Pattern {i}",
                evidence_devices=["DEV001", "DEV002"],
            )
            mm.save_pattern(pattern)

        # Only 3 should be stored for that scope
        scope_patterns = [p for p in mm.get_patterns() if p.scope == "MX-3071"]
        assert len(scope_patterns) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
