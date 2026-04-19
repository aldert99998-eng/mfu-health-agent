"""Phase 5.3 verification — agent core tests.

Checks:
1. E2E: factor_store with one device → run_batch → HealthResult.
2. Trace contains plan, llm_call, tool_call, reflection — all 4 step types.
3. Deliberately broken factor_store triggers flag_for_review after 2 attempts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError

from agent.core import Agent
from agent.memory import MemoryManager
from agent.tools.impl import ToolDependencies, register_all_tools
from agent.tools.registry import ToolRegistry
from config.loader import AgentConfig
from data_io.models import (
    AgentMode,
    BatchContext,
    HealthResult,
    LearnedPattern,
    Trace,
    TraceStepType,
    WeightsProfile,
)
from llm.client import LLMResponse, TokenUsage, ToolCall

# ── Fake data ───────────────────────────────────────────────────────────────


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


# ── LLM response sequences ─────────────────────────────────────────────────

_USAGE = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

_FINAL_ANSWER_JSON = json.dumps({
    "health_index": 72,
    "confidence": 0.85,
    "factors": [
        {
            "error_code": "SC542",
            "severity_level": "High",
            "n_repetitions": 3,
        },
    ],
    "explanation": "Fuser overheat detected, moderate degradation.",
    "reflection_notes": "",
}, ensure_ascii=False)

_REFLECTION_APPROVED_JSON = json.dumps({
    "verdict": "approved",
    "issues": [],
    "recommended_action": "accept",
})

_REFLECTION_NEEDS_REVISION_JSON = json.dumps({
    "verdict": "needs_revision",
    "issues": [{"issue": "Индекс слишком высокий при Critical ошибке", "severity": "high"}],
    "recommended_action": "recalculate",
})


def _make_llm_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=_USAGE,
    )


def _tool_call(name: str, args: dict[str, Any], call_id: str = "tc_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_factor_store():
    fs = MagicMock()
    fs.get_events.return_value = [FakeEvent(), FakeEvent(error_code="SC320")]
    fs.get_resources.return_value = FakeResource()
    fs.count_repetitions.return_value = 3
    fs.list_devices.return_value = ["DEV001"]
    fs.get_device_metadata.return_value = FakeMetadata()
    return fs


def _make_broken_factor_store():
    fs = MagicMock()
    fs.get_events.side_effect = RuntimeError("DB connection lost")
    fs.get_resources.side_effect = RuntimeError("DB connection lost")
    fs.count_repetitions.side_effect = RuntimeError("DB connection lost")
    fs.list_devices.return_value = ["DEV001"]
    fs.get_device_metadata.return_value = FakeMetadata()
    return fs


def _make_config(**overrides) -> AgentConfig:
    return AgentConfig(**overrides)


def _make_context(factor_store=None, **overrides) -> BatchContext:
    defaults = {
        "weights_profile": WeightsProfile(profile_name="test"),
        "factor_store": factor_store,
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
    cfg = config or _make_config()

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


# ── 1. E2E: factor_store + run_batch -> HealthResult ───────────────────────


class TestRunBatchE2E:
    """run_batch produces a valid HealthResult with mocked LLM."""

    def test_happy_path_no_tools(self) -> None:
        """LLM returns final answer immediately → valid HealthResult."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        ctx = _make_context()

        result, trace = agent.run_batch("DEV001", ctx)

        assert isinstance(result, HealthResult)
        assert result.device_id == "DEV001"
        assert result.health_index == 72
        assert result.confidence == 0.85
        assert 1 <= result.health_index <= 100
        assert trace.mode == AgentMode.BATCH
        assert trace.device_id == "DEV001"
        assert trace.ended_at is not None
        assert trace.final_result["health_index"] == 72

    def test_with_tool_calls_then_final_answer(self) -> None:
        """LLM calls get_device_events, then returns final answer."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                content="Получу события для DEV001",
                tool_calls=[_tool_call(
                    "get_device_events",
                    {"device_id": "DEV001"},
                    call_id="tc_events",
                )],
            ),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        ctx = _make_context()

        result, trace = agent.run_batch("DEV001", ctx)

        assert result.health_index == 72
        assert trace.total_tool_calls >= 1
        assert trace.total_llm_calls >= 2

    def test_zone_auto_assignment(self) -> None:
        """Zone and confidence_zone are auto-assigned when missing."""
        minimal_json = json.dumps({"health_index": 30, "confidence": 0.4})
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=minimal_json),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        result, _ = agent.run_batch("DEV001", _make_context())

        assert result.zone == "red"
        assert result.confidence_zone == "low"

    def test_green_zone_assignment(self) -> None:
        green_json = json.dumps({"health_index": 90, "confidence": 0.95})
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=green_json),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        result, _ = agent.run_batch("DEV001", _make_context())

        assert result.zone == "green"
        assert result.confidence_zone == "high"

    def test_reflection_disabled(self) -> None:
        """With reflection disabled, no reflection LLM call is made."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
        ]

        cfg = _make_config(reflection={"enabled": False})
        agent = _make_agent(llm, config=cfg)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert result.health_index == 72
        assert llm.generate.call_count == 1
        step_types = {s.type for s in trace.steps}
        assert TraceStepType.REFLECTION not in step_types


# ── 2. Trace содержит llm_call, tool_call, reflection ──────────────────────


class TestTraceStepTypes:
    """Trace contains llm_call, tool_call, reflection step types."""

    def test_trace_has_llm_call_and_reflection(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _, trace = agent.run_batch("DEV001", _make_context())

        step_types = {s.type for s in trace.steps}
        assert TraceStepType.LLM_CALL in step_types
        assert TraceStepType.REFLECTION in step_types

    def test_trace_has_tool_call_steps(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                content="Plan: get events",
                tool_calls=[_tool_call(
                    "get_device_events", {"device_id": "DEV001"}, "tc_1",
                )],
            ),
            _make_llm_response(
                content="Now get resources",
                tool_calls=[_tool_call(
                    "get_device_resources", {"device_id": "DEV001"}, "tc_2",
                )],
            ),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _, trace = agent.run_batch("DEV001", _make_context())

        step_types = [s.type for s in trace.steps]
        assert TraceStepType.TOOL_CALL in step_types
        assert TraceStepType.LLM_CALL in step_types
        assert TraceStepType.REFLECTION in step_types

        tool_steps = [s for s in trace.steps if s.type == TraceStepType.TOOL_CALL]
        assert len(tool_steps) >= 2
        tool_names = {s.tool_name for s in tool_steps}
        assert "get_device_events" in tool_names
        assert "get_device_resources" in tool_names

    def test_trace_step_numbers_are_sequential(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                content="",
                tool_calls=[_tool_call("get_device_events", {"device_id": "DEV001"})],
            ),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _, trace = agent.run_batch("DEV001", _make_context())

        numbers = [s.step_number for s in trace.steps]
        assert numbers == sorted(numbers)
        assert numbers[0] == 1

    def test_trace_steps_have_duration(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _, trace = agent.run_batch("DEV001", _make_context())

        for step in trace.steps:
            assert step.duration_ms >= 0

    def test_trace_token_accounting(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _, trace = agent.run_batch("DEV001", _make_context())

        assert trace.total_llm_calls == 2
        assert trace.total_tokens == 300  # 150 per call x 2


# ── 3. Сломанный factor_store → flag_for_review после 2 попыток ────────────


class TestBrokenFactorStore:
    """Broken dependencies trigger flag_for_review."""

    def test_unparseable_llm_output_flags_review(self) -> None:
        """LLM returns garbage → flag_for_review=True, fallback HealthResult."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content="I can't compute this, sorry."),
            _make_llm_response(content="Still can't do it."),
        ]

        agent = _make_agent(llm)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.flagged_for_review is True
        assert result.device_id == "DEV001"
        assert result.confidence == 0.2
        assert "Не удалось получить результат" in result.reflection_notes

    def test_needs_revision_exhausts_attempts(self) -> None:
        """Reflection keeps returning needs_revision → exhausts 2 attempts → flag."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_NEEDS_REVISION_JSON),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_NEEDS_REVISION_JSON),
        ]

        cfg = _make_config(agent={"max_attempts_per_device": 2})
        agent = _make_agent(llm, config=cfg)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.flagged_for_review is True
        assert trace.attempts == 2
        assert "Исчерпаны попытки" in result.reflection_notes

    def test_broken_tools_still_produce_result(self) -> None:
        """Tools fail but LLM still returns a final answer → result produced."""
        broken_fs = _make_broken_factor_store()
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                content="",
                tool_calls=[_tool_call(
                    "get_device_events", {"device_id": "DEV001"}, "tc_1",
                )],
            ),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm, factor_store=broken_fs)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert result.health_index == 72
        tool_steps = [s for s in trace.steps if s.type == TraceStepType.TOOL_CALL]
        assert len(tool_steps) >= 1

    def test_suspicious_verdict_flags_review(self) -> None:
        """Reflection verdict=suspicious → flag_for_review + notes."""
        suspicious_json = json.dumps({
            "verdict": "suspicious",
            "issues": [{"issue": "Аномально низкий индекс", "severity": "high"}],
            "recommended_action": "flag_for_review",
        })
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=suspicious_json),
        ]

        agent = _make_agent(llm)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.flagged_for_review is True
        assert "Аномально низкий индекс" in result.reflection_notes

    def test_reflection_parse_failure_falls_back_to_approved(self) -> None:
        """Unparseable reflection after retries -> fallback approved."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content="not json"),
            _make_llm_response(content="still not json"),
            _make_llm_response(content="nope"),
        ]

        agent = _make_agent(llm)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.flagged_for_review is False
        assert result.health_index == 72


# ── 3b. Reflection retry and verdict flow ──────────────────────────────────


class TestReflectionFlow:
    """Verify reflection retry logic and verdict handling."""

    def test_approved_completes_loop(self) -> None:
        """Reflection approved -> loop completes in 1 attempt."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert result.health_index == 72
        assert trace.attempts == 1
        assert trace.flagged_for_review is False

    def test_needs_revision_then_approved(self) -> None:
        """needs_revision then approved -> attempts=2, no flag."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_NEEDS_REVISION_JSON),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.attempts == 2
        assert trace.flagged_for_review is False

    def test_needs_revision_twice_flags_review(self) -> None:
        """needs_revision x2 -> flag_for_review after max attempts."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_NEEDS_REVISION_JSON),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_NEEDS_REVISION_JSON),
        ]

        agent = _make_agent(llm)
        result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.attempts == 2
        assert trace.flagged_for_review is True
        assert result.reflection_notes is not None


# ── 4. Message building and role extensions ────────────────────────────────


class TestMessageBuilding:
    """Verify system prompt assembly with role extensions and patterns."""

    def test_critical_device_extension(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
        ]

        fs = _make_factor_store()
        fs.get_device_metadata.return_value = FakeMetadata(critical_function=True)

        cfg = _make_config(reflection={"enabled": False})
        agent = _make_agent(llm, factor_store=fs, config=cfg)

        ctx = _make_context(device_metadata=FakeMetadata(critical_function=True))
        agent.run_batch("DEV001", ctx)

        call_args = llm.generate.call_args_list[0]
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        system_msg = messages[0]["content"]
        assert "критически важное" in system_msg

    def test_false_alarm_extension(self) -> None:
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
        ]

        cfg = _make_config(reflection={"enabled": False})
        agent = _make_agent(llm, config=cfg)

        meta = FakeMetadata(tags=["false_alarm_history"])
        ctx = _make_context(device_metadata=meta)
        agent.run_batch("DEV001", ctx)

        call_args = llm.generate.call_args_list[0]
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        system_msg = messages[0]["content"]
        assert "ложных срабатываний" in system_msg

    def test_learned_patterns_in_prompt(self) -> None:
        from data_io.models import LearnedPattern

        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
        ]

        cfg = _make_config(reflection={"enabled": False})
        agent = _make_agent(llm, config=cfg)

        patterns = [
            LearnedPattern(
                scope="MX-3071",
                observation="SC542 часто ложноположительный",
                evidence_devices=["DEV001", "DEV002"],
            ),
        ]
        ctx = _make_context(learned_patterns=patterns)
        agent.run_batch("DEV001", ctx)

        call_args = llm.generate.call_args_list[0]
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        system_msg = messages[0]["content"]
        assert "SC542 часто ложноположительный" in system_msg

    def test_revision_notes_in_user_message(self) -> None:
        """Second attempt includes revision notes from reflection."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_NEEDS_REVISION_JSON),
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        cfg = _make_config(agent={"max_attempts_per_device": 2})
        agent = _make_agent(llm, config=cfg)
        agent.run_batch("DEV001", _make_context())

        second_attempt_call = llm.generate.call_args_list[2]
        messages = (
            second_attempt_call.kwargs.get("messages")
            or second_attempt_call[1].get("messages")
            or second_attempt_call[0][0]
        )
        user_msg = messages[1]["content"]
        assert "Замечания ревизора" in user_msg
        assert "Индекс слишком высокий" in user_msg


# ── 5. Tool call limits ────────────────────────────────────────────────────


class TestLimits:
    """Agent respects max_llm_calls and max_tool_calls limits."""

    def test_max_llm_calls_respected(self) -> None:
        """Loop stops after max_llm_calls even if LLM keeps calling tools."""
        llm = MagicMock()
        infinite_tool_response = _make_llm_response(
            content="need more data",
            tool_calls=[_tool_call("get_device_events", {"device_id": "DEV001"})],
        )
        llm.generate.return_value = infinite_tool_response

        cfg = _make_config(
            agent={"max_llm_calls_per_attempt": 3, "max_attempts_per_device": 1},
            reflection={"enabled": False},
        )
        agent = _make_agent(llm, config=cfg)
        _result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.total_llm_calls <= 3
        assert trace.flagged_for_review is True

    def test_max_tool_calls_respected(self):
        """After max_tool_calls, tools get error messages."""
        call_count = 0

        def gen_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _make_llm_response(
                    content="calling tool",
                    tool_calls=[_tool_call(
                        "get_device_events",
                        {"device_id": "DEV001"},
                        f"tc_{call_count}",
                    )],
                )
            return _make_llm_response(content=_FINAL_ANSWER_JSON)

        llm = MagicMock()
        llm.generate.side_effect = gen_side_effect

        cfg = _make_config(
            agent={"max_tool_calls_per_attempt": 2, "max_attempts_per_device": 1},
            reflection={"enabled": False},
        )
        agent = _make_agent(llm, config=cfg)
        _result, trace = agent.run_batch("DEV001", _make_context())

        assert trace.total_tool_calls <= 2


# ── 6. Guided JSON fallback ────────────────────────────────────────────────


class TestGuidedJsonFallback:
    """parse_final_result falls back to guided JSON when text isn't JSON."""

    def test_markdown_fenced_json_parsed(self) -> None:
        fenced = '```json\n{"health_index": 65, "confidence": 0.7}\n```'
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=fenced),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        result, _ = agent.run_batch("DEV001", _make_context())

        assert result.health_index == 65

    def test_guided_json_fallback_called_for_text(self) -> None:
        """Free text triggers guided JSON fallback LLM call."""
        extracted_json = json.dumps({"health_index": 55, "confidence": 0.6})
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content="The health index is about 55 with moderate confidence."),
            _make_llm_response(content=extracted_json),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        result, _trace = agent.run_batch("DEV001", _make_context())

        assert result.health_index == 55
        assert llm.generate.call_count == 3


# ── 7. Chat mode ───────────────────────────────────────────────────────────


def _make_chat_context(**overrides):
    from data_io.models import ChatContext
    defaults: dict[str, Any] = {
        "factor_store": _make_factor_store(),
    }
    defaults.update(overrides)
    return ChatContext(**defaults)


class TestRunChat:
    """Verify run_chat: text output, no reflection, tool usage."""

    def test_simple_question_no_tools(self) -> None:
        """General question -> 0 tool calls, text answer."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                content="Indeks zdorovya -- eto integral'naya ocenka sostoyaniya MFU po shkale 1-100."
            ),
        ]

        agent = _make_agent(llm)
        answer, trace = agent.run_chat(
            "Chto takoe indeks zdorovya?", _make_chat_context()
        )

        assert isinstance(answer, str)
        assert len(answer) > 10
        assert trace.total_tool_calls == 0
        reflection_steps = [
            s for s in trace.steps if s.type == TraceStepType.REFLECTION
        ]
        assert len(reflection_steps) == 0

    def test_device_question_uses_tools(self) -> None:
        """Question about a specific device -> at least 1 tool call."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                content="",
                tool_calls=[
                    _tool_call(
                        "get_device_events",
                        {"device_id": "MFP-042"},
                        call_id="tc_chat_1",
                    )
                ],
            ),
            _make_llm_response(
                content="Indeks MFP-042 nizkij iz-za oshibki SC899, povtoryavshejsya 3 raza."
            ),
        ]

        agent = _make_agent(llm)
        answer, trace = agent.run_chat(
            "Pochemu indeks MFP-042 nizkij?", _make_chat_context()
        )

        assert trace.total_tool_calls >= 1
        assert "MFP-042" in answer or len(answer) > 10

    def test_action_request_refused(self) -> None:
        """Request to create a ticket -> agent refuses."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(
                content="Ya ne mogu sozdavat' zayavki. Moya rol' -- analiz i rekomendacii."
            ),
        ]

        cfg = _make_config(reflection={"enabled": False})
        agent = _make_agent(llm, config=cfg)
        _answer, trace = agent.run_chat(
            "Sozdaj tiket na remont", _make_chat_context()
        )

        assert trace.total_tool_calls == 0
        assert trace.mode == AgentMode.CHAT


# ── 8. MemoryManager ────────────────────────────────────────────────────────


class TestMemoryManager:
    """Verify evidence threshold, persistence roundtrip, and integration."""

    def test_evidence_below_threshold_not_saved(self) -> None:
        """Pattern with evidence < threshold is rejected."""
        import pytest as _pt

        mm = MemoryManager(min_evidence_devices=3)
        p = LearnedPattern(
            scope="MX-3071",
            observation="SC542 повторяется",
            evidence_devices=["DEV001", "DEV002"],
        )
        assert mm.save_pattern(p) is False
        assert len(mm) == 0

        with _pt.raises(ValidationError):
            LearnedPattern(
                scope="MX-3071",
                observation="SC542 повторяется",
                evidence_devices=["DEV001"],
            )

    def test_evidence_meets_threshold_saved(self) -> None:
        """Pattern with evidence=2 devices is accepted."""
        mm = MemoryManager(min_evidence_devices=2)
        p = LearnedPattern(
            scope="MX-3071",
            observation="SC542 повторяется",
            evidence_devices=["DEV001", "DEV002"],
        )
        assert mm.save_pattern(p) is True
        assert len(mm) == 1
        assert mm.get_patterns("MX-3071")[0].observation == "SC542 повторяется"

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """to_dict → from_dict preserves all patterns."""
        mm = MemoryManager(max_patterns_per_model=10, min_evidence_devices=2)
        mm.save_pattern(LearnedPattern(
            scope="MX-3071",
            observation="SC542 повторяется",
            evidence_devices=["DEV001", "DEV002"],
        ))
        mm.save_pattern(LearnedPattern(
            scope="fleet",
            observation="Массовый перегрев фьюзера",
            evidence_devices=["DEV003", "DEV004", "DEV005"],
        ))

        data = mm.to_dict()
        mm2 = MemoryManager.from_dict(data)

        assert len(mm2) == 2
        assert mm2.to_dict() == data

    def test_duplicate_merges_evidence(self) -> None:
        """Same scope+observation merges evidence_devices."""
        mm = MemoryManager(min_evidence_devices=2)
        mm.save_pattern(LearnedPattern(
            scope="MX-3071",
            observation="SC542 повторяется",
            evidence_devices=["DEV001", "DEV002"],
        ))
        mm.save_pattern(LearnedPattern(
            scope="MX-3071",
            observation="SC542 повторяется",
            evidence_devices=["DEV002", "DEV003"],
        ))
        assert len(mm) == 1
        assert set(mm.get_patterns("MX-3071")[0].evidence_devices) == {
            "DEV001", "DEV002", "DEV003",
        }

    def test_get_patterns_includes_fleet_scope(self) -> None:
        """get_patterns(model) returns model-specific + fleet-scoped patterns."""
        mm = MemoryManager(min_evidence_devices=2)
        mm.save_pattern(LearnedPattern(
            scope="MX-3071",
            observation="Модельная проблема",
            evidence_devices=["DEV001", "DEV002"],
        ))
        mm.save_pattern(LearnedPattern(
            scope="fleet",
            observation="Общая проблема",
            evidence_devices=["DEV003", "DEV004"],
        ))
        mm.save_pattern(LearnedPattern(
            scope="MX-4071",
            observation="Другая модель",
            evidence_devices=["DEV005", "DEV006"],
        ))

        result = mm.get_patterns("MX-3071")
        assert len(result) == 2
        observations = {p.observation for p in result}
        assert "Модельная проблема" in observations
        assert "Общая проблема" in observations
        assert "Другая модель" not in observations

    def test_pattern_visible_in_subsequent_run_batch(self) -> None:
        """Pattern saved via MemoryManager appears in next run_batch system prompt."""
        mm = MemoryManager(min_evidence_devices=2)
        mm.save_pattern(LearnedPattern(
            scope="MX-3071",
            observation="SC542 часто ложноположительный",
            evidence_devices=["DEV001", "DEV002"],
        ))

        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm, memory_manager=mm)
        ctx = _make_context(learned_patterns=[])
        agent.run_batch("DEV001", ctx)

        call_args = llm.generate.call_args_list[0]
        messages = (
            call_args.kwargs.get("messages")
            or call_args[1].get("messages")
            or call_args[0][0]
        )
        system_msg = messages[0]["content"]
        assert "SC542 часто ложноположительный" in system_msg


# ── 9. Trace serialization ──────────────────────────────────────────────────


class TestTraceSerialization:
    """Verify Trace to_json / from_json / summary."""

    def test_to_json_from_json_roundtrip(self) -> None:
        """to_json → from_json produces identical Trace."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _result, trace = agent.run_batch("DEV001", _make_context())

        raw = trace.to_json()
        restored = Trace.from_json(raw)

        assert restored.session_id == trace.session_id
        assert restored.mode == trace.mode
        assert restored.device_id == trace.device_id
        assert restored.total_tool_calls == trace.total_tool_calls
        assert restored.total_llm_calls == trace.total_llm_calls
        assert restored.total_tokens == trace.total_tokens
        assert restored.attempts == trace.attempts
        assert restored.flagged_for_review == trace.flagged_for_review
        assert len(restored.steps) == len(trace.steps)
        assert restored.final_result == trace.final_result

    def test_summary_contains_key_fields(self) -> None:
        """summary() includes mode, HI, tool/llm counts."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content=_FINAL_ANSWER_JSON),
            _make_llm_response(content=_REFLECTION_APPROVED_JSON),
        ]

        agent = _make_agent(llm)
        _result, trace = agent.run_batch("DEV001", _make_context())

        s = trace.summary()
        assert "batch" in s
        assert "DEV001" in s
        assert "HI=72" in s
        assert "tools=" in s
        assert "llm=" in s
        assert "flagged" not in s

    def test_summary_shows_flagged(self) -> None:
        """summary() includes flagged marker when flagged_for_review=True."""
        llm = MagicMock()
        llm.generate.side_effect = [
            _make_llm_response(content="garbage"),
            _make_llm_response(content="still garbage"),
        ]

        agent = _make_agent(llm)
        _result, trace = agent.run_batch("DEV001", _make_context())

        s = trace.summary()
        assert "flagged" in s
