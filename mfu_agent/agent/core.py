"""Agent core — Track B, Level 2.

Main Agent class implementing batch health-index calculation
with tool dispatch, self-check (reflection), and trace collection.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_io.models import (
    AgentMode,
    BatchContext,
    ChatContext,
    ConfidenceZone,
    FactorContribution,
    HealthResult,
    HealthZone,
    LearnedPattern,
    ReflectionAction,
    ReflectionResult,
    ReflectionVerdict,
    Trace,
    TraceStep,
    TraceStepType,
)

if TYPE_CHECKING:
    from agent.memory import MemoryManager
    from agent.tools.registry import ToolRegistry
    from config.loader import AgentConfig
    from data_io.factor_store import FactorStore
    from llm.client import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


class Agent:
    """MFU health-index calculation agent.

    Orchestrates the Plan → Tools → Reflection cycle
    for batch (per-device) health assessment.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        factor_store: FactorStore,
        config: AgentConfig,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._llm = llm_client
        self._tools = tool_registry
        self._factor_store = factor_store
        self._config = config
        self._memory = memory_manager

        self._system_batch_prompt = _load_prompt("system_batch.md")
        self._system_chat_prompt = _load_prompt("system_chat.md")
        self._reflection_prompt = _load_prompt("reflection.md")

    # ── Public API ───────────────────────────────────────────────────────

    def run_batch(
        self,
        device_id: str,
        context: BatchContext,
    ) -> tuple[HealthResult, Trace]:
        """Calculate health index for a single device.

        Runs up to ``max_attempts_per_device`` attempts with reflection
        between them.  Returns the final HealthResult and execution Trace.
        """
        loop_cfg = self._config.agent
        max_attempts = loop_cfg.max_attempts_per_device

        trace = Trace(
            session_id=str(uuid.uuid4()),
            mode=AgentMode.BATCH,
            device_id=device_id,
            started_at=datetime.now(UTC),
        )

        result: HealthResult | None = None
        revision_notes: str | None = None

        for attempt in range(1, max_attempts + 1):
            trace.attempts = attempt

            messages = self._build_messages(
                device_id=device_id,
                context=context,
                revision_notes=revision_notes,
            )

            loop_result, loop_trace_steps = self._agent_loop(
                messages=messages,
                trace=trace,
            )

            trace.steps.extend(loop_trace_steps)

            result = self._parse_final_result(loop_result, device_id)
            if result is None:
                logger.warning(
                    "Устройство %s: не удалось распарсить результат (попытка %d)",
                    device_id, attempt,
                )
                trace.flagged_for_review = True
                break

            if not self._config.reflection.enabled:
                break

            reflection = self._run_reflection(
                result=result,
                trace=trace,
                context=context,
            )

            if reflection.verdict == ReflectionVerdict.APPROVED:
                break

            if reflection.verdict == ReflectionVerdict.SUSPICIOUS:
                trace.flagged_for_review = True
                result.reflection_notes = "; ".join(
                    i.issue for i in reflection.issues
                )
                break

            if attempt < max_attempts:
                revision_notes = self._format_revision_notes(reflection)
                logger.info(
                    "Устройство %s: reflection → needs_revision, попытка %d/%d",
                    device_id, attempt + 1, max_attempts,
                )
            else:
                trace.flagged_for_review = True
                result.reflection_notes = (
                    "Исчерпаны попытки пересчёта. "
                    + "; ".join(i.issue for i in reflection.issues)
                )

        if result is None:
            result = HealthResult(
                device_id=device_id,
                health_index=50,
                confidence=0.2,
                zone=HealthZone.YELLOW,
                confidence_zone=ConfidenceZone.LOW,
                calculated_at=datetime.now(UTC),
                reflection_notes="Не удалось получить результат от LLM",
            )
            trace.flagged_for_review = True

        trace.ended_at = datetime.now(UTC)
        trace.final_result = result.model_dump(mode="json")

        if self._config.memory.enabled and not trace.flagged_for_review:
            self._save_learned_patterns(result, context, trace)
            if self._memory is not None:
                self._save_patterns_to_memory(result, context, trace)

        return result, trace

    def run_batch_lite(
        self,
        device_id: str,
        context: BatchContext,
    ) -> tuple[HealthResult, Trace]:
        """Lightweight batch: direct tool calls, no agent loop.

        Uses LLM only for classify_error_severity (small context).
        Everything else is deterministic. Works with 8K context LLMs.
        """
        trace = Trace(
            session_id=str(uuid.uuid4()),
            mode=AgentMode.BATCH,
            device_id=device_id,
            started_at=datetime.now(UTC),
        )
        trace.attempts = 1

        t0 = time.perf_counter()

        events_result = self._tools.execute(
            "get_device_events", {"device_id": device_id},
        )
        events = (events_result.data or {}).get("events", []) if events_result.success else []

        resources_result = self._tools.execute(
            "get_device_resources", {"device_id": device_id},
        )
        resources = resources_result.data if resources_result.success else None

        meta = self._factor_store.get_device_metadata(device_id)
        model_name = meta.model if meta else None

        trace.steps.append(TraceStep(
            step_number=1,
            type=TraceStepType.TOOL_CALL,
            tool_name="get_device_events",
            tool_result_summary=f"{len(events)} events",
            duration_ms=0,
        ))

        unique_errors: dict[str, dict] = {}
        for ev in events:
            code = ev.get("error_code") or ev.get("error_description") or ""
            if code and code not in unique_errors:
                unique_errors[code] = ev

        factors = []
        rag_missing = 0
        step_num = 2

        for code, ev in unique_errors.items():
            classify_result = self._tools.execute("classify_error_severity", {
                "error_code": code,
                "model": model_name or "",
                "error_description": ev.get("error_description", ""),
            })

            if classify_result.success and classify_result.data:
                severity = classify_result.data.get("severity", "Medium")
                source = classify_result.data.get("source")
            else:
                severity = "Medium"
                source = None
                rag_missing += 1

            rep_result = self._tools.execute("count_error_repetitions", {
                "device_id": device_id,
                "error_code": code,
            })
            n_reps = rep_result.data.get("count", 1) if rep_result.success and rep_result.data else 1

            ts = ev.get("timestamp", datetime.now(UTC).isoformat())

            factors.append({
                "error_code": code,
                "severity_level": severity,
                "n_repetitions": n_reps,
                "event_timestamp": ts,
                "applicable_modifiers": [],
                "source": source or "",
            })

            trace.steps.append(TraceStep(
                step_number=step_num,
                type=TraceStepType.TOOL_CALL,
                tool_name="classify_error_severity",
                tool_args={"error_code": code},
                tool_result_summary=f"{severity} (reps={n_reps})",
                duration_ms=0,
            ))
            step_num += 1

        calc_args = {
            "device_id": device_id,
            "factors": factors,
            "confidence_factors": {
                "rag_missing_count": rag_missing,
                "missing_resources": resources is None,
                "missing_model": model_name is None,
            },
        }
        calc_result = self._tools.execute("calculate_health_index", calc_args)

        duration_ms = int((time.perf_counter() - t0) * 1000)

        if calc_result.success and calc_result.data:
            d = calc_result.data

            result = HealthResult(
                device_id=device_id,
                health_index=d["health_index"],
                confidence=d["confidence"],
                zone=HealthZone(d["zone"]),
                confidence_zone=ConfidenceZone(d["confidence_zone"]),
                factor_contributions=[
                    FactorContribution(**fc)
                    for fc in d.get("factor_contributions", [])
                ],
                confidence_reasons=d.get("confidence_reasons", []),
                calculated_at=datetime.now(UTC),
            )
        else:
            result = HealthResult(
                device_id=device_id,
                health_index=50,
                confidence=0.2,
                zone=HealthZone.YELLOW,
                confidence_zone=ConfidenceZone.LOW,
                calculated_at=datetime.now(UTC),
                reflection_notes="Не удалось рассчитать индекс",
            )
            trace.flagged_for_review = True

        trace.steps.append(TraceStep(
            step_number=step_num,
            type=TraceStepType.TOOL_CALL,
            tool_name="calculate_health_index",
            tool_result_summary=f"index={result.health_index} zone={result.zone.value}",
            duration_ms=duration_ms,
        ))

        trace.ended_at = datetime.now(UTC)
        trace.final_result = result.model_dump(mode="json")
        trace.total_tool_calls = step_num

        return result, trace

    _REPORT_TOKEN_LIMIT = 8000
    _CHAT_MAX_TOOL_CALLS = 10

    def run_chat(
        self,
        user_message: str,
        context: ChatContext,
    ) -> tuple[str, Trace]:
        """Answer a user question in chat mode.

        Uses the same agent_loop but without reflection.
        Returns (text_answer, trace).
        """
        trace = Trace(
            session_id=str(uuid.uuid4()),
            mode=AgentMode.CHAT,
            user_query=user_message,
            started_at=datetime.now(UTC),
        )

        messages = self._build_chat_messages(user_message, context)

        chat_config = type(self._config.agent).model_validate(
            self._config.agent.model_dump()
            | {"max_tool_calls_per_attempt": self._CHAT_MAX_TOOL_CALLS},
        )
        saved = self._config.agent
        self._config.agent = chat_config
        try:
            answer, loop_steps = self._agent_loop(
                messages=messages,
                trace=trace,
            )
        finally:
            self._config.agent = saved

        trace.steps.extend(loop_steps)
        trace.attempts = 1
        trace.ended_at = datetime.now(UTC)
        trace.final_result = {"answer": answer}

        return answer, trace

    def _build_chat_messages(
        self,
        user_message: str,
        context: ChatContext,
    ) -> list[dict[str, Any]]:
        system_parts = [self._system_chat_prompt]

        if self._memory is not None:
            patterns = self._memory.get_patterns()
            preamble = self._format_learned_patterns(patterns)
            if preamble:
                system_parts.append(preamble)

        report_ctx = self._serialize_report_context(context.current_report)
        if report_ctx:
            system_parts.append(report_ctx)

        system_content = "\n\n".join(system_parts)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        for msg in context.conversation_history:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})
        return messages

    def _serialize_report_context(self, report: Any) -> str:
        """Serialize a Report into a context string for the chat prompt.

        If the full JSON exceeds ~8K tokens, falls back to a compact
        summary with fleet_summary + top-10 devices.
        """
        if report is None:
            return ""

        try:
            full = json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            return ""

        if len(full) <= self._REPORT_TOKEN_LIMIT * 4:
            return f"Последний отчёт:\n{full}"

        try:
            summary_data = {
                "fleet_summary": report.fleet_summary.model_dump(mode="json")
                if hasattr(report.fleet_summary, "model_dump")
                else report.fleet_summary,
                "top_devices": [
                    {
                        "device_id": d.device_id,
                        "health_index": d.health_index,
                        "zone": d.zone if isinstance(d.zone, str) else d.zone.value,
                    }
                    for d in (report.devices or [])[:10]
                ],
            }
            compact = json.dumps(summary_data, ensure_ascii=False, default=str)
            return f"Последний отчёт (сокращённый):\n{compact}"
        except Exception:
            return ""

    # ── Agent loop ───────────────────────────────────────────────────────

    def _agent_loop(
        self,
        messages: list[dict[str, Any]],
        trace: Trace,
    ) -> tuple[str, list[TraceStep]]:
        """Run the LLM ↔ tool loop until a final answer or limits hit.

        Returns the final text content and collected trace steps.
        """
        loop_cfg = self._config.agent
        max_llm_calls = loop_cfg.max_llm_calls_per_attempt
        max_tool_calls = loop_cfg.max_tool_calls_per_attempt

        llm_calls = 0
        tool_calls = 0
        steps: list[TraceStep] = []
        step_counter = len(trace.steps)

        tool_schemas = self._tools.get_all_schemas()

        while llm_calls < max_llm_calls:
            t0 = time.perf_counter()
            response = self._llm.generate(
                messages=messages,
                tools=tool_schemas if tool_calls < max_tool_calls else None,
                params=self._config.llm.batch_mode,
            )
            duration_ms = int((time.perf_counter() - t0) * 1000)
            llm_calls += 1
            trace.total_llm_calls += 1
            trace.total_tokens += response.usage.total_tokens

            step_counter += 1
            steps.append(TraceStep(
                step_number=step_counter,
                type=TraceStepType.LLM_CALL,
                thought=response.content[:500] if response.content else None,
                duration_ms=duration_ms,
                tokens_used=response.usage.total_tokens,
            ))

            if not response.tool_calls:
                return response.content, steps

            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(
                                tc.arguments, ensure_ascii=False,
                            ),
                        },
                    }
                    for tc in response.tool_calls
                ],
            })

            for tc in response.tool_calls:
                if tool_calls >= max_tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({
                            "error": "Лимит вызовов инструментов исчерпан",
                        }),
                    })
                    continue

                t0 = time.perf_counter()
                tool_result = self._tools.execute(tc.name, tc.arguments)
                tool_duration = int((time.perf_counter() - t0) * 1000)
                tool_calls += 1
                trace.total_tool_calls += 1

                result_json = tool_result.model_dump_json()
                if len(result_json) > 12000:
                    result_json = result_json[:12000] + '..."}'

                step_counter += 1
                steps.append(TraceStep(
                    step_number=step_counter,
                    type=TraceStepType.TOOL_CALL,
                    tool_name=tc.name,
                    tool_args=tc.arguments,
                    tool_result_summary=result_json[:300],
                    duration_ms=tool_duration,
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                })

        logger.warning("Лимит LLM-вызовов исчерпан (%d)", max_llm_calls)
        return messages[-1].get("content", ""), steps

    # ── Reflection ───────────────────────────────────────────────────────

    def _run_reflection(
        self,
        result: HealthResult,
        trace: Trace,
        context: BatchContext,
    ) -> ReflectionResult:
        """Self-check the calculation result via a separate LLM call."""
        t0 = time.perf_counter()

        # -- gather context via tools --
        history_result = self._tools.execute(
            "get_device_history", {"device_id": result.device_id},
        )
        device_history = history_result.data if history_result.success else []

        fleet_result = self._tools.execute("get_fleet_statistics", {})
        fleet_stats = fleet_result.data if fleet_result.success else {}

        events_result = self._tools.execute(
            "get_device_events", {"device_id": result.device_id},
        )
        device_events = events_result.data if events_result.success else []

        resources_result = self._tools.execute(
            "get_device_resources", {"device_id": result.device_id},
        )
        device_resources = resources_result.data if resources_result.success else {}

        if isinstance(device_events, dict):
            ev_list = device_events.get("events", [])
        else:
            ev_list = device_events if isinstance(device_events, list) else []
        ev_list = ev_list[-20:]

        if isinstance(fleet_stats, dict) and "devices" in fleet_stats:
            fleet_stats = {k: v for k, v in fleet_stats.items() if k != "devices"}

        reflection_context = {
            "device_id": result.device_id,
            "health_index": result.health_index,
            "confidence": result.confidence,
            "zone": result.zone,
            "factor_contributions": [
                fc.model_dump(mode="json")
                for fc in result.factor_contributions
            ],
            "device_input": {
                "events": ev_list,
                "resources": device_resources,
            },
            "device_history": device_history,
            "fleet_statistics": fleet_stats,
        }

        context_json = json.dumps(
            reflection_context, ensure_ascii=False, indent=2, default=str,
        )
        if len(context_json) > 15000:
            context_json = context_json[:15000] + "\n... (обрезано)"

        messages = [
            {"role": "system", "content": self._reflection_prompt},
            {"role": "user", "content": context_json},
        ]

        response = self._llm.generate(
            messages=messages,
            params=self._config.llm.reflection,
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        trace.total_llm_calls += 1
        trace.total_tokens += response.usage.total_tokens

        step_number = len(trace.steps) + 1
        trace.steps.append(TraceStep(
            step_number=step_number,
            type=TraceStepType.REFLECTION,
            thought=response.content[:500] if response.content else None,
            duration_ms=duration_ms,
            tokens_used=response.usage.total_tokens,
        ))

        return self._parse_reflection(response.content, messages, trace)

    # ── Message building ─────────────────────────────────────────────────

    def _build_messages(
        self,
        device_id: str,
        context: BatchContext,
        revision_notes: str | None = None,
    ) -> list[dict[str, Any]]:
        system_parts = [self._system_batch_prompt]

        role_ext = self._get_role_extensions(device_id, context)
        if role_ext:
            system_parts.append(role_ext)

        all_patterns = list(context.learned_patterns)
        if self._memory is not None:
            device_model = self._resolve_device_model(device_id, context)
            all_patterns.extend(self._memory.get_patterns(scope=device_model))

        if all_patterns:
            patterns_text = self._format_learned_patterns(all_patterns)
            if patterns_text:
                system_parts.append(patterns_text)

        system_content = "\n\n".join(system_parts)

        user_task = (
            f"Рассчитай индекс здоровья для устройства {device_id}.\n"
            f"Профиль весов: {context.weights_profile.profile_name}"
        )

        if revision_notes:
            user_task += f"\n\n⚠️ Замечания ревизора (предыдущая попытка):\n{revision_notes}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_task},
        ]

    @staticmethod
    def _resolve_device_model(device_id: str, context: BatchContext) -> str | None:
        meta = context.device_metadata
        if meta is not None:
            return getattr(meta, "model", None)
        return None

    def _get_role_extensions(
        self,
        device_id: str,
        context: BatchContext,
    ) -> str:
        if not self._config.role_extensions.enabled:
            return ""

        meta = context.device_metadata
        if meta is None:
            try:
                meta = self._factor_store.get_device_metadata(device_id)
            except Exception:
                return ""

        if meta is None:
            return ""

        parts: list[str] = []

        if getattr(meta, "critical_function", False):
            parts.append(
                "⚠️ Это устройство отмечено как критически важное. "
                "При оценке учитывай повышенные требования к надёжности."
            )

        tags = getattr(meta, "tags", [])
        if "false_alarm_history" in tags:
            parts.append(
                "ℹ️ У этого устройства история ложных срабатываний. "
                "Перепроверяй критичность ошибок через документацию."
            )

        return "\n".join(parts)

    @staticmethod
    def _format_learned_patterns(patterns: list[LearnedPattern]) -> str:
        if not patterns:
            return ""
        lines = ["Ранее обнаруженные закономерности в парке:"]
        for p in patterns[:10]:
            lines.append(f"— [{p.scope}] {p.observation}")
        return "\n".join(lines)

    @staticmethod
    def _format_revision_notes(reflection: ReflectionResult) -> str:
        parts: list[str] = []
        for issue in reflection.issues:
            parts.append(f"[{issue.severity}] {issue.issue}")
        return "\n".join(parts)

    # ── Parsing ──────────────────────────────────────────────────────────

    def _parse_final_result(
        self,
        content: str,
        device_id: str,
    ) -> HealthResult | None:
        """Parse LLM final answer into HealthResult.

        Tries direct JSON parse first. Falls back to guided JSON
        LLM call if the content is not valid JSON.
        """
        data = self._try_parse_json(content)

        if data is None:
            data = self._guided_json_fallback(content, device_id)

        if data is None:
            return None

        try:
            data.setdefault("device_id", device_id)
            data.setdefault("calculated_at", datetime.now(UTC).isoformat())
            if "zone" not in data:
                hi = data.get("health_index", 50)
                if hi >= 75:
                    data["zone"] = "green"
                elif hi >= 40:
                    data["zone"] = "yellow"
                else:
                    data["zone"] = "red"
            if "confidence_zone" not in data:
                conf = data.get("confidence", 0.5)
                if conf >= 0.85:
                    data["confidence_zone"] = "high"
                elif conf >= 0.6:
                    data["confidence_zone"] = "medium"
                else:
                    data["confidence_zone"] = "low"
            return HealthResult.model_validate(data)
        except Exception:
            logger.exception("Не удалось валидировать HealthResult для %s", device_id)
            return None

    def _guided_json_fallback(
        self,
        content: str,
        device_id: str,
    ) -> dict[str, Any] | None:
        """Ask LLM to extract structured JSON from free-text answer."""
        schema = {
            "type": "object",
            "properties": {
                "health_index": {"type": "integer", "minimum": 1, "maximum": 100},
                "confidence": {"type": "number", "minimum": 0.2, "maximum": 1.0},
                "factors": {"type": "array"},
                "explanation": {"type": "string"},
                "reflection_notes": {"type": "string"},
            },
            "required": ["health_index", "confidence"],
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "Извлеки из текста ниже результат расчёта индекса здоровья "
                    "МФУ и верни строго JSON по указанной схеме."
                ),
            },
            {"role": "user", "content": content},
        ]

        try:
            response = self._llm.generate(
                messages=messages,
                response_schema=schema,
                params=self._config.llm.batch_mode,
            )
            return self._try_parse_json(response.content)
        except Exception:
            logger.exception("Guided JSON fallback failed для %s", device_id)
            return None

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if text.startswith("```"):
            import re
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _parse_reflection(
        self,
        content: str,
        messages: list[dict[str, Any]],
        trace: Trace,
    ) -> ReflectionResult:
        """Parse reflection LLM output into ReflectionResult with retries."""
        max_retries = getattr(
            getattr(self._config, "llm_endpoint", None),
            "max_retries_invalid", 2,
        )

        parsed = self._try_validate_reflection(content)
        if parsed is not None:
            self._log_reflection(parsed)
            return parsed

        schema_reminder = (
            'Return ONLY valid JSON: '
            '{"verdict": "approved"|"needs_revision"|"suspicious", '
            '"issues": [{"issue": str, "severity": "high"|"medium"|"low"}], '
            '"recommended_action": "accept"|"recalculate"|"flag_for_review"}'
        )

        for retry in range(max_retries):
            logger.warning(
                "Reflection JSON parse failed, retry %d/%d",
                retry + 1, max_retries,
            )
            retry_messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": schema_reminder},
            ]
            response = self._llm.generate(
                messages=retry_messages,
                params=self._config.llm.reflection,
            )
            trace.total_llm_calls += 1
            trace.total_tokens += response.usage.total_tokens
            content = response.content

            parsed = self._try_validate_reflection(content)
            if parsed is not None:
                self._log_reflection(parsed)
                return parsed

        logger.warning("Reflection retries exhausted, falling back to approved")
        return ReflectionResult(
            verdict=ReflectionVerdict.APPROVED,
            issues=[],
            recommended_action=ReflectionAction.ACCEPT,
        )

    def _try_validate_reflection(self, content: str) -> ReflectionResult | None:
        data = self._try_parse_json(content)
        if data is not None:
            try:
                return ReflectionResult.model_validate(data)
            except Exception:
                return None
        return None

    @staticmethod
    def _log_reflection(result: ReflectionResult) -> None:
        logger.info(
            "Reflection verdict=%s, action=%s",
            result.verdict.value, result.recommended_action.value,
        )
        for issue in result.issues:
            logger.warning(
                "Reflection issue [%s]: %s",
                issue.severity, issue.issue,
            )

    # ── Learned patterns ─────────────────────────────────────────────────

    def _save_learned_patterns(
        self,
        result: HealthResult,
        context: BatchContext,
        trace: Trace,
    ) -> None:
        """Extract and save patterns if conditions are met.

        Patterns require at least ``pattern_min_evidence_devices`` devices
        with similar observations.
        """
        t0 = time.perf_counter()
        mem_cfg = self._config.memory

        if not context.fleet_stats:
            return

        try:
            fleet_data: dict[str, list[Any]] = context.fleet_stats
            if not isinstance(fleet_data, dict) or len(fleet_data) < mem_cfg.pattern_min_evidence_devices:
                return

            for fc in result.factor_contributions:
                matching_devices = []
                for did, history in fleet_data.items():
                    if not history:
                        continue
                    latest = history[-1]
                    for other_fc in getattr(latest, "factor_contributions", []):
                        if getattr(other_fc, "label", "") == fc.label:
                            matching_devices.append(did)
                            break

                if len(matching_devices) >= mem_cfg.pattern_min_evidence_devices:
                    pattern = LearnedPattern(
                        type="pattern",
                        scope=fc.source or "fleet",
                        observation=f"Фактор '{fc.label}' обнаружен у {len(matching_devices)} устройств",
                        evidence_devices=matching_devices[:10],
                    )
                    context.learned_patterns.append(pattern)

            duration_ms = int((time.perf_counter() - t0) * 1000)
            if context.learned_patterns:
                trace.steps.append(TraceStep(
                    step_number=len(trace.steps) + 1,
                    type=TraceStepType.MEMORY_SAVE,
                    thought=f"Сохранено {len(context.learned_patterns)} паттернов",
                    duration_ms=duration_ms,
                ))
        except Exception:
            logger.exception("Ошибка при сохранении паттернов")

    def _save_patterns_to_memory(
        self,
        result: HealthResult,
        context: BatchContext,
        trace: Trace,
    ) -> None:
        """Persist learned patterns from context into MemoryManager."""
        t0 = time.perf_counter()
        saved = 0
        for pattern in context.learned_patterns:
            if self._memory is not None and self._memory.save_pattern(pattern):
                saved += 1

        if saved:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            trace.steps.append(TraceStep(
                step_number=len(trace.steps) + 1,
                type=TraceStepType.MEMORY_SAVE,
                thought=f"MemoryManager: сохранено {saved} паттернов",
                duration_ms=duration_ms,
            ))
