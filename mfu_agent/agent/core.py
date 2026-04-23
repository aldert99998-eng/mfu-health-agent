"""Agent core — Track B, Level 2.

Main Agent class implementing batch health-index calculation
with tool dispatch, self-check (reflection), and trace collection.
"""

from __future__ import annotations

import json
import logging
import re
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
    DeepDeviceAnalysis,
    FactorContribution,
    HealthResult,
    HealthZone,
    LearnedPattern,
    MassErrorAnalysis,
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
    from data_io.models import WeightsProfile
    from llm.client import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _robust_json_parse(text: str) -> dict[str, Any] | None:
    """Extract first JSON object from LLM output tolerating extra text and truncation.

    Handles:
    - Markdown code fences (```json ... ```)
    - Extra text before/after the object ("Вот результат: {...} конец")
    - Extra sibling objects ("Extra data" error)
    - Truncated strings — tries to close a dangling string and outer braces
    """
    if not text:
        return None

    # Strip markdown fences
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 2:
            stripped = parts[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()

    # Find first '{' — start of JSON object
    start = stripped.find("{")
    if start < 0:
        return None
    candidate = stripped[start:]

    # First attempt: json.JSONDecoder.raw_decode — ignores trailing text
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Repair attempt: the response was cut off mid-string.
    # Strategy: walk char by char, track string state + brace depth.
    # At end: close any open string + pad missing closing braces.
    out: list[str] = []
    in_string = False
    escape = False
    depth = 0
    for ch in candidate:
        out.append(ch)
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Balanced top-level object found
                    tail = "".join(out)
                    try:
                        obj = json.loads(tail)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    # Close any dangling string
    if in_string:
        out.append('"')
    # Close missing braces
    while depth > 0:
        out.append("}")
        depth -= 1

    try:
        obj = json.loads("".join(out))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None

    return None


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

    def set_tools(self, registry: ToolRegistry) -> None:
        """Swap the tool registry on a cached Agent instance.

        The chat page builds a per-request registry with live
        ``current_report`` and ``mass_error_analyses`` so tools can
        answer questions about the loaded fleet.
        """
        self._tools = registry

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
                # Fallback: if calculate_health_index was already called
                # during the loop, its result is cached in ToolDependencies.
                cached = self._get_cached_health_result(device_id)
                if cached is not None:
                    logger.info(
                        "Устройство %s: финальный JSON не распарсен, "
                        "использую кэш calculate_health_index", device_id,
                    )
                    cached.reflection_notes = (
                        "LLM не смог оформить финальный JSON — "
                        "использован результат calculate_health_index из цепочки tools."
                    )
                    trace.flagged_for_review = True
                    return cached, trace
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
    ) -> tuple[HealthResult, Trace, dict[str, Any]]:
        """Lightweight batch: direct tool calls, no agent loop.

        Returns (HealthResult, Trace, calc_args) where calc_args contains
        raw factors and confidence_factors for weight recalculation.

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

        # Batch reads the full FactorStore directly — tool-output truncation
        # (50-event cap for chat context) must NOT affect health-index calculation.
        all_events = self._factor_store.get_events(device_id, window_days=30)
        resources = self._factor_store.get_resources(device_id)
        meta = self._factor_store.get_device_metadata(device_id)
        model_name = meta.model if meta else None

        trace.steps.append(TraceStep(
            step_number=1,
            type=TraceStepType.TOOL_CALL,
            tool_name="factor_store.get_events",
            tool_result_summary=f"{len(all_events)} events",
            duration_ms=0,
        ))

        unique_errors: dict[str, Any] = {}
        for e in all_events:
            code = e.error_code or e.error_description or ""
            if not code:
                continue
            prev = unique_errors.get(code)
            if prev is None or e.timestamp > prev.timestamp:
                unique_errors[code] = e

        factors = []
        rag_missing = 0
        step_num = 2

        for code, ev in unique_errors.items():
            classify_result = self._tools.execute("classify_error_severity", {
                "error_code": code,
                "model": model_name or "",
                "error_description": ev.error_description or "",
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

            ts = ev.timestamp.isoformat() if ev.timestamp else datetime.now(UTC).isoformat()

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

        return result, trace, calc_args

    _REPORT_TOKEN_LIMIT = 1500
    _CHAT_MAX_TOOL_CALLS = 10
    _CHAT_MAX_HISTORY_MESSAGES = 10

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

        rag_block, rag_hits = self._fetch_chat_rag_context(user_message)
        trace.rag_hits = rag_hits

        messages = self._build_chat_messages(user_message, context, rag_block=rag_block)

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
                llm_params=self._config.llm.chat_mode,
                strip_reasoning_on_final=True,
            )
        finally:
            self._config.agent = saved

        trace.steps.extend(loop_steps)
        trace.attempts = 1
        trace.ended_at = datetime.now(UTC)
        trace.final_result = {"answer": answer}

        return answer, trace

    # ── Mass-error LLM analysis ──────────────────────────────────────────

    def analyze_mass_error(
        self,
        error_code: str,
        description: str,
        affected_count: int,
        total_occurrences: int,
        fleet_total: int,
        severity: str = "",
        sample_device_ids: list[str] | None = None,
        sample_descriptions: list[str] | None = None,
    ) -> MassErrorAnalysis:
        """Single-call LLM analysis of one error code aggregated across fleet.

        Returns a MassErrorAnalysis. On any failure (no LLM, parse error,
        timeout) returns a degraded record with ``error`` populated and
        is_systemic=False.
        """
        now = datetime.now(UTC)

        if self._llm is None:
            return MassErrorAnalysis(
                error_code=error_code,
                description=description,
                affected_device_count=affected_count,
                total_occurrences=total_occurrences,
                analyzed_at=now,
                error="LLM недоступен",
            )

        try:
            prompt_template = (_PROMPTS_DIR / "mass_error_analysis.md").read_text(
                encoding="utf-8",
            )
        except FileNotFoundError:
            return MassErrorAnalysis(
                error_code=error_code,
                description=description,
                affected_device_count=affected_count,
                total_occurrences=total_occurrences,
                analyzed_at=now,
                error="Промпт mass_error_analysis.md не найден",
            )

        rag_context = self._fetch_rag_context(error_code, description)

        prompt_text = prompt_template.format(
            error_code=error_code,
            description=description or "нет описания",
            affected_count=affected_count,
            fleet_total=fleet_total,
            total_occurrences=total_occurrences,
            severity=severity or "неизвестно",
            sample_device_ids=", ".join((sample_device_ids or [])[:5]) or "—",
            sample_descriptions="\n".join((sample_descriptions or [])[:3]) or "—",
            rag_context=rag_context or "(нет фрагментов из документации)",
        )

        response_schema = {
            "type": "object",
            "properties": {
                "is_systemic": {"type": "boolean"},
                "what_is_this": {"type": "string"},
                "why_this_pattern": {"type": "string"},
                "business_impact": {"type": "string"},
                "immediate_action": {"type": "string"},
                "long_term_action": {"type": "string"},
                "indicators_to_watch": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "is_systemic",
                "what_is_this",
                "why_this_pattern",
                "business_impact",
                "immediate_action",
                "long_term_action",
                "indicators_to_watch",
            ],
        }

        try:
            from config.loader import LLMGenerationParams
            from llm.client import LLMClient

            params = LLMGenerationParams(temperature=0.3, top_p=1.0, max_tokens=2500)
            resp = self._llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты анализируешь агрегированные данные по ошибкам парка МФУ. "
                            "Возвращай строго ОДИН JSON-объект без Markdown-обёртки, "
                            "без вступительного текста, без пояснений. "
                            "Каждое текстовое поле ≤300 символов. "
                            "Запрещены расплывчатые фразы типа «возможные проблемы», "
                            "«необходимо следить», «сетевые сервисы» без уточнения."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                tools=None,
                response_schema=response_schema,
                params=params,
            )
            text = LLMClient.strip_reasoning_artifacts(resp.content).strip()
            data = _robust_json_parse(text)
            if data is None:
                raise ValueError("JSON не извлечён из ответа LLM")
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_mass_error failed for %s: %s", error_code, exc)
            return MassErrorAnalysis(
                error_code=error_code,
                description=description,
                affected_device_count=affected_count,
                total_occurrences=total_occurrences,
                analyzed_at=now,
                error=f"LLM error: {exc}",
            )

        def _clean(raw: Any, limit: int = 300) -> str:
            return str(raw or "").strip()[:limit]

        what = _clean(data.get("what_is_this"))
        why = _clean(data.get("why_this_pattern"))
        impact = _clean(data.get("business_impact"))
        imm = _clean(data.get("immediate_action"))
        long_ = _clean(data.get("long_term_action"))

        # what_is_this fallback: если LLM не справился — подставим из RAG/YAML
        if len(what) < 20:
            if rag_context and ("Tray" in rag_context or error_code in rag_context):
                what = rag_context.split("\n")[1][:300] if "\n" in rag_context else rag_context[:300]
            elif description:
                what = description[:300]
            else:
                what = f"Код {error_code}: недостаточно документации для расшифровки."

        # Индикаторы: до 5 пунктов, каждый ≤150 символов
        raw_indicators = data.get("indicators_to_watch") or []
        if not isinstance(raw_indicators, list):
            raw_indicators = [str(raw_indicators)]
        indicators = [str(x).strip()[:150] for x in raw_indicators if str(x).strip()][:5]

        # Legacy-поля заполняются из новых для обратной совместимости
        legacy_cause = why or impact or ""
        legacy_action = imm or long_ or ""
        legacy_explanation = "\n\n".join(filter(None, [
            f"Что это: {what}" if what else "",
            f"Почему массово: {why}" if why else "",
            f"Влияние: {impact}" if impact else "",
            f"Сейчас: {imm}" if imm else "",
            f"На будущее: {long_}" if long_ else "",
        ]))[:2000]

        return MassErrorAnalysis(
            error_code=error_code,
            description=description,
            affected_device_count=affected_count,
            total_occurrences=total_occurrences,
            is_systemic=bool(data.get("is_systemic", False)),
            what_is_this=what,
            why_this_pattern=why,
            business_impact=impact,
            immediate_action=imm,
            long_term_action=long_,
            indicators_to_watch=indicators,
            likely_cause=legacy_cause[:500],
            recommended_action=legacy_action[:500],
            explanation=legacy_explanation,
            analyzed_at=now,
        )

    # ── Single-device deep LLM analysis ──────────────────────────────────

    def analyze_device_deep(
        self,
        device_id: str,
        health_result: HealthResult,
        factor_store: FactorStore | None = None,
        weights_profile: WeightsProfile | None = None,
    ) -> DeepDeviceAnalysis:
        """Single-call LLM deep analysis of one red-zone device.

        Builds a rich prompt with device metadata, top factors, recent events,
        resources and RAG snippets, then asks the LLM for root cause + action
        in one shot. On any failure (no LLM, parse error, exception) returns
        a ``DeepDeviceAnalysis`` with ``error`` populated.
        """
        now = datetime.now(UTC)
        t0 = time.perf_counter()
        hi_orig = health_result.health_index

        def _elapsed_ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        if self._llm is None:
            return DeepDeviceAnalysis(
                device_id=device_id,
                health_index_original=hi_orig,
                analyzed_at=now,
                llm_calls=0,
                duration_ms=_elapsed_ms(),
                error="LLM недоступен",
            )

        try:
            prompt_template = (_PROMPTS_DIR / "device_deep_analysis.md").read_text(
                encoding="utf-8",
            )
        except FileNotFoundError:
            return DeepDeviceAnalysis(
                device_id=device_id,
                health_index_original=hi_orig,
                analyzed_at=now,
                llm_calls=0,
                duration_ms=_elapsed_ms(),
                error="Промпт device_deep_analysis.md не найден",
            )

        # ── Gather context (best-effort, never raises) ──
        meta = None
        events: list[Any] = []
        resources = None
        if factor_store is not None:
            try:
                meta = factor_store.get_device_metadata(device_id)
            except Exception:  # noqa: BLE001
                meta = None
            try:
                events = list(factor_store.get_events(device_id, window_days=30))
            except Exception:  # noqa: BLE001
                events = []
            try:
                resources = factor_store.get_resources(device_id)
            except Exception:  # noqa: BLE001
                resources = None

        model_name = getattr(meta, "model", None) or "неизвестна"

        # Top-3 factors
        top_factors_lines: list[str] = []
        for fc in (health_result.factor_contributions or [])[:3]:
            top_factors_lines.append(
                f"- {fc.label}: penalty={fc.penalty:.1f}, S={fc.S:.1f}, "
                f"R={fc.R:.2f}, C={fc.C:.2f}, source={fc.source}"
            )
        top_factors_str = "\n".join(top_factors_lines) or "—"

        # Recent events (last 20)
        ev_lines: list[str] = []
        for e in events[-20:]:
            ts = getattr(e, "timestamp", None)
            ts_str = ts.strftime("%Y-%m-%d") if ts is not None else "—"
            code = getattr(e, "error_code", None) or "—"
            desc = (getattr(e, "error_description", None) or "")[:120]
            ev_lines.append(f"- {ts_str} | {code} | {desc}")
        recent_events_str = "\n".join(ev_lines) or "—"

        # Resources
        resources_str = "—"
        if resources is not None:
            parts: list[str] = []
            for attr, label in (
                ("toner_level", "toner"),
                ("drum_level", "drum"),
                ("fuser_level", "fuser"),
                ("mileage", "mileage"),
            ):
                val = getattr(resources, attr, None)
                if val is not None:
                    suffix = "%" if attr != "mileage" else ""
                    parts.append(f"{label}={val}{suffix}")
            resources_str = ", ".join(parts) or "—"

        # Severity weights
        sw = getattr(weights_profile, "severity", None) if weights_profile else None
        severity_weights_str = (
            f"critical={sw.critical}, high={sw.high}, medium={sw.medium}, "
            f"low={sw.low}, info={sw.info}"
            if sw is not None
            else "default"
        )

        # RAG context by top-3 codes (label format: "<code> (<severity>)")
        rag_parts: list[str] = []
        seen: set[str] = set()
        for fc in (health_result.factor_contributions or [])[:3]:
            label = fc.label or ""
            code = label.split(" (")[0].strip() if " (" in label else label.strip()
            if not code or code in seen:
                continue
            seen.add(code)
            descr = ""
            for e in events:
                if getattr(e, "error_code", None) == code:
                    descr = getattr(e, "error_description", "") or ""
                    if descr:
                        break
            try:
                snippet = self._fetch_rag_context(code, descr, top_k=1)
            except Exception:  # noqa: BLE001
                snippet = ""
            if snippet:
                rag_parts.append(f"## {code}\n{snippet}")
        rag_context_str = ("\n\n".join(rag_parts))[:2000] or "(нет фрагментов из документации)"

        prompt_text = prompt_template.format(
            device_id=device_id,
            model=model_name,
            health_index_lite=hi_orig,
            zone=getattr(health_result.zone, "value", str(health_result.zone)),
            confidence=f"{health_result.confidence:.2f}",
            top_factors=top_factors_str,
            recent_events=recent_events_str,
            resources=resources_str,
            severity_weights=severity_weights_str,
            rag_context=rag_context_str,
        )

        response_schema = {
            "type": "object",
            "properties": {
                "health_index_llm": {"type": "integer", "minimum": 1, "maximum": 100},
                "root_cause": {"type": "string"},
                "recommended_action": {"type": "string"},
                "explanation": {"type": "string"},
                "related_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "health_index_llm",
                "root_cause",
                "recommended_action",
                "explanation",
                "related_codes",
            ],
        }

        try:
            from config.loader import LLMGenerationParams
            from llm.client import LLMClient

            params = LLMGenerationParams(temperature=0.3, top_p=1.0, max_tokens=2500)
            resp = self._llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты — инженер сервисной поддержки парка МФУ. "
                            "Анализируешь ОДНО устройство по его событиям, расходникам "
                            "и сервисной документации. Возвращай строго ОДИН JSON-объект "
                            "без Markdown-обёртки, без вступительного текста. "
                            "Запрещены расплывчатые фразы («проверить состояние», "
                            "«необходим мониторинг»). Каждое поле — содержательно."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                tools=None,
                response_schema=response_schema,
                params=params,
            )
            text = LLMClient.strip_reasoning_artifacts(resp.content).strip()
            data = _robust_json_parse(text)
            if data is None:
                raise ValueError("JSON не извлечён из ответа LLM")
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_device_deep failed for %s: %s", device_id, exc)
            return DeepDeviceAnalysis(
                device_id=device_id,
                health_index_original=hi_orig,
                analyzed_at=now,
                llm_calls=1,
                duration_ms=_elapsed_ms(),
                error=f"LLM error: {exc}",
            )

        def _clean(raw: Any, limit: int) -> str:
            return str(raw or "").strip()[:limit]

        try:
            h_llm = int(data.get("health_index_llm", hi_orig))
            h_llm = max(1, min(100, h_llm))
        except (TypeError, ValueError):
            h_llm = hi_orig

        related_raw = data.get("related_codes") or []
        if not isinstance(related_raw, list):
            related_raw = [str(related_raw)]
        related = [
            str(x).strip()[:50]
            for x in related_raw
            if str(x).strip()
        ][:10]

        return DeepDeviceAnalysis(
            device_id=device_id,
            health_index_original=hi_orig,
            health_index_llm=h_llm,
            root_cause=_clean(data.get("root_cause"), 400),
            recommended_action=_clean(data.get("recommended_action"), 400),
            explanation=_clean(data.get("explanation"), 1500),
            related_codes=related,
            reflection_verdict="single_shot",
            llm_calls=1,
            duration_ms=_elapsed_ms(),
            analyzed_at=now,
        )

    def _get_cached_health_result(self, device_id: str) -> HealthResult | None:
        """Return the latest HealthResult from any tool's health_cache.

        CalculateHealthIndexTool appends each invocation to
        ``deps.health_cache[device_id]``. When the LLM fails to emit a
        final JSON we can still surface the real calculation result.
        """
        tools = getattr(self, "_tools", None)
        if tools is None:
            return None
        registry_tools = getattr(tools, "_tools", {}) or {}
        for t in registry_tools.values():
            deps = getattr(t, "_deps", None)
            cache = getattr(deps, "health_cache", None) if deps is not None else None
            if not cache:
                continue
            series = cache.get(device_id)
            if series:
                return series[-1]
        return None

    def _fetch_rag_context(
        self,
        error_code: str,
        description: str,
        *,
        top_k: int = 3,
    ) -> str:
        """Try to pull relevant documentation fragments for a code.

        Strategy:
          1. Exact keyword match in `error_codes` collection via Qdrant
             scroll + payload filter. Display-codes like "71-535-00" are
             numeric and do not embed well — exact match is reliable.
          2. If nothing found, fall back to semantic search via
             search_service_docs tool across available collections.

        Best-effort — returns "" on any failure. Never raises.
        """
        tools = getattr(self, "_tools", None)
        snippets: list[str] = []

        # Extract Qdrant rest_client — try tools first, then singleton fallback
        client = None
        if tools is not None:
            registry_tools = getattr(tools, "_tools", {}) or {}
            for t in registry_tools.values():
                deps = getattr(t, "_deps", None)
                if deps is None:
                    continue
                searcher = getattr(deps, "searcher", None)
                if searcher is None:
                    continue
                qmgr = getattr(searcher, "_qdrant", None)
                if qmgr is not None:
                    client = getattr(qmgr, "rest_client", None)
                    if client is not None:
                        break

        if client is None:
            try:
                from state.singletons import get_qdrant_manager
                qmgr = get_qdrant_manager()
                client = getattr(qmgr, "rest_client", None)
            except Exception:
                client = None

        # 1) Exact keyword match on the error_codes collection
        try:
            if client is not None:
                from qdrant_client.http.models import (
                    FieldCondition,
                    Filter,
                    MatchAny,
                )
                points, _ = client.scroll(
                    "error_codes",
                    scroll_filter=Filter(
                        must=[FieldCondition(
                            key="error_codes",
                            match=MatchAny(any=[error_code]),
                        )],
                    ),
                    limit=top_k,
                    with_payload=True,
                )
                for p in points:
                    text = str((p.payload or {}).get("text", ""))[:400]
                    if text:
                        snippets.append(f"[error_codes reference]\n{text}")
        except Exception:
            logger.warning("RAG keyword lookup failed for %s", error_code, exc_info=True)

        # 2) Semantic fallback via search_service_docs tool
        if not snippets and tools is not None:
            query = f"{error_code} {description}".strip()[:400]
            try:
                result = tools.execute(
                    "search_service_docs",
                    {"query": query, "top_k": top_k},
                )
                if result.success and result.data:
                    for h in (result.data.get("hits") or [])[:top_k]:
                        text = str(h.get("text", ""))[:400]
                        src = h.get("document_id") or h.get("chunk_id") or "rag"
                        if text:
                            snippets.append(f"[{src}]\n{text}")
            except Exception:
                pass

        if not snippets:
            return ""
        return "\n\n---\n\n".join(snippets[:top_k])[:1500]

    _CHAT_RAG_TOP_K = 5
    _CHAT_RAG_SNIPPET_CHARS = 450
    _CHAT_RAG_TOTAL_CHARS = 3000

    def _fetch_chat_rag_context(
        self,
        user_query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """RAG-retrieval for chat: keyword scroll on error_codes + semantic search.

        Returns (formatted_text_block, raw_hits_for_ui). Both empty on any
        failure — never raises.
        """
        top_k = top_k or self._CHAT_RAG_TOP_K
        hits: list[dict[str, Any]] = []
        tools = getattr(self, "_tools", None)

        # 1) If the query mentions a Xerox-style code, keyword-scroll error_codes
        xerox_codes = re.findall(r"\b\d{2}-\d{3}-\d{2}\b", user_query or "")
        client = None
        if xerox_codes:
            if tools is not None:
                registry_tools = getattr(tools, "_tools", {}) or {}
                for t in registry_tools.values():
                    deps = getattr(t, "_deps", None)
                    searcher = getattr(deps, "searcher", None) if deps else None
                    qmgr = getattr(searcher, "_qdrant", None) if searcher else None
                    if qmgr is not None:
                        client = getattr(qmgr, "rest_client", None)
                        if client is not None:
                            break
            if client is None:
                try:
                    from state.singletons import get_qdrant_manager
                    qmgr = get_qdrant_manager()
                    client = getattr(qmgr, "rest_client", None)
                except Exception:
                    client = None

            if client is not None:
                try:
                    from qdrant_client.http.models import (
                        FieldCondition,
                        Filter,
                        MatchAny,
                    )
                    points, _ = client.scroll(
                        "error_codes",
                        scroll_filter=Filter(
                            must=[FieldCondition(
                                key="error_codes",
                                match=MatchAny(any=xerox_codes),
                            )],
                        ),
                        limit=top_k,
                        with_payload=True,
                    )
                    for p in points:
                        payload = p.payload or {}
                        text = str(payload.get("text", ""))[: self._CHAT_RAG_SNIPPET_CHARS]
                        if text:
                            hits.append({
                                "source": "error_codes",
                                "document_id": payload.get("document_id") or "",
                                "score": 1.0,  # keyword match — exact
                                "text": text,
                            })
                except Exception:
                    logger.warning("chat RAG keyword lookup failed", exc_info=True)

        # 2) Semantic search via search_service_docs
        if tools is not None and len(hits) < top_k:
            try:
                remaining = top_k - len(hits)
                result = tools.execute(
                    "search_service_docs",
                    {"query": user_query[:400], "top_k": remaining},
                )
                if result.success and result.data:
                    for h in (result.data.get("hits") or [])[:remaining]:
                        text = str(h.get("text", ""))[: self._CHAT_RAG_SNIPPET_CHARS]
                        if not text:
                            continue
                        hits.append({
                            "source": "service_manuals",
                            "document_id": h.get("document_id") or h.get("chunk_id") or "doc",
                            "score": float(h.get("score", 0.0)),
                            "text": text,
                        })
            except Exception:
                logger.warning("chat RAG semantic search failed", exc_info=True)

        if not hits:
            return "", []

        parts: list[str] = []
        total = 0
        for h in hits:
            block = f"[{h['source']}: {h['document_id']}] (score={h['score']:.2f})\n{h['text']}"
            if total + len(block) > self._CHAT_RAG_TOTAL_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts), hits

    def _build_chat_messages(
        self,
        user_message: str,
        context: ChatContext,
        rag_block: str = "",
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

        if rag_block:
            system_parts.append(
                "═══ Фрагменты из сервисной документации ═══\n" + rag_block
            )
        else:
            system_parts.append(
                "═══ Фрагменты из сервисной документации ═══\n"
                "(по этому запросу в документации ничего не найдено)"
            )

        system_content = "\n\n".join(system_parts)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        history = list(context.conversation_history or [])
        if (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == user_message
        ):
            history = history[:-1]
        if len(history) > self._CHAT_MAX_HISTORY_MESSAGES:
            history = history[-self._CHAT_MAX_HISTORY_MESSAGES:]

        for msg in history:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        history_chars = sum(len(m.get("content") or "") for m in history)
        logger.info(
            "chat prompt built: msgs=%d system=%dch history=%dch(%d msgs) rag=%dch report=%dch",
            len(messages),
            len(system_content),
            history_chars,
            len(history),
            len(rag_block),
            len(report_ctx),
        )
        return messages

    def _serialize_report_context(self, report: Any) -> str:
        """Serialize a Report into a short context block for the chat prompt.

        We ALWAYS emit a compact summary: fleet_summary, executive_summary
        (truncated), red-zone counters and a handful of IDs. Detailed
        queries (device list, factors, mass errors) go through tools —
        keeps the system prompt small enough for 16K-context local models.
        """
        if report is None:
            return ""

        try:
            devices = list(report.devices or [])

            def _zone_str(d: Any) -> str:
                z = d.zone
                return z if isinstance(z, str) else getattr(z, "value", str(z))

            red = [d for d in devices if _zone_str(d) == "red"]
            red_sorted = sorted(red, key=lambda d: d.health_index)

            summary_data: dict[str, Any] = {
                "report_id": report.report_id,
                "fleet_summary": report.fleet_summary.model_dump(mode="json")
                if hasattr(report.fleet_summary, "model_dump")
                else report.fleet_summary,
                "executive_summary": (getattr(report, "executive_summary", "") or "")[
                    :500
                ],
                "red_zone_total": len(red),
                "red_zone_sample_ids": [d.device_id for d in red_sorted[:5]],
                "hint": (
                    "Для списка проблемных устройств вызови "
                    "list_red_zone_devices, для массовых ошибок — list_mass_errors, "
                    "для сводки — get_current_report_summary."
                ),
            }

            compact = json.dumps(summary_data, ensure_ascii=False, default=str)
            if len(compact) > self._REPORT_TOKEN_LIMIT * 4:
                compact = compact[: self._REPORT_TOKEN_LIMIT * 4]
            return f"Последний отчёт (сводка):\n{compact}"
        except Exception:
            logger.warning("report context serialization failed", exc_info=True)
            return ""

    # ── Agent loop ───────────────────────────────────────────────────────

    def _agent_loop(
        self,
        messages: list[dict[str, Any]],
        trace: Trace,
        llm_params: Any = None,
        strip_reasoning_on_final: bool = False,
    ) -> tuple[str, list[TraceStep]]:
        """Run the LLM ↔ tool loop until a final answer or limits hit.

        Returns the final text content and collected trace steps.
        When ``strip_reasoning_on_final`` is True (chat mode), the final
        assistant message is passed through ``strip_reasoning_artifacts``
        to drop <think> blocks and ChatML turn-boundary leaks.
        """
        loop_cfg = self._config.agent
        max_llm_calls = loop_cfg.max_llm_calls_per_attempt
        max_tool_calls = loop_cfg.max_tool_calls_per_attempt

        llm_calls = 0
        tool_calls = 0
        steps: list[TraceStep] = []
        step_counter = len(trace.steps)

        tool_schemas = self._tools.get_all_schemas()
        params = llm_params if llm_params is not None else self._config.llm.batch_mode

        while llm_calls < max_llm_calls:
            t0 = time.perf_counter()
            response = self._llm.generate(
                messages=messages,
                tools=tool_schemas if tool_calls < max_tool_calls else None,
                params=params,
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
                content = response.content or ""
                if strip_reasoning_on_final:
                    from llm.client import LLMClient as _LC
                    content = _LC.strip_reasoning_artifacts(content)
                return content, steps

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
                # Small-context LLMs (8K tokens) die on long tool outputs.
                # 3000 chars ≈ 1000 tokens per tool message — safe for 8K budget.
                if len(result_json) > 3000:
                    result_json = result_json[:3000] + '..."}'

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
            preview = (content or "")[:800].replace("\n", " ⏎ ")
            logger.warning(
                "Не удалось распарсить финальный JSON для %s. "
                "Raw LLM content (первые 800 символов): %s",
                device_id, preview,
            )
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
        """Tolerant JSON extraction from LLM output.

        Handles: <think>...</think> blocks, leading/trailing prose,
        markdown code fences, extra data after the object, and strings
        truncated mid-content (max_tokens hit).
        """
        if not text:
            return None

        # Strip reasoning artifacts (<think> blocks, etc.)
        try:
            from llm.client import LLMClient
            cleaned = LLMClient.strip_reasoning_artifacts(text)
        except Exception:
            cleaned = text

        return _robust_json_parse(cleaned)

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
