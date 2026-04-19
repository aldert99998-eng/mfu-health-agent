"""Agent tool implementations — Track B, Level 1.

All 9 tools from the Track B tool-set, each with Pydantic validation
and structured ToolResult output.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from agent.tools.registry import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from agent.memory import MemoryManager
    from data_io.factor_store import FactorStore
    from data_io.models import HealthResult, WeightsProfile
    from llm.client import LLMClient
    from rag.search import HybridSearcher

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


# ── Shared dependencies container ────────────────────────────────────────────


class ToolDependencies:
    """Runtime dependencies injected into tools at registration time."""

    def __init__(
        self,
        *,
        factor_store: FactorStore,
        weights: WeightsProfile,
        searcher: HybridSearcher | None = None,
        llm_client: LLMClient | None = None,
        collection: str = "service_manuals",
        health_cache: dict[str, list[HealthResult]] | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.factor_store = factor_store
        self.weights = weights
        self.searcher = searcher
        self.llm_client = llm_client
        self.collection = collection
        self.health_cache: dict[str, list[HealthResult]] = health_cache or {}
        self.memory_manager = memory_manager


# ── 1. SearchServiceDocsTool ─────────────────────────────────────────────────


class _SearchArgs(BaseModel):
    query: str
    model: str | None = None
    content_type: str | None = None
    top_k: int = Field(default=5, ge=1, le=30)


class SearchServiceDocsTool:
    """Semantic search over MFU service documentation."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "search_service_docs"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Семантический поиск по сервисной документации МФУ. "
                    "Возвращает релевантные фрагменты с метаданными."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Текст запроса: код ошибки, симптом, название узла."},
                        "model": {"type": "string", "description": "Модель МФУ для фильтрации."},
                        "content_type": {
                            "type": "string",
                            "enum": ["symptom", "cause", "procedure", "specification", "reference"],
                            "description": "Тип содержимого для фильтрации.",
                        },
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _SearchArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        if self._deps.searcher is None:
            return ToolResult(success=False, error="RAG searcher not available")

        filters: dict[str, Any] = {}
        if parsed.model:
            filters["model"] = parsed.model
        if parsed.content_type:
            filters["doc_type"] = parsed.content_type

        try:
            results = self._deps.searcher.search(
                parsed.query,
                self._deps.collection,
                top_k=parsed.top_k,
                filters=filters if filters else None,
                use_reranker=True,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Search error: {exc}")

        hits = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text[:500],
                "score": round(r.score, 4),
                "document_id": r.document_id,
                "payload": {
                    k: v for k, v in r.payload.items()
                    if k in ("vendor", "model", "error_codes", "doc_type", "page", "section")
                },
            }
            for r in results
        ]
        return ToolResult(success=True, data={"hits": hits, "total": len(hits)})


# ── 2. ClassifyErrorSeverityTool ─────────────────────────────────────────────


class _ClassifyArgs(BaseModel):
    error_code: str
    model: str | None = None
    error_description: str | None = None


_CLASSIFY_PROMPT_PATH = _PROMPTS_DIR / "classify_severity.md"

_CLASSIFY_FALLBACK = {
    "severity": "Medium",
    "confidence": 0.3,
    "affected_components": [],
    "source": None,
    "reasoning": "Документация не найдена, используется default.",
}

# Xerox service codes with known severity (heuristic fallback when RAG is empty)
_XEROX_HEURISTIC: dict[str, tuple[str, list[str]]] = {
    # Critical — hardware failures requiring service call
    "09-594-00": ("Critical", ["fuser"]),
    "09-604-00": ("Critical", ["fuser"]),
    "09-605-00": ("Critical", ["fuser"]),
    "05-310-00": ("Critical", ["main_board"]),
    "05-320-00": ("Critical", ["main_board"]),
    # High — consumable replacement / paper path issues
    "75-530-00": ("High", ["toner"]),
    "72-535-00": ("High", ["paper_path"]),
    "73-530-00": ("High", ["paper_path"]),
    "73-535-00": ("High", ["paper_path"]),
    "17-562-00": ("High", ["drum"]),
    "17-570-00": ("High", ["drum"]),
    "07-535-00": ("High", ["paper_tray"]),
}

# Prefix-based heuristic for unknown Xerox codes
_XEROX_PREFIX_SEVERITY: dict[str, str] = {
    "09": "Critical",   # 09-xxx — hardware/fuser
    "05": "Critical",   # 05-xxx — controller
    "75": "High",       # 75-xxx — consumables
    "72": "High",       # 72-xxx — paper jam
    "73": "High",       # 73-xxx — paper jam
    "17": "High",       # 17-xxx — drum/imaging
    "07": "Medium",     # 07-xxx — tray
    "10": "Medium",     # 10-xxx — finisher
}


def _heuristic_classify(error_code: str) -> dict[str, Any] | None:
    """Return heuristic severity for known Xerox codes, or None."""
    import re

    if not re.match(r"^\d{2}-\d{3}-\d{2}$", error_code):
        return None

    if error_code in _XEROX_HEURISTIC:
        sev, comps = _XEROX_HEURISTIC[error_code]
        return {
            "severity": sev,
            "confidence": 0.6,
            "affected_components": comps,
            "source": "heuristic_xerox_lookup",
            "reasoning": f"Known Xerox code {error_code}, heuristic severity.",
        }

    prefix = error_code[:2]
    if prefix in _XEROX_PREFIX_SEVERITY:
        return {
            "severity": _XEROX_PREFIX_SEVERITY[prefix],
            "confidence": 0.4,
            "affected_components": [],
            "source": "heuristic_xerox_prefix",
            "reasoning": f"Xerox prefix {prefix}-xxx heuristic, no exact match.",
        }

    return None

_CLASSIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["Critical", "High", "Medium", "Low", "Info"]},
        "confidence": {"type": "number"},
        "affected_components": {"type": "array", "items": {"type": "string"}},
        "source": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["severity", "confidence", "affected_components", "reasoning"],
}


class ClassifyErrorSeverityTool:
    """Classify error severity via RAG search + LLM classification."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps
        self._cache: dict[str, dict[str, Any]] = {}
        self._prompt_template = self._load_prompt()

    @property
    def name(self) -> str:
        return "classify_error_severity"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Классифицирует критичность ошибки по сервисной документации. "
                    "Возвращает уровень (Critical/High/Medium/Low/Info), "
                    "источник, список затронутых узлов."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "error_code": {"type": "string"},
                        "model": {"type": "string"},
                        "error_description": {"type": "string"},
                    },
                    "required": ["error_code"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _ClassifyArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        cache_key = f"{parsed.error_code}|{parsed.model or ''}"
        if cache_key in self._cache:
            return ToolResult(success=True, data=self._cache[cache_key])

        context_text = ""
        if self._deps.searcher is not None:
            query = parsed.error_code
            if parsed.error_description:
                query += " " + parsed.error_description

            filters: dict[str, Any] = {}
            if parsed.model:
                filters["model"] = parsed.model

            try:
                results = self._deps.searcher.search(
                    query,
                    self._deps.collection,
                    top_k=5,
                    filters=filters if filters else None,
                    use_reranker=True,
                )
                if not results and filters:
                    results = self._deps.searcher.search(
                        query,
                        self._deps.collection,
                        top_k=5,
                        filters=None,
                        use_reranker=True,
                    )
                if results:
                    context_text = "\n\n---\n\n".join(
                        f"[{r.document_id}] (score={r.score:.3f})\n{r.text[:400]}"
                        for r in results
                    )
            except Exception:
                logger.warning("RAG search failed for %s", parsed.error_code, exc_info=True)

        if self._deps.llm_client is None:
            if not context_text:
                heuristic = _heuristic_classify(parsed.error_code)
                result_data = heuristic if heuristic else dict(_CLASSIFY_FALLBACK)
            else:
                result_data = dict(_CLASSIFY_FALLBACK)
                result_data["reasoning"] = "LLM client not available, используется default."
            self._cache[cache_key] = result_data
            return ToolResult(success=True, data=result_data)

        if not context_text and not parsed.error_description:
            heuristic = _heuristic_classify(parsed.error_code)
            result_data = heuristic if heuristic else dict(_CLASSIFY_FALLBACK)
            self._cache[cache_key] = result_data
            return ToolResult(success=True, data=result_data)

        if not context_text and parsed.error_description:
            context_text = (
                f"Описание из журнала устройства:\n{parsed.error_description}"
            )

        prompt_text = self._prompt_template.format(
            error_code=parsed.error_code,
            model=parsed.model or "неизвестна",
            error_description=parsed.error_description or "нет описания",
            context=context_text,
        )

        try:
            from config.loader import LLMGenerationParams

            params = LLMGenerationParams(temperature=0.0, top_p=1.0, max_tokens=300)
            resp = self._deps.llm_client.generate(
                messages=[
                    {"role": "system", "content": "Ты — классификатор критичности ошибок МФУ. Отвечай строго JSON."},
                    {"role": "user", "content": prompt_text},
                ],
                response_schema=_CLASSIFY_RESPONSE_SCHEMA,
                params=params,
            )

            text = resp.content.strip()
            if text.startswith("```"):
                import re
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
                text = text.strip()

            result_data = json.loads(text)
            if "severity" not in result_data:
                result_data = dict(_CLASSIFY_FALLBACK)
                result_data["reasoning"] = "LLM вернул невалидный JSON."
        except Exception:
            logger.warning("LLM classify failed for %s", parsed.error_code, exc_info=True)
            result_data = dict(_CLASSIFY_FALLBACK)
            result_data["reasoning"] = "LLM classify failed, используется default."

        self._cache[cache_key] = result_data
        return ToolResult(success=True, data=result_data)

    @staticmethod
    def _load_prompt() -> str:
        if _CLASSIFY_PROMPT_PATH.exists():
            return _CLASSIFY_PROMPT_PATH.read_text(encoding="utf-8")
        return (
            "Определи критичность ошибки {error_code} для модели {model}.\n"
            "Описание ошибки: {error_description}\n\n"
            "Контекст из документации:\n{context}\n\n"
            "Верни JSON: {{\"severity\": ..., \"confidence\": ..., "
            "\"affected_components\": [...], \"source\": ..., \"reasoning\": ...}}"
        )


# ── 3. GetDeviceEventsTool ───────────────────────────────────────────────────


class _DeviceEventsArgs(BaseModel):
    device_id: str
    window_days: int = Field(default=30, ge=1, le=365)


class GetDeviceEventsTool:
    """Return device events from FactorStore."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "get_device_events"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Возвращает все события (ошибки) устройства за заданное окно в днях из factor-store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "window_days": {"type": "integer", "default": 30},
                    },
                    "required": ["device_id"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _DeviceEventsArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        events = self._deps.factor_store.get_events(parsed.device_id, window_days=parsed.window_days)
        total = len(events)
        events = events[-50:]
        data = [
            {
                "device_id": e.device_id,
                "timestamp": e.timestamp.isoformat(),
                "error_code": e.error_code,
                "error_description": e.error_description,
                "model": e.model,
                "vendor": e.vendor,
                "location": e.location,
                "status": e.status,
            }
            for e in events
        ]
        return ToolResult(success=True, data={"events": data, "total": total, "truncated": total > 50})


# ── 4. GetDeviceResourcesTool ────────────────────────────────────────────────


class _DeviceResourcesArgs(BaseModel):
    device_id: str


class GetDeviceResourcesTool:
    """Return latest resource snapshot for a device."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "get_device_resources"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Возвращает последний известный снимок ресурсов устройства (тонер, барабан, фьюзер, пробег).",
                "parameters": {
                    "type": "object",
                    "properties": {"device_id": {"type": "string"}},
                    "required": ["device_id"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _DeviceResourcesArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        snap = self._deps.factor_store.get_resources(parsed.device_id)
        if snap is None:
            return ToolResult(success=True, data=None)

        return ToolResult(success=True, data={
            "device_id": snap.device_id,
            "timestamp": snap.timestamp.isoformat(),
            "toner_level": snap.toner_level,
            "drum_level": snap.drum_level,
            "fuser_level": snap.fuser_level,
            "mileage": snap.mileage,
            "service_interval": snap.service_interval,
        })


# ── 5. CountErrorRepetitionsTool ─────────────────────────────────────────────


class _CountRepArgs(BaseModel):
    device_id: str
    error_code: str
    window_days: int = Field(default=14, ge=1, le=365)


class CountErrorRepetitionsTool:
    """Count repetitions of a specific error code."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "count_error_repetitions"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Подсчитывает количество повторений одного и того же кода ошибки за окно.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "error_code": {"type": "string"},
                        "window_days": {"type": "integer", "default": 14},
                    },
                    "required": ["device_id", "error_code"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _CountRepArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        count = self._deps.factor_store.count_repetitions(
            parsed.device_id, parsed.error_code, window_days=parsed.window_days,
        )
        return ToolResult(success=True, data={"count": count})


# ── 6. CalculateHealthIndexTool ──────────────────────────────────────────────


class _FactorInput(BaseModel):
    error_code: str
    severity_level: str
    n_repetitions: int = Field(default=1, ge=1)
    event_timestamp: str
    applicable_modifiers: list[str] = Field(default_factory=list)
    source: str | None = None


class _CalcArgs(BaseModel):
    device_id: str
    factors: list[_FactorInput]
    confidence_factors: dict[str, Any] = Field(default_factory=dict)


class CalculateHealthIndexTool:
    """Deterministic health index calculation via the Track A formula."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "calculate_health_index"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Детерминированный расчёт индекса здоровья по формуле. "
                    "Принимает массив факторов, возвращает индекс, confidence, разложение."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "factors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "error_code": {"type": "string"},
                                    "severity_level": {"type": "string"},
                                    "n_repetitions": {"type": "integer"},
                                    "event_timestamp": {"type": "string", "format": "date-time"},
                                    "applicable_modifiers": {"type": "array", "items": {"type": "string"}},
                                    "source": {"type": "string"},
                                },
                            },
                        },
                        "confidence_factors": {
                            "type": "object",
                            "properties": {
                                "rag_missing_count": {"type": "integer"},
                                "missing_resources": {"type": "boolean"},
                                "missing_model": {"type": "boolean"},
                            },
                        },
                    },
                    "required": ["device_id", "factors"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _CalcArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        from data_io.models import ConfidenceFactors, Factor, SeverityLevel
        from tools.calculator import (
            calculate_health_index,
            compute_A,
            compute_C,
            compute_R,
        )

        weights = self._deps.weights
        now = datetime.now(UTC)

        built_factors: list[Factor] = []
        for fi in parsed.factors:
            try:
                severity = SeverityLevel(fi.severity_level)
            except ValueError:
                severity = SeverityLevel.MEDIUM

            S = getattr(weights.severity, severity.value.lower(), 10.0)
            ts = datetime.fromisoformat(fi.event_timestamp)
            age_days = max(0, (now - ts).days)

            R = compute_R(fi.n_repetitions, weights.repeatability.base, weights.repeatability.max_value)

            modifiers: list[float] = []
            for mod_name in fi.applicable_modifiers:
                mod = weights.context.modifiers.get(mod_name)
                if mod:
                    modifiers.append(mod.multiplier)
            C = compute_C(modifiers, weights.context.max_value)

            A = compute_A(age_days, weights.age.tau_days)

            built_factors.append(Factor(
                error_code=fi.error_code,
                severity_level=severity,
                S=S,
                n_repetitions=fi.n_repetitions,
                R=R,
                C=C,
                A=A,
                event_timestamp=ts,
                age_days=age_days,
                applicable_modifiers=fi.applicable_modifiers,
                source=fi.source,
            ))

        cf_data = parsed.confidence_factors
        confidence_factors = ConfidenceFactors(
            rag_missing_count=cf_data.get("rag_missing_count", 0),
            missing_resources=cf_data.get("missing_resources", False),
            missing_model=cf_data.get("missing_model", False),
        )

        result = calculate_health_index(
            built_factors, confidence_factors, weights, device_id=parsed.device_id,
        )

        self._deps.health_cache.setdefault(parsed.device_id, []).append(result)

        return ToolResult(success=True, data={
            "device_id": result.device_id,
            "health_index": result.health_index,
            "confidence": result.confidence,
            "zone": result.zone.value,
            "confidence_zone": result.confidence_zone.value,
            "factor_contributions": [
                {"label": fc.label, "penalty": fc.penalty, "S": fc.S, "R": fc.R, "C": fc.C, "A": fc.A, "source": fc.source}
                for fc in result.factor_contributions
            ],
            "confidence_reasons": result.confidence_reasons,
        })


# ── 7. GetFleetStatisticsTool ────────────────────────────────────────────────


class _FleetStatsArgs(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)


class GetFleetStatisticsTool:
    """Aggregate fleet statistics from factor_store + health cache."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "get_fleet_statistics"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Сводная статистика по парку: распределение индексов, зоны, средние по моделям/локациям.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "location": {"type": "string"},
                                "zone": {"type": "string", "enum": ["green", "yellow", "red"]},
                            },
                        },
                    },
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _FleetStatsArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        fs = self._deps.factor_store
        cache = self._deps.health_cache
        devices = fs.list_devices()

        zone_counts: Counter[str] = Counter()
        model_indices: dict[str, list[int]] = defaultdict(list)
        location_indices: dict[str, list[int]] = defaultdict(list)
        device_summaries: list[dict[str, Any]] = []

        filter_model = parsed.filters.get("model")
        filter_location = parsed.filters.get("location")
        filter_zone = parsed.filters.get("zone")

        for did in devices:
            meta = fs.get_device_metadata(did)
            dev_model = meta.model if meta else None
            dev_location = meta.location if meta else None

            if filter_model and dev_model and filter_model.lower() not in dev_model.lower():
                continue
            if filter_location and dev_location and filter_location.lower() not in dev_location.lower():
                continue

            history = cache.get(did, [])
            if not history:
                continue
            latest = history[-1]

            zone_str = latest.zone.value.lower()
            if filter_zone and zone_str != filter_zone.lower():
                continue

            zone_counts[zone_str] += 1
            if dev_model:
                model_indices[dev_model].append(latest.health_index)
            if dev_location:
                location_indices[dev_location].append(latest.health_index)

            device_summaries.append({
                "device_id": did,
                "health_index": latest.health_index,
                "zone": zone_str,
                "confidence": latest.confidence,
                "model": dev_model,
                "location": dev_location,
            })

        total = len(device_summaries)
        avg_index = round(sum(d["health_index"] for d in device_summaries) / total, 1) if total else 0.0

        model_avg = {
            m: round(sum(v) / len(v), 1)
            for m, v in sorted(model_indices.items())
        }
        location_avg = {
            loc: round(sum(v) / len(v), 1)
            for loc, v in sorted(location_indices.items())
        }

        return ToolResult(success=True, data={
            "total_devices": total,
            "average_health_index": avg_index,
            "zone_distribution": dict(zone_counts),
            "by_model": model_avg,
            "by_location": location_avg,
            "devices": device_summaries[:50],
        })


# ── 8. FindSimilarDevicesTool ────────────────────────────────────────────────


class _SimilarArgs(BaseModel):
    device_id: str
    similarity_dim: str = Field(default="errors", pattern=r"^(errors|model|location|error_and_model)$")


class FindSimilarDevicesTool:
    """Find devices with similar problem patterns (naive binary-vector KNN)."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "find_similar_devices"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Находит устройства со схожими паттернами проблем. Для поиска массовых проблем и локационных паттернов.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "similarity_dim": {
                            "type": "string",
                            "enum": ["errors", "model", "location", "error_and_model"],
                            "default": "errors",
                        },
                    },
                    "required": ["device_id"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _SimilarArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        fs = self._deps.factor_store
        target_events = fs.get_events(parsed.device_id, window_days=30)
        target_meta = fs.get_device_metadata(parsed.device_id)

        if not target_events and not target_meta:
            return ToolResult(success=True, data={"similar": [], "reason": "No data for target device"})

        target_codes = {e.error_code for e in target_events if e.error_code}
        target_model = target_meta.model if target_meta else None
        target_location = target_meta.location if target_meta else None

        all_devices = fs.list_devices()
        scored: list[dict[str, Any]] = []

        for did in all_devices:
            if did == parsed.device_id:
                continue

            meta = fs.get_device_metadata(did)
            events = fs.get_events(did, window_days=30)
            codes = {e.error_code for e in events if e.error_code}
            dev_model = meta.model if meta else None
            dev_location = meta.location if meta else None

            score = 0.0

            if parsed.similarity_dim in ("errors", "error_and_model") and target_codes and codes:
                intersection = target_codes & codes
                union = target_codes | codes
                score += len(intersection) / len(union) if union else 0.0

            if parsed.similarity_dim in ("model", "error_and_model") and target_model and dev_model and target_model == dev_model:
                score += 0.5

            if parsed.similarity_dim == "location" and target_location and dev_location and target_location == dev_location:
                score += 1.0

            if score > 0:
                latest = self._deps.health_cache.get(did, [None])[-1]
                scored.append({
                    "device_id": did,
                    "similarity_score": round(score, 3),
                    "shared_errors": sorted(target_codes & codes) if target_codes and codes else [],
                    "model": dev_model,
                    "location": dev_location,
                    "health_index": latest.health_index if latest else None,
                })

        scored.sort(key=lambda x: x["similarity_score"], reverse=True)
        return ToolResult(success=True, data={"similar": scored[:20]})


# ── 9. GetDeviceHistoryTool ──────────────────────────────────────────────────


class _DeviceHistoryArgs(BaseModel):
    device_id: str
    limit: int = Field(default=10, ge=1, le=100)


class GetDeviceHistoryTool:
    """Return health index calculation history for a device."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "get_device_history"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "История рассчитанных индексов устройства. Используется для self-check (резкие скачки) и трендов.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["device_id"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _DeviceHistoryArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        history = self._deps.health_cache.get(parsed.device_id, [])
        entries = history[-parsed.limit:]

        data = [
            {
                "health_index": h.health_index,
                "confidence": h.confidence,
                "zone": h.zone.value,
                "calculated_at": h.calculated_at.isoformat(),
                "num_factors": len(h.factor_contributions),
            }
            for h in entries
        ]
        return ToolResult(success=True, data={"history": data, "total": len(data)})


# ── 10. GetLearnedPatternsTool ─────────────────────────────────────────────


class _LearnedPatternsArgs(BaseModel):
    model: str | None = None


class GetLearnedPatternsTool:
    """Return learned patterns from MemoryManager."""

    def __init__(self, deps: ToolDependencies) -> None:
        self._deps = deps

    @property
    def name(self) -> str:
        return "get_learned_patterns"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Возвращает ранее обнаруженные закономерности в парке. "
                    "Можно фильтровать по модели устройства."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Модель МФУ для фильтрации паттернов. Если null — все паттерны.",
                        },
                    },
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        try:
            parsed = _LearnedPatternsArgs.model_validate(args)
        except ValidationError as exc:
            return ToolResult(success=False, error=f"Invalid args: {exc}")

        if self._deps.memory_manager is None:
            return ToolResult(success=True, data={"patterns": [], "total": 0})

        patterns = self._deps.memory_manager.get_patterns(scope=parsed.model)
        data = [
            {
                "type": p.type,
                "scope": p.scope,
                "observation": p.observation,
                "evidence_devices": p.evidence_devices,
            }
            for p in patterns
        ]
        return ToolResult(success=True, data={"patterns": data, "total": len(data)})


# ── Registration ─────────────────────────────────────────────────────────────


def register_all_tools(
    registry: ToolRegistry,
    deps: ToolDependencies,
) -> None:
    """Register all 10 agent tools into the given registry."""
    tools = [
        SearchServiceDocsTool(deps),
        ClassifyErrorSeverityTool(deps),
        GetDeviceEventsTool(deps),
        GetDeviceResourcesTool(deps),
        CountErrorRepetitionsTool(deps),
        CalculateHealthIndexTool(deps),
        GetFleetStatisticsTool(deps),
        FindSimilarDevicesTool(deps),
        GetDeviceHistoryTool(deps),
        GetLearnedPatternsTool(deps),
    ]
    for tool in tools:
        registry.register(tool)  # type: ignore[arg-type]

    logger.info("Зарегистрировано %d agent tools", len(tools))
