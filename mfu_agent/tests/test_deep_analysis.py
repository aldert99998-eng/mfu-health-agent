"""Unit tests for deep LLM analysis: Agent.analyze_mass_error and session helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from agent.core import Agent
from data_io.models import (
    AgentMode,
    DeepDeviceAnalysis,
    MassErrorAnalysis,
    Trace,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_agent_with_llm_mock(llm_response_content: str) -> Agent:
    """Build an Agent whose LLM returns the given content string."""
    from llm.client import LLMResponse, TokenUsage

    agent = Agent.__new__(Agent)
    agent._llm = MagicMock()
    agent._llm.generate.return_value = LLMResponse(
        content=llm_response_content,
        tool_calls=None,
        usage=TokenUsage(),
    )
    agent._tools = MagicMock()
    agent._factor_store = MagicMock()
    agent._config = MagicMock()
    agent._memory = None
    return agent


# ── MassErrorAnalysis model ─────────────────────────────────────────────────


class TestMassErrorAnalysisModel:
    def test_defaults(self) -> None:
        m = MassErrorAnalysis(error_code="75-530-00", analyzed_at=datetime.now(UTC))
        assert m.is_systemic is False
        assert m.affected_device_count == 0
        assert m.error is None

    def test_roundtrip(self) -> None:
        now = datetime.now(UTC)
        m = MassErrorAnalysis(
            error_code="14-517-00",
            description="Scanner error",
            affected_device_count=42,
            total_occurrences=500,
            is_systemic=True,
            likely_cause="Массовый дефект сканерного блока",
            recommended_action="Сервисный осмотр всех устройств",
            explanation="Подробный разбор...",
            analyzed_at=now,
        )
        data = m.model_dump(mode="json")
        m2 = MassErrorAnalysis.model_validate(data)
        assert m2.is_systemic is True
        assert m2.affected_device_count == 42


# ── DeepDeviceAnalysis model ────────────────────────────────────────────────


class TestDeepDeviceAnalysisModel:
    def test_defaults(self) -> None:
        d = DeepDeviceAnalysis(
            device_id="DEV001",
            health_index_original=25,
            analyzed_at=datetime.now(UTC),
        )
        assert d.health_index_llm is None
        assert d.error is None
        assert d.related_codes == []

    def test_bounded_health_index(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DeepDeviceAnalysis(
                device_id="X", health_index_original=200,
                analyzed_at=datetime.now(UTC),
            )


# ── Agent.analyze_mass_error ────────────────────────────────────────────────


class TestAnalyzeMassError:
    def test_happy_path_systemic(self) -> None:
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": True,
            "what_is_this": "Print Cartridge (R2) near end of life.",
            "why_this_pattern": "Централизованная проблема поставки тонера: 150 уст одной партии.",
            "business_impact": "Через 1-3 недели встанут десятки устройств.",
            "immediate_action": "Аудит контракта с поставщиком расходников.",
            "long_term_action": "Алерт при появлении у >5 новых устройств.",
            "indicators_to_watch": ["число уст >10/нед", "остаток на складе <20"],
        }))

        result = agent.analyze_mass_error(
            error_code="09-594-00",
            description="Print Cartridge near end of life",
            affected_count=150,
            total_occurrences=2500,
            fleet_total=319,
            severity="Medium",
            sample_device_ids=["A", "B", "C"],
            sample_descriptions=["desc1", "desc2"],
        )

        assert isinstance(result, MassErrorAnalysis)
        assert result.is_systemic is True
        assert result.error_code == "09-594-00"
        assert result.affected_device_count == 150
        assert "поставки" in result.why_this_pattern
        # legacy back-fill
        assert "поставки" in result.likely_cause
        assert result.error is None

    def test_llm_returns_think_prefix(self) -> None:
        """LLM may wrap JSON in <think>…</think>; must be stripped."""
        raw = (
            "<think>рассуждения модели о коде 75-530-00</think>\n"
            + json.dumps({
                "is_systemic": False,
                "likely_cause": "Разовые случаи",
                "recommended_action": "Следить",
                "explanation": "ok",
            })
        )
        agent = _make_agent_with_llm_mock(raw)
        result = agent.analyze_mass_error(
            error_code="75-530-00",
            description="Tray empty",
            affected_count=5,
            total_occurrences=20,
            fleet_total=319,
        )
        assert result.is_systemic is False
        assert result.error is None

    def test_llm_returns_markdown_codefence(self) -> None:
        raw = "```json\n" + json.dumps({
            "is_systemic": True,
            "likely_cause": "Systemic",
            "recommended_action": "Fix all",
            "explanation": "Long text",
        }) + "\n```"
        agent = _make_agent_with_llm_mock(raw)
        result = agent.analyze_mass_error(
            error_code="14-517-00",
            description="Scanner error",
            affected_count=30,
            total_occurrences=100,
            fleet_total=319,
        )
        assert result.is_systemic is True
        assert result.error is None

    def test_invalid_json_returns_error_record(self) -> None:
        agent = _make_agent_with_llm_mock("мусор, а не JSON")
        result = agent.analyze_mass_error(
            error_code="22-585-00",
            description="System error",
            affected_count=10,
            total_occurrences=50,
            fleet_total=319,
        )
        assert result.error is not None
        assert result.is_systemic is False
        assert result.error_code == "22-585-00"

    def test_rag_context_failure_does_not_break_analysis(self) -> None:
        """If RAG lookup raises, analysis still works via description only."""
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": False,
            "likely_cause": "Точечные случаи",
            "recommended_action": "Наблюдать",
            "explanation": "ок",
        }))
        # Break tools so _fetch_rag_context's scroll+search both raise
        agent._tools.execute.side_effect = RuntimeError("RAG down")
        agent._tools._deps = None

        result = agent.analyze_mass_error(
            error_code="71-535-00",
            description="Tray 1 is nearly empty",
            affected_count=100, total_occurrences=500, fleet_total=319,
        )
        assert result.error is None
        assert result.is_systemic is False

    def test_no_llm_returns_error_record(self) -> None:
        agent = Agent.__new__(Agent)
        agent._llm = None
        agent._tools = MagicMock()
        agent._factor_store = MagicMock()
        agent._config = MagicMock()
        agent._memory = None

        result = agent.analyze_mass_error(
            error_code="09-594-00",
            description="",
            affected_count=1,
            total_occurrences=1,
            fleet_total=319,
        )
        assert result.error == "LLM недоступен"
        assert result.is_systemic is False

    def test_extra_data_after_object(self) -> None:
        """LLM may emit object then repeat it — 'Extra data: line 7 column 1'."""
        raw = json.dumps({
            "is_systemic": True,
            "what_is_this": "Расшифровка кода A для тестирования robust JSON парсера.",
            "why_this_pattern": "Распределение B с подробностями для проверки.",
            "business_impact": "Влияние C на работу парка.",
            "immediate_action": "Сейчас D — конкретное действие.",
            "long_term_action": "Потом E — профилактика на неделю.",
            "indicators_to_watch": ["f1", "f2"],
        }) + "\n\n" + json.dumps({"extra": "ignored"})
        agent = _make_agent_with_llm_mock(raw)
        result = agent.analyze_mass_error(
            error_code="TEST",
            description="",
            affected_count=1, total_occurrences=1, fleet_total=319,
        )
        assert result.error is None
        assert result.is_systemic is True
        assert result.what_is_this.startswith("Расшифровка")
        assert len(result.indicators_to_watch) == 2

    def test_truncated_string_recovered(self) -> None:
        """LLM hits max_tokens mid-string — 'Unterminated string'."""
        raw = (
            '{"is_systemic": true, '
            '"what_is_this": "Print Cartridge near end of life, полное описание", '
            '"why_this_pattern": "проблема поставки тонера у 150 устройств", '
            '"business_impact": "риск остановки парка", '
            '"immediate_action": "аудит поставщика", '
            '"long_term_action": "Эта ошибка массово встре'  # cut mid-string
        )
        agent = _make_agent_with_llm_mock(raw)
        result = agent.analyze_mass_error(
            error_code="TEST",
            description="",
            affected_count=1, total_occurrences=1, fleet_total=319,
        )
        assert result.error is None, f"Should recover, got: {result.error}"
        assert result.is_systemic is True
        assert "поставки" in result.why_this_pattern

    def test_preamble_text_before_json(self) -> None:
        raw = "Вот результат анализа:\n" + json.dumps({
            "is_systemic": False,
            "likely_cause": "x",
            "recommended_action": "y",
            "explanation": "z",
        })
        agent = _make_agent_with_llm_mock(raw)
        result = agent.analyze_mass_error(
            error_code="TEST",
            description="",
            affected_count=1, total_occurrences=1, fleet_total=319,
        )
        assert result.error is None
        assert result.is_systemic is False

    def test_fields_are_truncated(self) -> None:
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": True,
            "what_is_this": "x" * 2000,
            "why_this_pattern": "y" * 2000,
            "business_impact": "z" * 2000,
            "immediate_action": "a" * 2000,
            "long_term_action": "b" * 2000,
            "indicators_to_watch": ["c" * 500] * 10,
        }))
        result = agent.analyze_mass_error(
            error_code="X", description="d",
            affected_count=1, total_occurrences=1, fleet_total=100,
        )
        assert len(result.what_is_this) <= 300
        assert len(result.why_this_pattern) <= 300
        assert len(result.business_impact) <= 300
        assert len(result.immediate_action) <= 300
        assert len(result.long_term_action) <= 300
        assert len(result.indicators_to_watch) <= 5
        assert all(len(ind) <= 150 for ind in result.indicators_to_watch)
        assert len(result.likely_cause) <= 500
        assert len(result.recommended_action) <= 500
        assert len(result.explanation) <= 2000

    def test_six_fields_parsed(self) -> None:
        """New structured fields round-trip through LLM response."""
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": True,
            "what_is_this": "Print Cartridge (R2) near end of life.",
            "why_this_pattern": "87 устройств, 4.7 срабатываний/уст — синхронный износ партии.",
            "business_impact": "Риск остановки десятков устройств одновременно через 1-3 недели.",
            "immediate_action": "Сверить список с остатком картриджей на складе.",
            "long_term_action": "Алерт при >5 новых устройств с кодом в сутки.",
            "indicators_to_watch": [
                "число устройств >10/нед",
                "появление 09-596-00",
                "остаток на складе <20",
            ],
        }))
        result = agent.analyze_mass_error(
            error_code="09-594-00",
            description="Print Cartridge near end of life",
            affected_count=87, total_occurrences=412, fleet_total=319,
            severity="Medium",
        )
        assert result.error is None
        assert "Print Cartridge" in result.what_is_this
        assert "87" in result.why_this_pattern or "синхр" in result.why_this_pattern
        assert result.business_impact
        assert result.immediate_action
        assert result.long_term_action
        assert len(result.indicators_to_watch) == 3
        assert any("09-596" in ind for ind in result.indicators_to_watch)

    def test_legacy_fields_populated_from_new(self) -> None:
        """likely_cause / recommended_action / explanation back-fill works."""
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": False,
            "what_is_this": "Это расшифровка кода.",
            "why_this_pattern": "Это объяснение паттерна.",
            "business_impact": "Это влияние.",
            "immediate_action": "Это срочное действие.",
            "long_term_action": "Это профилактика.",
            "indicators_to_watch": ["i1"],
        }))
        result = agent.analyze_mass_error(
            error_code="X", description="",
            affected_count=1, total_occurrences=1, fleet_total=100,
        )
        # likely_cause берётся из why_this_pattern
        assert "объяснение паттерна" in result.likely_cause
        # recommended_action — из immediate_action
        assert "срочное действие" in result.recommended_action
        # explanation содержит все секции
        assert "Что это" in result.explanation
        assert "Почему массово" in result.explanation
        assert "Влияние" in result.explanation

    def test_short_what_is_this_falls_back_to_description(self) -> None:
        """Если LLM дал пустой/короткий what_is_this, fallback на description."""
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": True,
            "what_is_this": "",  # LLM не справился
            "why_this_pattern": "Массовое распределение.",
            "business_impact": "Некоторое влияние.",
            "immediate_action": "Действие.",
            "long_term_action": "Профилактика.",
            "indicators_to_watch": ["ind1"],
        }))
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None
        result = agent.analyze_mass_error(
            error_code="77-777-77",
            description="Specific Zabbix description of the error",
            affected_count=10, total_occurrences=10, fleet_total=100,
        )
        assert result.error is None
        # what_is_this заполнен из description
        assert "Specific Zabbix description" in result.what_is_this

    def test_what_is_this_fallback_rag_tray_branch(self) -> None:
        """Если в rag_context есть 'Tray' — берём вторую строку rag_context."""
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": True,
            "what_is_this": "",
            "why_this_pattern": "Массовое.",
            "business_impact": "Impact.",
            "immediate_action": "Act.",
            "long_term_action": "Prof.",
            "indicators_to_watch": ["i"],
        }))
        rag_text = "Заголовок\nTray 2: paper misfeed — detailed explanation of the jam path"
        agent._fetch_rag_context = MagicMock(return_value=rag_text)  # type: ignore[method-assign]
        result = agent.analyze_mass_error(
            error_code="75-000-00", description="",
            affected_count=5, total_occurrences=5, fleet_total=100,
        )
        assert result.error is None
        assert "Tray 2" in result.what_is_this, (
            f"Ожидали подстановку из RAG (Tray-ветка), получили: {result.what_is_this!r}"
        )

    def test_what_is_this_fallback_rag_error_code_branch(self) -> None:
        """Если rag_context содержит error_code (без 'Tray') — всё равно берём RAG."""
        agent = _make_agent_with_llm_mock(json.dumps({
            "is_systemic": True,
            "what_is_this": "",
            "why_this_pattern": "Массовое.",
            "business_impact": "Impact.",
            "immediate_action": "Act.",
            "long_term_action": "Prof.",
            "indicators_to_watch": ["i"],
        }))
        rag_text = "Заголовок секции\nКод 75-530-00: описание неисправности узла фьюзера"
        agent._fetch_rag_context = MagicMock(return_value=rag_text)  # type: ignore[method-assign]
        result = agent.analyze_mass_error(
            error_code="75-530-00", description="",
            affected_count=5, total_occurrences=5, fleet_total=100,
        )
        assert result.error is None
        assert "75-530-00" in result.what_is_this, (
            f"Ожидали подстановку из RAG (error-code-ветка), получили: {result.what_is_this!r}"
        )


# ── Agent.analyze_device_deep ───────────────────────────────────────────────


def _make_health_result(
    device_id: str = "PRN001",
    hi: int = 25,
    codes: tuple[str, ...] = ("09-594-00",),
):
    from data_io.models import (
        ConfidenceZone,
        FactorContribution,
        HealthResult,
        HealthZone,
    )

    return HealthResult(
        device_id=device_id,
        health_index=hi,
        confidence=0.85,
        zone=HealthZone.RED,
        confidence_zone=ConfidenceZone.HIGH,
        factor_contributions=[
            FactorContribution(
                label=f"{c} (Medium)",
                penalty=20.0,
                S=10.0,
                R=2.5,
                C=1.0,
                A=1.0,
                source="error_code",
            )
            for c in codes
        ],
        calculated_at=datetime.now(UTC),
    )


def _make_factor_store_mock(events=None, resources=None, meta=None):
    fs = MagicMock()
    fs.get_events.return_value = events or []
    fs.get_resources.return_value = resources
    fs.get_device_metadata.return_value = meta
    return fs


def _make_event(code: str, desc: str = "", days_ago: int = 1):
    from datetime import timedelta

    ev = MagicMock()
    ev.error_code = code
    ev.error_description = desc
    ev.timestamp = datetime.now(UTC) - timedelta(days=days_ago)
    return ev


class TestAnalyzeDeviceDeep:
    def test_happy_path(self) -> None:
        agent = _make_agent_with_llm_mock(json.dumps({
            "health_index_llm": 22,
            "root_cause": "Картридж R2 исчерпан, toner=3%.",
            "recommended_action": "Заменить R2 сегодня, заказать резерв.",
            "explanation": "09-594-00 повторяется 12 раз за 14 дней (R=2.5). Toner=3%, drum и fuser в норме — проблема локализована в тонере.",
            "related_codes": ["09-594-00", "09-596-00"],
        }))
        # Make RAG fetch return nothing so test is deterministic
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None

        hr = _make_health_result(device_id="PRN042", hi=22, codes=("09-594-00",))
        fs = _make_factor_store_mock(
            events=[_make_event("09-594-00", "Print Cartridge R2 near EOL", 1)],
        )

        result = agent.analyze_device_deep(
            device_id="PRN042",
            health_result=hr,
            factor_store=fs,
        )

        assert isinstance(result, DeepDeviceAnalysis)
        assert result.error is None
        assert result.device_id == "PRN042"
        assert result.health_index_original == 22
        assert result.health_index_llm == 22
        assert "R2" in result.root_cause
        assert "Заменить" in result.recommended_action
        assert result.llm_calls == 1
        assert result.reflection_verdict == "single_shot"
        assert result.duration_ms >= 0
        assert "09-594-00" in result.related_codes

    def test_think_prefix_stripped(self) -> None:
        raw = (
            "<think>long reasoning about codes</think>\n"
            + json.dumps({
                "health_index_llm": 30,
                "root_cause": "Износ DADF",
                "recommended_action": "Чистка ролика DADF",
                "explanation": "05-231-00 устойчиво повторяется.",
                "related_codes": ["05-231-00"],
            })
        )
        agent = _make_agent_with_llm_mock(raw)
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None

        result = agent.analyze_device_deep(
            device_id="PRN118",
            health_result=_make_health_result(hi=32, codes=("05-231-00",)),
            factor_store=_make_factor_store_mock(),
        )
        assert result.error is None
        assert result.health_index_llm == 30

    def test_markdown_codefence(self) -> None:
        raw = "```json\n" + json.dumps({
            "health_index_llm": 18,
            "root_cause": "rc",
            "recommended_action": "act",
            "explanation": "exp",
            "related_codes": [],
        }) + "\n```"
        agent = _make_agent_with_llm_mock(raw)
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None

        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(),
            factor_store=_make_factor_store_mock(),
        )
        assert result.error is None
        assert result.health_index_llm == 18

    def test_invalid_json_returns_error_record(self) -> None:
        agent = _make_agent_with_llm_mock("не JSON совсем")
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None

        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(hi=25),
            factor_store=_make_factor_store_mock(),
        )
        assert result.error is not None
        assert result.llm_calls == 1
        assert result.health_index_llm is None
        assert result.health_index_original == 25

    def test_no_llm_returns_error_record(self) -> None:
        agent = Agent.__new__(Agent)
        agent._llm = None
        agent._tools = MagicMock()
        agent._factor_store = MagicMock()
        agent._config = MagicMock()
        agent._memory = None

        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(),
            factor_store=_make_factor_store_mock(),
        )
        assert result.error == "LLM недоступен"
        assert result.llm_calls == 0

    def test_rag_failure_does_not_break(self) -> None:
        agent = _make_agent_with_llm_mock(json.dumps({
            "health_index_llm": 40,
            "root_cause": "rc",
            "recommended_action": "act",
            "explanation": "exp",
            "related_codes": ["05-210-00"],
        }))
        agent._tools.execute.side_effect = RuntimeError("RAG down")
        agent._tools._deps = None

        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(),
            factor_store=_make_factor_store_mock(
                events=[_make_event("09-594-00", "desc")],
            ),
        )
        assert result.error is None

    def test_empty_events_and_resources(self) -> None:
        agent = _make_agent_with_llm_mock(json.dumps({
            "health_index_llm": 25,
            "root_cause": "Недостаточно телеметрии.",
            "recommended_action": "Собрать телеметрию за 7 дней.",
            "explanation": "Нет событий, нет ресурсов.",
            "related_codes": [],
        }))
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None

        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(hi=35, codes=()),
            factor_store=_make_factor_store_mock(events=[], resources=None, meta=None),
        )
        assert result.error is None
        assert "телеметри" in result.root_cause.lower()

    def test_truncated_json_recovered(self) -> None:
        raw = (
            '{"health_index_llm": 25, '
            '"root_cause": "Картридж R2 на исходе, требуется замена", '
            '"recommended_action": "Заменить картридж", '
            '"related_codes": [], '
            '"explanation": "Toner level 3%, drum 48%, fuser 72%, поэтому'  # cut mid-string
        )
        agent = _make_agent_with_llm_mock(raw)
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None

        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(),
            factor_store=_make_factor_store_mock(),
        )
        assert result.error is None, f"should recover, got: {result.error}"
        assert "Картридж" in result.root_cause

    def test_health_index_llm_bounded(self) -> None:
        # LLM returns out-of-range value — must clamp to 100
        agent = _make_agent_with_llm_mock(json.dumps({
            "health_index_llm": 150,
            "root_cause": "rc",
            "recommended_action": "act",
            "explanation": "exp",
            "related_codes": [],
        }))
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None
        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(hi=30),
            factor_store=_make_factor_store_mock(),
        )
        assert result.health_index_llm == 100

    def test_health_index_llm_non_numeric_falls_back(self) -> None:
        agent = _make_agent_with_llm_mock(json.dumps({
            "health_index_llm": "abc",
            "root_cause": "rc",
            "recommended_action": "act",
            "explanation": "exp",
            "related_codes": [],
        }))
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None
        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(hi=30),
            factor_store=_make_factor_store_mock(),
        )
        # fallback на health_index_original
        assert result.health_index_llm == 30

    def test_fields_and_codes_truncated(self) -> None:
        agent = _make_agent_with_llm_mock(json.dumps({
            "health_index_llm": 10,
            "root_cause": "x" * 2000,
            "recommended_action": "y" * 2000,
            "explanation": "z" * 5000,
            "related_codes": ["c" * 200] * 25,
        }))
        agent._tools.execute.side_effect = RuntimeError("no rag")
        agent._tools._deps = None
        result = agent.analyze_device_deep(
            device_id="X",
            health_result=_make_health_result(),
            factor_store=_make_factor_store_mock(),
        )
        assert result.error is None
        assert len(result.root_cause) <= 400
        assert len(result.recommended_action) <= 400
        assert len(result.explanation) <= 1500
        assert len(result.related_codes) <= 10
        assert all(len(c) <= 50 for c in result.related_codes)
