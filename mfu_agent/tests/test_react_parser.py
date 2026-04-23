"""Tests for LLMClient._parse_react_tool_calls.

Local llama-server models without native function-calling respond in the
ReAct text format ("Действие: <tool>\n Параметры: {...}"). The parser
has to turn that back into structured tool calls; this file pins down
the tolerant shape of that parsing.
"""

from __future__ import annotations

from llm.client import LLMClient


class TestReactParser:
    def test_standard_multiline(self) -> None:
        text = (
            "Мысль: Нужно найти код ошибки.\n"
            'Действие: search_service_docs\n'
            'Параметры: {"query": "81-200-00", "top_k": 1}\n'
            "Мысль: После этого дам ответ."
        )
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert calls[0].name == "search_service_docs"
        assert calls[0].arguments == {"query": "81-200-00", "top_k": 1}

    def test_single_line_no_newline_between_fields(self) -> None:
        """Real-world leak: Qwen/Nemotron sometimes emit both on one line."""
        text = (
            'Давайте попробуем: Действие: search_service_docs '
            'Параметры: {"query": "81-200-00", "limit": 1}'
        )
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert calls[0].name == "search_service_docs"
        assert calls[0].arguments == {"query": "81-200-00", "limit": 1}

    def test_english_action_keyword(self) -> None:
        text = (
            'Action: get_device_events\n'
            'Parameters: {"device_id": "DEV001", "window_days": 30}'
        )
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert calls[0].name == "get_device_events"
        assert calls[0].arguments == {"device_id": "DEV001", "window_days": 30}

    def test_nested_json_parsed(self) -> None:
        text = (
            'Действие: search_service_docs '
            'Параметры: {"query": "fuser", "filters": {"vendor": "Xerox", "type": "procedure"}}'
        )
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert calls[0].arguments["filters"]["vendor"] == "Xerox"

    def test_trailing_prose_after_json(self) -> None:
        """Model keeps thinking after the JSON — we take only the JSON object."""
        text = (
            'Действие: list_red_zone_devices '
            'Параметры: {"limit": 5}\nМысль: После получения результатов ...'
        )
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert calls[0].arguments == {"limit": 5}

    def test_quoted_tool_name(self) -> None:
        text = (
            'Действие: `list_mass_errors`\n'
            'Параметры: {"limit": 10}'
        )
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert calls[0].name == "list_mass_errors"

    def test_no_action_returns_none(self) -> None:
        text = "Простой ответ без вызова инструментов."
        assert LLMClient._parse_react_tool_calls(text) is None

    def test_malformed_json_falls_back_to_raw(self) -> None:
        text = 'Действие: search_service_docs Параметры: {это не JSON}'
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert "_raw" in calls[0].arguments

    def test_action_without_parameters_returns_raw_tail(self) -> None:
        """Action keyword without a JSON object at all."""
        text = 'Действие: search_service_docs Параметры: query=something'
        calls = LLMClient._parse_react_tool_calls(text)
        assert calls is not None
        assert "_raw" in calls[0].arguments
