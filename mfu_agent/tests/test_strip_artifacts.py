"""Tests for LLM artifact stripping and executive summary cleanup (Problems #1, #2, #5)."""

from __future__ import annotations

import pytest

from llm.client import LLMClient


class TestStripReasoningArtifacts:
    def test_strip_think_block(self) -> None:
        raw = "<think>рассуждения модели</think>\n\nФинальный текст."
        assert LLMClient.strip_reasoning_artifacts(raw) == "Финальный текст."

    def test_strip_think_block_case_insensitive(self) -> None:
        raw = "<THINK>мысли</THINK>Результат."
        assert LLMClient.strip_reasoning_artifacts(raw) == "Результат."

    def test_strip_multiline_think(self) -> None:
        raw = "<think>\nДлинные\nмногострочные\nрассуждения\n</think>\n\nТри абзаца текста."
        assert LLMClient.strip_reasoning_artifacts(raw) == "Три абзаца текста."

    def test_strip_unclosed_think(self) -> None:
        raw = "<think>начало мыслей\nпродолжение</think>Ответ после тега."
        result = LLMClient.strip_reasoning_artifacts(raw)
        assert "Ответ после тега." in result
        assert "<think>" not in result

    def test_strip_action_final_answer_json_dict(self) -> None:
        raw = '{"action": "final_answer", "final_answer": {"summary": "Текст.", "risk_areas": "Риски.", "recommendations": "Рек."}}'
        result = LLMClient.strip_reasoning_artifacts(raw)
        assert "Текст." in result
        assert "Риски." in result
        assert "Рек." in result
        assert '"action"' not in result
        assert "{" not in result

    def test_strip_action_final_answer_json_string(self) -> None:
        raw = '{"action": "final_answer", "final_answer": "Готовый текст."}'
        result = LLMClient.strip_reasoning_artifacts(raw)
        assert result == "Готовый текст."

    def test_strip_combined_think_and_json(self) -> None:
        raw = (
            "<think>Хорошо, мне нужно составить executive summary...\n"
            "Проверю структуру: три абзаца с общей оценкой...\n"
            "[ещё ~500 слов рассуждений]\n"
            "</think>\n\n"
            '{ "action": "final_answer", "final_answer": {\n'
            '  "summary": "Парк из 68 устройств.",\n'
            '  "risk_areas": "Большинство в жёлтой зоне.",\n'
            '  "recommendations": "Заменить тонеры."\n'
            "}}"
        )
        result = LLMClient.strip_reasoning_artifacts(raw)
        assert "<think>" not in result
        assert '"action"' not in result
        assert "Парк из 68 устройств." in result
        assert "Заменить тонеры." in result

    def test_strip_code_fences(self) -> None:
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = LLMClient.strip_reasoning_artifacts(raw)
        assert "```" not in result

    def test_strip_markdown_fence(self) -> None:
        raw = "```markdown\nОбычный текст.\n```"
        result = LLMClient.strip_reasoning_artifacts(raw)
        assert "```" not in result
        assert "Обычный текст." in result

    def test_plain_text_unchanged(self) -> None:
        raw = "Парк из 68 устройств. Средний индекс 61.8."
        assert LLMClient.strip_reasoning_artifacts(raw) == raw

    def test_empty_string(self) -> None:
        assert LLMClient.strip_reasoning_artifacts("") == ""

    def test_only_think_block_returns_empty(self) -> None:
        raw = "<think>только мысли, нет ответа</think>"
        assert LLMClient.strip_reasoning_artifacts(raw) == ""


class TestExecutiveSummaryPrompt:
    def test_prompt_template_has_placeholders(self) -> None:
        from pathlib import Path

        prompt_path = Path(__file__).resolve().parent.parent / "agent" / "prompts" / "executive_summary.md"
        content = prompt_path.read_text(encoding="utf-8")

        assert "{fleet_summary_json}" in content
        assert "{worst_devices_json}" in content
        assert "{top_error_codes_json}" in content
        assert "{top_patterns_json}" in content

    def test_prompt_forbids_think_tags(self) -> None:
        from pathlib import Path

        prompt_path = Path(__file__).resolve().parent.parent / "agent" / "prompts" / "executive_summary.md"
        content = prompt_path.read_text(encoding="utf-8")

        assert "<think>" in content.lower() or "think" in content.lower()
        assert "ЗАПРЕЩЕНО" in content or "запрещено" in content

    def test_prompt_contains_zone_thresholds(self) -> None:
        from pathlib import Path

        prompt_path = Path(__file__).resolve().parent.parent / "agent" / "prompts" / "executive_summary.md"
        content = prompt_path.read_text(encoding="utf-8")

        assert ">= 75" in content or "≥ 75" in content or "75" in content
        assert "< 40" in content or "40" in content


class TestReportBuilderSummaryNoRunChat:
    """Verify _generate_executive_summary uses LLM.generate directly, not run_chat."""

    def test_no_run_chat_in_summary_method(self) -> None:
        import inspect
        from reporting.report_builder import ReportBuilder

        source = inspect.getsource(ReportBuilder._generate_executive_summary)
        assert "run_chat" not in source, "Summary should use LLM.generate directly, not agent.run_chat"
        assert "tools=None" in source or "tools = None" in source, "Summary LLM call must have tools=None"
