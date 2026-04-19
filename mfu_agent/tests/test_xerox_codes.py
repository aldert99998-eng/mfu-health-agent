"""Tests for Xerox XX-YYY-ZZ error code support (Problems #3, #4)."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from data_io.normalizer import normalize_error_code

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# ── normalize_error_code ─────────────────────────────────────────────────────


class TestNormalizeXeroxCodes:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("75-530-00", "75-530-00"),
            ("72-535-00", "72-535-00"),
            ("07-535-00", "07-535-00"),
            ("09-594-00", "09-594-00"),
        ],
    )
    def test_xerox_codes_preserved(self, raw: str, expected: str) -> None:
        assert normalize_error_code(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Error 75-530-00 occurred", "75-530-00"),
            ("errdisp-[07-535-00 Tray empty]", "07-535-00"),
            ("код ошибки: 09-604-00", "09-604-00"),
        ],
    )
    def test_xerox_codes_extracted_from_text(self, raw: str, expected: str) -> None:
        assert normalize_error_code(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("C6000", "C6000"),
            ("C-6000", "C6000"),
            ("error SC543", "SC543"),
            ("E102", "E102"),
        ],
    )
    def test_classic_codes_still_work(self, raw: str, expected: str) -> None:
        assert normalize_error_code(raw) == expected

    def test_unknown_code_returns_none(self) -> None:
        assert normalize_error_code("random text no codes") is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_error_code("") is None
        assert normalize_error_code("   ") is None


# ── error_code_patterns.yaml ─────────────────────────────────────────────────


class TestErrorCodePatternsYaml:
    @pytest.fixture()
    def patterns(self) -> dict:
        path = CONFIGS_DIR / "error_code_patterns.yaml"
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_xerox_has_xx_yyy_zz_pattern(self, patterns: dict) -> None:
        xerox = patterns.get("xerox", [])
        has_pattern = any(
            re.compile(p).match("75-530-00") for p in xerox
        )
        assert has_pattern, f"No Xerox pattern matches 75-530-00: {xerox}"

    def test_generic_has_xx_yyy_zz_pattern(self, patterns: dict) -> None:
        generic = patterns.get("generic", [])
        has_pattern = any(
            re.compile(p).match("75-530-00") for p in generic
        )
        assert has_pattern, f"No generic pattern matches 75-530-00: {generic}"

    def test_all_patterns_compile(self, patterns: dict) -> None:
        for vendor, pats in patterns.items():
            for p in pats:
                try:
                    re.compile(p)
                except re.error as exc:
                    pytest.fail(f"{vendor}: invalid regex {p!r}: {exc}")


# ── Heuristic severity fallback ──────────────────────────────────────────────


class TestHeuristicClassify:
    def test_known_critical_code(self) -> None:
        from agent.tools.impl import _heuristic_classify

        result = _heuristic_classify("09-594-00")
        assert result is not None
        assert result["severity"] == "Critical"
        assert "fuser" in result["affected_components"]

    def test_known_high_code(self) -> None:
        from agent.tools.impl import _heuristic_classify

        result = _heuristic_classify("75-530-00")
        assert result is not None
        assert result["severity"] == "High"
        assert "toner" in result["affected_components"]

    def test_unknown_xerox_prefix_fallback(self) -> None:
        from agent.tools.impl import _heuristic_classify

        result = _heuristic_classify("09-999-99")
        assert result is not None
        assert result["severity"] == "Critical"
        assert result["confidence"] == 0.4

    def test_non_xerox_code_returns_none(self) -> None:
        from agent.tools.impl import _heuristic_classify

        assert _heuristic_classify("C6000") is None
        assert _heuristic_classify("SC543") is None

    def test_completely_unknown_xerox_prefix(self) -> None:
        from agent.tools.impl import _heuristic_classify

        result = _heuristic_classify("99-999-99")
        assert result is None


# ── ClassifyErrorSeverityTool with empty RAG ─────────────────────────────────


class TestClassifyToolHeuristicFallback:
    def test_xerox_code_with_empty_rag_uses_heuristic(self) -> None:
        from agent.tools.impl import ClassifyErrorSeverityTool, ToolDependencies
        from agent.tools.registry import ToolResult

        deps = ToolDependencies(
            factor_store=MagicMock(),
            weights=MagicMock(),
            searcher=None,
            llm_client=None,
            collection="service_manuals",
            health_cache={},
            memory_manager=MagicMock(),
        )
        tool = ClassifyErrorSeverityTool(deps)
        result = tool.execute({"error_code": "75-530-00", "model": "Xerox AltaLink B8045"})

        assert isinstance(result, ToolResult)
        assert result.success
        assert result.data["severity"] == "High"
        assert "toner" in result.data["affected_components"]

    def test_non_xerox_code_with_empty_rag_uses_generic_fallback(self) -> None:
        from agent.tools.impl import ClassifyErrorSeverityTool, ToolDependencies
        from agent.tools.registry import ToolResult

        deps = ToolDependencies(
            factor_store=MagicMock(),
            weights=MagicMock(),
            searcher=None,
            llm_client=None,
            collection="service_manuals",
            health_cache={},
            memory_manager=MagicMock(),
        )
        tool = ClassifyErrorSeverityTool(deps)
        result = tool.execute({"error_code": "C6000"})

        assert result.success
        assert result.data["severity"] == "Medium"
        assert result.data["confidence"] == 0.3
