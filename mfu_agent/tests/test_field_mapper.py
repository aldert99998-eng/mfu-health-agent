"""Tests for data_io/field_mapper.py — Phase 4.2 verification.

TC-FM-1  Синонимы: русские имена колонок → device_id, timestamp, error_code, model
TC-FM-2  Синонимы: английские варианты (serial_number, event_time, code)
TC-FM-3  Синонимы: варианты с дефисами, пробелами, скобками
TC-FM-4  Синонимы: не назначает один target на два столбца
TC-FM-5  ContentMatcher: колонка с кодами ошибок [CJEFS]\\d{3,5}
TC-FM-6  ContentMatcher: колонка с датами → timestamp
TC-FM-7  ContentMatcher: 0-100 + "toner" в имени → toner_level
TC-FM-8  ContentMatcher: большие числа + "pages" → mileage
TC-FM-9  ContentMatcher: уникальные строки (>20 строк) → device_id
TC-FM-10 ContentMatcher: пустая колонка → None
TC-FM-11 LLMMatcher: stub-клиент возвращает корректный JSON → маппинг
TC-FM-12 LLMMatcher: stub возвращает JSON в markdown-блоке → парсинг
TC-FM-13 LLMMatcher: stub возвращает мусор → fallback, пустой результат
TC-FM-14 LLMMatcher: без клиента → пустой результат
TC-FM-15 LLMMatcher: _ignore в ответе → не включается в маппинг
TC-FM-16 LLMMatcher: дубликат уже замапленного поля → отфильтровывается
TC-FM-17 Профиль: save → load → try_apply по signature
TC-FM-18 Профиль: другой набор колонок → профиль не применяется
TC-FM-19 Профиль: порядок колонок не влияет на signature
TC-FM-20 FieldMapper: полный pipeline synonym+content+llm
TC-FM-21 FieldMapper: повторный файл через профиль → без LLM
TC-FM-22 normalize_column_name: покрытие вариантов
TC-FM-23 compute_signature: детерминированный SHA-256
TC-FM-24 MappingResult: unmapped содержит нераспознанные колонки
TC-FM-25 FieldMapper: confidence levels корректны для каждого шага
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data_io.field_mapper import (
    ConfidenceLevel,
    ContentMatcher,
    FieldMapper,
    LLMClient,
    LLMMatcher,
    SynonymMatcher,
    compute_signature,
    normalize_column_name,
    save_profile,
    try_apply_profile,
)


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def synonyms_yaml(tmp: Path) -> Path:
    p = tmp / "field_synonyms.yaml"
    p.write_text(
        """\
device_id:
  - device_id
  - serial_number
  - id_устройства
  - идентификатор
timestamp:
  - timestamp
  - event_time
  - дата_события
error_code:
  - error_code
  - код_ошибки
  - code
model:
  - model
  - модель
vendor:
  - vendor
  - производитель
location:
  - location
  - офис
status:
  - status
  - статус
toner_level:
  - toner_level
  - тонер
drum_level:
  - drum_level
  - барабан
fuser_level:
  - fuser_level
  - фьюзер
mileage:
  - mileage
  - пробег
error_description:
  - error_description
  - описание
""",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def prompt_file(tmp: Path) -> Path:
    p = tmp / "field_mapping.md"
    p.write_text(
        "Маппинг полей.\n\n{columns_and_samples}\n\nВерни JSON.",
        encoding="utf-8",
    )
    return p


# ── Helpers ─────────────────────────────────────────────────────────────────


class StubLLMClient:
    """Fake LLM that returns a canned JSON response."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str) -> str:
        return self._response


class BrokenLLMClient:
    """Fake LLM that always returns garbage."""

    def complete(self, prompt: str) -> str:
        return "this is not json at all"


# ── TC-FM-22: normalize_column_name ─────────────────────────────────────────


class TestNormalizeColumnName:
    def test_lowercase_and_underscore(self) -> None:
        assert normalize_column_name("Device ID") == "device_id"

    def test_hyphen_to_underscore(self) -> None:
        assert normalize_column_name("device-id") == "device_id"

    def test_dot_to_underscore(self) -> None:
        assert normalize_column_name("device.id") == "device_id"

    def test_strip_parentheses(self) -> None:
        result = normalize_column_name("Уровень (%)")
        assert "(" not in result
        assert "%" not in result

    def test_diacritics_stripped(self) -> None:
        n1 = normalize_column_name("ёлка")
        n2 = normalize_column_name("елка")
        assert n1 == n2

    def test_й_stripped(self) -> None:
        n = normalize_column_name("устройства")
        assert "и" in n  # й → и after NFD + strip Mn

    def test_leading_trailing_whitespace(self) -> None:
        assert normalize_column_name("  id  ") == "id"


# ── TC-FM-23: compute_signature ─────────────────────────────────────────────


class TestComputeSignature:
    def test_deterministic(self) -> None:
        cols = ["device_id", "timestamp", "error_code"]
        assert compute_signature(cols) == compute_signature(cols)

    def test_order_independent(self) -> None:
        a = compute_signature(["device_id", "timestamp", "error_code"])
        b = compute_signature(["error_code", "device_id", "timestamp"])
        assert a == b

    def test_length_64_hex(self) -> None:
        sig = compute_signature(["col_a", "col_b"])
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


# ── TC-FM-1..4: SynonymMatcher ──────────────────────────────────────────────


class TestSynonymMatcher:
    def test_russian_columns(self, synonyms_yaml: Path) -> None:
        sm = SynonymMatcher(synonyms_yaml)
        result = sm.match(["id_устройства", "дата_события", "код_ошибки", "модель"])
        assert result == {
            "id_устройства": "device_id",
            "дата_события": "timestamp",
            "код_ошибки": "error_code",
            "модель": "model",
        }

    def test_english_variants(self, synonyms_yaml: Path) -> None:
        sm = SynonymMatcher(synonyms_yaml)
        result = sm.match(["serial_number", "event_time", "code", "vendor"])
        assert result["serial_number"] == "device_id"
        assert result["event_time"] == "timestamp"
        assert result["code"] == "error_code"
        assert result["vendor"] == "vendor"

    def test_hyphen_and_spaces(self, synonyms_yaml: Path) -> None:
        sm = SynonymMatcher(synonyms_yaml)
        result = sm.match(["Device ID", "device-id"])
        assert len(result) == 1
        target = next(iter(result.values()))
        assert target == "device_id"

    def test_no_duplicate_target(self, synonyms_yaml: Path) -> None:
        sm = SynonymMatcher(synonyms_yaml)
        result = sm.match(["device_id", "serial_number", "идентификатор"])
        targets = list(result.values())
        assert targets.count("device_id") == 1

    def test_missing_synonyms_file(self, tmp: Path) -> None:
        sm = SynonymMatcher(tmp / "nonexistent.yaml")
        result = sm.match(["device_id"])
        assert result == {}


# ── TC-FM-5..10: ContentMatcher ─────────────────────────────────────────────


class TestContentMatcher:
    def test_error_code_column(self) -> None:
        df = pd.DataFrame({"codes": ["C6000", "E200", "J1234", "F501", "SC345"]})
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert result.get("codes") == "error_code"

    def test_timestamp_column(self) -> None:
        df = pd.DataFrame({
            "when": [
                "2026-04-18 10:00",
                "2026-04-17 11:30",
                "2026-04-16 09:00",
                "2026-04-15 14:20",
                "2026-04-14 08:00",
            ]
        })
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert result.get("when") == "timestamp"

    def test_toner_level_by_content_and_name(self) -> None:
        df = pd.DataFrame({"toner_remaining": ["45", "78", "92", "10", "55"]})
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert result.get("toner_remaining") == "toner_level"

    def test_drum_level_cyrillic_hint(self) -> None:
        df = pd.DataFrame({"ресурс_барабана_%": ["45", "78", "92", "10", "55"]})
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert result.get("ресурс_барабана_%") == "drum_level"

    def test_mileage_column(self) -> None:
        df = pd.DataFrame({
            "total_pages": ["15000", "22000", "8500", "31000", "12000"]
        })
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert result.get("total_pages") == "mileage"

    def test_device_id_by_uniqueness(self) -> None:
        ids = [f"MFP-{i:03d}" for i in range(25)]
        df = pd.DataFrame({"asset": ids})
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert result.get("asset") == "device_id"

    def test_device_id_too_few_rows(self) -> None:
        df = pd.DataFrame({"asset": ["MFP-001", "MFP-002", "MFP-003"]})
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert "asset" not in result

    def test_empty_column_skipped(self) -> None:
        df = pd.DataFrame({"empty_col": [None, None, None]})
        cm = ContentMatcher()
        result = cm.match(df, {})
        assert result == {}

    def test_skips_already_mapped_targets(self) -> None:
        df = pd.DataFrame({
            "codes": ["C6000", "E200", "J1234"],
            "more_codes": ["C7000", "E300", "F100"],
        })
        cm = ContentMatcher()
        result = cm.match(df, {"codes": "error_code"})
        assert "more_codes" not in result


# ── TC-FM-11..16: LLMMatcher ───────────────────────────────────────────────


class TestLLMMatcher:
    def test_stub_returns_valid_json(self, prompt_file: Path) -> None:
        response = json.dumps({"mapping": {"Префикс_каб": "location", "Кол_замятий": "_ignore"}})
        client = StubLLMClient(response)
        matcher = LLMMatcher(client, prompt_file)

        df = pd.DataFrame({
            "Префикс_каб": ["Ленинг_5", "Тверск_12"],
            "Кол_замятий": ["0", "2"],
        })
        result = matcher.match(df, {})
        assert result == {"Префикс_каб": "location"}

    def test_markdown_code_block_stripped(self, prompt_file: Path) -> None:
        raw = '```json\n{"mapping": {"col_x": "vendor"}}\n```'
        client = StubLLMClient(raw)
        matcher = LLMMatcher(client, prompt_file)

        df = pd.DataFrame({"col_x": ["Kyocera", "HP"]})
        result = matcher.match(df, {})
        assert result == {"col_x": "vendor"}

    def test_broken_json_fallback(self, prompt_file: Path) -> None:
        client = BrokenLLMClient()
        matcher = LLMMatcher(client, prompt_file)

        df = pd.DataFrame({"col_x": ["a", "b"]})
        result = matcher.match(df, {})
        assert result == {}

    def test_no_client_returns_empty(self, prompt_file: Path) -> None:
        matcher = LLMMatcher(None, prompt_file)
        df = pd.DataFrame({"col_x": ["a", "b"]})
        result = matcher.match(df, {})
        assert result == {}

    def test_ignore_excluded(self, prompt_file: Path) -> None:
        response = json.dumps({"mapping": {"col_a": "_ignore", "col_b": "model"}})
        client = StubLLMClient(response)
        matcher = LLMMatcher(client, prompt_file)

        df = pd.DataFrame({"col_a": ["x"], "col_b": ["HP"]})
        result = matcher.match(df, {})
        assert "col_a" not in result
        assert result["col_b"] == "model"

    def test_duplicate_target_filtered(self, prompt_file: Path) -> None:
        response = json.dumps({"mapping": {"col_a": "device_id"}})
        client = StubLLMClient(response)
        matcher = LLMMatcher(client, prompt_file)

        df = pd.DataFrame({"col_a": ["D001"]})
        result = matcher.match(df, {"existing": "device_id"})
        assert result == {}

    def test_invalid_target_filtered(self, prompt_file: Path) -> None:
        response = json.dumps({"mapping": {"col_a": "nonexistent_field"}})
        client = StubLLMClient(response)
        matcher = LLMMatcher(client, prompt_file)

        df = pd.DataFrame({"col_a": ["x"]})
        result = matcher.match(df, {})
        assert result == {}

    def test_all_mapped_skips_llm(self, prompt_file: Path) -> None:
        call_count = 0

        class CountingClient:
            def complete(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                return '{"mapping": {}}'

        matcher = LLMMatcher(CountingClient(), prompt_file)
        df = pd.DataFrame({"col_a": ["x"]})
        matcher.match(df, {"col_a": "device_id"})
        assert call_count == 0


# ── TC-FM-17..19: Profile management ────────────────────────────────────────


class TestProfileManagement:
    def test_save_and_load(self, tmp: Path) -> None:
        profiles_dir = tmp / "profiles"
        columns = ["device_id", "timestamp", "error_code"]
        mapping = {"device_id": "device_id", "timestamp": "timestamp", "error_code": "error_code"}

        path = save_profile("test_prof", columns, mapping, profiles_dir)
        assert path.exists()

        loaded = try_apply_profile(columns, profiles_dir)
        assert loaded is not None
        loaded_mapping, loaded_name = loaded
        assert loaded_name == "test_prof"
        assert loaded_mapping == mapping

    def test_different_columns_no_match(self, tmp: Path) -> None:
        profiles_dir = tmp / "profiles"
        columns_a = ["device_id", "timestamp"]
        mapping_a = {"device_id": "device_id", "timestamp": "timestamp"}
        save_profile("prof_a", columns_a, mapping_a, profiles_dir)

        columns_b = ["serial", "date", "code"]
        loaded = try_apply_profile(columns_b, profiles_dir)
        assert loaded is None

    def test_column_order_independent(self, tmp: Path) -> None:
        profiles_dir = tmp / "profiles"
        columns = ["timestamp", "device_id", "error_code"]
        mapping = {"device_id": "device_id", "timestamp": "timestamp"}
        save_profile("prof_order", columns, mapping, profiles_dir)

        reordered = ["error_code", "device_id", "timestamp"]
        loaded = try_apply_profile(reordered, profiles_dir)
        assert loaded is not None

    def test_no_profiles_dir(self, tmp: Path) -> None:
        loaded = try_apply_profile(["a", "b"], tmp / "nonexistent")
        assert loaded is None

    def test_corrupt_profile_skipped(self, tmp: Path) -> None:
        profiles_dir = tmp / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "bad.yaml").write_text("not: [valid: yaml: {{", encoding="utf-8")
        loaded = try_apply_profile(["a", "b"], profiles_dir)
        assert loaded is None


# ── TC-FM-20..21, 24..25: FieldMapper integration ──────────────────────────


class TestFieldMapper:
    def test_full_pipeline_synonym_content_llm(
        self, synonyms_yaml: Path, prompt_file: Path, tmp: Path,
    ) -> None:
        llm_response = json.dumps({"mapping": {"Префикс_каб": "location"}})
        client = StubLLMClient(llm_response)
        profiles_dir = tmp / "profiles"

        mapper = FieldMapper(
            llm_client=client,
            synonyms_path=synonyms_yaml,
            prompt_path=prompt_file,
            profiles_dir=profiles_dir,
        )

        df = pd.DataFrame({
            "id_устройства": ["D001", "D002"],
            "дата_события": ["2026-01-01", "2026-01-02"],
            "код_ошибки": ["C6000", "E200"],
            "Префикс_каб": ["Ленинг_5", "Тверск_12"],
            "unknown_stuff": ["x", "y"],
        })

        result = mapper.map(df)

        assert result.auto_mapping["id_устройства"] == "device_id"
        assert result.auto_mapping["дата_события"] == "timestamp"
        assert result.auto_mapping["код_ошибки"] == "error_code"
        assert result.auto_mapping["Префикс_каб"] == "location"
        assert "unknown_stuff" in result.unmapped
        assert result.profile_applied is None

    def test_confidence_levels_correct(
        self, synonyms_yaml: Path, prompt_file: Path, tmp: Path,
    ) -> None:
        llm_response = json.dumps({"mapping": {"mystery_col": "vendor"}})
        client = StubLLMClient(llm_response)
        profiles_dir = tmp / "profiles"

        mapper = FieldMapper(
            llm_client=client,
            synonyms_path=synonyms_yaml,
            prompt_path=prompt_file,
            profiles_dir=profiles_dir,
        )

        df = pd.DataFrame({
            "id_устройства": ["D001", "D002", "D003", "D004", "D005"],
            "codes": ["C6000", "E200", "J1234", "F501", "SC345"],
            "mystery_col": ["Kyocera", "HP", "Canon", "Ricoh", "Xerox"],
            "garbage": ["x", "y", "z", "w", "v"],
        })

        result = mapper.map(df)
        detail_map = {d.source_column: d for d in result.column_details}

        assert detail_map["id_устройства"].confidence == ConfidenceLevel.SYNONYM
        assert detail_map["codes"].confidence == ConfidenceLevel.CONTENT
        assert detail_map["mystery_col"].confidence == ConfidenceLevel.LLM
        assert detail_map["garbage"].confidence == ConfidenceLevel.UNKNOWN

    def test_profile_applied_on_repeat(
        self, synonyms_yaml: Path, prompt_file: Path, tmp: Path,
    ) -> None:
        profiles_dir = tmp / "profiles"
        columns = ["device_id", "timestamp", "error_code"]
        mapping = {
            "device_id": "device_id",
            "timestamp": "timestamp",
            "error_code": "error_code",
        }
        save_profile("repeat_test", columns, mapping, profiles_dir)

        call_count = 0

        class TrackingClient:
            def complete(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                return '{"mapping": {}}'

        mapper = FieldMapper(
            llm_client=TrackingClient(),
            synonyms_path=synonyms_yaml,
            prompt_path=prompt_file,
            profiles_dir=profiles_dir,
        )

        df = pd.DataFrame({
            "device_id": ["D001"],
            "timestamp": ["2026-01-01"],
            "error_code": ["C6000"],
        })

        result = mapper.map(df)

        assert result.profile_applied == "repeat_test"
        assert result.auto_mapping == mapping
        assert call_count == 0
        for detail in result.column_details:
            assert detail.confidence == ConfidenceLevel.PROFILE

    def test_unmapped_columns_listed(
        self, synonyms_yaml: Path, prompt_file: Path, tmp: Path,
    ) -> None:
        mapper = FieldMapper(
            llm_client=None,
            synonyms_path=synonyms_yaml,
            prompt_path=prompt_file,
            profiles_dir=tmp / "profiles",
        )

        df = pd.DataFrame({
            "id_устройства": ["D001"],
            "weird_col_1": ["abc"],
            "weird_col_2": ["xyz"],
        })

        result = mapper.map(df)
        assert "weird_col_1" in result.unmapped
        assert "weird_col_2" in result.unmapped
        assert "id_устройства" not in result.unmapped

    def test_no_llm_still_works(
        self, synonyms_yaml: Path, prompt_file: Path, tmp: Path,
    ) -> None:
        mapper = FieldMapper(
            llm_client=None,
            synonyms_path=synonyms_yaml,
            prompt_path=prompt_file,
            profiles_dir=tmp / "profiles",
        )

        df = pd.DataFrame({
            "модель": ["HP 400"],
            "статус": ["active"],
            "производитель": ["HP"],
        })

        result = mapper.map(df)
        assert result.auto_mapping["модель"] == "model"
        assert result.auto_mapping["статус"] == "status"
        assert result.auto_mapping["производитель"] == "vendor"


# ── Protocol check ──────────────────────────────────────────────────────────


class TestLLMClientProtocol:
    def test_stub_satisfies_protocol(self) -> None:
        client = StubLLMClient('{"mapping": {}}')
        assert isinstance(client, LLMClient)

    def test_broken_satisfies_protocol(self) -> None:
        client = BrokenLLMClient()
        assert isinstance(client, LLMClient)
