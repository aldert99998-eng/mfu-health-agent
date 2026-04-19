"""Tests for data_io/normalizer.py — Phase 4.3 verification.

TC-N-1..5   parse_timestamp (ISO, dd.mm.yyyy, Unix, datetime passthrough, broken)
TC-N-6..10  normalize_error_code (plain, prefix strip, spaces/hyphens, parenthesized, invalid)
TC-N-11..13 canonicalize_model (alias hit, no alias, empty)
TC-N-14..16 detect_resource_unit (percent, fraction, raw)
TC-N-17..19 load_model_aliases (valid, missing file, corrupt)
TC-N-20..25 Normalizer.normalize (events only, resources only, mixed split,
            latest snapshot per device, invalid rows, model canonicalization)
TC-N-26..28 Normalizer unit detection integration (fraction 0.45 → 45, percent 45, raw 5000)
TC-N-29..33 ingest_file E2E (happy path, broken file, missing device_id,
            broken timestamps in invalid_records, factor_store filled correctly)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from data_io.factor_store import FactorStore
from data_io.normalizer import (
    Normalizer,
    canonicalize_model,
    detect_resource_unit,
    ingest_file,
    load_model_aliases,
    normalize_error_code,
    parse_timestamp,
)


@pytest.fixture()
def tmp(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def aliases_path(tmp: Path) -> Path:
    p = tmp / "aliases.yaml"
    p.write_text(
        '"Kyocera TASKalfa 3253ci":\n'
        "  - taskalfa 3253ci\n"
        "  - kyocera 3253ci\n"
        '"HP LaserJet Pro M404n":\n'
        "  - hp m404n\n"
        "  - m404n\n",
        encoding="utf-8",
    )
    return p


# ── TC-N-1..5: parse_timestamp ─────────────────────────────────────────────


class TestParseTimestamp:
    def test_iso_format(self) -> None:
        dt = parse_timestamp("2024-03-15T10:30:00")
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15
        assert dt.tzinfo is not None

    def test_dd_mm_yyyy(self) -> None:
        dt = parse_timestamp("15.03.2024 10:30:00")
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15

    def test_dd_mm_yyyy_short(self) -> None:
        dt = parse_timestamp("15.03.2024")
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15

    def test_unix_timestamp(self) -> None:
        dt = parse_timestamp("1710500000")
        assert dt.year == 2024
        assert dt.tzinfo is not None

    def test_unix_float(self) -> None:
        dt = parse_timestamp("1710500000.123")
        assert dt.tzinfo is not None

    def test_datetime_passthrough(self) -> None:
        orig = datetime(2024, 1, 1, tzinfo=UTC)
        result = parse_timestamp(orig)
        assert result == orig

    def test_naive_datetime_gets_utc(self) -> None:
        orig = datetime(2024, 1, 1)
        result = parse_timestamp(orig)
        assert result.tzinfo == UTC

    def test_broken_raises(self) -> None:
        with pytest.raises((ValueError, OverflowError)):
            parse_timestamp("not-a-date-at-all-xyz")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="пустое"):
            parse_timestamp("")


# ── TC-N-6..10: normalize_error_code ───────────────────────────────────────


class TestNormalizeErrorCode:
    def test_plain_code(self) -> None:
        assert normalize_error_code("C6000") == "C6000"

    def test_lowercase_uppercased(self) -> None:
        assert normalize_error_code("c6000") == "C6000"

    def test_prefix_error_stripped(self) -> None:
        assert normalize_error_code("Error C6000") == "C6000"

    def test_prefix_kod_stripped(self) -> None:
        assert normalize_error_code("Код: C6000") == "C6000"

    def test_prefix_oshibka_stripped(self) -> None:
        assert normalize_error_code("Ошибка E200") == "E200"

    def test_spaces_hyphens_removed(self) -> None:
        assert normalize_error_code("SC-001") == "SC001"

    def test_parenthesized_removed(self) -> None:
        assert normalize_error_code("C6000 (critical)") == "C6000"

    def test_invalid_returns_none(self) -> None:
        assert normalize_error_code("not_a_code") is None

    def test_empty_returns_none(self) -> None:
        assert normalize_error_code("") is None

    def test_three_letter_five_digit(self) -> None:
        assert normalize_error_code("ABC12345") == "ABC12345"

    def test_too_many_letters_returns_none(self) -> None:
        assert normalize_error_code("ABCD1234") is None


# ── TC-N-11..13: canonicalize_model ────────────────────────────────────────


class TestCanonicalizeModel:
    def test_alias_hit(self, aliases_path: Path) -> None:
        aliases = load_model_aliases(aliases_path)
        assert canonicalize_model("taskalfa 3253ci", aliases) == "Kyocera TASKalfa 3253ci"

    def test_canonical_itself(self, aliases_path: Path) -> None:
        aliases = load_model_aliases(aliases_path)
        assert canonicalize_model("Kyocera TASKalfa 3253ci", aliases) == "Kyocera TASKalfa 3253ci"

    def test_no_alias_returns_original(self, aliases_path: Path) -> None:
        aliases = load_model_aliases(aliases_path)
        assert canonicalize_model("Unknown Printer X", aliases) == "Unknown Printer X"

    def test_empty_returns_empty(self, aliases_path: Path) -> None:
        aliases = load_model_aliases(aliases_path)
        assert canonicalize_model("", aliases) == ""

    def test_case_insensitive(self, aliases_path: Path) -> None:
        aliases = load_model_aliases(aliases_path)
        assert canonicalize_model("HP M404N", aliases) == "HP LaserJet Pro M404n"


# ── TC-N-14..16: detect_resource_unit ──────────────────────────────────────


class TestDetectResourceUnit:
    def test_percent_range(self) -> None:
        vals = [45.0, 78.0, 12.0, 95.0]
        converted, unit_raw = detect_resource_unit(vals)
        assert converted == vals
        assert unit_raw is False

    def test_fraction_in_0_100_range(self) -> None:
        """Values in [0,1] are a subset of [0,100] — detect_resource_unit
        returns them as-is. The Normalizer._detect_units does the secondary
        check and applies the x100 multiplier."""
        vals = [0.45, 0.78, 0.12, 0.95]
        converted, unit_raw = detect_resource_unit(vals)
        assert converted == vals
        assert unit_raw is False

    def test_raw_out_of_range(self) -> None:
        vals = [5000.0, 12000.0, 3000.0]
        converted, unit_raw = detect_resource_unit(vals)
        assert converted == vals
        assert unit_raw is True

    def test_empty_list(self) -> None:
        converted, unit_raw = detect_resource_unit([])
        assert converted == []
        assert unit_raw is False

    def test_zero_to_one_boundary(self) -> None:
        vals = [0.0, 0.5, 1.0]
        _converted, unit_raw = detect_resource_unit(vals)
        assert unit_raw is False


# ── TC-N-17..19: load_model_aliases ────────────────────────────────────────


class TestLoadModelAliases:
    def test_valid_file(self, aliases_path: Path) -> None:
        aliases = load_model_aliases(aliases_path)
        assert "hp m404n" in aliases
        assert aliases["hp m404n"] == "HP LaserJet Pro M404n"

    def test_missing_file(self, tmp: Path) -> None:
        aliases = load_model_aliases(tmp / "nonexistent.yaml")
        assert aliases == {}

    def test_corrupt_file(self, tmp: Path) -> None:
        p = tmp / "bad.yaml"
        p.write_text("just a string, not a dict")
        aliases = load_model_aliases(p)
        assert aliases == {}


# ── TC-N-20..25: Normalizer.normalize ──────────────────────────────────────


class TestNormalizerNormalize:
    def test_events_only(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001", "D002"],
            "ts": ["2024-03-15 10:00:00", "2024-03-15 11:00:00"],
            "code": ["C6000", "E200"],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "code": "error_code"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert len(result.valid_events) == 2
        assert len(result.valid_resources) == 0
        assert result.valid_events[0].device_id == "D001"
        assert result.valid_events[0].error_code == "C6000"

    def test_resources_only(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001", "D001"],
            "ts": ["2024-03-15 10:00:00", "2024-03-15 11:00:00"],
            "toner": [45, 42],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "toner": "toner_level"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert len(result.valid_events) == 0
        assert "D001" in result.valid_resources
        snap = result.valid_resources["D001"]
        assert snap.toner_level == 42
        assert snap.timestamp.hour == 11

    def test_mixed_split_event_and_resource(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "code": ["C6000"],
            "toner": [80],
        })
        mapping = {
            "id": "device_id", "ts": "timestamp",
            "code": "error_code", "toner": "toner_level",
        }
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert len(result.valid_events) == 1
        assert "D001" in result.valid_resources
        assert result.valid_events[0].error_code == "C6000"
        assert result.valid_resources["D001"].toner_level == 80

    def test_latest_snapshot_per_device(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001", "D001", "D001"],
            "ts": [
                "2024-03-15 08:00:00",
                "2024-03-15 12:00:00",
                "2024-03-15 10:00:00",
            ],
            "toner": [90, 50, 70],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "toner": "toner_level"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert result.valid_resources["D001"].toner_level == 50
        assert result.valid_resources["D001"].timestamp.hour == 12

    def test_invalid_rows_collected(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001", "", "D003"],
            "ts": ["2024-03-15 10:00:00", "2024-03-15 11:00:00", "2024-03-15 12:00:00"],
            "code": ["C6000", "E200", "C6000"],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "code": "error_code"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert len(result.valid_events) == 2
        assert len(result.invalid_records) == 1
        assert result.invalid_records[0].field == "device_id"

    def test_model_canonicalization(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "code": ["C6000"],
            "model": ["taskalfa 3253ci"],
        })
        mapping = {
            "id": "device_id", "ts": "timestamp",
            "code": "error_code", "model": "model",
        }
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert result.valid_events[0].model == "Kyocera TASKalfa 3253ci"
        assert result.stats.model_canonicalized == 1


# ── TC-N-26..28: Unit detection integration ────────────────────────────────


class TestNormalizerUnitDetection:
    def test_fraction_0_45_becomes_45(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001", "D002"],
            "ts": ["2024-03-15 10:00:00", "2024-03-15 11:00:00"],
            "toner": [0.45, 0.78],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "toner": "toner_level"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert result.valid_resources["D001"].toner_level == 45
        assert result.valid_resources["D002"].toner_level == 78

    def test_percent_45_stays_45(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001", "D002"],
            "ts": ["2024-03-15 10:00:00", "2024-03-15 11:00:00"],
            "toner": [45, 78],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "toner": "toner_level"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert result.valid_resources["D001"].toner_level == 45
        assert result.valid_resources["D002"].toner_level == 78

    def test_raw_5000_flagged(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "toner": [5000],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "toner": "toner_level"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert result.stats.unit_raw_flagged > 0

    def test_hint_fraction_overrides_autodetect(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "toner": [0.45],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "toner": "toner_level"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(
            df, mapping, resource_unit_hints={"toner_level": "fraction"}
        )
        assert result.valid_resources["D001"].toner_level == 45


# ── TC-N-29..33: ingest_file E2E ──────────────────────────────────────────


class TestIngestFile:
    def test_e2e_happy_path(self, tmp: Path) -> None:
        csv_path = tmp / "fleet.csv"
        csv_path.write_text(
            "device_id,timestamp,error_code,toner_level,model\n"
            "D001,2024-03-15 10:00:00,C6000,80,taskalfa 3253ci\n"
            "D002,2024-03-15 11:00:00,E200,45,hp m404n\n"
            "D001,2024-03-15 12:00:00,,70,taskalfa 3253ci\n",
            encoding="utf-8",
        )
        ref_time = datetime(2024, 3, 16, tzinfo=UTC)
        store = FactorStore(reference_time=ref_time)
        result = ingest_file(csv_path, store)

        assert result.success is True
        assert result.total_records == 3
        assert result.valid_events_count == 2
        assert result.devices_count == 2

        devices = store.list_devices()
        assert "D001" in devices
        assert "D002" in devices

        d001_events = store.get_events("D001", window_days=30)
        assert len(d001_events) == 1
        assert d001_events[0].error_code == "C6000"

        d001_snap = store.get_resources("D001")
        assert d001_snap is not None
        assert d001_snap.toner_level == 70

    def test_e2e_broken_file(self, tmp: Path) -> None:
        p = tmp / "bad.json"
        p.write_text("{BROKEN")
        store = FactorStore()
        result = ingest_file(p, store)

        assert result.success is False
        assert len(result.errors) > 0

    def test_e2e_missing_device_id(self, tmp: Path) -> None:
        csv_path = tmp / "no_id.csv"
        csv_path.write_text(
            "date,value,description\n"
            "2024-03-15,100,test\n",
            encoding="utf-8",
        )
        store = FactorStore()
        result = ingest_file(csv_path, store)

        assert result.success is False
        assert any("device_id" in e for e in result.errors)

    def test_e2e_broken_timestamps_in_invalid(self, tmp: Path) -> None:
        csv_path = tmp / "bad_ts.csv"
        csv_path.write_text(
            "device_id,timestamp,error_code\n"
            "D001,2024-03-15 10:00:00,C6000\n"
            "D002,NOT_A_DATE,E200\n"
            "D003,2024-03-15 12:00:00,SC001\n",
            encoding="utf-8",
        )
        store = FactorStore()
        result = ingest_file(csv_path, store)

        assert result.success is True
        assert result.valid_events_count == 2
        assert len(result.invalid_records) == 1
        assert result.invalid_records[0].row_number == 3

    def test_e2e_factor_store_metadata(self, tmp: Path) -> None:
        csv_path = tmp / "meta.csv"
        csv_path.write_text(
            "device_id,timestamp,error_code,model,vendor,location\n"
            "D001,2024-03-15 10:00:00,C6000,HP M404n,HP,Moscow\n",
            encoding="utf-8",
        )
        store = FactorStore()
        result = ingest_file(csv_path, store)

        assert result.success is True

        meta = store.get_device_metadata("D001")
        assert meta is not None
        assert meta.vendor == "HP"
        assert meta.location == "Moscow"

        fleet = store.fleet_meta
        assert fleet is not None
        assert fleet.source_filename == "meta.csv"
        assert fleet.total_records == 1

    def test_e2e_date_range(self, tmp: Path) -> None:
        csv_path = tmp / "range.csv"
        csv_path.write_text(
            "device_id,timestamp,error_code\n"
            "D001,2024-01-01 00:00:00,C6000\n"
            "D002,2024-06-15 12:00:00,E200\n",
            encoding="utf-8",
        )
        store = FactorStore()
        result = ingest_file(csv_path, store)

        assert result.date_range is not None
        assert result.date_range[0].month == 1
        assert result.date_range[1].month == 6

    def test_e2e_unix_timestamps(self, tmp: Path) -> None:
        csv_path = tmp / "unix.csv"
        csv_path.write_text(
            "device_id,timestamp,error_code\n"
            "D001,1710500000,C6000\n"
            "D002,1710600000,E200\n",
            encoding="utf-8",
        )
        store = FactorStore()
        result = ingest_file(csv_path, store)

        assert result.success is True
        assert result.valid_events_count == 2

    def test_e2e_error_code_normalization(self, tmp: Path) -> None:
        csv_path = tmp / "codes.csv"
        csv_path.write_text(
            "device_id,timestamp,error_code\n"
            "D001,2024-03-15 10:00:00,Error C6000\n"
            'D002,2024-03-15 11:00:00,"Код: E200"\n',
            encoding="utf-8",
        )
        ref_time = datetime(2024, 3, 16, tzinfo=UTC)
        store = FactorStore(reference_time=ref_time)
        result = ingest_file(csv_path, store)

        assert result.success is True
        events = store.get_events("D001", window_days=30)
        assert events[0].error_code == "C6000"
        events2 = store.get_events("D002", window_days=30)
        assert events2[0].error_code == "E200"


# ── TC-N-34: NormalizationStats ────────────────────────────────────────────


class TestNormalizationStats:
    def test_stats_populated(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001", "D002", "D003"],
            "ts": ["2024-03-15 10:00:00", "2024-03-15 11:00:00", "2024-03-15 12:00:00"],
            "code": ["Error C6000", "E200", "SC001"],
            "model": ["taskalfa 3253ci", "SomeModel", "hp m404n"],
        })
        mapping = {
            "id": "device_id", "ts": "timestamp",
            "code": "error_code", "model": "model",
        }
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert result.stats.total_rows == 3
        assert result.stats.valid_events == 3
        assert result.stats.error_code_normalized == 3
        assert result.stats.model_canonicalized == 2


# ── TC-N-35: Row without error_code or resources → event ──────────────────


class TestRowWithoutErrorOrResource:
    def test_plain_row_becomes_event(self, aliases_path: Path) -> None:
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "status": ["active"],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "status": "status"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)

        assert len(result.valid_events) == 1
        assert result.valid_events[0].status == "active"
        assert len(result.valid_resources) == 0
