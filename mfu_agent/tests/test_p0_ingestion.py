"""P0 ingestion pipeline tests — TC-E-001 through TC-E-058."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data_io.parsers import parse_file, InvalidFileFormatError
from data_io.field_mapper import SynonymMatcher, FieldMapper, SYNONYMS_PATH
from data_io.normalizer import normalize_error_code, canonicalize_model, load_model_aliases, Normalizer
from data_io.factor_store import FactorStore, FactorStoreFrozenError
from data_io.models import NormalizedEvent, ResourceSnapshot


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_tmp(suffix: str, content: str | bytes) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    if isinstance(content, str):
        f.write(content.encode("utf-8"))
    else:
        f.write(content)
    f.close()
    return Path(f.name)


NOW = datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC)


def _make_event(dev: str, ts: datetime, code: str = "C6000") -> NormalizedEvent:
    return NormalizedEvent(device_id=dev, timestamp=ts, error_code=code)


def _make_snapshot(dev: str, ts: datetime) -> ResourceSnapshot:
    return ResourceSnapshot(device_id=dev, timestamp=ts, toner_level=50)


# ── TC-E-001: CSV parsing with standard columns ─────────────────────────────

class TestTCE001:
    def test_csv_standard_columns(self):
        csv_text = (
            "device_id,timestamp,error_code,model\n"
            "DEV001,2026-01-15 10:00:00,C6000,Kyocera TASKalfa 3253ci\n"
            "DEV002,2026-01-15 11:00:00,C6100,HP LaserJet Pro M404n\n"
            "DEV003,2026-01-15 12:00:00,J0511,Ricoh MP C3003\n"
        )
        p = _write_tmp(".csv", csv_text)
        try:
            df = parse_file(p)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            assert list(df.columns) == ["device_id", "timestamp", "error_code", "model"]
            assert df.iloc[0]["device_id"] == "DEV001"
            assert df.iloc[2]["error_code"] == "J0511"
        finally:
            p.unlink(missing_ok=True)


# ── TC-E-003: JSON array parsing ────────────────────────────────────────────

class TestTCE003:
    def test_json_array_parsing(self):
        records = [
            {"device_id": "DEV001", "timestamp": "2026-01-15T10:00:00", "error_code": "C6000"},
            {"device_id": "DEV002", "timestamp": "2026-01-15T11:00:00", "error_code": "C6100"},
        ]
        p = _write_tmp(".json", json.dumps(records))
        try:
            df = parse_file(p)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "device_id" in df.columns
            assert "error_code" in df.columns
            assert df.iloc[0]["device_id"] == "DEV001"
        finally:
            p.unlink(missing_ok=True)


# ── TC-E-006: Binary file with fake .csv extension ──────────────────────────

class TestTCE006:
    def test_binary_with_csv_extension_rejected(self):
        # Write random binary bytes that are not valid CSV text
        binary_data = bytes(range(256)) * 10
        p = _write_tmp(".csv", binary_data)
        try:
            # Binary content should either raise an error during parsing
            # or produce an unusable result. The key assertion: the pipeline
            # does not silently accept garbage as valid data.
            with pytest.raises(Exception):
                df = parse_file(p)
                # If parse_file doesn't raise, verify the result is degenerate
                if len(df) == 0:
                    raise ValueError("empty result from binary file")
        finally:
            p.unlink(missing_ok=True)


# ── TC-E-013: Missing required column (device_id) after mapping ─────────────

class TestTCE013:
    def test_missing_device_id_after_mapping(self):
        csv_text = (
            "foo,bar,baz\n"
            "1,2,3\n"
            "4,5,6\n"
        )
        p = _write_tmp(".csv", csv_text)
        try:
            df = parse_file(p)
            mapper = FieldMapper(llm_client=None)
            result = mapper.map(df)
            # With columns named foo/bar/baz, device_id should not be mapped
            assert "device_id" not in result.auto_mapping.values(), (
                "device_id should not be mapped from arbitrary column names"
            )
        finally:
            p.unlink(missing_ok=True)


# ── TC-E-020: Synonym matching from field_synonyms.yaml ─────────────────────

class TestTCE020:
    def test_synonym_matching(self):
        matcher = SynonymMatcher(SYNONYMS_PATH)
        # "serial_number" should map to "device_id" per synonyms
        result = matcher.match(["serial_number", "datetime", "error_code", "toner"])
        assert result.get("serial_number") == "device_id"
        assert result.get("datetime") == "timestamp"
        assert result.get("error_code") == "error_code"
        assert result.get("toner") == "toner_level"


# ── TC-E-030: Error code normalization ───────────────────────────────────────

class TestTCE030:
    def test_c_6000_normalized(self):
        assert normalize_error_code("C-6000") == "C6000"

    def test_c6000_already_normal(self):
        assert normalize_error_code("C6000") == "C6000"

    def test_with_prefix_stripped(self):
        assert normalize_error_code("error: C-6000") == "C6000"

    def test_spaces_stripped(self):
        assert normalize_error_code("C 6000") == "C6000"

    def test_invalid_returns_none(self):
        assert normalize_error_code("not_a_code") is None

    def test_empty_returns_none(self):
        assert normalize_error_code("") is None


# ── TC-E-031: Model name normalization via aliases ───────────────────────────

class TestTCE031:
    def test_alias_resolves_to_canonical(self):
        aliases = load_model_aliases()
        # "m2040dn" is an alias for "Kyocera ECOSYS M2040dn"
        result = canonicalize_model("m2040dn", aliases)
        assert result == "Kyocera ECOSYS M2040dn"

    def test_canonical_stays_canonical(self):
        aliases = load_model_aliases()
        result = canonicalize_model("Kyocera ECOSYS M2040dn", aliases)
        assert result == "Kyocera ECOSYS M2040dn"

    def test_unknown_model_returned_as_is(self):
        aliases = load_model_aliases()
        result = canonicalize_model("SomeUnknownPrinter XYZ", aliases)
        assert result == "SomeUnknownPrinter XYZ"


# ── TC-E-034: Invalid timestamp → row skipped, processing continues ─────────

class TestTCE034:
    def test_invalid_timestamp_row_skipped(self):
        df = pd.DataFrame({
            "dev": ["DEV1", "DEV2", "DEV3"],
            "ts": ["2026-01-15 10:00:00", "NOT_A_DATE", "2026-01-15 12:00:00"],
            "code": ["C6000", "C6100", "C6200"],
        })
        mapping = {"dev": "device_id", "ts": "timestamp", "code": "error_code"}
        normalizer = Normalizer()
        result = normalizer.normalize(df, mapping)
        # Row with bad timestamp should be in invalid_records
        assert result.stats.invalid_count >= 1
        # Valid rows should still be processed
        assert result.stats.valid_events >= 2
        # Processing continues, not aborted
        assert result.stats.total_rows == 3


# ── TC-E-038: Required field null → row skipped ─────────────────────────────

class TestTCE038:
    def test_null_device_id_row_skipped(self):
        df = pd.DataFrame({
            "dev": ["DEV1", None, "DEV3"],
            "ts": ["2026-01-15 10:00:00", "2026-01-15 11:00:00", "2026-01-15 12:00:00"],
            "code": ["C6000", "C6100", "C6200"],
        })
        mapping = {"dev": "device_id", "ts": "timestamp", "code": "error_code"}
        normalizer = Normalizer()
        result = normalizer.normalize(df, mapping)
        # Row with null device_id should be invalid
        assert result.stats.invalid_count >= 1
        # The other two rows should be valid
        assert result.stats.valid_events >= 2


# ── TC-E-050: FactorStore add_events stores correctly ────────────────────────

class TestTCE050:
    def test_add_events_stores_correctly(self):
        store = FactorStore(reference_time=NOW)
        e1 = [_make_event("DEV1", NOW - timedelta(days=1)),
              _make_event("DEV1", NOW - timedelta(days=2))]
        e2 = [_make_event("DEV2", NOW - timedelta(days=1))]
        store.add_events("DEV1", e1)
        store.add_events("DEV2", e2)
        got1 = store.get_events("DEV1", window_days=30)
        got2 = store.get_events("DEV2", window_days=30)
        assert len(got1) == 2
        assert all(e.device_id == "DEV1" for e in got1)
        assert len(got2) == 1
        assert got2[0].device_id == "DEV2"


# ── TC-E-053: list_devices() returns unique sorted list ──────────────────────

class TestTCE053:
    def test_list_devices_unique_sorted(self):
        store = FactorStore(reference_time=NOW)
        store.add_events("DEV3", [_make_event("DEV3", NOW - timedelta(days=1))])
        store.add_events("DEV1", [_make_event("DEV1", NOW - timedelta(days=1))])
        store.set_resources("DEV2", _make_snapshot("DEV2", NOW))
        store.set_resources("DEV1", _make_snapshot("DEV1", NOW))  # duplicate DEV1
        devices = store.list_devices()
        assert devices == ["DEV1", "DEV2", "DEV3"]
        # No duplicates
        assert len(devices) == len(set(devices))


# ── TC-E-054: freeze() makes store immutable ────────────────────────────────

class TestTCE054:
    def test_freeze_raises_on_add_events(self):
        store = FactorStore(reference_time=NOW)
        store.add_events("DEV1", [_make_event("DEV1", NOW - timedelta(days=1))])
        store.freeze()
        with pytest.raises(FactorStoreFrozenError):
            store.add_events("DEV1", [_make_event("DEV1", NOW - timedelta(days=2))])

    def test_freeze_raises_on_set_resources(self):
        store = FactorStore()
        store.freeze()
        with pytest.raises(FactorStoreFrozenError):
            store.set_resources("DEV1", _make_snapshot("DEV1", NOW))

    def test_freeze_raises_on_set_metadata(self):
        store = FactorStore()
        store.freeze()
        with pytest.raises(FactorStoreFrozenError):
            from data_io.factor_store import DeviceMetadata
            store.set_device_metadata("DEV1", DeviceMetadata(device_id="DEV1"))

    def test_getters_still_work_after_freeze(self):
        store = FactorStore(reference_time=NOW)
        store.add_events("DEV1", [_make_event("DEV1", NOW - timedelta(days=1))])
        store.freeze()
        assert len(store.get_events("DEV1", window_days=30)) == 1
        assert store.list_devices() == ["DEV1"]


# ── TC-E-056: get_events with window_days filter ────────────────────────────

class TestTCE056:
    def test_window_days_filter(self):
        store = FactorStore(reference_time=NOW)
        events = [
            _make_event("DEV1", NOW - timedelta(days=1)),
            _make_event("DEV1", NOW - timedelta(days=10)),
            _make_event("DEV1", NOW - timedelta(days=25)),
            _make_event("DEV1", NOW - timedelta(days=35)),  # outside 30-day window
            _make_event("DEV1", NOW - timedelta(days=60)),  # outside 30-day window
        ]
        store.add_events("DEV1", events)
        got_30 = store.get_events("DEV1", window_days=30)
        assert len(got_30) == 3
        got_7 = store.get_events("DEV1", window_days=7)
        assert len(got_7) == 1
        got_365 = store.get_events("DEV1", window_days=365)
        assert len(got_365) == 5


# ── TC-E-057: count_repetitions returns exact count ──────────────────────────

class TestTCE057:
    def test_count_repetitions_exact(self):
        store = FactorStore(reference_time=NOW)
        events = [
            _make_event("DEV1", NOW - timedelta(days=1), "C6000"),
            _make_event("DEV1", NOW - timedelta(days=5), "C6000"),
            _make_event("DEV1", NOW - timedelta(days=10), "C6000"),
            _make_event("DEV1", NOW - timedelta(days=3), "J0511"),  # different code
            _make_event("DEV1", NOW - timedelta(days=100), "C6000"),  # old
        ]
        store.add_events("DEV1", events)
        assert store.count_repetitions("DEV1", "C6000", window_days=14) == 3
        assert store.count_repetitions("DEV1", "J0511", window_days=14) == 1
        assert store.count_repetitions("DEV1", "C6000", window_days=365) == 4


# ── TC-E-058: Non-existent device_id → empty list, no crash ─────────────────

class TestTCE058:
    def test_nonexistent_device_returns_empty(self):
        store = FactorStore(reference_time=NOW)
        store.add_events("DEV1", [_make_event("DEV1", NOW - timedelta(days=1))])
        # Query for a device that does not exist
        events = store.get_events("NONEXISTENT", window_days=30)
        assert events == []
        resources = store.get_resources("NONEXISTENT")
        assert resources is None
        metadata = store.get_device_metadata("NONEXISTENT")
        assert metadata is None
        count = store.count_repetitions("NONEXISTENT", "C6000", window_days=30)
        assert count == 0
