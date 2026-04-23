"""Comprehensive bug-hunt test suite.

Tests discovered during systematic analysis of all modules.
Each test is named after the bug it targets.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from data_io.factor_store import FactorStore
from data_io.models import (
    ConfidenceFactors,
    Factor,
    HealthResult,
    HealthZone,
    NormalizedEvent,
    SeverityLevel,
    WeightsProfile,
)
from data_io.normalizer import Normalizer, normalize_error_code
from tools.calculator import calculate_health_index, compute_A, compute_C, compute_R


# ══════════════════════════════════════════════════════════════════════════════
# BUG #1: Error code regex doesn't support 3-digit codes (E200, E102, F500)
# ══════════════════════════════════════════════════════════════════════════════


class TestErrorCodeRegex:
    """ERROR_CODE_RE in normalizer.py:54 uses \\d{4,5} — rejects 3-digit codes."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("E200", "E200"),
            ("E102", "E102"),
            ("F500", "F500"),
            ("J100", "J100"),
            ("C100", "C100"),
        ],
    )
    def test_three_digit_letter_codes(self, raw: str, expected: str) -> None:
        """Letters + 3 digits should be valid error codes."""
        result = normalize_error_code(raw)
        assert result == expected, f"normalize_error_code({raw!r}) returned {result!r}"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("C6000", "C6000"),
            ("E1234", "E1234"),
            ("SC543", "SC543"),
            ("S1234", "S1234"),
        ],
    )
    def test_four_five_digit_codes_still_work(self, raw: str, expected: str) -> None:
        """Existing 4-5 digit codes must not break."""
        result = normalize_error_code(raw)
        assert result == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Ошибка E200", "E200"),
            ("Error E102", "E102"),
            ("Код: F500", "F500"),
            ("error C6000", "C6000"),
        ],
    )
    def test_prefix_stripping_with_short_codes(self, raw: str, expected: str) -> None:
        """Prefixes should be stripped before matching short codes."""
        result = normalize_error_code(raw)
        assert result == expected

    def test_xerox_dash_codes_still_work(self) -> None:
        """Xerox XX-YYY-ZZ codes must still be recognized."""
        assert normalize_error_code("07-535-00") == "07-535-00"
        assert normalize_error_code("75-530-00") == "75-530-00"

    def test_invalid_codes_still_rejected(self) -> None:
        """Random strings should still return None."""
        assert normalize_error_code("hello world") is None
        assert normalize_error_code("123") is None
        assert normalize_error_code("") is None


# ══════════════════════════════════════════════════════════════════════════════
# BUG #1b: NormalizedEvent.error_code pattern also rejects 3-digit codes
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizedEventPattern:
    """models.py:581 — NormalizedEvent.error_code pattern validation."""

    def test_three_digit_code_accepted(self) -> None:
        """E200 should be accepted by NormalizedEvent validation."""
        try:
            ev = NormalizedEvent(
                device_id="D001",
                timestamp=datetime.now(UTC),
                error_code="E200",
            )
            assert ev.error_code == "E200"
        except Exception as e:
            pytest.fail(f"NormalizedEvent rejected E200: {e}")

    def test_four_digit_code_accepted(self) -> None:
        """C6000 should still be accepted."""
        ev = NormalizedEvent(
            device_id="D001",
            timestamp=datetime.now(UTC),
            error_code="C6000",
        )
        assert ev.error_code == "C6000"

    def test_xerox_code_accepted(self) -> None:
        """07-535-00 should be accepted."""
        ev = NormalizedEvent(
            device_id="D001",
            timestamp=datetime.now(UTC),
            error_code="07-535-00",
        )
        assert ev.error_code == "07-535-00"


# ══════════════════════════════════════════════════════════════════════════════
# BUG #2: run_batch_lite return type hint mismatch
# ══════════════════════════════════════════════════════════════════════════════


class TestRunBatchLiteSignature:
    """agent/core.py — return type annotation must match actual return."""

    def test_return_type_annotation_has_three_elements(self) -> None:
        """run_batch_lite type hint should include calc_args."""
        import inspect
        from agent.core import Agent

        sig = inspect.signature(Agent.run_batch_lite)
        ret = sig.return_annotation
        # Should include 3 types, not 2
        ret_str = str(ret)
        assert "dict" in ret_str.lower() or ret_str.count(",") >= 2, (
            f"Return annotation is {ret_str!r} — should be 3-tuple including dict"
        )


# ══════════════════════════════════════════════════════════════════════════════
# BUG #3: WeightsProfile frozen mutation
# ══════════════════════════════════════════════════════════════════════════════


class TestWeightsProfileMutation:
    """pages/3_Weights.py mutates .severity sub-model — check if it's actually mutable."""

    def test_severity_weights_are_mutable(self) -> None:
        """SeverityWeights must allow attribute mutation (frozen=False)."""
        wp = WeightsProfile(profile_name="test")
        try:
            wp.severity.critical = 99.0
            assert wp.severity.critical == 99.0
        except Exception as e:
            pytest.fail(
                f"Cannot mutate SeverityWeights.critical: {e}. "
                "3_Weights.py relies on this — if frozen, sliders won't work."
            )

    def test_repeatability_config_mutable(self) -> None:
        """RepeatabilityConfig must be mutable."""
        wp = WeightsProfile(profile_name="test")
        try:
            wp.repeatability.base = 5
            assert wp.repeatability.base == 5
        except Exception as e:
            pytest.fail(f"Cannot mutate RepeatabilityConfig: {e}")

    def test_context_modifier_mutable(self) -> None:
        """ContextModifier must be mutable."""
        wp = WeightsProfile(profile_name="test")
        try:
            wp.context.max_value = 2.0
            assert wp.context.max_value == 2.0
        except Exception as e:
            pytest.fail(f"Cannot mutate ContextConfig: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BUG #4: Weights recalculation (the bug we just fixed)
# ══════════════════════════════════════════════════════════════════════════════


class TestWeightsRecalculation:
    """Verify the raw_factors mechanism works for weight recalculation."""

    def _make_raw_factors(self) -> dict[str, list[dict]]:
        return {
            "DEV001": {
                "factors": [
                    {
                        "error_code": "C6000",
                        "severity_level": "Medium",
                        "n_repetitions": 1,
                        "event_timestamp": "2026-04-15T10:00:00+00:00",
                        "applicable_modifiers": [],
                        "source": "test",
                    },
                ],
                "confidence_factors": {
                    "rag_missing_count": 0,
                    "missing_resources": False,
                    "missing_model": False,
                },
            },
        }

    def test_recalc_with_different_weights_produces_different_index(self) -> None:
        """Changing severity weight must change the health index."""
        raw = self._make_raw_factors()

        w1 = WeightsProfile(profile_name="default")
        w2 = WeightsProfile(profile_name="heavy")
        w2.severity.medium = 40.0

        results = []
        for weights in [w1, w2]:
            calc_args = raw["DEV001"]
            built = []
            now = datetime.now(UTC)

            for fi in calc_args["factors"]:
                severity = SeverityLevel(fi["severity_level"])
                S = getattr(weights.severity, severity.value.lower(), 10.0)
                ts = datetime.fromisoformat(fi["event_timestamp"])
                age_days = max(0, (now - ts).days)
                R = compute_R(fi["n_repetitions"], weights.repeatability.base, weights.repeatability.max_value)
                C = compute_C([], weights.context.max_value)
                A = compute_A(age_days, weights.age.tau_days)

                built.append(Factor(
                    error_code=fi["error_code"],
                    severity_level=severity,
                    S=S,
                    n_repetitions=fi["n_repetitions"],
                    R=R, C=C, A=A,
                    event_timestamp=ts,
                    age_days=age_days,
                ))

            cf = ConfidenceFactors()
            result = calculate_health_index(built, cf, weights, device_id="DEV001")
            results.append(result)

        assert results[0].health_index != results[1].health_index, (
            f"With S_medium=10 got {results[0].health_index}, "
            f"with S_medium=40 got {results[1].health_index} — should differ"
        )

    def test_empty_factors_gives_100(self) -> None:
        """Device with no factors should get index 100 (the old bug behavior)."""
        w = WeightsProfile(profile_name="default")
        cf = ConfidenceFactors()
        result = calculate_health_index([], cf, w, device_id="EMPTY")
        assert result.health_index == 100


# ══════════════════════════════════════════════════════════════════════════════
# BUG #5: Normalizer — rows without error_code or resources
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizerEdgeCases:
    """Edge cases in normalizer that may produce unexpected results."""

    @pytest.fixture
    def aliases_path(self, tmp_path: Path) -> Path:
        p = tmp_path / "aliases.yaml"
        p.write_text("{}", encoding="utf-8")
        return p

    def test_row_with_only_status_no_error(self, aliases_path: Path) -> None:
        """Row with device_id + timestamp + status but no error/resource is silently skipped."""
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "status": ["active"],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "status": "status"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)
        assert result.success is False
        assert len(result.valid_events) == 0
        assert len(result.valid_resources) == 0

    def test_row_with_empty_error_code(self, aliases_path: Path) -> None:
        """Row with empty error_code string should not crash."""
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "err": [""],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "err": "error_code"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)
        assert result.success is True

    def test_only_resource_data_no_error(self, aliases_path: Path) -> None:
        """Row with toner_level but no error should create resource snapshot."""
        df = pd.DataFrame({
            "id": ["D001"],
            "ts": ["2024-03-15 10:00:00"],
            "toner": [50],
        })
        mapping = {"id": "device_id", "ts": "timestamp", "toner": "toner_level"}
        norm = Normalizer(model_aliases_path=aliases_path)
        result = norm.normalize(df, mapping)
        assert result.success is True
        assert len(result.valid_resources) > 0 or len(result.valid_events) > 0


# ══════════════════════════════════════════════════════════════════════════════
# BUG #6: Regex inconsistency between normalizer and field_mapper
# ══════════════════════════════════════════════════════════════════════════════


class TestRegexConsistency:
    """Error code regexes must be consistent across modules."""

    def test_normalizer_and_model_patterns_match(self) -> None:
        """ERROR_CODE_RE in normalizer.py must accept everything NormalizedEvent accepts."""
        import re
        from data_io.normalizer import ERROR_CODE_RE

        # Model pattern from NormalizedEvent
        model_pattern = re.compile(
            r"^([CJEF]\d{4,5}|S\d{3,5}|SC\d{3,4}|\d{2}-\d{3}-\d{2})$"
        )

        test_codes = [
            "C6000", "E1234", "SC543", "S1234", "07-535-00",
            "C12345", "E12345", "F1234", "J1234",
        ]

        for code in test_codes:
            norm_match = bool(ERROR_CODE_RE.match(code))
            model_match = bool(model_pattern.match(code))
            assert norm_match == model_match, (
                f"Inconsistency for {code}: normalizer={norm_match}, model={model_match}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# BUG #7: FactorStore — edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestFactorStoreEdgeCases:
    def test_get_events_empty_device(self) -> None:
        """Getting events for non-existent device should return empty list."""
        fs = FactorStore()
        fs.freeze()
        assert fs.get_events("NONEXISTENT") == []

    def test_count_repetitions_no_events(self) -> None:
        """Count reps for device with no events should be 0."""
        fs = FactorStore()
        fs.freeze()
        assert fs.count_repetitions("NONEXISTENT", "C6000") == 0

    def test_list_devices_includes_metadata_only(self) -> None:
        """Device with only metadata (no events) should still be listed."""
        from data_io.factor_store import DeviceMetadata
        fs = FactorStore()
        fs.set_device_metadata("D001", DeviceMetadata(
            device_id="D001", model="TestModel"
        ))
        fs.freeze()
        assert "D001" in fs.list_devices()

    def test_thaw_allows_mutation(self) -> None:
        """After thaw(), store should accept new data."""
        fs = FactorStore()
        fs.freeze()
        fs.thaw()
        ev = NormalizedEvent(
            device_id="D001",
            timestamp=datetime.now(UTC),
            error_code="C6000",
        )
        fs.add_events("D001", [ev])
        assert len(fs.get_events("D001")) == 1


# ══════════════════════════════════════════════════════════════════════════════
# BUG #8: Calculator edge cases
# ══════════════════════════════════════════════════════════════════════════════


class TestCalculatorEdgeCases:
    def test_single_critical_very_old_event(self) -> None:
        """Very old event should decay to near-zero and not crash."""
        w = WeightsProfile(profile_name="test")
        f = Factor(
            error_code="C6000",
            severity_level=SeverityLevel.CRITICAL,
            S=60.0,
            n_repetitions=1,
            R=1.0, C=1.0,
            A=compute_A(365, w.age.tau_days),
            event_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            age_days=365,
        )
        result = calculate_health_index([f], ConfidenceFactors(), w, device_id="OLD")
        assert 1 <= result.health_index <= 100
        assert result.health_index > 90  # old event should barely affect

    def test_many_info_events_dont_overflow(self) -> None:
        """100 Info events should not push index below 1."""
        w = WeightsProfile(profile_name="test")
        factors = [
            Factor(
                error_code=f"I{i:04d}",
                severity_level=SeverityLevel.INFO,
                S=0.0,
                n_repetitions=1,
                R=1.0, C=1.0, A=1.0,
                event_timestamp=datetime.now(UTC),
                age_days=0,
            )
            for i in range(100)
        ]
        result = calculate_health_index(factors, ConfidenceFactors(), w, device_id="MANY")
        assert result.health_index == 100  # Info S=0 → no penalty

    def test_compute_R_single_rep(self) -> None:
        assert compute_R(1, 2, 5.0) == 1.0

    def test_compute_C_empty_modifiers(self) -> None:
        assert compute_C([], 1.5) == 1.0

    def test_compute_A_zero_days(self) -> None:
        assert compute_A(0, 14) == 1.0

    def test_compute_A_negative_days(self) -> None:
        assert compute_A(-5, 14) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# BUG #9: Ingest full CSV files from fixtures
# ══════════════════════════════════════════════════════════════════════════════


class TestIngestFixtures:
    """End-to-end ingestion of real fixture files."""

    FIXTURES_DIR = Path(__file__).resolve().parent.parent / "files(8)"

    @pytest.fixture
    def aliases_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "configs" / "model_aliases.yaml"

    def _ingest(self, filename: str) -> tuple:
        from data_io.normalizer import ingest_file
        path = self.FIXTURES_DIR / filename
        if not path.exists():
            pytest.skip(f"Fixture not found: {path}")
        fs = FactorStore(reference_time=datetime(2026, 4, 19, tzinfo=UTC))
        result = ingest_file(path, fs)
        return result, fs

    def test_smoke_csv_ingests(self) -> None:
        """01_valid_smoke.csv should ingest without errors."""
        result, fs = self._ingest("01_valid_smoke.csv")
        assert result.success is True
        devices = fs.list_devices()
        assert len(devices) > 0, "No devices loaded from smoke CSV"

    def test_single_critical_csv(self) -> None:
        """04_single_critical.csv should produce one device."""
        result, fs = self._ingest("04_single_critical.csv")
        assert result.success is True

    def test_empty_headers_only(self) -> None:
        """10_empty_headers_only.csv should not crash."""
        result, fs = self._ingest("10_empty_headers_only.csv")
        assert fs.list_devices() == []

    def test_malformed_csv(self) -> None:
        """11_malformed.csv should not crash."""
        result, fs = self._ingest("11_malformed.csv")
        # May or may not succeed, but must not raise

    def test_duplicates_csv(self) -> None:
        """12_duplicates_and_code_formats.csv should handle dups gracefully."""
        result, fs = self._ingest("12_duplicates_and_code_formats.csv")
        assert result.success is True


# ══════════════════════════════════════════════════════════════════════════════
# BUG #10: Session state helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionHelpers:
    """Test the raw_factors session helpers we just added."""

    def test_raw_factors_round_trip(self) -> None:
        """set/get raw_factors should preserve data."""
        from unittest.mock import MagicMock, patch

        mock_state: dict = {}
        with patch("state.session.st") as mock_st:
            mock_st.session_state = mock_state

            from state.session import get_raw_factors, set_raw_factors

            assert get_raw_factors() == {}

            data = {"DEV001": {"factors": [{"code": "C6000"}], "confidence_factors": {}}}
            set_raw_factors(data)
            assert get_raw_factors() == data
