"""Tests for tools/calculator.py — Phase 2.2, TC-A-1 through TC-A-20."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_io.models import (
    ConfidenceFactors,
    ConfidenceZone,
    Factor,
    HealthZone,
    SeverityLevel,
    WeightsProfile,
)
from tools.calculator import (
    calculate_health_index,
    compute_A,
    compute_C,
    compute_R,
    select_one_critical_per_day,
)

NOW = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)
GOLDEN_DIR = Path(__file__).parent / "fixtures"


# ── Fixtures ─────────────────────────────────────────────────────────────────────


@pytest.fixture()
def weights_default() -> WeightsProfile:
    return WeightsProfile(profile_name="default")


@pytest.fixture()
def confidence_empty() -> ConfidenceFactors:
    return ConfidenceFactors()


def _factor(
    *,
    code: str = "C6000",
    severity: SeverityLevel = SeverityLevel.CRITICAL,
    S: float = 60,
    n: int = 1,
    R: float = 1.0,
    C: float = 1.0,
    A: float = 1.0,
    days_ago: float = 0,
    modifiers: list[str] | None = None,
    source: str = "",
) -> Factor:
    return Factor(
        error_code=code,
        severity_level=severity,
        S=S,
        n_repetitions=n,
        R=R,
        C=C,
        A=A,
        event_timestamp=NOW - timedelta(days=days_ago),
        age_days=max(0, int(days_ago)),
        applicable_modifiers=modifiers or [],
        source=source,
    )


# ── TC-A-1: Healthy device (no factors) ─────────────────────────────────────────


class TestTCA1:
    def test_optimistic_mode(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        result = calculate_health_index(
            [], confidence_empty, weights_default,
            device_id="D001", silent_device_mode="optimistic",
        )
        assert result.health_index == 100
        assert result.confidence == 1.0
        assert result.zone == HealthZone.GREEN
        assert result.confidence_zone == ConfidenceZone.HIGH
        assert result.factor_contributions == []

    def test_data_quality_mode(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        result = calculate_health_index(
            [], confidence_empty, weights_default,
            device_id="D001", silent_device_mode="data_quality",
        )
        assert result.health_index == 100
        assert result.confidence == 0.9
        assert result.zone == HealthZone.GREEN

    def test_carry_forward_mode(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        result = calculate_health_index(
            [], confidence_empty, weights_default,
            device_id="D001", silent_device_mode="carry_forward",
        )
        assert result.health_index == 100
        assert result.confidence == 0.9


# ── TC-A-2: Single Medium factor ────────────────────────────────────────────────


class TestTCA2:
    def test_single_medium(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0,
        )
        result = calculate_health_index(
            [f], confidence_empty, weights_default, device_id="D001",
        )
        assert result.health_index == 90
        assert result.zone == HealthZone.GREEN


# ── TC-A-3: Critical + fuser context ────────────────────────────────────────────


class TestTCA3:
    def test_critical_with_fuser(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f = _factor(S=60, R=1.0, C=1.5, A=1.0)
        result = calculate_health_index(
            [f], confidence_empty, weights_default, device_id="D001",
        )
        assert result.health_index == 10
        assert result.zone == HealthZone.RED


# ── TC-A-4: R coefficient and ceiling ────────────────────────────────────────────


class TestTCA4:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1, 1.0),
            (2, 2.0),
            (4, 3.0),
            (8, 4.0),
            (16, 5.0),
            (100, 5.0),
        ],
    )
    def test_compute_R_parametrized(self, n: int, expected: float) -> None:
        assert math.isclose(compute_R(n, base=2, max_value=5.0), expected, abs_tol=1e-6)

    def test_R_ceiling_large_n(self) -> None:
        assert compute_R(100, base=2, max_value=5.0) == 5.0


# ── TC-A-5: C coefficient and ceiling ────────────────────────────────────────────


class TestTCA5:
    def test_multiple_modifiers_capped(self) -> None:
        assert compute_C([1.5, 1.3], max_value=1.5) == 1.5

    def test_single_modifier_below_cap(self) -> None:
        assert math.isclose(compute_C([1.3], max_value=1.5), 1.3)

    def test_empty_modifiers(self) -> None:
        assert compute_C([], max_value=1.5) == 1.0


# ── TC-A-6: Age decay ───────────────────────────────────────────────────────────


class TestTCA6:
    def test_compute_A_30_days(self) -> None:
        A = compute_A(age_days=30, tau_days=14)
        assert math.isclose(A, math.exp(-30 / 14), abs_tol=1e-6)

    def test_compute_A_zero_days(self) -> None:
        assert compute_A(age_days=0, tau_days=14) == 1.0

    def test_compute_A_negative_days(self) -> None:
        assert compute_A(age_days=-1, tau_days=14) == 1.0

    def test_old_critical_event_health(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        A = round(compute_A(30, 14), 4)
        f = _factor(S=60, R=1.0, C=1.0, A=A, days_ago=30)
        result = calculate_health_index(
            [f], confidence_empty, weights_default, device_id="D001",
        )
        expected_penalty = 60 * 1 * 1 * A
        expected_h = max(1, round(100 - expected_penalty))
        assert result.health_index == expected_h
        assert result.zone == HealthZone.GREEN


# ── TC-A-7: One-Critical-per-day rule ────────────────────────────────────────────


class TestTCA7:
    def test_two_critical_same_day_keeps_highest(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f1 = _factor(code="C6000", S=60, R=1.0, C=1.5, A=1.0, days_ago=0)
        f2 = _factor(code="C6020", S=60, R=1.0, C=1.0, A=1.0, days_ago=0)
        result = calculate_health_index(
            [f1, f2], confidence_empty, weights_default, device_id="D001",
        )
        assert result.health_index == 10
        assert len(result.factor_contributions) == 1

    def test_critical_different_days_both_kept(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f1 = _factor(code="C6000", S=60, R=1.0, C=1.0, A=1.0, days_ago=0)
        f2 = _factor(code="C6020", S=60, R=1.0, C=1.0, A=1.0, days_ago=1)
        result = calculate_health_index(
            [f1, f2], confidence_empty, weights_default, device_id="D001",
        )
        assert len(result.factor_contributions) == 2

    def test_non_critical_not_filtered(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f1 = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0, days_ago=0,
        )
        f2 = _factor(
            code="A1002", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0, days_ago=0,
        )
        result = calculate_health_index(
            [f1, f2], confidence_empty, weights_default, device_id="D001",
        )
        assert len(result.factor_contributions) == 2
        assert result.health_index == 80

    def test_select_function_directly(self) -> None:
        f1 = _factor(code="C6000", S=60, R=1.0, C=1.5, A=1.0, days_ago=0)
        f2 = _factor(code="C6020", S=60, R=1.0, C=1.0, A=1.0, days_ago=0)
        f3 = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0, days_ago=0,
        )
        selected = select_one_critical_per_day([f1, f2, f3], limit=1)
        critical = [f for f in selected if f.severity_level == SeverityLevel.CRITICAL]
        assert len(critical) == 1
        assert critical[0].error_code == "C6000"
        assert len(selected) == 2


# ── TC-A-8: Health index floor at 1 ─────────────────────────────────────────────


class TestTCA8:
    def test_huge_penalty_floors_at_1(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f = _factor(S=60, R=5.0, C=1.5, A=1.0)
        result = calculate_health_index(
            [f], confidence_empty, weights_default, device_id="D001",
        )
        assert result.health_index == 1
        assert result.zone == HealthZone.RED


# ── TC-A-9: Confidence floor at 0.2 ─────────────────────────────────────────────


class TestTCA9:
    def test_many_rag_misses_floors_at_02(self, weights_default: WeightsProfile) -> None:
        cf = ConfidenceFactors(rag_missing_count=20)
        f = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0,
        )
        result = calculate_health_index(
            [f], cf, weights_default, device_id="D001",
        )
        assert result.confidence == 0.2
        assert result.confidence_zone == ConfidenceZone.LOW


# ── TC-A-10: Context modifier applicability ──────────────────────────────────────


class TestTCA10:
    def test_fuser_modifier_applies(self) -> None:
        assert math.isclose(compute_C([1.5], max_value=1.5), 1.5)

    def test_no_modifier_C_is_1(self) -> None:
        assert compute_C([], max_value=1.5) == 1.0

    def test_factor_with_vs_without_modifier(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f_plain = _factor(S=60, C=1.0, A=1.0)
        f_fuser = _factor(S=60, C=1.5, A=1.0)

        r1 = calculate_health_index(
            [f_plain], confidence_empty, weights_default, device_id="D001",
        )
        r2 = calculate_health_index(
            [f_fuser], confidence_empty, weights_default, device_id="D001",
        )
        assert r1.health_index > r2.health_index


# ── TC-A-11: Multiple modifiers multiply to ceiling ─────────────────────────────


class TestTCA11:
    def test_product_above_cap(self) -> None:
        assert compute_C([1.5, 1.3], max_value=1.5) == 1.5

    def test_product_below_cap(self) -> None:
        result = compute_C([1.1, 1.2], max_value=1.5)
        assert math.isclose(result, 1.32, abs_tol=1e-6)

    def test_three_modifiers(self) -> None:
        result = compute_C([1.1, 1.1, 1.1], max_value=1.5)
        assert math.isclose(result, 1.331, abs_tol=1e-3)


# ── TC-A-12: Recalc with different zone thresholds ──────────────────────────────


class TestTCA12:
    def test_same_penalty_different_zones(self, confidence_empty: ConfidenceFactors) -> None:
        f = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0,
        )
        w_default = WeightsProfile(profile_name="default")
        w_strict = WeightsProfile(
            profile_name="strict",
            zones={"green_threshold": 95, "red_threshold": 70},
        )

        r1 = calculate_health_index([f], confidence_empty, w_default, device_id="D001")
        r2 = calculate_health_index([f], confidence_empty, w_strict, device_id="D001")

        assert r1.health_index == r2.health_index == 90
        assert r1.zone == HealthZone.GREEN
        assert r2.zone == HealthZone.YELLOW


# ── TC-A-13: Same error, different context → different H ─────────────────────────


class TestTCA13:
    def test_context_changes_result(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f_no = _factor(S=60, C=1.0, A=1.0)
        f_ctx = _factor(S=60, C=1.5, A=1.0)

        r1 = calculate_health_index(
            [f_no], confidence_empty, weights_default, device_id="D001",
        )
        r2 = calculate_health_index(
            [f_ctx], confidence_empty, weights_default, device_id="D001",
        )
        assert r1.health_index == 40
        assert r2.health_index == 10
        assert r1.zone == HealthZone.YELLOW
        assert r2.zone == HealthZone.RED


# ── TC-A-14: Silent device data_quality ──────────────────────────────────────────


class TestTCA14:
    def test_data_quality(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        result = calculate_health_index(
            [], confidence_empty, weights_default,
            device_id="D001", silent_device_mode="data_quality",
        )
        assert result.health_index == 100
        assert result.confidence == 0.9
        assert result.zone == HealthZone.GREEN
        assert len(result.confidence_reasons) == 1


# ── TC-A-15: Silent device optimistic ────────────────────────────────────────────


class TestTCA15:
    def test_optimistic(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        result = calculate_health_index(
            [], confidence_empty, weights_default,
            device_id="D001", silent_device_mode="optimistic",
        )
        assert result.health_index == 100
        assert result.confidence == 1.0
        assert result.confidence_reasons == []


# ── TC-A-16: Missing model → confidence penalty ─────────────────────────────────


class TestTCA16:
    def test_missing_model(self, weights_default: WeightsProfile) -> None:
        cf = ConfidenceFactors(missing_model=True)
        f = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0,
        )
        result = calculate_health_index(
            [f], cf, weights_default, device_id="D001",
        )
        assert math.isclose(result.confidence, 0.6, abs_tol=1e-4)
        assert result.confidence_zone == ConfidenceZone.MEDIUM


# ── TC-A-17: RAG missing 3 codes → 0.7³ ─────────────────────────────────────────


class TestTCA17:
    def test_all_confidence_penalties(self, weights_default: WeightsProfile) -> None:
        cf = ConfidenceFactors(
            rag_missing_count=1,
            missing_resources=True,
            missing_model=True,
            abnormal_daily_jump=True,
            anomalous_event_count=True,
            no_events_and_no_resources=True,
        )
        f = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0,
        )
        result = calculate_health_index(
            [f], cf, weights_default, device_id="D001",
        )
        expected = 0.7 * 0.85 * 0.6 * 0.8 * 0.7 * 0.9
        assert result.confidence == max(0.2, round(expected, 4))
        assert len(result.confidence_reasons) == 6

    def test_rag_missing_3(self, weights_default: WeightsProfile) -> None:
        cf = ConfidenceFactors(rag_missing_count=3)
        f = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0,
        )
        result = calculate_health_index(
            [f], cf, weights_default, device_id="D001",
        )
        expected = round(0.7**3, 4)
        assert math.isclose(result.confidence, expected, abs_tol=1e-4)
        assert result.confidence_zone == ConfidenceZone.LOW


# ── TC-A-18: Snapshot reproducibility (golden file) ──────────────────────────────


class TestTCA18:
    @staticmethod
    def _canonical_result(weights: WeightsProfile) -> dict:
        f1 = _factor(
            code="C6000", S=60, R=1.5, C=1.2, A=0.7,
            days_ago=3, source="manual",
        )
        f2 = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=0.9,
            days_ago=1, source="rag",
        )
        cf = ConfidenceFactors(rag_missing_count=1)
        result = calculate_health_index(
            [f1, f2], cf, weights, device_id="D001",
        )
        data = result.model_dump(mode="json")
        data.pop("calculated_at", None)
        data.pop("calculation_snapshot", None)
        return data

    def test_matches_golden_file(self, weights_default: WeightsProfile) -> None:
        golden_path = GOLDEN_DIR / "golden_health_result.json"
        actual = self._canonical_result(weights_default)
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        assert actual == golden

    def test_deterministic(self, weights_default: WeightsProfile) -> None:
        r1 = self._canonical_result(weights_default)
        r2 = self._canonical_result(weights_default)
        assert r1 == r2


# ── TC-A-19: Config validation (Pydantic rejects invalid values) ─────────────────


class TestTCA19:
    def test_negative_severity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WeightsProfile(
                profile_name="bad",
                severity={"critical": -10, "high": 20, "medium": 10, "low": 3, "info": 0},
            )

    def test_factor_S_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _factor(S=-5)

    def test_factor_R_below_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _factor(R=0.5)

    def test_factor_C_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _factor(C=2.0)

    def test_factor_A_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _factor(A=1.5)

    def test_health_index_below_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            from data_io.models import HealthResult
            HealthResult(
                device_id="D001",
                health_index=0,
                confidence=0.5,
                zone=HealthZone.RED,
                confidence_zone=ConfidenceZone.LOW,
                calculated_at=NOW,
            )


# ── TC-A-20: Input immutability after calculation ────────────────────────────────


class TestTCA20:
    def test_factors_not_mutated(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        f = _factor(S=60, R=2.0, C=1.5, A=0.8)
        snap = (f.S, f.R, f.C, f.A)
        calculate_health_index(
            [f], confidence_empty, weights_default, device_id="D001",
        )
        assert snap == (f.S, f.R, f.C, f.A)

    def test_confidence_factors_not_mutated(self, weights_default: WeightsProfile) -> None:
        cf = ConfidenceFactors(rag_missing_count=3, missing_model=True)
        snap = (cf.rag_missing_count, cf.missing_model)
        f = _factor(
            code="A1001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.0, C=1.0, A=1.0,
        )
        calculate_health_index([f], cf, weights_default, device_id="D001")
        assert (cf.rag_missing_count, cf.missing_model) == snap

    def test_factors_list_not_modified(
        self, weights_default: WeightsProfile, confidence_empty: ConfidenceFactors
    ) -> None:
        factors = [
            _factor(code="C6000", S=60, R=1.0, C=1.0, A=1.0, days_ago=0),
            _factor(code="C6020", S=60, R=1.0, C=1.0, A=1.0, days_ago=0),
        ]
        original_len = len(factors)
        calculate_health_index(
            factors, confidence_empty, weights_default, device_id="D001",
        )
        assert len(factors) == original_len
