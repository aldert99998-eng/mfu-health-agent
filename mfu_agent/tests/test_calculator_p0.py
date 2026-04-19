"""P0 test cases for tools/calculator.py — comprehensive verification."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from data_io.models import (
    ConfidenceFactors,
    ConfidenceZone,
    Factor,
    HealthResult,
    HealthZone,
    SeverityLevel,
    WeightsProfile,
)
from tools.calculator import (
    calculate_health_index,
    compute_A,
    compute_C,
    compute_R,
)

NOW = datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _weights() -> WeightsProfile:
    return WeightsProfile(profile_name="test-p0")


def _cf(**kwargs) -> ConfidenceFactors:
    return ConfidenceFactors(**kwargs)


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
        source=source,
    )


def _calc(factors, cf=None, weights=None, **kwargs) -> HealthResult:
    return calculate_health_index(
        factors,
        cf or _cf(),
        weights or _weights(),
        device_id="P0-DEV",
        **kwargs,
    )


# ── TC-A-001: Determinism — same input 10x gives same output ────────────────


class TestTCA001:
    def test_determinism_10x(self):
        """Same input called 10 times must produce identical results."""
        factors = [
            _factor(code="C6000", S=60, R=1.5, C=1.2, A=0.8),
            _factor(code="A1001", severity=SeverityLevel.MEDIUM, S=10, R=1.0, C=1.0, A=0.9),
        ]
        cf = _cf(rag_missing_count=1)
        w = _weights()

        results = [
            calculate_health_index(factors, cf, w, device_id="D-DET")
            for _ in range(10)
        ]

        hi_values = [r.health_index for r in results]
        conf_values = [r.confidence for r in results]
        zone_values = [r.zone for r in results]

        assert len(set(hi_values)) == 1, f"H not deterministic: {hi_values}"
        assert len(set(conf_values)) == 1, f"Conf not deterministic: {conf_values}"
        assert len(set(zone_values)) == 1, f"Zone not deterministic: {zone_values}"


# ── TC-A-002: Factor order independence ─────────────────────────────────────


class TestTCA002:
    def test_factor_order_independence(self):
        """Shuffling factor order must not change the result."""
        base_factors = [
            _factor(code="C6000", S=60, R=1.0, C=1.0, A=1.0, days_ago=0),
            _factor(code="A1001", severity=SeverityLevel.MEDIUM, S=10, R=1.2, C=1.0, A=0.9, days_ago=2),
            _factor(code="B2001", severity=SeverityLevel.HIGH, S=20, R=1.0, C=1.1, A=0.95, days_ago=1),
            _factor(code="D3001", severity=SeverityLevel.LOW, S=3, R=1.0, C=1.0, A=1.0, days_ago=5),
        ]
        cf = _cf()
        w = _weights()

        baseline = calculate_health_index(base_factors, cf, w, device_id="D-ORD")

        rng = random.Random(42)
        for _ in range(5):
            shuffled = list(base_factors)
            rng.shuffle(shuffled)
            result = calculate_health_index(shuffled, cf, w, device_id="D-ORD")
            assert result.health_index == baseline.health_index
            assert result.confidence == baseline.confidence


# ── TC-A-003: No hidden state — sequential calls independent ────────────────


class TestTCA003:
    def test_no_hidden_state(self):
        """Call with bad factors, then good factors — second call must be clean."""
        w = _weights()
        cf = _cf()

        # First call: heavy penalty
        bad = [_factor(S=60, R=5.0, C=1.5, A=1.0)]
        r_bad = _calc(bad, cf, w)
        assert r_bad.health_index == 1

        # Second call: no factors
        r_good = _calc([], cf, w)
        assert r_good.health_index == 100
        assert r_good.zone == HealthZone.GREEN

        # Third call: same bad factors again — must match first
        r_bad2 = _calc(bad, cf, w)
        assert r_bad2.health_index == r_bad.health_index


# ── TC-A-010: No errors -> H=100, green zone ───────────────────────────────


class TestTCA010:
    def test_no_errors_gives_100_green(self):
        """Empty factor list: H=100, zone=green."""
        result = _calc([], silent_device_mode="optimistic")
        assert result.health_index == 100
        assert result.zone == HealthZone.GREEN
        assert result.factor_contributions == []


# ── TC-A-011: 1 minor factor -> moderate drop, still green ──────────────────


class TestTCA011:
    def test_single_low_factor_still_green(self):
        """Single Low factor (S=3): H=97, zone=green."""
        f = _factor(code="L001", severity=SeverityLevel.LOW, S=3, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 97
        assert result.zone == HealthZone.GREEN

    def test_single_info_factor_still_green(self):
        """Single Info factor (S=0): H=100, zone=green."""
        f = _factor(code="I001", severity=SeverityLevel.INFO, S=0, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 100
        assert result.zone == HealthZone.GREEN


# ── TC-A-013: 1 critical factor -> significant drop ────────────────────────


class TestTCA013:
    def test_single_critical_significant_drop(self):
        """Single Critical factor (S=60): H=40, penalty=60."""
        f = _factor(S=60, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 40
        assert result.zone == HealthZone.YELLOW  # 40 >= red_threshold(40)

    def test_critical_with_context_drops_to_red(self):
        """Critical + C=1.5: penalty=90, H=10, red zone."""
        f = _factor(S=60, R=1.0, C=1.5, A=1.0)
        result = _calc([f])
        assert result.health_index == 10
        assert result.zone == HealthZone.RED


# ── TC-A-014: Mixed factors -> matches hand calculation ─────────────────────


class TestTCA014:
    def test_mixed_factors_hand_calculation(self):
        """Hand-verify: Critical(S=60,R=1,C=1,A=1)=60 + Medium(S=10,R=1.5,C=1,A=0.8)=12 = 72 penalty.
        H = max(1, round(100 - 72)) = 28."""
        f1 = _factor(code="C6000", S=60, R=1.0, C=1.0, A=1.0, days_ago=0)
        f2 = _factor(
            code="M001", severity=SeverityLevel.MEDIUM,
            S=10, R=1.5, C=1.0, A=0.8, days_ago=3,
        )
        result = _calc([f1, f2])

        expected_p1 = 60 * 1.0 * 1.0 * 1.0  # 60
        expected_p2 = 10 * 1.5 * 1.0 * 0.8  # 12
        expected_h = max(1, round(100 - (expected_p1 + expected_p2)))
        assert expected_h == 28
        assert result.health_index == expected_h
        assert result.zone == HealthZone.RED


# ── TC-A-030: R=1 no penalty ───────────────────────────────────────────────


class TestTCA030:
    def test_R_1_no_penalty(self):
        """compute_R(n=1) == 1.0 — no repeat penalty."""
        assert compute_R(1, base=2, max_value=5.0) == 1.0

    def test_R_1_in_health_calc(self):
        """Factor with n_repetitions=1, R=1 — penalty is just S*C*A."""
        f = _factor(S=20, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 80  # 100 - 20


# ── TC-A-031: R=2 increases penalty ────────────────────────────────────────


class TestTCA031:
    def test_R_2_increases_penalty(self):
        """compute_R(n=2) == 2.0 — doubles the penalty."""
        assert compute_R(2, base=2, max_value=5.0) == 2.0

    def test_R_2_in_health_calc(self):
        """Factor with R=2 doubles effective penalty."""
        f_r1 = _factor(code="H001", severity=SeverityLevel.HIGH, S=20, R=1.0, C=1.0, A=1.0)
        f_r2 = _factor(code="H001", severity=SeverityLevel.HIGH, S=20, R=2.0, C=1.0, A=1.0)

        r1 = _calc([f_r1])
        r2 = _calc([f_r2])

        assert r1.health_index == 80  # 100 - 20
        assert r2.health_index == 60  # 100 - 40
        assert r2.health_index < r1.health_index


# ── TC-A-040: Context modifier applied when component matches ──────────────


class TestTCA040:
    def test_context_modifier_applied(self):
        """C=1.3 increases penalty vs C=1.0."""
        f_no_ctx = _factor(code="H001", severity=SeverityLevel.HIGH, S=20, R=1.0, C=1.0, A=1.0)
        f_ctx = _factor(code="H001", severity=SeverityLevel.HIGH, S=20, R=1.0, C=1.3, A=1.0)

        r_no = _calc([f_no_ctx])
        r_ctx = _calc([f_ctx])

        assert r_no.health_index == 80   # 100 - 20
        assert r_ctx.health_index == 74  # 100 - 26 = 74
        assert r_ctx.health_index < r_no.health_index


# ── TC-A-041: Context modifier NOT applied when out of scope ────────────────


class TestTCA041:
    def test_no_context_modifier_C_is_1(self):
        """Without context modifier, C=1.0, no extra penalty."""
        assert compute_C([], max_value=1.5) == 1.0

    def test_factor_without_modifier_uses_base_penalty(self):
        """Factor with C=1.0: penalty = S * R * 1.0 * A = S*R*A."""
        f = _factor(S=20, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 80  # 100 - 20*1*1*1


# ── TC-A-043: C ceiling <= 1.5 ─────────────────────────────────────────────


class TestTCA043:
    def test_C_ceiling_clamped(self):
        """Product of modifiers exceeding 1.5 is clamped to 1.5."""
        assert compute_C([1.3, 1.4], max_value=1.5) == 1.5
        assert compute_C([1.5, 1.5], max_value=1.5) == 1.5

    def test_C_at_max_in_factor(self):
        """Factor with C=1.5 (max allowed by Pydantic): penalty = S*R*1.5*A."""
        f = _factor(S=20, R=1.0, C=1.5, A=1.0)
        result = _calc([f])
        assert result.health_index == 70  # 100 - 30

    def test_C_above_1_5_rejected_by_pydantic(self):
        """Factor C > 1.5 is rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            _factor(C=1.6)


# ── TC-A-050: Fresh event (age=0) -> full weight (A~1.0) ───────────────────


class TestTCA050:
    def test_fresh_event_A_is_1(self):
        """compute_A(0, tau) == 1.0 for any tau."""
        assert compute_A(0, tau_days=14) == 1.0
        assert compute_A(0, tau_days=30) == 1.0

    def test_fresh_factor_full_penalty(self):
        """A=1.0 factor: full penalty applied."""
        f = _factor(S=10, A=1.0, days_ago=0)
        result = _calc([f])
        assert result.health_index == 90  # 100 - 10


# ── TC-A-051: 30-day-old event -> reduced weight ───────────────────────────


class TestTCA051:
    def test_30_day_old_event_reduced_A(self):
        """compute_A(30, 14) = exp(-30/14) ~ 0.1173."""
        A = compute_A(30, 14)
        expected = math.exp(-30 / 14)
        assert math.isclose(A, expected, abs_tol=1e-6)
        assert A < 0.15  # Significantly reduced

    def test_30_day_old_critical_reduced_penalty(self):
        """30-day-old critical event with decay has much less impact."""
        A_fresh = 1.0
        A_old = round(compute_A(30, 14), 4)

        f_fresh = _factor(S=60, A=A_fresh, days_ago=0)
        f_old = _factor(S=60, A=A_old, days_ago=30)

        r_fresh = _calc([f_fresh])
        r_old = _calc([f_old])

        assert r_old.health_index > r_fresh.health_index
        assert r_old.zone == HealthZone.GREEN  # Old event barely affects health


# ── TC-A-060-063: Zone boundaries ───────────────────────────────────────────


class TestTCA060_063:
    """Default thresholds: green >= 75, 40 <= yellow < 75, red < 40."""

    def test_060_h_ge_75_is_green(self):
        """H=75 -> green zone."""
        # penalty=25 => H=75
        f = _factor(code="H001", severity=SeverityLevel.HIGH, S=25, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 75
        assert result.zone == HealthZone.GREEN

    def test_061_h_74_is_yellow(self):
        """H=74 -> yellow zone."""
        # penalty=26 => H=74
        f = _factor(code="H001", severity=SeverityLevel.HIGH, S=26, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 74
        assert result.zone == HealthZone.YELLOW

    def test_062_h_40_is_yellow(self):
        """H=40 -> yellow zone (40 >= red_threshold)."""
        f = _factor(S=60, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 40
        assert result.zone == HealthZone.YELLOW

    def test_063_h_39_is_red(self):
        """H=39 -> red zone (39 < red_threshold=40)."""
        # penalty=61 => H=39
        f = _factor(S=61, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 39
        assert result.zone == HealthZone.RED

    def test_exact_green_boundary(self):
        """H exactly at green_threshold (75) is GREEN."""
        w = _weights()
        assert w.zones.green_threshold == 75
        f = _factor(code="X", severity=SeverityLevel.HIGH, S=25, R=1.0, C=1.0, A=1.0)
        result = _calc([f], weights=w)
        assert result.health_index == 75
        assert result.zone == HealthZone.GREEN

    def test_exact_red_boundary(self):
        """H exactly at red_threshold (40) is YELLOW (>= red_threshold)."""
        w = _weights()
        assert w.zones.red_threshold == 40
        f = _factor(S=60, R=1.0, C=1.0, A=1.0)
        result = _calc([f], weights=w)
        assert result.health_index == 40
        assert result.zone == HealthZone.YELLOW


# ── TC-A-070: Confidence=1.0 when all flags clear ──────────────────────────


class TestTCA070:
    def test_confidence_1_when_all_clear(self):
        """All confidence flags clear -> confidence=1.0."""
        cf = _cf()
        f = _factor(code="M001", severity=SeverityLevel.MEDIUM, S=10, R=1.0, C=1.0, A=1.0)
        result = _calc([f], cf)
        assert result.confidence == 1.0
        assert result.confidence_zone == ConfidenceZone.HIGH
        assert result.confidence_reasons == []


# ── TC-A-074: Confidence floor = 0.2 ───────────────────────────────────────


class TestTCA074:
    def test_confidence_floor_0_2(self):
        """Even with all penalties maxed, confidence >= 0.2."""
        cf = _cf(
            rag_missing_count=50,
            missing_resources=True,
            missing_model=True,
            abnormal_daily_jump=True,
            anomalous_event_count=True,
            no_events_and_no_resources=True,
        )
        f = _factor(code="M001", severity=SeverityLevel.MEDIUM, S=10, R=1.0, C=1.0, A=1.0)
        result = _calc([f], cf)
        assert result.confidence == 0.2
        assert result.confidence_zone == ConfidenceZone.LOW

    def test_confidence_never_below_0_2(self):
        """Pydantic validates confidence >= 0.2."""
        with pytest.raises(ValidationError):
            HealthResult(
                device_id="X",
                health_index=50,
                confidence=0.1,  # Below floor
                zone=HealthZone.YELLOW,
                confidence_zone=ConfidenceZone.LOW,
                calculated_at=NOW,
            )


# ── TC-A-080: H ceiling = 100 ──────────────────────────────────────────────


class TestTCA080:
    def test_h_ceiling_100(self):
        """Health index never exceeds 100."""
        result = _calc([], silent_device_mode="optimistic")
        assert result.health_index == 100

    def test_h_ceiling_with_zero_penalty(self):
        """Factor with S=0 doesn't make H > 100."""
        f = _factor(code="I001", severity=SeverityLevel.INFO, S=0, R=1.0, C=1.0, A=1.0)
        result = _calc([f])
        assert result.health_index == 100

    def test_pydantic_rejects_h_above_100(self):
        """Pydantic validation prevents H > 100."""
        with pytest.raises(ValidationError):
            HealthResult(
                device_id="X",
                health_index=101,
                confidence=1.0,
                zone=HealthZone.GREEN,
                confidence_zone=ConfidenceZone.HIGH,
                calculated_at=NOW,
            )


# ── TC-A-081: H floor = 1 (not 0 or negative) ─────────────────────────────


class TestTCA081:
    def test_h_floor_1_with_extreme_penalty(self):
        """Enormous penalty still gives H=1, never 0 or negative."""
        f = _factor(S=60, R=5.0, C=1.5, A=1.0)  # penalty = 450
        result = _calc([f])
        assert result.health_index == 1
        assert result.health_index >= 1

    def test_h_floor_1_multiple_factors(self):
        """Multiple heavy factors: H floors at 1."""
        factors = [
            _factor(code=f"C{i}", S=60, R=3.0, C=1.5, A=1.0, days_ago=i)
            for i in range(5)
        ]
        result = _calc(factors)
        assert result.health_index == 1

    def test_pydantic_rejects_h_0(self):
        """Pydantic validation prevents H=0."""
        with pytest.raises(ValidationError):
            HealthResult(
                device_id="X",
                health_index=0,
                confidence=0.5,
                zone=HealthZone.RED,
                confidence_zone=ConfidenceZone.LOW,
                calculated_at=NOW,
            )


# ── TC-A-083: NaN/inf protection ───────────────────────────────────────────


class TestTCA083:
    def test_inf_S_raises(self):
        """Factor with S=inf raises ValueError (non-finite penalty)."""
        # Pydantic allows float('inf') for S (ge=0 is satisfied),
        # but calculator should raise on non-finite penalty.
        f = Factor(
            error_code="X",
            severity_level=SeverityLevel.CRITICAL,
            S=float("inf"),
            n_repetitions=1,
            R=1.0,
            C=1.0,
            A=1.0,
            event_timestamp=NOW,
            age_days=0,
        )
        with pytest.raises(ValueError, match="Non-finite"):
            _calc([f])

    def test_nan_S_raises(self):
        """Factor with S=NaN raises either ValueError or ValidationError."""
        # NaN fails ge=0 check in Pydantic
        with pytest.raises((ValidationError, ValueError)):
            Factor(
                error_code="X",
                severity_level=SeverityLevel.CRITICAL,
                S=float("nan"),
                n_repetitions=1,
                R=1.0,
                C=1.0,
                A=1.0,
                event_timestamp=NOW,
                age_days=0,
            )


# ── TC-A-090: Sum of contributions ~ total drop from 100 ───────────────────


class TestTCA090:
    def test_contributions_sum_matches_drop(self):
        """Sum of contribution penalties should approximately equal (100 - H)."""
        factors = [
            _factor(code="C6000", S=60, R=1.0, C=1.0, A=1.0, days_ago=0),
            _factor(code="M001", severity=SeverityLevel.MEDIUM, S=10, R=1.0, C=1.0, A=0.9, days_ago=2),
            _factor(code="L001", severity=SeverityLevel.LOW, S=3, R=1.5, C=1.0, A=0.8, days_ago=5),
        ]
        result = _calc(factors)

        penalty_sum = sum(c.penalty for c in result.factor_contributions)
        drop = 100 - result.health_index

        # Allow rounding tolerance: each contribution is rounded to 2 decimals,
        # and H = max(1, round(100 - total_raw_penalty))
        assert abs(penalty_sum - drop) <= len(result.factor_contributions), (
            f"penalty_sum={penalty_sum}, drop={drop}, diff={abs(penalty_sum - drop)}"
        )

    def test_contributions_sum_matches_single_factor(self):
        """Single factor: contribution penalty == drop exactly."""
        f = _factor(code="H001", severity=SeverityLevel.HIGH, S=20, R=1.0, C=1.0, A=1.0)
        result = _calc([f])

        assert len(result.factor_contributions) == 1
        assert result.factor_contributions[0].penalty == 20.0
        assert result.health_index == 80
        assert result.factor_contributions[0].penalty == (100 - result.health_index)
