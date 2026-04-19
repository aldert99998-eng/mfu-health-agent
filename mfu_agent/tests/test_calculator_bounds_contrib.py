"""Tests for tools/calculator.py — Sections 9-11: Bounds, FactorContribution, CalculationSnapshot.

TC-A-080 .. TC-A-083  (Section 9  — Bounds)
TC-A-090 .. TC-A-093  (Section 10 — FactorContribution)
TC-A-100 .. TC-A-102  (Section 11 — CalculationSnapshot)
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.calculator import calculate_health_index
from data_io.models import (
    Factor,
    ConfidenceFactors,
    WeightsProfile,
    SeverityLevel,
    HealthZone,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_factor(
    error_code: str = "E001",
    severity: SeverityLevel = SeverityLevel.LOW,
    S: float = 3.0,
    R: float = 1.0,
    C: float = 1.0,
    A: float = 1.0,
) -> Factor:
    return Factor(
        error_code=error_code,
        severity_level=severity,
        S=S,
        n_repetitions=1,
        R=R,
        C=C,
        A=A,
        event_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        age_days=0,
    )


def default_weights() -> WeightsProfile:
    return WeightsProfile(profile_name="default-test", version="1.0")


def empty_confidence() -> ConfidenceFactors:
    return ConfidenceFactors()


def run_calc(factors: list[Factor]) -> "HealthResult":  # noqa: F821
    return calculate_health_index(
        factors,
        empty_confidence(),
        default_weights(),
        device_id="dev-test",
        silent_device_mode="data_quality",
    )


def report(tc: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {tc}{suffix}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9 — Bounds
# ═══════════════════════════════════════════════════════════════════════════════


def test_tc_a_080_empty_factors_gives_100():
    """TC-A-080: Empty factors list yields H=100."""
    result = run_calc([])
    passed = result.health_index == 100
    report("TC-A-080", passed, f"H={result.health_index}")
    assert passed


def test_tc_a_081_floor_clamp_to_1():
    """TC-A-081: Many heavy factors clamp H to 1 (never 0 or negative)."""
    factors = [
        make_factor(error_code=f"E{i:03d}", severity=SeverityLevel.CRITICAL,
                    S=60.0, R=1.0, C=1.0, A=1.0)
        for i in range(10)
    ]
    # Total penalty = 10 * 60 = 600, raw = 100 - 600 = -500 -> clamped to 1.
    # Note: one-critical-per-day rule may filter some out if same day.
    # All factors share the same event_timestamp so only `limit` per day survive.
    # With default limit=1, only 1 critical survives -> penalty=60 -> H=40.
    # To truly exceed 100 penalty we use non-critical factors instead.
    factors_nc = [
        make_factor(error_code=f"E{i:03d}", severity=SeverityLevel.HIGH,
                    S=60.0, R=1.0, C=1.0, A=1.0)
        for i in range(10)
    ]
    result = run_calc(factors_nc)
    passed = result.health_index == 1
    report("TC-A-081", passed, f"H={result.health_index} (expected 1)")
    assert passed


def test_tc_a_082_health_index_is_int():
    """TC-A-082: health_index is always int type."""
    result_empty = run_calc([])
    result_one = run_calc([make_factor(S=10.0)])
    ok1 = isinstance(result_empty.health_index, int)
    ok2 = isinstance(result_one.health_index, int)
    passed = ok1 and ok2
    report("TC-A-082", passed,
           f"types: empty={type(result_empty.health_index).__name__}, "
           f"one={type(result_one.health_index).__name__}")
    assert passed


def test_tc_a_083_inf_and_nan():
    """TC-A-083: Factor with S=inf raises ValueError; S=nan raises ValidationError."""
    # S=inf: Pydantic accepts (inf >= 0 is True), but calculator rejects non-finite penalty.
    ok_inf = False
    detail_inf = ""
    try:
        f_inf = make_factor(S=float("inf"))
        run_calc([f_inf])
        detail_inf = "No error raised — unexpected"
    except ValueError:
        ok_inf = True
        detail_inf = "ValueError raised as expected"

    # S=nan: nan >= 0 is False in Python, so Pydantic should reject it.
    ok_nan = False
    detail_nan = ""
    try:
        make_factor(S=float("nan"))
        detail_nan = "Factor created (no error) — unexpected"
    except (ValidationError, ValueError):
        ok_nan = True
        detail_nan = "ValidationError raised as expected"

    passed = ok_inf and ok_nan
    report("TC-A-083", passed, f"inf: {detail_inf}; nan: {detail_nan}")
    assert passed


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10 — FactorContribution
# ═══════════════════════════════════════════════════════════════════════════════


def _mixed_factors() -> list[Factor]:
    """1 High (S=20, R=1.5, C=1.2, A=0.8) + 2 Low (S=3, R=1, C=1, A=1)."""
    return [
        make_factor(error_code="E100", severity=SeverityLevel.HIGH,
                    S=20.0, R=1.5, C=1.2, A=0.8),
        make_factor(error_code="E200", severity=SeverityLevel.LOW,
                    S=3.0, R=1.0, C=1.0, A=1.0),
        make_factor(error_code="E201", severity=SeverityLevel.LOW,
                    S=3.0, R=1.0, C=1.0, A=1.0),
    ]


def test_tc_a_090_penalty_sum_matches_index():
    """TC-A-090: sum(penalty) ~ (100 - H) with +/-1 rounding tolerance."""
    factors = _mixed_factors()
    result = run_calc(factors)
    penalty_sum = sum(c.penalty for c in result.factor_contributions)
    expected_gap = 100 - result.health_index
    diff = abs(penalty_sum - expected_gap)
    passed = diff <= 1.0
    report("TC-A-090", passed,
           f"penalty_sum={penalty_sum:.2f}, 100-H={expected_gap}, diff={diff:.2f}")
    assert passed


def test_tc_a_091_contribution_fields_filled():
    """TC-A-091: Each contribution has all required fields filled (not None/empty except source)."""
    result = run_calc(_mixed_factors())
    all_ok = True
    details: list[str] = []
    for i, c in enumerate(result.factor_contributions):
        problems: list[str] = []
        if not c.label:
            problems.append("label empty")
        if c.penalty is None:
            problems.append("penalty None")
        if c.S is None:
            problems.append("S None")
        if c.R is None:
            problems.append("R None")
        if c.C is None:
            problems.append("C None")
        if c.A is None:
            problems.append("A None")
        if c.source is None:
            problems.append("source None")
        if problems:
            all_ok = False
            details.append(f"contrib[{i}]: {', '.join(problems)}")

    passed = all_ok and len(result.factor_contributions) > 0
    report("TC-A-091", passed,
           "; ".join(details) if details else f"{len(result.factor_contributions)} contributions OK")
    assert passed


def test_tc_a_092_no_negative_penalty():
    """TC-A-092: No contribution has a negative penalty."""
    result = run_calc(_mixed_factors())
    negatives = [c for c in result.factor_contributions if c.penalty < 0]
    passed = len(negatives) == 0
    report("TC-A-092", passed,
           f"negative count={len(negatives)}")
    assert passed


def test_tc_a_093_contributions_sorted_descending():
    """TC-A-093: Contributions are sorted by penalty descending."""
    result = run_calc(_mixed_factors())
    penalties = [c.penalty for c in result.factor_contributions]
    sorted_desc = sorted(penalties, reverse=True)
    passed = penalties == sorted_desc
    report("TC-A-093", passed, f"penalties={penalties}")
    assert passed


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11 — CalculationSnapshot
# ═══════════════════════════════════════════════════════════════════════════════


def test_tc_a_100_snapshot_contains_weights_data():
    """TC-A-100: calculation_snapshot contains WeightsProfile data."""
    w = default_weights()
    result = calculate_health_index(
        [], empty_confidence(), w, device_id="snap-dev",
    )
    snap = result.calculation_snapshot
    # The snapshot should contain keys from WeightsProfile
    ok_severity = "severity" in snap
    ok_zones = "zones" in snap
    ok_age = "age" in snap
    passed = ok_severity and ok_zones and ok_age
    report("TC-A-100", passed,
           f"keys present: severity={ok_severity}, zones={ok_zones}, age={ok_age}")
    assert passed


def test_tc_a_101_snapshot_has_profile_name_and_version():
    """TC-A-101: snapshot contains profile_name and version."""
    w = WeightsProfile(profile_name="custom-profile", version="2.5")
    result = calculate_health_index(
        [], empty_confidence(), w, device_id="snap-dev",
    )
    snap = result.calculation_snapshot
    ok_name = snap.get("profile_name") == "custom-profile"
    ok_ver = snap.get("version") == "2.5"
    passed = ok_name and ok_ver
    report("TC-A-101", passed,
           f"profile_name={snap.get('profile_name')!r}, version={snap.get('version')!r}")
    assert passed


def test_tc_a_102_snapshot_is_plain_dict_and_isolated():
    """TC-A-102: snapshot is a plain dict (not Pydantic model); mutating it does not affect original weights."""
    w = WeightsProfile(profile_name="immutable-check", version="1.0")
    result = calculate_health_index(
        [], empty_confidence(), w, device_id="snap-dev",
    )
    snap = result.calculation_snapshot

    # Must be a plain dict
    ok_type = type(snap) is dict
    detail_type = f"type={type(snap).__name__}"

    # Mutate the snapshot and verify original weights are unaffected
    original_name = w.profile_name
    snap["profile_name"] = "MUTATED"
    ok_isolated = w.profile_name == original_name
    detail_iso = (f"weights.profile_name={w.profile_name!r} "
                  f"(expected {original_name!r})")

    passed = ok_type and ok_isolated
    report("TC-A-102", passed, f"{detail_type}; {detail_iso}")
    assert passed


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        # Section 9
        test_tc_a_080_empty_factors_gives_100,
        test_tc_a_081_floor_clamp_to_1,
        test_tc_a_082_health_index_is_int,
        test_tc_a_083_inf_and_nan,
        # Section 10
        test_tc_a_090_penalty_sum_matches_index,
        test_tc_a_091_contribution_fields_filled,
        test_tc_a_092_no_negative_penalty,
        test_tc_a_093_contributions_sorted_descending,
        # Section 11
        test_tc_a_100_snapshot_contains_weights_data,
        test_tc_a_101_snapshot_has_profile_name_and_version,
        test_tc_a_102_snapshot_is_plain_dict_and_isolated,
    ]

    total = len(tests)
    passed_count = 0
    failed_count = 0

    print(f"\n{'=' * 70}")
    print("Track A Calculator — Sections 9-11: Bounds, Contribution, Snapshot")
    print(f"{'=' * 70}\n")

    for fn in tests:
        try:
            fn()
            passed_count += 1
        except Exception as exc:
            failed_count += 1
            # report() already printed FAIL inside the test for assertion
            # failures; only log unexpected exceptions here.
            if "Assertion" not in type(exc).__name__:
                report(fn.__name__, False, f"EXCEPTION: {exc}")

    print(f"\n{'─' * 70}")
    print(f"Total: {total}  |  Passed: {passed_count}  |  Failed: {failed_count}")
    print(f"{'─' * 70}\n")
