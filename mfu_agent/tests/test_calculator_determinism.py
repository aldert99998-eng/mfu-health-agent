"""Track A Calculator tests -- sections 1-2: Determinism + Basic Formula.

TC-A-001 .. TC-A-004  (Section 1: Determinism)
TC-A-010 .. TC-A-014  (Section 2: Basic Formula)
"""

from __future__ import annotations

import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.calculator import calculate_health_index, compute_R, compute_C, compute_A
from data_io.models import (
    Factor,
    ConfidenceFactors,
    WeightsProfile,
    SeverityLevel,
    HealthZone,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = datetime(2025, 1, 1, tzinfo=UTC)


def _weights() -> WeightsProfile:
    return WeightsProfile(profile_name="test-default")


def _cf() -> ConfidenceFactors:
    return ConfidenceFactors()


def _factor(
    error_code: str = "E001",
    severity: SeverityLevel = SeverityLevel.LOW,
    S: float = 3.0,
    R: float = 1.0,
    C: float = 1.0,
    A: float = 1.0,
    n_repetitions: int = 1,
    age_days: int = 0,
) -> Factor:
    return Factor(
        error_code=error_code,
        severity_level=severity,
        S=S,
        n_repetitions=n_repetitions,
        R=R,
        C=C,
        A=A,
        event_timestamp=TS,
        age_days=age_days,
    )


def _run_test(tc_id: str, description: str, test_fn):
    """Run a single test function and print PASS/FAIL."""
    try:
        test_fn()
        print(f"PASS  {tc_id}: {description}")
    except Exception as exc:
        print(f"FAIL  {tc_id}: {description}")
        print(f"      Error: {exc}")


# ===========================================================================
# Section 1 -- Determinism
# ===========================================================================


def test_tc_a_001():
    """TC-A-001: Same input 10 times -> identical results (except calculated_at)."""
    factors = [_factor()]
    results = [
        calculate_health_index(factors, _cf(), _weights(), device_id="dev-001")
        for _ in range(10)
    ]
    ref = results[0]
    for i, r in enumerate(results[1:], start=2):
        assert r.health_index == ref.health_index, (
            f"Run {i}: health_index {r.health_index} != {ref.health_index}"
        )
        assert r.confidence == ref.confidence, (
            f"Run {i}: confidence {r.confidence} != {ref.confidence}"
        )
        assert r.zone == ref.zone, f"Run {i}: zone {r.zone} != {ref.zone}"
        assert r.factor_contributions == ref.factor_contributions, (
            f"Run {i}: contributions differ"
        )


def test_tc_a_002():
    """TC-A-002: Different factor order -> same health_index."""
    f1 = _factor("ERR_A", SeverityLevel.CRITICAL, S=60)
    f2 = _factor("ERR_B", SeverityLevel.HIGH, S=20)
    f3 = _factor("ERR_C", SeverityLevel.LOW, S=3)

    r_abc = calculate_health_index([f1, f2, f3], _cf(), _weights())
    r_cba = calculate_health_index([f3, f2, f1], _cf(), _weights())
    r_bac = calculate_health_index([f2, f1, f3], _cf(), _weights())

    assert r_abc.health_index == r_cba.health_index == r_bac.health_index, (
        f"Order-dependent: {r_abc.health_index}, {r_cba.health_index}, {r_bac.health_index}"
    )


def test_tc_a_003():
    """TC-A-003: No hidden state between calls (A, then B, then A again)."""
    factors_a = [_factor("ERR_A", SeverityLevel.HIGH, S=20)]
    factors_b = [
        _factor("ERR_B", SeverityLevel.CRITICAL, S=60),
        _factor("ERR_C", SeverityLevel.CRITICAL, S=60),
    ]

    r_a1 = calculate_health_index(factors_a, _cf(), _weights())
    _r_b = calculate_health_index(factors_b, _cf(), _weights())
    r_a2 = calculate_health_index(factors_a, _cf(), _weights())

    assert r_a1.health_index == r_a2.health_index, (
        f"Hidden state detected: first={r_a1.health_index}, second={r_a2.health_index}"
    )
    assert r_a1.confidence == r_a2.confidence


def test_tc_a_004():
    """TC-A-004: Thread safety -- 10 threads same input -> identical results."""
    factors = [
        _factor("ERR_X", SeverityLevel.HIGH, S=20),
        _factor("ERR_Y", SeverityLevel.LOW, S=3),
    ]
    results: list = [None] * 10
    errors: list = []

    def worker(idx: int):
        try:
            results[idx] = calculate_health_index(
                factors, _cf(), _weights(), device_id=f"thread-{idx}"
            )
        except Exception as exc:
            errors.append((idx, exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"

    ref = results[0]
    for i, r in enumerate(results[1:], start=1):
        assert r is not None, f"Thread {i} returned None"
        assert r.health_index == ref.health_index, (
            f"Thread {i}: {r.health_index} != {ref.health_index}"
        )
        assert r.confidence == ref.confidence, (
            f"Thread {i}: confidence {r.confidence} != {ref.confidence}"
        )


# ===========================================================================
# Section 2 -- Basic Formula
# ===========================================================================


def test_tc_a_010():
    """TC-A-010: Empty factors -> H=100, zone=green, confidence >= 0.9."""
    r = calculate_health_index([], _cf(), _weights())
    assert r.health_index == 100, f"Expected H=100, got {r.health_index}"
    assert r.zone == HealthZone.GREEN, f"Expected green, got {r.zone}"
    assert r.confidence >= 0.9, f"Expected confidence >= 0.9, got {r.confidence}"


def test_tc_a_011():
    """TC-A-011: One Low factor (S=3, R=1, C=1, A=1) -> H=97, green."""
    f = _factor("LOW_001", SeverityLevel.LOW, S=3, R=1, C=1, A=1)
    r = calculate_health_index([f], _cf(), _weights())
    # H = max(1, round(100 - 3*1*1*1)) = 97
    assert r.health_index == 97, f"Expected H=97, got {r.health_index}"
    assert r.zone == HealthZone.GREEN, f"Expected green, got {r.zone}"


def test_tc_a_012():
    """TC-A-012: One High factor (S=20, R=1, C=1, A=1) -> H=80, green/yellow boundary."""
    f = _factor("HIGH_001", SeverityLevel.HIGH, S=20, R=1, C=1, A=1)
    r = calculate_health_index([f], _cf(), _weights())
    # H = max(1, round(100 - 20)) = 80
    assert r.health_index == 80, f"Expected H=80, got {r.health_index}"
    # green_threshold=75 by default, so 80 >= 75 -> green
    assert r.zone == HealthZone.GREEN, f"Expected green, got {r.zone}"


def test_tc_a_013():
    """TC-A-013: One Critical factor (S=60, R=1, C=1, A=1) -> H=40, yellow."""
    f = _factor("CRIT_001", SeverityLevel.CRITICAL, S=60, R=1, C=1, A=1)
    r = calculate_health_index([f], _cf(), _weights())
    # H = max(1, round(100 - 60)) = 40
    assert r.health_index == 40, f"Expected H=40, got {r.health_index}"
    # red_threshold=40 by default, so 40 >= 40 -> yellow (not red)
    assert r.zone == HealthZone.YELLOW, f"Expected yellow, got {r.zone}"


def test_tc_a_014():
    """TC-A-014: Mixed factors (1 critical + 2 high + 5 low) -> verify manual calculation."""
    factors = [
        _factor("CRIT_001", SeverityLevel.CRITICAL, S=60, R=1, C=1, A=1),
        _factor("HIGH_001", SeverityLevel.HIGH, S=20, R=1, C=1, A=1),
        _factor("HIGH_002", SeverityLevel.HIGH, S=20, R=1, C=1, A=1),
        _factor("LOW_001", SeverityLevel.LOW, S=3, R=1, C=1, A=1),
        _factor("LOW_002", SeverityLevel.LOW, S=3, R=1, C=1, A=1),
        _factor("LOW_003", SeverityLevel.LOW, S=3, R=1, C=1, A=1),
        _factor("LOW_004", SeverityLevel.LOW, S=3, R=1, C=1, A=1),
        _factor("LOW_005", SeverityLevel.LOW, S=3, R=1, C=1, A=1),
    ]
    r = calculate_health_index(factors, _cf(), _weights())

    # Manual: penalty = 60 + 20 + 20 + 5*3 = 115
    # H = max(1, round(100 - 115)) = max(1, -15) = 1
    expected_penalty = 60 + 20 + 20 + 5 * 3  # = 115
    expected_h = max(1, round(100 - expected_penalty))  # = 1
    assert r.health_index == expected_h, (
        f"Expected H={expected_h}, got {r.health_index} "
        f"(total penalty should be {expected_penalty})"
    )
    assert r.zone == HealthZone.RED, f"Expected red, got {r.zone}"

    # Verify total penalty from contributions matches manual calculation
    actual_penalty = sum(c.penalty for c in r.factor_contributions)
    assert actual_penalty == expected_penalty, (
        f"Contribution penalty sum {actual_penalty} != expected {expected_penalty}"
    )


# ===========================================================================
# CLI runner -- print PASS / FAIL per test case
# ===========================================================================

if __name__ == "__main__":
    tests = [
        ("TC-A-001", "Same input 10x -> identical results", test_tc_a_001),
        ("TC-A-002", "Different factor order -> same health_index", test_tc_a_002),
        ("TC-A-003", "No hidden state between calls", test_tc_a_003),
        ("TC-A-004", "Thread safety -- 10 threads same input", test_tc_a_004),
        ("TC-A-010", "Empty factors -> H=100, green, conf>=0.9", test_tc_a_010),
        ("TC-A-011", "One Low (S=3) -> H=97, green", test_tc_a_011),
        ("TC-A-012", "One High (S=20) -> H=80, green", test_tc_a_012),
        ("TC-A-013", "One Critical (S=60) -> H=40, yellow", test_tc_a_013),
        ("TC-A-014", "Mixed 1C+2H+5L -> H=1, red", test_tc_a_014),
    ]

    print("=" * 60)
    print("Track A Calculator -- Sections 1-2")
    print("=" * 60)
    for tc_id, desc, fn in tests:
        _run_test(tc_id, desc, fn)
    print("=" * 60)
