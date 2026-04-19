"""Track A Calculator tests -- Sections 3-5.

Section 3: Severity weights
Section 4: Repetitions (compute_R)
Section 5: Component modifiers (compute_C)
"""

from __future__ import annotations

import sys
import math
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.calculator import calculate_health_index, compute_R, compute_C, compute_A
from data_io.models import (
    Factor,
    ConfidenceFactors,
    WeightsProfile,
    SeverityLevel,
    HealthZone,
    SeverityWeights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = datetime(2025, 1, 1, tzinfo=UTC)
EMPTY_CONF = ConfidenceFactors()

passed = 0
failed = 0


def default_weights() -> WeightsProfile:
    return WeightsProfile(profile_name="test")


def make_factor(
    severity: SeverityLevel,
    S: float,
    *,
    n_repetitions: int = 1,
    R: float = 1.0,
    C: float = 1.0,
    A: float = 1.0,
    error_code: str = "E001",
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
        age_days=0,
    )


def check(test_id: str, description: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if not condition:
        failed += 1
        suffix = f" -- {detail}" if detail else ""
        print(f"  [{status}] {test_id}: {description}{suffix}")
    else:
        passed += 1
        print(f"  [{status}] {test_id}: {description}")


# ===========================================================================
# Section 3 -- Severity weights
# ===========================================================================

def test_section_3():
    print("\n=== Section 3: Severity weights ===")
    w = default_weights()

    # TC-A-020: Low weight S=3 -> H=97
    f = make_factor(SeverityLevel.LOW, S=3)
    result = calculate_health_index([f], EMPTY_CONF, w)
    check("TC-A-020", "Low weight S=3 -> H=97", result.health_index == 97,
          f"got H={result.health_index}")

    # TC-A-021: High weight S=20 -> H=80
    f = make_factor(SeverityLevel.HIGH, S=20)
    result = calculate_health_index([f], EMPTY_CONF, w)
    check("TC-A-021", "High weight S=20 -> H=80", result.health_index == 80,
          f"got H={result.health_index}")

    # TC-A-022: Critical weight S=60 -> H=40
    f = make_factor(SeverityLevel.CRITICAL, S=60)
    result = calculate_health_index([f], EMPTY_CONF, w)
    check("TC-A-022", "Critical weight S=60 -> H=40", result.health_index == 40,
          f"got H={result.health_index}")

    # TC-A-023: Custom WeightsProfile with modified severity weights
    custom_w = WeightsProfile(
        profile_name="custom",
        severity=SeverityWeights(critical=50, high=15, medium=8, low=2, info=0),
    )
    f = make_factor(SeverityLevel.CRITICAL, S=50)
    result = calculate_health_index([f], EMPTY_CONF, custom_w)
    check("TC-A-023", "Custom severity weights respected (critical=50 -> H=50)",
          result.health_index == 50,
          f"got H={result.health_index}")

    # TC-A-024: Zero weight (S=0) -> no penalty, H=100
    f = make_factor(SeverityLevel.INFO, S=0)
    result = calculate_health_index([f], EMPTY_CONF, w)
    check("TC-A-024", "Zero weight S=0 -> H=100", result.health_index == 100,
          f"got H={result.health_index}")


# ===========================================================================
# Section 4 -- Repetitions (compute_R)
# ===========================================================================

def test_section_4():
    print("\n=== Section 4: Repetitions (compute_R) ===")

    # TC-A-030: compute_R(1, 2, 5.0) = 1.0
    r = compute_R(1, 2, 5.0)
    check("TC-A-030", "compute_R(1, 2, 5.0) == 1.0", r == 1.0,
          f"got {r}")

    # TC-A-031: compute_R(2, 2, 5.0) = 1 + log2(2) = 2.0
    r = compute_R(2, 2, 5.0)
    check("TC-A-031", "compute_R(2, 2, 5.0) == 2.0", abs(r - 2.0) < 1e-9,
          f"got {r}")

    # TC-A-032: compute_R(10, 2, 5.0) saturates at max_value
    r = compute_R(10, 2, 5.0)
    raw = 1.0 + math.log(10, 2)  # ~4.3219
    expected = min(5.0, raw)
    check("TC-A-032", "compute_R(10, 2, 5.0) saturates correctly",
          abs(r - expected) < 1e-9 and r <= 5.0,
          f"got {r}, expected {expected}")

    # TC-A-033: n_repetitions=0 rejected by Factor(n_repetitions ge=1)
    rejected = False
    try:
        Factor(
            error_code="E001",
            severity_level=SeverityLevel.LOW,
            S=3,
            n_repetitions=0,
            R=1.0, C=1.0, A=1.0,
            event_timestamp=TS,
            age_days=0,
        )
    except Exception:
        rejected = True
    check("TC-A-033", "n_repetitions=0 rejected by validation", rejected)

    # TC-A-034: compute_R(100, 2, 5.0) capped at 5.0
    r = compute_R(100, 2, 5.0)
    check("TC-A-034", "compute_R(100, 2, 5.0) capped at 5.0", r == 5.0,
          f"got {r}")


# ===========================================================================
# Section 5 -- Component modifiers (compute_C)
# ===========================================================================

def test_section_5():
    print("\n=== Section 5: Component modifiers (compute_C) ===")

    # TC-A-040: compute_C([1.3], 1.5) = 1.3
    c = compute_C([1.3], 1.5)
    check("TC-A-040", "compute_C([1.3], 1.5) == 1.3", abs(c - 1.3) < 1e-9,
          f"got {c}")

    # TC-A-041: compute_C([], 1.5) = 1.0
    c = compute_C([], 1.5)
    check("TC-A-041", "compute_C([], 1.5) == 1.0", c == 1.0,
          f"got {c}")

    # TC-A-042: compute_C([1.3, 1.2], 1.5) = min(1.5, 1.56) = 1.5
    c = compute_C([1.3, 1.2], 1.5)
    check("TC-A-042", "compute_C([1.3, 1.2], 1.5) == 1.5 (clamped)",
          abs(c - 1.5) < 1e-9,
          f"got {c}")

    # TC-A-043: Ceiling C<=1.5 with multiple multipliers
    c = compute_C([1.3, 1.4, 1.2], 1.5)
    check("TC-A-043", "compute_C([1.3, 1.4, 1.2], 1.5) == 1.5 (ceiling)",
          abs(c - 1.5) < 1e-9,
          f"got {c}")

    # TC-A-044: 1 Critical + C=1.5 + R=2 -> H clamped to 1
    w = default_weights()
    f = make_factor(SeverityLevel.CRITICAL, S=60, R=2.0, C=1.5, A=1.0,
                    n_repetitions=2)
    result = calculate_health_index([f], EMPTY_CONF, w)
    # penalty = 60 * 2 * 1.5 * 1.0 = 180, H = max(1, 100 - 180) = 1
    check("TC-A-044", "Critical S=60, R=2, C=1.5, A=1 -> H=1 (floor clamp)",
          result.health_index == 1,
          f"got H={result.health_index}")

    # TC-A-045: Modifier by model -- compute_C with varied multiplier lists
    c1 = compute_C([1.1], 1.5)
    c2 = compute_C([1.1, 1.05], 1.5)
    c3 = compute_C([1.5], 1.5)
    check("TC-A-045a", "compute_C([1.1], 1.5) == 1.1",
          abs(c1 - 1.1) < 1e-9, f"got {c1}")
    check("TC-A-045b", "compute_C([1.1, 1.05], 1.5) == 1.155",
          abs(c2 - 1.155) < 1e-9, f"got {c2}")
    check("TC-A-045c", "compute_C([1.5], 1.5) == 1.5",
          abs(c3 - 1.5) < 1e-9, f"got {c3}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Track A Calculator -- Sections 3-5 test suite")
    test_section_3()
    test_section_4()
    test_section_5()
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed:
        sys.exit(1)
    else:
        print("All tests passed.")
