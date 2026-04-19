"""Track A Calculator tests -- Sections 6-8: Age decay, Zones, Confidence.

Covers TC-A-050..054 (age decay), TC-A-060..065 (zones),
TC-A-070..076 (confidence).
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.calculator import (
    calculate_health_index,
    compute_A,
    _determine_zone,
    _determine_confidence_zone,
    _compute_confidence,
)
from data_io.models import (
    Factor,
    ConfidenceFactors,
    WeightsProfile,
    SeverityLevel,
    HealthZone,
    ConfidenceZone,
    ZoneThresholds,
    ConfidenceConfig,
    ConfidencePenalties,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0


def _check(tc_id: str, condition: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tc_id}: {status}{suffix}")
    if condition:
        _pass_count += 1
    else:
        _fail_count += 1


def _approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


def _make_weights(**zone_kw) -> WeightsProfile:
    """Return a WeightsProfile with default zone thresholds (75 / 40)."""
    return WeightsProfile(
        profile_name="test",
        zones=ZoneThresholds(
            green_threshold=zone_kw.get("green", 75),
            red_threshold=zone_kw.get("red", 40),
        ),
    )


# ── Section 6: Age decay (compute_A) ─────────────────────────────────────────

def test_section_6_age_decay() -> None:
    print("\n=== Section 6: Age decay (compute_A) ===")

    # TC-A-050: fresh event (age_days=0) -> 1.0
    result = compute_A(0, 14)
    _check("TC-A-050", _approx(result, 1.0),
           f"compute_A(0, 14) = {result}, expected 1.0")

    # TC-A-051: 30-day-old event, tau=14 -> exp(-30/14) ~ 0.1173
    expected = math.exp(-30 / 14)
    result = compute_A(30, 14)
    _check("TC-A-051", _approx(result, expected),
           f"compute_A(30, 14) = {result:.6f}, expected {expected:.6f}")

    # TC-A-052: 365-day-old event -> near zero
    expected = math.exp(-365 / 14)
    result = compute_A(365, 14)
    _check("TC-A-052", _approx(result, expected, tol=1e-10) and result < 1e-10,
           f"compute_A(365, 14) = {result:.2e}, expected ~0")

    # TC-A-053: future event (age_days=-1) -> clamped to 1.0
    result = compute_A(-1, 14)
    _check("TC-A-053", _approx(result, 1.0),
           f"compute_A(-1, 14) = {result}, expected 1.0")

    # TC-A-054: boundary at tau (age_days == tau_days) -> exp(-1) ~ 0.3679
    expected = math.exp(-1)
    result = compute_A(14, 14)
    _check("TC-A-054", _approx(result, expected),
           f"compute_A(14, 14) = {result:.6f}, expected {expected:.6f}")


# ── Section 7: Zones (_determine_zone) ───────────────────────────────────────

def test_section_7_zones() -> None:
    print("\n=== Section 7: Zones (_determine_zone) ===")
    w = _make_weights()

    # TC-A-060: H=75 -> green; H=74 -> yellow
    _check("TC-A-060a", _determine_zone(75, w) == HealthZone.GREEN,
           f"H=75 -> {_determine_zone(75, w)}, expected GREEN")
    _check("TC-A-060b", _determine_zone(74, w) == HealthZone.YELLOW,
           f"H=74 -> {_determine_zone(74, w)}, expected YELLOW")

    # TC-A-061: H=50 -> yellow
    _check("TC-A-061", _determine_zone(50, w) == HealthZone.YELLOW,
           f"H=50 -> {_determine_zone(50, w)}, expected YELLOW")

    # TC-A-062: H=39 -> red; H=40 -> yellow
    _check("TC-A-062a", _determine_zone(39, w) == HealthZone.RED,
           f"H=39 -> {_determine_zone(39, w)}, expected RED")
    _check("TC-A-062b", _determine_zone(40, w) == HealthZone.YELLOW,
           f"H=40 -> {_determine_zone(40, w)}, expected YELLOW")

    # TC-A-063: boundary values explicitly
    _check("TC-A-063a", _determine_zone(40, w) == HealthZone.YELLOW,
           f"H=40 boundary -> {_determine_zone(40, w)}, expected YELLOW")
    _check("TC-A-063b", _determine_zone(75, w) == HealthZone.GREEN,
           f"H=75 boundary -> {_determine_zone(75, w)}, expected GREEN")

    # TC-A-064: H=1 -> red
    _check("TC-A-064", _determine_zone(1, w) == HealthZone.RED,
           f"H=1 -> {_determine_zone(1, w)}, expected RED")

    # TC-A-065: H=100 -> green
    _check("TC-A-065", _determine_zone(100, w) == HealthZone.GREEN,
           f"H=100 -> {_determine_zone(100, w)}, expected GREEN")


# ── Section 8: Confidence (_compute_confidence, _determine_confidence_zone) ──

def test_section_8_confidence() -> None:
    print("\n=== Section 8: Confidence ===")
    w = _make_weights()

    # TC-A-070: all flags false -> confidence=1.0, no reasons
    cf = ConfidenceFactors()
    score, reasons = _compute_confidence(cf, w)
    _check("TC-A-070", _approx(score, 1.0) and len(reasons) == 0,
           f"score={score}, reasons={reasons}")

    # TC-A-071: rag_missing_count=1 -> 0.7
    cf = ConfidenceFactors(rag_missing_count=1)
    score, reasons = _compute_confidence(cf, w)
    _check("TC-A-071", _approx(score, 0.7),
           f"score={score}, expected 0.7")

    # TC-A-072: rag_missing_count=2 -> 0.7*0.7 = 0.49
    cf = ConfidenceFactors(rag_missing_count=2)
    score, reasons = _compute_confidence(cf, w)
    _check("TC-A-072", _approx(score, 0.49),
           f"score={score}, expected 0.49")

    # TC-A-073: missing_resources=True -> 0.85
    cf = ConfidenceFactors(missing_resources=True)
    score, reasons = _compute_confidence(cf, w)
    _check("TC-A-073", _approx(score, 0.85),
           f"score={score}, expected 0.85")

    # TC-A-074: all flags true + rag_missing_count=5 -> clamped to 0.2
    cf = ConfidenceFactors(
        rag_missing_count=5,
        missing_resources=True,
        missing_model=True,
        abnormal_daily_jump=True,
        anomalous_event_count=True,
        no_events_and_no_resources=True,
    )
    score, reasons = _compute_confidence(cf, w)
    raw = (0.7 ** 5) * 0.85 * 0.6 * 0.8 * 0.7 * 0.9
    _check("TC-A-074", _approx(score, 0.2) and raw < 0.2,
           f"score={score}, expected 0.2 (raw={raw:.6f} < 0.2, clamped)")

    # TC-A-075: confidence zone boundaries
    _check("TC-A-075a", _determine_confidence_zone(0.85) == ConfidenceZone.HIGH,
           f"0.85 -> {_determine_confidence_zone(0.85)}, expected HIGH")
    _check("TC-A-075b", _determine_confidence_zone(0.60) == ConfidenceZone.MEDIUM,
           f"0.60 -> {_determine_confidence_zone(0.60)}, expected MEDIUM")
    _check("TC-A-075c", _determine_confidence_zone(0.59) == ConfidenceZone.LOW,
           f"0.59 -> {_determine_confidence_zone(0.59)}, expected LOW")

    # TC-A-076: flag_for_review -- NOT_APPLICABLE
    print("  TC-A-076: NOT_APPLICABLE  (flag_for_review is agent-level logic "
          "in Track B, not part of calculate_health_index in Track A calculator)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_section_6_age_decay()
    test_section_7_zones()
    test_section_8_confidence()

    total = _pass_count + _fail_count
    print(f"\n{'=' * 50}")
    print(f"Total: {total}  |  PASS: {_pass_count}  |  FAIL: {_fail_count}")
    if _fail_count == 0:
        print("All tests passed.")
    else:
        print(f"*** {_fail_count} test(s) FAILED ***")
        sys.exit(1)
