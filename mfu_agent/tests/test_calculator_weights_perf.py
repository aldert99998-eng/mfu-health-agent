"""Track A Calculator tests -- sections 12-15.

TC-A-110..116  WeightsManager
TC-A-120..122  Performance
TC-A-130..131  Regression dataset
TC-A-140..142  Config errors
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

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
from config.weights_manager import WeightsManager
from config.loader import ConfigValidationError

# ── Paths ────────────────────────────────────────────────────────────────────

_MFU_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_YAML = _MFU_ROOT / "configs" / "weights" / "default.yaml"

# ── Helpers ──────────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0


def _report(tc_id: str, passed: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    status = "PASS" if passed else "FAIL"
    if not passed:
        _fail_count += 1
    else:
        _pass_count += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"[{status}] {tc_id}{suffix}")


def make_factor(
    S: float = 3.0,
    R: float = 1.0,
    C: float = 1.0,
    A: float = 1.0,
    severity: SeverityLevel = SeverityLevel.LOW,
) -> Factor:
    return Factor(
        error_code="E001",
        severity_level=severity,
        S=S,
        n_repetitions=1,
        R=R,
        C=C,
        A=A,
        event_timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        age_days=0,
    )


def _default_weights() -> WeightsProfile:
    return WeightsProfile(profile_name="test_default")


def _default_confidence() -> ConfidenceFactors:
    return ConfidenceFactors()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12 -- WeightsManager
# ═══════════════════════════════════════════════════════════════════════════════


def test_tc_a_110_load_default_yaml() -> None:
    """Load default.yaml via reset_to_default, verify profile_name and severity.critical."""
    tc = "TC-A-110"
    try:
        wm = WeightsManager(default_path=_DEFAULT_YAML)
        profile = wm.reset_to_default()
        ok = profile.profile_name == "default" and profile.severity.critical == 60
        _report(tc, ok, f"name={profile.profile_name!r}, critical={profile.severity.critical}")
    except Exception as exc:
        _report(tc, False, str(exc))


def test_tc_a_111_save_and_load_named_profile() -> None:
    """Save a custom profile to tempdir, load it back, verify values match."""
    tc = "TC-A-111"
    tmpdir = tempfile.mkdtemp(prefix="wm_test_")
    try:
        profiles_dir = Path(tmpdir) / "profiles"
        history_path = Path(tmpdir) / "history.log"
        wm = WeightsManager(
            profiles_dir=profiles_dir,
            history_path=history_path,
            default_path=_DEFAULT_YAML,
        )
        custom = WeightsProfile(
            profile_name="custom_one",
            severity=SeverityWeights(critical=80, high=30, medium=15, low=5, info=1),
        )
        wm.save_profile(custom, author="test_runner")
        loaded = wm.load_profile("custom_one")
        ok = (
            loaded.profile_name == "custom_one"
            and loaded.severity.critical == 80
            and loaded.severity.high == 30
            and loaded.severity.info == 1
        )
        _report(tc, ok, f"critical={loaded.severity.critical}")
    except Exception as exc:
        _report(tc, False, str(exc))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_tc_a_112_invalid_yaml_raises() -> None:
    """Write invalid YAML content, load_profile must raise ConfigValidationError."""
    tc = "TC-A-112"
    tmpdir = tempfile.mkdtemp(prefix="wm_test_")
    try:
        profiles_dir = Path(tmpdir) / "profiles"
        profiles_dir.mkdir(parents=True)
        bad_file = profiles_dir / "broken.yaml"
        bad_file.write_text("{{{{", encoding="utf-8")
        wm = WeightsManager(
            profiles_dir=profiles_dir,
            history_path=Path(tmpdir) / "history.log",
            default_path=_DEFAULT_YAML,
        )
        try:
            wm.load_profile("broken")
            _report(tc, False, "No exception raised")
        except ConfigValidationError:
            _report(tc, True)
        except Exception as exc:
            _report(tc, False, f"Wrong exception type: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_tc_a_113_pydantic_validation_negative_weight() -> None:
    """SeverityWeights rejects negative values (ge=0 constraint)."""
    tc = "TC-A-113"
    try:
        WeightsProfile(
            profile_name="bad",
            severity=SeverityWeights(critical=-1),
        )
        _report(tc, False, "No ValidationError raised")
    except ValidationError:
        _report(tc, True)
    except Exception as exc:
        _report(tc, False, f"Wrong exception type: {type(exc).__name__}")


def test_tc_a_114_different_weights_different_h() -> None:
    """Same factors but different weight profiles produce different H values."""
    tc = "TC-A-114"
    try:
        factors = [make_factor(S=10.0, severity=SeverityLevel.MEDIUM)]
        conf = _default_confidence()

        w1 = WeightsProfile(profile_name="w1")
        w2 = WeightsProfile(
            profile_name="w2",
            severity=SeverityWeights(critical=100, high=50, medium=25, low=8, info=2),
        )

        # Recalculate S based on severity weight from the profile
        f1 = [make_factor(S=w1.severity.medium)]
        f2 = [make_factor(S=w2.severity.medium)]

        r1 = calculate_health_index(f1, conf, w1, device_id="d1")
        r2 = calculate_health_index(f2, conf, w2, device_id="d2")

        ok = r1.health_index != r2.health_index
        _report(tc, ok, f"H1={r1.health_index}, H2={r2.health_index}")
    except Exception as exc:
        _report(tc, False, str(exc))


def test_tc_a_115_save_profile_creates_file() -> None:
    """save_profile returns a path and the file exists on disk."""
    tc = "TC-A-115"
    tmpdir = tempfile.mkdtemp(prefix="wm_test_")
    try:
        profiles_dir = Path(tmpdir) / "profiles"
        wm = WeightsManager(
            profiles_dir=profiles_dir,
            history_path=Path(tmpdir) / "history.log",
            default_path=_DEFAULT_YAML,
        )
        profile = WeightsProfile(profile_name="saved_one")
        result_path = wm.save_profile(profile, author="tester")
        ok = result_path.exists() and result_path.name == "saved_one.yaml"
        _report(tc, ok, f"path={result_path}")
    except Exception as exc:
        _report(tc, False, str(exc))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_tc_a_116_missing_profile_raises() -> None:
    """load_profile for a nonexistent name raises ConfigValidationError."""
    tc = "TC-A-116"
    tmpdir = tempfile.mkdtemp(prefix="wm_test_")
    try:
        profiles_dir = Path(tmpdir) / "profiles"
        profiles_dir.mkdir(parents=True)
        wm = WeightsManager(
            profiles_dir=profiles_dir,
            history_path=Path(tmpdir) / "history.log",
            default_path=_DEFAULT_YAML,
        )
        try:
            wm.load_profile("does_not_exist")
            _report(tc, False, "No exception raised")
        except ConfigValidationError:
            _report(tc, True)
        except Exception as exc:
            _report(tc, False, f"Wrong exception: {type(exc).__name__}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 13 -- Performance
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_factors(count: int) -> list[Factor]:
    """Generate a list of varied factors for performance tests."""
    severities = list(SeverityLevel)
    factors = []
    for i in range(count):
        sev = severities[i % len(severities)]
        s_map = {
            SeverityLevel.CRITICAL: 60,
            SeverityLevel.HIGH: 20,
            SeverityLevel.MEDIUM: 10,
            SeverityLevel.LOW: 3,
            SeverityLevel.INFO: 0,
        }
        factors.append(
            Factor(
                error_code=f"E{i:04d}",
                severity_level=sev,
                S=float(s_map[sev]),
                n_repetitions=1 + (i % 3),
                R=min(5.0, 1.0 + (i % 5) * 0.5),
                C=min(1.5, 1.0 + (i % 4) * 0.1),
                A=max(0.0, 1.0 - (i % 10) * 0.1),
                event_timestamp=datetime(2025, 1, 1 + (i % 28), tzinfo=UTC),
                age_days=i % 30,
            )
        )
    return factors


def test_tc_a_120_perf_100_devices_10_factors() -> None:
    """100 devices x 10 factors each must complete in < 100 ms."""
    tc = "TC-A-120"
    try:
        w = _default_weights()
        conf = _default_confidence()
        device_factors = [_generate_factors(10) for _ in range(100)]

        start = time.perf_counter()
        for i, factors in enumerate(device_factors):
            calculate_health_index(factors, conf, w, device_id=f"dev_{i:04d}")
        elapsed_ms = (time.perf_counter() - start) * 1000

        ok = elapsed_ms < 100
        _report(tc, ok, f"{elapsed_ms:.1f} ms (limit 100 ms)")
    except Exception as exc:
        _report(tc, False, str(exc))


def test_tc_a_121_perf_10000_devices_5_factors() -> None:
    """10 000 devices x 5 factors each must complete in < 10 s."""
    tc = "TC-A-121"
    try:
        w = _default_weights()
        conf = _default_confidence()
        batch = _generate_factors(5)

        start = time.perf_counter()
        for i in range(10_000):
            calculate_health_index(batch, conf, w, device_id=f"dev_{i:05d}")
        elapsed_s = time.perf_counter() - start

        ok = elapsed_s < 10
        _report(tc, ok, f"{elapsed_s:.2f} s (limit 10 s)")
    except Exception as exc:
        _report(tc, False, str(exc))


def test_tc_a_122_linearity_check() -> None:
    """Time with 10, 100, 1000 factors. Ratio should be roughly linear, not quadratic."""
    tc = "TC-A-122"
    try:
        w = _default_weights()
        conf = _default_confidence()
        times: dict[int, float] = {}
        for n in (10, 100, 1000):
            factors = _generate_factors(n)
            start = time.perf_counter()
            for _ in range(50):
                calculate_health_index(factors, conf, w, device_id="linearity")
            times[n] = time.perf_counter() - start

        # If quadratic, ratio_1000_100 would be ~100; linear would be ~10
        ratio_1000_100 = times[1000] / times[100] if times[100] > 0 else float("inf")
        ratio_100_10 = times[100] / times[10] if times[10] > 0 else float("inf")

        # Allow generous margin: ratio should be < 30 (well under quadratic ~100)
        ok = ratio_1000_100 < 30 and ratio_100_10 < 30
        _report(
            tc,
            ok,
            f"100/10={ratio_100_10:.1f}x, 1000/100={ratio_1000_100:.1f}x",
        )
    except Exception as exc:
        _report(tc, False, str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 14 -- Regression dataset
# ═══════════════════════════════════════════════════════════════════════════════


def _build_regression_devices() -> list[tuple[str, list[Factor]]]:
    """Create 20 devices with deterministic known inputs."""
    devices = []
    for i in range(20):
        n_factors = 1 + (i % 5)
        factors = []
        for j in range(n_factors):
            sev_idx = (i + j) % 5
            sev = list(SeverityLevel)[sev_idx]
            s_map = {
                SeverityLevel.CRITICAL: 60.0,
                SeverityLevel.HIGH: 20.0,
                SeverityLevel.MEDIUM: 10.0,
                SeverityLevel.LOW: 3.0,
                SeverityLevel.INFO: 0.0,
            }
            factors.append(
                Factor(
                    error_code=f"R{i:02d}{j:02d}",
                    severity_level=sev,
                    S=s_map[sev],
                    n_repetitions=1,
                    R=1.0,
                    C=1.0,
                    A=1.0,
                    event_timestamp=datetime(2025, 1, 15, tzinfo=UTC),
                    age_days=0,
                )
            )
        devices.append((f"reg_dev_{i:03d}", factors))
    return devices


def test_tc_a_130_regression_20_devices() -> None:
    """Calculate H for 20 known devices, store expected, verify all match on re-run."""
    tc = "TC-A-130"
    try:
        w = _default_weights()
        conf = _default_confidence()
        devices = _build_regression_devices()

        # First pass: compute expected values
        expected: dict[str, int] = {}
        for dev_id, factors in devices:
            result = calculate_health_index(factors, conf, w, device_id=dev_id)
            expected[dev_id] = result.health_index

        # Second pass: verify determinism
        all_match = True
        mismatches = []
        for dev_id, factors in devices:
            result = calculate_health_index(factors, conf, w, device_id=dev_id)
            if result.health_index != expected[dev_id]:
                all_match = False
                mismatches.append(
                    f"{dev_id}: got {result.health_index}, expected {expected[dev_id]}"
                )

        _report(tc, all_match, "; ".join(mismatches) if mismatches else "all 20 match")
    except Exception as exc:
        _report(tc, False, str(exc))


def test_tc_a_131_regression_with_default_profile() -> None:
    """Run regression with default WeightsProfile, store reference values."""
    tc = "TC-A-131"
    try:
        w = WeightsProfile(profile_name="default")
        conf = _default_confidence()
        devices = _build_regression_devices()

        reference: list[tuple[str, int]] = []
        for dev_id, factors in devices:
            result = calculate_health_index(factors, conf, w, device_id=dev_id)
            reference.append((dev_id, result.health_index))

        # Verify reference values are plausible (1..100) and consistent
        plausible = all(1 <= h <= 100 for _, h in reference)
        # Re-run once to confirm determinism
        second: list[int] = []
        for dev_id, factors in devices:
            result = calculate_health_index(factors, conf, w, device_id=dev_id)
            second.append(result.health_index)

        deterministic = all(
            second[i] == reference[i][1] for i in range(len(reference))
        )

        ok = plausible and deterministic
        sample = ", ".join(f"{d}={h}" for d, h in reference[:5])
        _report(tc, ok, f"sample: {sample}...")
    except Exception as exc:
        _report(tc, False, str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 15 -- Config errors
# ═══════════════════════════════════════════════════════════════════════════════


def test_tc_a_140_inverted_weights_still_calculates() -> None:
    """critical < low weight -- should still calculate (just unusual policy)."""
    tc = "TC-A-140"
    try:
        w = WeightsProfile(
            profile_name="inverted",
            severity=SeverityWeights(critical=1, high=2, medium=5, low=50, info=0),
        )
        factors = [make_factor(S=w.severity.low, severity=SeverityLevel.LOW)]
        conf = _default_confidence()
        result = calculate_health_index(factors, conf, w, device_id="inv")
        ok = 1 <= result.health_index <= 100
        _report(tc, ok, f"H={result.health_index} (low weight=50, critical weight=1)")
    except Exception as exc:
        _report(tc, False, str(exc))


def test_tc_a_141_negative_weight_rejected() -> None:
    """Pydantic ge=0 constraint rejects negative severity weight."""
    tc = "TC-A-141"
    try:
        SeverityWeights(critical=-5, high=20, medium=10, low=3, info=0)
        _report(tc, False, "No ValidationError raised")
    except ValidationError:
        _report(tc, True)
    except Exception as exc:
        _report(tc, False, f"Wrong exception: {type(exc).__name__}")


def test_tc_a_142_very_large_weights_clamp() -> None:
    """Very large S values -- formula still works, H clamps to 1."""
    tc = "TC-A-142"
    try:
        w = WeightsProfile(
            profile_name="huge",
            severity=SeverityWeights(critical=99999, high=50000, medium=10000, low=5000, info=1000),
        )
        factors = [
            make_factor(S=99999.0, severity=SeverityLevel.CRITICAL),
            make_factor(S=50000.0, severity=SeverityLevel.HIGH),
        ]
        conf = _default_confidence()
        result = calculate_health_index(factors, conf, w, device_id="huge")
        ok = result.health_index == 1
        _report(tc, ok, f"H={result.health_index} (expected 1)")
    except Exception as exc:
        _report(tc, False, str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    # Section 12
    test_tc_a_110_load_default_yaml,
    test_tc_a_111_save_and_load_named_profile,
    test_tc_a_112_invalid_yaml_raises,
    test_tc_a_113_pydantic_validation_negative_weight,
    test_tc_a_114_different_weights_different_h,
    test_tc_a_115_save_profile_creates_file,
    test_tc_a_116_missing_profile_raises,
    # Section 13
    test_tc_a_120_perf_100_devices_10_factors,
    test_tc_a_121_perf_10000_devices_5_factors,
    test_tc_a_122_linearity_check,
    # Section 14
    test_tc_a_130_regression_20_devices,
    test_tc_a_131_regression_with_default_profile,
    # Section 15
    test_tc_a_140_inverted_weights_still_calculates,
    test_tc_a_141_negative_weight_rejected,
    test_tc_a_142_very_large_weights_clamp,
]


def main() -> None:
    print("=" * 70)
    print("Track A Calculator -- Sections 12-15")
    print("=" * 70)
    print()

    for test_fn in ALL_TESTS:
        test_fn()

    print()
    print("-" * 70)
    print(f"Total: {_pass_count + _fail_count}  |  PASS: {_pass_count}  |  FAIL: {_fail_count}")
    print("-" * 70)

    if _fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
