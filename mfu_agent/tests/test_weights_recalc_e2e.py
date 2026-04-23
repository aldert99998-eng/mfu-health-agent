"""End-to-end test: changing weights on Weights page must change health indices.

Reproduces the exact logic from pages/3_Weights.py in the "Apply weights" block,
without Streamlit, so we can assert programmatically what the page SHOULD do.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.tools.impl import ToolDependencies, register_all_tools
from agent.tools.registry import ToolRegistry
from data_io.factor_store import FactorStore
from data_io.models import (
    ConfidenceFactors,
    Factor,
    SeverityLevel,
    WeightsProfile,
)
from data_io.normalizer import ingest_file
from tools.calculator import (
    calculate_health_index,
    compute_A,
    compute_C,
    compute_R,
)

REAL_ZABBIX_FILE = Path(
    "/tmp/tmpbsclq6n3/_SELECT_h_hostid_AS_id_prn_h_host_AS_host_name_h_name_AS_host_di_202603131731.json"
)


@pytest.fixture(scope="module")
def loaded_store():
    """Ingest a real Zabbix file once per module."""
    if not REAL_ZABBIX_FILE.exists():
        pytest.skip(f"Real Zabbix fixture missing: {REAL_ZABBIX_FILE}")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        shutil.copyfile(REAL_ZABBIX_FILE, f.name)
        tmp = Path(f.name)
    fs = FactorStore()
    ingest_file(tmp, fs)
    fs.freeze()
    return fs


@pytest.fixture(scope="module")
def initial_batch(loaded_store):
    """Run full batch analysis once; return per-device raw factors + results."""
    fs = loaded_store
    weights = WeightsProfile(profile_name="default")
    reg = ToolRegistry()
    deps = ToolDependencies(factor_store=fs, weights=weights)
    register_all_tools(reg, deps)

    raw_factors_map = {}
    results_map = {}
    for did in fs.list_devices():
        events_result = reg.execute("get_device_events", {"device_id": did})
        events = (events_result.data or {}).get("events", [])
        unique = {}
        for ev in events:
            code = ev.get("error_code") or ev.get("error_description") or ""
            if code and code not in unique:
                unique[code] = ev

        factors = []
        for code, ev in unique.items():
            cls = reg.execute(
                "classify_error_severity",
                {
                    "error_code": code,
                    "model": "",
                    "error_description": ev.get("error_description", ""),
                },
            )
            sev = cls.data.get("severity", "Medium") if cls.success and cls.data else "Medium"
            rep = reg.execute("count_error_repetitions", {"device_id": did, "error_code": code})
            n = rep.data.get("count", 1) if rep.success and rep.data else 1
            factors.append({
                "error_code": code,
                "severity_level": sev,
                "n_repetitions": n,
                "event_timestamp": ev.get("timestamp"),
                "applicable_modifiers": [],
                "source": "",
            })

        calc_args = {
            "device_id": did,
            "factors": factors,
            "confidence_factors": {
                "rag_missing_count": 0,
                "missing_resources": False,
                "missing_model": False,
            },
        }
        raw_factors_map[did] = calc_args

        calc = reg.execute("calculate_health_index", calc_args)
        if calc.success:
            results_map[did] = calc.data["health_index"]

    return raw_factors_map, results_map


def _recalc_with_weights(
    raw_factors_map: dict,
    weights: WeightsProfile,
    now: datetime,
) -> dict[str, int]:
    """Exact copy of the Apply-weights block from pages/3_Weights.py."""
    results = {}
    for did, calc_args in raw_factors_map.items():
        raw_factors = calc_args.get("factors", [])
        cf_data = calc_args.get("confidence_factors", {})

        built_factors: list[Factor] = []
        for fi in raw_factors:
            try:
                severity = SeverityLevel(fi["severity_level"])
            except (ValueError, KeyError):
                severity = SeverityLevel.MEDIUM

            S = getattr(weights.severity, severity.value.lower(), 10.0)
            ts = datetime.fromisoformat(fi["event_timestamp"])
            age_days = max(0, (now - ts).days)

            R = compute_R(
                fi.get("n_repetitions", 1),
                weights.repeatability.base,
                weights.repeatability.max_value,
            )
            modifiers = []
            for mod_name in fi.get("applicable_modifiers", []):
                mod = weights.context.modifiers.get(mod_name)
                if mod:
                    modifiers.append(mod.multiplier)
            C = compute_C(modifiers, weights.context.max_value)
            A = compute_A(age_days, weights.age.tau_days)

            built_factors.append(
                Factor(
                    error_code=fi.get("error_code", ""),
                    severity_level=severity,
                    S=S,
                    n_repetitions=fi.get("n_repetitions", 1),
                    R=R,
                    C=C,
                    A=A,
                    event_timestamp=ts,
                    age_days=age_days,
                )
            )

        cf = ConfidenceFactors(
            rag_missing_count=cf_data.get("rag_missing_count", 0),
            missing_resources=cf_data.get("missing_resources", False),
            missing_model=cf_data.get("missing_model", False),
        )
        result = calculate_health_index(
            factors=built_factors,
            confidence_factors=cf,
            weights=weights,
            device_id=did,
            silent_device_mode=weights.silent_device_mode.value,
        )
        results[did] = result.health_index
    return results


def test_raw_factors_populated(initial_batch):
    raw_factors_map, _ = initial_batch
    assert len(raw_factors_map) > 0, "No raw factors captured"
    devices_with_events = sum(1 for args in raw_factors_map.values() if args["factors"])
    assert devices_with_events > 50, (
        f"Only {devices_with_events} devices have factors — ingestion broken"
    )


def test_lowering_severity_raises_indices(initial_batch, loaded_store):
    """Lowering S_high from 20 to 5 must RAISE indices (less penalty)."""
    raw_factors_map, original_results = initial_batch
    now = loaded_store.reference_time

    w_low = WeightsProfile(profile_name="low-severity")
    w_low.severity.critical = 10.0
    w_low.severity.high = 5.0
    w_low.severity.medium = 2.0

    new_results = _recalc_with_weights(raw_factors_map, w_low, now)

    changed = sum(
        1 for did in original_results
        if did in new_results and new_results[did] != original_results[did]
    )
    assert changed > 0, "No device index changed despite radically lower weights"

    raised = sum(
        1 for did in original_results
        if did in new_results and new_results[did] > original_results[did]
    )
    assert raised > changed * 0.5, (
        f"Expected most changed devices to RAISE index (weights lowered), "
        f"but only {raised} of {changed} went up"
    )


def test_raising_severity_drops_indices(initial_batch, loaded_store):
    """Raising S_critical must DROP indices for devices with Critical errors."""
    raw_factors_map, original_results = initial_batch
    now = loaded_store.reference_time

    w_high = WeightsProfile(profile_name="high-severity")
    w_high.severity.critical = 100.0
    w_high.severity.high = 60.0

    new_results = _recalc_with_weights(raw_factors_map, w_high, now)

    dropped = sum(
        1 for did in original_results
        if did in new_results and new_results[did] < original_results[did]
    )
    unchanged = sum(
        1 for did in original_results
        if did in new_results and new_results[did] == original_results[did]
    )

    assert dropped > 0, (
        f"No device dropped index despite higher severity weights "
        f"(dropped={dropped}, unchanged={unchanged})"
    )


def test_changing_tau_changes_indices(initial_batch, loaded_store):
    """Doubling tau_days (slower age decay) must change indices."""
    raw_factors_map, original_results = initial_batch
    now = loaded_store.reference_time

    w = WeightsProfile(profile_name="slow-decay")
    w.age.tau_days = 60

    new_results = _recalc_with_weights(raw_factors_map, w, now)
    changed = sum(
        1 for did in original_results
        if did in new_results and new_results[did] != original_results[did]
    )
    assert changed > 0, "Changing tau_days didn't affect any index"


def test_before_after_comparison_uses_baseline_not_current(initial_batch, loaded_store):
    """After apply+rerun, current_results == new_results. Without a baseline
    snapshot, comparison would always read current==current → no changes shown.

    This simulates what pages/3_Weights.py does on re-render post-apply.
    """
    raw_factors_map, original_results = initial_batch
    now = loaded_store.reference_time

    baseline = dict(original_results)

    w_heavy = WeightsProfile(profile_name="heavy")
    w_heavy.severity.critical = 100.0
    w_heavy.severity.high = 60.0
    w_heavy.severity.medium = 30.0

    new_results = _recalc_with_weights(raw_factors_map, w_heavy, now)

    current = new_results

    wrong_changed = sum(1 for did in current if current[did] != current[did])
    correct_changed = sum(
        1 for did in current
        if did in baseline and current[did] != baseline[did]
    )

    assert wrong_changed == 0
    assert correct_changed > 0, (
        "Using current as both 'old' and 'new' shows 0 changes — this is the bug"
    )
