"""Tests for reporting.report_builder — Track C, Phase 6.1."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from config.loader import ReportConfig
from data_io.factor_store import DeviceMetadata, FactorStore
from data_io.models import (
    AgentMode,
    CalculationSnapshot,
    ConfidenceZone,
    FactorContribution,
    FileFormat,
    HealthResult,
    HealthZone,
    PatternType,
    ResourceSnapshot,
    SourceFileInfo,
    Trace,
)
from reporting.report_builder import ReportBuilder

NOW = datetime.now(UTC)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_result(
    device_id: str,
    health_index: int,
    confidence: float = 0.85,
    zone: str | None = None,
    model: str | None = None,
    location: str | None = None,
    error_label: str = "C6000 (Critical)",
) -> HealthResult:
    if zone is None:
        if health_index >= 75:
            zone = "green"
        elif health_index >= 40:
            zone = "yellow"
        else:
            zone = "red"
    cz = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.6 else "low")
    return HealthResult(
        device_id=device_id,
        health_index=health_index,
        confidence=confidence,
        zone=HealthZone(zone),
        confidence_zone=ConfidenceZone(cz),
        factor_contributions=[
            FactorContribution(
                label=error_label,
                penalty=10.0,
                S=60,
                R=1.0,
                C=1.0,
                A=1.0,
                source="test",
            ),
        ],
        calculated_at=NOW,
    )


def _make_factor_store(
    results: list[HealthResult],
    models: dict[str, str] | None = None,
    locations: dict[str, str] | None = None,
) -> FactorStore:
    store = FactorStore()
    models = models or {}
    locations = locations or {}
    for r in results:
        store.set_device_metadata(
            r.device_id,
            DeviceMetadata(
                device_id=r.device_id,
                model=models.get(r.device_id),
                location=locations.get(r.device_id),
            ),
        )
    store.freeze()
    return store


def _make_snapshot() -> CalculationSnapshot:
    return CalculationSnapshot(
        weights_profile_name="default",
        weights_profile_version="1.0",
        weights_data={"severity": {"critical": 60}, "age": {"window_days": 30}},
    )


def _make_source_info() -> SourceFileInfo:
    return SourceFileInfo(
        file_name="test.csv",
        file_hash="sha256:abc",
        file_format=FileFormat.CSV,
        uploaded_at=NOW,
    )


_DEFAULT_SUMMARY = (
    "Парк из 10 устройств в хорошем состоянии. Средний индекс здоровья "
    "составляет 75. Рекомендуется плановое обслуживание."
)


def _make_builder() -> ReportBuilder:
    agent = MagicMock()
    llm_response = MagicMock()
    llm_response.content = _DEFAULT_SUMMARY
    agent._llm.generate.return_value = llm_response
    config = ReportConfig()
    return ReportBuilder(agent=agent, config=config)


# ── TC-C-1: FleetSummary correctness ────────────────────────────────────────


class TestFleetSummary:
    def test_zone_counts_and_average(self) -> None:
        results = [
            _make_result("D001", 20),
            _make_result("D002", 55),
            _make_result("D003", 90),
        ]
        builder = _make_builder()
        store = _make_factor_store(results)
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.fleet_summary.zone_counts == {"green": 1, "yellow": 1, "red": 1}
        assert report.fleet_summary.average_index == 55.0
        assert report.fleet_summary.total_devices == 3

    def test_empty_results(self) -> None:
        builder = _make_builder()
        store = FactorStore()
        report = builder.build([], store, _make_snapshot(), _make_source_info())

        assert report.fleet_summary.total_devices == 0
        assert report.fleet_summary.average_index == 0.0
        assert report.fleet_summary.zone_counts == {"green": 0, "yellow": 0, "red": 0}

    def test_median_calculation(self) -> None:
        results = [
            _make_result("D001", 10),
            _make_result("D002", 50),
            _make_result("D003", 90),
            _make_result("D004", 95),
        ]
        builder = _make_builder()
        store = _make_factor_store(results)
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.fleet_summary.median_index == 70.0


# ── TC-C-2: Mass issue detection ────────────────────────────────────────────


class TestPatternDetection:
    def test_mass_issue_detected(self) -> None:
        results = [
            _make_result(f"D{i:03d}", 35, error_label="C6000")
            for i in range(4)
        ]
        models = {f"D{i:03d}": "Kyocera 3253ci" for i in range(4)}
        store = _make_factor_store(results, models=models)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        mass_patterns = [
            p for p in report.top_patterns
            if p.pattern_type == PatternType.MASS_ISSUE
        ]
        assert len(mass_patterns) >= 1
        assert len(mass_patterns[0].affected_device_ids) == 4

    def test_location_cluster_detected(self) -> None:
        results = [
            _make_result(f"D{i:03d}", 30 + i, error_label=f"E{i:04d}")
            for i in range(5)
        ]
        locations = {f"D{i:03d}": "Офис Ленинградка" for i in range(5)}
        store = _make_factor_store(results, locations=locations)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        loc_patterns = [
            p for p in report.top_patterns
            if p.pattern_type == PatternType.LOCATION_CLUSTER
        ]
        assert len(loc_patterns) >= 1

    def test_critical_single_detected(self) -> None:
        results = [
            _make_result("D001", 15, confidence=0.9),
            _make_result("D002", 80, confidence=0.9),
        ]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        critical = [
            p for p in report.top_patterns
            if p.pattern_type == PatternType.CRITICAL_SINGLE
        ]
        assert len(critical) >= 1
        assert "D001" in critical[0].affected_device_ids

    def test_patterns_limited_to_5(self) -> None:
        results = []
        models = {}
        for i in range(20):
            did = f"D{i:03d}"
            results.extend([
                _make_result(f"{did}_a", 30, error_label=f"ERR_{i}"),
                _make_result(f"{did}_b", 30, error_label=f"ERR_{i}"),
                _make_result(f"{did}_c", 30, error_label=f"ERR_{i}"),
            ])
            for suffix in ("_a", "_b", "_c"):
                models[f"{did}{suffix}"] = f"Model_{i}"

        store = _make_factor_store(results, models=models)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert len(report.top_patterns) <= 5

    def test_single_device_no_patterns(self) -> None:
        results = [_make_result("D001", 50)]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.top_patterns == []


# ── TC-C-4: Distribution ────────────────────────────────────────────────────


class TestDistribution:
    def test_10_bins(self) -> None:
        results = [_make_result(f"D{i:03d}", i * 10 + 5) for i in range(10)]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert len(report.index_distribution) == 10
        assert report.index_distribution[0].range_start == 0
        assert report.index_distribution[9].range_end == 100

    def test_all_in_one_bin(self) -> None:
        results = [_make_result(f"D{i:03d}", 85) for i in range(5)]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        eighth_bin = report.index_distribution[8]
        assert eighth_bin.count == 5
        total = sum(b.count for b in report.index_distribution)
        assert total == 5


# ── TC-C-5: Category Breakdown ──────────────────────────────────────────────


class TestCategoryBreakdown:
    def test_breakdown_by_model(self) -> None:
        results = [
            _make_result("D001", 80),
            _make_result("D002", 60),
        ]
        models = {"D001": "Kyocera 3253ci", "D002": "HP M404"}
        store = _make_factor_store(results, models=models)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.category_breakdown is not None
        assert report.category_breakdown.category_field == "model"
        assert len(report.category_breakdown.groups) == 2

    def test_single_model_breakdown_by_location(self) -> None:
        results = [
            _make_result("D001", 80),
            _make_result("D002", 60),
        ]
        models = {"D001": "Kyocera 3253ci", "D002": "Kyocera 3253ci"}
        locations = {"D001": "Офис A", "D002": "Офис B"}
        store = _make_factor_store(results, models=models, locations=locations)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.category_breakdown is not None
        assert report.category_breakdown.category_field == "location"

    def test_no_model_no_location_confidence_zone(self) -> None:
        results = [
            _make_result("D001", 80, confidence=0.9),
            _make_result("D002", 60, confidence=0.5),
        ]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.category_breakdown is not None
        assert report.category_breakdown.category_field == "confidence_zone"


# ── TC-C-6: Device Reports ──────────────────────────────────────────────────


class TestDeviceReports:
    def test_sorted_by_index_ascending(self) -> None:
        results = [
            _make_result("D003", 90),
            _make_result("D001", 20),
            _make_result("D002", 55),
        ]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        indices = [d.health_index for d in report.devices]
        assert indices == sorted(indices)

    def test_resource_state_from_factor_store(self) -> None:
        results = [_make_result("D001", 50)]
        store = FactorStore()
        store.set_device_metadata(
            "D001", DeviceMetadata(device_id="D001"),
        )
        store.set_resources(
            "D001",
            ResourceSnapshot(
                device_id="D001",
                timestamp=NOW,
                toner_level=45,
                drum_level=72,
                fuser_level=95,
                mileage=180_000,
            ),
        )
        store.freeze()

        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        dev = report.devices[0]
        assert dev.resource_state.toner == 45
        assert dev.resource_state.drum == 72
        assert dev.resource_state.fuser == 95
        assert dev.resource_state.mileage == 180_000

    def test_top_problem_tag_set(self) -> None:
        results = [_make_result("D001", 30, error_label="Fuser failure")]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.devices[0].top_problem_tag == "Fuser failure"


# ── TC-C-7: Agent Trace Summary ─────────────────────────────────────────────


class TestAgentTraceSummary:
    def test_trace_summary_included_when_flag_set(self) -> None:
        results = [_make_result("D001", 50)]
        store = _make_factor_store(results)
        traces = {
            "D001": Trace(
                session_id="s1",
                mode=AgentMode.BATCH,
                device_id="D001",
                started_at=NOW,
                total_tool_calls=5,
                total_llm_calls=3,
                total_tokens=2000,
                flagged_for_review=True,
            ),
        }
        builder = _make_builder()
        report = builder.build(
            results, store, _make_snapshot(), _make_source_info(),
            traces=traces, include_agent_trace=True,
        )

        assert report.agent_trace_summary is not None
        assert report.agent_trace_summary.average_tool_calls_per_device == 5.0
        assert "D001" in report.agent_trace_summary.devices_flagged_for_review

    def test_no_trace_summary_without_flag(self) -> None:
        results = [_make_result("D001", 50)]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.agent_trace_summary is None


# ── TC-C-8: Executive Summary via Agent ──────────────────────────────────────


class TestExecutiveSummary:
    def test_summary_generated_via_agent(self) -> None:
        results = [_make_result("D001", 50)]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.executive_summary == _DEFAULT_SUMMARY

    def test_fallback_on_agent_failure(self) -> None:
        agent = MagicMock()
        agent._llm.generate.side_effect = RuntimeError("LLM unavailable")
        config = ReportConfig()
        builder = ReportBuilder(agent=agent, config=config)

        results = [_make_result("D001", 80)]
        store = _make_factor_store(results)
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert "1 устройств" in report.executive_summary or "Парк из" in report.executive_summary

    def test_empty_results_summary(self) -> None:
        builder = _make_builder()
        store = FactorStore()
        report = builder.build([], store, _make_snapshot(), _make_source_info())

        assert report.executive_summary == "Нет устройств для анализа."


# ── TC-C-9: Report structure ────────────────────────────────────────────────


class TestReportStructure:
    def test_report_has_all_fields(self) -> None:
        results = [_make_result(f"D{i:03d}", 50 + i) for i in range(5)]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.report_id
        assert report.generated_at
        assert report.source_file_name == "test.csv"
        assert report.source_file_hash == "sha256:abc"
        assert report.fleet_summary.total_devices == 5
        assert len(report.index_distribution) == 10
        assert report.category_breakdown is not None
        assert len(report.devices) == 5
        assert report.calculation_snapshot is not None

    def test_all_green_zone(self) -> None:
        results = [_make_result(f"D{i:03d}", 90) for i in range(5)]
        store = _make_factor_store(results)
        builder = _make_builder()
        report = builder.build(results, store, _make_snapshot(), _make_source_info())

        assert report.fleet_summary.zone_counts["green"] == 5
        assert report.fleet_summary.zone_counts["red"] == 0
        assert report.fleet_summary.zone_counts["yellow"] == 0
