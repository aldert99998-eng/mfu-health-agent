"""E2E test — full batch pipeline: ingest → agent.run_batch → build_report → render_pdf.

10 devices with varied health states, mock LLM, Qdrant testcontainer.
Target: < 2 minutes with mock LLM.
"""

from __future__ import annotations

import csv
import json
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from agent.core import Agent
from agent.tools.impl import ToolDependencies, register_all_tools
from agent.tools.registry import ToolRegistry
from config.loader import AgentConfig, ConfigManager, ReportConfig
from data_io.factor_store import FactorStore
from data_io.models import (
    BatchContext,
    CalculationSnapshot,
    FileFormat,
    HealthResult,
    HealthZone,
    Report,
    SourceFileInfo,
    Trace,
    WeightsProfile,
)
from data_io.normalizer import ingest_file
from llm.client import LLMResponse, TokenUsage, ToolCall
from reporting.pdf_generator import PDFGenerator
from reporting.report_builder import ReportBuilder

pytestmark = pytest.mark.e2e

# ── Constants ───────────────────────────────────────────────────────────────

NOW = datetime.now(UTC)
DEVICE_IDS = [f"D{i:03d}" for i in range(1, 11)]

_HEALTHY_CODES = ["I0001"]
_MEDIUM_CODES = ["C3000", "C4000"]
_CRITICAL_CODES = ["C6000", "C7000", "C8000"]


# ── CSV fixture generation ──────────────────────────────────────────────────


def _generate_fixture_csv(path: Path) -> None:
    """Generate CSV with 10 devices in varied health states."""
    rows: list[dict[str, str]] = []

    devices_spec: list[tuple[str, str, str, list[tuple[str, int]]]] = [
        # (device_id, model, location, [(error_code, repeat_count)])
        ("D001", "Kyocera TASKalfa 3253ci", "Офис 1", [("I0001", 1)]),
        ("D002", "Kyocera TASKalfa 3253ci", "Офис 1", [("I0001", 2)]),
        ("D003", "Kyocera TASKalfa 4053ci", "Офис 2", [("I0001", 1)]),
        ("D004", "Kyocera TASKalfa 4053ci", "Офис 2", [("C3000", 2), ("I0001", 1)]),
        ("D005", "Kyocera TASKalfa 3253ci", "Офис 3", [("C3000", 3), ("C4000", 1)]),
        ("D006", "Kyocera TASKalfa 5053ci", "Офис 3", [("C4000", 2), ("C3000", 2)]),
        ("D007", "Kyocera TASKalfa 5053ci", "Офис 4", [("C6000", 1), ("C3000", 1)]),
        ("D008", "Kyocera TASKalfa 3253ci", "Офис 4", [("C6000", 2), ("C7000", 1)]),
        ("D009", "Kyocera TASKalfa 4053ci", "Офис 5", [("C6000", 3), ("C8000", 1), ("C7000", 2)]),
        ("D010", "Kyocera TASKalfa 5053ci", "Офис 5", [("C8000", 2), ("C7000", 2), ("C6000", 1)]),
    ]

    for device_id, model, location, errors in devices_spec:
        for error_code, count in errors:
            for i in range(count):
                ts = NOW - timedelta(days=i, hours=device_id.__hash__() % 12)
                rows.append({
                    "Serial Number": device_id,
                    "Date/Time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "Error Code": error_code,
                    "Error Description": f"Error {error_code} occurrence",
                    "Model Name": model,
                    "Location": location,
                    "Status": "error",
                    "Toner Level": str(max(10, 90 - int(device_id[1:]) * 8)),
                    "Drum Level": str(max(15, 85 - int(device_id[1:]) * 5)),
                    "Fuser Level": str(max(5, 95 - int(device_id[1:]) * 9)),
                    "Mileage": str(50000 + int(device_id[1:]) * 15000),
                })

    fieldnames = [
        "Serial Number", "Date/Time", "Error Code", "Error Description",
        "Model Name", "Location", "Status",
        "Toner Level", "Drum Level", "Fuser Level", "Mileage",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Mock LLM ────────────────────────────────────────────────────────────────


class MockLLMClient:
    """LLM mock that returns deterministic tool-call sequences and final answers."""

    def __init__(self) -> None:
        self._call_count: dict[str, int] = {}

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        params: Any = None,
    ) -> LLMResponse:
        session_key = self._extract_device_id(messages)
        self._call_count.setdefault(session_key, 0)
        self._call_count[session_key] += 1
        call_num = self._call_count[session_key]

        if response_schema is not None:
            return self._guided_json_response(messages)

        if self._is_reflection(messages):
            return self._reflection_response()

        if self._is_executive_summary(messages):
            return self._executive_summary_response()

        if tools and call_num == 1:
            return self._first_tool_calls(session_key)
        if tools and call_num == 2:
            return self._second_tool_calls(session_key)

        return self._final_answer(session_key)

    def _extract_device_id(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                for did in DEVICE_IDS:
                    if did in content:
                        return did
        return "unknown"

    def _is_reflection(self, messages: list[dict[str, Any]]) -> bool:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "ревизор" in content.lower():
                return True
        return False

    def _is_executive_summary(self, messages: list[dict[str, Any]]) -> bool:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "executive" in content.lower():
                return True
            if isinstance(content, str) and "резюме" in content.lower():
                return True
        return False

    def _reflection_response(self) -> LLMResponse:
        return LLMResponse(
            content=json.dumps({
                "verdict": "approved",
                "issues": [],
                "recommended_action": "accept",
            }),
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

    def _executive_summary_response(self) -> LLMResponse:
        return LLMResponse(
            content="Парк из 10 устройств: 3 в зелёной зоне, 4 в жёлтой, 3 в красной. "
                    "Основные проблемы: отказы фьюзера (C6000) на TASKalfa 3253ci/4053ci.",
            usage=TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300),
        )

    def _first_tool_calls(self, device_id: str) -> LLMResponse:
        call_id_1 = f"call_{uuid.uuid4().hex[:8]}"
        call_id_2 = f"call_{uuid.uuid4().hex[:8]}"
        return LLMResponse(
            content="Получаю данные устройства.",
            tool_calls=[
                ToolCall(
                    id=call_id_1,
                    name="get_device_events",
                    arguments={"device_id": device_id, "window_days": 30},
                ),
                ToolCall(
                    id=call_id_2,
                    name="get_device_resources",
                    arguments={"device_id": device_id},
                ),
            ],
            usage=TokenUsage(prompt_tokens=300, completion_tokens=50, total_tokens=350),
        )

    def _second_tool_calls(self, device_id: str) -> LLMResponse:
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        return LLMResponse(
            content="Считаю индекс здоровья.",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="calculate_health_index",
                    arguments={"device_id": device_id},
                ),
            ],
            usage=TokenUsage(prompt_tokens=400, completion_tokens=50, total_tokens=450),
        )

    def _final_answer(self, device_id: str) -> LLMResponse:
        idx = int(device_id[1:])
        if idx <= 3:
            health_index, confidence, zone = 85 + (3 - idx) * 3, 0.9, "green"
        elif idx <= 6:
            health_index, confidence, zone = 55 + (6 - idx) * 5, 0.7, "yellow"
        else:
            health_index, confidence, zone = 25 + (10 - idx) * 5, 0.5, "red"

        result = {
            "health_index": health_index,
            "confidence": confidence,
            "zone": zone,
            "factor_contributions": [],
            "confidence_reasons": [],
        }
        return LLMResponse(
            content=json.dumps(result),
            usage=TokenUsage(prompt_tokens=500, completion_tokens=100, total_tokens=600),
        )

    def _guided_json_response(self, messages: list[dict[str, Any]]) -> LLMResponse:
        device_id = self._extract_device_id(messages)
        return self._final_answer(device_id)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def csv_fixture_path() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = Path(f.name)
    _generate_fixture_csv(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture(scope="module")
def factor_store(csv_fixture_path: Path, mock_llm: MockLLMClient) -> FactorStore:
    fs = FactorStore()
    result = ingest_file(csv_fixture_path, fs, llm_client=mock_llm)
    assert result.success, f"Ingestion failed: {result.errors}"
    assert result.devices_count == 10
    fs.freeze()
    return fs


@pytest.fixture(scope="module")
def weights_profile() -> WeightsProfile:
    try:
        cm = ConfigManager()
        profiles = cm.list_profiles()
        if profiles:
            return cm.load_weights(profiles[0])
    except Exception:
        pass
    return WeightsProfile(profile_name="default")


@pytest.fixture(scope="module")
def agent_config() -> AgentConfig:
    try:
        cm = ConfigManager()
        return cm.load_agent_config()
    except Exception:
        return AgentConfig()


@pytest.fixture(scope="module")
def report_config() -> ReportConfig:
    try:
        cm = ConfigManager()
        return cm.load_report_config()
    except Exception:
        return ReportConfig()


@pytest.fixture(scope="module")
def agent(
    mock_llm: MockLLMClient,
    factor_store: FactorStore,
    weights_profile: WeightsProfile,
    agent_config: AgentConfig,
) -> Agent:
    registry = ToolRegistry()
    deps = ToolDependencies(
        factor_store=factor_store,
        weights=weights_profile,
        searcher=None,
        llm_client=mock_llm,
    )
    register_all_tools(registry, deps)
    return Agent(
        llm_client=mock_llm,
        tool_registry=registry,
        factor_store=factor_store,
        config=agent_config,
    )


@pytest.fixture(scope="module")
def batch_results(
    agent: Agent,
    factor_store: FactorStore,
    weights_profile: WeightsProfile,
) -> list[tuple[HealthResult, Trace]]:
    context = BatchContext(
        weights_profile=weights_profile,
        factor_store=factor_store,
    )
    results = []
    for device_id in factor_store.list_devices():
        result, trace = agent.run_batch(device_id, context)
        results.append((result, trace))
    return results


@pytest.fixture(scope="module")
def health_results(batch_results: list[tuple[HealthResult, Trace]]) -> list[HealthResult]:
    return [r for r, _ in batch_results]


@pytest.fixture(scope="module")
def traces(batch_results: list[tuple[HealthResult, Trace]]) -> dict[str, Trace]:
    return {t.device_id: t for _, t in batch_results}


@pytest.fixture(scope="module")
def report(
    agent: Agent,
    health_results: list[HealthResult],
    factor_store: FactorStore,
    weights_profile: WeightsProfile,
    report_config: ReportConfig,
    traces: dict[str, Trace],
    csv_fixture_path: Path,
) -> Report:

    builder = ReportBuilder(agent, report_config)

    calc_snapshot = CalculationSnapshot(
        weights_profile_name=weights_profile.profile_name,
        weights_profile_version=weights_profile.version,
        weights_data=weights_profile.model_dump(mode="json"),
        llm_model="mock-llm",
        source_file_hash="test-hash",
        input_record_count=30,
        valid_record_count=30,
        discarded_record_count=0,
    )
    source_info = SourceFileInfo(
        file_name=csv_fixture_path.name,
        file_hash="sha256:test",
        file_format=FileFormat.CSV,
        uploaded_at=NOW,
    )

    return builder.build(
        health_results=health_results,
        factor_store=factor_store,
        calculation_snapshot=calc_snapshot,
        source_file_info=source_info,
        traces=traces,
    )


@pytest.fixture(scope="module")
def pdf_bytes(report: Report, report_config: ReportConfig) -> bytes:
    gen = PDFGenerator(report_config)
    return gen.generate(report)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestIngestion:
    def test_factor_store_has_10_devices(self, factor_store: FactorStore) -> None:
        devices = factor_store.list_devices()
        assert len(devices) == 10

    def test_all_device_ids_present(self, factor_store: FactorStore) -> None:
        devices = set(factor_store.list_devices())
        for did in DEVICE_IDS:
            assert did in devices, f"Device {did} missing from FactorStore"

    def test_events_exist_for_all_devices(self, factor_store: FactorStore) -> None:
        for did in factor_store.list_devices():
            events = factor_store.get_events(did)
            assert len(events) > 0, f"No events for {did}"


class TestBatchResults:
    def test_10_results(self, health_results: list[HealthResult]) -> None:
        assert len(health_results) == 10

    def test_no_duplicate_device_ids(self, health_results: list[HealthResult]) -> None:
        ids = [r.device_id for r in health_results]
        assert len(ids) == len(set(ids))

    def test_health_index_range(self, health_results: list[HealthResult]) -> None:
        for r in health_results:
            assert 1 <= r.health_index <= 100, f"{r.device_id}: index={r.health_index}"

    def test_confidence_range(self, health_results: list[HealthResult]) -> None:
        for r in health_results:
            assert 0.2 <= r.confidence <= 1.0, f"{r.device_id}: confidence={r.confidence}"

    def test_all_zones_represented(self, health_results: list[HealthResult]) -> None:
        zones = {r.zone for r in health_results}
        for z in (HealthZone.GREEN, HealthZone.YELLOW, HealthZone.RED):
            assert z in zones, f"Zone {z} not found in results"

    def test_traces_have_steps(self, batch_results: list[tuple[HealthResult, Trace]]) -> None:
        for result, trace in batch_results:
            assert trace.total_llm_calls > 0, f"{result.device_id}: no LLM calls"
            assert len(trace.steps) > 0, f"{result.device_id}: no trace steps"

    def test_calculated_at_set(self, health_results: list[HealthResult]) -> None:
        for r in health_results:
            assert r.calculated_at is not None


class TestReport:
    def test_report_has_fleet_summary(self, report: Report) -> None:
        assert report.fleet_summary is not None
        assert report.fleet_summary.total_devices == 10

    def test_report_average_index_reasonable(self, report: Report) -> None:
        avg = report.fleet_summary.average_index
        assert 1 <= avg <= 100

    def test_report_has_executive_summary(self, report: Report) -> None:
        assert report.executive_summary
        assert len(report.executive_summary) > 10

    def test_report_has_all_devices(self, report: Report) -> None:
        assert len(report.devices) == 10
        report_ids = {d.device_id for d in report.devices}
        for did in DEVICE_IDS:
            assert did in report_ids

    def test_report_zone_counts_sum(self, report: Report) -> None:
        zc = report.fleet_summary.zone_counts
        total = sum(zc.values())
        assert total == 10

    def test_report_has_distribution(self, report: Report) -> None:
        assert report.index_distribution


class TestPDF:
    def test_pdf_not_empty(self, pdf_bytes: bytes) -> None:
        assert len(pdf_bytes) > 0

    def test_pdf_starts_with_magic(self, pdf_bytes: bytes) -> None:
        assert pdf_bytes[:5] == b"%PDF-"

    def test_pdf_contains_device_ids(self, pdf_bytes: bytes) -> None:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        for did in DEVICE_IDS:
            assert did in full_text, f"Device {did} not found in PDF"


class TestPerformance:
    def test_full_pipeline_under_2_minutes(
        self,
        csv_fixture_path: Path,
        weights_profile: WeightsProfile,
        agent_config: AgentConfig,
        report_config: ReportConfig,
    ) -> None:
        t0 = time.perf_counter()

        mock = MockLLMClient()
        fs = FactorStore()
        ingest_file(csv_fixture_path, fs, llm_client=mock)
        fs.freeze()

        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=weights_profile,
            searcher=None,
            llm_client=mock,
        )
        register_all_tools(registry, deps)
        ag = Agent(
            llm_client=mock,
            tool_registry=registry,
            factor_store=fs,
            config=agent_config,
        )

        context = BatchContext(weights_profile=weights_profile, factor_store=fs)
        results = []
        trace_map: dict[str, Trace] = {}
        for did in fs.list_devices():
            r, t = ag.run_batch(did, context)
            results.append(r)
            trace_map[did] = t

        builder = ReportBuilder(ag, report_config)
        calc_snap = CalculationSnapshot(
            weights_profile_name="default",
            weights_profile_version="1.0",
            weights_data={},
            llm_model="mock",
            source_file_hash="perf-test",
            input_record_count=30,
            valid_record_count=30,
            discarded_record_count=0,
        )
        src_info = SourceFileInfo(
            file_name="perf.csv",
            file_hash="sha256:perf",
            file_format=FileFormat.CSV,
            uploaded_at=NOW,
        )
        report = builder.build(
            health_results=results,
            factor_store=fs,
            calculation_snapshot=calc_snap,
            source_file_info=src_info,
            traces=trace_map,
        )

        pdf_gen = PDFGenerator(report_config)
        pdf = pdf_gen.generate(report)

        elapsed = time.perf_counter() - t0
        assert elapsed < 120, f"Full pipeline took {elapsed:.1f}s (limit: 120s)"
        assert len(pdf) > 0
