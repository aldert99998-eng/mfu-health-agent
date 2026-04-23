"""P0 tests for the MFU Reporting system — Track C.

TC-C-001 .. TC-C-100: covers aggregation, device cards, snapshots,
pattern detection, HTML rendering, XSS escaping, Cyrillic, PDF,
executive summary with LLM mock, and Pydantic validation.
"""

from __future__ import annotations

import html as html_mod
import statistics
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from config.loader import ReportConfig
from data_io.factor_store import DeviceMetadata, FactorStore, ResourceSnapshot
from data_io.models import (
    AgentMode,
    CalculationSnapshot,
    CategoryBreakdown,
    ConfidenceZone,
    DeviceReport,
    DistributionBin,
    FactorContribution,
    FileFormat,
    FleetSummary,
    HealthResult,
    HealthZone,
    Report,
    ResourceState,
    SourceFileInfo,
    Trace,
)
from reporting.report_builder import ReportBuilder

NOW = datetime.now(UTC)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _hr(
    device_id: str,
    health_index: int,
    confidence: float = 0.85,
    zone: str | None = None,
    error_label: str = "C6000 (Critical)",
    reflection_notes: str | None = None,
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
                label=error_label, penalty=10.0, S=60, R=1.0, C=1.0, A=1.0, source="test",
            ),
        ],
        calculated_at=NOW,
        reflection_notes=reflection_notes,
    )


def _store(
    results: list[HealthResult],
    models: dict[str, str] | None = None,
    locations: dict[str, str] | None = None,
    resources: dict[str, ResourceSnapshot] | None = None,
) -> FactorStore:
    store = FactorStore()
    models = models or {}
    locations = locations or {}
    resources = resources or {}
    for r in results:
        store.set_device_metadata(
            r.device_id,
            DeviceMetadata(
                device_id=r.device_id,
                model=models.get(r.device_id),
                location=locations.get(r.device_id),
            ),
        )
        if r.device_id in resources:
            store.set_resources(r.device_id, resources[r.device_id])
    store.freeze()
    return store


def _snapshot() -> CalculationSnapshot:
    return CalculationSnapshot(
        weights_profile_name="default",
        weights_profile_version="1.0",
        weights_data={"severity": {"critical": 60}, "age": {"window_days": 30}},
    )


def _source() -> SourceFileInfo:
    return SourceFileInfo(
        file_name="fleet_data.csv",
        file_hash="sha256:deadbeef",
        file_format=FileFormat.CSV,
        uploaded_at=NOW,
    )


def _builder(llm_answer: str = "Парк из 10 устройств в хорошем состоянии. Средний индекс здоровья составляет 75.") -> ReportBuilder:
    agent = MagicMock()
    llm_response = MagicMock()
    llm_response.content = llm_answer
    agent._llm.generate.return_value = llm_response
    return ReportBuilder(agent=agent, config=ReportConfig())


def _build_report(
    results: list[HealthResult],
    models: dict[str, str] | None = None,
    locations: dict[str, str] | None = None,
    llm_answer: str = "Парк из 10 устройств в хорошем состоянии. Средний индекс здоровья составляет 75.",
) -> Report:
    builder = _builder(llm_answer)
    store = _store(results, models=models, locations=locations)
    return builder.build(results, store, _snapshot(), _source())


# ── TC-C-001: FleetSummary aggregation ───────────────────────────────────────


class TestTCC001FleetSummaryAggregation:
    """Given N HealthResults, verify total_devices, zone_counts, avg_health_index."""

    def test_total_devices_equals_input_count(self) -> None:
        results = [_hr(f"D{i:03d}", 50 + i) for i in range(7)]
        report = _build_report(results)
        assert report.fleet_summary.total_devices == 7

    def test_zone_counts_correct(self) -> None:
        results = [
            _hr("R1", 20),   # red
            _hr("R2", 30),   # red
            _hr("Y1", 50),   # yellow
            _hr("Y2", 60),   # yellow
            _hr("Y3", 70),   # yellow
            _hr("G1", 80),   # green
            _hr("G2", 95),   # green
        ]
        report = _build_report(results)
        assert report.fleet_summary.zone_counts == {"green": 2, "yellow": 3, "red": 2}

    def test_average_health_index(self) -> None:
        results = [_hr("A", 20), _hr("B", 40), _hr("C", 60), _hr("D", 80)]
        report = _build_report(results)
        expected_avg = round(statistics.mean([20, 40, 60, 80]), 1)
        assert report.fleet_summary.average_index == expected_avg

    def test_median_health_index(self) -> None:
        results = [_hr("A", 10), _hr("B", 50), _hr("C", 90)]
        report = _build_report(results)
        assert report.fleet_summary.median_index == 50.0

    def test_average_confidence(self) -> None:
        results = [
            _hr("A", 50, confidence=0.9),
            _hr("B", 50, confidence=0.6),
        ]
        report = _build_report(results)
        assert report.fleet_summary.average_confidence == 0.75


# ── TC-C-002: Per-device cards contain all required fields ───────────────────


class TestTCC002DeviceCardFields:
    """Per-device cards contain device_id, model, location, health_index, zone, confidence."""

    def test_all_required_fields_present(self) -> None:
        results = [_hr("MFU-001", 65, confidence=0.88)]
        models = {"MFU-001": "Kyocera 3253ci"}
        locations = {"MFU-001": "Офис Москва"}
        report = _build_report(results, models=models, locations=locations)

        dev = report.devices[0]
        assert dev.device_id == "MFU-001"
        assert dev.model == "Kyocera 3253ci"
        assert dev.location == "Офис Москва"
        assert dev.health_index == 65
        assert dev.zone == HealthZone.YELLOW
        assert dev.confidence == 0.88
        assert dev.confidence_zone == ConfidenceZone.HIGH

    def test_missing_model_location_are_none(self) -> None:
        results = [_hr("MFU-002", 50)]
        report = _build_report(results)
        dev = report.devices[0]
        assert dev.model is None
        assert dev.location is None

    def test_top_problem_tag_set(self) -> None:
        results = [_hr("MFU-003", 30, error_label="Fuser overheating")]
        report = _build_report(results)
        assert report.devices[0].top_problem_tag == "Fuser overheating"


# ── TC-C-003: CalculationSnapshot included in report ────────────────────────


class TestTCC003CalculationSnapshot:
    """CalculationSnapshot is included and preserves original data."""

    def test_snapshot_present(self) -> None:
        results = [_hr("D001", 50)]
        report = _build_report(results)
        assert report.calculation_snapshot is not None

    def test_snapshot_fields_match(self) -> None:
        results = [_hr("D001", 50)]
        report = _build_report(results)
        snap = report.calculation_snapshot
        assert snap.weights_profile_name == "default"
        assert snap.weights_profile_version == "1.0"
        assert snap.weights_data["severity"]["critical"] == 60

    def test_analysis_window_from_snapshot(self) -> None:
        results = [_hr("D001", 50)]
        report = _build_report(results)
        assert report.analysis_window_days == 30


# ── TC-C-021: HTML contains all mandatory sections ──────────────────────────


class TestTCC021HTMLMandatorySections:
    """HTML output contains all mandatory div/section elements."""

    def test_mandatory_sections_present(self) -> None:
        results = [_hr(f"D{i:03d}", 50 + i * 5) for i in range(5)]
        models = {f"D{i:03d}": f"Model{i}" for i in range(5)}
        report = _build_report(results, models=models)
        builder = _builder()
        html = builder.render_html(report)

        assert html, "render_html returned empty string"

        mandatory_ids = [
            'id="header"',
            'id="executive-summary"',
            'id="distribution"',
            'id="device-table"',
            'id="device-cards"',
            'id="calc-snapshot"',
        ]
        for section_id in mandatory_ids:
            assert section_id in html, f"Missing section: {section_id}"

    def test_category_breakdown_section_present(self) -> None:
        results = [_hr("D001", 80), _hr("D002", 60)]
        models = {"D001": "ModelA", "D002": "ModelB"}
        report = _build_report(results, models=models)
        builder = _builder()
        html = builder.render_html(report)
        assert 'id="category-breakdown"' in html


# ── TC-C-022: HTML escaping — XSS prevention ────────────────────────────────


class TestTCC022HTMLEscaping:
    """Inject <script>alert('xss')</script> in device_id, verify it's escaped."""

    def test_script_tag_escaped_in_device_id(self) -> None:
        xss_payload = "<script>alert('xss')</script>"
        results = [_hr(xss_payload, 50)]
        report = _build_report(results)
        builder = _builder()
        html = builder.render_html(report)

        assert html, "render_html returned empty"
        # The raw script tag must NOT appear
        assert "<script>alert" not in html
        # The escaped form should be present
        assert html_mod.escape(xss_payload) in html or "&lt;script&gt;" in html

    def test_script_in_model_escaped(self) -> None:
        results = [_hr("D001", 50)]
        xss_model = "<img onerror=alert(1) src=x>"
        models = {"D001": xss_model}
        report = _build_report(results, models=models)
        builder = _builder()
        html = builder.render_html(report)
        # The raw tag must not appear — angle brackets must be escaped
        assert xss_model not in html
        assert "&lt;img" in html, "Expected HTML-escaped angle brackets"


# ── TC-C-023: Cyrillic renders correctly in HTML ────────────────────────────


class TestTCC023CyrillicHTML:
    """Cyrillic text renders without encoding issues in HTML."""

    def test_cyrillic_in_executive_summary(self) -> None:
        cyrillic_summary = "Парк из 5 устройств в отличном состоянии. Средний индекс здоровья составляет 90."
        results = [_hr("D001", 90)]
        report = _build_report(results, llm_answer=cyrillic_summary)
        builder = _builder(cyrillic_summary)
        html = builder.render_html(report)
        assert "Парк из 5 устройств" in html

    def test_cyrillic_location_in_html(self) -> None:
        results = [_hr("D001", 50)]
        locations = {"D001": "Москва, ул. Ленина 42"}
        report = _build_report(results, locations=locations)
        builder = _builder()
        html = builder.render_html(report)
        assert "Москва" in html

    def test_html_charset_utf8(self) -> None:
        results = [_hr("D001", 50)]
        report = _build_report(results)
        builder = _builder()
        html = builder.render_html(report)
        assert 'charset="UTF-8"' in html or "charset=UTF-8" in html


# ── TC-C-030: PDF generates without errors ──────────────────────────────────


class TestTCC030PDFGeneration:
    """PDF generates without errors (skip gracefully if WeasyPrint unavailable)."""

    def test_pdf_generates_successfully(self) -> None:
        try:
            import weasyprint  # noqa: F401
        except ImportError:
            pytest.skip("WeasyPrint not installed")

        results = [_hr(f"D{i:03d}", 40 + i * 6) for i in range(5)]
        models = {f"D{i:03d}": "Kyocera TASKalfa" for i in range(5)}
        report = _build_report(results, models=models)
        builder = _builder()
        pdf_bytes = builder.render_pdf(report)

        assert pdf_bytes, "render_pdf returned empty bytes"
        assert pdf_bytes[:5] == b"%PDF-", "Output does not start with %PDF-"

    def test_pdf_has_pages(self) -> None:
        try:
            import weasyprint  # noqa: F401
            from pypdf import PdfReader
        except ImportError:
            pytest.skip("WeasyPrint or pypdf not installed")

        import io

        results = [_hr(f"D{i:03d}", 50) for i in range(3)]
        report = _build_report(results)
        builder = _builder()
        pdf_bytes = builder.render_pdf(report)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) >= 1


# ── TC-C-040: Special chars in PDF — no squares/boxes ───────────────────────


class TestTCC040SpecialCharsInPDF:
    """Special characters (Cyrillic, symbols) render without tofu/boxes in PDF."""

    def test_cyrillic_extractable_from_pdf(self) -> None:
        try:
            import weasyprint  # noqa: F401
            from pypdf import PdfReader
        except ImportError:
            pytest.skip("WeasyPrint or pypdf not installed")

        import io

        results = [_hr("Принтер-001", 50)]
        locations = {"Принтер-001": "Офис Санкт-Петербург"}
        report = _build_report(results, locations=locations)
        builder = _builder()
        pdf_bytes = builder.render_pdf(report)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        # Check no replacement characters (U+FFFD) or box chars
        assert "\ufffd" not in full_text, "Found replacement character U+FFFD in PDF"

        # Check that key Cyrillic text is extractable
        assert "Отчёт" in full_text or "здоровь" in full_text, (
            "Expected Cyrillic report title text in PDF"
        )


# ── TC-C-050: Executive summary mentions key facts ─────────────────────────


class TestTCC050ExecutiveSummary:
    """Executive summary (via mocked LLM) mentions key facts."""

    def test_llm_summary_used_when_available(self) -> None:
        llm_text = "Общий индекс парка 55. Обнаружено 2 критических устройства."
        results = [_hr("D001", 30), _hr("D002", 80)]
        report = _build_report(results, llm_answer=llm_text)
        assert report.executive_summary == llm_text

    def test_fallback_summary_on_llm_failure(self) -> None:
        agent = MagicMock()
        agent._llm.generate.side_effect = RuntimeError("LLM timeout")
        builder = ReportBuilder(agent=agent, config=ReportConfig())

        results = [
            _hr("D001", 20),   # red
            _hr("D002", 50),   # yellow
            _hr("D003", 90),   # green
        ]
        store = _store(results)
        report = builder.build(results, store, _snapshot(), _source())

        summary = report.executive_summary
        # Fallback should mention device count and stats
        assert "3" in summary, "Fallback summary should mention total device count"
        assert "красн" in summary.lower() or "red" in summary.lower() or "1" in summary

    def test_empty_fleet_summary_text(self) -> None:
        builder = _builder()
        store = FactorStore()
        report = builder.build([], store, _snapshot(), _source())
        assert "Нет устройств" in report.executive_summary

    def test_llm_receives_fleet_data(self) -> None:
        """Verify the LLM is called directly with fleet data context."""
        agent = MagicMock()
        llm_response = MagicMock()
        llm_response.content = "Summary text with enough characters to pass validation check."
        agent._llm.generate.return_value = llm_response
        builder = ReportBuilder(agent=agent, config=ReportConfig())

        results = [_hr("D001", 50), _hr("D002", 80)]
        store = _store(results)
        builder.build(results, store, _snapshot(), _source())

        assert agent._llm.generate.call_count == 1
        call_kwargs = agent._llm.generate.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages") or call_kwargs[0][0]
        prompt_text = messages[0]["content"] if isinstance(messages, list) else str(messages)
        assert "total_devices" in prompt_text or "2" in prompt_text


# ── TC-C-100: Report data model Pydantic validation ─────────────────────────


class TestTCC100PydanticValidation:
    """Report and sub-model Pydantic validation works correctly."""

    def test_health_result_index_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _hr("D001", 0)  # below minimum (1)

        with pytest.raises(ValidationError):
            _hr("D001", 101)  # above maximum (100)

    def test_health_result_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _hr("D001", 50, confidence=0.1)  # below 0.2

        with pytest.raises(ValidationError):
            _hr("D001", 50, confidence=1.5)  # above 1.0

    def test_fleet_summary_valid(self) -> None:
        fs = FleetSummary(
            total_devices=5,
            average_index=72.3,
            median_index=75.0,
            zone_counts={"green": 3, "yellow": 1, "red": 1},
            average_confidence=0.82,
        )
        assert fs.total_devices == 5

    def test_device_report_valid(self) -> None:
        dr = DeviceReport(
            device_id="MFU-001",
            health_index=65,
            confidence=0.85,
            zone=HealthZone.YELLOW,
            confidence_zone=ConfidenceZone.HIGH,
        )
        assert dr.device_id == "MFU-001"
        assert dr.model is None
        assert dr.resource_state is not None

    def test_device_report_index_validation(self) -> None:
        with pytest.raises(ValidationError):
            DeviceReport(
                device_id="BAD",
                health_index=200,
                confidence=0.85,
                zone=HealthZone.GREEN,
                confidence_zone=ConfidenceZone.HIGH,
            )

    def test_report_roundtrip_json(self) -> None:
        results = [_hr("D001", 50)]
        report = _build_report(results)
        json_str = report.model_dump_json()
        restored = Report.model_validate_json(json_str)
        assert restored.report_id == report.report_id
        assert restored.fleet_summary.total_devices == 1
        assert len(restored.devices) == 1

    def test_calculation_snapshot_standalone_valid(self) -> None:
        snap = CalculationSnapshot(
            weights_profile_name="custom",
            weights_profile_version="2.0",
            weights_data={"key": "value"},
            llm_model="gpt-4",
            input_record_count=100,
            valid_record_count=95,
            discarded_record_count=5,
        )
        assert snap.weights_profile_name == "custom"
        assert snap.discarded_record_count == 5
