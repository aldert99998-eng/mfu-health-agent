"""Visual regression tests for report rendering.

Renders a fixture report to PDF via WeasyPrint, converts pages to PNG
via PyMuPDF, and compares against golden files using PIL pixel diff.

Threshold: pixel difference < 1% → pass, ≥ 1% → fail.
Update golden files:  pytest tests/visual/ --update-golden

Requires: WeasyPrint system deps (libpango, libcairo, libgdk-pixbuf).
Run:  python -m pytest tests/visual/test_report_rendering.py -v -s
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageChops

from config.loader import ReportConfig
from data_io.models import (
    CalculationSnapshot,
    ConfidenceZone,
    DeviceReport,
    DistributionBin,
    FactorContribution,
    FleetSummary,
    HealthZone,
    PatternGroup,
    PatternType,
    Report,
    ResourceState,
)
from reporting.pdf_generator import PDFGenerator

# ── Config ───────────────────────────────────────────────────────────────────

_GOLDEN_DIR = Path(__file__).parent / "golden"
_DPI = 150
_DIFF_THRESHOLD_PERCENT = 1.0

# ── Rich fixture report ─────────────────────────────────────────────────────

_NOW = datetime(2025, 3, 15, 10, 0, 0, tzinfo=UTC)

_FACTOR_FUSER = FactorContribution(
    label="Отказ фьюзера (C6000)", penalty=30.96,
    S=60.0, R=2.58, C=1.5, A=0.8,
    source="Service Manual TASKalfa 3253ci, §7.2",
)
_FACTOR_DRUM = FactorContribution(
    label="Износ барабана (C0800)", penalty=8.4,
    S=15.0, R=1.0, C=1.2, A=0.7,
    source="Service Manual TASKalfa 3253ci, §8.1",
)
_FACTOR_JAM = FactorContribution(
    label="Замятие бумаги (J0510)", penalty=4.2,
    S=10.0, R=1.4, C=1.0, A=0.3,
    source="Service Manual TASKalfa 3253ci, §6.1",
)

_DEVICES = [
    DeviceReport(
        device_id="D001", model="Kyocera TASKalfa 3253ci",
        location="Офис 3, этаж 2", health_index=42, confidence=0.7,
        zone=HealthZone.YELLOW, confidence_zone=ConfidenceZone.MEDIUM,
        top_problem_tag="fuser",
        factor_contributions=[_FACTOR_FUSER, _FACTOR_DRUM],
        resource_state=ResourceState(toner=45, drum=72, fuser=95, mileage=182_000),
        agent_recommendation="Заменить фьюзер в ближайшее ТО.",
    ),
    DeviceReport(
        device_id="D002", model="Kyocera TASKalfa 3253ci",
        location="Офис 3, этаж 2", health_index=38, confidence=0.65,
        zone=HealthZone.RED, confidence_zone=ConfidenceZone.MEDIUM,
        top_problem_tag="fuser",
        factor_contributions=[_FACTOR_FUSER],
        resource_state=ResourceState(toner=12, drum=55, fuser=98, mileage=220_000),
        agent_recommendation="Критический износ фьюзера. Требуется немедленная замена.",
        flag_for_review=True,
    ),
    DeviceReport(
        device_id="D003", model="HP LaserJet Pro M404n",
        location="Офис 1, этаж 1", health_index=91, confidence=0.95,
        zone=HealthZone.GREEN, confidence_zone=ConfidenceZone.HIGH,
        top_problem_tag="",
        resource_state=ResourceState(toner=78, drum=85, mileage=45_000),
    ),
    DeviceReport(
        device_id="D004", model="HP LaserJet Pro M404n",
        location="Офис 1, этаж 1", health_index=88, confidence=0.92,
        zone=HealthZone.GREEN, confidence_zone=ConfidenceZone.HIGH,
        top_problem_tag="",
        resource_state=ResourceState(toner=65, drum=90, mileage=32_000),
    ),
    DeviceReport(
        device_id="D005", model="Ricoh MP C3003",
        location="Офис 2, этаж 3", health_index=55, confidence=0.78,
        zone=HealthZone.YELLOW, confidence_zone=ConfidenceZone.MEDIUM,
        top_problem_tag="paper_feed",
        factor_contributions=[_FACTOR_JAM],
        resource_state=ResourceState(toner=30, drum=40, fuser=60, mileage=150_000),
        agent_recommendation="Заменить ролики подачи бумаги.",
    ),
    DeviceReport(
        device_id="D006", model="Xerox VersaLink C405",
        location="Офис 2, этаж 3", health_index=95, confidence=0.98,
        zone=HealthZone.GREEN, confidence_zone=ConfidenceZone.HIGH,
        top_problem_tag="",
        resource_state=ResourceState(toner=88, drum=92, mileage=18_000),
    ),
    DeviceReport(
        device_id="D007", model="Kyocera ECOSYS M2040dn",
        location="Склад", health_index=25, confidence=0.55,
        zone=HealthZone.RED, confidence_zone=ConfidenceZone.LOW,
        top_problem_tag="motor",
        factor_contributions=[_FACTOR_FUSER, _FACTOR_DRUM, _FACTOR_JAM],
        resource_state=ResourceState(toner=5, drum=15, fuser=98, mileage=310_000),
        agent_recommendation="Устройство требует капитального ремонта.",
        flag_for_review=True,
    ),
    DeviceReport(
        device_id="D008", model="Kyocera ECOSYS M2040dn",
        location="Склад", health_index=78, confidence=0.85,
        zone=HealthZone.GREEN, confidence_zone=ConfidenceZone.HIGH,
        top_problem_tag="",
        resource_state=ResourceState(toner=55, drum=70, mileage=95_000),
    ),
]

_DISTRIBUTION = [
    DistributionBin(range_start=0, range_end=10, count=0),
    DistributionBin(range_start=10, range_end=20, count=0),
    DistributionBin(range_start=20, range_end=30, count=1),
    DistributionBin(range_start=30, range_end=40, count=1),
    DistributionBin(range_start=40, range_end=50, count=1),
    DistributionBin(range_start=50, range_end=60, count=1),
    DistributionBin(range_start=60, range_end=70, count=0),
    DistributionBin(range_start=70, range_end=80, count=1),
    DistributionBin(range_start=80, range_end=90, count=1),
    DistributionBin(range_start=90, range_end=100, count=2),
]

_PATTERNS = [
    PatternGroup(
        pattern_type=PatternType.MASS_ISSUE,
        title="C6000 на TASKalfa 3253ci (2 устройства)",
        affected_device_ids=["D001", "D002"],
        average_index=40.0,
        explanation="Массовый отказ фьюзера на устройствах одной модели. "
        "Рекомендуется проверить партию фьюзеров.",
    ),
    PatternGroup(
        pattern_type=PatternType.CRITICAL_SINGLE,
        title="D007 — критический износ",
        affected_device_ids=["D007"],
        average_index=25.0,
        explanation="Устройство в критическом состоянии с множественными проблемами.",
    ),
]

VISUAL_REPORT = Report(
    report_id="visual-test-001",
    generated_at=_NOW,
    source_file_name="fleet_visual_test.csv",
    source_file_hash="sha256:visual_test_hash",
    analysis_window_days=30,
    fleet_summary=FleetSummary(
        total_devices=8,
        average_index=64.0,
        median_index=66.5,
        zone_counts={"green": 4, "yellow": 2, "red": 2},
        average_confidence=0.81,
        delta_vs_previous=-3.2,
    ),
    executive_summary=(
        "Парк из 8 устройств показывает средний индекс здоровья 64.0 — "
        "умеренное состояние. Два устройства Kyocera TASKalfa 3253ci страдают "
        "от массового отказа фьюзера (C6000). Устройство D007 (ECOSYS M2040dn) "
        "в критическом состоянии и требует капитального ремонта. "
        "Рекомендуется приоритизировать замену фьюзеров на TASKalfa 3253ci "
        "и провести диагностику D007."
    ),
    top_patterns=_PATTERNS,
    index_distribution=_DISTRIBUTION,
    devices=_DEVICES,
    calculation_snapshot=CalculationSnapshot(
        weights_profile_name="default",
        weights_profile_version="1.0",
        weights_data={"severity": {"critical": 60, "high": 30, "medium": 10}},
        llm_model="qwen2.5-7b-instruct",
        source_file_hash="sha256:visual_test_hash",
        input_record_count=250,
        valid_record_count=248,
        discarded_record_count=2,
    ),
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _weasyprint_available() -> bool:
    try:
        from weasyprint import HTML  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


requires_weasyprint = pytest.mark.skipif(
    not _weasyprint_available(),
    reason="WeasyPrint not installed or system deps missing",
)


def _render_report_to_pngs(report: Report) -> list[Image.Image]:
    """Render report → PDF → list of PIL Images (one per page)."""
    cfg = ReportConfig()
    gen = PDFGenerator(cfg)
    pdf_bytes = gen.generate(report)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(dpi=_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def _pixel_diff_percent(img_a: Image.Image, img_b: Image.Image) -> float:
    """Compute percentage of differing pixels between two images.

    Images are resized to the same dimensions before comparison.
    A pixel is considered different if any channel differs by more than 2.
    """
    w = max(img_a.width, img_b.width)
    h = max(img_a.height, img_b.height)
    a = img_a.resize((w, h), Image.LANCZOS)
    b = img_b.resize((w, h), Image.LANCZOS)

    diff = ImageChops.difference(a, b)
    pixels = diff.load()
    total = w * h
    changed = 0
    for y in range(h):
        for x in range(w):
            r, g, b_val = pixels[x, y]
            if r > 2 or g > 2 or b_val > 2:
                changed += 1

    return (changed / total) * 100.0


def _save_diff_image(
    img_a: Image.Image, img_b: Image.Image, path: Path,
) -> None:
    """Save amplified diff image for debugging."""
    w = max(img_a.width, img_b.width)
    h = max(img_a.height, img_b.height)
    a = img_a.resize((w, h), Image.LANCZOS)
    b = img_b.resize((w, h), Image.LANCZOS)

    diff = ImageChops.difference(a, b)
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Brightness(diff)
    amplified = enhancer.enhance(10.0)
    amplified.save(str(path))


# ── Pytest fixtures & config ─────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden files for visual tests",
    )


@pytest.fixture(scope="module")
def update_golden(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--update-golden", default=False)


@pytest.fixture(scope="module")
def rendered_pages() -> list[Image.Image]:
    return _render_report_to_pngs(VISUAL_REPORT)


# ── Tests ────────────────────────────────────────────────────────────────────


@requires_weasyprint
class TestReportRendering:
    """Visual regression: rendered report vs golden PNG files."""

    def test_renders_without_error(self, rendered_pages: list[Image.Image]) -> None:
        """Report renders to at least 1 page."""
        assert len(rendered_pages) >= 1

    def test_page_dimensions_reasonable(self, rendered_pages: list[Image.Image]) -> None:
        """Pages have A4-ish dimensions at the configured DPI."""
        for img in rendered_pages:
            assert img.width > 500, f"Page too narrow: {img.width}px"
            assert img.height > 700, f"Page too short: {img.height}px"

    def test_page_not_blank(self, rendered_pages: list[Image.Image]) -> None:
        """Pages are not entirely white/blank."""
        for i, img in enumerate(rendered_pages):
            extrema = img.getextrema()
            is_solid = all(lo == hi for lo, hi in extrema)
            assert not is_solid, f"Page {i} is a solid color — rendering likely failed"

    def test_golden_comparison(
        self,
        rendered_pages: list[Image.Image],
        update_golden: bool,
    ) -> None:
        """Each page matches its golden file within the threshold."""
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

        created_new = False
        for i, current in enumerate(rendered_pages):
            golden_path = _GOLDEN_DIR / f"report_page_{i}.png"
            diff_path = _GOLDEN_DIR / f"report_page_{i}_diff.png"

            if update_golden or not golden_path.exists():
                current.save(str(golden_path))
                if diff_path.exists():
                    diff_path.unlink()
                created_new = True
                continue

            golden = Image.open(str(golden_path))
            diff_pct = _pixel_diff_percent(golden, current)

            if diff_pct >= _DIFF_THRESHOLD_PERCENT:
                _save_diff_image(golden, current, diff_path)
                current.save(str(_GOLDEN_DIR / f"report_page_{i}_actual.png"))
                pytest.fail(
                    f"Page {i}: pixel diff {diff_pct:.2f}% exceeds "
                    f"threshold {_DIFF_THRESHOLD_PERCENT}%.\n"
                    f"  Golden:  {golden_path}\n"
                    f"  Actual:  {_GOLDEN_DIR / f'report_page_{i}_actual.png'}\n"
                    f"  Diff:    {diff_path}\n"
                    f"Review and run with --update-golden to accept."
                )
            else:
                if diff_path.exists():
                    diff_path.unlink()

        if created_new and not update_golden:
            pytest.skip(
                "Golden files created for the first time. "
                "Re-run to compare against them."
            )

    def test_multi_page_report(self, rendered_pages: list[Image.Image]) -> None:
        """Report with 8 devices produces multiple pages."""
        assert len(rendered_pages) >= 2, (
            f"Expected ≥ 2 pages for 8 devices, got {len(rendered_pages)}"
        )


@requires_weasyprint
class TestRenderingEdgeCases:
    """Edge cases that should not break rendering."""

    def test_minimal_report(self) -> None:
        """Report with 1 device, no patterns, no distribution renders OK."""
        minimal = Report(
            report_id="minimal-001",
            generated_at=_NOW,
            source_file_name="minimal.csv",
            source_file_hash="sha256:minimal",
            analysis_window_days=30,
            fleet_summary=FleetSummary(
                total_devices=1,
                average_index=85.0,
                median_index=85.0,
                zone_counts={"green": 1, "yellow": 0, "red": 0},
                average_confidence=0.95,
            ),
            devices=[
                DeviceReport(
                    device_id="SOLO-001",
                    health_index=85,
                    confidence=0.95,
                    zone=HealthZone.GREEN,
                    confidence_zone=ConfidenceZone.HIGH,
                ),
            ],
            calculation_snapshot=CalculationSnapshot(
                weights_profile_name="default",
                weights_profile_version="1.0",
                weights_data={},
                llm_model="test",
                source_file_hash="sha256:minimal",
                input_record_count=10,
                valid_record_count=10,
                discarded_record_count=0,
            ),
        )
        pages = _render_report_to_pngs(minimal)
        assert len(pages) >= 1

    def test_all_red_zone(self) -> None:
        """Report where all devices are in red zone renders OK."""
        red_devices = [
            DeviceReport(
                device_id=f"RED-{i:03d}",
                model="Problem Model X",
                health_index=15 + i,
                confidence=0.5,
                zone=HealthZone.RED,
                confidence_zone=ConfidenceZone.LOW,
                flag_for_review=True,
                agent_recommendation="Срочный ремонт.",
            )
            for i in range(5)
        ]
        red_report = Report(
            report_id="all-red-001",
            generated_at=_NOW,
            source_file_name="red_fleet.csv",
            source_file_hash="sha256:red",
            analysis_window_days=30,
            fleet_summary=FleetSummary(
                total_devices=5,
                average_index=17.0,
                median_index=17.0,
                zone_counts={"green": 0, "yellow": 0, "red": 5},
                average_confidence=0.5,
            ),
            devices=red_devices,
            calculation_snapshot=CalculationSnapshot(
                weights_profile_name="default",
                weights_profile_version="1.0",
                weights_data={},
                llm_model="test",
                source_file_hash="sha256:red",
                input_record_count=50,
                valid_record_count=50,
                discarded_record_count=0,
            ),
        )
        pages = _render_report_to_pngs(red_report)
        assert len(pages) >= 1
        for img in pages:
            extrema = img.getextrema()
            assert not all(lo == hi for lo, hi in extrema)

    def test_empty_executive_summary(self) -> None:
        """Report with no executive summary renders without error."""
        report = Report(
            report_id="no-summary-001",
            generated_at=_NOW,
            source_file_name="test.csv",
            source_file_hash="sha256:test",
            analysis_window_days=30,
            fleet_summary=FleetSummary(
                total_devices=1,
                average_index=75.0,
                median_index=75.0,
                zone_counts={"green": 1, "yellow": 0, "red": 0},
                average_confidence=0.9,
            ),
            executive_summary="",
            devices=[
                DeviceReport(
                    device_id="X001",
                    health_index=75,
                    confidence=0.9,
                    zone=HealthZone.GREEN,
                    confidence_zone=ConfidenceZone.HIGH,
                ),
            ],
            calculation_snapshot=CalculationSnapshot(
                weights_profile_name="default",
                weights_profile_version="1.0",
                weights_data={},
                llm_model="test",
                source_file_hash="sha256:test",
                input_record_count=1,
                valid_record_count=1,
                discarded_record_count=0,
            ),
        )
        pages = _render_report_to_pngs(report)
        assert len(pages) >= 1
