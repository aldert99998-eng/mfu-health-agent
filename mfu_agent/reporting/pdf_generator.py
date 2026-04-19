"""PDF generation via WeasyPrint — Track C, Phase 6.3.

System dependencies (Linux):
    apt install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf2.0-0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, StrictUndefined

if TYPE_CHECKING:
    from config.loader import ReportConfig
    from data_io.models import Report

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PDFGenerationError(Exception):
    """Raised when WeasyPrint fails to produce a valid PDF."""


class PDFGenerator:
    """Renders a Report to PDF bytes via Jinja2 + WeasyPrint."""

    def __init__(self, config: ReportConfig) -> None:
        self._config = config

    def generate(self, report: Report) -> bytes:
        html_string = self._render_html(report)
        pdf_bytes = self._html_to_pdf(html_string)
        self._validate(pdf_bytes)
        return pdf_bytes

    def _render_html(self, report: Report) -> str:
        template_dir = _PROJECT_ROOT / Path(self._config.rendering.template_path).parent
        template_name = Path(self._config.rendering.template_path).name

        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True,
            undefined=StrictUndefined,
        )
        return env.get_template(template_name).render(
            report=report,
            interactive=False,
        )

    def _html_to_pdf(self, html_string: str) -> bytes:
        try:
            from weasyprint import CSS, HTML
        except ImportError as exc:
            raise PDFGenerationError(
                "WeasyPrint не установлен. "
                "Установите: pip install weasyprint"
            ) from exc

        css_path = _PROJECT_ROOT / self._config.rendering.css_path
        print_css_path = _PROJECT_ROOT / self._config.rendering.print_css_path
        base_url = str(_PROJECT_ROOT / "reporting")

        stylesheets = []
        if css_path.exists():
            stylesheets.append(CSS(filename=str(css_path)))
        if print_css_path.exists():
            stylesheets.append(CSS(filename=str(print_css_path)))

        try:
            return HTML(  # type: ignore[no-any-return]
                string=html_string,
                base_url=base_url,
            ).write_pdf(stylesheets=stylesheets)
        except Exception as exc:
            raise PDFGenerationError(
                f"WeasyPrint не смог сгенерировать PDF: {exc}. "
                "Проверьте наличие шрифтов и системных зависимостей "
                "(libpango, libcairo)."
            ) from exc

    @staticmethod
    def _validate(pdf_bytes: bytes) -> None:
        if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
            raise PDFGenerationError(
                "Результат не является валидным PDF (отсутствует заголовок %PDF-)."
            )

        try:
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
        except ImportError:
            logger.debug("pypdf не установлен, пропускаем валидацию страниц")
            return
        except Exception as exc:
            raise PDFGenerationError(
                f"Сгенерированный PDF не читается pypdf: {exc}"
            ) from exc

        if num_pages == 0:
            raise PDFGenerationError("PDF содержит 0 страниц.")

        text = ""
        for page in reader.pages[:3]:
            text += page.extract_text() or ""

        if "Отчёт" not in text and "здоровь" not in text:
            logger.warning(
                "PDF не содержит ожидаемого текста 'Отчёт о здоровье' "
                "на первых страницах — возможна проблема с кодировкой шрифтов"
            )
