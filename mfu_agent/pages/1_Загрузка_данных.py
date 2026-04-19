"""Страница 1 — Загрузка данных мониторинга и результаты анализа."""

from __future__ import annotations

import re
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

_SQL_KW = {"SELECT", "UNION", "FROM", "WHERE", "INNER", "LEFT", "JOIN", "INSERT", "CREATE"}


def _strip_csv_preamble(dest: Path, raw: bytes) -> bytes:
    """Remove SQL/comment preamble from CSV files before writing to disk."""
    if dest.suffix.lower() not in (".csv", ".tsv"):
        return raw

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw

    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            start = i + 1
            continue
        if stripped.startswith("--"):
            start = i + 1
            continue
        tokens = set(re.split(r"[\s(,;]+", stripped.upper()[:300]))
        if tokens & _SQL_KW:
            start = i + 1
            continue
        break

    if start > 0:
        cleaned = "\n".join(lines[start:])
        return cleaned.encode("utf-8")
    return raw

from state.session import (
    get_active_llm_endpoint,
    get_active_weights_profile,
    get_current_report,
    set_current_factor_store,
    set_current_health_results,
    set_current_report,
)

# ── Background worker ────────────────────────────────────────────────────────

from state import _analysis_bg as _bg


def _run_analysis_worker(
    file_bytes: bytes,
    file_name: str,
    tmpdir: str,
    progress: dict,
    llm_endpoint: str,
    weights_profile,
) -> None:
    """Run the full analysis pipeline in a background thread."""
    try:
        import importlib
        import data_io.normalizer
        import data_io.parsers
        import data_io.zabbix_transform

        _bg.log(f"Загрузка файла **{file_name}** ({len(file_bytes)} байт)")
        _bg.log("Перезагрузка модулей парсинга...")
        importlib.reload(data_io.parsers)
        importlib.reload(data_io.zabbix_transform)
        importlib.reload(data_io.normalizer)

        from agent.core import Agent
        from agent.tools.impl import ToolDependencies, register_all_tools
        from agent.tools.registry import ToolRegistry
        from config.loader import ConfigManager, ReportConfig
        from data_io.factor_store import FactorStore
        from data_io.models import (
            BatchContext,
            CalculationSnapshot,
            FileFormat,
            SourceFileInfo,
            WeightsProfile,
        )
        from data_io.normalizer import ingest_file
        from reporting.report_builder import ReportBuilder

        dest = Path(tmpdir) / file_name
        raw_bytes = _strip_csv_preamble(dest, file_bytes)
        if len(raw_bytes) != len(file_bytes):
            _bg.log("Обнаружена SQL-преамбула в файле, очищена")
        dest.write_bytes(raw_bytes)

        # ── Шаг 1: Ingestion ────────────────────────────────────────────
        progress["step"] = "ingestion"
        progress["step_label"] = "Загрузка и нормализация данных..."
        _bg.log("**Шаг 1/3** — Парсинг файла и нормализация данных...")

        ext = Path(file_name).suffix.lower()
        _bg.log(f"Формат файла: {ext}")

        fs = FactorStore()
        result = ingest_file(dest, fs)

        if not result.success:
            progress["status"] = "error"
            progress["errors"] = result.errors
            for err in result.errors:
                _bg.log(f"ОШИБКА: {err}")
            return

        fs.freeze()
        device_ids = fs.list_devices()

        progress["devices_count"] = result.devices_count
        progress["events_count"] = result.valid_events_count
        progress["snapshots_count"] = result.valid_snapshots_count
        progress["warnings"] = result.warnings

        _bg.log(
            f"Парсинг завершён: {result.devices_count} устройств, "
            f"{result.valid_events_count} событий, "
            f"{result.valid_snapshots_count} ресурсных снимков"
        )
        if result.warnings:
            for w in result.warnings:
                _bg.log(f"Предупреждение: {w}")
        if result.invalid_records:
            _bg.log(f"Отброшено некорректных записей: {len(result.invalid_records)}")

        # ── Шаг 2: Расчёт индексов ──────────────────────────────────────
        progress["step"] = "calculation"
        progress["step_label"] = f"Расчёт индексов для {len(device_ids)} устройств..."
        _bg.log(f"**Шаг 2/3** — Расчёт индексов здоровья для {len(device_ids)} устройств")

        cm = ConfigManager()
        agent_config = cm.load_agent_config()
        _bg.log("Конфигурация агента загружена")

        weights = weights_profile
        if weights is None:
            profiles = cm.list_profiles()
            weights = cm.load_weights(profiles[0]) if profiles else WeightsProfile(
                profile_name="default",
            )
        _bg.log(f"Весовой профиль: **{weights.profile_name}**")

        try:
            from state.singletons import get_llm_client
            _bg.log(f"Подключение к LLM (эндпоинт: {llm_endpoint})...")
            llm_client = get_llm_client(llm_endpoint)
            _bg.log("LLM подключён")
        except Exception as exc:
            progress["status"] = "error"
            progress["errors"] = [
                "Не удалось подключиться к LLM. "
                "Проверьте настройки эндпоинта в configs/llm_endpoints.yaml"
            ]
            _bg.log(f"ОШИБКА подключения к LLM: {exc}")
            return

        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=fs,
            weights=weights,
        )
        register_all_tools(registry, deps)

        agent = Agent(
            llm_client=llm_client,
            tool_registry=registry,
            factor_store=fs,
            config=agent_config,
        )

        context = BatchContext(
            weights_profile=weights,
            factor_store=fs,
        )

        health_results = []
        traces = {}

        _bg.log(
            "Начинаю расчёт. Для каждого устройства: классификация ошибок через LLM, "
            "затем детерминированный расчёт индекса"
        )

        for i, did in enumerate(device_ids):
            n_events = len(fs.get_events(did))
            snap = fs.get_resources(did)
            snap_info = "есть" if snap else "нет"
            _bg.log(
                f"[{i+1}/{len(device_ids)}] Устройство **{did}**: "
                f"событий={n_events}, снимок={snap_info}"
            )

            hr, trace = agent.run_batch_lite(did, context)
            health_results.append(hr)
            traces[did] = trace
            progress["calc_done"] = i + 1
            progress["calc_total"] = len(device_ids)
            progress["calc_last"] = f"{did}: индекс {hr.health_index}, зона {hr.zone}"

            _bg.log(
                f"[{i+1}/{len(device_ids)}] **{did}** → "
                f"индекс={hr.health_index}, зона={hr.zone}, "
                f"уверенность={hr.confidence:.0%}"
            )

        set_current_health_results(health_results)
        _bg.log(f"Расчёт завершён для всех {len(health_results)} устройств")

        # ── Шаг 3: Построение отчёта ────────────────────────────────────
        progress["step"] = "report"
        progress["step_label"] = "Формирование отчёта..."
        _bg.log("**Шаг 3/3** — Формирование итогового отчёта...")

        try:
            report_config = cm.load_report_config()
        except Exception:
            report_config = ReportConfig()

        builder = ReportBuilder(agent, report_config)
        _bg.log("Агрегация данных по флоту, расчёт сводных метрик...")

        calc_snapshot = CalculationSnapshot(
            weights_profile_name=weights.profile_name,
            weights_profile_version=weights.version,
            weights_data=weights.model_dump(mode="json"),
            llm_model=llm_endpoint,
            source_file_hash=result.source_file_info.file_hash,
            input_record_count=result.total_records,
            valid_record_count=result.valid_events_count,
            discarded_record_count=len(result.invalid_records),
        )

        format_map = {
            ".csv": FileFormat.CSV,
            ".tsv": FileFormat.TSV,
            ".json": FileFormat.JSON,
            ".jsonl": FileFormat.JSONL,
            ".ndjson": FileFormat.JSONL,
            ".xlsx": FileFormat.XLSX,
        }
        file_format = format_map.get(ext, FileFormat.CSV)

        source_info = SourceFileInfo(
            file_name=file_name,
            file_hash=result.source_file_info.file_hash,
            file_size_bytes=len(file_bytes),
            file_format=file_format,
            uploaded_at=datetime.now(UTC),
        )

        _bg.log("Генерация executive summary через LLM...")
        report = builder.build(
            health_results=health_results,
            factor_store=fs,
            calculation_snapshot=calc_snapshot,
            source_file_info=source_info,
            traces=traces,
        )

        progress["status"] = "complete"
        progress["_result_report"] = report
        progress["_result_fs"] = fs
        progress["_result_health"] = health_results

        _bg.log(f"Отчёт сформирован (ID: {report.report_id})")
        _bg.log("**Анализ завершён!**")

    except Exception as exc:
        progress["status"] = "error"
        progress["errors"] = [f"Непредвиденная ошибка: {exc}"]
        _bg.log(f"ОШИБКА: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════

st.header("Загрузка данных мониторинга")

# ═══════════════════════════════════════════════════════════════════════════════
# Блок 1 — Загрузка файла с данными устройств
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("Загрузка данных")

uploaded = st.file_uploader(
    "Файл с данными устройств",
    type=["csv", "tsv", "json", "jsonl", "ndjson", "xlsx"],
    help="Поддерживаются CSV, TSV, JSON, JSONL, XLSX",
)

# ── Check if background analysis is running ─────────────────────────────────

def _render_log() -> None:
    if not _bg.log_lines:
        return
    with st.expander("Журнал операций", expanded=True):
        st.markdown("\n\n".join(_bg.log_lines))


if _bg.thread is not None and _bg.thread.is_alive():
    st.info("⏳ Идёт анализ...")

    step = _bg.progress.get("step", "")
    step_label = _bg.progress.get("step_label", "Загрузка...")

    if step == "calculation":
        done = _bg.progress.get("calc_done", 0)
        total = _bg.progress.get("calc_total", 1)
        st.progress(done / total if total else 0, text=step_label)

    _render_log()

    time.sleep(1)
    st.rerun()

elif _bg.thread is not None and not _bg.thread.is_alive():
    status = _bg.progress.get("status")

    if status == "complete":
        report_obj = _bg.progress.get("_result_report")
        fs_obj = _bg.progress.get("_result_fs")
        health_obj = _bg.progress.get("_result_health")

        if report_obj and fs_obj and health_obj:
            set_current_factor_store(fs_obj)
            set_current_health_results(health_obj)
            set_current_report(report_obj)

        st.success(
            f"Анализ завершён — "
            f"устройств: {_bg.progress.get('devices_count', '?')}, "
            f"событий: {_bg.progress.get('events_count', '?')}, "
            f"снимков: {_bg.progress.get('snapshots_count', '?')}"
        )
        if _bg.progress.get("warnings"):
            for w in _bg.progress["warnings"]:
                st.warning(w)

    elif status == "error":
        for err in _bg.progress.get("errors", ["Неизвестная ошибка"]):
            st.error(err)

    _render_log()

    _bg.thread = None
    _bg.progress.clear()

# ── Start button ─────────────────────────────────────────────────────────────

is_running = _bg.thread is not None and _bg.thread.is_alive()

if uploaded and not is_running and st.button(
    "Загрузить и рассчитать", type="primary", use_container_width=True
):
    _bg.progress.clear()
    _bg.progress["status"] = "running"
    _bg.log_lines.clear()

    tmpdir = tempfile.mkdtemp()

    llm_endpoint = get_active_llm_endpoint()
    weights = get_active_weights_profile()

    _bg.thread = threading.Thread(
        target=_run_analysis_worker,
        args=(
            uploaded.getvalue(),
            uploaded.name,
            tmpdir,
            _bg.progress,
            llm_endpoint,
            weights,
        ),
        daemon=True,
    )
    _bg.thread.start()
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# Блок 2 — Результаты (если отчёт уже загружен)
# ═══════════════════════════════════════════════════════════════════════════════

report = get_current_report()

if report is None:
    st.divider()
    st.info("Загрузите файл с данными устройств для расчёта индекса здоровья.")
    st.stop()

st.divider()
st.subheader("Результаты анализа")

# ── Метрики ──────────────────────────────────────────────────────────────────

fs_summary = report.fleet_summary

cols = st.columns(5)
cols[0].metric("Устройств", fs_summary.total_devices)
cols[1].metric("Средний индекс", f"{fs_summary.average_index:.1f}")
cols[2].metric(
    "Медиана",
    f"{fs_summary.median_index:.1f}",
    delta=f"{fs_summary.delta_vs_previous:+.1f}" if fs_summary.delta_vs_previous else None,
)
cols[3].metric("Уверенность", f"{fs_summary.average_confidence:.0%}")
cols[4].metric("Источник", report.source_file_name)

_ZONE_COLORS = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
_ZONE_LABELS = {"green": "Зелёная", "yellow": "Жёлтая", "red": "Красная"}

zone_parts = []
for z in ("green", "yellow", "red"):
    count = fs_summary.zone_counts.get(z, 0)
    zone_parts.append(f"{_ZONE_COLORS[z]} {_ZONE_LABELS[z]}: **{count}**")
st.markdown(" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(zone_parts))

# ── Таблица устройств ────────────────────────────────────────────────────────

st.subheader("Устройства")

if report.devices:
    rows = []
    for d in report.devices:
        rows.append({
            "ID": d.device_id,
            "Модель": d.model or "—",
            "Локация": d.location or "—",
            "Индекс": d.health_index,
            "Уверенность": f"{d.confidence:.0%}",
            "Зона": _ZONE_LABELS.get(d.zone, d.zone),
            "Проблема": d.top_problem_tag or "—",
            "На проверку": "да" if d.flag_for_review else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Индекс": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        },
    )

# ── Резюме ───────────────────────────────────────────────────────────────────

if report.executive_summary:
    st.subheader("Краткое резюме")
    st.markdown(report.executive_summary)

# ── Скачать PDF ──────────────────────────────────────────────────────────────

st.divider()
if st.button("Сгенерировать PDF-отчёт", use_container_width=True):
    from config.loader import ConfigManager, ReportConfig
    from reporting.pdf_generator import PDFGenerator

    try:
        pdf_config = ConfigManager().load_report_config()
    except Exception:
        pdf_config = ReportConfig()

    with st.spinner("Генерация PDF..."):
        gen = PDFGenerator(pdf_config)
        pdf_bytes = gen.generate(report)

    st.download_button(
        label="Скачать PDF",
        data=pdf_bytes,
        file_name=f"health_report_{report.report_id}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
