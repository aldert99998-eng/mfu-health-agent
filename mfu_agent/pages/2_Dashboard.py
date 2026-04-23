"""Страница 1 — Дашборд здоровья парка МФУ."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from state import _deep_bg as _deep
from state.session import (
    claim_bg_results_if_any,
    get_active_llm_endpoint,
    get_active_weights_profile,
    get_current_factor_store,
    get_current_health_results,
    get_current_report,
    get_deep_device_analyses,
    get_mass_error_analyses,
    set_deep_device_analyses,
    set_mass_error_analyses,
)

st.header("Дашборд здоровья парка")

from ui.endpoint_selector import render_endpoint_selector

render_endpoint_selector(location="sidebar", key_suffix="dashboard")

claim_bg_results_if_any()
report = get_current_report()

if report is None:
    st.info(
        "Отчёт ещё не сформирован. "
        "Перейдите на страницу **Загрузка данных** и загрузите файл с данными устройств."
    )
    st.stop()

# ── Block 1: Fleet summary metrics ──────────────────────────────────────────

fs = report.fleet_summary

cols = st.columns(5)
cols[0].metric("Устройств", fs.total_devices)
cols[1].metric("Средний индекс", f"{fs.average_index:.1f}")
cols[2].metric(
    "Медиана",
    f"{fs.median_index:.1f}",
    delta=f"{fs.delta_vs_previous:+.1f}" if fs.delta_vs_previous is not None else None,
)
cols[3].metric("Уверенность", f"{fs.average_confidence:.0%}")
cols[4].metric("Источник", report.source_file_name)

_ZONE_LABELS = {"green": "Зелёная", "yellow": "Жёлтая", "red": "Красная"}
_ZONE_COLORS = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

zone_parts = []
for zone_key in ("green", "yellow", "red"):
    count = fs.zone_counts.get(zone_key, 0)
    icon = _ZONE_COLORS.get(zone_key, "")
    label = _ZONE_LABELS.get(zone_key, zone_key)
    zone_parts.append(f"{icon} {label}: **{count}**")
st.markdown(" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(zone_parts))


# ── Block 2: Index distribution chart ──────────────────────────────────────

st.subheader("Распределение индекса здоровья")

if report.index_distribution:
    dist_df = pd.DataFrame(
        {
            "Диапазон": [
                f"{b.range_start}–{b.range_end}" for b in report.index_distribution
            ],
            "Кол-во": [b.count for b in report.index_distribution],
        }
    )
    st.bar_chart(dist_df, x="Диапазон", y="Кол-во")
else:
    st.caption("Данные о распределении отсутствуют.")


# ── Block 3: Executive summary ─────────────────────────────────────────────

st.subheader("Краткое резюме")

if report.executive_summary:
    st.markdown(report.executive_summary)
else:
    st.caption("Резюме не сформировано.")


# ── Block 4.5: Deep LLM analysis ───────────────────────────────────────────


def _build_agent_for_deep_analysis(factor_store, weights, endpoint_name: str):
    """Construct a fully-wired Agent with tools, RAG and LLM.

    ``endpoint_name`` MUST be captured on the Streamlit main thread and
    passed in — Streamlit session_state (the source of the "active
    endpoint" key) is not readable from background worker threads, so
    calling ``get_active_llm_endpoint()`` here would silently fall back
    to the default (local) endpoint regardless of the user's choice.
    """
    try:
        from agent.core import Agent
        from agent.tools.impl import ToolDependencies, register_all_tools
        from agent.tools.registry import ToolRegistry
        from config.loader import get_config_manager
        from state.singletons import (
            get_hybrid_searcher,
            get_llm_client,
            get_memory_manager,
        )

        llm = get_llm_client(endpoint_name)

        try:
            searcher = get_hybrid_searcher()
        except Exception as exc:  # noqa: BLE001
            _deep.log(f"⚠️ RAG недоступен: {exc}")
            searcher = None

        try:
            memory = get_memory_manager()
        except Exception:
            memory = None

        agent_config = get_config_manager().load_agent_config()

        registry = ToolRegistry()
        deps = ToolDependencies(
            factor_store=factor_store,
            weights=weights,
            searcher=searcher,
            llm_client=llm,
            memory_manager=memory,
        )
        register_all_tools(registry, deps)

        return Agent(
            llm_client=llm,
            tool_registry=registry,
            factor_store=factor_store,
            config=agent_config,
            memory_manager=memory,
        )
    except Exception as exc:  # noqa: BLE001
        _deep.log(f"❌ Не удалось инициализировать Agent: {exc}")
        return None


def _render_deep_log(*, running: bool) -> None:
    if not _deep.log_lines:
        return
    header_cols = st.columns([5, 1])
    with header_cols[0]:
        icon = "⏳" if running else "📋"
        st.markdown(f"**{icon} Журнал глубокого анализа** — {len(_deep.log_lines)} строк")
    with header_cols[1]:
        if not running and st.button("Очистить", key="clear_deep_log"):
            _deep.log_lines.clear()
            st.rerun()
    with st.container(height=300, border=True):
        st.markdown("\n\n".join(_deep.log_lines))


def _run_mass_error_worker(results, fs, weights, endpoint_name: str) -> None:
    """Background worker: LLM-analyse top-10 mass errors.

    All session-state objects must be passed from the main thread —
    Streamlit session_state is not accessible from worker threads.
    ``endpoint_name`` picks which LLM provider runs the analysis.
    """
    _deep.progress.clear()
    _deep.progress["kind"] = "mass"
    _deep.progress["status"] = "running"
    _deep.progress["should_stop"] = False
    _deep.log_lines.clear()
    _deep.log("Старт анализа массовых ошибок (top-10)…")

    if not results:
        _deep.log("❌ Нет результатов анализа")
        _deep.progress["status"] = "error"
        return

    _deep.log(f"🧠 Провайдер: {endpoint_name}")
    agent = _build_agent_for_deep_analysis(fs, weights, endpoint_name)
    if agent is None:
        _deep.progress["status"] = "error"
        return

    # Aggregate codes by affected-device count
    code_devices: dict[str, set[str]] = defaultdict(set)
    code_total: Counter[str] = Counter()
    code_severity: dict[str, str] = {}
    for r in results:
        for fc in r.factor_contributions:
            code = fc.label.split(" (")[0] if " (" in fc.label else fc.label
            code_devices[code].add(r.device_id)
            code_total[code] += 1
            # Extract severity from "CODE (Severity)" label
            if " (" in fc.label and fc.label.endswith(")"):
                code_severity[code] = fc.label.rsplit(" (", 1)[1][:-1]

    top = sorted(
        code_devices.items(), key=lambda kv: len(kv[1]), reverse=True,
    )[:10]

    if not top:
        _deep.log("⚠️ Не найдено ни одного кода ошибки в результатах")
        _deep.progress["status"] = "complete"
        return

    fleet_total = len(results)
    output: dict = {}
    # Publish the output dict reference immediately so the UI can render
    # partial results while the worker is still running.
    _deep.progress["_result_mass"] = output

    _deep.progress["total"] = len(top)
    _deep.progress["done"] = 0

    for i, (code, devs) in enumerate(top, 1):
        if _deep.progress.get("should_stop"):
            _deep.log(f"⏹ Остановлено пользователем на {i-1}/{len(top)}")
            _deep.progress["status"] = "stopped"
            return

        _deep.progress["label"] = f"Анализ кода {code} ({i}/{len(top)})…"
        affected = len(devs)
        total_occ = code_total[code]
        sev = code_severity.get(code, "")

        # Try to pull descriptions from events in factor_store
        samples: list[str] = []
        if fs is not None:
            for did in list(devs)[:3]:
                for ev in fs.get_events(did, window_days=365):
                    if ev.error_code == code and ev.error_description:
                        samples.append(ev.error_description[:200])
                        break

        description = samples[0] if samples else ""

        try:
            analysis = agent.analyze_mass_error(
                error_code=code,
                description=description,
                affected_count=affected,
                total_occurrences=total_occ,
                fleet_total=fleet_total,
                severity=sev,
                sample_device_ids=list(devs)[:5],
                sample_descriptions=samples[:3],
            )
            output[code] = analysis  # mutates the dict UI has reference to
            tag = "🔴 Системно" if analysis.is_systemic else "🟡 Точечно"
            if analysis.error:
                _deep.log(f"[{i}/{len(top)}] **{code}** — ошибка LLM: {analysis.error}")
            else:
                _deep.log(f"[{i}/{len(top)}] **{code}** {tag} ({affected} уст.)")
        except Exception as exc:  # noqa: BLE001
            _deep.log(f"[{i}/{len(top)}] **{code}** — исключение: {exc}")

        _deep.progress["done"] = i

    _deep.progress["status"] = "complete"
    _deep.log(f"✅ Готово: проанализировано {len(output)} кодов")


def _run_red_zone_worker(results, fs, weights, endpoint_name: str) -> None:
    """Background worker: run full agent.run_batch on every red-zone device.

    All session-state objects must be passed from the main thread.
    ``endpoint_name`` picks which LLM provider runs the analysis.
    """
    _deep.progress.clear()
    _deep.progress["kind"] = "red"
    _deep.progress["status"] = "running"
    _deep.progress["should_stop"] = False
    _deep.log_lines.clear()
    _deep.log("Старт глубокого анализа красной зоны…")

    red_devices = [r for r in results if getattr(r.zone, "value", r.zone) == "red"]

    if not red_devices:
        _deep.log("⚠️ Нет устройств в красной зоне")
        _deep.progress["status"] = "complete"
        return

    _deep.log(f"🧠 Провайдер: {endpoint_name}")
    agent = _build_agent_for_deep_analysis(fs, weights, endpoint_name)
    if agent is None:
        _deep.progress["status"] = "error"
        return

    from data_io.models import DeepDeviceAnalysis

    output: dict = {}
    # Publish the output dict reference immediately so the UI can render
    # partial results while the worker is still running.
    _deep.progress["_result_red"] = output
    _deep.progress["total"] = len(red_devices)
    _deep.progress["done"] = 0

    for i, r in enumerate(red_devices, 1):
        if _deep.progress.get("should_stop"):
            _deep.log(f"⏹ Остановлено пользователем на {i-1}/{len(red_devices)}")
            _deep.progress["status"] = "stopped"
            return

        did = r.device_id
        _deep.progress["label"] = f"Анализ {did} ({i}/{len(red_devices)})…"

        try:
            analysis = agent.analyze_device_deep(
                device_id=did,
                health_result=r,
                factor_store=fs,
                weights_profile=weights,
            )
            output[did] = analysis
            if analysis.error:
                _deep.log(f"[{i}/{len(red_devices)}] **{did}** — ошибка LLM: {analysis.error}")
            else:
                _deep.log(
                    f"[{i}/{len(red_devices)}] **{did}** → H_lite={r.health_index} "
                    f"/ H_llm={analysis.health_index_llm} ({analysis.duration_ms} мс)"
                )
        except Exception as exc:  # noqa: BLE001
            output[did] = DeepDeviceAnalysis(
                device_id=did,
                health_index_original=r.health_index,
                analyzed_at=datetime.now(UTC),
                error=f"{type(exc).__name__}: {exc}",
            )
            _deep.log(f"[{i}/{len(red_devices)}] **{did}** — исключение: {exc}")

        _deep.progress["done"] = i

    _deep.progress["status"] = "complete"
    _deep.log(f"✅ Готово: проанализировано {len(output)} устройств")


def _snapshot_session_for_worker():
    """Capture session state in the main thread for the worker to use.

    Note: ``endpoint_name`` is captured HERE because st.session_state is
    only accessible from the Streamlit main thread — the worker threads
    cannot read ``get_active_llm_endpoint()``.
    """
    from data_io.models import WeightsProfile
    results = get_current_health_results()
    fs = get_current_factor_store()
    weights = get_active_weights_profile()
    if weights is None:
        weights = WeightsProfile(profile_name="default")
    endpoint_name = get_active_llm_endpoint()
    return results, fs, weights, endpoint_name


def _start_mass_error_worker() -> None:
    results, fs, weights, endpoint_name = _snapshot_session_for_worker()
    _deep.thread = threading.Thread(
        target=_run_mass_error_worker,
        args=(results, fs, weights, endpoint_name),
        daemon=True,
    )
    _deep.thread.start()


def _start_red_zone_worker() -> None:
    results, fs, weights, endpoint_name = _snapshot_session_for_worker()
    _deep.thread = threading.Thread(
        target=_run_red_zone_worker,
        args=(results, fs, weights, endpoint_name),
        daemon=True,
    )
    _deep.thread.start()


st.subheader("🧠 Глубокий LLM-анализ")

red_count = sum(1 for d in report.devices if d.zone == "red")
is_running = _deep.thread is not None and _deep.thread.is_alive()

col_a, col_b = st.columns(2)
with col_a:
    btn_red_label = f"🔬 Красная зона ({red_count} уст., ~{max(1, red_count * 15 // 60)} мин)"
    if st.button(
        btn_red_label,
        disabled=(red_count == 0 or is_running),
        use_container_width=True,
        help="Full agent loop с reflection для каждого устройства в красной зоне. "
             "LLM видит все события, ресурсы, метаданные. Выдаёт root cause, "
             "рекомендуемое действие и полное объяснение.",
        key="btn_red_zone",
    ):
        _start_red_zone_worker()
        st.rerun()

with col_b:
    if st.button(
        "🔥 Массовые ошибки (top-10)",
        disabled=is_running,
        use_container_width=True,
        help="Top-10 кодов по числу затронутых устройств. "
             "LLM определяет системная это проблема или локальная + рекомендация.",
        key="btn_mass",
    ):
        _start_mass_error_worker()
        st.rerun()

# Running indicator + progress
if is_running:
    step_label = _deep.progress.get("label", "Обработка…")
    row = st.columns([4, 1])
    with row[0]:
        st.info(f"⏳ {step_label}")
    with row[1]:
        if st.button(
            "⏹ Остановить",
            disabled=bool(_deep.progress.get("should_stop")),
            use_container_width=True,
            key="btn_stop_deep",
        ):
            _deep.progress["should_stop"] = True
            _deep.log("⏹ Запрошена остановка — завершу текущий шаг и выйду")
            st.rerun()

    done = _deep.progress.get("done", 0)
    total = _deep.progress.get("total", 1)
    if total:
        st.progress(min(1.0, done / total), text=f"{done}/{total}")
    _render_deep_log(running=True)

# Transition: thread finished → persist final results in session state
if _deep.thread is not None and not _deep.thread.is_alive():
    status = _deep.progress.get("status")
    if status in ("complete", "stopped"):
        mass = _deep.progress.get("_result_mass")
        if mass:
            set_mass_error_analyses(mass)
        red = _deep.progress.get("_result_red")
        if red:
            set_deep_device_analyses(red)
    _deep.thread = None

# Persistent log when idle
if not is_running and _deep.log_lines:
    _render_deep_log(running=False)

# While running, pull partial results from module-level progress dict;
# once idle, fall back to persisted session state.
if is_running:
    mass_analyses = _deep.progress.get("_result_mass") or {}
    deep_analyses = _deep.progress.get("_result_red") or {}
else:
    mass_analyses = get_mass_error_analyses()
    deep_analyses = get_deep_device_analyses()

if mass_analyses:
    fleet_total = report.fleet_summary.total_devices or 1
    with st.expander(f"🔥 Массовые ошибки — LLM-взгляд ({len(mass_analyses)})", expanded=True):
        ordered = sorted(
            mass_analyses.values(),
            key=lambda a: a.affected_device_count, reverse=True,
        )
        for a in ordered:
            with st.container(border=True):
                pct = a.affected_device_count * 100 // fleet_total if fleet_total else 0
                reps_per_dev = a.total_occurrences // max(1, a.affected_device_count)

                if a.error:
                    st.markdown(
                        f"### {a.error_code} — ⚠️ LLM-ошибка  \n"
                        f"{a.description or '(нет описания)'}"
                    )
                    m_cols = st.columns(3)
                    m_cols[0].metric("Устройств", f"{a.affected_device_count}", delta=f"{pct}% парка")
                    m_cols[1].metric("Событий", f"{a.total_occurrences}")
                    m_cols[2].metric("Ср. повторов / уст.", f"{reps_per_dev}")
                    st.caption(f"Детали: {a.error}")
                    continue

                tag = "🔴 Системная" if a.is_systemic else "🟡 Точечная"
                scale = (
                    "в масштабе парка"
                    if pct >= 20
                    else ("на значимой доле парка" if pct >= 5 else "на отдельных устройствах")
                )
                st.markdown(
                    f"### {a.error_code} — {tag} ({scale})"
                )

                m_cols = st.columns(3)
                m_cols[0].metric("Устройств", f"{a.affected_device_count}", delta=f"{pct}% парка")
                m_cols[1].metric("Событий всего", f"{a.total_occurrences}")
                m_cols[2].metric("Ср. повторов / уст.", f"{reps_per_dev}")

                # ── 6 содержательных секций ─────────────────────────────
                if a.what_is_this:
                    st.markdown(f"📋 **Что это:** {a.what_is_this}")
                elif a.description:
                    st.markdown(f"📋 **Что это:** {a.description}")

                if a.why_this_pattern:
                    st.markdown(f"🔍 **Почему массово:** {a.why_this_pattern}")

                if a.business_impact:
                    st.markdown(f"⚠️ **Влияние:** {a.business_impact}")

                if a.immediate_action:
                    st.markdown(f"⚡ **Сейчас:** {a.immediate_action}")

                if a.long_term_action:
                    st.markdown(f"🧭 **На будущее:** {a.long_term_action}")

                if a.indicators_to_watch:
                    st.markdown("👀 **Мониторить:**")
                    for ind in a.indicators_to_watch:
                        st.markdown(f"- {ind}")

                # Legacy fallback — если новых полей нет, старое поведение
                if not a.what_is_this and a.likely_cause:
                    st.markdown(f"**Причина:** {a.likely_cause}")
                if not a.immediate_action and a.recommended_action:
                    st.markdown(f"**Что делать:** {a.recommended_action}")

                if a.explanation:
                    with st.expander("📖 Полное объяснение LLM", expanded=False):
                        st.markdown(a.explanation)

if deep_analyses:
    _running_red = is_running and _deep.progress.get("kind") == "red"
    with st.expander(
        f"🔬 Глубокий анализ красной зоны ({len(deep_analyses)})",
        expanded=_running_red,
    ):
        ok_count = sum(1 for a in deep_analyses.values() if not a.error)
        err_count = len(deep_analyses) - ok_count
        st.caption(f"Успешно: {ok_count} · Ошибок: {err_count}")

        for did in sorted(deep_analyses.keys()):
            a = deep_analyses[did]
            with st.container(border=True):
                if a.error:
                    st.markdown(f"**{did}** — ⚠️ {a.error}")
                    continue
                verdict_badge = (
                    "✅" if a.reflection_verdict == "approved" else "⚠️"
                )
                st.markdown(
                    f"**{did}** {verdict_badge} — "
                    f"H_lite={a.health_index_original} / "
                    f"H_llm={a.health_index_llm} · "
                    f"{a.llm_calls} LLM · {a.duration_ms / 1000:.1f}s"
                )
                if a.root_cause:
                    st.markdown(f"**Корневая причина:** {a.root_cause}")
                if a.recommended_action:
                    st.markdown(f"**Действие:** {a.recommended_action}")
                if a.related_codes:
                    st.caption(f"Коды: {', '.join(a.related_codes[:6])}")
                if a.explanation:
                    with st.expander("Объяснение LLM", expanded=False):
                        st.markdown(a.explanation)

# Auto-refresh while the worker is running, after rendering partial results.
if is_running:
    time.sleep(2)
    st.rerun()


# ── Block 5: Device table ──────────────────────────────────────────────────

st.subheader("Таблица устройств")

if report.devices:
    rows = []
    for d in report.devices:
        rows.append(
            {
                "ID": d.device_id,
                "Модель": d.model or "—",
                "Локация": d.location or "—",
                "Индекс": d.health_index,
                "Уверенность": f"{d.confidence:.0%}",
                "Зона": _ZONE_LABELS.get(d.zone, d.zone),
                "Проблема": d.top_problem_tag or "—",
                "На проверку": "да" if d.flag_for_review else "",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Индекс": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d"
            ),
        },
    )
else:
    st.caption("Нет данных по устройствам.")
