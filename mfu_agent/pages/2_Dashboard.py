"""Страница 1 — Дашборд здоровья парка МФУ."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from state.session import get_current_report

st.header("Дашборд здоровья парка")

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


# ── Block 4: Top patterns ─────────────────────────────────────────────────

st.subheader("Ключевые паттерны")

_PATTERN_BADGES = {
    "mass_issue": ":red[Массовая проблема]",
    "location_cluster": ":orange[Кластер по локации]",
    "critical_single": ":violet[Критическое устройство]",
    "model_defect": ":blue[Дефект модели]",
    "resource_depletion": ":gray[Расход ресурсов]",
}

if report.top_patterns:
    for pat in report.top_patterns:
        badge = _PATTERN_BADGES.get(pat.pattern_type, pat.pattern_type)
        with st.expander(f"{badge}  {pat.title}", expanded=True):
            st.markdown(pat.explanation)
            mcols = st.columns(2)
            mcols[0].metric("Средний индекс", f"{pat.average_index:.1f}")
            mcols[1].metric("Устройств", len(pat.affected_device_ids))
            if pat.affected_device_ids:
                st.caption("Устройства: " + ", ".join(pat.affected_device_ids[:10]))
                if len(pat.affected_device_ids) > 10:
                    st.caption(f"… и ещё {len(pat.affected_device_ids) - 10}")
else:
    st.caption("Паттерны не обнаружены.")


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
