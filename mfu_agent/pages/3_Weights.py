"""Страница 2 — Управление весами формулы здоровья."""

from __future__ import annotations

import streamlit as st

from config.loader import get_config_manager
from data_io.models import (
    HealthResult,
    SilentDeviceMode,
    WeightsProfile,
)
from state.session import (
    get_active_weights_profile,
    get_current_factor_store,
    get_current_health_results,
    set_active_weights_profile,
    set_current_health_results,
)

st.header("Управление весами")

_DRAFT_KEY = "weights_profile_draft"

# ── Load initial draft from active profile or config ───────────────────────

def _load_default_profile() -> WeightsProfile:
    active = get_active_weights_profile()
    if active is not None:
        return active
    cm = get_config_manager()
    profiles = cm.list_profiles()
    if profiles:
        return cm.load_weights(profiles[0])
    return WeightsProfile(profile_name="default")


def _get_draft() -> WeightsProfile:
    if _DRAFT_KEY not in st.session_state:
        st.session_state[_DRAFT_KEY] = _load_default_profile()
    return st.session_state[_DRAFT_KEY]  # type: ignore[no-any-return]


def _set_draft(profile: WeightsProfile) -> None:
    st.session_state[_DRAFT_KEY] = profile


draft = _get_draft()

# ── Profile selector ──────────────────────────────────────────────────────

cm = get_config_manager()
profiles = cm.list_profiles()

if profiles:
    col_prof, col_load = st.columns([3, 1])
    selected = col_prof.selectbox(
        "Профиль",
        profiles,
        index=profiles.index(draft.profile_name) if draft.profile_name in profiles else 0,
    )
    if col_load.button("Загрузить", use_container_width=True):
        _set_draft(cm.load_weights(selected))
        st.rerun()

st.divider()

# ── Severity sliders ─────────────────────────────────────────────────────

st.subheader("Severity — базовый штраф")

sev = draft.severity
sev_cols = st.columns(5)
new_critical = sev_cols[0].slider("Critical", 0.0, 100.0, sev.critical, 1.0, key="sev_critical")
new_high = sev_cols[1].slider("High", 0.0, 60.0, sev.high, 1.0, key="sev_high")
new_medium = sev_cols[2].slider("Medium", 0.0, 30.0, sev.medium, 0.5, key="sev_medium")
new_low = sev_cols[3].slider("Low", 0.0, 15.0, sev.low, 0.5, key="sev_low")
new_info = sev_cols[4].slider("Info", 0.0, 5.0, sev.info, 0.5, key="sev_info")

draft.severity.critical = new_critical
draft.severity.high = new_high
draft.severity.medium = new_medium
draft.severity.low = new_low
draft.severity.info = new_info

# ── Repeatability ─────────────────────────────────────────────────────────

st.subheader("Repeatability — повторяемость")

rep_cols = st.columns(3)
draft.repeatability.base = rep_cols[0].slider(
    "Основание log", 2, 10, draft.repeatability.base, key="rep_base",
)
draft.repeatability.max_value = rep_cols[1].slider(
    "Потолок R", 1.0, 10.0, draft.repeatability.max_value, 0.5, key="rep_max",
)
draft.repeatability.window_days = rep_cols[2].slider(
    "Окно (дней)", 7, 60, draft.repeatability.window_days, key="rep_window",
)

# ── Context modifiers ─────────────────────────────────────────────────────

st.subheader("Context — контекстные модификаторы")

ctx = draft.context
ctx_cols = st.columns(2)
with ctx_cols[0]:
    for name, mod in ctx.modifiers.items():
        mod.multiplier = st.slider(
            f"{name} (множитель)",
            1.0, 3.0, mod.multiplier, 0.1,
            key=f"ctx_mult_{name}",
        )
        if mod.threshold is not None:
            mod.threshold = st.slider(
                f"{name} (порог)",
                0.0, 100.0, mod.threshold, 1.0,
                key=f"ctx_thresh_{name}",
            )

with ctx_cols[1]:
    ctx.max_value = st.slider(
        "Потолок C", 1.0, 3.0, ctx.max_value, 0.1, key="ctx_max",
    )

# ── Age decay ─────────────────────────────────────────────────────────────

st.subheader("Age — затухание по давности")

age_cols = st.columns(2)
draft.age.tau_days = age_cols[0].slider(
    "τ (полураспад, дней)", 3, 60, draft.age.tau_days, key="age_tau",
)
draft.age.window_days = age_cols[1].slider(
    "Окно анализа (дней)", 7, 90, draft.age.window_days, key="age_window",
)

# ── Confidence ────────────────────────────────────────────────────────────

st.subheader("Confidence — штрафы уверенности")

conf = draft.confidence
conf_cols = st.columns(3)
conf.min_value = conf_cols[0].slider(
    "Минимум", 0.05, 0.5, conf.min_value, 0.05, key="conf_min",
)

pen = conf.penalties
pen.rag_not_found = conf_cols[1].slider(
    "RAG не нашёл", 0.3, 1.0, pen.rag_not_found, 0.05, key="conf_rag",
)
pen.missing_resources = conf_cols[2].slider(
    "Нет ресурсов", 0.3, 1.0, pen.missing_resources, 0.05, key="conf_res",
)
pen_cols2 = st.columns(4)
pen.missing_model = pen_cols2[0].slider(
    "Нет модели", 0.3, 1.0, pen.missing_model, 0.05, key="conf_model",
)
pen.abnormal_daily_jump = pen_cols2[1].slider(
    "Скачок >30", 0.3, 1.0, pen.abnormal_daily_jump, 0.05, key="conf_jump",
)
pen.anomalous_event_count = pen_cols2[2].slider(
    "Аномальные события", 0.3, 1.0, pen.anomalous_event_count, 0.05, key="conf_anomal",
)
pen.no_events_and_no_resources = pen_cols2[3].slider(
    "Нет событий и ресурсов", 0.3, 1.0, pen.no_events_and_no_resources, 0.05, key="conf_nodata",
)

# ── Zones & misc ──────────────────────────────────────────────────────────

st.subheader("Зоны и режим")

zone_cols = st.columns(3)
draft.zones.green_threshold = zone_cols[0].slider(
    "Зелёная ≥", 50, 100, draft.zones.green_threshold, key="zone_green",
)
draft.zones.red_threshold = zone_cols[1].slider(
    "Красная <", 10, draft.zones.green_threshold - 1, draft.zones.red_threshold, key="zone_red",
)
draft.silent_device_mode = SilentDeviceMode(
    zone_cols[2].selectbox(
        "Режим тихих устройств",
        [m.value for m in SilentDeviceMode],
        index=[m.value for m in SilentDeviceMode].index(draft.silent_device_mode.value),
        key="silent_mode",
    )
)

st.divider()

# ── Apply button + live before/after summary ──────────────────────────────

fs = get_current_factor_store()
old_results = get_current_health_results()

if not fs:
    st.info(
        "Для пересчёта нужен загруженный FactorStore. "
        "Перейдите на страницу **Загрузка данных** и загрузите файл."
    )
elif st.button("Применить веса и пересчитать", type="primary", use_container_width=True):
    from data_io.models import ConfidenceFactors
    from tools.calculator import calculate_health_index

    device_ids = fs.list_devices()
    new_results = []
    for did in device_ids:
        factors = getattr(fs, "_factors", {}).get(did, [])
        cf = ConfidenceFactors()
        result = calculate_health_index(
            factors=factors,
            confidence_factors=cf,
            weights=draft,
            device_id=did,
            silent_device_mode=draft.silent_device_mode.value,
        )
        new_results.append(result)

    set_current_health_results(new_results)
    set_active_weights_profile(draft)
    st.success(f"Пересчитано {len(new_results)} устройств.")
    st.rerun()

# ── Live before/after comparison ──────────────────────────────────────────

if old_results:
    st.subheader("Сводка «было / стало»")

    active = get_active_weights_profile()
    current_results = get_current_health_results()

    old_map = {r.device_id: r for r in old_results}
    new_map = {r.device_id: r for r in current_results}

    old_avg = sum(r.health_index for r in old_results) / len(old_results) if old_results else 0
    new_avg = sum(r.health_index for r in current_results) / len(current_results) if current_results else 0

    def _count_zones(results: list[HealthResult]) -> dict[str, int]:
        z: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
        for r in results:
            z[r.zone] = z.get(r.zone, 0) + 1
        return z

    old_zones = _count_zones(old_results)
    new_zones = _count_zones(current_results)

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("**Было**")
        st.metric("Средний индекс", f"{old_avg:.1f}")
        st.caption(
            f"🟢 {old_zones['green']}  🟡 {old_zones['yellow']}  🔴 {old_zones['red']}"
        )

    with col_after:
        st.markdown("**Стало**")
        delta = new_avg - old_avg
        st.metric("Средний индекс", f"{new_avg:.1f}", delta=f"{delta:+.1f}")
        st.caption(
            f"🟢 {new_zones['green']}  🟡 {new_zones['yellow']}  🔴 {new_zones['red']}"
        )

    changed = []
    for did in sorted(old_map.keys() & new_map.keys()):
        old_h = old_map[did].health_index
        new_h = new_map[did].health_index
        if old_h != new_h:
            changed.append({"ID": did, "Было": old_h, "Стало": new_h, "Δ": new_h - old_h})

    if changed:
        import pandas as pd
        st.markdown(f"**Изменилось {len(changed)} устройств:**")
        st.dataframe(pd.DataFrame(changed), use_container_width=True, hide_index=True)
    elif current_results:
        st.caption("Индексы не изменились.")
