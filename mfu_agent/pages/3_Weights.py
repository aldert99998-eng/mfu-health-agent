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
    claim_bg_results_if_any,
    get_active_weights_profile,
    get_baseline_health_results,
    get_current_factor_store,
    get_current_health_results,
    get_raw_factors,
    set_active_weights_profile,
    set_current_health_results,
)

st.header("Управление весами")

claim_bg_results_if_any()

with st.expander("📐 Как считается индекс здоровья", expanded=False):
    st.markdown(
        "Индекс считается детерминированно по формуле из архитектуры (Track A). "
        "Каждый фактор (уникальный код ошибки у устройства) даёт штраф **S · R · C · A**, "
        "штрафы суммируются, вычитаются из 100."
    )
    st.latex(r"H \;=\; \max\!\left(1,\; 100 - \sum_i S_i \cdot R_i \cdot C_i \cdot A_i\right)")

    st.markdown("**Компоненты штрафа:**")
    st.latex(r"R \;=\; \min\!\left(R_{max},\; 1 + \log_{base}(n)\right)")
    st.caption("R — множитель повторяемости, где n — число повторов кода в окне.")
    st.latex(r"C \;=\; \min\!\left(C_{max},\; \prod_k m_k\right)")
    st.caption("C — контекстный множитель (низкий тонер, простой и т.п.).")
    st.latex(r"A \;=\; e^{-d/\tau}")
    st.caption("A — затухание по возрасту d (дней). τ = период полузатухания.")

    st.markdown("**Уверенность:**")
    st.latex(r"\text{Conf} \;=\; \max\!\left(\text{min}_{conf},\; \prod_j p_j\right)")
    st.caption("Conf — произведение штрафов за отсутствие данных, не ниже min_conf.")

    st.markdown("**Зоны:**")
    st.markdown(
        "- 🟢 **Зелёная**: H ≥ green_threshold\n"
        "- 🟡 **Жёлтая**: red_threshold ≤ H < green_threshold\n"
        "- 🔴 **Красная**: H < red_threshold"
    )

_DRAFT_KEY = "weights_profile_draft"
_VERSION_KEY = "weights_profile_version"

_ver = st.session_state.setdefault(_VERSION_KEY, 0)


def _k(name: str) -> str:
    """Widget key suffixed with version so reload resets widget state."""
    return f"{name}_v{_ver}"

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
        st.session_state[_VERSION_KEY] = _ver + 1
        st.rerun()

st.divider()

# ── Severity sliders ─────────────────────────────────────────────────────

st.subheader("Severity — базовый штраф")
st.caption("S — сколько единиц штрафа даёт **одна** ошибка каждого уровня (до учёта повторов, контекста и возраста).")

sev = draft.severity
sev_cols = st.columns(5)
new_critical = sev_cols[0].slider(
    "Critical", 0.0, 100.0, sev.critical, 1.0, key=_k("sev_critical"),
    help="Базовый штраф S для ошибок Critical (отказ узла, требуется сервис). "
         "Один такой фактор с R=1, C=1, A=1 отнимет этот % от индекса.",
)
new_high = sev_cols[1].slider(
    "High", 0.0, 60.0, sev.high, 1.0, key=_k("sev_high"),
    help="Базовый штраф S для ошибок High (серьёзная проблема расходников или бумажного тракта).",
)
new_medium = sev_cols[2].slider(
    "Medium", 0.0, 30.0, sev.medium, 0.5, key=_k("sev_medium"),
    help="Базовый штраф S для ошибок Medium (умеренные неполадки, требуют внимания).",
)
new_low = sev_cols[3].slider(
    "Low", 0.0, 15.0, sev.low, 0.5, key=_k("sev_low"),
    help="Базовый штраф S для ошибок Low (мелкие инциденты: открыта крышка, пустой лоток).",
)
new_info = sev_cols[4].slider(
    "Info", 0.0, 5.0, sev.info, 0.5, key=_k("sev_info"),
    help="Базовый штраф S для информационных событий. Обычно 0 — не влияют на индекс.",
)

draft.severity.critical = new_critical
draft.severity.high = new_high
draft.severity.medium = new_medium
draft.severity.low = new_low
draft.severity.info = new_info

# ── Repeatability ─────────────────────────────────────────────────────────

st.subheader("Repeatability — повторяемость")
st.caption("R — насколько усиливается штраф за повторение одного и того же кода: R = min(R_max, 1 + log_base(n)).")

rep_cols = st.columns(3)
draft.repeatability.base = rep_cols[0].slider(
    "Основание log", 2, 10, draft.repeatability.base, key=_k("rep_base"),
    help="Основание логарифма. Чем больше — тем медленнее R растёт при увеличении n. "
         "base=2: одно повторение → R=1, 4 повтора → R=3, 16 → R=5. "
         "base=10: 10 повторов → R=2, 100 → R=3.",
)
draft.repeatability.max_value = rep_cols[1].slider(
    "Потолок R", 1.0, 10.0, draft.repeatability.max_value, 0.5, key=_k("rep_max"),
    help="Максимум R. Защита от разрастания штрафа при тысячах повторений одной ошибки. "
         "Например, 5.0 = штраф вырастет максимум в 5 раз относительно одной ошибки.",
)
draft.repeatability.window_days = rep_cols[2].slider(
    "Окно (дней)", 7, 60, draft.repeatability.window_days, key=_k("rep_window"),
    help="Окно подсчёта повторений. Повторы того же кода старше этого окна не учитываются.",
)

# ── Context modifiers ─────────────────────────────────────────────────────

st.subheader("Context — контекстные модификаторы")
st.caption("C — перемножение множителей, применимых к конкретному устройству (низкий тонер, простой и т.п.): C = min(C_max, Π m_k).")

ctx = draft.context
ctx_cols = st.columns(2)
with ctx_cols[0]:
    for name, mod in ctx.modifiers.items():
        mod.multiplier = st.slider(
            f"{name} (множитель)",
            1.0, 3.0, mod.multiplier, 0.1,
            key=_k(f"ctx_mult_{name}"),
            help=f"Во сколько раз увеличить штраф, если активен модификатор «{name}». "
                 "1.0 = без влияния, 2.0 = удвоение штрафа.",
        )
        if mod.threshold is not None:
            mod.threshold = st.slider(
                f"{name} (порог)",
                0.0, 100.0, mod.threshold, 1.0,
                key=_k(f"ctx_thresh_{name}"),
                help=f"Порог срабатывания модификатора «{name}» "
                     "(например: уровень тонера ниже порога → модификатор активен).",
            )

with ctx_cols[1]:
    ctx.max_value = st.slider(
        "Потолок C", 1.0, 3.0, ctx.max_value, 0.1, key=_k("ctx_max"),
        help="Максимум контекстного множителя. Защита от комбинаторного взрыва "
             "при срабатывании нескольких модификаторов одновременно.",
    )

# ── Age decay ─────────────────────────────────────────────────────────────

st.subheader("Age — затухание по давности")
st.caption("A — насколько ослабевает штраф у старых событий: A = exp(-d/τ), где d — возраст события в днях.")

age_cols = st.columns(2)
draft.age.tau_days = age_cols[0].slider(
    "τ (полураспад, дней)", 3, 60, draft.age.tau_days, key=_k("age_tau"),
    help="Постоянная затухания. Событие возрастом τ дней имеет A ≈ 0.37. "
         "τ=14 → событие недельной давности A ≈ 0.61, двухнедельное A ≈ 0.37, месячной A ≈ 0.14. "
         "Больше τ → старые события дольше влияют на индекс.",
)
draft.age.window_days = age_cols[1].slider(
    "Окно анализа (дней)", 7, 90, draft.age.window_days, key=_k("age_window"),
    help="Максимальный возраст событий в расчёте. События старше выбрасываются ещё до применения A. "
         "Служит фильтром отсечения архивных данных.",
)

# ── Confidence ────────────────────────────────────────────────────────────

st.subheader("Confidence — штрафы уверенности")
st.caption("Conf — доверие к оценке индекса. Каждый негативный фактор домножает Conf на свой множитель (< 1.0 → уверенность падает).")

conf = draft.confidence
conf_cols = st.columns(3)
conf.min_value = conf_cols[0].slider(
    "Минимум", 0.05, 0.5, conf.min_value, 0.05, key=_k("conf_min"),
    help="Нижняя граница Conf. Даже при срабатывании всех негативных факторов уверенность не упадёт ниже.",
)

pen = conf.penalties
pen.rag_not_found = conf_cols[1].slider(
    "RAG не нашёл", 0.3, 1.0, pen.rag_not_found, 0.05, key=_k("conf_rag"),
    help="Множитель Conf за каждый код ошибки, которого нет в сервисной документации RAG. "
         "Применяется в степени — если не нашлось 3 кода, штраф = x³.",
)
pen.missing_resources = conf_cols[2].slider(
    "Нет ресурсов", 0.3, 1.0, pen.missing_resources, 0.05, key=_k("conf_res"),
    help="Множитель Conf, если у устройства нет снимка уровней расходников (тонер, барабан и т.п.).",
)
pen_cols2 = st.columns(4)
pen.missing_model = pen_cols2[0].slider(
    "Нет модели", 0.3, 1.0, pen.missing_model, 0.05, key=_k("conf_model"),
    help="Множитель Conf, если модель устройства не определена — нельзя уверенно классифицировать severity.",
)
pen.abnormal_daily_jump = pen_cols2[1].slider(
    "Скачок >30", 0.3, 1.0, pen.abnormal_daily_jump, 0.05, key=_k("conf_jump"),
    help="Множитель Conf, если индекс прыгнул больше чем на 30 пунктов за день — подозрительная динамика.",
)
pen.anomalous_event_count = pen_cols2[2].slider(
    "Аномальные события", 0.3, 1.0, pen.anomalous_event_count, 0.05, key=_k("conf_anomal"),
    help="Множитель Conf, если количество событий у устройства резко выше типичного — возможен шум в данных.",
)
pen.no_events_and_no_resources = pen_cols2[3].slider(
    "Нет событий и ресурсов", 0.3, 1.0, pen.no_events_and_no_resources, 0.05, key=_k("conf_nodata"),
    help="Множитель Conf, когда нет ни событий, ни снимка. Оценка строится на одной только репутации устройства.",
)

# ── Zones & misc ──────────────────────────────────────────────────────────

st.subheader("Зоны и режим")
st.caption("Пороги разделяют парк на зоны для отчёта и обработки silent-устройств (без событий).")

zone_cols = st.columns(3)
draft.zones.green_threshold = zone_cols[0].slider(
    "Зелёная ≥", 50, 100, draft.zones.green_threshold, key=_k("zone_green"),
    help="Порог зелёной зоны: устройства с H ≥ этого значения считаются здоровыми.",
)
draft.zones.red_threshold = zone_cols[1].slider(
    "Красная <", 10, draft.zones.green_threshold - 1, draft.zones.red_threshold, key=_k("zone_red"),
    help="Порог красной зоны: устройства с H < этого значения требуют срочного вмешательства. "
         "Между red_threshold и green_threshold — жёлтая зона.",
)
draft.silent_device_mode = SilentDeviceMode(
    zone_cols[2].selectbox(
        "Режим тихих устройств",
        [m.value for m in SilentDeviceMode],
        index=[m.value for m in SilentDeviceMode].index(draft.silent_device_mode.value),
        key=_k("silent_mode"),
        help="Как обрабатывать устройства БЕЗ событий и снимков ресурсов.\n\n"
             "• **data_quality** — H=100, Conf=0.9 с пометкой «нет данных» (осторожный режим).\n"
             "• **optimistic** — H=100, Conf=1.0 (предполагаем всё хорошо).\n"
             "• **carry_forward** — берёт последнюю известную оценку из истории (если есть).",
    )
)

st.divider()

# ── Save profile to YAML ──────────────────────────────────────────────────

save_cols = st.columns([3, 1])
save_name = save_cols[0].text_input(
    "Имя профиля для сохранения",
    value=draft.profile_name,
    key="weights_save_name",
    help="Сохранить текущие значения как YAML-профиль. "
         "Если имя совпадает с существующим — перезапишет. Новое имя — создаст новый профиль.",
)
if save_cols[1].button("💾 Сохранить профиль", use_container_width=True):
    save_name_clean = (save_name or "").strip()
    if not save_name_clean:
        st.error("Укажи имя профиля.")
    else:
        draft.profile_name = save_name_clean
        path = cm.save_weights_profile(draft)
        set_active_weights_profile(draft)
        st.session_state["_weights_save_flash"] = str(path)
        st.rerun()

if "_weights_save_flash" in st.session_state:
    st.success(f"Профиль сохранён: `{st.session_state.pop('_weights_save_flash')}`")

st.divider()

# ── Apply button + live before/after summary ──────────────────────────────

raw_factors_map = get_raw_factors()
baseline_results = get_baseline_health_results()

if not raw_factors_map:
    st.info(
        "Для пересчёта нужны результаты анализа. "
        "Перейдите на страницу **Загрузка данных** и загрузите файл."
    )
elif st.button("Применить веса и пересчитать", type="primary", use_container_width=True):
    from datetime import UTC, datetime

    from data_io.models import ConfidenceFactors, Factor, SeverityLevel
    from tools.calculator import (
        calculate_health_index,
        compute_A,
        compute_C,
        compute_R,
    )

    fs_for_ref = get_current_factor_store()
    now = fs_for_ref.reference_time if fs_for_ref else datetime.now(UTC)
    new_results = []

    for did, calc_args in raw_factors_map.items():
        raw_factors = calc_args.get("factors", [])
        cf_data = calc_args.get("confidence_factors", {})

        built_factors: list[Factor] = []
        for fi in raw_factors:
            try:
                severity = SeverityLevel(fi["severity_level"])
            except (ValueError, KeyError):
                severity = SeverityLevel.MEDIUM

            S = getattr(draft.severity, severity.value.lower(), 10.0)
            ts = datetime.fromisoformat(fi["event_timestamp"])
            age_days = max(0, (now - ts).days)

            R = compute_R(
                fi.get("n_repetitions", 1),
                draft.repeatability.base,
                draft.repeatability.max_value,
            )

            modifiers: list[float] = []
            for mod_name in fi.get("applicable_modifiers", []):
                mod = draft.context.modifiers.get(mod_name)
                if mod:
                    modifiers.append(mod.multiplier)
            C = compute_C(modifiers, draft.context.max_value)

            A = compute_A(age_days, draft.age.tau_days)

            built_factors.append(Factor(
                error_code=fi.get("error_code", ""),
                severity_level=severity,
                S=S,
                n_repetitions=fi.get("n_repetitions", 1),
                R=R,
                C=C,
                A=A,
                event_timestamp=ts,
                age_days=age_days,
                applicable_modifiers=fi.get("applicable_modifiers", []),
                source=fi.get("source"),
            ))

        confidence_factors = ConfidenceFactors(
            rag_missing_count=cf_data.get("rag_missing_count", 0),
            missing_resources=cf_data.get("missing_resources", False),
            missing_model=cf_data.get("missing_model", False),
        )

        result = calculate_health_index(
            factors=built_factors,
            confidence_factors=confidence_factors,
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

if baseline_results:
    st.subheader("Сводка «было / стало»")

    current_results = get_current_health_results()

    old_map = {r.device_id: r for r in baseline_results}
    new_map = {r.device_id: r for r in current_results}

    old_avg = sum(r.health_index for r in baseline_results) / len(baseline_results) if baseline_results else 0
    new_avg = sum(r.health_index for r in current_results) / len(current_results) if current_results else 0

    def _count_zones(results: list[HealthResult]) -> dict[str, int]:
        z: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
        for r in results:
            z[r.zone] = z.get(r.zone, 0) + 1
        return z

    old_zones = _count_zones(baseline_results)
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
