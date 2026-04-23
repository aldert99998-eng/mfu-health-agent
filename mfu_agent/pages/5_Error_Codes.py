"""Страница 5 — Справочник ошибок устройств (Xerox / Lexmark / Ricoh)."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from error_codes import (
    SUPPORTED_VENDORS,
    AliasConflict,
    Conflict,
    ErrorCode,
    ModelErrorCodes,
    ParseError,
    add_alias,
    delete,
    find_conflicts,
    get_canonical_for_alias,
    list_models,
    load_codes,
    model_slug,
    models_sharing_slug,
    parse_file,
    remove_alias,
    save,
    sync_to_all_models,
)

st.header("🔧 Справочник ошибок устройств")

# ── RAG reindex control ──────────────────────────────────────────────────────
with st.container(border=True):
    cols = st.columns([4, 1])
    with cols[0]:
        st.caption(
            "💡 Правки severity вступают в силу для расчёта сразу после сохранения. "
            "RAG-коллекция `error_codes` (семантический поиск по кодам в чате) "
            "пересобирается отдельно."
        )
    with cols[1]:
        if st.button("🔄 Пересобрать RAG-индекс", use_container_width=True):
            with st.spinner("Переиндексация коллекции error_codes…"):
                try:
                    from rag.reindex_error_codes import reindex_error_codes
                    stats = reindex_error_codes()
                    by_vendor = ", ".join(
                        f"{v}={n}" for v, n in sorted(stats["by_vendor"].items())
                    )
                    st.success(
                        f"Готово: upserted={stats['upserted']} ({by_vendor})"
                    )
                except Exception as exc:
                    st.error(f"Ошибка переиндексации: {exc}")

# ── Session keys ─────────────────────────────────────────────────────────────
_DIRTY = "ec_dirty"          # dict[(vendor, model_slug)] -> DataFrame
_CONFIRM_DELETE = "ec_confirm_delete"
if _DIRTY not in st.session_state:
    st.session_state[_DIRTY] = {}

SEVERITY_OPTIONS = ["Critical", "High", "Medium", "Low", "Info"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _codes_to_df(doc: ModelErrorCodes | None) -> pd.DataFrame:
    if doc is None or not doc.codes:
        return pd.DataFrame(
            columns=["code", "description", "severity", "component", "notes"]
        )
    rows = []
    for code, info in doc.codes.items():
        rows.append({
            "code": code,
            "description": info.description,
            "severity": info.severity,
            "component": info.component,
            "notes": info.notes,
        })
    return pd.DataFrame(rows)


def _df_to_model(df: pd.DataFrame, vendor: str, model: str) -> ModelErrorCodes:
    codes: dict[str, ErrorCode] = {}
    errors: list[str] = []
    for idx, row in df.iterrows():
        code = str(row.get("code") or "").strip()
        if not code:
            continue
        try:
            codes[code] = ErrorCode(
                description=str(row.get("description") or "").strip(),
                severity=str(row.get("severity") or "Medium").strip() or "Medium",
                component=str(row.get("component") or "").strip(),
                notes=str(row.get("notes") or "").strip(),
            )
        except Exception as exc:
            errors.append(f"Строка {int(idx) + 2} (код {code}): {exc}")
    if errors:
        raise ValueError("Ошибки валидации:\n" + "\n".join(errors[:10]))
    return ModelErrorCodes(vendor=vendor, model=model, codes=codes)


def _key(vendor: str, model: str) -> tuple[str, str]:
    return (vendor, model_slug(model))


# ── Brand tabs ───────────────────────────────────────────────────────────────

vendor_tabs = st.tabs(list(SUPPORTED_VENDORS))

for tab, vendor in zip(vendor_tabs, SUPPORTED_VENDORS):
    with tab:
        # --- Model selector ---
        existing = list_models(vendor)
        cols = st.columns([3, 2, 2])
        with cols[0]:
            if existing:
                selected_slug = st.selectbox(
                    "Модель",
                    options=existing,
                    key=f"sel_model_{vendor}",
                )
            else:
                st.info(f"Для {vendor} пока нет справочников. Создайте новую модель или загрузите файл ниже.")
                selected_slug = None
        with cols[1]:
            new_model_name = st.text_input(
                "Новая модель",
                key=f"new_model_{vendor}",
                placeholder="Напр. B8090",
            )
        with cols[2]:
            if st.button(
                "➕ Создать",
                key=f"create_model_{vendor}",
                disabled=not new_model_name,
                use_container_width=True,
            ):
                new_slug = model_slug(new_model_name)
                if new_slug in existing:
                    st.warning(f"Модель {new_slug!r} уже существует.")
                else:
                    empty_doc = ModelErrorCodes(
                        vendor=vendor,
                        model=new_model_name,
                        codes={},
                    )
                    save(empty_doc)
                    st.success(f"Создан справочник для {new_model_name}.")
                    st.rerun()

        if selected_slug is None:
            # still allow upload section below for empty vendor
            doc = None
        else:
            doc = load_codes(vendor, selected_slug)

        key = _key(vendor, selected_slug or "")

        # --- Load base DataFrame ---
        if key in st.session_state[_DIRTY]:
            df = st.session_state[_DIRTY][key]
            dirty = True
        else:
            df = _codes_to_df(doc)
            dirty = False

        if doc is not None:
            st.caption(
                f"📊 {vendor} / {doc.model} — {len(doc.codes)} кодов. "
                f"Обновлено: {doc.updated_at:%Y-%m-%d %H:%M UTC}"
            )
            # ── Linked models (aliases) — dropdown + attach + detach ───────
            shared = models_sharing_slug(selected_slug or "")
            with st.container(border=True):
                st.markdown("**🔗 Этот справочник применяется к моделям**")
                if not shared:
                    st.caption("(нет дополнительных алиасов)")
                else:
                    # Determine the canonical name (first hit from shared, longest)
                    # and separate canonical from attachable aliases.
                    canonical_for_slug = (
                        get_canonical_for_alias(shared[0]) or doc.model
                    )
                    aliases = [
                        s for s in shared
                        if s.strip().lower() != canonical_for_slug.strip().lower()
                    ]
                    st.caption(
                        f"Каноническое имя: **{canonical_for_slug}** "
                        f"(удалить нельзя — сломает регистр)"
                    )

                    cols = st.columns([4, 1])
                    with cols[0]:
                        if aliases:
                            chosen_alias = st.selectbox(
                                "Привязанные модели",
                                options=aliases,
                                key=f"alias_sel_{vendor}_{selected_slug}",
                            )
                        else:
                            chosen_alias = None
                            st.caption("Пока нет дополнительных привязок.")
                    with cols[1]:
                        st.write(" ")
                        if st.button(
                            "🗑 Открепить",
                            key=f"alias_del_{vendor}_{selected_slug}",
                            disabled=not chosen_alias,
                            use_container_width=True,
                        ):
                            try:
                                removed_from = remove_alias(chosen_alias)
                                if removed_from:
                                    st.success(
                                        f"Модель '{chosen_alias}' откреплена от "
                                        f"'{removed_from}'."
                                    )
                                else:
                                    st.warning(
                                        f"Модель '{chosen_alias}' не найдена в aliases."
                                    )
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))

                    # ── Attach form ────────────────────────────────────────
                    add_cols = st.columns([4, 1])
                    with add_cols[0]:
                        new_name = st.text_input(
                            "Привязать новую модель",
                            key=f"alias_add_{vendor}_{selected_slug}",
                            placeholder="Напр. B8145, PrimeLink B9165",
                            label_visibility="collapsed",
                        )
                    with add_cols[1]:
                        st.write(" ")
                        clicked = st.button(
                            "➕ Привязать",
                            key=f"alias_add_btn_{vendor}_{selected_slug}",
                            disabled=not new_name,
                            use_container_width=True,
                            type="primary",
                        )

                    conflict_key = f"alias_conflict_{vendor}_{selected_slug}"
                    if clicked:
                        try:
                            result = add_alias(canonical_for_slug, new_name)
                            if result["action"] == "noop":
                                st.info("Эта модель уже привязана к текущему файлу.")
                            else:
                                st.success(
                                    f"Модель '{new_name}' привязана к "
                                    f"'{canonical_for_slug}'."
                                )
                            st.rerun()
                        except AliasConflict as exc:
                            # Stash conflict info for confirm dialog
                            st.session_state[conflict_key] = {
                                "alias": exc.alias,
                                "existing": exc.existing_canonical,
                                "new_name_raw": new_name,
                            }
                        except Exception as exc:
                            st.error(str(exc))

                    # Confirm dialog for conflicts
                    conflict_data = st.session_state.get(conflict_key)
                    if conflict_data:
                        with st.container(border=True):
                            st.warning(
                                f"⚠ Модель **{conflict_data['alias']}** уже "
                                f"привязана к **{conflict_data['existing']}**. "
                                f"Переназначить на **{canonical_for_slug}**?"
                            )
                            cf = st.columns(2)
                            with cf[0]:
                                if st.button(
                                    "🔄 Да, переназначить",
                                    key=f"conflict_yes_{vendor}_{selected_slug}",
                                    type="primary",
                                ):
                                    try:
                                        add_alias(
                                            canonical_for_slug,
                                            conflict_data["new_name_raw"],
                                            force_reassign=True,
                                        )
                                        st.success(
                                            f"Модель '{conflict_data['alias']}' "
                                            f"переназначена на '{canonical_for_slug}'."
                                        )
                                        st.session_state.pop(conflict_key, None)
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(str(exc))
                            with cf[1]:
                                if st.button(
                                    "❌ Отмена",
                                    key=f"conflict_no_{vendor}_{selected_slug}",
                                ):
                                    st.session_state.pop(conflict_key, None)
                                    st.rerun()

        # --- Quick edit panel: fix severity of one code in one click ---
        if doc is not None and doc.codes:
            with st.expander("⚡ Быстрая правка одного кода", expanded=False):
                qcols = st.columns([2, 2, 1])
                with qcols[0]:
                    q_code = st.selectbox(
                        "Код",
                        options=sorted(doc.codes.keys()),
                        key=f"quick_code_{vendor}_{selected_slug}",
                    )
                current = doc.codes[q_code]
                with qcols[1]:
                    q_sev = st.selectbox(
                        "Severity",
                        options=SEVERITY_OPTIONS,
                        index=SEVERITY_OPTIONS.index(current.severity),
                        key=f"quick_sev_{vendor}_{selected_slug}",
                    )
                with qcols[2]:
                    st.write(" ")
                    if st.button(
                        "💾 Применить",
                        key=f"quick_save_{vendor}_{selected_slug}",
                        type="primary",
                        disabled=q_sev == current.severity,
                        use_container_width=True,
                    ):
                        new_codes = dict(doc.codes)
                        new_codes[q_code] = ErrorCode(
                            description=current.description,
                            severity=q_sev,
                            component=current.component,
                            notes=current.notes,
                        )
                        updated = ModelErrorCodes(
                            vendor=vendor,
                            model=doc.model,
                            codes=new_codes,
                        )
                        save(updated)
                        st.success(
                            f"`{q_code}` → severity={q_sev} сохранено. "
                            "Новое значение применяется к следующему расчёту здоровья."
                        )
                        st.rerun()
                st.caption(
                    f"📝 Описание: _{current.description}_ · "
                    f"компонент: `{current.component or '—'}`"
                )

        # --- Consistency warning ---
        if doc is not None:
            try:
                conflicts = find_conflicts(vendor, doc)
            except Exception:
                conflicts = []
            if conflicts:
                other_models = sorted({c.other_model for c in conflicts})
                with st.expander(
                    f"⚠ Расхождения с другими моделями {vendor}: "
                    f"{len(conflicts)} правок по {len(other_models)} моделям",
                    expanded=False,
                ):
                    for c in conflicts[:50]:
                        st.markdown(
                            f"`{c.code}` / **{c.field}**: "
                            f"здесь = `{c.this_value}`, "
                            f"в `{c.other_model}` = `{c.other_value}`"
                        )
                    if st.button(
                        f"🔄 Синхронизировать все модели {vendor} по текущей версии",
                        key=f"sync_{vendor}",
                    ):
                        affected = sync_to_all_models(vendor, doc)
                        st.success(f"Обновлено моделей: {len(affected)} — {', '.join(affected)}")
                        st.rerun()

        # --- Data editor ---
        if selected_slug is not None:
            edited = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"editor_{vendor}_{selected_slug}",
                column_config={
                    "code": st.column_config.TextColumn("Код", width="small"),
                    "description": st.column_config.TextColumn("Описание", width="medium"),
                    "severity": st.column_config.SelectboxColumn(
                        "Severity",
                        options=SEVERITY_OPTIONS,
                        width="small",
                        required=True,
                    ),
                    "component": st.column_config.TextColumn("Компонент", width="small"),
                    "notes": st.column_config.TextColumn("Примечание", width="medium"),
                },
            )

            st.caption(
                "💡 Кликни в любую ячейку таблицы, чтобы отредактировать. "
                "Severity меняется через дропдаун в колонке."
            )

            if not edited.equals(df):
                st.session_state[_DIRTY][key] = edited
                dirty = True

            if dirty:
                st.info("💡 Изменения не сохранены — нажми «💾 Сохранить» ниже.")

            # --- Save / Revert / Delete ---
            btn_cols = st.columns([2, 2, 2, 2])
            with btn_cols[0]:
                if st.button(
                    "💾 Сохранить",
                    type="primary",
                    key=f"save_{vendor}_{selected_slug}",
                    disabled=not dirty,
                    use_container_width=True,
                ):
                    try:
                        new_doc = _df_to_model(
                            edited,
                            vendor=vendor,
                            model=doc.model if doc else selected_slug,
                        )
                        save(new_doc)
                        st.session_state[_DIRTY].pop(key, None)
                        st.success(
                            f"Сохранено {len(new_doc.codes)} кодов. "
                            "Новые значения применяются к следующему расчёту здоровья."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            with btn_cols[1]:
                if st.button(
                    "↩ Отменить",
                    key=f"revert_{vendor}_{selected_slug}",
                    disabled=not dirty,
                    use_container_width=True,
                ):
                    st.session_state[_DIRTY].pop(key, None)
                    st.rerun()

            with btn_cols[2]:
                if st.button(
                    "🗑 Удалить модель",
                    key=f"delete_{vendor}_{selected_slug}",
                    use_container_width=True,
                ):
                    st.session_state[_CONFIRM_DELETE] = (vendor, selected_slug)
                    st.rerun()

            with btn_cols[3]:
                # --- Export ---
                if doc is not None and doc.codes:
                    df_export = _codes_to_df(doc)
                    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ CSV",
                        data=csv_bytes,
                        file_name=f"{vendor.lower()}_{selected_slug}.csv",
                        mime="text/csv",
                        key=f"dl_csv_{vendor}_{selected_slug}",
                        use_container_width=True,
                    )

            # --- Delete confirmation modal ---
            if st.session_state.get(_CONFIRM_DELETE) == (vendor, selected_slug):
                with st.container(border=True):
                    st.error(
                        f"Удалить справочник **{vendor} / {selected_slug}**? "
                        f"Файл будет перемещён в `_trash/`."
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Да, удалить", key=f"confirm_del_{vendor}"):
                            trash_path = delete(vendor, selected_slug)
                            st.session_state.pop(_CONFIRM_DELETE, None)
                            st.session_state[_DIRTY].pop(key, None)
                            if trash_path:
                                st.success(f"Файл перемещён в {trash_path.name}")
                            st.rerun()
                    with col2:
                        if st.button("❌ Отмена", key=f"cancel_del_{vendor}"):
                            st.session_state.pop(_CONFIRM_DELETE, None)
                            st.rerun()

        # --- Upload zone ---
        st.markdown("---")
        st.markdown("### 📤 Загрузить справочник")
        up_cols = st.columns([3, 2])
        with up_cols[0]:
            uploaded = st.file_uploader(
                "CSV / XLSX / YAML",
                type=["csv", "xlsx", "yaml", "yml"],
                key=f"upload_{vendor}",
                label_visibility="collapsed",
            )
        with up_cols[1]:
            replace_mode = st.checkbox(
                "Заменить полностью",
                key=f"replace_{vendor}",
                help="Иначе — upsert (добавить новые + обновить существующие)",
            )
            upload_target = st.text_input(
                "Модель для загрузки",
                value=selected_slug or (new_model_name or ""),
                key=f"upload_target_{vendor}",
                help="Имя модели — будет нормализовано в slug",
            )

        if uploaded is not None and upload_target and st.button(
            "📥 Импортировать", key=f"import_{vendor}", type="primary"
        ):
            try:
                data = uploaded.read()
                parsed_doc = parse_file(
                    uploaded.name,
                    data,
                    vendor=vendor,
                    model=upload_target,
                )
                if replace_mode:
                    final_doc = parsed_doc
                else:
                    existing_doc = load_codes(vendor, upload_target)
                    if existing_doc is None:
                        final_doc = parsed_doc
                    else:
                        final_doc = existing_doc.merge(parsed_doc, replace=False)
                save(final_doc)
                st.success(
                    f"Импортировано {len(parsed_doc.codes)} кодов в "
                    f"{vendor}/{upload_target}. Режим: "
                    f"{'замена' if replace_mode else 'upsert'}."
                )
                st.rerun()
            except ParseError as exc:
                st.error(f"Ошибка парсинга: {exc}")
            except Exception as exc:
                st.error(f"Непредвиденная ошибка: {exc}")
