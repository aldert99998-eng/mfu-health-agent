"""Страница 4 — Администрирование RAG-базы."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import streamlit as st

from config.loader import get_config_manager
from state import _rag_bg as _rbg
from state.singletons import (
    get_embedder,
    get_hybrid_searcher,
    get_qdrant_manager,
)

st.header("RAG — Администрирование")

cm = get_config_manager()
rag_cfg = cm.load_rag_config()

# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Коллекции
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Коллекции")

qm = get_qdrant_manager()

if not qm.healthcheck():
    st.error("Qdrant недоступен. Проверьте подключение.")
    st.stop()

configured_names = [c.name for c in rag_cfg.qdrant.collections]

col_refresh, col_init = st.columns([1, 1])
if col_init.button("Создать все коллекции", use_container_width=True):
    with st.status("Создание коллекций…", expanded=True) as status:
        created = qm.ensure_all_collections()
        for name, was_new in created.items():
            st.write(f"{'✓ создана' if was_new else '— уже существует'}: **{name}**")
        status.update(label="Готово", state="complete")

infos: list[Any] = []
for name in configured_names:
    try:
        infos.append(qm.collection_info(name))
    except Exception:
        infos.append(None)

if infos:
    rows = []
    for info in infos:
        if info is None:
            rows.append({"Коллекция": "—", "Точек": 0, "Сегментов": 0, "Статус": "не найдена"})
        else:
            rows.append({
                "Коллекция": info.name,
                "Точек": info.points_count,
                "Сегментов": info.segments_count,
                "Статус": info.status,
            })
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.caption("Коллекции не сконфигурированы в rag_config.yaml.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Загрузка документов
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Загрузка документов")

upload_col_sel, upload_col_file = st.columns([1, 2])

target_collection = upload_col_sel.selectbox(
    "Коллекция",
    configured_names or ["service_manuals"],
    key="rag_upload_collection",
)

uploaded_file = upload_col_file.file_uploader(
    "PDF / TXT документ",
    type=["pdf", "txt"],
    key="rag_upload_file",
)

_STAGE_LABELS = {
    "parse": "1/7 Парсинг документа",
    "preprocess": "2/7 Предобработка текста",
    "identify_model": "3/7 Определение модели устройства",
    "chunk": "4/7 Разбиение на чанки",
    "enrich": "5/7 Обогащение метаданных",
    "embed": "6/7 Вычисление эмбеддингов",
    "upsert": "7/7 Загрузка в Qdrant",
}



def _render_rag_log() -> None:
    lines = list(_rbg.log_lines)
    if lines:
        with st.expander("Журнал индексации", expanded=True):
            st.markdown("\n\n".join(lines))


# --- Background process status display ---
if _rbg.process is not None and _rbg.process.is_alive():
    st.info("⏳ Идёт индексация документа…")
    stage = _rbg.progress.get("stage", "")
    if stage:
        label = _STAGE_LABELS.get(stage, stage)
        st.caption(f"Текущий этап: {label}")
    _render_rag_log()
    if st.button("Отменить индексацию", type="secondary", use_container_width=True, key="btn_cancel_indexing"):
        _rbg.process.terminate()
        _rbg.process.join(timeout=5)
        _rbg.log("⛔ Индексация отменена пользователем")
        _rbg.reset()
        st.rerun()
    time.sleep(1)
    st.rerun()
elif _rbg.process is not None and not _rbg.process.is_alive():
    res = dict(_rbg.result)
    if res.get("success"):
        st.success(
            f"Документ **{res.get('document_id', '')}** — "
            f"{res.get('chunks_count', 0)} чанков за {res.get('duration', 0):.1f} сек."
        )
        for err in res.get("errors", []):
            st.warning(err)
    elif "error" in res:
        st.error(f"Ошибка индексации: {res['error']}")
    _render_rag_log()
    _rbg.reset()

if uploaded_file and st.button("Индексировать", type="primary", use_container_width=True):
    if _rbg.process is not None and _rbg.process.is_alive():
        st.warning("Индексация уже запущена, дождитесь завершения.")
    else:
        _rbg.reset()

        uploads_dir = Path(rag_cfg.storage.uploads_dir)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = uploads_dir / uploaded_file.name
        dest.write_bytes(uploaded_file.getvalue())

        file_mb = len(uploaded_file.getvalue()) / 1024 / 1024
        _rbg.log(f"Файл сохранён: {dest.name} ({file_mb:.1f} МБ)")

        llm_endpoint = None
        try:
            from state.session import get_active_llm_endpoint
            llm_endpoint = get_active_llm_endpoint()
        except Exception:
            pass

        shared_progress, shared_log, shared_result = _rbg.make_shared()
        _rbg.progress = shared_progress
        _rbg.log_lines = shared_log
        _rbg.result = shared_result

        _rbg.log(f"Файл сохранён: {dest.name} ({file_mb:.1f} МБ)")
        _rbg.log("Запуск фонового процесса индексации…")

        from rag._indexing_worker import run as _indexing_run

        ctx = mp.get_context("spawn")
        _rbg.process = ctx.Process(
            target=_indexing_run,
            args=(
                str(dest),
                target_collection,
                llm_endpoint,
                shared_progress,
                shared_log,
                shared_result,
            ),
            daemon=True,
        )
        _rbg.process.start()
        st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Тестовый поиск
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Поиск (тест)")

search_col_left, search_col_right = st.columns([3, 1])

search_query = search_col_left.text_input("Запрос", key="rag_search_query")
search_collection = search_col_right.selectbox(
    "Коллекция",
    configured_names or ["service_manuals"],
    key="rag_search_collection",
)

filter_cols = st.columns(3)
filter_model = filter_cols[0].text_input("Фильтр: модель", key="rag_filter_model")
filter_content_type = filter_cols[1].selectbox(
    "Фильтр: тип контента",
    ["—", "symptom", "cause", "procedure", "specification", "reference"],
    key="rag_filter_ctype",
)
search_top_k = filter_cols[2].number_input("Top-K", min_value=1, max_value=50, value=8, key="rag_search_topk")

use_reranker = st.checkbox("Использовать reranker", value=True, key="rag_use_reranker")

if search_query and st.button("Искать", use_container_width=True):
    searcher = get_hybrid_searcher()
    filters: dict[str, Any] = {}
    if filter_model:
        filters["model"] = filter_model
    if filter_content_type != "—":
        filters["content_type"] = filter_content_type

    with st.status("Поиск…", expanded=False) as status:
        results = searcher.search(
            query=search_query,
            collection=search_collection,
            top_k=search_top_k,
            filters=filters or None,
            use_reranker=use_reranker,
        )
        status.update(label=f"Найдено {len(results)} результатов", state="complete")

    if results:
        for i, r in enumerate(results, 1):
            score_label = f"score={r.score:.4f}"
            if r.dense_score or r.sparse_score:
                score_label += f"  (dense={r.dense_score:.4f}, sparse={r.sparse_score:.4f})"
            with st.expander(f"#{i}  {r.chunk_id}  —  {score_label}", expanded=(i <= 3)):
                st.markdown(r.text[:1000])
                if len(r.text) > 1000:
                    st.caption(f"… ещё {len(r.text) - 1000} символов")
                meta_cols = st.columns(4)
                meta_cols[0].caption(f"doc: {r.document_id}")
                meta_cols[1].caption(f"type: {r.payload.get('content_type', '—')}")
                meta_cols[2].caption(f"models: {r.payload.get('models', '—')}")
                meta_cols[3].caption(f"codes: {r.payload.get('error_codes', '—')}")
    else:
        st.info("Ничего не найдено.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Метрики
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Метрики качества")

eval_collection = st.selectbox(
    "Коллекция для eval",
    configured_names or ["service_manuals"],
    key="rag_eval_collection",
)

if st.button("Запустить eval", use_container_width=True):
    from rag.evaluation import RAGEvaluator

    searcher = get_hybrid_searcher()
    evaluator = RAGEvaluator(searcher=searcher, config=rag_cfg)

    with st.status("Прогон eval-датасета…", expanded=True) as status:
        try:
            report = evaluator.run_eval(collection=eval_collection)
            evaluator.save_report(report)
            status.update(label="Eval завершён", state="complete")
        except FileNotFoundError:
            status.update(label="Файл eval-датасета не найден", state="error")
            st.error(f"Файл {rag_cfg.evaluation.dataset_path} не найден.")
            st.stop()
        except Exception as exc:
            status.update(label="Ошибка eval", state="error")
            st.error(str(exc))
            st.stop()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Recall@5", f"{report.recall_at_5:.2f}")
    metric_cols[1].metric("Recall@10", f"{report.recall_at_10:.2f}")
    metric_cols[2].metric("MRR", f"{report.mrr:.2f}")
    metric_cols[3].metric("Precision@5", f"{report.precision_at_5:.2f}")
    metric_cols[4].metric("nDCG@10", f"{report.ndcg_at_10:.2f}")

    checks = evaluator.check_thresholds(report)
    if checks:
        for chk in checks:
            if chk.passed:
                st.success(f"{chk.metric}: {chk.value:.3f} ≥ {chk.threshold:.3f}")
            else:
                st.error(f"{chk.metric}: {chk.value:.3f} < {chk.threshold:.3f}")

    deltas = evaluator.delta_vs_previous(report)
    if deltas:
        st.markdown("**Дельта vs предыдущий запуск:**")
        delta_rows = []
        for d in deltas:
            delta_rows.append({
                "Метрика": d.metric,
                "Текущая": f"{d.current:.3f}",
                "Предыдущая": f"{d.previous:.3f}",
                "Δ": f"{d.delta:+.3f}",
            })
        st.dataframe(delta_rows, use_container_width=True, hide_index=True)

    if report.per_scenario:
        st.markdown("**По сценариям:**")
        sc_rows = []
        for sc in report.per_scenario:
            sc_rows.append({
                "Сценарий": sc.scenario,
                "Запросов": sc.num_queries,
                "Recall@5": f"{sc.recall_at_5:.2f}",
                "MRR": f"{sc.mrr:.2f}",
            })
        st.dataframe(sc_rows, use_container_width=True, hide_index=True)

# ── Eval history ─────────────────────────────────────────────────────────

with st.expander("История eval-запусков", expanded=False):
    try:
        from rag.evaluation import RAGEvaluator

        searcher = get_hybrid_searcher()
        evaluator = RAGEvaluator(searcher=searcher, config=rag_cfg)
        history = evaluator.get_history(last_n=10)
        if history:
            hist_rows = []
            for h in history:
                hist_rows.append({
                    "Время": h.timestamp,
                    "Коллекция": h.collection,
                    "Запросов": h.num_queries,
                    "Recall@5": f"{h.recall_at_5:.2f}",
                    "MRR": f"{h.mrr:.2f}",
                    "Пороги": "OK" if h.all_thresholds_passed else "FAIL",
                })
            st.dataframe(hist_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Нет сохранённых запусков.")
    except Exception as exc:
        st.caption(f"Не удалось загрузить историю: {exc}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Операции
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("Операции")

op_tabs = st.tabs(["Переиндексация", "Удаление", "Диагностика"])

# ── Tab: Переиндексация ──────────────────────────────────────────────────

with op_tabs[0]:
    reindex_collection = st.selectbox(
        "Коллекция",
        configured_names or ["service_manuals"],
        key="rag_reindex_collection",
    )
    reindex_full = st.checkbox("Полная (удалить и пересоздать)", key="rag_reindex_full")

    if st.button("Переиндексировать", use_container_width=True, key="btn_reindex"):
        from rag.ingestion import index_document

        uploads_dir = Path(rag_cfg.storage.uploads_dir)
        if not uploads_dir.exists():
            st.warning("Папка uploads пуста.")
        else:
            files = list(uploads_dir.glob("*.pdf")) + list(uploads_dir.glob("*.txt"))
            if not files:
                st.warning("Нет файлов для индексации в uploads/.")
            else:
                embedder = get_embedder()
                with st.status(
                    f"Переиндексация {len(files)} файлов…", expanded=True
                ) as status:
                    if reindex_full:
                        st.write("Удаление коллекции…")
                        with contextlib.suppress(Exception):
                            qm.drop_collection(reindex_collection, confirm=True)
                        qm.ensure_collection(reindex_collection)
                        st.write("Коллекция пересоздана.")

                    llm_client = None
                    try:
                        from state.session import get_active_llm_endpoint
                        from state.singletons import get_llm_client
                        llm_client = get_llm_client(get_active_llm_endpoint())
                    except Exception:
                        pass

                    ok_count = 0
                    err_count = 0
                    for i, fpath in enumerate(files, 1):
                        st.write(f"[{i}/{len(files)}] {fpath.name}")
                        try:
                            index_document(
                                file_path=fpath,
                                collection=reindex_collection,
                                qdrant_manager=qm,
                                embedder=embedder,
                                config=rag_cfg,
                                llm_client=llm_client,  # type: ignore[arg-type]
                            )
                            ok_count += 1
                        except Exception as exc:
                            st.write(f"  ⚠ {exc}")
                            err_count += 1

                    label = f"Готово: {ok_count} ок, {err_count} ошибок"
                    status.update(
                        label=label,
                        state="complete" if err_count == 0 else "error",
                    )

# ── Tab: Удаление ────────────────────────────────────────────────────────

with op_tabs[1]:
    del_collection = st.selectbox(
        "Коллекция",
        configured_names or ["service_manuals"],
        key="rag_del_collection",
    )

    del_mode = st.radio(
        "Режим",
        ["Очистить (все точки)", "Удалить коллекцию"],
        key="rag_del_mode",
    )

    if st.button("Выполнить", type="primary", use_container_width=True, key="btn_delete"):
        with st.status("Удаление…", expanded=True) as status:
            try:
                if del_mode == "Удалить коллекцию":
                    qm.drop_collection(del_collection, confirm=True)
                    st.write(f"Коллекция **{del_collection}** удалена.")
                else:
                    qm.drop_collection(del_collection, confirm=True)
                    qm.ensure_collection(del_collection)
                    st.write(f"Коллекция **{del_collection}** очищена.")
                status.update(label="Готово", state="complete")
            except Exception as exc:
                status.update(label="Ошибка", state="error")
                st.error(str(exc))

# ── Tab: Диагностика ─────────────────────────────────────────────────────

with op_tabs[2]:
    diag_collection = st.selectbox(
        "Коллекция",
        configured_names or ["service_manuals"],
        key="rag_diag_collection",
    )

    if st.button("Запустить диагностику", use_container_width=True, key="btn_diag"):
        with st.status("Диагностика…", expanded=True) as status:
            try:
                info = qm.collection_info(diag_collection)
                st.write(f"Точек: **{info.points_count}**")
                st.write(f"Сегментов: **{info.segments_count}**")
                st.write(f"Проиндексировано векторов: **{info.indexed_vectors_count}**")
                st.write(f"Статус: **{info.status}**")

                embedder = get_embedder()
                st.write(f"Модель эмбеддингов: **{embedder.embedding_version()}**")

                uploads_dir = Path(rag_cfg.storage.uploads_dir)
                if uploads_dir.exists():
                    upload_files = list(uploads_dir.glob("*.pdf")) + list(uploads_dir.glob("*.txt"))
                    st.write(f"Файлов в uploads: **{len(upload_files)}**")
                else:
                    st.write("Папка uploads не существует.")

                status.update(label="Диагностика завершена", state="complete")
            except Exception as exc:
                status.update(label="Ошибка", state="error")
                st.error(str(exc))
