"""Background indexing worker — runs in a spawn'ed subprocess."""

from __future__ import annotations

import logging
import time
from pathlib import Path

STAGE_LABELS = {
    "parse": "1/7 Парсинг документа",
    "preprocess": "2/7 Предобработка текста",
    "identify_model": "3/7 Определение модели устройства",
    "chunk": "4/7 Разбиение на чанки",
    "enrich": "5/7 Обогащение метаданных",
    "embed": "6/7 Вычисление эмбеддингов",
    "upsert": "7/7 Загрузка в Qdrant",
}


def run(
    file_path_str: str,
    collection: str,
    llm_endpoint: str | None,
    shared_progress: dict,
    shared_log: list,
    shared_result: dict,
) -> None:
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logging.basicConfig(level=logging.INFO)

    file_path = Path(file_path_str)

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        shared_log.append(f"`{ts}` {msg}")

    try:
        from config.loader import get_config_manager

        cm = get_config_manager()
        rag_config = cm.load_rag_config()

        _log("Загрузка модели эмбеддингов BGE-M3…")
        from rag.embeddings import BGEEmbedder
        embedder = BGEEmbedder(rag_config.embeddings)
        _log("✓ Модель эмбеддингов загружена")

        from rag.qdrant_client import QdrantManager
        qdrant_mgr = QdrantManager(rag_config)

        llm_client = None
        if llm_endpoint:
            try:
                import yaml as _yaml
                from config.loader import CONFIGS_DIR, LLMEndpointConfig
                from llm.client import LLMClient

                path = CONFIGS_DIR / "llm_endpoints.yaml"
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        data = _yaml.safe_load(f) or {}
                    endpoints = data.get("endpoints", data)
                    if isinstance(endpoints, dict) and llm_endpoint in endpoints:
                        ep_cfg = LLMEndpointConfig.model_validate(endpoints[llm_endpoint])
                    else:
                        ep_cfg = LLMEndpointConfig()
                else:
                    ep_cfg = LLMEndpointConfig()
                llm_client = LLMClient(ep_cfg)
                _log("LLM-клиент подключён для обогащения метаданных")
            except Exception as exc:
                _log(f"LLM недоступен: {exc}")

        _log(f"Начинается индексация **{file_path.name}** → коллекция *{collection}*")

        def _progress_cb(stage: str, pct: float) -> None:
            label = STAGE_LABELS.get(stage, stage)
            shared_progress["stage"] = stage
            shared_progress["pct"] = pct
            if pct >= 1.0:
                _log(f"✓ {label} — завершено")
            elif pct == 0.0:
                _log(f"→ {label}…")

        from rag.ingestion import index_document
        res = index_document(
            file_path=file_path,
            collection=collection,
            qdrant_manager=qdrant_mgr,
            embedder=embedder,
            config=rag_config,
            llm_client=llm_client,
            progress_callback=_progress_cb,
        )
        shared_result["success"] = True
        shared_result["document_id"] = res.document_id
        shared_result["chunks_count"] = res.chunks_count
        shared_result["duration"] = res.duration_seconds
        shared_result["errors"] = list(res.errors)
        _log(
            f"✅ Индексация завершена: **{res.chunks_count}** чанков "
            f"за **{res.duration_seconds:.1f}** сек."
        )
    except Exception as exc:
        shared_result["success"] = False
        shared_result["error"] = str(exc)
        _log(f"❌ Ошибка индексации: {exc}")
