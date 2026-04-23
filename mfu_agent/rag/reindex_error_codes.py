"""Re-index the `error_codes` Qdrant collection from per-model registries.

Source: every file under ``configs/error_codes/{vendor}/*.yaml`` for each
vendor in ``error_codes.SUPPORTED_VENDORS``. The same files are edited on
the "Справочник ошибок устройств" page — one source of truth.

Each code becomes one chunk with text:
  "{code} {description}. Severity: {severity}. Component: {component}."

Run with:
  python -m rag.reindex_error_codes
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client.http import models as qmodels

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import get_config_manager  # noqa: E402
from rag.embeddings import BGEEmbedder  # noqa: E402
from rag.ingestion import EmbeddedChunk  # noqa: E402
from rag.qdrant_client import QdrantManager  # noqa: E402

logger = logging.getLogger(__name__)

_COLLECTION = "error_codes"
_DOC_PREFIX = "error_codes_registry"


def _collect_entries() -> list[dict[str, Any]]:
    """Read every vendor/model file under configs/error_codes/{vendor}/*.yaml."""
    from error_codes import SUPPORTED_VENDORS, list_models, load_codes

    entries: list[dict[str, Any]] = []
    for vendor in SUPPORTED_VENDORS:
        for model_key in list_models(vendor):
            doc = load_codes(vendor, model_key)
            if doc is None:
                continue
            for code, info in doc.codes.items():
                entries.append({
                    "code": str(code),
                    "description": info.description,
                    "severity": info.severity,
                    "component": info.component,
                    "notes": info.notes,
                    "vendor": vendor,
                    "model": doc.model,
                })
    return entries


def _format_text(entry: dict[str, Any]) -> str:
    parts = [entry["code"]]
    if entry["description"]:
        parts.append(entry["description"])
    if entry["severity"]:
        parts.append(f"Severity: {entry['severity']}")
    if entry["component"]:
        parts.append(f"Component: {entry['component']}")
    if entry["notes"]:
        parts.append(entry["notes"])
    return ". ".join(parts) + "."


def _recreate_collection(mgr: QdrantManager, dense_dim: int) -> None:
    client = mgr.rest_client
    try:
        client.delete_collection(_COLLECTION)
        logger.info("Dropped old '%s' collection", _COLLECTION)
    except Exception as exc:  # noqa: BLE001
        logger.info("Could not drop '%s' (may not exist): %s", _COLLECTION, exc)

    from rag.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=dense_dim,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info("Created '%s' with size=%d", _COLLECTION, dense_dim)


def _embed_and_upsert(
    entries: list[dict[str, Any]],
    embedder: BGEEmbedder,
    mgr: QdrantManager,
) -> int:
    from rag.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    texts = [_format_text(e) for e in entries]

    logger.info("Encoding %d texts (dense + sparse)…", len(texts))
    result = embedder.encode(texts, return_sparse=True)
    dense_vectors = result.dense
    sparse_vectors = result.sparse or [None] * len(texts)

    now_iso = datetime.now(UTC).isoformat()
    points = []
    for e, text, dv, sv in zip(entries, texts, dense_vectors, sparse_vectors, strict=True):
        src_slug = f"{e['vendor'].lower()}_{e['model'].lower().replace(' ', '_')}"
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS, f"{_DOC_PREFIX}:{src_slug}:{e['code']}"
        ))
        payload = {
            "chunk_id": f"ec_{e['code']}",
            "document_id": f"{_DOC_PREFIX}_{src_slug}",
            "collection": _COLLECTION,
            "vendor": e["vendor"],
            "model": e["model"],
            "models": [e["model"]],
            "doc_title": f"{e['vendor']} {e['model']} error codes",
            "section": None,
            "page_number": None,
            "content_type": "reference",
            "error_codes": [e["code"]],
            "components": [e["component"]] if e["component"] else [],
            "text": text,
            "language": "ru",
            "indexed_at": now_iso,
            "severity": e["severity"],
        }
        vectors: dict[str, Any] = {DENSE_VECTOR_NAME: dv.tolist()}
        if sv is not None and getattr(sv, "indices", None):
            vectors[SPARSE_VECTOR_NAME] = sv
        points.append(qmodels.PointStruct(id=point_id, vector=vectors, payload=payload))

    mgr.rest_client.upsert(collection_name=_COLLECTION, points=points, wait=True)
    return len(points)


def reindex_error_codes() -> dict[str, Any]:
    """Rebuild the `error_codes` Qdrant collection from per-model registries.

    Returns ``{"total": K, "upserted": K, "by_vendor": {vendor: count, ...}}``.
    Raises on empty registry / Qdrant / embedding failures; caller handles.
    """
    entries = _collect_entries()
    by_vendor = Counter(e["vendor"] for e in entries)
    logger.info(
        "Total codes to index: %d (%s)",
        len(entries),
        ", ".join(f"{v}={n}" for v, n in sorted(by_vendor.items())) or "empty",
    )

    if not entries:
        raise RuntimeError(
            "No codes found in configs/error_codes/. Add entries via the "
            "«Справочник ошибок устройств» page first."
        )

    cfg = get_config_manager().load_rag_config()
    mgr = QdrantManager(cfg)
    embedder = BGEEmbedder(cfg.embeddings)

    sample = embedder.encode(["probe"], return_sparse=False)
    dense_dim = sample.dense.shape[1]
    logger.info("Dense dim detected: %d", dense_dim)

    _recreate_collection(mgr, dense_dim)
    upserted = _embed_and_upsert(entries, embedder, mgr)
    logger.info("Upserted %d points to '%s'", upserted, _COLLECTION)

    return {
        "total": len(entries),
        "upserted": upserted,
        "by_vendor": dict(by_vendor),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        stats = reindex_error_codes()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    by_vendor_str = ", ".join(
        f"{v}={n}" for v, n in sorted(stats["by_vendor"].items())
    )
    logger.info(
        "Collection '%s' upserted=%d (by vendor: %s)",
        _COLLECTION, stats["upserted"], by_vendor_str,
    )


if __name__ == "__main__":
    main()
