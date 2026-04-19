"""Qdrant vector-store manager — Track D, Level 1.

Wraps qdrant_client.QdrantClient with typed helpers for collection
lifecycle, health checks, and named-vector configuration (dense + sparse).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

if TYPE_CHECKING:
    from config.loader import QdrantCollectionConfig, RAGConfig

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

KEYWORD_INDEXES = (
    "doc_type",
    "source_file",
    "vendor",
    "model",
    "error_code",
    "language",
)


# ── Custom errors ───────────────────────────────────────────────────────────


class QdrantUnavailableError(Exception):
    """Raised when Qdrant server is unreachable."""


class CollectionExistsError(Exception):
    """Raised on attempt to create a collection that already exists."""


# ── CollectionInfo ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Lightweight snapshot of collection state."""

    name: str
    points_count: int
    segments_count: int
    indexed_vectors_count: int
    status: str


# ── QdrantManager ───────────────────────────────────────────────────────────


class QdrantManager:
    """High-level wrapper around :class:`qdrant_client.QdrantClient`.

    Parameters
    ----------
    config:
        Full RAGConfig; ``config.qdrant`` supplies connection params
        and per-collection vector settings.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        qcfg = config.qdrant
        try:
            self._client = QdrantClient(
                host=qcfg.host,
                port=qcfg.port,
                grpc_port=qcfg.grpc_port,
                prefer_grpc=qcfg.prefer_grpc,
                timeout=qcfg.timeout_seconds,
                check_compatibility=False,
            )
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Не удалось подключиться к Qdrant ({qcfg.host}:{qcfg.port}): {exc}"
            ) from exc

        logger.info(
            "QdrantManager инициализирован: %s:%s (grpc=%s)",
            qcfg.host,
            qcfg.port,
            qcfg.prefer_grpc,
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def rest_client(self) -> QdrantClient:
        """REST-only client for operations incompatible with gRPC on older servers."""
        if not hasattr(self, "_rest_client"):
            qcfg = self._config.qdrant
            self._rest_client = QdrantClient(
                host=qcfg.host,
                port=qcfg.port,
                prefer_grpc=False,
                timeout=qcfg.timeout_seconds,
                check_compatibility=False,
            )
        return self._rest_client

    # ── ensure_collection ───────────────────────────────────────────────────

    def ensure_collection(self, collection_name: str) -> bool:
        """Create collection if it does not exist.

        Dense vectors use COSINE distance with configurable HNSW params.
        Sparse vectors use IDF modifier.  Six keyword payload indexes
        are created automatically.

        Returns ``True`` if a new collection was created, ``False`` if it
        already existed.

        Raises
        ------
        QdrantUnavailableError
            If the Qdrant server is unreachable.
        """
        col_cfg = self._find_collection_config(collection_name)
        dense_size = col_cfg.dense_size if col_cfg else 1024
        hnsw_m = (col_cfg.hnsw_m if col_cfg else None) or 16
        hnsw_ef = (col_cfg.hnsw_ef_construct if col_cfg else None) or 128

        try:
            if self._client.collection_exists(collection_name):
                logger.info("Коллекция уже существует: %s", collection_name)
                return False
        except (ResponseHandlingException, Exception) as exc:
            raise QdrantUnavailableError(
                f"Qdrant недоступен при проверке коллекции {collection_name}: {exc}"
            ) from exc

        dense_config = qmodels.VectorParams(
            size=dense_size,
            distance=qmodels.Distance.COSINE,
            hnsw_config=qmodels.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef),
        )

        sparse_config = qmodels.SparseVectorParams(
            modifier=qmodels.Modifier.IDF,
        )

        try:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={DENSE_VECTOR_NAME: dense_config},
                sparse_vectors_config={SPARSE_VECTOR_NAME: sparse_config},
            )
        except (ResponseHandlingException, Exception) as exc:
            raise QdrantUnavailableError(
                f"Не удалось создать коллекцию {collection_name}: {exc}"
            ) from exc

        self._create_keyword_indexes(collection_name)

        logger.info(
            "Коллекция создана: %s (dense=%d, hnsw_m=%d, hnsw_ef=%d)",
            collection_name,
            dense_size,
            hnsw_m,
            hnsw_ef,
        )
        return True

    # ── ensure_all_collections ──────────────────────────────────────────────

    def ensure_all_collections(self) -> dict[str, bool]:
        """Apply :meth:`ensure_collection` to every collection in config.

        Returns mapping ``{collection_name: was_created}``.
        """
        results: dict[str, bool] = {}
        for col_cfg in self._config.qdrant.collections:
            created = self.ensure_collection(col_cfg.name)
            results[col_cfg.name] = created

        created_count = sum(results.values())
        logger.info(
            "ensure_all_collections: %d создано, %d уже существовало",
            created_count,
            len(results) - created_count,
        )
        return results

    # ── drop_collection ─────────────────────────────────────────────────────

    def drop_collection(self, name: str, *, confirm: bool = True) -> None:
        """Delete a collection.

        Parameters
        ----------
        name:
            Collection name to drop.
        confirm:
            Safety guard — must be ``True`` to actually delete.

        Raises
        ------
        ValueError
            If *confirm* is not ``True``.
        QdrantUnavailableError
            If the Qdrant server is unreachable.
        """
        if not confirm:
            raise ValueError(
                f"Удаление коллекции {name} отклонено: confirm должен быть True"
            )

        try:
            self._client.delete_collection(collection_name=name)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise QdrantUnavailableError(
                f"Не удалось удалить коллекцию {name}: {exc}"
            ) from exc

        logger.info("Коллекция удалена: %s", name)

    # ── collection_info ─────────────────────────────────────────────────────

    def collection_info(self, name: str) -> CollectionInfo:
        """Return lightweight status snapshot for a collection.

        Raises
        ------
        QdrantUnavailableError
            If the Qdrant server is unreachable or the collection
            does not exist.
        """
        try:
            info = self._client.get_collection(collection_name=name)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise QdrantUnavailableError(
                f"Не удалось получить информацию о коллекции {name}: {exc}"
            ) from exc

        logger.info(
            "collection_info(%s): points=%s, status=%s",
            name,
            info.points_count,
            info.status,
        )

        return CollectionInfo(
            name=name,
            points_count=info.points_count or 0,
            segments_count=info.segments_count or 0,
            indexed_vectors_count=info.indexed_vectors_count or 0,
            status=str(info.status),
        )

    # ── healthcheck ─────────────────────────────────────────────────────────

    def healthcheck(self) -> bool:
        """Return ``True`` if Qdrant server responds to a health probe."""
        try:
            self._client.get_collections()
            logger.info("Qdrant healthcheck: OK")
            return True
        except Exception:
            logger.warning("Qdrant healthcheck: FAIL")
            return False

    # ── Private helpers ─────────────────────────────────────────────────────

    def _find_collection_config(self, name: str) -> QdrantCollectionConfig | None:
        """Lookup QdrantCollectionConfig by name, or return None."""
        for col in self._config.qdrant.collections:
            if col.name == name:
                return col
        return None

    def _create_keyword_indexes(self, collection_name: str) -> None:
        for field_name in KEYWORD_INDEXES:
            try:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug(
                    "Не удалось создать индекс %s.%s (возможно, уже существует)",
                    collection_name,
                    field_name,
                )
