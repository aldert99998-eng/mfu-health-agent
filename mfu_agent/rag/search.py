"""Hybrid search engine — Track D, Level 2.

Combines dense (BGE-M3) and sparse (BM25/IDF) retrieval with
Reciprocal Rank Fusion.  Supports server-side fusion via the
Qdrant Query API (>= 1.10) and a client-side RRF fallback.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from qdrant_client.http import models as qmodels

from rag.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

if TYPE_CHECKING:
    from qdrant_client.http.models import SparseVector

    from config.loader import RAGConfig
    from rag.embeddings import BGEEmbedder
    from rag.qdrant_client import QdrantManager
    from rag.reranker import BGEReranker

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_MODEL_ALIASES_PATH = _CONFIGS_DIR / "model_aliases.yaml"

_MIN_QDRANT_FUSION_VERSION = (1, 10)


# ── SearchResult ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single hit from hybrid search (before reranking)."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    dense_score: float
    sparse_score: float
    payload: dict[str, Any] = field(default_factory=dict)


# ── HybridSearcher ─────────────────────────────────────────────────────────


class HybridSearcher:
    """Dense + sparse hybrid search with optional reranking.

    Parameters
    ----------
    qdrant_manager:
        Initialized QdrantManager.
    embedder:
        BGEEmbedder for query encoding.
    config:
        Full RAGConfig — uses ``hybrid_search`` and ``reranker`` sections.
    reranker:
        Optional BGEReranker; if provided, results are re-scored.
    """

    def __init__(
        self,
        qdrant_manager: QdrantManager,
        embedder: BGEEmbedder,
        config: RAGConfig,
        *,
        reranker: BGEReranker | None = None,
    ) -> None:
        self._qdrant = qdrant_manager
        self._embedder = embedder
        self._config = config
        self._hs_cfg = config.hybrid_search
        self._reranker = reranker
        self._model_aliases: dict[str, str] = {}
        self._qdrant_version: tuple[int, ...] | None = None

        self._load_model_aliases()

        logger.info(
            "HybridSearcher инициализирован (fusion=%s, rrf_k=%d, reranker=%s)",
            self._hs_cfg.use_qdrant_fusion,
            self._hs_cfg.rrf_k,
            "да" if reranker else "нет",
        )

    # ── search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        collection: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_reranker: bool = True,
    ) -> list[SearchResult]:
        """Run hybrid search and optional reranking.

        Parameters
        ----------
        query:
            Natural-language query string.
        collection:
            Qdrant collection name.
        top_k:
            Final number of results.  Defaults to
            ``reranker.top_n_output`` (with reranker) or
            ``hybrid_search.top_k_per_branch`` (without).
        filters:
            Optional payload filters.  Keys ``model`` and ``vendor``
            are normalized against model_aliases.yaml.
        use_reranker:
            Whether to apply the cross-encoder reranker (if available).

        Returns
        -------
        list[SearchResult]
            Sorted by score descending.
        """
        t0 = time.perf_counter()

        if top_k is None:
            if use_reranker and self._reranker:
                top_k = self._config.reranker.top_n_output
            else:
                top_k = self._hs_cfg.top_k_per_branch

        dense_vec, sparse_vec = self._embedder.encode_query(query)

        qdrant_filter = self._build_filter(filters) if filters else None

        use_server_fusion = (
            self._hs_cfg.use_qdrant_fusion and self._supports_fusion()
        )

        if use_server_fusion:
            results = self._search_with_qdrant_fusion(
                collection, dense_vec, sparse_vec, qdrant_filter, top_k,
            )
        else:
            results = self._search_with_manual_rrf(
                collection, dense_vec, sparse_vec, qdrant_filter, top_k,
            )

        t_search = time.perf_counter() - t0

        reranker_failed = False
        if use_reranker and self._reranker and results:
            candidates = [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "text": r.text,
                    "score": r.score,
                    "payload": r.payload,
                }
                for r in results
            ]
            try:
                scored = self._reranker.rerank(query, candidates, top_n=top_k)
                results = [
                    SearchResult(
                        chunk_id=s.chunk_id,
                        document_id=s.document_id,
                        text=s.text,
                        score=s.rerank_score,
                        dense_score=0.0,
                        sparse_score=0.0,
                        payload=s.payload,
                    )
                    for s in scored
                ]
            except Exception as exc:
                # Не рушим поиск из-за reranker (OOM/init/GPU недоступна).
                # Возвращаем исходные results, обрезаем до top_k.
                reranker_failed = True
                logger.warning(
                    "Reranker упал, деградируем до hybrid-only: %s",
                    exc,
                )
                results = results[:top_k]

        elapsed = time.perf_counter() - t0
        logger.debug(
            "search(%s): %d результатов за %.3f с (поиск=%.3f с, fusion=%s, reranker=%s)",
            collection,
            len(results),
            elapsed,
            t_search,
            "server" if use_server_fusion else "manual_rrf",
            "упал" if reranker_failed else ("да" if (use_reranker and self._reranker) else "нет"),
        )

        return results

    # ── Server-side fusion (Qdrant Query API) ──────────────────────────────

    def _search_with_qdrant_fusion(
        self,
        collection: str,
        dense_vec: Any,
        sparse_vec: SparseVector | None,
        qdrant_filter: qmodels.Filter | None,
        top_k: int,
    ) -> list[SearchResult]:
        prefetches = [
            qmodels.Prefetch(
                query=dense_vec.tolist(),
                using=DENSE_VECTOR_NAME,
                limit=self._hs_cfg.top_k_per_branch,
                filter=qdrant_filter,
            ),
        ]

        if sparse_vec and sparse_vec.indices:
            prefetches.append(
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=self._hs_cfg.top_k_per_branch,
                    filter=qdrant_filter,
                ),
            )

        rerank_limit = top_k
        if self._reranker:
            rerank_limit = self._config.reranker.top_n_input

        points = self._qdrant.rest_client.query_points(
            collection_name=collection,
            prefetch=prefetches,
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=rerank_limit,
            with_payload=True,
        ).points

        return self._points_to_results(points)

    # ── Client-side manual RRF ─────────────────────────────────────────────

    def _search_with_manual_rrf(
        self,
        collection: str,
        dense_vec: Any,
        sparse_vec: SparseVector | None,
        qdrant_filter: qmodels.Filter | None,
        top_k: int,
    ) -> list[SearchResult]:
        branch_limit = self._hs_cfg.top_k_per_branch

        dense_hits = self._qdrant.rest_client.search(  # type: ignore[attr-defined]
            collection_name=collection,
            query_vector=qmodels.NamedVector(
                name=DENSE_VECTOR_NAME,
                vector=dense_vec.tolist(),
            ),
            query_filter=qdrant_filter,
            limit=branch_limit,
            with_payload=True,
        )

        sparse_hits = []
        if sparse_vec and sparse_vec.indices:
            sparse_hits = self._qdrant.rest_client.search(  # type: ignore[attr-defined]
                collection_name=collection,
                query_vector=qmodels.NamedSparseVector(
                    name=SPARSE_VECTOR_NAME,
                    vector=qmodels.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                ),
                query_filter=qdrant_filter,
                limit=branch_limit,
                with_payload=True,
            )

        merged = self._rrf_merge(dense_hits, sparse_hits)

        rerank_limit = top_k
        if self._reranker:
            rerank_limit = self._config.reranker.top_n_input

        return merged[:rerank_limit]

    def _rrf_merge(
        self,
        dense_hits: list[Any],
        sparse_hits: list[Any],
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of two hit lists."""
        k = self._hs_cfg.rrf_k
        dense_w = self._hs_cfg.dense_weight
        sparse_w = self._hs_cfg.sparse_weight

        score_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            pid = str(hit.id)
            rrf_score = dense_w / (k + rank + 1)
            if pid not in score_map:
                score_map[pid] = {
                    "hit": hit,
                    "dense_score": hit.score or 0.0,
                    "sparse_score": 0.0,
                    "rrf": 0.0,
                }
            score_map[pid]["rrf"] += rrf_score
            score_map[pid]["dense_score"] = hit.score or 0.0

        for rank, hit in enumerate(sparse_hits):
            pid = str(hit.id)
            rrf_score = sparse_w / (k + rank + 1)
            if pid not in score_map:
                score_map[pid] = {
                    "hit": hit,
                    "dense_score": 0.0,
                    "sparse_score": hit.score or 0.0,
                    "rrf": 0.0,
                }
            score_map[pid]["rrf"] += rrf_score
            score_map[pid]["sparse_score"] = hit.score or 0.0

        sorted_entries = sorted(
            score_map.values(), key=lambda e: e["rrf"], reverse=True,
        )

        results: list[SearchResult] = []
        for entry in sorted_entries:
            hit = entry["hit"]
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    score=entry["rrf"],
                    dense_score=entry["dense_score"],
                    sparse_score=entry["sparse_score"],
                    payload=payload,
                ),
            )

        return results

    # ── Filter building ────────────────────────────────────────────────────

    def _build_filter(
        self, filters: dict[str, Any],
    ) -> qmodels.Filter:
        """Build Qdrant Filter from a dict of field→value pairs.

        The ``model`` field is normalized via model_aliases.yaml so that
        user queries like "m2040dn" match the canonical name stored in
        the payload.
        """
        must_conditions: list[qmodels.FieldCondition] = []

        for key, value in filters.items():
            if value is None:
                continue

            if key == "model":
                value = self._normalize_model(str(value))

            if isinstance(value, list):
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchAny(any=value),
                    ),
                )
            else:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value),
                    ),
                )

        return qmodels.Filter(must=must_conditions)  # type: ignore[arg-type]

    # ── Model alias normalization ──────────────────────────────────────────

    def _load_model_aliases(self) -> None:
        if not _MODEL_ALIASES_PATH.exists():
            logger.warning("model_aliases.yaml не найден: %s", _MODEL_ALIASES_PATH)
            return

        with open(_MODEL_ALIASES_PATH, encoding="utf-8") as f:
            raw: dict[str, list[str]] = yaml.safe_load(f) or {}

        for canonical, aliases in raw.items():
            canonical_lower = canonical.lower()
            self._model_aliases[canonical_lower] = canonical
            for alias in aliases:
                self._model_aliases[alias.lower()] = canonical

        logger.debug(
            "Загружено %d алиасов моделей из model_aliases.yaml",
            len(self._model_aliases),
        )

    def _normalize_model(self, model_query: str) -> str:
        """Return canonical model name if an alias matches, else original."""
        lower = model_query.strip().lower()
        if lower in self._model_aliases:
            canonical = self._model_aliases[lower]
            logger.debug("Модель нормализована: %r → %r", model_query, canonical)
            return canonical
        return model_query

    # ── Qdrant version detection ───────────────────────────────────────────

    def _supports_fusion(self) -> bool:
        """Check if Qdrant server supports the Query API with fusion."""
        if self._qdrant_version is not None:
            return self._qdrant_version >= _MIN_QDRANT_FUSION_VERSION

        try:
            self._qdrant_version = self._detect_qdrant_version()
            supported = self._qdrant_version >= _MIN_QDRANT_FUSION_VERSION
            logger.info(
                "Qdrant версия %s — fusion %s",
                ".".join(str(v) for v in self._qdrant_version),
                "поддерживается" if supported else "не поддерживается",
            )
            return supported
        except Exception:
            logger.warning(
                "Не удалось определить версию Qdrant — используем ручной RRF"
            )
            self._qdrant_version = (0, 0)
            return False

    def _detect_qdrant_version(self) -> tuple[int, ...]:
        """Query Qdrant telemetry endpoint for version info."""
        try:
            info = self._qdrant.rest_client.http.cluster_api.cluster_status()  # type: ignore[attr-defined]
            version_str = getattr(info, "version", None)
        except Exception:
            version_str = None

        if not version_str:
            try:
                import httpx

                qcfg = self._config.qdrant
                resp = httpx.get(
                    f"http://{qcfg.host}:{qcfg.port}/", timeout=5.0,
                )
                version_str = resp.json().get("version", "")
            except Exception:
                version_str = ""

        if not version_str:
            return (0, 0)

        match = re.search(r"(\d+)\.(\d+)", str(version_str))
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _points_to_results(points: list[Any]) -> list[SearchResult]:
        """Convert Qdrant ScoredPoint list to SearchResult list."""
        results: list[SearchResult] = []
        for pt in points:
            payload = pt.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(pt.id),
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    score=pt.score if pt.score is not None else 0.0,
                    dense_score=0.0,
                    sparse_score=0.0,
                    payload=payload,
                ),
            )
        return results
