"""BGE-M3 embedder — Track D, Level 2.

Wraps FlagEmbedding's BGEM3FlagModel to produce dense and sparse
vectors in a single forward pass.  Used by the ingestion pipeline
(index-time) and HybridSearcher (query-time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from qdrant_client.http.models import SparseVector

if TYPE_CHECKING:
    from config.loader import EmbeddingsConfig

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"


# ── Custom errors ───────────────────────────────────────────────────────────


class ModelLoadError(Exception):
    """Raised when the embedding model cannot be downloaded or loaded."""


class OOMError(Exception):
    """Raised on out-of-memory during encoding."""


# ── EmbeddingResult ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Result of encoding a batch of texts."""

    dense: np.ndarray
    sparse: list[SparseVector] | None = field(default=None)


# ── BGEEmbedder ─────────────────────────────────────────────────────────────


class BGEEmbedder:
    """High-level wrapper around ``BGEM3FlagModel``.

    Parameters
    ----------
    config:
        ``EmbeddingsConfig`` from ``RAGConfig.embeddings``.
    """

    def __init__(self, config: EmbeddingsConfig) -> None:
        self._config = config
        self._model_name = config.model
        self._device = config.device
        self._batch_size = config.batch_size
        self._normalize = config.normalize
        self._max_length = config.max_length
        self._fp16 = config.fp16

        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Загрузка модели %s (device=%s, fp16=%s, cache=%s)…",
            self._model_name,
            self._device,
            self._fp16,
            MODELS_CACHE_DIR,
        )

        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                model_name_or_path=self._model_name,
                normalize_embeddings=self._normalize,
                use_fp16=self._fp16,
                devices=[self._device] if self._device else None,
                cache_dir=str(MODELS_CACHE_DIR),
                batch_size=self._batch_size,
                passage_max_length=self._max_length,
                query_max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except ImportError as exc:
            raise ModelLoadError(
                "FlagEmbedding не установлен: pip install FlagEmbedding"
            ) from exc
        except MemoryError as exc:
            raise OOMError(
                f"Недостаточно памяти для загрузки {self._model_name}. "
                "Попробуйте уменьшить batch_size или включить fp16."
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Не удалось загрузить модель {self._model_name}: {exc}"
            ) from exc

        logger.info("Модель %s загружена.", self._model_name)

    # ── encode ──────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        *,
        return_sparse: bool = True,
    ) -> EmbeddingResult:
        """Encode a list of texts into dense (and optionally sparse) vectors.

        Processes in batches of ``config.batch_size``.

        Returns
        -------
        EmbeddingResult
            ``.dense`` — ``np.ndarray`` of shape ``(n, dim)``.
            ``.sparse`` — list of ``SparseVector`` or ``None``.

        Raises
        ------
        OOMError
            If encoding runs out of memory.
        """
        if not texts:
            return EmbeddingResult(
                dense=np.empty((0, 0), dtype=np.float32),
                sparse=[] if return_sparse else None,
            )

        try:
            output = self._model.encode(
                texts,
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
            )
        except (MemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower() or isinstance(exc, MemoryError):
                raise OOMError(
                    f"OOM при кодировании {len(texts)} текстов (batch_size={self._batch_size}). "
                    "Уменьшите batch_size в конфиге embeddings."
                ) from exc
            raise

        dense: np.ndarray = output["dense_vecs"]

        sparse_vectors: list[SparseVector] | None = None
        if return_sparse:
            lexical_weights: list[dict[str, float]] = output["lexical_weights"]
            sparse_vectors = [
                self._lexical_to_sparse(lw) for lw in lexical_weights
            ]

        logger.debug(
            "Закодировано %d текстов: dense=%s, sparse=%s",
            len(texts),
            dense.shape,
            len(sparse_vectors) if sparse_vectors else 0,
        )

        return EmbeddingResult(dense=dense, sparse=sparse_vectors)

    # ── encode_query ────────────────────────────────────────────────────────

    def encode_query(
        self,
        query: str,
        *,
        return_sparse: bool = True,
    ) -> tuple[np.ndarray, SparseVector | None]:
        """Convenience method for a single query string.

        Returns ``(dense_vector_1d, sparse_vector_or_none)``.
        """
        result = self.encode([query], return_sparse=return_sparse)
        dense_vec = result.dense[0]
        sparse_vec = result.sparse[0] if result.sparse else None
        return dense_vec, sparse_vec

    # ── embedding_version ───────────────────────────────────────────────────

    def embedding_version(self) -> str:
        """Return a version string like ``bge-m3@<version>`` for payload tagging."""
        version = "unknown"
        try:
            if hasattr(self._model, "model") and hasattr(self._model.model, "config"):
                cfg = self._model.model.config
                if hasattr(cfg, "model_version"):
                    version = str(cfg.model_version)
                elif hasattr(cfg, "_name_or_path"):
                    version = str(cfg._name_or_path).rsplit("/", 1)[-1]
        except Exception:
            pass

        short_name = self._model_name.rsplit("/", 1)[-1]
        return f"{short_name}@{version}"

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _lexical_to_sparse(lw: dict[str, float]) -> SparseVector:
        """Convert FlagEmbedding lexical_weights dict to Qdrant SparseVector."""
        if not lw:
            return SparseVector(indices=[], values=[])

        indices: list[int] = []
        values: list[float] = []
        for token_id_str, weight in lw.items():
            indices.append(int(token_id_str))
            values.append(float(weight))

        return SparseVector(indices=indices, values=values)
