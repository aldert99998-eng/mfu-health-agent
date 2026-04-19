"""BGE cross-encoder reranker — Track D, Level 2.

Wraps FlagEmbedding's FlagReranker to re-score candidate chunks
returned by HybridSearcher before they reach the LLM context.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.loader import RerankerConfig

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path(__file__).resolve().parent.parent / "models"


# ── Custom errors ───────────────────────────────────────────────────────────


class RerankerLoadError(Exception):
    """Raised when the reranker model cannot be loaded."""


class RerankerScoringError(Exception):
    """Raised on failure during cross-encoder scoring."""


# ── ScoredChunk ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A search result that has been re-scored by the cross-encoder."""

    chunk_id: str
    document_id: str
    text: str
    rerank_score: float
    original_score: float
    payload: dict[str, Any]


# ── BGEReranker ─────────────────────────────────────────────────────────────


class BGEReranker:
    """Cross-encoder reranker using ``BAAI/bge-reranker-v2-m3``.

    Parameters
    ----------
    config:
        ``RerankerConfig`` from ``RAGConfig.reranker``.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._model_name = config.model
        self._device = config.device
        self._batch_size = config.batch_size
        self._top_n_input = config.top_n_input
        self._top_n_output = config.top_n_output

        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Загрузка реранкера %s (device=%s)…",
            self._model_name,
            self._device,
        )

        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                model_name_or_path=self._model_name,
                use_fp16=self._device != "cpu",
                cache_dir=str(MODELS_CACHE_DIR),
                device=self._device,
            )
        except ImportError as exc:
            raise RerankerLoadError(
                "FlagEmbedding не установлен: pip install FlagEmbedding"
            ) from exc
        except Exception as exc:
            raise RerankerLoadError(
                f"Не удалось загрузить реранкер {self._model_name}: {exc}"
            ) from exc

        logger.info("Реранкер %s загружен.", self._model_name)

    # ── rerank ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_n: int | None = None,
    ) -> list[ScoredChunk]:
        """Re-score candidates with the cross-encoder.

        Parameters
        ----------
        query:
            User query string.
        candidates:
            List of dicts with keys: ``chunk_id``, ``document_id``,
            ``text``, ``score``, ``payload``.
        top_n:
            Max results to return.  Defaults to ``config.top_n_output``.

        Returns
        -------
        list[ScoredChunk]
            Sorted by ``rerank_score`` descending, truncated to *top_n*.
        """
        if not candidates:
            return []

        top_n = top_n or self._top_n_output
        candidates = candidates[: self._top_n_input]

        t0 = time.perf_counter()

        pairs = [[query, c["text"]] for c in candidates]

        try:
            scores = self._compute_scores(pairs)
        except Exception as exc:
            raise RerankerScoringError(
                f"Ошибка скоринга реранкера на {len(pairs)} парах: {exc}"
            ) from exc

        scored: list[ScoredChunk] = []
        for candidate, score in zip(candidates, scores, strict=True):
            scored.append(
                ScoredChunk(
                    chunk_id=candidate["chunk_id"],
                    document_id=candidate["document_id"],
                    text=candidate["text"],
                    rerank_score=float(score),
                    original_score=float(candidate.get("score", 0.0)),
                    payload=candidate.get("payload", {}),
                )
            )

        scored.sort(key=lambda s: s.rerank_score, reverse=True)
        result = scored[:top_n]

        elapsed = time.perf_counter() - t0
        logger.debug(
            "Реранкинг: %d кандидатов → %d результатов за %.3f с",
            len(candidates),
            len(result),
            elapsed,
        )

        return result

    # ── Private helpers ─────────────────────────────────────────────────────

    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        """Score query-text pairs in batches."""
        all_scores: list[float] = []

        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            batch_scores = self._model.compute_score(
                batch, normalize=True,
            )
            if isinstance(batch_scores, (int, float)):
                all_scores.append(float(batch_scores))
            else:
                all_scores.extend(float(s) for s in batch_scores)

        return all_scores
