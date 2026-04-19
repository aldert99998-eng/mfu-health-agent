"""RAG evaluation engine — Track D, Level 2.

Deterministic evaluation of retrieval quality using IR metrics:
Recall@5, Recall@10, MRR, Precision@5, nDCG@10.

Results are grouped by scenario and compared against acceptance
thresholds defined in ``RAGConfig.evaluation.acceptance_thresholds``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from config.loader import RAGConfig
    from rag.search import HybridSearcher

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Pydantic models ───────────────────────────────────────────────────────────


class EvalQuery(BaseModel):
    """Single evaluation query from the dataset."""

    id: str
    query: str
    expected_chunks: list[str]
    must_contain_codes: list[str] = Field(default_factory=list)
    must_contain_components: list[str] = Field(default_factory=list)
    scenario: str = "general"
    difficulty: str = "medium"
    collection: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryEvalResult(BaseModel):
    """Evaluation result for a single query."""

    query_id: str
    query: str
    scenario: str
    difficulty: str
    expected_chunks: list[str]
    retrieved_chunks: list[str]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_5: float
    ndcg_at_10: float
    hit: bool


class ScenarioMetrics(BaseModel):
    """Aggregated metrics for a scenario group."""

    scenario: str
    num_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_5: float
    ndcg_at_10: float


class ThresholdCheck(BaseModel):
    """Result of checking a metric against its acceptance threshold."""

    metric: str
    value: float
    threshold: float
    passed: bool


class MetricDelta(BaseModel):
    """Delta between current and previous report for one metric."""

    metric: str
    current: float
    previous: float
    delta: float
    improved: bool


class EvalReport(BaseModel):
    """Complete evaluation report."""

    timestamp: str
    duration_seconds: float
    dataset_path: str
    num_queries: int
    collection: str
    use_reranker: bool

    recall_at_5: float
    recall_at_10: float
    mrr: float
    precision_at_5: float
    ndcg_at_10: float

    per_scenario: list[ScenarioMetrics]
    per_query: list[QueryEvalResult]
    threshold_checks: list[ThresholdCheck] = Field(default_factory=list)
    all_thresholds_passed: bool = True
    deltas: list[MetricDelta] = Field(default_factory=list)


# ── IR metric functions (deterministic, pure) ─────────────────────────────────


def _recall_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for cid in retrieved[:k] if cid in expected)
    return hits / len(expected)


def _precision_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in expected)
    return hits / k


def _mrr(retrieved: list[str], expected: set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in expected:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved: list[str], expected: set[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in expected:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 1.0
    return dcg / idcg


# ── RAGEvaluator ──────────────────────────────────────────────────────────────


class RAGEvaluator:
    """Deterministic RAG retrieval quality evaluator.

    Parameters
    ----------
    searcher:
        Initialized HybridSearcher instance.
    config:
        Full RAGConfig — uses ``evaluation`` section.
    """

    def __init__(self, searcher: HybridSearcher, config: RAGConfig) -> None:
        self._searcher = searcher
        self._config = config
        self._eval_cfg = config.evaluation
        self._thresholds = config.evaluation.acceptance_thresholds

        self._dataset_path = _PROJECT_ROOT / self._eval_cfg.dataset_path
        self._history_dir = _PROJECT_ROOT / self._eval_cfg.save_history_path
        self._history_dir.mkdir(parents=True, exist_ok=True)

    # ── run_eval ──────────────────────────────────────────────────────────

    def run_eval(
        self,
        collection: str,
        *,
        use_reranker: bool = True,
        dataset_path: str | Path | None = None,
    ) -> EvalReport:
        """Load eval dataset, run search for each query, compute metrics.

        Parameters
        ----------
        collection:
            Qdrant collection to search.
        use_reranker:
            Whether to enable cross-encoder reranking.
        dataset_path:
            Override for the eval dataset YAML path.

        Returns
        -------
        EvalReport
        """
        t0 = time.perf_counter()
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        ds_path = Path(dataset_path) if dataset_path else self._dataset_path
        queries = self._load_dataset(ds_path)

        per_query: list[QueryEvalResult] = []
        for eq in queries:
            result = self._evaluate_query(
                eq, collection, use_reranker=use_reranker,
            )
            per_query.append(result)

        agg = self._aggregate_metrics(per_query)
        per_scenario = self._group_by_scenario(per_query)

        duration = time.perf_counter() - t0

        report = EvalReport(
            timestamp=ts,
            duration_seconds=round(duration, 3),
            dataset_path=str(ds_path),
            num_queries=len(queries),
            collection=collection,
            use_reranker=use_reranker,
            **agg,  # type: ignore[arg-type]
            per_scenario=per_scenario,
            per_query=per_query,
        )

        report.threshold_checks = self.check_thresholds(report)
        report.all_thresholds_passed = all(
            tc.passed for tc in report.threshold_checks
        )

        report.deltas = self.delta_vs_previous(report)

        logger.info(
            "Evaluation: %d запросов за %.1f с — Recall@5=%.3f MRR=%.3f %s",
            report.num_queries,
            report.duration_seconds,
            report.recall_at_5,
            report.mrr,
            "PASS" if report.all_thresholds_passed else "FAIL",
        )

        return report

    # ── save_report ───────────────────────────────────────────────────────

    def save_report(self, report: EvalReport) -> Path:
        """Persist report as JSON to the history directory.

        Returns the path to the saved file.
        """
        safe_ts = report.timestamp.replace(":", "-")
        filename = f"eval_{safe_ts}.json"
        path = self._history_dir / filename

        path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Отчёт сохранён: %s", path)
        return path

    # ── get_history ───────────────────────────────────────────────────────

    def get_history(self, last_n: int = 10) -> list[EvalReport]:
        """Load last *N* reports from the history directory, newest first."""
        files = sorted(
            self._history_dir.glob("eval_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:last_n]

        reports: list[EvalReport] = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                reports.append(EvalReport.model_validate(data))
            except Exception:
                logger.warning("Не удалось загрузить отчёт: %s", f)

        return reports

    # ── check_thresholds ──────────────────────────────────────────────────

    def check_thresholds(self, report: EvalReport) -> list[ThresholdCheck]:
        """Compare report metrics against acceptance thresholds."""
        checks: list[ThresholdCheck] = []

        threshold_map = {
            "recall_at_5": self._thresholds.recall_at_5,
            "mrr": self._thresholds.mrr,
        }

        for metric, threshold in threshold_map.items():
            value = getattr(report, metric)
            checks.append(
                ThresholdCheck(
                    metric=metric,
                    value=round(value, 4),
                    threshold=threshold,
                    passed=value >= threshold,
                ),
            )

        return checks

    # ── delta_vs_previous ─────────────────────────────────────────────────

    def delta_vs_previous(self, report: EvalReport) -> list[MetricDelta]:
        """Compute metric deltas between *report* and the most recent saved report."""
        history = self.get_history(last_n=1)
        if not history:
            return []

        prev = history[0]
        metrics = (
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "precision_at_5",
            "ndcg_at_10",
        )

        deltas: list[MetricDelta] = []
        for m in metrics:
            cur_val = getattr(report, m)
            prev_val = getattr(prev, m)
            delta = cur_val - prev_val
            deltas.append(
                MetricDelta(
                    metric=m,
                    current=round(cur_val, 4),
                    previous=round(prev_val, 4),
                    delta=round(delta, 4),
                    improved=delta > 0,
                ),
            )

        return deltas

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_dataset(self, path: Path) -> list[EvalQuery]:
        """Load evaluation dataset from YAML."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        queries_raw: list[dict[str, Any]] = raw.get("queries", [])
        queries = [EvalQuery.model_validate(q) for q in queries_raw]
        logger.debug("Загружено %d eval-запросов из %s", len(queries), path)
        return queries

    def _evaluate_query(
        self,
        eq: EvalQuery,
        collection: str,
        *,
        use_reranker: bool,
    ) -> QueryEvalResult:
        """Run search for one eval query and compute per-query metrics."""
        filters = eq.filters if eq.filters else None
        target_collection = eq.collection or collection

        results = self._searcher.search(
            eq.query,
            target_collection,
            top_k=10,
            filters=filters,
            use_reranker=use_reranker,
        )

        retrieved = [r.chunk_id for r in results]
        expected = set(eq.expected_chunks)

        r5 = _recall_at_k(retrieved, expected, 5)
        r10 = _recall_at_k(retrieved, expected, 10)
        m = _mrr(retrieved, expected)
        p5 = _precision_at_k(retrieved, expected, 5)
        n10 = _ndcg_at_k(retrieved, expected, 10)

        return QueryEvalResult(
            query_id=eq.id,
            query=eq.query,
            scenario=eq.scenario,
            difficulty=eq.difficulty,
            expected_chunks=eq.expected_chunks,
            retrieved_chunks=retrieved,
            recall_at_5=round(r5, 4),
            recall_at_10=round(r10, 4),
            mrr=round(m, 4),
            precision_at_5=round(p5, 4),
            ndcg_at_10=round(n10, 4),
            hit=r10 > 0,
        )

    @staticmethod
    def _aggregate_metrics(
        per_query: list[QueryEvalResult],
    ) -> dict[str, float]:
        """Compute macro-averaged metrics across all queries."""
        if not per_query:
            return {
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "precision_at_5": 0.0,
                "ndcg_at_10": 0.0,
            }

        n = len(per_query)
        return {
            "recall_at_5": round(sum(q.recall_at_5 for q in per_query) / n, 4),
            "recall_at_10": round(sum(q.recall_at_10 for q in per_query) / n, 4),
            "mrr": round(sum(q.mrr for q in per_query) / n, 4),
            "precision_at_5": round(sum(q.precision_at_5 for q in per_query) / n, 4),
            "ndcg_at_10": round(sum(q.ndcg_at_10 for q in per_query) / n, 4),
        }

    @staticmethod
    def _group_by_scenario(
        per_query: list[QueryEvalResult],
    ) -> list[ScenarioMetrics]:
        """Group per-query results by scenario and aggregate."""
        groups: dict[str, list[QueryEvalResult]] = {}
        for q in per_query:
            groups.setdefault(q.scenario, []).append(q)

        scenarios: list[ScenarioMetrics] = []
        for scenario, items in sorted(groups.items()):
            n = len(items)
            scenarios.append(
                ScenarioMetrics(
                    scenario=scenario,
                    num_queries=n,
                    recall_at_5=round(sum(q.recall_at_5 for q in items) / n, 4),
                    recall_at_10=round(sum(q.recall_at_10 for q in items) / n, 4),
                    mrr=round(sum(q.mrr for q in items) / n, 4),
                    precision_at_5=round(sum(q.precision_at_5 for q in items) / n, 4),
                    ndcg_at_10=round(sum(q.ndcg_at_10 for q in items) / n, 4),
                ),
            )

        return scenarios
