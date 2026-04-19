"""E2E verification for RAG evaluation (Phase 4.4 — evaluation checks).

Checks from the playbook:
1. Eval on empty collection → all metrics 0, report saved.
2. Eval on seeded collection → metrics > 0.
3. Two consecutive evals on the same data → identical metrics.

Requires: running Qdrant on localhost:6333 and BGE-M3 model cached.
Run:  python -m pytest tests/e2e/test_evaluation.py -v -s
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# ── Small fixture dataset (3 queries covering 2 scenarios) ────────────────

_EMPTY_COLLECTION = f"_test_eval_empty_{uuid.uuid4().hex[:8]}"
_SEEDED_COLLECTION = f"_test_eval_seeded_{uuid.uuid4().hex[:8]}"

_CHUNKS = [
    {
        "text": "Error Code C6000 — Abnormal fuser heater temperature. The fuser unit fails to reach target temperature within 30 seconds. Replace the fuser thermistor or the fuser heater lamp.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C6000"],
        "doc_type": "service_manual",
    },
    {
        "text": "Error Code C6020 — Fuser overheat. The fuser temperature exceeds the upper safety limit. Check thermistor wiring and fuser unit connector CN201.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": ["C6020"],
        "doc_type": "service_manual",
    },
    {
        "text": "The drum unit (OPC drum) should be replaced every 200,000 pages. Check drum surface for scratches or coating damage.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
    },
    {
        "text": "Developer unit replacement procedure: 1) Open front cover. 2) Remove developer unit by pulling the green handle. 3) Install the new developer unit.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
    },
    {
        "text": "HP LaserJet Pro M404n error 49.4C02 — Critical firmware error. Power-cycle the printer. If the error persists, update firmware.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": ["49.4C02"],
        "doc_type": "service_manual",
    },
    {
        "text": "HP M404n fuser replacement: open the rear door. Release the two blue fuser latches. Pull the fuser straight out. Insert the replacement.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": [],
        "doc_type": "service_manual",
    },
    {
        "text": "HP LaserJet M404n 13.B2 paper jam in tray 2. Remove tray 2 and clear jammed paper. Check the pickup roller and separation pad for wear.",
        "vendor": "HP",
        "model": "HP LaserJet Pro M404n",
        "error_codes": ["13.B2"],
        "doc_type": "service_manual",
    },
    {
        "text": "Paper feed roller replacement: Remove cassette tray. Release the roller retainer clips. Replace the pickup roller and feed roller set.",
        "vendor": "Kyocera",
        "model": "Kyocera TASKalfa 3253ci",
        "error_codes": [],
        "doc_type": "service_manual",
    },
]

_CHUNK_IDS = [
    str(uuid.uuid5(uuid.NAMESPACE_DNS, f"eval_chunk_{i}"))
    for i in range(len(_CHUNKS))
]


def _make_eval_dataset(tmp_dir: Path) -> Path:
    """Write a small eval dataset YAML referencing our chunk IDs."""
    dataset = {
        "queries": [
            {
                "id": "q_001",
                "query": "ошибка C6000 фьюзер не нагревается Kyocera",
                "expected_chunks": [_CHUNK_IDS[0]],
                "must_contain_codes": ["C6000"],
                "scenario": "error_code_lookup",
                "difficulty": "easy",
            },
            {
                "id": "q_002",
                "query": "как заменить девелопер на TASKalfa 3253ci",
                "expected_chunks": [_CHUNK_IDS[3]],
                "scenario": "procedure_lookup",
                "difficulty": "medium",
            },
            {
                "id": "q_003",
                "query": "HP M404n ошибка 49.4C02 firmware",
                "expected_chunks": [_CHUNK_IDS[4]],
                "must_contain_codes": ["49.4C02"],
                "scenario": "error_code_lookup",
                "difficulty": "easy",
            },
        ],
    }
    path = tmp_dir / "eval_dataset.yaml"
    path.write_text(yaml.dump(dataset, allow_unicode=True), encoding="utf-8")
    return path


# ── Availability checks ──────────────────────────────────────────────────


def _qdrant_available() -> bool:
    try:
        c = QdrantClient(host="localhost", port=6333, prefer_grpc=False, timeout=3)
        c.get_collections()
        return True
    except Exception:
        return False


def _embedder_available() -> bool:
    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: F401

        return True
    except ImportError:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(), reason="Qdrant not running on localhost:6333",
)
requires_embedder = pytest.mark.skipif(
    not _embedder_available(), reason="FlagEmbedding not installed",
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qdrant_rest():
    return QdrantClient(host="localhost", port=6333, prefer_grpc=False, timeout=30)


@pytest.fixture(scope="module")
def embedder():
    from config.loader import EmbeddingsConfig
    from rag.embeddings import BGEEmbedder

    cfg = EmbeddingsConfig(device="cpu", fp16=False)
    return BGEEmbedder(cfg)


@pytest.fixture(scope="module")
def rag_config():
    from config.loader import RAGConfig

    cfg = RAGConfig()
    cfg.qdrant.host = "localhost"
    cfg.qdrant.port = 6333
    cfg.qdrant.prefer_grpc = False
    return cfg


@pytest.fixture(scope="module")
def qdrant_manager(rag_config):
    from rag.qdrant_client import QdrantManager

    return QdrantManager(rag_config)


@pytest.fixture(scope="module")
def searcher(embedder, qdrant_manager, rag_config):
    from rag.search import HybridSearcher

    return HybridSearcher(qdrant_manager, embedder, rag_config)


@pytest.fixture(scope="module")
def tmp_dir() -> None:
    d = Path(tempfile.mkdtemp(prefix="eval_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def eval_dataset_path(tmp_dir):
    return _make_eval_dataset(tmp_dir)


@pytest.fixture(scope="module")
def history_dir(tmp_dir):
    d = tmp_dir / "eval_history"
    d.mkdir()
    return d


def _create_empty_collection(qdrant_rest) -> None:
    """Create empty collection with dense+sparse vectors."""
    qdrant_rest.create_collection(
        collection_name=_EMPTY_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=1024, distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF),
        },
    )


def _create_seeded_collection(qdrant_rest, embedder) -> None:
    """Create collection with 8 chunks."""
    qdrant_rest.create_collection(
        collection_name=_SEEDED_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=1024, distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF),
        },
    )

    for field_name in ("vendor", "model", "error_codes", "doc_type"):
        with contextlib.suppress(Exception):
            qdrant_rest.create_payload_index(
                collection_name=_SEEDED_COLLECTION,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    texts = [c["text"] for c in _CHUNKS]
    result = embedder.encode(texts, return_sparse=True)

    points = []
    for i, chunk in enumerate(_CHUNKS):
        vectors = {"dense": result.dense[i].tolist()}
        sv = result.sparse[i] if result.sparse else None
        if sv and sv.indices:
            vectors["sparse"] = qmodels.SparseVector(
                indices=sv.indices, values=sv.values,
            )

        payload = {
            "text": chunk["text"],
            "vendor": chunk["vendor"],
            "model": chunk["model"],
            "error_codes": chunk["error_codes"],
            "doc_type": chunk["doc_type"],
            "document_id": f"test_doc_{chunk['vendor'].lower()}",
        }
        points.append(
            qmodels.PointStruct(id=_CHUNK_IDS[i], vector=vectors, payload=payload),
        )

    qdrant_rest.upsert(collection_name=_SEEDED_COLLECTION, points=points)

    info = qdrant_rest.get_collection(_SEEDED_COLLECTION)
    assert info.points_count == len(_CHUNKS)


@pytest.fixture(scope="module")
def collections(qdrant_rest, embedder) -> None:
    """Create both empty and seeded collections, tear down after tests."""
    _create_empty_collection(qdrant_rest)
    _create_seeded_collection(qdrant_rest, embedder)

    yield {
        "empty": _EMPTY_COLLECTION,
        "seeded": _SEEDED_COLLECTION,
    }

    with contextlib.suppress(Exception):
        qdrant_rest.delete_collection(_EMPTY_COLLECTION)
    with contextlib.suppress(Exception):
        qdrant_rest.delete_collection(_SEEDED_COLLECTION)


# ── Tests ─────────────────────────────────────────────────────────────────


@requires_qdrant
@requires_embedder
class TestEvaluation:
    """Phase 4.4 evaluation verification suite."""

    def _make_evaluator(self, searcher, rag_config, eval_dataset_path, history_dir):
        from rag.evaluation import RAGEvaluator

        rag_config.evaluation.dataset_path = str(eval_dataset_path)
        rag_config.evaluation.save_history_path = str(history_dir)
        return RAGEvaluator(searcher, rag_config)

    def test_eval_on_empty_collection_all_zeros(
        self, searcher, rag_config, eval_dataset_path, history_dir, collections,
    ) -> None:
        """Eval on empty collection → all metrics 0, report saved."""
        evaluator = self._make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            collections["empty"],
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        assert report.num_queries == 3
        assert report.recall_at_5 == 0.0
        assert report.recall_at_10 == 0.0
        assert report.mrr == 0.0
        assert report.precision_at_5 == 0.0
        assert report.ndcg_at_10 == 0.0

        for qr in report.per_query:
            assert qr.recall_at_5 == 0.0
            assert qr.mrr == 0.0
            assert not qr.hit

        saved_path = evaluator.save_report(report)
        assert saved_path.exists()

        loaded = json.loads(saved_path.read_text(encoding="utf-8"))
        assert loaded["recall_at_5"] == 0.0
        assert loaded["mrr"] == 0.0

    def test_eval_on_seeded_collection_metrics_positive(
        self, searcher, rag_config, eval_dataset_path, history_dir, collections,
    ) -> None:
        """Eval on seeded collection → at least some metrics > 0."""
        evaluator = self._make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            collections["seeded"],
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        assert report.num_queries == 3
        assert report.recall_at_5 > 0.0, f"recall@5 should be > 0, got {report.recall_at_5}"
        assert report.recall_at_10 > 0.0, f"recall@10 should be > 0, got {report.recall_at_10}"
        assert report.mrr > 0.0, f"MRR should be > 0, got {report.mrr}"

        hits = sum(1 for qr in report.per_query if qr.hit)
        assert hits > 0, "Expected at least one query to hit"

        assert len(report.per_scenario) >= 1
        assert len(report.threshold_checks) >= 2

    def test_deterministic_two_runs_identical(
        self, searcher, rag_config, eval_dataset_path, history_dir, collections,
    ) -> None:
        """Two consecutive evals on the same data → identical metrics."""
        evaluator = self._make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report_1 = evaluator.run_eval(
            collections["seeded"],
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )
        report_2 = evaluator.run_eval(
            collections["seeded"],
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        assert report_1.recall_at_5 == report_2.recall_at_5
        assert report_1.recall_at_10 == report_2.recall_at_10
        assert report_1.mrr == report_2.mrr
        assert report_1.precision_at_5 == report_2.precision_at_5
        assert report_1.ndcg_at_10 == report_2.ndcg_at_10

        for q1, q2 in zip(report_1.per_query, report_2.per_query, strict=True):
            assert q1.query_id == q2.query_id
            assert q1.retrieved_chunks == q2.retrieved_chunks
            assert q1.recall_at_5 == q2.recall_at_5
            assert q1.mrr == q2.mrr
            assert q1.ndcg_at_10 == q2.ndcg_at_10

    def test_save_and_load_history(
        self, searcher, rag_config, eval_dataset_path, history_dir, collections,
    ) -> None:
        """save_report + get_history round-trips correctly."""
        evaluator = self._make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            collections["seeded"],
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )
        evaluator.save_report(report)

        history = evaluator.get_history(last_n=5)
        assert len(history) >= 1

        latest = history[0]
        assert latest.recall_at_5 == report.recall_at_5
        assert latest.mrr == report.mrr
        assert latest.num_queries == report.num_queries

    def test_delta_vs_previous(
        self, searcher, rag_config, eval_dataset_path, history_dir, collections,
    ) -> None:
        """delta_vs_previous computes deltas against saved report."""
        evaluator = self._make_evaluator(
            searcher, rag_config, eval_dataset_path, history_dir,
        )

        report = evaluator.run_eval(
            collections["seeded"],
            use_reranker=False,
            dataset_path=eval_dataset_path,
        )

        deltas = evaluator.delta_vs_previous(report)
        assert len(deltas) == 5

        for d in deltas:
            assert d.metric in (
                "recall_at_5", "recall_at_10", "mrr", "precision_at_5", "ndcg_at_10",
            )
            assert d.delta == round(d.current - d.previous, 4)
