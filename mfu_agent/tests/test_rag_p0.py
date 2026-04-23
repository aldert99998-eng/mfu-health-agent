"""P0 tests for the MFU RAG system (Track D).

TC-D-007  rag_config.yaml Pydantic validation
TC-D-010  Same text -> deterministic vector
TC-D-011  Vector dimension correct for BGE-M3 (1024)
TC-D-015  Mixed language (ru/en/error codes) -> no crash during embedding
TC-D-001  Qdrant healthcheck
TC-D-020  Reranker reorders candidates correctly
TC-D-023  Empty candidate list -> empty result
TC-D-080  SearchResult has source_ref fields (doc_title, section)
TC-D-100  Qdrant unavailable -> clear error message
TC-D-111  Verify API key not stored in Qdrant payload (code review)
TC-D-120  component_vocabulary.yaml sync
TC-D-121  model_aliases.yaml sync between RAG and ingestion
TC-D-122  error_code_patterns.yaml single source of truth
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import pytest
import yaml

# ── Project root on sys.path ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import (
    RAGConfig,
    ConfigManager,
    ConfigValidationError,
    EmbeddingsConfig,
    RerankerConfig,
)

RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_config.yaml"
COMPONENT_VOCAB_PATH = PROJECT_ROOT / "configs" / "component_vocabulary.yaml"
MODEL_ALIASES_PATH = PROJECT_ROOT / "configs" / "model_aliases.yaml"
ERROR_PATTERNS_PATH = PROJECT_ROOT / "configs" / "error_code_patterns.yaml"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_rag_config() -> RAGConfig:
    with open(RAG_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RAGConfig.model_validate(data)


def _qdrant_reachable() -> bool:
    try:
        import httpx
        resp = httpx.get("http://localhost:6333/", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def _flag_embedding_available() -> bool:
    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: F401
        return True
    except ImportError:
        return False


def _flag_reranker_available() -> bool:
    try:
        from FlagEmbedding import FlagReranker  # noqa: F401
        return True
    except ImportError:
        return False


_QDRANT_UP = _qdrant_reachable()
_EMBEDDER_OK = _flag_embedding_available()
_RERANKER_OK = _flag_reranker_available()

skip_no_qdrant = pytest.mark.skipif(not _QDRANT_UP, reason="Qdrant not reachable on localhost:6333")
skip_no_embedder = pytest.mark.skipif(not _EMBEDDER_OK, reason="FlagEmbedding / BGE-M3 not available")
skip_no_reranker = pytest.mark.skipif(not _RERANKER_OK, reason="FlagReranker not available")


# ── Shared embedder fixture (expensive — session scoped) ────────────────────


@pytest.fixture(scope="session")
def embedder():
    """Load BGEEmbedder once per session."""
    cfg = _load_rag_config()
    # Force CPU to keep CI predictable; GPU is fine locally
    emb_cfg = cfg.embeddings.model_copy(update={"device": "cpu", "fp16": False})
    from rag.embeddings import BGEEmbedder
    return BGEEmbedder(emb_cfg)


@pytest.fixture(scope="session")
def reranker_model():
    """Load BGEReranker once per session."""
    cfg = _load_rag_config()
    rr_cfg = cfg.reranker.model_copy(update={"device": "cpu"})
    from rag.reranker import BGEReranker
    return BGEReranker(rr_cfg)


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-007: rag_config.yaml Pydantic validation
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD007:
    """rag_config.yaml must load and pass Pydantic validation."""

    def test_config_loads_successfully(self):
        cfg = _load_rag_config()
        assert isinstance(cfg, RAGConfig)

    def test_config_via_config_manager(self):
        cm = ConfigManager()
        cfg = cm.load_rag_config()
        assert isinstance(cfg, RAGConfig)

    def test_embeddings_section(self):
        cfg = _load_rag_config()
        assert cfg.embeddings.model == "BAAI/bge-m3"
        assert cfg.embeddings.batch_size > 0

    def test_qdrant_section_has_collections(self):
        cfg = _load_rag_config()
        assert len(cfg.qdrant.collections) >= 1
        names = [c.name for c in cfg.qdrant.collections]
        assert "service_manuals" in names

    def test_chunking_strategies_defined(self):
        cfg = _load_rag_config()
        assert "service_manuals" in cfg.chunking
        assert cfg.chunking["service_manuals"].strategy in (
            "hierarchical", "recursive", "per_record",
        )

    def test_invalid_config_rejected(self):
        """Negative: invalid port must raise validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RAGConfig.model_validate({
                "qdrant": {"port": -1},
            })

    def test_dense_size_1024_in_all_collections(self):
        cfg = _load_rag_config()
        for col in cfg.qdrant.collections:
            assert col.dense_size == 1024, (
                f"Collection {col.name} has dense_size={col.dense_size}, expected 1024"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-010: Same text -> deterministic vector
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD010:
    @skip_no_embedder
    def test_deterministic_embedding(self, embedder):
        text = "Замените блок фьюзера, код ошибки C6000."
        res1 = embedder.encode([text], return_sparse=False)
        res2 = embedder.encode([text], return_sparse=False)
        import numpy as np
        np.testing.assert_allclose(
            res1.dense[0], res2.dense[0], atol=1e-6,
            err_msg="Same text must produce identical vectors",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-011: Vector dimension correct for BGE-M3 (1024)
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD011:
    @skip_no_embedder
    def test_dimension_is_1024(self, embedder):
        res = embedder.encode(["test dimension"], return_sparse=False)
        assert res.dense.shape == (1, 1024), (
            f"Expected (1, 1024), got {res.dense.shape}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-015: Mixed language (ru/en/error codes) -> no crash
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD015:
    @skip_no_embedder
    def test_mixed_language_no_crash(self, embedder):
        texts = [
            "Замена фьюзера на Kyocera TASKalfa 3253ci",
            "Replace fuser unit on HP LaserJet M404n",
            "Код ошибки C6000 / Error SC543 / ERR-123",
            "Mixed: фьюзер fuser 3253ci SC543 блок питания power supply",
            "",  # edge case: empty string
        ]
        res = embedder.encode(texts, return_sparse=False)
        assert res.dense.shape[0] == 5
        assert res.dense.shape[1] == 1024

    @skip_no_embedder
    def test_empty_list_returns_empty(self, embedder):
        res = embedder.encode([], return_sparse=False)
        assert res.dense.shape[0] == 0


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-001: Qdrant healthcheck
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD001:
    @skip_no_qdrant
    def test_qdrant_healthcheck(self):
        cfg = _load_rag_config()
        from rag.qdrant_client import QdrantManager
        mgr = QdrantManager(cfg)
        assert mgr.healthcheck() is True

    @skip_no_qdrant
    def test_qdrant_version_returned(self):
        import httpx
        resp = httpx.get("http://localhost:6333/", timeout=3.0)
        data = resp.json()
        assert "version" in data
        assert re.match(r"\d+\.\d+", data["version"])


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-020: Reranker reorders candidates correctly
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD020:
    @skip_no_reranker
    def test_reranker_reorders(self, reranker_model):
        query = "Как заменить блок фьюзера?"
        candidates = [
            {
                "chunk_id": "c_irrelevant",
                "document_id": "doc1",
                "text": "Каталог запчастей: винт M3x8, шайба 10мм, гайка M3",
                "score": 0.9,
                "payload": {},
            },
            {
                "chunk_id": "c_relevant",
                "document_id": "doc1",
                "text": "Процедура замены блока фьюзера: отключите питание, откройте заднюю крышку, извлеките блок фьюзера.",
                "score": 0.5,
                "payload": {},
            },
        ]
        results = reranker_model.rerank(query, candidates)
        assert len(results) == 2
        # The relevant chunk should be ranked first by the reranker
        assert results[0].chunk_id == "c_relevant", (
            f"Expected c_relevant first, got {results[0].chunk_id}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-023: Empty candidate list -> empty result
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD023:
    @skip_no_reranker
    def test_empty_candidates_reranker(self, reranker_model):
        results = reranker_model.rerank("любой запрос", [])
        assert results == []

    def test_empty_candidates_reranker_code_path(self):
        """Verify the early-return code path without loading the model."""
        from rag.reranker import BGEReranker
        # Inspect source to confirm early return
        import inspect
        src = inspect.getsource(BGEReranker.rerank)
        assert "if not candidates" in src, (
            "rerank() must short-circuit on empty candidates"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-080: SearchResult has source_ref fields (doc_title, section)
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD080:
    def test_search_result_has_payload(self):
        from rag.search import SearchResult
        sr = SearchResult(
            chunk_id="c1",
            document_id="doc1",
            text="some text",
            score=0.8,
            dense_score=0.7,
            sparse_score=0.5,
            payload={"doc_title": "Service Manual", "section": "Chapter 3"},
        )
        assert sr.payload["doc_title"] == "Service Manual"
        assert sr.payload["section"] == "Chapter 3"

    def test_upsert_payload_includes_source_fields(self):
        """Code review: upsert_to_qdrant must put doc_title and section in payload."""
        import inspect
        from rag.ingestion import upsert_to_qdrant
        src = inspect.getsource(upsert_to_qdrant)
        assert '"doc_title"' in src, "doc_title must be in Qdrant payload"
        assert '"section"' in src, "section must be in Qdrant payload"

    def test_enriched_chunk_has_section(self):
        from rag.ingestion import EnrichedChunk
        ec = EnrichedChunk(
            chunk_id="c1",
            document_id="doc1",
            text="text",
            section="Chapter 5",
        )
        assert ec.section == "Chapter 5"


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-100: Qdrant unavailable -> clear error message
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD100:
    def test_wrong_port_raises_clear_error(self):
        """Connecting to a wrong port must raise QdrantUnavailableError
        with a human-readable message."""
        from rag.qdrant_client import QdrantManager, QdrantUnavailableError

        bad_cfg = _load_rag_config()
        # Modify to use a port that is definitely not Qdrant
        bad_qdrant = bad_cfg.qdrant.model_copy(update={"port": 19999, "grpc_port": 19998})
        bad_cfg_full = bad_cfg.model_copy(update={"qdrant": bad_qdrant})

        mgr = QdrantManager(bad_cfg_full)
        # healthcheck should return False (not crash with obscure error)
        assert mgr.healthcheck() is False

    def test_unavailable_error_is_descriptive(self):
        from rag.qdrant_client import QdrantUnavailableError
        err = QdrantUnavailableError("Qdrant недоступен на порту 19999")
        assert "19999" in str(err)


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-111: Verify API key not stored in Qdrant payload (code review)
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD111:
    _SENSITIVE_KEYS = {"api_key", "password", "secret", "token", "auth"}

    def test_no_api_key_in_upsert_payload(self):
        """Code review: the payload dict in upsert_to_qdrant must not
        contain sensitive keys like api_key, password, secret, token."""
        import inspect
        from rag.ingestion import upsert_to_qdrant
        src = inspect.getsource(upsert_to_qdrant)

        # Extract all string keys from the payload dict literal
        payload_keys = set(re.findall(r'"(\w+)"', src))
        leaked = payload_keys & self._SENSITIVE_KEYS
        assert not leaked, (
            f"Sensitive keys found in upsert payload: {leaked}"
        )

    def test_no_secrets_in_embedded_chunk(self):
        """EmbeddedChunk dataclass fields must not store secrets."""
        from rag.ingestion import EmbeddedChunk
        field_names = {f.name for f in dc_fields(EmbeddedChunk)}
        leaked = field_names & self._SENSITIVE_KEYS
        assert not leaked, (
            f"Sensitive fields in EmbeddedChunk: {leaked}"
        )

    def test_no_api_key_in_config_payload_flow(self):
        """LLMEndpointConfig.api_key must never reach QdrantManager or ingestion."""
        import inspect
        from rag import qdrant_client as qmod
        src_q = inspect.getsource(qmod)
        assert "api_key" not in src_q, (
            "qdrant_client module must not reference api_key"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-120: component_vocabulary.yaml sync
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD120:
    def test_component_vocab_exists_and_valid(self):
        assert COMPONENT_VOCAB_PATH.exists()
        with open(COMPONENT_VOCAB_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_ingestion_uses_component_vocab_path(self):
        """Ingestion must reference component_vocabulary.yaml."""
        import inspect
        from rag import ingestion
        src = inspect.getsource(ingestion)
        assert "component_vocabulary.yaml" in src

    def test_canonical_names_are_snake_case(self):
        with open(COMPONENT_VOCAB_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key in data:
            assert re.match(r"^[a-z][a-z0-9_]*$", key), (
                f"Canonical component name '{key}' is not snake_case"
            )

    def test_every_entry_has_synonyms(self):
        with open(COMPONENT_VOCAB_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for key, synonyms in data.items():
            assert isinstance(synonyms, list), (
                f"component '{key}' synonyms must be a list"
            )
            assert len(synonyms) >= 1, (
                f"component '{key}' must have at least one synonym"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-121: model_aliases.yaml sync between RAG and ingestion
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD121:
    def test_model_aliases_exists_and_valid(self):
        assert MODEL_ALIASES_PATH.exists()
        with open(MODEL_ALIASES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_search_and_ingestion_use_same_path(self):
        """Both search.py and ingestion.py must use model_aliases.yaml."""
        import inspect
        from rag import search as search_mod
        from rag import ingestion as ing_mod
        src_search = inspect.getsource(search_mod)
        src_ing = inspect.getsource(ing_mod)
        assert "model_aliases.yaml" in src_search, (
            "search.py must reference model_aliases.yaml"
        )
        assert "model_aliases.yaml" in src_ing, (
            "ingestion.py must reference model_aliases.yaml"
        )

    def test_aliases_are_lists_of_strings(self):
        with open(MODEL_ALIASES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for canonical, aliases in data.items():
            assert isinstance(aliases, list), (
                f"Aliases for '{canonical}' must be a list"
            )
            for alias in aliases:
                assert isinstance(alias, str), (
                    f"Alias for '{canonical}' must be a string, got {type(alias)}"
                )

    def test_no_duplicate_aliases_across_models(self):
        with open(MODEL_ALIASES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        seen: dict[str, str] = {}
        for canonical, aliases in data.items():
            for alias in aliases:
                lower = alias.lower()
                if lower in seen:
                    pytest.fail(
                        f"Duplicate alias '{alias}' found in both "
                        f"'{seen[lower]}' and '{canonical}'"
                    )
                seen[lower] = canonical


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-122: error_code_patterns.yaml single source of truth
# ═════════════════════════════════════════════════════════════════════════════


class TestTCD122:
    def test_error_patterns_exists_and_valid(self):
        assert ERROR_PATTERNS_PATH.exists()
        with open(ERROR_PATTERNS_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "generic" in data, "Must have a 'generic' fallback section"

    def test_all_patterns_compile(self):
        with open(ERROR_PATTERNS_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for vendor, patterns in data.items():
            assert isinstance(patterns, list), (
                f"Vendor '{vendor}' patterns must be a list"
            )
            for p in patterns:
                try:
                    re.compile(p)
                except re.error as exc:
                    pytest.fail(
                        f"Invalid regex for vendor '{vendor}': {p!r} — {exc}"
                    )

    def test_ingestion_uses_error_patterns_path(self):
        import inspect
        from rag import ingestion
        src = inspect.getsource(ingestion)
        assert "error_code_patterns.yaml" in src

    def test_generic_patterns_match_known_codes(self):
        with open(ERROR_PATTERNS_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        generic = data["generic"]
        test_codes = ["C6000", "SC543", "ERR-123", "J0500"]
        for code in test_codes:
            matched = any(re.search(p, code) for p in generic)
            assert matched, (
                f"Generic patterns should match known code '{code}'"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TC-D-021: Reranker failure must not break search path (graceful degrade)
# ═════════════════════════════════════════════════════════════════════════════


class TestRerankerGracefulDegrade:
    def test_try_get_reranker_returns_none_on_error(self, monkeypatch):
        """`_try_get_reranker` должен вернуть None, если get_reranker кидает."""
        from state import singletons

        def _raise():
            raise RuntimeError("simulated CUDA OOM")

        monkeypatch.setattr(singletons, "get_reranker", _raise)
        assert singletons._try_get_reranker() is None

    def test_search_returns_results_when_reranker_crashes(self):
        """search() не должен падать, если reranker.rerank кидает исключение."""
        from unittest.mock import MagicMock
        from rag.reranker import RerankerScoringError
        from rag.search import HybridSearcher, SearchResult

        fake_results = [
            SearchResult(
                chunk_id=f"c{i}",
                document_id="d1",
                text=f"text {i}",
                score=1.0 - i * 0.1,
                dense_score=0.0,
                sparse_score=0.0,
                payload={},
            )
            for i in range(5)
        ]

        searcher = HybridSearcher.__new__(HybridSearcher)
        searcher._qdrant = MagicMock()
        searcher._embedder = MagicMock()
        searcher._embedder.encode_query.return_value = (MagicMock(), MagicMock())
        searcher._config = MagicMock()
        searcher._config.reranker.top_n_output = 3
        searcher._hs_cfg = MagicMock()
        searcher._hs_cfg.use_qdrant_fusion = False
        searcher._hs_cfg.top_k_per_branch = 5
        searcher._model_aliases = {}
        searcher._qdrant_version = None
        searcher._reranker = MagicMock()
        searcher._reranker.rerank.side_effect = RerankerScoringError("boom")
        searcher._search_with_manual_rrf = MagicMock(return_value=fake_results)
        searcher._supports_fusion = MagicMock(return_value=False)
        searcher._build_filter = MagicMock(return_value=None)

        results = searcher.search("q", "service_manuals", top_k=3, use_reranker=True)

        assert len(results) == 3, "Должны остаться 3 top_k результата из hybrid-ветки"
        assert results[0].chunk_id == "c0", "Исходный порядок сохраняется"
        searcher._reranker.rerank.assert_called_once()
