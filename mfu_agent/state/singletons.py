"""Streamlit cached singletons — Phase 7.1.

Long-lived objects that survive reruns via @st.cache_resource.
Endpoint name is part of the cache key so switching endpoints
automatically invalidates dependent singletons.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import streamlit as st

from config.loader import get_config_manager

if TYPE_CHECKING:
    from agent.core import Agent
    from agent.memory import MemoryManager
    from agent.tools.registry import ToolRegistry
    from config.loader import LLMEndpointConfig
    from llm.client import LLMClient
    from rag.embeddings import BGEEmbedder
    from rag.qdrant_client import QdrantManager
    from rag.reranker import BGEReranker
    from rag.search import HybridSearcher

logger = logging.getLogger(__name__)


@st.cache_resource
def get_qdrant_manager() -> QdrantManager:
    """Singleton QdrantManager, connects on first call."""
    from rag.qdrant_client import QdrantManager

    config = get_config_manager().load_rag_config()
    mgr = QdrantManager(config)
    logger.info("QdrantManager создан (%s:%d)", config.qdrant.host, config.qdrant.port)
    return mgr


@st.cache_resource
def get_embedder() -> BGEEmbedder:
    """Singleton BGE-M3 embedder."""
    from rag.embeddings import BGEEmbedder

    config = get_config_manager().load_rag_config()
    embedder = BGEEmbedder(config.embeddings)
    logger.info("BGEEmbedder создан (model=%s)", config.embeddings.model)
    return embedder


@st.cache_resource
def get_reranker() -> BGEReranker:
    """Singleton BGE reranker."""
    from rag.reranker import BGEReranker

    config = get_config_manager().load_rag_config()
    reranker = BGEReranker(config.reranker)
    logger.info("BGEReranker создан (model=%s)", config.reranker.model)
    return reranker


def _try_get_reranker() -> BGEReranker | None:
    """Graceful wrapper around ``get_reranker``.

    Reranker может упасть при инициализации (CUDA OOM, отсутствие
    FlagEmbedding, несовместимая модель). В таких случаях отдаём
    ``None`` — HybridSearcher тогда пропустит ре-ранкинг.
    """
    try:
        return get_reranker()
    except Exception as exc:
        logger.warning(
            "Reranker init failed, hybrid search будет работать без re-ranking: %s",
            exc,
        )
        return None


@st.cache_resource
def get_hybrid_searcher() -> HybridSearcher:
    """Singleton hybrid searcher wired to Qdrant + embedder + reranker."""
    from rag.search import HybridSearcher

    config = get_config_manager().load_rag_config()
    return HybridSearcher(
        qdrant_manager=get_qdrant_manager(),
        embedder=get_embedder(),
        config=config,
        reranker=_try_get_reranker(),
    )


@st.cache_resource
def get_llm_client(endpoint_name: str) -> LLMClient:
    """Singleton LLM client, keyed by endpoint name.

    Changing ``endpoint_name`` yields a new client automatically
    because Streamlit hashes the argument.
    """
    from llm.client import LLMClient

    cm = get_config_manager()
    config = cm.load_agent_config()
    _ = config.llm
    client = LLMClient(
        _resolve_endpoint(endpoint_name),
    )
    logger.info("LLMClient создан для endpoint=%s", endpoint_name)
    return client


@st.cache_resource
def get_tool_registry() -> ToolRegistry:
    """Singleton tool registry with all agent tools registered."""
    from agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    logger.info("ToolRegistry создан")
    return registry


@st.cache_resource
def get_memory_manager() -> MemoryManager:
    """Singleton MemoryManager for learned patterns."""
    from agent.memory import MemoryManager

    config = get_config_manager().load_agent_config()
    manager = MemoryManager(config.memory)  # type: ignore[arg-type]
    logger.info("MemoryManager создан")
    return manager


@st.cache_resource
def get_agent(endpoint_name: str) -> Agent:
    """Singleton Agent, keyed by LLM endpoint name.

    Switching the endpoint invalidates both LLMClient and Agent
    because ``endpoint_name`` is part of the cache key.
    """
    from agent.core import Agent

    config = get_config_manager().load_agent_config()
    return Agent(
        llm_client=get_llm_client(endpoint_name),
        tool_registry=get_tool_registry(),
        factor_store=None,  # type: ignore[arg-type]
        config=config,
        memory_manager=get_memory_manager(),
    )


def invalidate_llm_singletons() -> None:
    """Clear LLMClient and Agent caches after endpoint change."""
    get_llm_client.clear()
    get_agent.clear()
    logger.info("LLM-зависимые кэши сброшены")


def invalidate_rag_singletons() -> None:
    """Clear all RAG-related caches (e.g. after config change)."""
    get_hybrid_searcher.clear()
    get_qdrant_manager.clear()
    get_embedder.clear()
    get_reranker.clear()
    logger.info("RAG-кэши сброшены")


def invalidate_all() -> None:
    """Nuclear option — clear every cached singleton."""
    invalidate_llm_singletons()
    invalidate_rag_singletons()
    get_tool_registry.clear()
    get_memory_manager.clear()
    logger.info("Все кэши сброшены")


# ── helpers ──────────────────────────────────────────────────────────────────


def _resolve_endpoint(endpoint_name: str) -> LLMEndpointConfig:
    """Load LLMEndpointConfig by name from llm_endpoints.yaml.

    Falls back to agent_config defaults if the endpoints file
    is missing or the name is not found.
    """
    import yaml

    from config.loader import CONFIGS_DIR, LLMEndpointConfig

    path = CONFIGS_DIR / "llm_endpoints.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        endpoints = data.get("endpoints", data)
        if isinstance(endpoints, dict) and endpoint_name in endpoints:
            return LLMEndpointConfig.model_validate(endpoints[endpoint_name])

    return LLMEndpointConfig()
