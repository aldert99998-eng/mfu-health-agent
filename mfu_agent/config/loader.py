"""Configuration loading, validation, and access."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator

from data_io.models import SchemaProfile, WeightsProfile

_T = TypeVar("_T", bound=BaseModel)

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = _PROJECT_ROOT / "configs"
WEIGHTS_DIR = CONFIGS_DIR / "weights"
SCHEMA_PROFILES_DIR = _PROJECT_ROOT / "config" / "schema_profiles"


# ── Errors ────────────────────────────────────────────────────────────────────


class ConfigValidationError(Exception):
    """Raised when a configuration file fails validation.

    Attributes:
        config_path: Path to the problematic config file.
        details: Human-readable description of what went wrong.
    """

    def __init__(self, config_path: Path | str, details: str) -> None:
        self.config_path = Path(config_path)
        self.details = details
        super().__init__(
            f"Invalid config '{self.config_path.name}': {details}"
        )


# ── LLM endpoint config (track B) ────────────────────────────────────────────


class GigaChatAuthConfig(BaseModel):
    """OAuth-based auth for the GigaChat provider.

    The `auth_key` itself is NEVER stored in YAML — only the name of the ENV
    variable that holds it. The provider reads it at first-use time.
    """

    type: Literal["gigachat_oauth"]
    oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    scope: str = "GIGACHAT_API_PERS"
    auth_key_env: str = "GIGACHAT_AUTH_KEY"
    verify_tls: bool = False
    ca_bundle: str | None = None


class LLMEndpointConfig(BaseModel):
    """Single OpenAI-compatible LLM endpoint.

    When ``auth`` is ``None`` the endpoint uses the plain ``api_key`` (local
    llama-server / vLLM path). When ``auth`` is set, ``LLMClient`` swaps in a
    token provider that refreshes an OAuth access token periodically.
    """

    url: str = "http://localhost:8000/v1"
    api_key: str = "dummy-for-local"
    model: str = "qwen2.5-7b-instruct"
    tool_strategy: str = ""
    timeout_seconds: float = 120.0
    max_retries_network: int = 3
    max_retries_invalid: int = 2
    display_name: str = ""
    auth: GigaChatAuthConfig | None = None


# ── Agent config models (track B) ────────────────────────────────────────────


class LLMGenerationParams(BaseModel):
    """LLM generation parameters for a specific mode."""

    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0)
    max_tokens: int = Field(ge=1)


class LLMConfig(BaseModel):
    """LLM generation parameters per agent mode."""

    batch_mode: LLMGenerationParams = Field(
        default_factory=lambda: LLMGenerationParams(
            temperature=0.2, top_p=0.9, max_tokens=2000
        )
    )
    chat_mode: LLMGenerationParams = Field(
        default_factory=lambda: LLMGenerationParams(
            temperature=0.4, top_p=0.9, max_tokens=3000
        )
    )
    reflection: LLMGenerationParams = Field(
        default_factory=lambda: LLMGenerationParams(
            temperature=0.1, top_p=1.0, max_tokens=500
        )
    )
    classify_severity: LLMGenerationParams = Field(
        default_factory=lambda: LLMGenerationParams(
            temperature=0.0, top_p=1.0, max_tokens=300
        )
    )


class AgentLoopConfig(BaseModel):
    """Core agent loop parameters."""

    max_attempts_per_device: int = Field(default=2, ge=1, le=5)
    max_tool_calls_per_attempt: int = Field(default=15, ge=1)
    max_llm_calls_per_attempt: int = Field(default=20, ge=1)
    trace_retention_days: int = Field(default=30, ge=1)


class ReflectionConfig(BaseModel):
    """Self-check / reflection configuration."""

    enabled: bool = True
    apply_in_chat_mode: bool = False


class MemoryConfig(BaseModel):
    """Learned patterns memory configuration."""

    enabled: bool = True
    max_patterns_per_model: int = Field(default=50, ge=1)
    pattern_min_evidence_devices: int = Field(default=2, ge=1)


class RoleExtensionsConfig(BaseModel):
    """Dynamic role extension configuration."""

    enabled: bool = True
    critical_function_devices: list[dict[str, Any]] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Complete agent configuration (track B)."""

    agent: AgentLoopConfig = Field(default_factory=AgentLoopConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    role_extensions: RoleExtensionsConfig = Field(default_factory=RoleExtensionsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


# ── Report config models (track C) ───────────────────────────────────────────


class RenderingConfig(BaseModel):
    """Paths to report templates and styles."""

    template_path: str = "reporting/templates/report.jinja2"
    css_path: str = "reporting/styles/report.css"
    print_css_path: str = "reporting/styles/report_print.css"


class ReportThresholds(BaseModel):
    """Zone and confidence thresholds for report rendering."""

    green_zone: int = Field(default=75, ge=0, le=100)
    red_zone: int = Field(default=40, ge=0, le=100)
    high_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    medium_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("red_zone")
    @classmethod
    def red_below_green(cls, v: int, info: Any) -> int:
        green = info.data.get("green_zone")
        if green is not None and v >= green:
            msg = f"red_zone ({v}) must be below green_zone ({green})"
            raise ValueError(msg)
        return v


class DisplayConfig(BaseModel):
    """Display limits and history depth."""

    max_devices_in_table_html: int = Field(default=500, ge=1)
    max_devices_in_pdf_warning: int = Field(default=300, ge=1)
    sparkline_history_points: int = Field(default=5, ge=1)


class CategoryBreakdownRule(BaseModel):
    """Single rule for auto-selecting the category breakdown."""

    condition: str | None = None
    use: str | None = None
    default: str | None = None


class AgentTraceConfig(BaseModel):
    """Configuration for the agent trace section of the report."""

    devices_to_include: int = Field(default=3, ge=1)
    selection_strategy: str = "worst_best_flagged"


class ReportConfig(BaseModel):
    """Complete report configuration (track C)."""

    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    thresholds: ReportThresholds = Field(default_factory=ReportThresholds)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    category_breakdown_auto: list[CategoryBreakdownRule] = Field(default_factory=list)
    agent_trace: AgentTraceConfig = Field(default_factory=AgentTraceConfig)


# ── RAG config models (track D) ──────────────────────────────────────────────


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""

    model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    max_length: int = Field(default=1024, ge=1)
    fp16: bool = False


class RerankerConfig(BaseModel):
    """Reranker model configuration."""

    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    batch_size: int = Field(default=16, ge=1)
    top_n_input: int = Field(default=30, ge=1)
    top_n_output: int = Field(default=8, ge=1)


class HybridSearchConfig(BaseModel):
    """Hybrid search parameters."""

    use_qdrant_fusion: bool = True
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0.0)
    sparse_weight: float = Field(default=1.0, ge=0.0)
    top_k_per_branch: int = Field(default=30, ge=1)


class ChunkingStrategyConfig(BaseModel):
    """Chunking strategy for a single collection."""

    strategy: str
    max_tokens: int | None = None
    min_tokens: int | None = None
    overlap_tokens: int | None = None
    use_bookmarks: bool | None = None
    record_separator: str | None = None


class QdrantCollectionConfig(BaseModel):
    """Single Qdrant collection definition."""

    name: str
    dense_size: int = Field(default=1024, ge=1)
    hnsw_m: int | None = None
    hnsw_ef_construct: int | None = None


class QdrantConfig(BaseModel):
    """Qdrant connection and collection configuration."""

    host: str = "localhost"
    port: int = Field(default=6333, ge=1, le=65535)
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    prefer_grpc: bool = True
    timeout_seconds: int = Field(default=30, ge=1)
    collections: list[QdrantCollectionConfig] = Field(default_factory=list)


class LLMEnrichmentConfig(BaseModel):
    """LLM enrichment settings for the ingestion pipeline."""

    enabled: bool = True
    classify_batch_size: int = Field(default=15, ge=1)
    fallback_content_type: str = "reference"


class IngestionConfig(BaseModel):
    """Document ingestion pipeline configuration."""

    pdf_parser: str = "pymupdf"
    ocr_enabled: bool = False
    llm_enrichment: LLMEnrichmentConfig = Field(default_factory=LLMEnrichmentConfig)
    upsert_batch_size: int = Field(default=100, ge=1)


class AcceptanceThresholds(BaseModel):
    """Minimum acceptable RAG quality thresholds."""

    recall_at_5: float = Field(default=0.70, ge=0.0, le=1.0)
    mrr: float = Field(default=0.50, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """RAG evaluation configuration."""

    dataset_path: str = "configs/eval_dataset.yaml"
    save_history_path: str = "storage/eval_history/"
    auto_run_on_reindex: bool = True
    acceptance_thresholds: AcceptanceThresholds = Field(default_factory=AcceptanceThresholds)


class StorageConfig(BaseModel):
    """Storage paths for uploads and snapshots."""

    uploads_dir: str = "storage/uploads"
    qdrant_snapshots_dir: str = "storage/qdrant_snapshots"


class PIIPattern(BaseModel):
    """Single PII detection pattern."""

    pattern: str
    replacement: str


class PIIFilterConfig(BaseModel):
    """PII filtering configuration."""

    enabled: bool = True
    patterns: list[PIIPattern] = Field(default_factory=list)


class RAGConfig(BaseModel):
    """Complete RAG configuration (track D)."""

    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    chunking: dict[str, ChunkingStrategyConfig] = Field(default_factory=dict)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    pii_filter: PIIFilterConfig = Field(default_factory=PIIFilterConfig)


# ── ConfigManager ─────────────────────────────────────────────────────────────


class ConfigManager:
    """Loads, validates, and caches project configuration from YAML files."""

    def __init__(
        self,
        configs_dir: Path = CONFIGS_DIR,
        weights_dir: Path = WEIGHTS_DIR,
        schema_profiles_dir: Path = SCHEMA_PROFILES_DIR,
    ) -> None:
        self._configs_dir = configs_dir
        self._weights_dir = weights_dir
        self._schema_profiles_dir = schema_profiles_dir

    # ── public API ────────────────────────────────────────────────────────

    def load_weights(self, profile_name: str = "default") -> WeightsProfile:
        path = self._weights_dir / f"{profile_name}.yaml"
        data = self._read_yaml(path)
        return self._validate(path, WeightsProfile, data)

    def load_agent_config(self) -> AgentConfig:
        path = self._configs_dir / "agent_config.yaml"
        data = self._read_yaml(path)
        return self._validate(path, AgentConfig, data)

    def load_report_config(self) -> ReportConfig:
        path = self._configs_dir / "report_config.yaml"
        data = self._read_yaml(path)
        return self._validate(path, ReportConfig, data)

    def load_rag_config(self) -> RAGConfig:
        path = self._configs_dir / "rag_config.yaml"
        data = self._read_yaml(path)
        return self._validate(path, RAGConfig, data)

    def list_profiles(self) -> list[str]:
        if not self._weights_dir.exists():
            return []
        return sorted(
            p.stem for p in self._weights_dir.glob("*.yaml")
        )

    def save_weights_profile(self, profile: WeightsProfile) -> Path:
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        path = self._weights_dir / f"{profile.profile_name}.yaml"
        data = profile.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return path

    def list_schema_profiles(self) -> list[str]:
        if not self._schema_profiles_dir.exists():
            return []
        return sorted(
            p.stem for p in self._schema_profiles_dir.glob("*.yaml")
        )

    def load_schema_profile(self, name: str) -> SchemaProfile:
        path = self._schema_profiles_dir / f"{name}.yaml"
        data = self._read_yaml(path)
        return self._validate(path, SchemaProfile, data)

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ConfigValidationError(path, f"File not found: {path}")
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigValidationError(path, f"YAML parse error: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigValidationError(path, "Expected a YAML mapping at the top level")
        return data

    @staticmethod
    def _validate(path: Path, model: type[_T], data: dict[str, Any]) -> _T:
        try:
            return model.model_validate(data)
        except Exception as exc:
            raise ConfigValidationError(path, str(exc)) from exc


@lru_cache(maxsize=1)
def get_config_manager() -> ConfigManager:
    """Return a singleton ConfigManager instance."""
    return ConfigManager()
