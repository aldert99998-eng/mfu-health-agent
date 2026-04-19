"""Unified data models for the MFU Health Agent."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ── Enums ──────────────────────────────────────────────────────────────────────


class SeverityLevel(StrEnum):
    """Error severity classification level."""

    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"


class HealthZone(StrEnum):
    """Health index zone classification."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class ConfidenceZone(StrEnum):
    """Confidence score zone classification."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SilentDeviceMode(StrEnum):
    """Strategy for devices with no events."""

    OPTIMISTIC = "optimistic"
    DATA_QUALITY = "data_quality"
    CARRY_FORWARD = "carry_forward"


class TraceStepType(StrEnum):
    """Type of agent execution step."""

    PLAN = "plan"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    REFLECTION = "reflection"
    MEMORY_SAVE = "memory_save"


class AgentMode(StrEnum):
    """Agent execution mode."""

    BATCH = "batch"
    CHAT = "chat"


class ReflectionVerdict(StrEnum):
    """Agent self-check verdict."""

    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    SUSPICIOUS = "suspicious"


class ReflectionAction(StrEnum):
    """Recommended action after agent self-check."""

    ACCEPT = "accept"
    RECALCULATE = "recalculate"
    FLAG_FOR_REVIEW = "flag_for_review"


class PatternType(StrEnum):
    """Fleet-wide problem pattern classification."""

    MASS_ISSUE = "mass_issue"
    LOCATION_CLUSTER = "location_cluster"
    CRITICAL_SINGLE = "critical_single"


class FileFormat(StrEnum):
    """Supported input file formats."""

    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    JSONL = "jsonl"
    XLSX = "xlsx"


class ContentType(StrEnum):
    """RAG document content type for search filtering."""

    SYMPTOM = "symptom"
    CAUSE = "cause"
    PROCEDURE = "procedure"
    SPECIFICATION = "specification"
    REFERENCE = "reference"


class SimilarityDimension(StrEnum):
    """Dimension for finding similar devices."""

    ERRORS = "errors"
    MODEL = "model"
    LOCATION = "location"
    ERROR_AND_MODEL = "error_and_model"


# ── Track A: Health Index ──────────────────────────────────────────────────────


class Factor(BaseModel):
    """Single problem/event factor for health index calculation."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    error_code: str
    severity_level: SeverityLevel
    S: float = Field(ge=0)
    n_repetitions: int = Field(ge=1)
    R: float = Field(ge=1.0, le=5.0)
    C: float = Field(ge=1.0, le=1.5)
    A: float = Field(ge=0.0, le=1.0)
    event_timestamp: datetime
    age_days: int = Field(ge=0)
    applicable_modifiers: list[str] = Field(default_factory=list)
    source: str | None = None
    confidence_flags: list[str] = Field(default_factory=list)


class ConfidenceFactors(BaseModel):
    """Input flags for confidence score calculation."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    rag_missing_count: int = Field(default=0, ge=0)
    missing_resources: bool = False
    missing_model: bool = False
    abnormal_daily_jump: bool = False
    anomalous_event_count: bool = False
    no_events_and_no_resources: bool = False


class FactorContribution(BaseModel):
    """One factor's contribution to the health index breakdown."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    label: str
    penalty: float
    S: float
    R: float
    C: float
    A: float
    source: str


class HealthResult(BaseModel):
    """Output of the health index calculation for one device."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    device_id: str
    health_index: int = Field(ge=1, le=100)
    confidence: float = Field(ge=0.2, le=1.0)
    zone: HealthZone
    confidence_zone: ConfidenceZone
    factor_contributions: list[FactorContribution] = Field(default_factory=list)
    confidence_reasons: list[str] = Field(default_factory=list)
    calculation_snapshot: dict[str, Any] = Field(default_factory=dict)
    calculated_at: datetime
    reflection_notes: str | None = None


# ── Track A: WeightsProfile (nested) ──────────────────────────────────────────


class SeverityWeights(BaseModel):
    """Severity score mapping by level."""

    critical: float = Field(default=60, ge=0)
    high: float = Field(default=20, ge=0)
    medium: float = Field(default=10, ge=0)
    low: float = Field(default=3, ge=0)
    info: float = Field(default=0, ge=0)


class RepeatabilityConfig(BaseModel):
    """Repeatability coefficient parameters."""

    base: int = 2
    max_value: float = 5.0
    window_days: int = 14


class ContextModifier(BaseModel):
    """Single context modifier rule."""

    threshold: float | None = None
    multiplier: float
    applies_to_components: list[str]


class ContextConfig(BaseModel):
    """Context modifier configuration."""

    modifiers: dict[str, ContextModifier] = Field(default_factory=dict)
    max_value: float = 1.5


class AgeConfig(BaseModel):
    """Age decay parameters."""

    tau_days: int = 14
    window_days: int = 30


class ConfidencePenalties(BaseModel):
    """Penalty multipliers for confidence reduction."""

    rag_not_found: float = 0.7
    missing_resources: float = 0.85
    missing_model: float = 0.6
    abnormal_daily_jump: float = 0.8
    anomalous_event_count: float = 0.7
    no_events_and_no_resources: float = 0.9


class ConfidenceConfig(BaseModel):
    """Confidence calculation parameters."""

    min_value: float = 0.2
    penalties: ConfidencePenalties = Field(default_factory=ConfidencePenalties)


class ZoneThresholds(BaseModel):
    """Health zone threshold configuration."""

    green_threshold: int = 75
    red_threshold: int = 40


class WeightsProfile(BaseModel):
    """Complete weights profile loaded from YAML."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    profile_name: str
    version: str = "1.0"
    severity: SeverityWeights = Field(default_factory=SeverityWeights)
    repeatability: RepeatabilityConfig = Field(default_factory=RepeatabilityConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    age: AgeConfig = Field(default_factory=AgeConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    zones: ZoneThresholds = Field(default_factory=ZoneThresholds)
    critical_per_day_limit: int = 1
    silent_device_mode: SilentDeviceMode = SilentDeviceMode.DATA_QUALITY


# ── Track B: Agent ─────────────────────────────────────────────────────────────


class TraceStep(BaseModel):
    """Single step in agent execution trace."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    step_number: int
    type: TraceStepType
    thought: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result_summary: str | None = None
    duration_ms: int
    tokens_used: int | None = None


class Trace(BaseModel):
    """Complete agent execution trace."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    session_id: str
    mode: AgentMode
    device_id: str | None = None
    user_query: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    steps: list[TraceStep] = Field(default_factory=list)
    final_result: dict[str, Any] = Field(default_factory=dict)
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    attempts: int = Field(default=1, ge=1, le=2)
    flagged_for_review: bool = False

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, raw: str) -> Trace:
        return cls.model_validate_json(raw)

    def summary(self) -> str:
        duration_s = (
            round((self.ended_at - self.started_at).total_seconds(), 1)
            if self.ended_at
            else "n/a"
        )
        hi = self.final_result.get("health_index", "—")
        flag = " ⚠️ flagged" if self.flagged_for_review else ""
        return (
            f"[{self.mode.value}] session={self.session_id[:8]}… "
            f"device={self.device_id or '—'} "
            f"HI={hi} "
            f"tools={self.total_tool_calls} llm={self.total_llm_calls} "
            f"tokens={self.total_tokens} "
            f"attempts={self.attempts} "
            f"duration={duration_s}s{flag}"
        )


class LearnedPattern(BaseModel):
    """Pattern discovered by agent across devices."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    type: str = "pattern"
    scope: str
    observation: str
    evidence_devices: list[str] = Field(min_length=2)


class SeverityResult(BaseModel):
    """Result of error severity classification."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    severity: SeverityLevel
    confidence: float
    affected_components: list[str] = Field(default_factory=list)
    source: str | None = None
    reasoning: str = ""


class ReflectionIssue(BaseModel):
    """Single issue found during agent self-check."""

    issue: str
    severity: str


class ReflectionResult(BaseModel):
    """Result of agent self-check reflection."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    verdict: ReflectionVerdict
    issues: list[ReflectionIssue] = Field(default_factory=list)
    recommended_action: ReflectionAction


class BatchContext(BaseModel):
    """Context for batch-mode agent execution."""

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True,
        revalidate_instances="never",
    )

    weights_profile: WeightsProfile
    factor_store: Any = None
    fleet_stats: Any = None
    device_metadata: Any = None
    learned_patterns: list[LearnedPattern] = Field(default_factory=list)


class ChatContext(BaseModel):
    """Context for chat-mode agent execution."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    current_report: Report | None = None
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    factor_store: Any = None
    fleet_stats: Any = None


# ── Track C: Report ────────────────────────────────────────────────────────────


class DocReference(BaseModel):
    """Reference to service documentation."""

    title: str
    section: str
    url: str | None = None


class ResourceState(BaseModel):
    """Current resource levels for a device."""

    toner: int | None = None
    drum: int | None = None
    fuser: int | None = None
    mileage: int | None = None
    service_interval: int | None = None


class HistoryPoint(BaseModel):
    """Single point in health index history for sparkline."""

    date: datetime
    health_index: int = Field(ge=1, le=100)


class FleetSummary(BaseModel):
    """Aggregate fleet health statistics."""

    total_devices: int
    average_index: float
    median_index: float
    zone_counts: dict[str, int]
    average_confidence: float
    delta_vs_previous: float | None = None


class PatternGroup(BaseModel):
    """Group of devices sharing a common problem pattern."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    pattern_type: PatternType
    title: str = Field(max_length=60)
    affected_device_ids: list[str]
    average_index: float
    explanation: str
    doc_references: list[DocReference] = Field(default_factory=list)


class DeviceReport(BaseModel):
    """Health report data for a single device."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    device_id: str
    model: str | None = None
    location: str | None = None
    health_index: int = Field(ge=1, le=100)
    confidence: float = Field(ge=0.2, le=1.0)
    zone: HealthZone
    confidence_zone: ConfidenceZone
    top_problem_tag: str = ""
    factor_contributions: list[FactorContribution] = Field(default_factory=list)
    resource_state: ResourceState = Field(default_factory=ResourceState)
    index_history: list[HistoryPoint] = Field(default_factory=list)
    agent_recommendation: str = ""
    flag_for_review: bool = False


class DistributionBin(BaseModel):
    """Single bin for health index distribution histogram."""

    range_start: int
    range_end: int
    count: int


class CategoryBreakdown(BaseModel):
    """Fleet breakdown by category (model, location, or confidence zone)."""

    category_field: str
    groups: dict[str, FleetSummary]


class AgentTraceSummary(BaseModel):
    """Summary of agent execution across the fleet."""

    average_tool_calls_per_device: float = 0.0
    average_llm_calls_per_device: float = 0.0
    self_check_restart_count: int = 0
    devices_flagged_for_review: list[str] = Field(default_factory=list)
    rag_request_count: int = 0
    rag_average_rerank_score: float = 0.0
    rag_no_match_percent: float = 0.0


class RAGSummary(BaseModel):
    """Summary of RAG usage in calculation."""

    total_queries: int = 0
    cache_hit_rate: float = 0.0
    average_rerank_score: float = 0.0
    no_match_count: int = 0
    collections_used: list[str] = Field(default_factory=list)


class CalculationSnapshot(BaseModel):
    """Frozen snapshot of all parameters at calculation time."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    weights_profile_name: str
    weights_profile_version: str
    weights_data: dict[str, Any]
    llm_endpoint: str = ""
    llm_model: str = ""
    llm_tool_strategy: str = ""
    embedding_model: str = ""
    reranker_model: str = ""
    rag_collections: dict[str, int] = Field(default_factory=dict)
    source_file_hash: str = ""
    input_record_count: int = 0
    valid_record_count: int = 0
    discarded_record_count: int = 0
    field_mapping_profile: str | None = None


class SourceFileInfo(BaseModel):
    """Metadata about the ingested source file."""

    file_name: str
    file_hash: str
    file_size_bytes: int = 0
    file_format: FileFormat
    uploaded_at: datetime


class Report(BaseModel):
    """Complete fleet health report."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    report_id: str
    generated_at: datetime
    source_file_name: str
    source_file_hash: str
    analysis_window_days: int
    fleet_summary: FleetSummary
    executive_summary: str = ""
    top_patterns: list[PatternGroup] = Field(default_factory=list, max_length=5)
    index_distribution: list[DistributionBin] = Field(default_factory=list)
    category_breakdown: CategoryBreakdown | None = None
    devices: list[DeviceReport] = Field(default_factory=list)
    agent_trace_summary: AgentTraceSummary | None = None
    rag_summary: RAGSummary | None = None
    calculation_snapshot: CalculationSnapshot
    include_agent_trace: bool = False
    include_devices_collapsed: bool = True


# ── Track E: Ingestion ─────────────────────────────────────────────────────────


class NormalizedEvent(BaseModel):
    """Single normalized device event."""

    model_config = ConfigDict(frozen=True)

    device_id: str = Field(min_length=1, max_length=100)
    timestamp: datetime
    error_code: str | None = Field(default=None, pattern=r"^[A-Z]{1,3}\d{3,5}$")
    error_description: str | None = None
    model: str | None = None
    vendor: str | None = None
    location: str | None = None
    status: str | None = None


class ResourceSnapshot(BaseModel):
    """Resource levels snapshot for a device at a point in time."""

    model_config = ConfigDict(frozen=True)

    device_id: str = Field(min_length=1, max_length=100)
    timestamp: datetime
    toner_level: int | None = Field(default=None, ge=0, le=100)
    drum_level: int | None = Field(default=None, ge=0, le=100)
    fuser_level: int | None = Field(default=None, ge=0, le=100)
    mileage: int | None = Field(default=None, ge=0)
    service_interval: int | None = None
    unit_raw: bool = False


class InvalidRecord(BaseModel):
    """Record that failed validation during ingestion."""

    row_number: int
    raw_data: dict[str, Any]
    reason: str
    field: str | None = None


class SchemaProfile(BaseModel):
    """Predefined field mapping profile for known data sources."""

    name: str
    vendor: str | None = None
    description: str = ""
    mapping: dict[str, str]
    date_format: str | None = None
    encoding: str | None = None


class IngestionResult(BaseModel):
    """Result of the data ingestion pipeline."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    success: bool
    source_file_info: SourceFileInfo
    mapping_used: dict[str, str] = Field(default_factory=dict)
    profile_applied: str | None = None
    total_records: int = 0
    valid_events_count: int = 0
    valid_snapshots_count: int = 0
    invalid_records: list[InvalidRecord] = Field(default_factory=list)
    devices_count: int = 0
    date_range: tuple[datetime, datetime] | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    ingestion_duration_seconds: float = 0.0
    data_checksum: str | None = None


# ── Exports ────────────────────────────────────────────────────────────────────

__all__ = [
    "AgeConfig",
    "AgentMode",
    "AgentTraceSummary",
    "BatchContext",
    "CalculationSnapshot",
    "CategoryBreakdown",
    "ChatContext",
    "ConfidenceConfig",
    "ConfidenceFactors",
    "ConfidencePenalties",
    "ConfidenceZone",
    "ContentType",
    "ContextConfig",
    "ContextModifier",
    "DeviceReport",
    "DistributionBin",
    # Track C: Report
    "DocReference",
    # Track A: Health Index
    "Factor",
    "FactorContribution",
    "FileFormat",
    "FleetSummary",
    "HealthResult",
    "HealthZone",
    "HistoryPoint",
    "IngestionResult",
    "InvalidRecord",
    "LearnedPattern",
    # Track E: Ingestion
    "NormalizedEvent",
    "PatternGroup",
    "PatternType",
    "RAGSummary",
    "ReflectionAction",
    "ReflectionIssue",
    "ReflectionResult",
    "ReflectionVerdict",
    "RepeatabilityConfig",
    "Report",
    "ResourceSnapshot",
    "ResourceState",
    "SchemaProfile",
    # Enums
    "SeverityLevel",
    "SeverityResult",
    "SeverityWeights",
    "SilentDeviceMode",
    "SimilarityDimension",
    "SourceFileInfo",
    "Trace",
    # Track B: Agent
    "TraceStep",
    "TraceStepType",
    "WeightsProfile",
    "ZoneThresholds",
]
