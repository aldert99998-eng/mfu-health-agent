"""Example instances of all data models for testing."""

from datetime import UTC, datetime

from data_io.models import (
    AgentMode,
    BatchContext,
    CalculationSnapshot,
    ConfidenceFactors,
    ConfidenceZone,
    DeviceReport,
    Factor,
    FactorContribution,
    FileFormat,
    FleetSummary,
    HealthResult,
    HealthZone,
    IngestionResult,
    InvalidRecord,
    LearnedPattern,
    NormalizedEvent,
    ReflectionAction,
    ReflectionResult,
    ReflectionVerdict,
    Report,
    ResourceSnapshot,
    ResourceState,
    SchemaProfile,
    SeverityLevel,
    SeverityResult,
    SourceFileInfo,
    Trace,
    TraceStep,
    TraceStepType,
    WeightsProfile,
)

NOW = datetime.now(UTC)

FACTOR_EXAMPLE = Factor(
    error_code="C6000",
    severity_level=SeverityLevel.CRITICAL,
    S=60,
    n_repetitions=3,
    R=2.58,
    C=1.5,
    A=0.8,
    event_timestamp=NOW,
    age_days=2,
    applicable_modifiers=["fuser>90"],
    source="Service Manual TASKalfa 3253ci, раздел 7.2",
    confidence_flags=[],
)

CONFIDENCE_FACTORS_EXAMPLE = ConfidenceFactors(
    rag_missing_count=1,
    missing_resources=False,
    missing_model=False,
)

FACTOR_CONTRIBUTION_EXAMPLE = FactorContribution(
    label="Отказ фьюзера (C6000)",
    penalty=30.96,
    S=60,
    R=2.58,
    C=1.5,
    A=0.8,
    source="Service Manual TASKalfa 3253ci, раздел 7.2",
)

HEALTH_RESULT_EXAMPLE = HealthResult(
    device_id="D001",
    health_index=69,
    confidence=0.7,
    zone=HealthZone.YELLOW,
    confidence_zone=ConfidenceZone.MEDIUM,
    factor_contributions=[FACTOR_CONTRIBUTION_EXAMPLE],
    confidence_reasons=["RAG: 1 код без документации"],
    calculated_at=NOW,
)

WEIGHTS_PROFILE_EXAMPLE = WeightsProfile(profile_name="default")

TRACE_STEP_EXAMPLE = TraceStep(
    step_number=1,
    type=TraceStepType.TOOL_CALL,
    tool_name="get_device_events",
    tool_args={"device_id": "D001", "window_days": 30},
    tool_result_summary="12 events found",
    duration_ms=150,
)

TRACE_EXAMPLE = Trace(
    session_id="sess-001",
    mode=AgentMode.BATCH,
    device_id="D001",
    started_at=NOW,
    total_tool_calls=5,
    total_llm_calls=3,
    total_tokens=2400,
)

LEARNED_PATTERN_EXAMPLE = LearnedPattern(
    scope="model:Kyocera TASKalfa 3253ci",
    observation="C6000 часто сопровождается C6020 в течение 48 часов",
    evidence_devices=["D001", "D017"],
)

SEVERITY_RESULT_EXAMPLE = SeverityResult(
    severity=SeverityLevel.CRITICAL,
    confidence=0.9,
    affected_components=["fuser", "heat_roller"],
    source="Service Manual TASKalfa 3253ci, раздел 7.2",
    reasoning="C6000 — отказ термоблока, критичный для работы устройства",
)

REFLECTION_RESULT_EXAMPLE = ReflectionResult(
    verdict=ReflectionVerdict.APPROVED,
    issues=[],
    recommended_action=ReflectionAction.ACCEPT,
)

BATCH_CONTEXT_EXAMPLE = BatchContext(weights_profile=WEIGHTS_PROFILE_EXAMPLE)

FLEET_SUMMARY_EXAMPLE = FleetSummary(
    total_devices=152,
    average_index=76.3,
    median_index=82.0,
    zone_counts={"green": 94, "yellow": 41, "red": 17},
    average_confidence=0.87,
)

DEVICE_REPORT_EXAMPLE = DeviceReport(
    device_id="D001",
    model="Kyocera TASKalfa 3253ci",
    location="Офис 3, этаж 2",
    health_index=69,
    confidence=0.7,
    zone=HealthZone.YELLOW,
    confidence_zone=ConfidenceZone.MEDIUM,
    top_problem_tag="fuser",
    factor_contributions=[FACTOR_CONTRIBUTION_EXAMPLE],
    resource_state=ResourceState(toner=45, drum=72, fuser=95, mileage=182_000),
    agent_recommendation="Заменить фьюзер в ближайшее ТО.",
)

CALCULATION_SNAPSHOT_EXAMPLE = CalculationSnapshot(
    weights_profile_name="default",
    weights_profile_version="1.0",
    weights_data={"severity": {"critical": 60}},
    llm_model="qwen2.5-7b-instruct",
    source_file_hash="abc123",
    input_record_count=1500,
    valid_record_count=1480,
    discarded_record_count=20,
)

SOURCE_FILE_INFO_EXAMPLE = SourceFileInfo(
    file_name="fleet_data_2024.csv",
    file_hash="sha256:abcdef",
    file_format=FileFormat.CSV,
    uploaded_at=NOW,
)

REPORT_EXAMPLE = Report(
    report_id="rpt-001",
    generated_at=NOW,
    source_file_name="fleet_data_2024.csv",
    source_file_hash="sha256:abcdef",
    analysis_window_days=30,
    fleet_summary=FLEET_SUMMARY_EXAMPLE,
    executive_summary="Парк в целом в хорошем состоянии.",
    devices=[DEVICE_REPORT_EXAMPLE],
    calculation_snapshot=CALCULATION_SNAPSHOT_EXAMPLE,
)

NORMALIZED_EVENT_EXAMPLE = NormalizedEvent(
    device_id="D001",
    timestamp=NOW,
    error_code="C6000",
    error_description="Fuser unit failure",
    model="Kyocera TASKalfa 3253ci",
    location="Офис 3",
)

RESOURCE_SNAPSHOT_EXAMPLE = ResourceSnapshot(
    device_id="D001",
    timestamp=NOW,
    toner_level=45,
    drum_level=72,
    fuser_level=95,
    mileage=182_000,
)

SCHEMA_PROFILE_EXAMPLE = SchemaProfile(
    name="kyocera_fleet_csv",
    vendor="Kyocera",
    description="Standard Kyocera fleet export CSV format",
    mapping={
        "Serial Number": "device_id",
        "Date/Time": "timestamp",
        "Error Code": "error_code",
        "Model Name": "model",
    },
)

INGESTION_RESULT_EXAMPLE = IngestionResult(
    success=True,
    source_file_info=SOURCE_FILE_INFO_EXAMPLE,
    mapping_used={"Serial Number": "device_id", "Date/Time": "timestamp"},
    total_records=1500,
    valid_events_count=1200,
    valid_snapshots_count=280,
    invalid_records=[
        InvalidRecord(row_number=42, raw_data={"Serial Number": ""}, reason="empty device_id"),
    ],
    devices_count=152,
    date_range=(NOW, NOW),
    ingestion_duration_seconds=3.2,
)
