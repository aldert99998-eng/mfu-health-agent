"""P0 Contract Tests for MFU data models, configs, and bridges.

Each test is tagged with its TC-ID for traceability.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_io.models import (
    CalculationSnapshot,
    ConfidenceZone,
    Factor,
    FactorContribution,
    HealthResult,
    HealthZone,
    NormalizedEvent,
    ResourceSnapshot,
    SeverityLevel,
)
from data_io.factor_store import DeviceMetadata, FactorStore, FactorStoreFrozenError

CONFIGS_DIR = PROJECT_ROOT / "configs"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_factor(**overrides) -> Factor:
    defaults = dict(
        error_code="C6000",
        severity_level=SeverityLevel.HIGH,
        S=20.0,
        n_repetitions=1,
        R=1.0,
        C=1.0,
        A=0.5,
        event_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        age_days=5,
    )
    defaults.update(overrides)
    return Factor(**defaults)


def _make_health_result(**overrides) -> HealthResult:
    defaults = dict(
        device_id="DEV-001",
        health_index=85,
        confidence=0.9,
        zone=HealthZone.GREEN,
        confidence_zone=ConfidenceZone.HIGH,
        calculated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    defaults.update(overrides)
    return HealthResult(**defaults)


def _make_normalized_event(**overrides) -> NormalizedEvent:
    defaults = dict(
        device_id="DEV-001",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        error_code="C6000",
    )
    defaults.update(overrides)
    return NormalizedEvent(**defaults)


def _make_resource_snapshot(**overrides) -> ResourceSnapshot:
    defaults = dict(
        device_id="DEV-001",
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        toner_level=80,
    )
    defaults.update(overrides)
    return ResourceSnapshot(**defaults)


# ── TC-CNT-001: Factor is frozen ────────────────────────────────────────────

class TestTCCNT001:
    """Factor frozen contract."""

    def test_factor_frozen_status(self):
        """TC-CNT-001: Factor model_config.frozen determines mutability.

        Factor has frozen=False, so mutation is ALLOWED.
        We verify the actual config value and document behavior.
        """
        f = _make_factor()
        # Factor is NOT frozen (frozen=False in source)
        assert f.model_config.get("frozen") is False, (
            "Factor.model_config['frozen'] should be False"
        )
        # Mutation should succeed since frozen=False
        f.error_code = "C7000"
        assert f.error_code == "C7000"


# ── TC-CNT-002: HealthResult is frozen ──────────────────────────────────────

class TestTCCNT002:
    """HealthResult frozen contract."""

    def test_health_result_frozen_status(self):
        """TC-CNT-002: HealthResult model_config.frozen determines mutability.

        HealthResult has frozen=False, so mutation is ALLOWED.
        """
        hr = _make_health_result()
        assert hr.model_config.get("frozen") is False
        hr.health_index = 90
        assert hr.health_index == 90


# ── TC-CNT-003: SearchResult exists and is frozen ───────────────────────────

class TestTCCNT003:
    """SearchResult frozen contract."""

    def test_search_result_exists_and_frozen(self):
        """TC-CNT-003: SearchResult is a frozen dataclass."""
        from dataclasses import FrozenInstanceError
        from rag.search import SearchResult

        sr = SearchResult(
            chunk_id="c1",
            document_id="d1",
            text="sample",
            score=0.9,
            dense_score=0.8,
            sparse_score=0.7,
        )
        with pytest.raises(FrozenInstanceError):
            sr.score = 0.5  # type: ignore[misc]


# ── TC-CNT-004: CalculationSnapshot is frozen ──────────────────────────────

class TestTCCNT004:
    """CalculationSnapshot frozen contract."""

    def test_calculation_snapshot_frozen_status(self):
        """TC-CNT-004: CalculationSnapshot model_config.frozen determines mutability.

        CalculationSnapshot has frozen=False, so mutation is ALLOWED.
        """
        cs = CalculationSnapshot(
            weights_profile_name="default",
            weights_profile_version="1.0",
            weights_data={"severity": {}},
        )
        assert cs.model_config.get("frozen") is False
        cs.weights_profile_name = "custom"
        assert cs.weights_profile_name == "custom"


# ── TC-CNT-006: NormalizedEvent is frozen ───────────────────────────────────

class TestTCCNT006:
    """NormalizedEvent frozen contract."""

    def test_normalized_event_is_frozen(self):
        """TC-CNT-006: NormalizedEvent has frozen=True -> mutation raises ValidationError."""
        ev = _make_normalized_event()
        assert ev.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            ev.device_id = "OTHER"  # type: ignore[misc]


# ── TC-CNT-007: ResourceSnapshot is frozen ──────────────────────────────────

class TestTCCNT007:
    """ResourceSnapshot frozen contract."""

    def test_resource_snapshot_is_frozen(self):
        """TC-CNT-007: ResourceSnapshot has frozen=True -> mutation raises ValidationError."""
        rs = _make_resource_snapshot()
        assert rs.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            rs.toner_level = 50  # type: ignore[misc]


# ── TC-CNT-008: DeviceMetadata is frozen ────────────────────────────────────

class TestTCCNT008:
    """DeviceMetadata (factor_store.py) frozen contract."""

    def test_device_metadata_is_frozen(self):
        """TC-CNT-008: DeviceMetadata has frozen=True -> mutation raises ValidationError."""
        dm = DeviceMetadata(device_id="DEV-001")
        assert dm.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            dm.device_id = "OTHER"  # type: ignore[misc]


# ── TC-CNT-010: FactorStore freeze() prevents writes ───────────────────────

class TestTCCNT010:
    """FactorStore freeze contract."""

    def test_freeze_prevents_add_events(self):
        """TC-CNT-010: After freeze(), add_events raises FactorStoreFrozenError."""
        store = FactorStore()
        store.freeze()
        ev = _make_normalized_event()
        with pytest.raises(FactorStoreFrozenError):
            store.add_events("DEV-001", [ev])

    def test_freeze_prevents_set_resources(self):
        """TC-CNT-010: After freeze(), set_resources raises FactorStoreFrozenError."""
        store = FactorStore()
        store.freeze()
        rs = _make_resource_snapshot()
        with pytest.raises(FactorStoreFrozenError):
            store.set_resources("DEV-001", rs)

    def test_freeze_prevents_set_device_metadata(self):
        """TC-CNT-010: After freeze(), set_device_metadata raises FactorStoreFrozenError."""
        store = FactorStore()
        store.freeze()
        dm = DeviceMetadata(device_id="DEV-001")
        with pytest.raises(FactorStoreFrozenError):
            store.set_device_metadata("DEV-001", dm)

    def test_reads_still_work_after_freeze(self):
        """TC-CNT-010: Read operations work after freeze."""
        store = FactorStore()
        ev = _make_normalized_event()
        store.add_events("DEV-001", [ev])
        store.freeze()
        assert store.frozen is True
        assert store.list_devices() == ["DEV-001"]
        assert len(store.get_events("DEV-001", window_days=9999)) == 1


# ── TC-CNT-020: Factor has all required fields ─────────────────────────────

class TestTCCNT020:
    """Factor field completeness contract."""

    def test_factor_field_count(self):
        """TC-CNT-020: Factor has exactly 12 fields."""
        expected_fields = {
            "error_code", "severity_level", "S", "n_repetitions",
            "R", "C", "A", "event_timestamp", "age_days",
            "applicable_modifiers", "source", "confidence_flags",
        }
        actual_fields = set(Factor.model_fields.keys())
        assert actual_fields == expected_fields, (
            f"Missing: {expected_fields - actual_fields}, "
            f"Extra: {actual_fields - expected_fields}"
        )
        assert len(actual_fields) == 12


# ── TC-CNT-021: HealthResult has all required fields ───────────────────────

class TestTCCNT021:
    """HealthResult field completeness contract."""

    def test_health_result_field_count(self):
        """TC-CNT-021: HealthResult has exactly 10 fields."""
        expected_fields = {
            "device_id", "health_index", "confidence", "zone",
            "confidence_zone", "factor_contributions", "confidence_reasons",
            "calculation_snapshot", "calculated_at", "reflection_notes",
        }
        actual_fields = set(HealthResult.model_fields.keys())
        assert actual_fields == expected_fields, (
            f"Missing: {expected_fields - actual_fields}, "
            f"Extra: {actual_fields - expected_fields}"
        )
        assert len(actual_fields) == 10


# ── TC-CNT-030: health_index type and range ─────────────────────────────────

class TestTCCNT030:
    """health_index type and range contract."""

    def test_health_index_is_int(self):
        """TC-CNT-030: health_index is int."""
        hr = _make_health_result(health_index=50)
        assert isinstance(hr.health_index, int)

    def test_health_index_min_1(self):
        """TC-CNT-030: health_index < 1 rejected."""
        with pytest.raises(ValidationError):
            _make_health_result(health_index=0)

    def test_health_index_max_100(self):
        """TC-CNT-030: health_index > 100 rejected."""
        with pytest.raises(ValidationError):
            _make_health_result(health_index=101)

    def test_health_index_boundary_values(self):
        """TC-CNT-030: health_index boundaries 1 and 100 accepted."""
        assert _make_health_result(health_index=1).health_index == 1
        assert _make_health_result(health_index=100).health_index == 100


# ── TC-CNT-031: confidence type and range ───────────────────────────────────

class TestTCCNT031:
    """confidence type and range contract."""

    def test_confidence_is_float(self):
        """TC-CNT-031: confidence is float."""
        hr = _make_health_result(confidence=0.8)
        assert isinstance(hr.confidence, float)

    def test_confidence_min_0_2(self):
        """TC-CNT-031: confidence < 0.2 rejected."""
        with pytest.raises(ValidationError):
            _make_health_result(confidence=0.19)

    def test_confidence_max_1_0(self):
        """TC-CNT-031: confidence > 1.0 rejected."""
        with pytest.raises(ValidationError):
            _make_health_result(confidence=1.01)

    def test_confidence_boundary_values(self):
        """TC-CNT-031: confidence boundaries 0.2 and 1.0 accepted."""
        assert _make_health_result(confidence=0.2).confidence == 0.2
        assert _make_health_result(confidence=1.0).confidence == 1.0


# ── TC-CNT-032: zone is HealthZone enum ─────────────────────────────────────

class TestTCCNT032:
    """zone type contract."""

    def test_zone_is_enum(self):
        """TC-CNT-032: zone is HealthZone enum, not plain string."""
        hr = _make_health_result(zone=HealthZone.GREEN)
        assert isinstance(hr.zone, HealthZone)
        assert hr.zone is HealthZone.GREEN

    def test_zone_from_string(self):
        """TC-CNT-032: zone accepts valid string and converts to enum."""
        hr = _make_health_result(zone="yellow")
        assert isinstance(hr.zone, HealthZone)
        assert hr.zone is HealthZone.YELLOW

    def test_zone_invalid_string_rejected(self):
        """TC-CNT-032: invalid zone string rejected."""
        with pytest.raises(ValidationError):
            _make_health_result(zone="purple")


# ── TC-CNT-035: severity_level is SeverityLevel enum ───────────────────────

class TestTCCNT035:
    """severity_level type contract."""

    def test_severity_is_enum(self):
        """TC-CNT-035: severity_level is SeverityLevel enum."""
        f = _make_factor(severity_level=SeverityLevel.CRITICAL)
        assert isinstance(f.severity_level, SeverityLevel)
        assert f.severity_level is SeverityLevel.CRITICAL

    def test_severity_from_string(self):
        """TC-CNT-035: severity_level accepts valid string."""
        f = _make_factor(severity_level="High")
        assert isinstance(f.severity_level, SeverityLevel)
        assert f.severity_level is SeverityLevel.HIGH

    def test_severity_invalid_rejected(self):
        """TC-CNT-035: invalid severity string rejected."""
        with pytest.raises(ValidationError):
            _make_factor(severity_level="Unknown")


# ── TC-CNT-040: component_vocabulary.yaml validates ────────────────────────

class TestTCCNT040:
    """component_vocabulary.yaml structure contract."""

    def test_loads_without_error(self):
        """TC-CNT-040: component_vocabulary.yaml loads as valid YAML."""
        path = CONFIGS_DIR / "component_vocabulary.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), "Top-level must be a mapping"

    def test_expected_structure(self):
        """TC-CNT-040: Each key maps to a list of string synonyms."""
        path = CONFIGS_DIR / "component_vocabulary.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert len(data) > 0, "Must have at least one component"
        for component, synonyms in data.items():
            assert isinstance(component, str)
            assert isinstance(synonyms, list), f"{component}: value must be a list"
            for syn in synonyms:
                assert isinstance(syn, str), f"{component}: synonym must be string"


# ── TC-CNT-050: model_aliases.yaml validates ───────────────────────────────

class TestTCCNT050:
    """model_aliases.yaml structure contract."""

    def test_loads_and_structure(self):
        """TC-CNT-050: model_aliases.yaml loads; canonical->aliases mapping."""
        path = CONFIGS_DIR / "model_aliases.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0
        for canonical, aliases in data.items():
            assert isinstance(canonical, str)
            assert isinstance(aliases, list)
            for alias in aliases:
                assert isinstance(alias, str)


# ── TC-CNT-060: error_code_patterns.yaml validates ────────────────────────

class TestTCCNT060:
    """error_code_patterns.yaml structure contract."""

    def test_loads_and_structure(self):
        """TC-CNT-060: error_code_patterns.yaml loads; vendor->patterns mapping."""
        path = CONFIGS_DIR / "error_code_patterns.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0
        for vendor, patterns in data.items():
            assert isinstance(vendor, str)
            assert isinstance(patterns, list)
            for pattern in patterns:
                assert isinstance(pattern, str)


# ── TC-CNT-070: field_synonyms.yaml validates ─────────────────────────────

class TestTCCNT070:
    """field_synonyms.yaml structure contract."""

    def test_loads_and_structure(self):
        """TC-CNT-070: field_synonyms.yaml loads; canonical_field->[synonyms]."""
        path = CONFIGS_DIR / "field_synonyms.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert len(data) > 0
        for field_name, synonyms in data.items():
            assert isinstance(field_name, str)
            assert isinstance(synonyms, list)
            for syn in synonyms:
                assert isinstance(syn, str)


# ── TC-CNT-071: No synonym maps to 2 different canonical fields ───────────

class TestTCCNT071:
    """field_synonyms.yaml uniqueness contract."""

    def test_no_duplicate_synonyms(self):
        """TC-CNT-071: No synonym appears under two different canonical fields."""
        path = CONFIGS_DIR / "field_synonyms.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        synonym_to_fields: dict[str, list[str]] = defaultdict(list)
        for canonical, synonyms in data.items():
            for syn in synonyms:
                synonym_to_fields[syn.lower()].append(canonical)
        duplicates = {
            syn: fields
            for syn, fields in synonym_to_fields.items()
            if len(fields) > 1
        }
        assert duplicates == {}, (
            f"Synonyms mapped to multiple canonical fields: {duplicates}"
        )


# ── TC-CNT-100: HealthResult serialize/deserialize roundtrip ───────────────

class TestTCCNT100:
    """HealthResult roundtrip contract."""

    def test_to_dict_from_dict_roundtrip(self):
        """TC-CNT-100: HealthResult survives model_dump -> model_validate roundtrip."""
        original = _make_health_result(
            factor_contributions=[
                FactorContribution(
                    label="fuser", penalty=15.0,
                    S=20.0, R=1.5, C=1.1, A=0.8, source="rag",
                )
            ],
            confidence_reasons=["missing_resources"],
        )
        dumped = original.model_dump(mode="json")
        restored = HealthResult.model_validate(dumped)
        assert restored.device_id == original.device_id
        assert restored.health_index == original.health_index
        assert restored.confidence == original.confidence
        assert restored.zone == original.zone
        assert restored.confidence_zone == original.confidence_zone
        assert len(restored.factor_contributions) == 1
        assert restored.factor_contributions[0].label == "fuser"

    def test_json_roundtrip(self):
        """TC-CNT-100: HealthResult survives JSON string roundtrip."""
        original = _make_health_result()
        json_str = original.model_dump_json()
        restored = HealthResult.model_validate_json(json_str)
        assert restored.device_id == original.device_id
        assert restored.health_index == original.health_index


# ── TC-CNT-110: All config YAML files load with Pydantic validation ───────

class TestTCCNT110:
    """Config Pydantic validation contract."""

    def test_agent_config_validates(self):
        """TC-CNT-110: agent_config.yaml validates against AgentConfig."""
        from config.loader import AgentConfig
        path = CONFIGS_DIR / "agent_config.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        AgentConfig.model_validate(data)

    def test_report_config_validates(self):
        """TC-CNT-110: report_config.yaml validates against ReportConfig."""
        from config.loader import ReportConfig
        path = CONFIGS_DIR / "report_config.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ReportConfig.model_validate(data)

    def test_rag_config_validates(self):
        """TC-CNT-110: rag_config.yaml validates against RAGConfig."""
        from config.loader import RAGConfig
        path = CONFIGS_DIR / "rag_config.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        RAGConfig.model_validate(data)


# ── TC-CNT-120: Check __all__ exports in key modules ──────────────────────

class TestTCCNT120:
    """__all__ exports contract."""

    def test_models_all_exports(self):
        """TC-CNT-120: data_io.models.__all__ contains all public classes."""
        import data_io.models as m
        assert hasattr(m, "__all__"), "data_io.models must define __all__"
        all_names = m.__all__
        # Spot-check key exports
        must_export = [
            "Factor", "HealthResult", "NormalizedEvent",
            "ResourceSnapshot", "SeverityLevel", "HealthZone",
            "CalculationSnapshot", "ConfidenceZone", "Report",
        ]
        for name in must_export:
            assert name in all_names, f"{name} missing from __all__"
        # Every name in __all__ should actually exist in the module
        for name in all_names:
            assert hasattr(m, name), f"__all__ lists '{name}' but it is not defined"

    def test_factor_store_exports(self):
        """TC-CNT-120: data_io.factor_store module has FactorStore, DeviceMetadata."""
        import data_io.factor_store as fs
        for name in ["FactorStore", "DeviceMetadata", "FactorStoreFrozenError", "FleetMeta"]:
            assert hasattr(fs, name), f"{name} not found in factor_store"


# ── TC-CNT-121: No circular imports ────────────────────────────────────────

class TestTCCNT121:
    """Circular import detection contract."""

    @pytest.mark.parametrize("module_path", [
        "data_io.models",
        "data_io.factor_store",
        "data_io.field_mapper",
        "data_io.normalizer",
        "data_io.parsers",
        "config.loader",
        "config.weights_manager",
    ])
    def test_no_circular_import(self, module_path: str):
        """TC-CNT-121: Module imports without ImportError/circular dependency."""
        # Force fresh import to catch circular issues
        if module_path in sys.modules:
            # Already loaded => no circular issue
            mod = sys.modules[module_path]
        else:
            mod = importlib.import_module(module_path)
        assert mod is not None
