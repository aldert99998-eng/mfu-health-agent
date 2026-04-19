"""QA tests for data_io/field_mapper.py — TC-E-020 through TC-E-027."""

import sys
import os
import json
import shutil
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from mfu_agent.data_io.field_mapper import (
    SynonymMatcher,
    ContentMatcher,
    LLMMatcher,
    FieldMapper,
    MappingResult,
    ColumnMapping,
    ConfidenceLevel,
    save_profile,
    try_apply_profile,
    SYNONYMS_PATH,
    PROFILES_DIR,
)

results = []

# ── TC-E-020 ── Synonym matching: "device_serial" → "device_id"
def test_020():
    tag = "TC-E-020"
    try:
        # Check synonyms file exists
        assert SYNONYMS_PATH.exists(), f"synonyms file not found at {SYNONYMS_PATH}"
        with open(SYNONYMS_PATH) as f:
            syns = yaml.safe_load(f)
        device_id_syns = syns.get("device_id", [])
        has_device_serial = "device_serial" in device_id_syns
        note_syn = f"'device_serial' in synonyms list: {has_device_serial}"

        df = pd.DataFrame({"device_serial": ["SN001", "SN002"]})
        matcher = SynonymMatcher()
        mapping = matcher.match(list(df.columns))

        if mapping.get("device_serial") == "device_id":
            results.append(f"{tag} | PASS | {note_syn}; mapped device_serial→device_id")
        else:
            results.append(
                f"{tag} | FAIL | {note_syn}; mapping={mapping}. "
                f"'device_serial' is NOT listed in field_synonyms.yaml under device_id"
            )
    except Exception as e:
        results.append(f"{tag} | FAIL | Exception: {e}")

# ── TC-E-021 ── Profile signature match
def test_021():
    tag = "TC-E-021"
    profile_name = "__test_profile_021"
    profile_path = None
    try:
        columns = ["col_a", "col_b", "col_c"]
        mapping = {"col_a": "device_id", "col_b": "timestamp"}
        profile_path = save_profile(profile_name, columns, mapping)
        assert profile_path.exists(), "profile file not created"

        result = try_apply_profile(columns)
        assert result is not None, "try_apply_profile returned None"
        returned_mapping, returned_name = result
        assert returned_mapping == mapping, f"mapping mismatch: {returned_mapping}"
        assert returned_name == profile_name, f"name mismatch: {returned_name}"
        results.append(f"{tag} | PASS | profile saved, signature matched, mapping+name returned correctly")
    except Exception as e:
        results.append(f"{tag} | FAIL | Exception: {e}")
    finally:
        if profile_path and profile_path.exists():
            profile_path.unlink()

# ── TC-E-022 ── LLM matching with mock valid JSON
def test_022():
    tag = "TC-E-022"
    try:
        class MockLLMClient:
            def complete(self, prompt: str) -> str:
                return json.dumps({"mapping": {"mystery_col": "device_id"}})

        df = pd.DataFrame({
            "mystery_col": ["ABC123", "DEF456", "GHI789"],
            "other_col": [1, 2, 3],
        })
        matcher = LLMMatcher(client=MockLLMClient())
        mapping = matcher.match(df, already_mapped={})
        assert mapping.get("mystery_col") == "device_id", f"mapping={mapping}"
        results.append(f"{tag} | PASS | mock LLM returned valid JSON, mystery_col→device_id mapped")
    except Exception as e:
        results.append(f"{tag} | FAIL | Exception: {e}")

# ── TC-E-023 ── LLM returns invalid (plain text) response
def test_023():
    tag = "TC-E-023"
    try:
        class BadLLMClient:
            def complete(self, prompt: str) -> str:
                return "I don't know how to map these columns, sorry!"

        df = pd.DataFrame({"some_col": ["x", "y", "z"]})
        matcher = LLMMatcher(client=BadLLMClient())
        mapping = matcher.match(df, already_mapped={}, max_retries=2)
        assert mapping == {}, f"Expected empty mapping, got {mapping}"
        results.append(f"{tag} | PASS | invalid LLM response → graceful fallback, empty mapping")
    except Exception as e:
        results.append(f"{tag} | FAIL | Exception: {e}")

# ── TC-E-025 ── Ambiguous: two columns look like device_id, only one wins
def test_025():
    tag = "TC-E-025"
    try:
        n = 30  # > MIN_ROWS_FOR_UNIQUENESS (20)
        df = pd.DataFrame({
            "serial_number": [f"SN{i:04d}" for i in range(n)],
            "device_code":   [f"DC{i:04d}" for i in range(n)],
        })
        # serial_number is a synonym for device_id, so SynonymMatcher should grab it first.
        # ContentMatcher should not re-assign device_id to device_code.
        synonym_matcher = SynonymMatcher()
        syn_map = synonym_matcher.match(list(df.columns))

        content_matcher = ContentMatcher()
        content_map = content_matcher.match(df, syn_map)

        combined = {**syn_map, **content_map}
        device_id_cols = [c for c, t in combined.items() if t == "device_id"]
        if len(device_id_cols) == 1:
            results.append(
                f"{tag} | PASS | only '{device_id_cols[0]}' mapped to device_id (first wins)"
            )
        else:
            results.append(f"{tag} | FAIL | device_id mapped to {device_id_cols}, expected exactly 1")
    except Exception as e:
        results.append(f"{tag} | FAIL | Exception: {e}")

# ── TC-E-026 ── Save new profile, verify YAML structure
def test_026():
    tag = "TC-E-026"
    profile_name = "new_prof_026"
    profile_path = None
    try:
        columns = ["alpha", "beta", "gamma"]
        mapping = {"alpha": "device_id", "beta": "error_code"}
        profile_path = save_profile(profile_name, columns, mapping)

        assert profile_path.exists(), "YAML file not created"
        assert profile_path.name == f"{profile_name}.yaml"

        with open(profile_path) as f:
            data = yaml.safe_load(f)

        assert data["profile_name"] == profile_name
        assert "signature_hash" in data and len(data["signature_hash"]) == 64
        assert data["column_mapping"] == mapping
        results.append(
            f"{tag} | PASS | {profile_path} created with correct name, signature, and mapping"
        )
    except Exception as e:
        results.append(f"{tag} | FAIL | Exception: {e}")
    finally:
        if profile_path and profile_path.exists():
            profile_path.unlink()

# ── TC-E-027 ── FieldMapper.map() with existing profile → needs_confirmation=True
def test_027():
    tag = "TC-E-027"
    profile_name = "__test_profile_027"
    profile_path = None
    try:
        columns = ["x_col", "y_col", "z_col"]
        mapping = {"x_col": "device_id", "y_col": "timestamp"}
        profile_path = save_profile(profile_name, columns, mapping)

        df = pd.DataFrame({
            "x_col": ["A", "B"],
            "y_col": ["2024-01-01", "2024-01-02"],
            "z_col": [1, 2],
        })

        mapper = FieldMapper(llm_client=None)
        result = mapper.map(df)

        assert result.needs_confirmation is True, f"needs_confirmation={result.needs_confirmation}"
        assert result.profile_applied is not None, f"profile_applied is None"
        assert result.profile_applied == profile_name
        results.append(
            f"{tag} | PASS | needs_confirmation=True, profile_applied='{result.profile_applied}'"
        )
    except Exception as e:
        results.append(f"{tag} | FAIL | Exception: {e}")
    finally:
        if profile_path and profile_path.exists():
            profile_path.unlink()


# ── Run all ──
if __name__ == "__main__":
    test_020()
    test_021()
    test_022()
    test_023()
    test_025()
    test_026()
    test_027()

    print()
    for r in results:
        print(r)
