"""Tests for the per-model error_codes registry."""

from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import pytest
import yaml

import error_codes
from error_codes import (
    ErrorCode,
    ModelErrorCodes,
    ParseError,
    delete,
    find_conflicts,
    invalidate_cache,
    list_models,
    list_vendors,
    load_codes,
    model_slug,
    parse_csv,
    parse_file,
    parse_xlsx,
    parse_yaml,
    save,
    vendor_slug,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Redirect ERROR_CODES_ROOT to a temp dir and clear cache."""
    ec_root = tmp_path / "error_codes"
    ec_root.mkdir()
    trash = ec_root / "_trash"
    trash.mkdir()
    monkeypatch.setattr(error_codes.loader, "ERROR_CODES_ROOT", ec_root)
    monkeypatch.setattr(error_codes.loader, "TRASH_DIR", trash)
    invalidate_cache()
    yield ec_root
    invalidate_cache()


def _sample_doc(vendor="Xerox", model="B8090") -> ModelErrorCodes:
    return ModelErrorCodes(
        vendor=vendor,
        model=model,
        codes={
            "09-605-00": ErrorCode(
                description="Fuser EOL",
                severity="Critical",
                component="fuser",
            ),
            "07-545-00": ErrorCode(
                description="Tray 2 empty",
                severity="Low",
                component="tray",
            ),
        },
    )


# ── Slugs ───────────────────────────────────────────────────────────────────


class TestSlug:
    def test_vendor_slug_canonical(self) -> None:
        assert vendor_slug("Xerox") == "xerox"
        assert vendor_slug(" LEXMARK ") == "lexmark"
        assert vendor_slug("") == "unknown"

    def test_model_slug_strips_vendor_prefix(self) -> None:
        # Aliases map all B80xx to "Xerox AltaLink B8090" → slug should omit "xerox"
        assert model_slug("B8090") == "altalink_b8090"
        assert model_slug("Xerox AltaLink B8090") == "altalink_b8090"

    def test_model_slug_handles_punctuation(self) -> None:
        assert model_slug("WorkCentre 6515/DN") == "workcentre_6515_dn"


# ── Schema ──────────────────────────────────────────────────────────────────


class TestSchema:
    def test_severity_must_be_enum(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ErrorCode(description="x", severity="Bogus")  # type: ignore[arg-type]

    def test_description_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ErrorCode(description="", severity="Low")

    def test_merge_upsert(self) -> None:
        a = _sample_doc()
        b = ModelErrorCodes(
            vendor="Xerox",
            model="B8090",
            codes={
                "09-605-00": ErrorCode(description="changed", severity="High"),
                "14-517-00": ErrorCode(description="new code", severity="Medium"),
            },
        )
        merged = a.merge(b, replace=False)
        assert len(merged.codes) == 3
        assert merged.codes["09-605-00"].severity == "High"  # overwritten
        assert merged.codes["14-517-00"].description == "new code"  # added
        assert merged.codes["07-545-00"].severity == "Low"  # untouched

    def test_merge_replace(self) -> None:
        a = _sample_doc()
        b = ModelErrorCodes(
            vendor="Xerox",
            model="B8090",
            codes={"99-999-99": ErrorCode(description="only", severity="Info")},
        )
        merged = a.merge(b, replace=True)
        assert set(merged.codes.keys()) == {"99-999-99"}


# ── Writer + Loader ─────────────────────────────────────────────────────────


class TestWriterLoader:
    def test_save_and_load_roundtrip(self, tmp_registry: Path) -> None:
        doc = _sample_doc()
        save(doc)
        loaded = load_codes("Xerox", "B8090")
        assert loaded is not None
        assert loaded.vendor == "Xerox"
        assert "09-605-00" in loaded.codes
        assert loaded.codes["09-605-00"].severity == "Critical"

    def test_list_vendors_and_models(self, tmp_registry: Path) -> None:
        save(_sample_doc(vendor="Xerox", model="B8090"))
        save(_sample_doc(vendor="Lexmark", model="MX321"))
        assert "Xerox" in list_vendors()
        assert "Lexmark" in list_vendors()
        assert "altalink_b8090" in list_models("Xerox")
        assert "mx321" in list_models("Lexmark")

    def test_delete_moves_to_trash(self, tmp_registry: Path) -> None:
        save(_sample_doc())
        file_path = tmp_registry / "xerox" / "altalink_b8090.yaml"
        assert file_path.exists()
        trash_path = delete("Xerox", "B8090")
        assert not file_path.exists()
        assert trash_path is not None and trash_path.exists()
        assert trash_path.parent.name == "_trash"

    def test_load_returns_none_for_missing(self, tmp_registry: Path) -> None:
        assert load_codes("Xerox", "Nonexistent") is None


# ── Parsers ─────────────────────────────────────────────────────────────────


class TestParsers:
    def test_parse_csv_basic(self) -> None:
        csv_text = (
            "code,description,severity,component,notes\n"
            "09-605-00,Fuser EOL,Critical,fuser,\n"
            "07-545-00,Tray 2 empty,Low,tray,\n"
        )
        doc = parse_csv(csv_text, vendor="Xerox", model="B8090")
        assert len(doc.codes) == 2
        assert doc.codes["09-605-00"].severity == "Critical"

    def test_parse_csv_rejects_missing_headers(self) -> None:
        bad = "code,description\n09-605-00,x\n"
        with pytest.raises(ParseError):
            parse_csv(bad, vendor="Xerox", model="B8090")

    def test_parse_csv_severity_synonym_major(self) -> None:
        csv_text = "code,description,severity\n09-605-00,Fuser,major\n"
        doc = parse_csv(csv_text, vendor="Xerox", model="B8090")
        assert doc.codes["09-605-00"].severity == "High"

    def test_parse_csv_invalid_severity(self) -> None:
        csv_text = "code,description,severity\n09-605-00,Fuser,Bogus\n"
        with pytest.raises(ParseError):
            parse_csv(csv_text, vendor="Xerox", model="B8090")

    def test_parse_xlsx_roundtrip(self) -> None:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.append(["code", "description", "severity", "component"])
        ws.append(["09-605-00", "Fuser EOL", "Critical", "fuser"])
        ws.append(["07-545-00", "Tray empty", "Low", "tray"])
        buf = io.BytesIO()
        wb.save(buf)
        doc = parse_xlsx(buf.getvalue(), vendor="Xerox", model="B8090")
        assert len(doc.codes) == 2
        assert doc.codes["09-605-00"].component == "fuser"

    def test_parse_yaml(self) -> None:
        yaml_text = yaml.safe_dump({
            "vendor": "Xerox",
            "model": "B8090",
            "codes": {
                "09-605-00": {
                    "description": "Fuser EOL",
                    "severity": "Critical",
                    "component": "fuser",
                },
            },
        })
        doc = parse_yaml(yaml_text, vendor="Xerox", model="B8090")
        assert "09-605-00" in doc.codes

    def test_parse_file_dispatch(self) -> None:
        csv_text = "code,description,severity\n09-605-00,x,Critical\n"
        doc = parse_file(
            "dict.csv", csv_text.encode(), vendor="Xerox", model="B8090",
        )
        assert len(doc.codes) == 1


# ── Consistency ─────────────────────────────────────────────────────────────


class TestConsistency:
    def test_find_conflicts_on_severity(self, tmp_registry: Path) -> None:
        # Two distinct models with the same code but different severity
        d1 = ModelErrorCodes(
            vendor="Xerox", model="Phaser",
            codes={"09-605-00": ErrorCode(description="Fuser", severity="Critical")},
        )
        d2 = ModelErrorCodes(
            vendor="Xerox", model="WorkCentre 6515",
            codes={"09-605-00": ErrorCode(description="Fuser", severity="High")},
        )
        save(d1)
        save(d2)
        conflicts = find_conflicts("Xerox", d1)
        severity_conflicts = [c for c in conflicts if c.field == "severity"]
        assert len(severity_conflicts) == 1
        assert severity_conflicts[0].this_value == "Critical"
        assert severity_conflicts[0].other_value == "High"

    def test_no_conflicts_when_identical(self, tmp_registry: Path) -> None:
        d1 = _sample_doc(model="Phaser")
        d2 = _sample_doc(model="WorkCentre 6515")
        save(d1)
        save(d2)
        conflicts = find_conflicts("Xerox", d1)
        assert conflicts == []


# ── Per-model registry is the only source ──────────────────────────────────


class TestRegistryLookup:
    def test_registry_returns_full_hit(self, tmp_registry: Path) -> None:
        """When registry has the code, _lookup_model_registry returns confidence 0.9."""
        from agent.tools.impl import _lookup_model_registry
        save(_sample_doc())
        result = _lookup_model_registry("09-605-00", model_hint="B8090")
        assert result is not None
        assert result["confidence"] == 0.9
        assert result["severity"] == "Critical"
        assert "_registry" in result["source"]

    def test_missing_code_returns_none(self, tmp_registry: Path) -> None:
        """Code not in registry → None, caller falls through to RAG/LLM or default."""
        from agent.tools.impl import _lookup_model_registry
        save(_sample_doc())
        assert _lookup_model_registry("99-999-99", model_hint="B8090") is None

    def test_no_model_hint_returns_none(self, tmp_registry: Path) -> None:
        """Without model_hint registry lookup is skipped entirely."""
        from agent.tools.impl import _lookup_model_registry
        save(_sample_doc())
        assert _lookup_model_registry("09-605-00", model_hint=None) is None


# ── Alias editing ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_aliases(tmp_path, monkeypatch):
    """Redirect MODEL_ALIASES_PATH to a throwaway file."""
    aliases_path = tmp_path / "model_aliases.yaml"
    aliases_path.write_text(
        yaml.safe_dump(
            {
                "Xerox AltaLink B8090": ["b8090", "altalink b8090"],
                "Kyocera TASKalfa 3253ci": ["taskalfa 3253ci"],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    backups = tmp_path / "_backups"
    # Patch both normalizer (source of truth) and aliases module (caches path)
    import data_io.normalizer as normalizer_mod
    import error_codes.aliases as aliases_mod
    monkeypatch.setattr(normalizer_mod, "MODEL_ALIASES_PATH", aliases_path)
    monkeypatch.setattr(aliases_mod, "MODEL_ALIASES_PATH", aliases_path)
    monkeypatch.setattr(aliases_mod, "BACKUPS_DIR", backups)
    yield aliases_path


class TestAliases:
    def test_add_new_alias(self, tmp_aliases: Path) -> None:
        from error_codes import add_alias, get_canonical_for_alias
        result = add_alias("Xerox AltaLink B8090", "B8145")
        assert result["action"] == "added"
        assert get_canonical_for_alias("b8145") == "Xerox AltaLink B8090"

    def test_add_existing_alias_noop(self, tmp_aliases: Path) -> None:
        from error_codes import add_alias
        result = add_alias("Xerox AltaLink B8090", "b8090")
        assert result["action"] == "noop"

    def test_add_conflict_without_force(self, tmp_aliases: Path) -> None:
        from error_codes import AliasConflict, add_alias
        with pytest.raises(AliasConflict) as exc:
            add_alias("Xerox AltaLink B8090", "taskalfa 3253ci")
        assert exc.value.existing_canonical == "Kyocera TASKalfa 3253ci"

    def test_add_conflict_with_force(self, tmp_aliases: Path) -> None:
        from error_codes import add_alias, get_canonical_for_alias
        result = add_alias(
            "Xerox AltaLink B8090", "taskalfa 3253ci", force_reassign=True
        )
        assert result["action"] == "reassigned"
        assert get_canonical_for_alias("taskalfa 3253ci") == "Xerox AltaLink B8090"

    def test_add_unknown_canonical(self, tmp_aliases: Path) -> None:
        from error_codes import add_alias
        with pytest.raises(ValueError):
            add_alias("NoSuchModel", "x")

    def test_remove_alias(self, tmp_aliases: Path) -> None:
        from error_codes import get_canonical_for_alias, remove_alias
        canonical = remove_alias("b8090")
        assert canonical == "Xerox AltaLink B8090"
        assert get_canonical_for_alias("b8090") is None

    def test_remove_canonical_forbidden(self, tmp_aliases: Path) -> None:
        from error_codes import remove_alias
        with pytest.raises(ValueError):
            remove_alias("Xerox AltaLink B8090")

    def test_remove_nonexistent_returns_none(self, tmp_aliases: Path) -> None:
        from error_codes import remove_alias
        assert remove_alias("completely_unknown_alias") is None

    def test_backup_created_on_write(self, tmp_aliases: Path) -> None:
        from error_codes.aliases import BACKUPS_DIR
        from error_codes import add_alias
        add_alias("Xerox AltaLink B8090", "B9999")
        assert BACKUPS_DIR.exists()
        backups = list(BACKUPS_DIR.glob("model_aliases_*.yaml"))
        assert len(backups) >= 1
