"""Semantic field mapping — Track E, Level 2.

Four-step pipeline: Synonyms → Content heuristics → LLM → User confirmation.
Maps arbitrary column names to the internal normalized schema.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

SYNONYMS_PATH = Path(__file__).resolve().parent.parent / "configs" / "field_synonyms.yaml"
PROFILES_DIR = Path(__file__).resolve().parent.parent / "configs" / "schema_profiles"
PROMPT_PATH = Path(__file__).resolve().parent.parent / "agent" / "prompts" / "field_mapping.md"

INTERNAL_FIELDS = frozenset({
    "device_id", "timestamp", "error_code", "error_description",
    "model", "vendor", "location", "status",
    "mileage", "toner_level", "drum_level", "fuser_level",
})

SAMPLE_SIZE = 5
MIN_ROWS_FOR_UNIQUENESS = 20


# ── Enums & models ──────────────────────────────────────────────────────────


class ConfidenceLevel(StrEnum):
    """How the mapping was determined."""

    SYNONYM = "synonym"
    CONTENT = "content"
    LLM = "llm"
    PROFILE = "profile"
    UNKNOWN = "unknown"


class ColumnMapping(BaseModel):
    """Mapping result for a single column."""

    model_config = ConfigDict(frozen=True)

    source_column: str
    target_field: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    samples: list[str] = Field(default_factory=list)


class MappingResult(BaseModel):
    """Complete mapping result for a DataFrame."""

    model_config = ConfigDict(frozen=True)

    auto_mapping: dict[str, str]
    column_details: list[ColumnMapping] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)
    profile_applied: str | None = None
    needs_confirmation: bool = False


class FieldMappingResponse(BaseModel):
    """Pydantic model for LLM JSON response."""

    mapping: dict[str, str]


# ── Column name normalization ────────────────────────────────────────────────


def normalize_column_name(name: str) -> str:
    """Normalize column name for synonym matching.

    lowercase, replace spaces/hyphens/dots with _, strip diacritics,
    remove parenthesized content.
    """
    s = name.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = s.strip("_")
    return s


def compute_signature(columns: list[str]) -> str:
    """SHA-256 of sorted normalized column names."""
    normalized = sorted(normalize_column_name(c) for c in columns)
    blob = json.dumps(normalized, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ── SynonymMatcher ───────────────────────────────────────────────────────────


class SynonymMatcher:
    """Step A: match columns by name using a synonym dictionary."""

    def __init__(self, synonyms_path: Path = SYNONYMS_PATH) -> None:
        self._lookup: dict[str, str] = {}
        self._load(synonyms_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Файл синонимов не найден: %s", path)
            return
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("Некорректный формат файла синонимов: %s", path)
            return
        for target, synonyms in data.items():
            if not isinstance(synonyms, list):
                continue
            for syn in synonyms:
                norm = normalize_column_name(str(syn))
                self._lookup[norm] = target

    def match(self, columns: list[str]) -> dict[str, str]:
        """Return {source_column: target_field} for matched columns."""
        result: dict[str, str] = {}
        used_targets: set[str] = set()
        for col in columns:
            norm = normalize_column_name(col)
            target = self._lookup.get(norm)
            if target and target not in used_targets:
                result[col] = target
                used_targets.add(target)
        return result


# ── ContentMatcher ───────────────────────────────────────────────────────────


class ContentMatcher:
    """Step B: match columns by inspecting their values."""

    _ERROR_CODE_RE = re.compile(r"^(?:[CJEF]\d{3,5}|S\d{3,5}|SC\d{3,4}|\d{2}-\d{3}-\d{2})$")

    def match(self, df: pd.DataFrame, already_mapped: dict[str, str]) -> dict[str, str]:
        """Return additional mappings based on column content."""
        result: dict[str, str] = {}
        used_targets = set(already_mapped.values())
        unmapped_cols = [c for c in df.columns if c not in already_mapped]

        for col in unmapped_cols:
            target = self._classify_column(df, col, used_targets)
            if target:
                result[col] = target
                used_targets.add(target)

        return result

    def _classify_column(
        self, df: pd.DataFrame, col: str, used: set[str]
    ) -> str | None:
        series = df[col].dropna().astype(str).str.strip()
        series = series[series != ""]
        if series.empty:
            return None

        sample = series.head(200)
        norm_name = normalize_column_name(col)

        if "error_code" not in used and self._is_error_code(sample):
            return "error_code"

        if self._is_numeric_0_100(sample):
            for field, hints in [
                ("toner_level", ("toner", "тонер")),
                ("drum_level", ("drum", "барабан")),
                ("fuser_level", ("fuser", "фьюзер")),
            ]:
                if field not in used and any(h in norm_name for h in hints):
                    return field

        if "timestamp" not in used and self._is_timestamp(sample):
            return "timestamp"

        if "mileage" not in used and self._is_large_int(sample, norm_name):
            return "mileage"

        if "device_id" not in used and self._is_device_id(series):
            return "device_id"

        return None

    def _is_error_code(self, sample: pd.Series) -> bool:
        return bool(sample.apply(lambda v: bool(self._ERROR_CODE_RE.match(v.upper()))).all())

    @staticmethod
    def _is_timestamp(sample: pd.Series) -> bool:
        from dateutil import parser as dateutil_parser

        parsed = 0
        for val in sample:
            try:
                dateutil_parser.parse(str(val))
                parsed += 1
            except (ValueError, OverflowError):
                pass
        return parsed >= len(sample) * 0.9

    @staticmethod
    def _is_numeric_0_100(sample: pd.Series) -> bool:
        try:
            nums = pd.to_numeric(sample, errors="coerce").dropna()
            if len(nums) < len(sample) * 0.8:
                return False
            return bool((nums >= 0).all() and (nums <= 100).all())
        except Exception:
            return False

    @staticmethod
    def _is_large_int(sample: pd.Series, norm_name: str) -> bool:
        hints = ("count", "pages", "pagecount", "пробег", "счетчик", "страниц")
        if not any(h in norm_name for h in hints):
            return False
        try:
            nums = pd.to_numeric(sample, errors="coerce").dropna()
            return bool(len(nums) >= len(sample) * 0.8 and (nums > 1000).any())
        except Exception:
            return False

    @staticmethod
    def _is_device_id(series: pd.Series) -> bool:
        if len(series) < MIN_ROWS_FOR_UNIQUENESS:
            return False
        uniqueness = series.nunique() / len(series)
        lengths = series.str.len()
        return bool(
            uniqueness > 0.9
            and lengths.min() >= 3
            and lengths.max() <= 20
        )


# ── LLMMatcher ───────────────────────────────────────────────────────────────


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM calls — allows testing without a real LLM."""

    def complete(self, prompt: str) -> str:
        """Send prompt, return text response."""
        ...


class LLMMatcher:
    """Step C: classify remaining columns via a single LLM call."""

    def __init__(
        self, client: LLMClient | None = None, prompt_path: Path = PROMPT_PATH
    ) -> None:
        self._client = client
        self._prompt_template = self._load_prompt(prompt_path)

    @staticmethod
    def _load_prompt(path: Path) -> str:
        if not path.exists():
            logger.warning("Промпт-файл не найден: %s", path)
            return ""
        return path.read_text(encoding="utf-8")

    def match(
        self,
        df: pd.DataFrame,
        already_mapped: dict[str, str],
        *,
        max_retries: int = 2,
    ) -> dict[str, str]:
        """Return additional mappings from LLM for unmapped columns."""
        if not self._client or not self._prompt_template:
            return {}

        unmapped = [c for c in df.columns if c not in already_mapped]
        if not unmapped:
            return {}

        columns_and_samples = self._format_samples(df, unmapped)
        prompt = self._prompt_template.replace("{columns_and_samples}", columns_and_samples)

        for attempt in range(1, max_retries + 1):
            try:
                raw = self._client.complete(prompt)
                response = self._parse_response(raw)
                return self._filter_valid(response.mapping, already_mapped)
            except Exception:
                logger.warning("LLM маппинг, попытка %d/%d не удалась", attempt, max_retries)

        return {}

    @staticmethod
    def _format_samples(df: pd.DataFrame, columns: list[str]) -> str:
        parts: list[str] = []
        for col in columns:
            vals = df[col].dropna().astype(str).head(SAMPLE_SIZE).tolist()
            samples_str = ", ".join(f'"{v}"' for v in vals)
            parts.append(f'Колонка: "{col}"\nПримеры значений: {samples_str}')
        return "\n\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> FieldMappingResponse:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)
        data = json.loads(text)
        return FieldMappingResponse.model_validate(data)

    @staticmethod
    def _filter_valid(
        mapping: dict[str, str], already_mapped: dict[str, str]
    ) -> dict[str, str]:
        used = set(already_mapped.values())
        result: dict[str, str] = {}
        for col, target in mapping.items():
            if target == "_ignore":
                continue
            if target in INTERNAL_FIELDS and target not in used:
                result[col] = target
                used.add(target)
        return result


# ── Profile management ───────────────────────────────────────────────────────


def save_profile(
    name: str,
    columns: list[str],
    mapping: dict[str, str],
    profiles_dir: Path = PROFILES_DIR,
) -> Path:
    """Save a mapping profile to YAML."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    sig = compute_signature(columns)
    data: dict[str, Any] = {
        "profile_name": name,
        "signature_hash": sig,
        "column_mapping": mapping,
    }
    path = profiles_dir / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Профиль сохранён: %s → %s", name, path)
    return path


def try_apply_profile(
    columns: list[str],
    profiles_dir: Path = PROFILES_DIR,
) -> tuple[dict[str, str], str] | None:
    """Find and apply a saved profile matching the column signature.

    Returns (mapping, profile_name) or None.
    """
    if not profiles_dir.exists():
        return None
    sig = compute_signature(columns)
    for path in profiles_dir.glob("*.yaml"):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            if data.get("signature_hash") == sig:
                mapping = data.get("column_mapping", {})
                profile_name = data.get("profile_name", path.stem)
                logger.info("Профиль найден: %s (hash=%s)", profile_name, sig[:12])
                return mapping, profile_name
        except Exception:
            continue
    return None


# ── FieldMapper coordinator ──────────────────────────────────────────────────


class FieldMapper:
    """Orchestrator: runs synonym → content → LLM steps, builds MappingResult."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        synonyms_path: Path = SYNONYMS_PATH,
        prompt_path: Path = PROMPT_PATH,
        profiles_dir: Path = PROFILES_DIR,
    ) -> None:
        self._synonym = SynonymMatcher(synonyms_path)
        self._content = ContentMatcher()
        self._llm = LLMMatcher(llm_client, prompt_path)
        self._profiles_dir = profiles_dir

    def map(self, df: pd.DataFrame) -> MappingResult:
        """Run the full mapping pipeline on a DataFrame."""
        columns = list(df.columns)

        profile_result = try_apply_profile(columns, self._profiles_dir)
        if profile_result:
            mapping, profile_name = profile_result
            details = [
                ColumnMapping(
                    source_column=col,
                    target_field=mapping.get(col),
                    confidence=ConfidenceLevel.PROFILE,
                    samples=df[col].dropna().astype(str).head(SAMPLE_SIZE).tolist(),
                )
                for col in columns
            ]
            unmapped = [col for col in columns if col not in mapping]
            return MappingResult(
                auto_mapping=mapping,
                column_details=details,
                unmapped=unmapped,
                profile_applied=profile_name,
                needs_confirmation=True,
            )

        combined: dict[str, str] = {}
        confidence_map: dict[str, ConfidenceLevel] = {}

        syn = self._synonym.match(columns)
        for col, target in syn.items():
            combined[col] = target
            confidence_map[col] = ConfidenceLevel.SYNONYM

        content = self._content.match(df, combined)
        for col, target in content.items():
            combined[col] = target
            confidence_map[col] = ConfidenceLevel.CONTENT

        llm = self._llm.match(df, combined)
        for col, target in llm.items():
            combined[col] = target
            confidence_map[col] = ConfidenceLevel.LLM

        details = []
        unmapped = []
        for col in columns:
            target = combined.get(col)  # type: ignore[assignment]
            conf = confidence_map.get(col, ConfidenceLevel.UNKNOWN)
            samples = df[col].dropna().astype(str).head(SAMPLE_SIZE).tolist()
            details.append(
                ColumnMapping(
                    source_column=col,
                    target_field=target,
                    confidence=conf,
                    samples=samples,
                )
            )
            if target is None:
                unmapped.append(col)

        logger.info(
            "Маппинг: %d/%d колонок опознано (synonym=%d, content=%d, llm=%d)",
            len(combined), len(columns),
            sum(1 for v in confidence_map.values() if v == ConfidenceLevel.SYNONYM),
            sum(1 for v in confidence_map.values() if v == ConfidenceLevel.CONTENT),
            sum(1 for v in confidence_map.values() if v == ConfidenceLevel.LLM),
        )

        return MappingResult(
            auto_mapping=combined,
            column_details=details,
            unmapped=unmapped,
        )
