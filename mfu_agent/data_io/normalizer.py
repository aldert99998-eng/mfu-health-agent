"""Data normalizer — Track E, Level 3.

Transforms mapped DataFrame rows into NormalizedEvent / ResourceSnapshot
objects with validation, timestamp parsing, error code normalization,
model canonicalization, and resource unit auto-detection.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import ValidationError

from data_io.factor_store import DeviceMetadata, FactorStore, FleetMeta
from data_io.field_mapper import FieldMapper, LLMClient
from data_io.models import (
    FileFormat,
    IngestionResult,
    InvalidRecord,
    NormalizedEvent,
    ResourceSnapshot,
    SourceFileInfo,
)
from data_io.parsers import parse_file
from data_io.preamble import has_sql_preamble_in_columns, strip_sql_preamble_text
from data_io.zabbix_transform import is_zabbix_long_format, transform_zabbix

logger = logging.getLogger(__name__)

MODEL_ALIASES_PATH = Path(__file__).resolve().parent.parent / "configs" / "model_aliases.yaml"

FALLBACK_FORMATS = [
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
]

UNIX_TS_THRESHOLD = 1_000_000_000

ERROR_CODE_RE = re.compile(r"^(?:[CJEF]\d{3,5}|S\d{3,5}|SC\d{3,4}|\d{2}-\d{3}-\d{2})$")
XEROX_CODE_RE = re.compile(r"^\d{2}-\d{3}-\d{2}$")
NUMERIC_DASH_CODE_RE = re.compile(r"\b\d{2}-\d{3}-\d{2}\b")
ERROR_PREFIX_RE = re.compile(
    r"^(?:error|err|код|ошибка|code)[\s:_\-]*",
    re.IGNORECASE,
)

RESOURCE_FIELDS = frozenset({"toner_level", "drum_level", "fuser_level", "mileage"})
EVENT_INDICATOR = "error_code"


# ── Normalization result ────────────────────────────────────────────────────


class NormalizationStats:
    """Counters collected during normalization."""

    def __init__(self) -> None:
        self.total_rows: int = 0
        self.valid_events: int = 0
        self.valid_snapshots: int = 0
        self.invalid_count: int = 0
        self.timestamp_fallback_count: int = 0
        self.unix_ts_count: int = 0
        self.error_code_normalized: int = 0
        self.model_canonicalized: int = 0
        self.unit_fraction_converted: int = 0
        self.unit_raw_flagged: int = 0


class NormalizationResult:
    """Output of Normalizer.normalize()."""

    def __init__(
        self,
        valid_events: list[NormalizedEvent],
        valid_resources: dict[str, ResourceSnapshot],
        invalid_records: list[InvalidRecord],
        stats: NormalizationStats,
    ) -> None:
        self.valid_events = valid_events
        self.valid_resources = valid_resources
        self.invalid_records = invalid_records
        self.stats = stats

    @property
    def success(self) -> bool:
        return len(self.valid_events) > 0 or len(self.valid_resources) > 0


# ── Helper functions ────────────────────────────────────────────────────────


def parse_timestamp(value: Any) -> datetime:
    """Parse a timestamp value using dateutil, fallback formats, and Unix detection."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    s = str(value).strip()
    if not s:
        raise ValueError("пустое значение timestamp")

    if _looks_like_unix_ts(s):
        ts = float(s)
        return datetime.fromtimestamp(ts, tz=UTC)

    for fmt in FALLBACK_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue

    from dateutil import parser as dateutil_parser

    dt = dateutil_parser.parse(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _looks_like_unix_ts(s: str) -> bool:
    try:
        val = float(s)
        return val > UNIX_TS_THRESHOLD
    except (ValueError, OverflowError):
        return False


def normalize_error_code(raw: str) -> str | None:
    """Normalize an error code: strip prefixes, validate pattern.

    Supports both letter-prefixed codes (C6000, SC543) and
    Xerox numeric codes (75-530-00, 07-535-00).
    """
    s = raw.strip()
    if not s:
        return None

    # Xerox XX-YYY-ZZ: extract before stripping hyphens
    m = NUMERIC_DASH_CODE_RE.search(s)
    if m:
        return m.group(0)

    s = re.sub(r"\(.*?\)", "", s).strip()
    s = ERROR_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"[\s\-]+", "", s)
    s = s.upper()

    if ERROR_CODE_RE.match(s):
        return s
    return None


def load_model_aliases(path: Path = MODEL_ALIASES_PATH) -> dict[str, str]:
    """Load model alias lookup: {normalized_alias: canonical_name}."""
    if not path.exists():
        logger.warning("Файл алиасов моделей не найден: %s", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    lookup: dict[str, str] = {}
    for canonical, aliases in data.items():
        canon_lower = str(canonical).strip().lower()
        lookup[canon_lower] = str(canonical)
        if isinstance(aliases, list):
            for alias in aliases:
                lookup[str(alias).strip().lower()] = str(canonical)
    return lookup


_TRAILING_SERIAL_RE = re.compile(r"\s+[A-Z0-9]{6,}$")


def canonicalize_model(raw: str, aliases: dict[str, str]) -> str:
    """Return canonical model name or stripped original."""
    s = raw.strip()
    if not s:
        return s
    key = s.lower()
    if key in aliases:
        return aliases[key]
    cleaned = _TRAILING_SERIAL_RE.sub("", s)
    key2 = cleaned.lower()
    return aliases.get(key2, cleaned)


def detect_resource_unit(values: list[float]) -> tuple[list[float], bool]:
    """Auto-detect resource unit: percent [0,100], fraction [0,1], or raw.

    Returns (converted_values, unit_raw_flag).
    """
    if not values:
        return values, False

    max_val = max(values)
    min_val = min(values)

    if min_val >= 0 and max_val <= 100:
        return values, False
    if min_val >= 0 and max_val <= 1.0:
        return [v * 100 for v in values], False
    return values, True


def _safe_int(val: Any) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return round(f)
    except (ValueError, TypeError, OverflowError):
        return None


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (ValueError, TypeError, OverflowError):
        return None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_format(path: Path) -> FileFormat:
    ext = path.suffix.lower()
    fmt_map = {
        ".csv": FileFormat.CSV,
        ".tsv": FileFormat.TSV,
        ".json": FileFormat.JSON,
        ".jsonl": FileFormat.JSONL,
        ".ndjson": FileFormat.JSONL,
        ".xlsx": FileFormat.XLSX,
    }
    return fmt_map.get(ext, FileFormat.CSV)


# ── Normalizer ──────────────────────────────────────────────────────────────


class Normalizer:
    """Transform mapped DataFrame into validated events and resource snapshots."""

    def __init__(
        self,
        model_aliases_path: Path = MODEL_ALIASES_PATH,
    ) -> None:
        self._aliases = load_model_aliases(model_aliases_path)

    def normalize(
        self,
        df: pd.DataFrame,
        mapping: dict[str, str],
        resource_unit_hints: dict[str, str] | None = None,
    ) -> NormalizationResult:
        """Run full normalization pipeline on a mapped DataFrame."""
        stats = NormalizationStats()
        stats.total_rows = len(df)
        events: list[NormalizedEvent] = []
        snapshots_by_device: dict[str, list[ResourceSnapshot]] = defaultdict(list)
        invalid: list[InvalidRecord] = []

        reverse_map = {v: k for k, v in mapping.items()}

        resource_columns_present = any(
            field in reverse_map for field in RESOURCE_FIELDS
        )

        level_values = self._collect_level_values(df, reverse_map)
        unit_info = self._detect_units(level_values, resource_unit_hints)

        for idx, row in df.iterrows():
            row_num = int(idx) + 2  # type: ignore[call-overload]
            raw_data = {str(k): str(v) for k, v in row.to_dict().items() if pd.notna(v)}

            try:
                parsed = self._parse_row(row, reverse_map, unit_info, stats)
            except Exception as exc:
                invalid.append(InvalidRecord(
                    row_number=row_num,
                    raw_data=raw_data,
                    reason=str(exc),
                ))
                stats.invalid_count += 1
                continue

            device_id = parsed.get("device_id")
            timestamp = parsed.get("timestamp")

            if not device_id:
                invalid.append(InvalidRecord(
                    row_number=row_num,
                    raw_data=raw_data,
                    reason="отсутствует device_id",
                    field="device_id",
                ))
                stats.invalid_count += 1
                continue

            if not timestamp:
                invalid.append(InvalidRecord(
                    row_number=row_num,
                    raw_data=raw_data,
                    reason="отсутствует или некорректный timestamp",
                    field="timestamp",
                ))
                stats.invalid_count += 1
                continue

            has_error = parsed.get("error_code") is not None
            has_error_desc = parsed.get("error_description") is not None
            has_error_code_col = "error_code" in reverse_map
            has_resources = any(
                parsed.get(f) is not None for f in RESOURCE_FIELDS
            )

            if has_error_code_col:
                is_event = has_error or has_error_desc
            else:
                is_event = has_error

            if is_event:
                event = self._build_event(parsed, row_num, invalid, raw_data, stats)
                if event:
                    events.append(event)

            if has_resources:
                snap = self._build_snapshot(parsed, row_num, invalid, raw_data, stats)
                if snap:
                    snapshots_by_device[device_id].append(snap)

            if not is_event and not has_resources:
                if resource_columns_present:
                    snap = self._build_snapshot(parsed, row_num, invalid, raw_data, stats)
                    if snap:
                        snapshots_by_device[device_id].append(snap)
                elif has_error_code_col:
                    event = self._build_event(parsed, row_num, invalid, raw_data, stats)
                    if event:
                        events.append(event)

        latest_resources = self._pick_latest_snapshots(snapshots_by_device)

        logger.info(
            "Нормализация: %d событий, %d снапшотов, %d невалидных из %d строк",
            len(events), len(latest_resources), len(invalid), stats.total_rows,
        )

        return NormalizationResult(
            valid_events=events,
            valid_resources=latest_resources,
            invalid_records=invalid,
            stats=stats,
        )

    def _collect_level_values(
        self,
        df: pd.DataFrame,
        reverse_map: dict[str, str],
    ) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for field in ("toner_level", "drum_level", "fuser_level"):
            col = reverse_map.get(field)
            if col is None or col not in df.columns:
                continue
            vals: list[float] = []
            for v in df[col].dropna():
                f = _safe_float(v)
                if f is not None:
                    vals.append(f)
            if vals:
                result[field] = vals
        return result

    def _detect_units(
        self,
        level_values: dict[str, list[float]],
        hints: dict[str, str] | None,
    ) -> dict[str, tuple[float, bool]]:
        """Return {field: (multiplier, unit_raw)} for resource fields."""
        info: dict[str, tuple[float, bool]] = {}
        hints = hints or {}
        for field, vals in level_values.items():
            hint = hints.get(field)
            if hint == "percent":
                info[field] = (1.0, False)
            elif hint == "fraction":
                info[field] = (100.0, False)
            elif hint == "raw":
                info[field] = (1.0, True)
            else:
                _, unit_raw = detect_resource_unit(vals)
                max_val = max(vals) if vals else 0
                if not unit_raw and max_val <= 1.0 and max_val > 0:
                    info[field] = (100.0, False)
                else:
                    info[field] = (1.0, unit_raw)
        return info

    def _parse_row(
        self,
        row: pd.Series,
        reverse_map: dict[str, str],
        unit_info: dict[str, tuple[float, bool]],
        stats: NormalizationStats,
    ) -> dict[str, Any]:
        parsed: dict[str, Any] = {}

        dev_col = reverse_map.get("device_id")
        if dev_col and pd.notna(row.get(dev_col)):
            parsed["device_id"] = str(row[dev_col]).strip()

        ts_col = reverse_map.get("timestamp")
        if ts_col and pd.notna(row.get(ts_col)):
            raw_ts = row[ts_col]
            try:
                parsed["timestamp"] = parse_timestamp(raw_ts)
                raw_s = str(raw_ts).strip()
                if _looks_like_unix_ts(raw_s):
                    stats.unix_ts_count += 1
            except (ValueError, OverflowError, TypeError) as exc:
                raise ValueError(
                    f"timestamp (колонка '{ts_col}'): {exc}"
                ) from exc

        ec_col = reverse_map.get("error_code")
        if ec_col and pd.notna(row.get(ec_col)):
            raw_code = str(row[ec_col]).strip()
            if raw_code:
                normalized = normalize_error_code(raw_code)
                if normalized:
                    parsed["error_code"] = normalized
                    stats.error_code_normalized += 1
                else:
                    parsed.setdefault("error_description", raw_code)

        for str_field in ("error_description", "location", "status"):
            col = reverse_map.get(str_field)
            if col and pd.notna(row.get(col)):
                parsed[str_field] = str(row[col]).strip()

        if "error_code" not in parsed and parsed.get("error_description"):
            extracted = normalize_error_code(parsed["error_description"])
            if extracted:
                parsed["error_code"] = extracted
                stats.error_code_normalized += 1

        model_col = reverse_map.get("model")
        if model_col and pd.notna(row.get(model_col)):
            raw_model = str(row[model_col]).strip()
            canonical = canonicalize_model(raw_model, self._aliases)
            parsed["model"] = canonical
            if canonical != raw_model:
                stats.model_canonicalized += 1

        vendor_col = reverse_map.get("vendor")
        if vendor_col and pd.notna(row.get(vendor_col)):
            parsed["vendor"] = str(row[vendor_col]).strip()

        for field in ("toner_level", "drum_level", "fuser_level"):
            col = reverse_map.get(field)
            if col and pd.notna(row.get(col)):
                raw_val = _safe_float(row[col])
                if raw_val is not None:
                    multiplier, unit_raw = unit_info.get(field, (1.0, False))
                    converted = raw_val * multiplier
                    clamped = max(0, min(100, round(converted)))
                    parsed[field] = clamped
                    if unit_raw:
                        stats.unit_raw_flagged += 1
                    if multiplier != 1.0:
                        stats.unit_fraction_converted += 1
                    parsed.setdefault("_unit_raw", unit_raw)

        mileage_col = reverse_map.get("mileage")
        if mileage_col and pd.notna(row.get(mileage_col)):
            parsed["mileage"] = _safe_int(row[mileage_col])

        return parsed

    def _build_event(
        self,
        parsed: dict[str, Any],
        row_num: int,
        invalid: list[InvalidRecord],
        raw_data: dict[str, str],
        stats: NormalizationStats,
    ) -> NormalizedEvent | None:
        try:
            event = NormalizedEvent(
                device_id=parsed["device_id"],
                timestamp=parsed["timestamp"],
                error_code=parsed.get("error_code"),
                error_description=parsed.get("error_description"),
                model=parsed.get("model"),
                vendor=parsed.get("vendor"),
                location=parsed.get("location"),
                status=parsed.get("status"),
            )
            stats.valid_events += 1
            return event
        except ValidationError as exc:
            invalid.append(InvalidRecord(
                row_number=row_num,
                raw_data=raw_data,
                reason=f"валидация NormalizedEvent: {exc.error_count()} ошибок",
            ))
            stats.invalid_count += 1
            return None

    def _build_snapshot(
        self,
        parsed: dict[str, Any],
        row_num: int,
        invalid: list[InvalidRecord],
        raw_data: dict[str, str],
        stats: NormalizationStats,
    ) -> ResourceSnapshot | None:
        try:
            snap = ResourceSnapshot(
                device_id=parsed["device_id"],
                timestamp=parsed["timestamp"],
                toner_level=parsed.get("toner_level"),
                drum_level=parsed.get("drum_level"),
                fuser_level=parsed.get("fuser_level"),
                mileage=parsed.get("mileage"),
                unit_raw=parsed.get("_unit_raw", False),
            )
            stats.valid_snapshots += 1
            return snap
        except ValidationError as exc:
            invalid.append(InvalidRecord(
                row_number=row_num,
                raw_data=raw_data,
                reason=f"валидация ResourceSnapshot: {exc.error_count()} ошибок",
            ))
            stats.invalid_count += 1
            return None

    @staticmethod
    def _pick_latest_snapshots(
        by_device: dict[str, list[ResourceSnapshot]],
    ) -> dict[str, ResourceSnapshot]:
        result: dict[str, ResourceSnapshot] = {}
        for device_id, snaps in by_device.items():
            if snaps:
                result[device_id] = max(snaps, key=lambda s: s.timestamp)
        return result


# ── Integration function ────────────────────────────────────────────────────


def _extract_metadata_from_df(
    df: pd.DataFrame,
    reverse_map: dict[str, str],
    device_meta_map: dict[str, dict[str, str | None]],
) -> None:
    """Extract per-device metadata (model, vendor, location) directly from the DataFrame."""
    dev_col = reverse_map.get("device_id")
    if not dev_col or dev_col not in df.columns:
        return
    model_col = reverse_map.get("model")
    vendor_col = reverse_map.get("vendor")
    location_col = reverse_map.get("location")

    for device_id, grp in df.groupby(dev_col):
        did = str(device_id).strip()
        if did in device_meta_map:
            continue
        model = _first_notnull(grp, model_col)
        vendor = _first_notnull(grp, vendor_col)
        location = _first_notnull(grp, location_col)
        device_meta_map[did] = {"model": model, "vendor": vendor, "location": location}


def _has_sql_preamble(df: pd.DataFrame) -> bool:
    """Check if DataFrame column names contain SQL query text."""
    return has_sql_preamble_in_columns(df.columns)


def _strip_sql_lines(file_path: Path) -> Path | None:
    """Read file, strip SQL/comment preamble lines, write cleaned temp file.

    Returns path to cleaned file, or None if no preamble found.
    """
    raw = file_path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None

    cleaned, skipped = strip_sql_preamble_text(text)
    if skipped == 0:
        return None

    logger.info("SQL-преамбула: пропущено %d строк, перепарсинг файла", skipped)
    clean_path = file_path.with_suffix(".cleaned" + file_path.suffix)
    clean_path.write_text(cleaned, encoding="utf-8")
    return clean_path


_META_EMPTY = frozenset({"nan", "none", "null", "na", ""})


def _first_notnull(df: pd.DataFrame, col: str | None) -> str | None:
    if not col or col not in df.columns:
        return None
    for v in df[col].dropna():
        s = str(v).strip()
        if s.lower() not in _META_EMPTY:
            return s
    return None


MAX_FILE_SIZE_MB = 200


def ingest_file(
    file_path: Path,
    factor_store: FactorStore,
    *,
    llm_client: LLMClient | None = None,
    resource_unit_hints: dict[str, str] | None = None,
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
) -> IngestionResult:
    """Full ingestion pipeline: parse → map → normalize → fill store."""
    start = time.monotonic()
    warnings_list: list[str] = []
    errors_list: list[str] = []

    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_file_size_mb:
            return IngestionResult(
                success=False,
                source_file_info=SourceFileInfo(
                    file_name=file_path.name,
                    file_hash="",
                    file_size_bytes=file_path.stat().st_size,
                    file_format=_detect_format(file_path),
                    uploaded_at=datetime.now(UTC),
                ),
                errors=[
                    f"Файл слишком большой: {size_mb:.1f} МБ "
                    f"(максимум {max_file_size_mb:.0f} МБ)"
                ],
                ingestion_duration_seconds=time.monotonic() - start,
            )

    try:
        df = parse_file(file_path)
    except Exception as exc:
        return IngestionResult(
            success=False,
            source_file_info=SourceFileInfo(
                file_name=file_path.name,
                file_hash="",
                file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                file_format=_detect_format(file_path),
                uploaded_at=datetime.now(UTC),
            ),
            errors=[f"Ошибка парсинга: {exc}"],
            ingestion_duration_seconds=time.monotonic() - start,
        )

    if _has_sql_preamble(df):
        clean_path = _strip_sql_lines(file_path)
        if clean_path:
            try:
                df = parse_file(clean_path)
            finally:
                clean_path.unlink(missing_ok=True)
            warnings_list.append("SQL-преамбула пропущена автоматически")

    if is_zabbix_long_format(df):
        logger.info("Обнаружен формат Zabbix (long), трансформация...")
        warnings_list.append("Обнаружен формат Zabbix — данные преобразованы из длинного формата")
        df = transform_zabbix(df)

    mapper = FieldMapper(llm_client=llm_client)
    mapping_result = mapper.map(df)
    mapping = mapping_result.auto_mapping

    if mapping_result.unmapped:
        warnings_list.append(
            f"Нераспознанные колонки: {', '.join(mapping_result.unmapped)}"
        )

    if "device_id" not in mapping.values():
        cols_list = ", ".join(f"«{c}»" for c in df.columns)
        mapped_info = (
            "; ".join(f"«{c}» → {t}" for c, t in mapping.items())
            if mapping
            else "ни одна колонка не распознана"
        )
        errors_list.append(
            f"Не удалось определить колонку device_id.\n"
            f"Колонки в файле: {cols_list}.\n"
            f"Распознано: {mapped_info}.\n"
            f"Переименуйте колонку с ID устройства в «device_id», "
            f"«serial_number» или «id»."
        )
        return IngestionResult(
            success=False,
            source_file_info=SourceFileInfo(
                file_name=file_path.name,
                file_hash=_file_hash(file_path),
                file_size_bytes=file_path.stat().st_size,
                file_format=_detect_format(file_path),
                uploaded_at=datetime.now(UTC),
            ),
            mapping_used=mapping,
            profile_applied=mapping_result.profile_applied,
            total_records=len(df),
            errors=errors_list,
            ingestion_duration_seconds=time.monotonic() - start,
        )

    normalizer = Normalizer()
    norm_result = normalizer.normalize(df, mapping, resource_unit_hints)

    events_by_device: dict[str, list[NormalizedEvent]] = defaultdict(list)
    device_meta_map: dict[str, dict[str, str | None]] = {}

    reverse_map = {v: k for k, v in mapping.items()}
    _extract_metadata_from_df(df, reverse_map, device_meta_map)

    for event in norm_result.valid_events:
        events_by_device[event.device_id].append(event)

    for device_id, device_events in events_by_device.items():
        factor_store.add_events(device_id, device_events)

    for device_id, snap in norm_result.valid_resources.items():
        factor_store.set_resources(device_id, snap)
        device_meta_map.setdefault(device_id, {"model": None, "vendor": None, "location": None})

    for device_id, meta_info in device_meta_map.items():
        factor_store.set_device_metadata(device_id, DeviceMetadata(
            device_id=device_id,
            model=meta_info.get("model"),
            vendor=meta_info.get("vendor"),
            location=meta_info.get("location"),
        ))

    fhash = _file_hash(file_path)
    factor_store.set_fleet_meta(FleetMeta(
        file_hash=fhash,
        upload_timestamp=datetime.now(UTC),
        mapping_profile=mapping_result.profile_applied,
        source_filename=file_path.name,
        total_records=len(df),
    ))

    all_timestamps: list[datetime] = []
    for ev in norm_result.valid_events:
        all_timestamps.append(ev.timestamp)
    for snap in norm_result.valid_resources.values():
        all_timestamps.append(snap.timestamp)

    date_range = None
    if all_timestamps:
        date_range = (min(all_timestamps), max(all_timestamps))
        factor_store.set_reference_time(max(all_timestamps))

    all_device_ids = set(events_by_device.keys()) | set(norm_result.valid_resources.keys())

    if norm_result.stats.invalid_count > 0:
        warnings_list.append(
            f"Невалидных записей: {norm_result.stats.invalid_count}"
        )

    duration = time.monotonic() - start

    data_checksum = hashlib.sha256(
        f"{len(df)}:{norm_result.stats.valid_events}:{len(norm_result.valid_resources)}"
        f":{norm_result.stats.invalid_count}:{fhash}".encode()
    ).hexdigest()[:16]

    return IngestionResult(
        success=True,
        source_file_info=SourceFileInfo(
            file_name=file_path.name,
            file_hash=fhash,
            file_size_bytes=file_path.stat().st_size,
            file_format=_detect_format(file_path),
            uploaded_at=datetime.now(UTC),
        ),
        mapping_used=mapping,
        profile_applied=mapping_result.profile_applied,
        total_records=len(df),
        valid_events_count=norm_result.stats.valid_events,
        valid_snapshots_count=len(norm_result.valid_resources),
        invalid_records=norm_result.invalid_records,
        devices_count=len(all_device_ids),
        date_range=date_range,
        warnings=warnings_list,
        errors=errors_list,
        ingestion_duration_seconds=duration,
        data_checksum=data_checksum,
    )
