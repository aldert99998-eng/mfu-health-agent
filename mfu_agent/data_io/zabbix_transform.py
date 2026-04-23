"""Zabbix long-format → wide-format transform.

Zabbix exports monitoring data as one row per metric reading (item_key + metric_value).
This module detects that format and pivots it into the wide format
expected by the normalizer: one row per event/snapshot with device_id,
timestamp, error_description, toner_level, drum_level, etc.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

_ZABBIX_MARKERS = {"item_key", "metric_value"}

_TONER_HINTS = ("toner",)
_DRUM_HINTS = ("drum", "print cartridge", "принт-картридж", "барабан")
_FUSER_HINTS = ("fuser", "фьюзер")
_MAINT_HINTS = ("maintenance", "transfer roller", "обслуживан")

_ERROR_PREFIXES = ("errdisp", "errorondisplay")


def is_zabbix_long_format(df: pd.DataFrame) -> bool:
    lower_cols = {c.lower().strip() for c in df.columns}
    return _ZABBIX_MARKERS.issubset(lower_cols)


def transform_zabbix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot Zabbix long-format data into wide-format rows."""
    col = _ColFinder(df)

    item_key_col = col("item_key")
    value_col = col("metric_value", "value")
    device_col = col("id_prn", "hostid", "host_name", "host")
    clock_col = col("clock", "clock_time", "timestamp", "ts")

    if not all([item_key_col, value_col, device_col, clock_col]):
        logger.warning(
            "Zabbix: не найдены обязательные колонки, пропуск трансформации. "
            "item_key=%s value=%s device=%s clock=%s. Доступные: %s",
            item_key_col, value_col, device_col, clock_col, list(df.columns),
        )
        return df

    inv_model = col("inventory_model")
    inv_type = col("inventory_type", "inventory_type_full")
    inv_vendor = col("inventory_vendor")
    inv_location = col("inventory_location")
    display_col = col("host_display_name")

    rows: list[dict[str, object]] = []

    for device_id, grp in df.groupby(device_col):
        meta = _extract_metadata(grp, item_key_col, value_col, inv_model, inv_type, inv_vendor, inv_location, display_col)
        supplies = _extract_supplies(grp, item_key_col, value_col, clock_col)
        mileage = _latest_numeric(grp, item_key_col, value_col, clock_col, "LifeCount")

        base: dict[str, object] = {
            "device_id": str(device_id),
            "model": meta.get("model"),
            "vendor": meta.get("vendor"),
            "location": meta.get("location"),
            "toner_level": supplies.get("toner"),
            "drum_level": supplies.get("drum"),
            "fuser_level": supplies.get("fuser"),
            "mileage": mileage,
        }

        err_rows = _error_rows(grp, item_key_col, value_col, clock_col)

        if err_rows:
            for ts, desc in err_rows:
                rows.append({**base, "timestamp": ts, "error_description": desc})
        else:
            rows.append({**base, "timestamp": str(grp[clock_col].max())})

    if not rows:
        return df

    result = pd.DataFrame(rows)
    n_devices = result["device_id"].nunique()
    logger.info(
        "Zabbix трансформация: %d строк → %d строк, %d устройств",
        len(df), len(result), n_devices,
    )
    return result


class _ColFinder:
    def __init__(self, df: pd.DataFrame) -> None:
        self._map = {c.lower().strip(): c for c in df.columns}

    def __call__(self, *candidates: str) -> str | None:
        for c in candidates:
            if c in self._map:
                return self._map[c]
        return None


_EMPTY_VALS = frozenset({"nan", "none", "null", "na", ""})


def _first_notempty(series: pd.Series) -> str | None:
    for v in series.dropna():
        s = str(v).strip()
        if s.lower() not in _EMPTY_VALS:
            return s
    return None


def _extract_metadata(
    grp: pd.DataFrame,
    item_key_col: str,
    value_col: str,
    inv_model: str | None,
    inv_type: str | None,
    inv_vendor: str | None,
    inv_location: str | None,
    display_col: str | None,
) -> dict[str, str | None]:
    model = _col_value(grp, inv_model) or _col_value(grp, inv_type) or _col_value(grp, display_col)
    vendor = _col_value(grp, inv_vendor)
    location = _col_value(grp, inv_location)
    return {"model": model, "vendor": vendor, "location": location}


def _col_value(grp: pd.DataFrame, col: str | None) -> str | None:
    if not col or col not in grp.columns:
        return None
    return _first_notempty(grp[col])


def _classify_supply(item_key: str) -> str | None:
    low = item_key.lower()
    for hint in _TONER_HINTS:
        if hint in low:
            return "toner"
    for hint in _DRUM_HINTS:
        if hint in low:
            return "drum"
    for hint in _FUSER_HINTS:
        if hint in low:
            return "fuser"
    return None


def _extract_supplies(
    grp: pd.DataFrame,
    item_key_col: str,
    value_col: str,
    clock_col: str,
) -> dict[str, float | None]:
    levels: dict[str, list[tuple[str, float]]] = {}
    maxes: dict[str, list[float]] = {}

    for _, row in grp.iterrows():
        key = str(row[item_key_col])
        val_str = str(row[value_col]).strip() if pd.notna(row[value_col]) else ""

        is_level = key.startswith(("SupplLev", "prtMarkerSuppliesLevel"))
        is_max = key.startswith(("SupplMax", "prtMarkerSuppliesMaxCapacity"))

        if not (is_level or is_max):
            continue

        category = _classify_supply(key)
        if category is None:
            continue

        try:
            num = float(val_str)
        except (ValueError, TypeError):
            continue

        if is_level:
            ts = str(row[clock_col])
            levels.setdefault(category, []).append((ts, num))
        else:
            maxes.setdefault(category, []).append(num)

    result: dict[str, float | None] = {}
    for cat in ("toner", "drum", "fuser"):
        cat_levels = levels.get(cat)
        cat_maxes = maxes.get(cat)
        if not cat_levels:
            result[cat] = None
            continue

        latest_val = max(cat_levels, key=lambda x: x[0])[1]
        max_val = max(cat_maxes) if cat_maxes else None

        if max_val and max_val > 0:
            if max_val == 100:
                result[cat] = min(latest_val, 100.0)
            else:
                result[cat] = round(latest_val / max_val * 100, 1)
        elif latest_val <= 100:
            result[cat] = latest_val
        else:
            result[cat] = None

    return result


def _latest_numeric(
    grp: pd.DataFrame,
    item_key_col: str,
    value_col: str,
    clock_col: str,
    key_match: str,
) -> str | None:
    mask = grp[item_key_col].astype(str) == key_match
    sub = grp[mask]
    if sub.empty:
        return None
    latest = sub.loc[sub[clock_col].idxmax()]
    val = str(latest[value_col]).strip() if pd.notna(latest[value_col]) else None
    return val


def _error_rows(
    grp: pd.DataFrame,
    item_key_col: str,
    value_col: str,
    clock_col: str,
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in grp.iterrows():
        key = str(row[item_key_col]).lower()
        if not any(key.startswith(p) for p in _ERROR_PREFIXES):
            continue

        ts = str(row[clock_col])
        desc = str(row[value_col]).strip() if pd.notna(row[value_col]) else ""
        if not desc:
            continue

        pair = (ts, desc)
        if pair not in seen:
            seen.add(pair)
            result.append(pair)

    return result
