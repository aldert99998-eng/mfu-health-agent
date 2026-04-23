"""Writer: save ModelErrorCodes to disk with backup on delete."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .loader import (
    ERROR_CODES_ROOT,
    TRASH_DIR,
    _file_path,
    invalidate_cache,
    vendor_slug,
    model_slug,
)
from .schema import ModelErrorCodes


def save(model_codes: ModelErrorCodes) -> Path:
    """Persist dictionary to configs/error_codes/{vendor}/{model}.yaml.

    Sets updated_at=now and invalidates the cache for this (vendor, model).
    """
    model_codes.updated_at = datetime.now(UTC)
    path = _file_path(model_codes.vendor, model_codes.model)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "vendor": model_codes.vendor,
        "model": model_codes.model,
        "updated_at": model_codes.updated_at.isoformat(),
        "codes": {
            code: {
                "description": info.description,
                "severity": info.severity,
                "component": info.component,
                "notes": info.notes,
            }
            for code, info in model_codes.codes.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    invalidate_cache(model_codes.vendor, model_codes.model)
    return path


def delete(vendor: str, model: str) -> Path | None:
    """Move file to _trash/ with timestamp. Returns new trash path or None."""
    src = _file_path(vendor, model)
    if not src.exists():
        invalidate_cache(vendor, model)
        return None
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    dest = TRASH_DIR / f"{vendor_slug(vendor)}_{model_slug(model)}_{ts}.yaml"
    shutil.move(str(src), str(dest))
    invalidate_cache(vendor, model)
    return dest
