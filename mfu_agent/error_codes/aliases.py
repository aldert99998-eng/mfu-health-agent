"""Safe editing of configs/model_aliases.yaml from UI: add / remove alias.

All mutations make a timestamped backup into configs/_backups/ first.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from data_io.normalizer import MODEL_ALIASES_PATH

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
BACKUPS_DIR = _CONFIGS_DIR / "_backups"


class AliasConflict(Exception):
    """Raised when a requested alias is already attached to a different canonical."""

    def __init__(self, alias: str, existing_canonical: str) -> None:
        super().__init__(
            f"Alias '{alias}' уже привязан к '{existing_canonical}'."
        )
        self.alias = alias
        self.existing_canonical = existing_canonical


def _load_raw() -> dict[str, list[str]]:
    if not MODEL_ALIASES_PATH.exists():
        return {}
    with MODEL_ALIASES_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for canonical, aliases in data.items():
        if isinstance(aliases, list):
            out[str(canonical)] = [str(a) for a in aliases]
        elif aliases is None:
            out[str(canonical)] = []
    return out


def _backup_then_write(data: dict[str, list[str]]) -> Path:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    backup_path = BACKUPS_DIR / f"model_aliases_{ts}.yaml"
    if MODEL_ALIASES_PATH.exists():
        shutil.copy2(MODEL_ALIASES_PATH, backup_path)
    with MODEL_ALIASES_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return backup_path


def get_canonical_for_alias(alias: str) -> str | None:
    """Return canonical name this alias belongs to (if any)."""
    needle = (alias or "").strip().lower()
    if not needle:
        return None
    data = _load_raw()
    for canonical, aliases in data.items():
        if needle == canonical.strip().lower():
            return canonical
        if any(needle == a.strip().lower() for a in aliases):
            return canonical
    return None


def add_alias(
    canonical: str,
    new_alias: str,
    *,
    force_reassign: bool = False,
) -> dict[str, str]:
    """Attach new_alias to given canonical model name.

    Returns a status dict:
      {"action": "noop"|"added"|"reassigned", "previous_canonical": str|None}

    Raises:
      ValueError — canonical not found, alias empty/invalid.
      AliasConflict — alias already attached to a different canonical and
                      force_reassign is False.
    """
    new_alias_norm = (new_alias or "").strip().lower()
    if not new_alias_norm:
        raise ValueError("Имя модели не может быть пустым.")
    if len(new_alias_norm) > 120:
        raise ValueError("Имя модели слишком длинное (>120 символов).")

    data = _load_raw()
    if canonical not in data:
        raise ValueError(f"Канонический ключ '{canonical}' не найден в aliases.yaml.")

    existing = get_canonical_for_alias(new_alias_norm)
    if existing is not None:
        if existing == canonical:
            # Already in place
            return {"action": "noop", "previous_canonical": existing}
        if not force_reassign:
            raise AliasConflict(new_alias_norm, existing)
        # Remove from old canonical
        data[existing] = [
            a for a in data.get(existing, [])
            if a.strip().lower() != new_alias_norm
        ]

    # Add to new canonical (if not present already)
    if new_alias_norm not in [a.strip().lower() for a in data[canonical]]:
        data[canonical].append(new_alias_norm)

    _backup_then_write(data)

    try:
        from .loader import invalidate_cache
        invalidate_cache()
    except Exception:
        pass

    return {
        "action": "reassigned" if existing else "added",
        "previous_canonical": existing,
    }


def remove_alias(alias: str) -> str | None:
    """Detach an alias. Returns canonical it was attached to (or None).

    Raises ValueError if caller tries to remove a canonical top-level key
    (which would destroy the whole registry entry).
    """
    alias_norm = (alias or "").strip().lower()
    if not alias_norm:
        raise ValueError("Имя alias не может быть пустым.")

    data = _load_raw()
    if alias_norm in {k.strip().lower() for k in data}:
        raise ValueError(
            f"'{alias}' — это каноническое имя, а не alias. "
            f"Удалять канонические ключи нельзя (сломается регистр)."
        )

    target_canonical: str | None = None
    for canonical, aliases in data.items():
        lowered = [a.strip().lower() for a in aliases]
        if alias_norm in lowered:
            target_canonical = canonical
            data[canonical] = [
                a for a in aliases
                if a.strip().lower() != alias_norm
            ]
            break

    if target_canonical is None:
        return None

    _backup_then_write(data)

    try:
        from .loader import invalidate_cache
        invalidate_cache()
    except Exception:
        pass

    return target_canonical


def list_aliases_for_canonical(canonical: str) -> list[str]:
    """Return all aliases attached to given canonical (sorted)."""
    data = _load_raw()
    return sorted(data.get(canonical, []))
