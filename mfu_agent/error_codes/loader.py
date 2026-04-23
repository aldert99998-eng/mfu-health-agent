"""Loader + slug helpers for per-model error code dictionaries.

Storage layout:
  configs/error_codes/{vendor_slug}/{model_slug}.yaml
  configs/error_codes/_trash/            ← moved-out files on delete
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import yaml

from data_io.normalizer import canonicalize_model, load_model_aliases

from .schema import SUPPORTED_VENDORS, ErrorCode, ModelErrorCodes

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ERROR_CODES_ROOT = _CONFIGS_DIR / "error_codes"
TRASH_DIR = ERROR_CODES_ROOT / "_trash"

_cache: dict[tuple[str, str], ModelErrorCodes] = {}
_cache_lock = threading.Lock()

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def vendor_slug(vendor: str) -> str:
    """Normalize vendor name to a directory slug."""
    s = (vendor or "").strip().lower()
    s = _SLUG_RE.sub("_", s).strip("_")
    return s or "unknown"


def model_slug(model: str) -> str:
    """Normalize model name to a file slug. Applies canonicalization first.

    Strips known vendor prefixes from canonical name so the slug contains
    only the model identifier (e.g. "Xerox AltaLink B8090" → "altalink_b8090").
    """
    raw = (model or "").strip()
    if not raw:
        return "unknown"
    try:
        aliases = load_model_aliases()
        canonical = canonicalize_model(raw, aliases)
    except Exception:
        canonical = raw
    s = canonical.lower()
    for prefix in ("xerox ", "lexmark ", "ricoh ", "kyocera ", "hp ", "canon "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = _SLUG_RE.sub("_", s).strip("_")
    return s or "unknown"


def _file_path(vendor: str, model: str) -> Path:
    return ERROR_CODES_ROOT / vendor_slug(vendor) / f"{model_slug(model)}.yaml"


def models_sharing_slug(slug: str) -> list[str]:
    """Return all human-readable model names that resolve to given slug.

    Reverse lookup over model_aliases.yaml. Useful for showing "Applies to
    models: X, Y, Z" in UI when one file serves a whole product series.
    """
    try:
        aliases = load_model_aliases()
    except Exception:
        return []

    # First, find the canonical model whose slug matches the target slug
    canonicals: set[str] = set()
    for _alias, canonical in aliases.items():
        if model_slug(canonical) == slug:
            canonicals.add(canonical)

    if not canonicals:
        return []

    names: set[str] = set()
    # Add canonical names themselves
    names.update(canonicals)
    # Add all aliases whose canonical is in our set
    for alias, canonical in aliases.items():
        if canonical in canonicals:
            names.add(alias)
    # Return short unique model identifiers (drop vendor-prefix duplicates)
    cleaned: set[str] = set()
    for n in names:
        s = n.strip()
        if not s:
            continue
        # Drop pure-numeric aliases like "8045" — not useful as model name.
        if not any(c.isalpha() for c in s):
            continue
        cleaned.add(s)
    # Canonical first, then by length ascending, then alphabetic
    return sorted(
        cleaned,
        key=lambda x: (0 if x in canonicals else 1, len(x), x.lower()),
    )


def invalidate_cache(vendor: str | None = None, model: str | None = None) -> None:
    """Invalidate cache: a single pair, a whole vendor, or everything.

    Also clears downstream severity caches (classify tool + YAML maps) so
    the next health-index calculation sees the new values.
    """
    with _cache_lock:
        if vendor is None:
            _cache.clear()
        else:
            v_slug = vendor_slug(vendor)
            if model is None:
                to_del = [k for k in _cache if vendor_slug(k[0]) == v_slug]
            else:
                m_slug = model_slug(model)
                to_del = [
                    k for k in _cache
                    if vendor_slug(k[0]) == v_slug and model_slug(k[1]) == m_slug
                ]
            for k in to_del:
                _cache.pop(k, None)

    # Late import: agent.tools.severity_cache transitively imports state/UI
    # modules, which in turn depend on error_codes — direct import is cyclic.
    try:
        from agent.tools.severity_cache import invalidate_severity_caches
    except Exception:
        return
    invalidate_severity_caches()


def _read_yaml(path: Path) -> ModelErrorCodes | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception:
        return None
    codes_raw = data.get("codes") or {}
    codes: dict[str, ErrorCode] = {}
    for code, info in codes_raw.items():
        if not isinstance(info, dict):
            continue
        try:
            codes[str(code)] = ErrorCode(**info)
        except Exception:
            continue
    try:
        return ModelErrorCodes(
            vendor=data.get("vendor") or path.parent.name,
            model=data.get("model") or path.stem,
            updated_at=data.get("updated_at") or None,  # pydantic will default
            codes=codes,
        )
    except Exception:
        return None


def load_codes(vendor: str, model: str) -> ModelErrorCodes | None:
    """Load dictionary for (vendor, model). Returns None if no file exists."""
    if not vendor or not model:
        return None
    key = (vendor_slug(vendor), model_slug(model))
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    path = _file_path(vendor, model)
    doc = _read_yaml(path)
    if doc is None:
        return None
    with _cache_lock:
        _cache[key] = doc
    return doc


def list_vendors() -> list[str]:
    """Return canonical vendor names that have any model files."""
    if not ERROR_CODES_ROOT.exists():
        return []
    present: set[str] = set()
    slug_to_canonical = {vendor_slug(v): v for v in SUPPORTED_VENDORS}
    for p in ERROR_CODES_ROOT.iterdir():
        if not p.is_dir() or p.name.startswith("_"):
            continue
        canonical = slug_to_canonical.get(p.name, p.name.capitalize())
        # only include if it has at least one .yaml file
        if any(child.suffix == ".yaml" for child in p.iterdir()):
            present.add(canonical)
    return sorted(present)


def list_models(vendor: str) -> list[str]:
    """Return model slugs (file stems) that have dictionary for given vendor."""
    v_dir = ERROR_CODES_ROOT / vendor_slug(vendor)
    if not v_dir.exists():
        return []
    return sorted(p.stem for p in v_dir.iterdir() if p.suffix == ".yaml")
