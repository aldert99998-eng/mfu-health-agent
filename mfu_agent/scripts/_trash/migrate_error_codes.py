"""One-shot migration: legacy xerox_*.yaml → per-model registry.

Reads:
  configs/xerox_display_codes.yaml  (52 display codes)
  configs/xerox_severity_map.yaml   (366 AltaLink B8090 service codes)

Writes:
  configs/error_codes/xerox/b8090.yaml   — merge of both files
  configs/error_codes/xerox/b8045.yaml   — display codes only (fleet fallback)
  configs/error_codes/xerox/workcentre.yaml — display codes only

Legacy files are NOT removed — they remain as backward-compat fallback.

Run:  python -m scripts.migrate_error_codes
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


SEVERITY_ALIASES = {"critical": "Critical", "major": "High", "minor": "Low"}


def _canonical_sev(raw: str) -> str:
    s = (raw or "").strip()
    lower = s.lower()
    if lower in SEVERITY_ALIASES:
        return SEVERITY_ALIASES[lower]
    if s in ("Critical", "High", "Medium", "Low", "Info"):
        return s
    return "Medium"


def _load_display_codes() -> dict[str, dict]:
    """From xerox_display_codes.yaml → {code: {description, severity, component}}."""
    data = _load_yaml(CONFIGS / "xerox_display_codes.yaml")
    result: dict[str, dict] = {}
    for code, info in (data.get("codes") or {}).items():
        if not isinstance(info, dict):
            continue
        result[str(code)] = {
            "description": str(info.get("description") or "").strip(),
            "severity": _canonical_sev(str(info.get("severity") or "")),
            "component": str(info.get("component") or "").strip(),
            "notes": "",
        }
    return result


def _load_service_codes() -> dict[str, dict]:
    data = _load_yaml(CONFIGS / "xerox_severity_map.yaml")
    result: dict[str, dict] = {}
    for code, info in (data.get("codes") or {}).items():
        if not isinstance(info, dict):
            continue
        result[str(code)] = {
            "description": str(info.get("description") or "").strip()
            or f"Xerox service code {code}",
            "severity": _canonical_sev(str(info.get("severity") or "")),
            "component": str(info.get("component") or "").strip(),
            "notes": "",
        }
    return result


def _dedup_missing_description(codes: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for code, info in codes.items():
        if not info.get("description"):
            continue
        out[code] = info
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from error_codes import ModelErrorCodes, ErrorCode  # noqa: E402
    from error_codes.writer import save  # noqa: E402

    display = _load_display_codes()
    service = _load_service_codes()

    # Note: model_aliases.yaml maps the whole AltaLink B80xx series
    # (B8045, B8065, B8090, B8145) onto a single canonical name.
    # Therefore we write a single file that covers the series.
    altalink_codes = {**display, **service}
    altalink_codes = _dedup_missing_description(altalink_codes)
    workcentre_codes = _dedup_missing_description(dict(display))

    targets = [
        ("AltaLink B8090", altalink_codes),
        ("WorkCentre 6515", workcentre_codes),
    ]

    total_written = 0
    seen_slugs: set[str] = set()
    from error_codes import model_slug  # noqa: E402
    for model_name, codes_dict in targets:
        if not codes_dict:
            print(f"  skip {model_name}: empty")
            continue
        slug = model_slug(model_name)
        if slug in seen_slugs:
            print(f"  skip {model_name}: same slug as previously written ({slug})")
            continue
        seen_slugs.add(slug)
        codes = {c: ErrorCode(**info) for c, info in codes_dict.items()}
        doc = ModelErrorCodes(vendor="Xerox", model=model_name, codes=codes)
        path = save(doc)
        print(f"  wrote {path.relative_to(ROOT)} ({len(codes)} codes)")
        total_written += 1

    print(f"\nDone. Files written: {total_written}")
    print("Legacy xerox_*.yaml kept as-is (backward compat).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
