"""Consistency checker: find differences between models of the same vendor."""

from __future__ import annotations

from dataclasses import dataclass

from .loader import list_models, load_codes
from .schema import ModelErrorCodes


@dataclass(frozen=True)
class Conflict:
    """One code differs between this model and another model of the same vendor."""

    code: str
    field: str  # "severity" | "description" | "component"
    this_value: str
    other_model: str
    other_value: str


def find_conflicts(vendor: str, this: ModelErrorCodes) -> list[Conflict]:
    """Compare `this` model against every other model of same vendor.

    Only codes present in BOTH models are compared. Fields checked:
    severity, description, component.
    """
    result: list[Conflict] = []
    my_model_slug = this.model
    for other_slug in list_models(vendor):
        if other_slug == my_model_slug.lower():
            continue
        other = load_codes(vendor, other_slug)
        if other is None or other.model == this.model:
            continue
        for code, info in this.codes.items():
            other_info = other.codes.get(code)
            if other_info is None:
                continue
            for field in ("severity", "description", "component"):
                a = getattr(info, field)
                b = getattr(other_info, field)
                if a != b:
                    result.append(Conflict(
                        code=code,
                        field=field,
                        this_value=str(a),
                        other_model=other.model,
                        other_value=str(b),
                    ))
    return result


def sync_to_all_models(vendor: str, source: ModelErrorCodes) -> list[str]:
    """Push source's codes into every other model of vendor, upsert.

    For each code in source: if that code exists in target, overwrite
    its fields with source's. Codes not in source are left alone.
    Returns list of affected model names.
    """
    from .writer import save

    affected: list[str] = []
    for other_slug in list_models(vendor):
        if other_slug == source.model.lower():
            continue
        other = load_codes(vendor, other_slug)
        if other is None:
            continue
        changed = False
        for code, info in source.codes.items():
            if code in other.codes and other.codes[code] != info:
                other.codes[code] = info
                changed = True
        if changed:
            save(other)
            affected.append(other.model)
    return affected
