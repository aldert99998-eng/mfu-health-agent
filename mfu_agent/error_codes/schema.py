"""Pydantic schemas for per-device-model error code dictionaries."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SeverityLiteral = Literal["Critical", "High", "Medium", "Low", "Info"]
SUPPORTED_VENDORS = ("Xerox", "Lexmark", "Ricoh")


class ErrorCode(BaseModel):
    """A single error code entry in a model dictionary."""

    model_config = ConfigDict(frozen=False, extra="ignore")

    description: str = Field(min_length=1, max_length=500)
    severity: SeverityLiteral
    component: str = Field(default="", max_length=80)
    notes: str = Field(default="", max_length=1000)

    @field_validator("component", "notes", mode="before")
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()


class ModelErrorCodes(BaseModel):
    """Error-code dictionary for one (vendor, model) pair."""

    model_config = ConfigDict(frozen=False, extra="ignore")

    vendor: str = Field(min_length=1)
    model: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    codes: dict[str, ErrorCode] = Field(default_factory=dict)

    @field_validator("vendor")
    @classmethod
    def _normalize_vendor(cls, v: str) -> str:
        s = v.strip()
        for canonical in SUPPORTED_VENDORS:
            if s.lower() == canonical.lower():
                return canonical
        return s

    def merge(self, other: "ModelErrorCodes", *, replace: bool = False) -> "ModelErrorCodes":
        """Return a new ModelErrorCodes merged with `other`.

        replace=True: other wins entirely (this = other).
        replace=False: upsert — new codes added, existing updated with other's values.
        """
        if replace:
            return ModelErrorCodes(
                vendor=self.vendor,
                model=self.model,
                updated_at=datetime.now(UTC),
                codes=dict(other.codes),
            )
        merged = dict(self.codes)
        for k, v in other.codes.items():
            merged[k] = v
        return ModelErrorCodes(
            vendor=self.vendor,
            model=self.model,
            updated_at=datetime.now(UTC),
            codes=merged,
        )
