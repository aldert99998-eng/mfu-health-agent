"""MemoryManager — in-memory learned-pattern storage.

Stores patterns discovered across devices, with deduplication,
evidence merging, and scope-based retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from data_io.models import LearnedPattern

logger = logging.getLogger(__name__)


class MemoryManager:
    """In-memory store for fleet-wide learned patterns.

    Designed for Streamlit session_state persistence via to_dict / from_dict.
    """

    def __init__(
        self,
        max_patterns_per_model: int = 50,
        min_evidence_devices: int = 2,
    ) -> None:
        self._max_per_model = max_patterns_per_model
        self._min_evidence = min_evidence_devices
        self._patterns: list[LearnedPattern] = []

    def save_pattern(self, pattern: LearnedPattern) -> bool:
        """Save a pattern if it meets evidence threshold; dedup/merge if exists.

        Returns True if the pattern was saved (new or merged), False if skipped.
        """
        if len(pattern.evidence_devices) < self._min_evidence:
            return False

        existing = self._find_duplicate(pattern)
        if existing is not None:
            merged_devices = list(
                dict.fromkeys(
                    existing.evidence_devices + pattern.evidence_devices
                ),
            )
            existing.evidence_devices = merged_devices[:50]
            return True

        scope_count = sum(1 for p in self._patterns if p.scope == pattern.scope)
        if scope_count >= self._max_per_model:
            logger.warning(
                "Scope '%s' at capacity (%d patterns), skipping",
                pattern.scope, self._max_per_model,
            )
            return False

        self._patterns.append(pattern)
        return True

    def get_patterns(self, scope: str | None = None) -> list[LearnedPattern]:
        """Return patterns matching scope. None returns all."""
        if scope is None:
            return list(self._patterns)
        return [
            p for p in self._patterns
            if p.scope == scope or p.scope == "fleet"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_patterns_per_model": self._max_per_model,
            "min_evidence_devices": self._min_evidence,
            "patterns": [p.model_dump(mode="json") for p in self._patterns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryManager:
        mgr = cls(
            max_patterns_per_model=data.get("max_patterns_per_model", 50),
            min_evidence_devices=data.get("min_evidence_devices", 2),
        )
        for raw in data.get("patterns", []):
            mgr._patterns.append(LearnedPattern.model_validate(raw))
        return mgr

    def _find_duplicate(self, pattern: LearnedPattern) -> LearnedPattern | None:
        for existing in self._patterns:
            if (
                existing.scope == pattern.scope
                and existing.observation == pattern.observation
            ):
                return existing
        return None

    def __len__(self) -> int:
        return len(self._patterns)
