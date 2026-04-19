"""Deterministic health index calculator — Track A core.

Pure function: takes factors and weights, returns a numeric result.
No LLM calls, no external dependencies, no global state.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime

from data_io.models import (
    ConfidenceFactors,
    ConfidenceZone,
    Factor,
    FactorContribution,
    HealthResult,
    HealthZone,
    SeverityLevel,
    WeightsProfile,
)


def compute_R(n_repetitions: int, base: int, max_value: float) -> float:
    """R = min(max_value, 1 + log_base(n))."""
    if n_repetitions <= 1:
        return 1.0
    return min(max_value, 1.0 + math.log(n_repetitions, base))


def compute_C(applicable_multipliers: list[float], max_value: float) -> float:
    """C = min(max_value, product of applicable multipliers)."""
    if not applicable_multipliers:
        return 1.0
    product = 1.0
    for m in applicable_multipliers:
        product *= m
    return min(max_value, product)


def compute_A(age_days: int, tau_days: int) -> float:
    """A = exp(-d / tau)."""
    if age_days < 0:
        return 1.0
    return math.exp(-age_days / tau_days)


def select_one_critical_per_day(
    factors: list[Factor],
    limit: int = 1,
) -> list[Factor]:
    """Apply the one-Critical-per-day rule.

    For each calendar day, keep only `limit` Critical factors
    (those with the highest individual penalty S*R*C*A).
    Non-Critical factors pass through unchanged.
    """
    non_critical: list[Factor] = []
    critical_by_day: dict[str, list[Factor]] = defaultdict(list)

    for f in factors:
        if f.severity_level == SeverityLevel.CRITICAL:
            day_key = f.event_timestamp.strftime("%Y-%m-%d")
            critical_by_day[day_key].append(f)
        else:
            non_critical.append(f)

    selected: list[Factor] = list(non_critical)
    for _day, day_factors in critical_by_day.items():
        day_factors.sort(key=lambda f: f.S * f.R * f.C * f.A, reverse=True)
        selected.extend(day_factors[:limit])

    return selected


def _compute_confidence(
    confidence_factors: ConfidenceFactors,
    weights: WeightsProfile,
) -> tuple[float, list[str]]:
    """Compute confidence score and collect reasons for reduction."""
    penalties = weights.confidence.penalties
    min_value = weights.confidence.min_value

    product = 1.0
    reasons: list[str] = []

    if confidence_factors.rag_missing_count > 0:
        product *= penalties.rag_not_found ** confidence_factors.rag_missing_count
        reasons.append(
            f"RAG не нашёл {confidence_factors.rag_missing_count} код(ов) ошибок"
        )

    if confidence_factors.missing_resources:
        product *= penalties.missing_resources
        reasons.append("Нет данных о ресурсах")

    if confidence_factors.missing_model:
        product *= penalties.missing_model
        reasons.append("Модель устройства не определена")

    if confidence_factors.abnormal_daily_jump:
        product *= penalties.abnormal_daily_jump
        reasons.append("Резкий скачок индекса (>30 пунктов)")

    if confidence_factors.anomalous_event_count:
        product *= penalties.anomalous_event_count
        reasons.append("Аномальное количество событий")

    if confidence_factors.no_events_and_no_resources:
        product *= penalties.no_events_and_no_resources
        reasons.append("Нет событий и нет снимка ресурсов")

    return max(min_value, product), reasons


def _determine_zone(health_index: int, weights: WeightsProfile) -> HealthZone:
    """Determine health zone from index value."""
    if health_index >= weights.zones.green_threshold:
        return HealthZone.GREEN
    if health_index >= weights.zones.red_threshold:
        return HealthZone.YELLOW
    return HealthZone.RED


def _determine_confidence_zone(confidence: float) -> ConfidenceZone:
    """Determine confidence zone from score."""
    if confidence >= 0.85:
        return ConfidenceZone.HIGH
    if confidence >= 0.60:
        return ConfidenceZone.MEDIUM
    return ConfidenceZone.LOW


def calculate_health_index(
    factors: list[Factor],
    confidence_factors: ConfidenceFactors,
    weights: WeightsProfile,
    *,
    device_id: str = "",
    silent_device_mode: str = "data_quality",
) -> HealthResult:
    """Calculate health index for a single device.

    H = max(1, 100 - Σ(S·R·C·A))
    Conf = max(0.2, Π(penalties))
    """
    now = datetime.now(UTC)

    if not factors:
        confidence = 1.0
        reasons: list[str] = []
        if silent_device_mode == "optimistic":
            confidence = 1.0
        elif silent_device_mode == "data_quality":
            confidence = 0.9
            reasons.append("Нет событий — режим data_quality")
        else:
            confidence = 0.9
            reasons.append("Нет событий — carry_forward без истории")

        return HealthResult(
            device_id=device_id,
            health_index=100,
            confidence=confidence,
            zone=HealthZone.GREEN,
            confidence_zone=_determine_confidence_zone(confidence),
            factor_contributions=[],
            confidence_reasons=reasons,
            calculation_snapshot=weights.model_dump(mode="json"),
            calculated_at=now,
        )

    filtered = select_one_critical_per_day(
        factors, limit=weights.critical_per_day_limit
    )

    contributions: list[FactorContribution] = []
    total_penalty = 0.0

    for f in filtered:
        penalty = f.S * f.R * f.C * f.A
        if not math.isfinite(penalty):
            raise ValueError(f"Non-finite penalty for {f.error_code}: S={f.S}, R={f.R}, C={f.C}, A={f.A}")
        total_penalty += penalty
        contributions.append(
            FactorContribution(
                label=f"{f.error_code} ({f.severity_level})",
                penalty=round(penalty, 2),
                S=f.S,
                R=round(f.R, 4),
                C=round(f.C, 4),
                A=round(f.A, 4),
                source=f.source or "",
            )
        )

    contributions.sort(key=lambda c: c.penalty, reverse=True)

    health_index = max(1, round(100 - total_penalty))

    confidence, confidence_reasons = _compute_confidence(
        confidence_factors, weights
    )

    return HealthResult(
        device_id=device_id,
        health_index=health_index,
        confidence=round(confidence, 4),
        zone=_determine_zone(health_index, weights),
        confidence_zone=_determine_confidence_zone(confidence),
        factor_contributions=contributions,
        confidence_reasons=confidence_reasons,
        calculation_snapshot=weights.model_dump(mode="json"),
        calculated_at=now,
    )
