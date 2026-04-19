"""Report assembly — Track C, Phase 6.1.

Builds a Report object from HealthResults, generates executive summary
via Agent, detects fleet-wide patterns, and delegates HTML/PDF rendering
to Jinja2 + WeasyPrint (Phase 6.2–6.3).
"""

from __future__ import annotations

import json
import logging
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_io.models import (
    AgentTraceSummary,
    CalculationSnapshot,
    CategoryBreakdown,
    DeviceReport,
    DistributionBin,
    FleetSummary,
    HealthResult,
    PatternGroup,
    PatternType,
    Report,
    ResourceState,
    SourceFileInfo,
    Trace,
)

if TYPE_CHECKING:
    from agent.core import Agent
    from config.loader import ReportConfig
    from data_io.factor_store import FactorStore

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agent" / "prompts"
_STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"


class ReportBuilder:
    """Assembles a Report from health calculation results.

    Uses Agent for LLM-powered sections (executive summary,
    per-device recommendations) and deterministic logic for
    statistics and pattern detection.
    """

    def __init__(
        self,
        agent: Agent,
        config: ReportConfig,
    ) -> None:
        self._agent = agent
        self._config = config

    def build(
        self,
        health_results: list[HealthResult],
        factor_store: FactorStore,
        calculation_snapshot: CalculationSnapshot,
        source_file_info: SourceFileInfo,
        traces: dict[str, Trace] | None = None,
        include_agent_trace: bool = False,
    ) -> Report:
        fleet_summary = self._build_fleet_summary(health_results)

        executive_summary = self._generate_executive_summary(
            fleet_summary, health_results,
        )

        top_patterns = self._detect_patterns(
            health_results, factor_store,
        )

        index_distribution = self._build_distribution(health_results)

        category_breakdown = self._build_category_breakdown(
            health_results, factor_store,
        )

        devices = self._build_device_reports(
            health_results, factor_store,
        )

        agent_trace_summary = None
        if include_agent_trace and traces:
            agent_trace_summary = self._build_agent_trace_summary(
                health_results, traces,
            )

        return Report(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now(UTC),
            source_file_name=source_file_info.file_name,
            source_file_hash=source_file_info.file_hash,
            analysis_window_days=calculation_snapshot.weights_data.get(
                "age", {},
            ).get("window_days", 30),
            fleet_summary=fleet_summary,
            executive_summary=executive_summary,
            top_patterns=top_patterns,
            index_distribution=index_distribution,
            category_breakdown=category_breakdown,
            devices=devices,
            agent_trace_summary=agent_trace_summary,
            calculation_snapshot=calculation_snapshot,
            include_agent_trace=include_agent_trace,
        )

    def render_html(self, report: Report) -> str:
        try:
            from jinja2 import Environment, FileSystemLoader, StrictUndefined

            project_root = Path(__file__).resolve().parent.parent
            template_dir = project_root / Path(self._config.rendering.template_path).parent
            template_name = Path(self._config.rendering.template_path).name

            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=True,
                undefined=StrictUndefined,
            )
            template = env.get_template(template_name)
            return template.render(report=report, interactive=True)
        except ImportError:
            logger.warning("Jinja2 not installed, render_html unavailable")
            return ""
        except Exception:
            logger.exception("Failed to render HTML")
            return ""

    def render_pdf(self, report: Report) -> bytes:
        try:
            from jinja2 import Environment, FileSystemLoader, StrictUndefined
            from weasyprint import CSS, HTML

            project_root = Path(__file__).resolve().parent.parent
            template_dir = project_root / Path(self._config.rendering.template_path).parent
            template_name = Path(self._config.rendering.template_path).name

            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=True,
                undefined=StrictUndefined,
            )
            template = env.get_template(template_name)
            html_string = template.render(report=report, interactive=False)

            css_path = project_root / self._config.rendering.css_path
            print_css_path = project_root / self._config.rendering.print_css_path

            stylesheets = []
            if css_path.exists():
                stylesheets.append(CSS(filename=str(css_path)))
            if print_css_path.exists():
                stylesheets.append(CSS(filename=str(print_css_path)))

            return HTML(  # type: ignore[no-any-return]
                string=html_string,
                base_url=str(project_root),
            ).write_pdf(stylesheets=stylesheets)
        except ImportError:
            logger.warning("WeasyPrint not installed, render_pdf unavailable")
            return b""
        except Exception:
            logger.exception("Failed to render PDF")
            return b""

    # ── Fleet Summary ────────────────────────────────────────────────────

    def _build_fleet_summary(
        self, results: list[HealthResult],
    ) -> FleetSummary:
        if not results:
            return FleetSummary(
                total_devices=0,
                average_index=0.0,
                median_index=0.0,
                zone_counts={"green": 0, "yellow": 0, "red": 0},
                average_confidence=0.0,
            )

        indices = [r.health_index for r in results]
        confidences = [r.confidence for r in results]

        zone_counts: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
        for r in results:
            zone_key = r.zone if isinstance(r.zone, str) else r.zone.value
            zone_counts[zone_key] = zone_counts.get(zone_key, 0) + 1

        delta = self._load_previous_delta(statistics.mean(indices))

        return FleetSummary(
            total_devices=len(results),
            average_index=round(statistics.mean(indices), 1),
            median_index=round(statistics.median(indices), 1),
            zone_counts=zone_counts,
            average_confidence=round(statistics.mean(confidences), 2),
            delta_vs_previous=delta,
        )

    def _load_previous_delta(self, current_avg: float) -> float | None:
        previous_path = _STORAGE_DIR / "previous_reports.json"
        if not previous_path.exists():
            return None
        try:
            data = json.loads(previous_path.read_text(encoding="utf-8"))
            prev_avg = data.get("average_index")
            if prev_avg is not None:
                return round(current_avg - prev_avg, 1)  # type: ignore[no-any-return]
        except Exception:
            logger.debug("Could not load previous report data")
        return None

    # ── Executive Summary ────────────────────────────────────────────────

    def _generate_executive_summary(
        self,
        fleet_summary: FleetSummary,
        results: list[HealthResult],
    ) -> str:
        if not results:
            return "Нет устройств для анализа."

        worst = sorted(results, key=lambda r: r.health_index)[:5]
        worst_data = [
            {
                "device_id": r.device_id,
                "health_index": r.health_index,
                "zone": r.zone if isinstance(r.zone, str) else r.zone.value,
                "top_factor": (
                    r.factor_contributions[0].label
                    if r.factor_contributions
                    else "—"
                ),
            }
            for r in worst
        ]

        code_counter: Counter[str] = Counter()
        for r in results:
            for fc in r.factor_contributions:
                code_counter[fc.label] += 1
        top_codes = [
            {"code": code, "count": cnt}
            for code, cnt in code_counter.most_common(5)
        ]

        try:
            prompt_template = (_PROMPTS_DIR / "executive_summary.md").read_text(
                encoding="utf-8",
            )
        except FileNotFoundError:
            prompt_template = "Напиши executive summary по данным парка МФУ.\n\n{fleet_summary_json}"

        fleet_dict = {
            "total_devices": fleet_summary.total_devices,
            "average_index": fleet_summary.average_index,
            "median_index": fleet_summary.median_index,
            "zone_counts": fleet_summary.zone_counts,
            "average_confidence": fleet_summary.average_confidence,
            "delta_vs_previous": fleet_summary.delta_vs_previous,
        }

        prompt_text = prompt_template.format(
            fleet_summary_json=json.dumps(fleet_dict, ensure_ascii=False, indent=2, default=str),
            worst_devices_json=json.dumps(worst_data, ensure_ascii=False, indent=2),
            top_error_codes_json=json.dumps(top_codes, ensure_ascii=False, indent=2),
            top_patterns_json="[]",
        )

        try:
            from config.loader import LLMGenerationParams
            from llm.client import LLMClient

            params = LLMGenerationParams(temperature=0.4, top_p=1.0, max_tokens=800)
            resp = self._agent._llm.generate(
                messages=[{"role": "user", "content": prompt_text}],
                tools=None,
                params=params,
            )
            clean = LLMClient.strip_reasoning_artifacts(resp.content)

            if not clean or len(clean) < 50:
                logger.warning("Executive summary too short after cleanup, using fallback")
                return self._fallback_executive_summary(fleet_summary)

            return clean
        except Exception:
            logger.exception("Failed to generate executive summary via LLM")
            return self._fallback_executive_summary(fleet_summary)

    @staticmethod
    def _fallback_executive_summary(fs: FleetSummary) -> str:
        red = fs.zone_counts.get("red", 0)
        yellow = fs.zone_counts.get("yellow", 0)
        green = fs.zone_counts.get("green", 0)
        return (
            f"Парк из {fs.total_devices} устройств. "
            f"Средний индекс здоровья: {fs.average_index}. "
            f"В зелёной зоне: {green}, жёлтой: {yellow}, красной: {red}."
        )

    # ── Pattern Detection ────────────────────────────────────────────────

    def _detect_patterns(
        self,
        results: list[HealthResult],
        factor_store: FactorStore,
    ) -> list[PatternGroup]:
        if len(results) < 2:
            return []

        patterns: list[PatternGroup] = []
        cfg = self._config.top_patterns

        patterns.extend(self._detect_mass_issues(results, factor_store, cfg.mass_issue_min_devices))
        patterns.extend(self._detect_location_clusters(results, factor_store))
        patterns.extend(self._detect_critical_singles(results, cfg))

        seen_devices: set[str] = set()
        deduped: list[PatternGroup] = []
        for p in sorted(patterns, key=lambda x: len(x.affected_device_ids), reverse=True):
            new_devices = set(p.affected_device_ids) - seen_devices
            if new_devices:
                seen_devices.update(new_devices)
                deduped.append(p)

        return deduped[: cfg.max_count]

    def _detect_mass_issues(
        self,
        results: list[HealthResult],
        factor_store: FactorStore,
        min_devices: int,
    ) -> list[PatternGroup]:
        model_error_devices: dict[tuple[str, str], list[str]] = defaultdict(list)

        for r in results:
            meta = factor_store.get_device_metadata(r.device_id)
            model = meta.model if meta and meta.model else None
            if not model:
                continue
            for fc in r.factor_contributions:
                label = fc.label.split(" (")[0] if " (" in fc.label else fc.label
                model_error_devices[(model, label)].append(r.device_id)

        patterns: list[PatternGroup] = []
        for (model, error_label), device_ids in model_error_devices.items():
            if len(device_ids) < min_devices:
                continue
            affected_results = [r for r in results if r.device_id in device_ids]
            avg_index = statistics.mean(r.health_index for r in affected_results)
            patterns.append(PatternGroup(
                pattern_type=PatternType.MASS_ISSUE,
                title=f"{error_label} на {model} ({len(device_ids)} уст.)"[:60],
                affected_device_ids=device_ids,
                average_index=round(avg_index, 1),
                explanation=(
                    f"Ошибка «{error_label}» обнаружена на {len(device_ids)} "
                    f"устройствах модели {model}. Средний индекс: {avg_index:.0f}."
                ),
            ))

        return patterns

    def _detect_location_clusters(
        self,
        results: list[HealthResult],
        factor_store: FactorStore,
    ) -> list[PatternGroup]:
        location_devices: dict[str, list[HealthResult]] = defaultdict(list)

        for r in results:
            meta = factor_store.get_device_metadata(r.device_id)
            loc = meta.location if meta and meta.location else None
            if not loc:
                continue
            location_devices[loc].append(r)

        patterns: list[PatternGroup] = []
        for loc, loc_results in location_devices.items():
            if len(loc_results) < 3:
                continue
            problem_count = sum(
                1
                for r in loc_results
                if (r.zone if isinstance(r.zone, str) else r.zone.value) in ("red", "yellow")
            )
            if problem_count < len(loc_results) * 0.6:
                continue
            avg_index = statistics.mean(r.health_index for r in loc_results)
            device_ids = [r.device_id for r in loc_results]
            patterns.append(PatternGroup(
                pattern_type=PatternType.LOCATION_CLUSTER,
                title=f"Проблемы в локации «{loc}» ({len(device_ids)} уст.)"[:60],
                affected_device_ids=device_ids,
                average_index=round(avg_index, 1),
                explanation=(
                    f"В локации «{loc}» {problem_count} из {len(loc_results)} "
                    f"устройств в жёлтой или красной зоне. "
                    f"Средний индекс: {avg_index:.0f}."
                ),
            ))

        return patterns

    @staticmethod
    def _detect_critical_singles(
        results: list[HealthResult],
        cfg: Any,
    ) -> list[PatternGroup]:
        patterns: list[PatternGroup] = []
        for r in results:
            is_critical = (
                r.health_index < cfg.critical_single_index_threshold
                or r.confidence < cfg.critical_single_confidence_threshold
            )
            if not is_critical:
                continue

            reason_parts = []
            if r.health_index < cfg.critical_single_index_threshold:
                reason_parts.append(f"индекс {r.health_index}")
            if r.confidence < cfg.critical_single_confidence_threshold:
                reason_parts.append(f"confidence {r.confidence}")

            top_factor = (
                r.factor_contributions[0].label if r.factor_contributions else "—"
            )
            patterns.append(PatternGroup(
                pattern_type=PatternType.CRITICAL_SINGLE,
                title=f"Критическое: {r.device_id} ({', '.join(reason_parts)})"[:60],
                affected_device_ids=[r.device_id],
                average_index=float(r.health_index),
                explanation=(
                    f"Устройство {r.device_id} требует внимания: "
                    f"{', '.join(reason_parts)}. "
                    f"Основной фактор: {top_factor}."
                ),
            ))

        return patterns

    # ── Distribution ─────────────────────────────────────────────────────

    @staticmethod
    def _build_distribution(
        results: list[HealthResult],
    ) -> list[DistributionBin]:
        bins: list[DistributionBin] = []
        counts = Counter[int]()
        for r in results:
            bin_idx = min(r.health_index // 10, 9)
            counts[bin_idx] += 1

        for i in range(10):
            start = i * 10
            end = start + 9 if i < 9 else 100
            bins.append(DistributionBin(
                range_start=start,
                range_end=end,
                count=counts.get(i, 0),
            ))
        return bins

    # ── Category Breakdown ───────────────────────────────────────────────

    def _build_category_breakdown(
        self,
        results: list[HealthResult],
        factor_store: FactorStore,
    ) -> CategoryBreakdown | None:
        if not results:
            return None

        models: set[str] = set()
        locations: set[str] = set()
        for r in results:
            meta = factor_store.get_device_metadata(r.device_id)
            if meta:
                if meta.model:
                    models.add(meta.model)
                if meta.location:
                    locations.add(meta.location)

        if len(models) > 1:
            return self._breakdown_by_field(results, factor_store, "model")
        if len(locations) > 1:
            return self._breakdown_by_field(results, factor_store, "location")
        return self._breakdown_by_confidence_zone(results)

    def _breakdown_by_field(
        self,
        results: list[HealthResult],
        factor_store: FactorStore,
        field: str,
    ) -> CategoryBreakdown:
        groups: dict[str, list[HealthResult]] = defaultdict(list)
        for r in results:
            meta = factor_store.get_device_metadata(r.device_id)
            key = getattr(meta, field, None) if meta else None
            groups[key or "Неизвестно"].append(r)

        return CategoryBreakdown(
            category_field=field,
            groups={
                name: self._build_fleet_summary(group_results)
                for name, group_results in groups.items()
            },
        )

    def _breakdown_by_confidence_zone(
        self, results: list[HealthResult],
    ) -> CategoryBreakdown:
        groups: dict[str, list[HealthResult]] = defaultdict(list)
        for r in results:
            cz = r.confidence_zone if isinstance(r.confidence_zone, str) else r.confidence_zone.value
            groups[cz].append(r)

        return CategoryBreakdown(
            category_field="confidence_zone",
            groups={
                name: self._build_fleet_summary(group_results)
                for name, group_results in groups.items()
            },
        )

    # ── Device Reports ───────────────────────────────────────────────────

    def _build_device_reports(
        self,
        results: list[HealthResult],
        factor_store: FactorStore,
    ) -> list[DeviceReport]:
        devices: list[DeviceReport] = []
        for r in sorted(results, key=lambda x: x.health_index):
            meta = factor_store.get_device_metadata(r.device_id)
            resource_snapshot = factor_store.get_resources(r.device_id)

            resource_state = ResourceState()
            if resource_snapshot:
                resource_state = ResourceState(
                    toner=resource_snapshot.toner_level,
                    drum=resource_snapshot.drum_level,
                    fuser=resource_snapshot.fuser_level,
                    mileage=resource_snapshot.mileage,
                    service_interval=resource_snapshot.service_interval,
                )

            index_history = self._load_device_history(r.device_id)

            top_problem = ""
            if r.factor_contributions:
                top_problem = r.factor_contributions[0].label

            recommendation = r.reflection_notes or ""

            devices.append(DeviceReport(
                device_id=r.device_id,
                model=meta.model if meta else None,
                location=meta.location if meta else None,
                health_index=r.health_index,
                confidence=r.confidence,
                zone=r.zone,
                confidence_zone=r.confidence_zone,
                top_problem_tag=top_problem,
                factor_contributions=list(r.factor_contributions),
                resource_state=resource_state,
                index_history=index_history,
                agent_recommendation=recommendation,
                flag_for_review=getattr(r, "flag_for_review", False)
                or bool(r.reflection_notes),
            ))
        return devices

    @staticmethod
    def _load_device_history(device_id: str) -> list[Any]:
        history_path = _STORAGE_DIR / "history" / f"{device_id}.json"
        if not history_path.exists():
            return []
        try:
            from data_io.models import HistoryPoint

            data = json.loads(history_path.read_text(encoding="utf-8"))
            return [HistoryPoint.model_validate(p) for p in data]
        except Exception:
            logger.debug("Could not load history for %s", device_id)
            return []

    # ── Agent Trace Summary ──────────────────────────────────────────────

    def _build_agent_trace_summary(
        self,
        results: list[HealthResult],
        traces: dict[str, Trace],
    ) -> AgentTraceSummary | None:
        if not traces:
            return None

        all_tool_calls = [t.total_tool_calls for t in traces.values()]
        all_llm_calls = [t.total_llm_calls for t in traces.values()]
        restart_count = sum(1 for t in traces.values() if t.attempts > 1)
        flagged = [
            did for did, t in traces.items() if t.flagged_for_review
        ]

        return AgentTraceSummary(
            average_tool_calls_per_device=(
                statistics.mean(all_tool_calls) if all_tool_calls else 0.0
            ),
            average_llm_calls_per_device=(
                statistics.mean(all_llm_calls) if all_llm_calls else 0.0
            ),
            self_check_restart_count=restart_count,
            devices_flagged_for_review=flagged,
        )

    # ── Utility ──────────────────────────────────────────────────────────

    def save_for_delta(self, report: Report) -> None:
        """Persist summary data so the next report can compute delta."""
        previous_path = _STORAGE_DIR / "previous_reports.json"
        previous_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "average_index": report.fleet_summary.average_index,
            "generated_at": report.generated_at.isoformat(),
            "report_id": report.report_id,
        }
        previous_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
