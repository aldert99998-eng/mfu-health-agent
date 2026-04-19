"""In-memory factor store for device events and resources."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_io.models import NormalizedEvent, ResourceSnapshot

# ── Errors ────────────────────────────────────────────────────────────────────


class FactorStoreFrozenError(Exception):
    """Raised on mutation attempt after freeze()."""

    def __init__(self) -> None:
        super().__init__("FactorStore is frozen and cannot be modified")


# ── Supporting models ─────────────────────────────────────────────────────────


class DeviceMetadata(BaseModel):
    """Static metadata for a single device."""

    model_config = ConfigDict(frozen=True)

    device_id: str = Field(min_length=1, max_length=100)
    model: str | None = None
    vendor: str | None = None
    location: str | None = None
    critical_function: bool = False
    tags: list[str] = Field(default_factory=list)


class FleetMeta(BaseModel):
    """Metadata about the ingested data file / fleet snapshot."""

    model_config = ConfigDict(frozen=True)

    file_hash: str = ""
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    mapping_profile: str | None = None
    source_filename: str | None = None
    total_records: int = 0


# ── FactorStore ───────────────────────────────────────────────────────────────


class FactorStore:
    """In-memory storage that the agent uses to access device data.

    Lifecycle:
        1. Ingestion phase — add events, resources, metadata via mutators.
        2. freeze() — lock store; all subsequent mutations raise FactorStoreFrozenError.
        3. Query phase — agent reads via getters with windowed queries.
    """

    def __init__(
        self,
        *,
        reference_time: datetime | None = None,
    ) -> None:
        self._events: dict[str, list[NormalizedEvent]] = {}
        self._resources: dict[str, ResourceSnapshot] = {}
        self._device_metadata: dict[str, DeviceMetadata] = {}
        self._fleet_meta: FleetMeta | None = None
        self._frozen: bool = False
        self._reference_time = reference_time

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def reference_time(self) -> datetime:
        if self._reference_time is not None:
            return self._reference_time
        return datetime.now(UTC)

    @property
    def fleet_meta(self) -> FleetMeta | None:
        return self._fleet_meta

    def set_reference_time(self, ref: datetime) -> None:
        self._reference_time = ref

    # ── Ingestion (mutators) ──────────────────────────────────────────────

    def _check_not_frozen(self) -> None:
        if self._frozen:
            raise FactorStoreFrozenError()

    def add_events(self, device_id: str, events: list[NormalizedEvent]) -> None:
        self._check_not_frozen()
        if device_id not in self._events:
            self._events[device_id] = []
        self._events[device_id].extend(events)

    def set_resources(self, device_id: str, snapshot: ResourceSnapshot) -> None:
        self._check_not_frozen()
        self._resources[device_id] = snapshot

    def set_device_metadata(self, device_id: str, meta: DeviceMetadata) -> None:
        self._check_not_frozen()
        self._device_metadata[device_id] = meta

    def set_fleet_meta(self, meta: FleetMeta) -> None:
        self._check_not_frozen()
        self._fleet_meta = meta

    def freeze(self) -> None:
        self._frozen = True

    def thaw(self) -> None:
        self._frozen = False

    # ── Getters ───────────────────────────────────────────────────────────

    def list_devices(self) -> list[str]:
        device_ids: set[str] = set()
        device_ids.update(self._events)
        device_ids.update(self._resources)
        device_ids.update(self._device_metadata)
        return sorted(device_ids)

    def get_events(
        self,
        device_id: str,
        window_days: int = 30,
    ) -> list[NormalizedEvent]:
        all_events = self._events.get(device_id, [])
        if not all_events:
            return []
        cutoff = self.reference_time - timedelta(days=window_days)
        return [e for e in all_events if e.timestamp >= cutoff]

    def get_resources(self, device_id: str) -> ResourceSnapshot | None:
        return self._resources.get(device_id)

    def get_device_metadata(self, device_id: str) -> DeviceMetadata | None:
        return self._device_metadata.get(device_id)

    def count_repetitions(
        self,
        device_id: str,
        error_code: str,
        window_days: int = 14,
    ) -> int:
        events = self.get_events(device_id, window_days=window_days)
        return sum(1 for e in events if e.error_code == error_code)

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": {
                did: [e.model_dump(mode="json") for e in evts]
                for did, evts in self._events.items()
            },
            "resources": {
                did: snap.model_dump(mode="json")
                for did, snap in self._resources.items()
            },
            "device_metadata": {
                did: meta.model_dump(mode="json")
                for did, meta in self._device_metadata.items()
            },
            "fleet_meta": (
                self._fleet_meta.model_dump(mode="json")
                if self._fleet_meta
                else None
            ),
            "frozen": self._frozen,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FactorStore:
        store = cls()
        for did, evts in data.get("events", {}).items():
            store._events[did] = [NormalizedEvent.model_validate(e) for e in evts]
        for did, snap in data.get("resources", {}).items():
            store._resources[did] = ResourceSnapshot.model_validate(snap)
        for did, meta in data.get("device_metadata", {}).items():
            store._device_metadata[did] = DeviceMetadata.model_validate(meta)
        fm = data.get("fleet_meta")
        if fm is not None:
            store._fleet_meta = FleetMeta.model_validate(fm)
        if data.get("frozen", False):
            store._frozen = True
        return store

    # ── Dunder ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FactorStore(devices={len(self.list_devices())}, "
            f"frozen={self._frozen})"
        )
