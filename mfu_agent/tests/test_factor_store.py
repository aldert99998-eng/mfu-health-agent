"""Tests for data_io/factor_store.py — Phase 1.3 verification."""

from datetime import UTC, datetime, timedelta

import pytest

from data_io.factor_store import (
    DeviceMetadata,
    FactorStore,
    FactorStoreFrozenError,
    FleetMeta,
)
from data_io.models import NormalizedEvent, ResourceSnapshot

NOW = datetime(2026, 4, 18, 12, 0, 0, tzinfo=UTC)


def _event(device_id: str, days_ago: float, code: str = "C6000") -> NormalizedEvent:
    return NormalizedEvent(
        device_id=device_id,
        timestamp=NOW - timedelta(days=days_ago),
        error_code=code,
        error_description="test",
    )


def _snapshot(device_id: str) -> ResourceSnapshot:
    return ResourceSnapshot(
        device_id=device_id,
        timestamp=NOW,
        toner_level=45,
        drum_level=72,
    )


def _meta(device_id: str) -> DeviceMetadata:
    return DeviceMetadata(
        device_id=device_id,
        model="Kyocera TASKalfa 3253ci",
        vendor="Kyocera",
        location="Офис 1",
    )


# ── Freeze / thaw semantics ──────────────────────────────────────────────────


class TestFreeze:
    def test_add_events_after_freeze_raises(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [_event("D001", 1)])
        store.freeze()
        with pytest.raises(FactorStoreFrozenError):
            store.add_events("D001", [_event("D001", 0)])

    def test_set_resources_after_freeze_raises(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.freeze()
        with pytest.raises(FactorStoreFrozenError):
            store.set_resources("D001", _snapshot("D001"))

    def test_set_device_metadata_after_freeze_raises(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.freeze()
        with pytest.raises(FactorStoreFrozenError):
            store.set_device_metadata("D001", _meta("D001"))

    def test_set_fleet_meta_after_freeze_raises(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.freeze()
        with pytest.raises(FactorStoreFrozenError):
            store.set_fleet_meta(FleetMeta(file_hash="abc"))

    def test_thaw_allows_mutations(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.freeze()
        assert store.frozen is True
        store.thaw()
        assert store.frozen is False
        store.add_events("D001", [_event("D001", 1)])
        assert len(store.get_events("D001")) == 1

    def test_frozen_property(self) -> None:
        store = FactorStore()
        assert store.frozen is False
        store.freeze()
        assert store.frozen is True


# ── Windowed queries ─────────────────────────────────────────────────────────


class TestWindowedQueries:
    def test_events_inside_window(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [
            _event("D001", 5),
            _event("D001", 15),
            _event("D001", 29),
        ])
        result = store.get_events("D001", window_days=30)
        assert len(result) == 3

    def test_events_outside_window_excluded(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [
            _event("D001", 5),
            _event("D001", 31),
            _event("D001", 60),
        ])
        result = store.get_events("D001", window_days=30)
        assert len(result) == 1

    def test_boundary_exactly_30_days(self) -> None:
        store = FactorStore(reference_time=NOW)
        exactly_30 = _event("D001", 30)
        just_inside = _event("D001", 29.999)
        just_outside = _event("D001", 30.001)
        store.add_events("D001", [exactly_30, just_inside, just_outside])
        result = store.get_events("D001", window_days=30)
        assert exactly_30 in result
        assert just_inside in result
        assert just_outside not in result

    def test_custom_window(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [
            _event("D001", 5),
            _event("D001", 10),
            _event("D001", 20),
        ])
        result = store.get_events("D001", window_days=14)
        assert len(result) == 2

    def test_empty_device_returns_empty(self) -> None:
        store = FactorStore(reference_time=NOW)
        assert store.get_events("NONEXISTENT") == []

    def test_count_repetitions(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [
            _event("D001", 1, "C6000"),
            _event("D001", 3, "C6000"),
            _event("D001", 5, "A1001"),
            _event("D001", 20, "C6000"),
        ])
        assert store.count_repetitions("D001", "C6000", window_days=14) == 2
        assert store.count_repetitions("D001", "A1001", window_days=14) == 1
        assert store.count_repetitions("D001", "C6000", window_days=2) == 1

    def test_count_repetitions_no_match(self) -> None:
        store = FactorStore(reference_time=NOW)
        assert store.count_repetitions("D001", "X9999") == 0


# ── Getters ──────────────────────────────────────────────────────────────────


class TestGetters:
    def test_list_devices_union(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [_event("D001", 1)])
        store.set_resources("D002", _snapshot("D002"))
        store.set_device_metadata("D003", _meta("D003"))
        devices = store.list_devices()
        assert devices == ["D001", "D002", "D003"]

    def test_list_devices_sorted(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D003", [_event("D003", 1)])
        store.add_events("D001", [_event("D001", 1)])
        assert store.list_devices() == ["D001", "D003"]

    def test_get_resources(self) -> None:
        store = FactorStore(reference_time=NOW)
        snap = _snapshot("D001")
        store.set_resources("D001", snap)
        assert store.get_resources("D001") is snap
        assert store.get_resources("NONE") is None

    def test_get_device_metadata(self) -> None:
        store = FactorStore(reference_time=NOW)
        meta = _meta("D001")
        store.set_device_metadata("D001", meta)
        assert store.get_device_metadata("D001") is meta
        assert store.get_device_metadata("NONE") is None

    def test_fleet_meta(self) -> None:
        store = FactorStore(reference_time=NOW)
        assert store.fleet_meta is None
        fm = FleetMeta(file_hash="sha256:abc", mapping_profile="kyocera_csv")
        store.set_fleet_meta(fm)
        assert store.fleet_meta is fm


# ── Serialization roundtrip ──────────────────────────────────────────────────


class TestSerialization:
    def _filled_store(self) -> FactorStore:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [
            _event("D001", 1, "C6000"),
            _event("D001", 10, "A1001"),
        ])
        store.add_events("D002", [_event("D002", 5)])
        store.set_resources("D001", _snapshot("D001"))
        store.set_device_metadata("D001", _meta("D001"))
        store.set_fleet_meta(FleetMeta(
            file_hash="sha256:abc",
            upload_timestamp=NOW,
            mapping_profile="kyocera_csv",
            source_filename="fleet.csv",
            total_records=100,
        ))
        store.freeze()
        return store

    def test_roundtrip_preserves_data(self) -> None:
        original = self._filled_store()
        data = original.to_dict()
        restored = FactorStore.from_dict(data)

        assert restored.list_devices() == original.list_devices()
        assert restored.frozen is True

        for did in original.list_devices():
            orig_events = original._events.get(did, [])
            rest_events = restored._events.get(did, [])
            assert len(rest_events) == len(orig_events)
            for o, r in zip(orig_events, rest_events, strict=True):
                assert o.model_dump() == r.model_dump()

            orig_res = original.get_resources(did)
            rest_res = restored.get_resources(did)
            if orig_res is not None:
                assert rest_res is not None
                assert orig_res.model_dump() == rest_res.model_dump()

            orig_meta = original.get_device_metadata(did)
            rest_meta = restored.get_device_metadata(did)
            if orig_meta is not None:
                assert rest_meta is not None
                assert orig_meta.model_dump() == rest_meta.model_dump()

        assert restored.fleet_meta is not None
        assert original.fleet_meta is not None
        assert restored.fleet_meta.model_dump() == original.fleet_meta.model_dump()

    def test_roundtrip_empty_store(self) -> None:
        store = FactorStore()
        data = store.to_dict()
        restored = FactorStore.from_dict(data)
        assert restored.list_devices() == []
        assert restored.frozen is False
        assert restored.fleet_meta is None

    def test_to_dict_structure(self) -> None:
        store = self._filled_store()
        data = store.to_dict()
        assert set(data.keys()) == {"events", "resources", "device_metadata", "fleet_meta", "frozen"}
        assert isinstance(data["events"], dict)
        assert isinstance(data["resources"], dict)
        assert data["frozen"] is True


# ── Repr ─────────────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr(self) -> None:
        store = FactorStore(reference_time=NOW)
        store.add_events("D001", [_event("D001", 1)])
        assert "devices=1" in repr(store)
        assert "frozen=False" in repr(store)
