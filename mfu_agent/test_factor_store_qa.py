"""QA test cases TC-E-050 through TC-E-059 for factor_store module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import UTC, datetime, timedelta
from data_io.factor_store import FactorStore, FactorStoreFrozenError, DeviceMetadata, FleetMeta
from data_io.models import NormalizedEvent, ResourceSnapshot

results = []
now = datetime(2026, 4, 19, 12, 0, 0, tzinfo=UTC)


def make_event(dev, ts, code='C6000'):
    return NormalizedEvent(device_id=dev, timestamp=ts, error_code=code)


def make_snapshot(dev, ts):
    return ResourceSnapshot(device_id=dev, timestamp=ts, toner_level=50)


# TC-E-050: add_events stores under correct device_id
try:
    store = FactorStore(reference_time=now)
    e1 = [make_event('DEV1', now - timedelta(days=1)), make_event('DEV1', now - timedelta(days=2))]
    e2 = [make_event('DEV2', now - timedelta(days=1))]
    store.add_events('DEV1', e1)
    store.add_events('DEV2', e2)
    got = store.get_events('DEV1', window_days=30)
    ok = len(got) == 2 and all(e.device_id == 'DEV1' for e in got)
    results.append(f'TC-E-050 | {"PASS" if ok else "FAIL"} | get_events(DEV1) returned {len(got)} events, all DEV1={ok}')
except Exception as ex:
    results.append(f'TC-E-050 | FAIL | exception: {ex}')

# TC-E-051: set_resources replaces previous snapshot
try:
    store = FactorStore(reference_time=now)
    s1 = ResourceSnapshot(device_id='DEV1', timestamp=now - timedelta(days=2), toner_level=80)
    s2 = ResourceSnapshot(device_id='DEV1', timestamp=now - timedelta(days=1), toner_level=20)
    store.set_resources('DEV1', s1)
    store.set_resources('DEV1', s2)
    got = store.get_resources('DEV1')
    ok = got is not None and got.toner_level == 20
    results.append(f'TC-E-051 | {"PASS" if ok else "FAIL"} | toner_level={got.toner_level if got else None}, expected 20')
except Exception as ex:
    results.append(f'TC-E-051 | FAIL | exception: {ex}')

# TC-E-052: set_device_metadata idempotent
try:
    store = FactorStore(reference_time=now)
    meta = DeviceMetadata(device_id='DEV1', model='M1', vendor='V1')
    store.set_device_metadata('DEV1', meta)
    store.set_device_metadata('DEV1', meta)
    got = store.get_device_metadata('DEV1')
    ok = got is not None and got.model == 'M1' and got.vendor == 'V1'
    results.append(f'TC-E-052 | {"PASS" if ok else "FAIL"} | no error, model={got.model if got else None}, vendor={got.vendor if got else None}')
except Exception as ex:
    results.append(f'TC-E-052 | FAIL | exception: {ex}')

# TC-E-053: list_devices no duplicates
try:
    store = FactorStore(reference_time=now)
    store.add_events('DEV1', [make_event('DEV1', now - timedelta(days=1))])
    store.set_resources('DEV1', make_snapshot('DEV1', now))
    store.set_device_metadata('DEV1', DeviceMetadata(device_id='DEV1'))
    devs = store.list_devices()
    ok = devs.count('DEV1') == 1
    results.append(f'TC-E-053 | {"PASS" if ok else "FAIL"} | list_devices()={devs}')
except Exception as ex:
    results.append(f'TC-E-053 | FAIL | exception: {ex}')

# TC-E-054: freeze makes store immutable
try:
    store = FactorStore(reference_time=now)
    store.add_events('DEV1', [make_event('DEV1', now - timedelta(days=1))])
    store.freeze()
    raised = False
    try:
        store.add_events('DEV1', [make_event('DEV1', now - timedelta(days=2))])
    except FactorStoreFrozenError:
        raised = True
    events_unchanged = len(store.get_events('DEV1', window_days=30)) == 1
    ok = raised and events_unchanged
    results.append(f'TC-E-054 | {"PASS" if ok else "FAIL"} | raised={raised}, events_unchanged={events_unchanged}')
except Exception as ex:
    results.append(f'TC-E-054 | FAIL | exception: {ex}')

# TC-E-055: multiple freeze calls idempotent
try:
    store = FactorStore()
    store.freeze()
    store.freeze()
    store.freeze()
    ok = store.frozen
    results.append(f'TC-E-055 | {"PASS" if ok else "FAIL"} | 3x freeze(), frozen={ok}, no crash')
except Exception as ex:
    results.append(f'TC-E-055 | FAIL | exception: {ex}')

# TC-E-056: get_events window_days filtering
try:
    store = FactorStore(reference_time=now)
    recent = [make_event('DEV1', now - timedelta(days=d)) for d in [1, 10, 25]]
    old = [make_event('DEV1', now - timedelta(days=d)) for d in [35, 60]]
    store.add_events('DEV1', recent + old)
    got = store.get_events('DEV1', window_days=30)
    ok = len(got) == 3
    results.append(f'TC-E-056 | {"PASS" if ok else "FAIL"} | expected 3, got {len(got)}')
except Exception as ex:
    results.append(f'TC-E-056 | FAIL | exception: {ex}')

# TC-E-057: count_repetitions with window
try:
    store = FactorStore(reference_time=now)
    within = [make_event('DEV1', now - timedelta(days=d), 'C6000') for d in [1, 5, 10, 20]]
    older = [make_event('DEV1', now - timedelta(days=100), 'C6000')]
    store.add_events('DEV1', within + older)
    c30 = store.count_repetitions('DEV1', 'C6000', window_days=30)
    c365 = store.count_repetitions('DEV1', 'C6000', window_days=365)
    ok = c30 == 4 and c365 == 5
    results.append(f'TC-E-057 | {"PASS" if ok else "FAIL"} | count(30d)={c30} expected 4, count(365d)={c365} expected 5')
except Exception as ex:
    results.append(f'TC-E-057 | FAIL | exception: {ex}')

# TC-E-058: access non-existent device
try:
    store = FactorStore(reference_time=now)
    evts = store.get_events('NOPE', window_days=30)
    res = store.get_resources('NOPE')
    meta = store.get_device_metadata('NOPE')
    cnt = store.count_repetitions('NOPE', 'C6000', window_days=30)
    ok = evts == [] and res is None and meta is None and cnt == 0
    results.append(f'TC-E-058 | {"PASS" if ok else "FAIL"} | events={evts}, resources={res}, meta={meta}, count={cnt}')
except Exception as ex:
    results.append(f'TC-E-058 | FAIL | exception: {ex}')

# TC-E-059: freeze/thaw lifecycle - getters work after freeze
try:
    store = FactorStore(reference_time=now)
    store.add_events('DEV1', [make_event('DEV1', now - timedelta(days=1))])
    store.set_resources('DEV1', make_snapshot('DEV1', now))
    store.set_device_metadata('DEV1', DeviceMetadata(device_id='DEV1', model='X'))
    store.freeze()
    # all getters should work
    e = store.get_events('DEV1', window_days=30)
    r = store.get_resources('DEV1')
    m = store.get_device_metadata('DEV1')
    d = store.list_devices()
    c = store.count_repetitions('DEV1', 'C6000', window_days=30)
    ok = len(e) == 1 and r is not None and m is not None and d == ['DEV1'] and c == 1
    results.append(f'TC-E-059 | {"PASS" if ok else "FAIL"} | all getters work after freeze: events={len(e)}, resources={r is not None}, meta={m is not None}, devices={d}, count={c}')
except Exception as ex:
    results.append(f'TC-E-059 | FAIL | exception: {ex}')

print()
for r in results:
    print(r)
