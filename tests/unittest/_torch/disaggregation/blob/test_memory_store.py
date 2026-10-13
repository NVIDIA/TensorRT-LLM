# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The private driver boundary, exercised through the in-memory store and driver.

These tests pin what any driver owes the backend: bytes move whole and never alias the source,
results are per row and never pass a timeout off as a miss, access end is separate evidence, and
failed registrations stay cleanable.
"""

from __future__ import annotations

import threading

import pytest
from host_memory import HostBuffer
from memory_store import MemoryDriver, MemoryOp, MemoryStore

from tensorrt_llm._torch.disaggregation.blob.format import (
    TRAILER_BYTES,
    check_trailer,
    encode_trailer,
)
from tensorrt_llm._torch.disaggregation.blob.keys import KEY_FORMAT_VERSION, encode_key
from tensorrt_llm._torch.disaggregation.blob.store import (
    ProbeStatus,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowStatus,
    Span,
    StoreDriver,
    StoreError,
    concluded,
)

pytestmark = pytest.mark.cpu_only

SLOT = 8
NAMES = {"A": b"name-A", "B": b"name-B", "C": b"name-C"}


class Endpoint:
    """One side of a transfer: a registered payload region plus registered trailer buffers."""

    def __init__(self, driver: MemoryDriver, labels: list[str], fill: int = 0) -> None:
        self.driver = driver
        self.labels = labels
        self.payload = HostBuffer(SLOT * len(labels), fill)
        self.trailer = HostBuffer(TRAILER_BYTES * len(labels), fill)
        self.handles = [
            driver.register(
                self.payload.address, self.payload.nbytes, RegionLayout(SLOT, len(labels), "kv")
            ),
            driver.register(
                self.trailer.address,
                self.trailer.nbytes,
                RegionLayout(TRAILER_BYTES, len(labels), "trailer"),
            ),
        ]
        self.rows = [
            RowIO(
                encode_key("t", NAMES[label]),
                self.payload.span(i * SLOT, SLOT),
                self.trailer.span(i * TRAILER_BYTES, TRAILER_BYTES),
                (0, i),
            )
            for i, label in enumerate(labels)
        ]

    def fill_row(self, index: int, data: bytes) -> None:
        self.payload.write(data, index * SLOT)
        self.trailer.write(encode_trailer(NAMES[self.labels[index]], SLOT), index * TRAILER_BYTES)

    def read_row(self, index: int) -> bytes:
        return self.payload.read(index * SLOT, SLOT)

    def trailer_ok(self, index: int) -> str | None:
        trailer = self.trailer.read(index * TRAILER_BYTES, TRAILER_BYTES)
        return check_trailer(trailer, NAMES[self.labels[index]], SLOT)

    def intact(self) -> bool:
        return self.payload.canaries_intact() and self.trailer.canaries_intact()


def publish(store: MemoryStore, labels: dict[str, bytes]) -> None:
    """Put ``labels`` into ``store`` through an automatic driver of its own."""
    writer = Endpoint(MemoryDriver(store), list(labels))
    for i, data in enumerate(labels.values()):
        writer.fill_row(i, data)
    op = writer.driver.put(writer.rows)
    assert all(r.status is RowStatus.READY for r in writer.driver.poll(op))


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


def statuses(driver: MemoryDriver, op: MemoryOp) -> list[RowStatus]:
    return [r.status for r in driver.poll(op)]


# Bytes and ownership.


def test_put_keeps_its_own_copy_so_the_source_may_change(store: MemoryStore) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    writer.fill_row(0, b"aaaaaaaa")
    op = writer.driver.put(writer.rows)
    assert statuses(writer.driver, op) == [RowStatus.READY]
    assert writer.driver.access_ended(op)

    writer.payload.write(b"xxxxxxxx")
    assert store.committed_payload(writer.rows[0].key) == b"aaaaaaaa"

    reader = Endpoint(MemoryDriver(store), ["A"], fill=0x2D)
    got = reader.driver.get(reader.rows)
    assert statuses(reader.driver, got) == [RowStatus.READY]
    assert reader.read_row(0) == b"aaaaaaaa"
    assert reader.trailer_ok(0) is None
    assert reader.intact()


def test_get_reports_hit_hit_miss_per_row_and_leaves_the_miss_untouched(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    reader = Endpoint(MemoryDriver(store), ["A", "B", "C"], fill=0x2D)
    op = reader.driver.get(reader.rows)
    assert statuses(reader.driver, op) == [RowStatus.READY, RowStatus.READY, RowStatus.MISS]
    assert concluded(reader.driver.poll(op))
    assert reader.read_row(0) == b"aaaaaaaa"
    assert reader.read_row(1) == b"bbbbbbbb"
    assert reader.read_row(2) == b"-" * SLOT
    assert reader.trailer_ok(2) is not None
    assert reader.intact()


def test_get_rejects_a_size_mismatch_before_writing(store: MemoryStore) -> None:
    key = encode_key("t", NAMES["A"])
    assert store.commit(key, b"short")
    reader = Endpoint(MemoryDriver(store), ["A"], fill=0x2D)
    op = reader.driver.get(reader.rows)
    (result,) = reader.driver.poll(op)
    assert result.status is RowStatus.FAULT
    assert result.fault == f"object is 5 bytes, expected {SLOT + TRAILER_BYTES}"
    assert reader.read_row(0) == b"-" * SLOT
    assert reader.intact()


def test_rows_outside_a_live_registration_are_refused_before_any_effect(store: MemoryStore) -> None:
    driver = MemoryDriver(store)
    ok = Endpoint(driver, ["A"])
    stray = HostBuffer(SLOT)
    row = RowIO(ok.rows[0].key, stray.span(), ok.rows[0].trailer)
    with pytest.raises(ValueError, match="outside"):
        driver.get([row])
    assert driver.open_ops() == []
    driver.unregister(ok.handles[0])
    with pytest.raises(ValueError, match="outside"):
        driver.get(ok.rows)


# Format.


def test_trailer_round_trip_and_mismatches() -> None:
    trailer = encode_trailer(b"name", 64)
    assert len(trailer) == TRAILER_BYTES
    assert check_trailer(trailer, b"name", 64) is None
    assert check_trailer(trailer, b"other", 64) == "object name mismatch"
    assert check_trailer(trailer, b"name", 65) == "payload length 64, expected 65"
    assert check_trailer(b"\0" * TRAILER_BYTES, b"name", 64) == "bad magic"
    assert check_trailer(trailer[:-1], b"name", 64) == (
        f"trailer is {TRAILER_BYTES - 1} bytes, expected {TRAILER_BYTES}"
    )


def test_key_encoding_is_prefix_version_and_hex() -> None:
    key = encode_key("prod", b"\x01\xff")
    assert key == f"prod/v{KEY_FORMAT_VERSION}/01ff"
    for prefix in ("", "a/b", "a b"):
        with pytest.raises(ValueError):
            encode_key(prefix, b"\x01")
    with pytest.raises(ValueError):
        encode_key("prod", b"")


def test_row_refuses_overlapping_payload_and_trailer() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RowIO("k", Span(0x1000, 64), Span(0x1020, 32))


# Publication.


def test_duplicate_put_reuses_identical_object_and_rejects_different_bytes(
    store: MemoryStore,
) -> None:
    first = Endpoint(MemoryDriver(store), ["A"])
    first.fill_row(0, b"aaaaaaaa")
    assert statuses(first.driver, first.driver.put(first.rows)) == [RowStatus.READY]

    same = Endpoint(MemoryDriver(store), ["A"])
    same.fill_row(0, b"aaaaaaaa")
    assert statuses(same.driver, same.driver.put(same.rows)) == [RowStatus.READY]

    other = Endpoint(MemoryDriver(store), ["A"])
    other.fill_row(0, b"zzzzzzzz")
    (result,) = other.driver.poll(other.driver.put(other.rows))
    assert result.status is RowStatus.FAULT
    assert store.committed_payload(first.rows[0].key) == b"aaaaaaaa"
    assert len(store) == 1


def test_a_failed_retry_does_not_damage_the_committed_object(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    retry = Endpoint(driver, ["A"])
    retry.fill_row(0, b"aaaaaaaa")
    op = driver.put(retry.rows)
    driver.fail(op, "connection reset")
    (result,) = driver.poll(op)
    assert result.status is RowStatus.FAULT
    assert not driver.access_ended(op)
    assert store.committed_payload(retry.rows[0].key) == b"aaaaaaaa"


def test_readers_never_see_a_partial_object(store: MemoryStore) -> None:
    pattern_a, pattern_b = b"a" * 4096, b"b" * 4096
    keys = [f"k{i}" for i in range(200)]
    stop = threading.Event()
    seen_partial = []

    def write() -> None:
        for i, key in enumerate(keys):
            store.commit(key, pattern_a if i % 2 == 0 else pattern_b)
        stop.set()

    def read() -> None:
        while not stop.is_set():
            for key in keys:
                data = store.get(key)
                if data is not None and data not in (pattern_a, pattern_b):
                    seen_partial.append(key)

    readers = [threading.Thread(target=read) for _ in range(2)]
    writer = threading.Thread(target=write)
    for t in readers + [writer]:
        t.start()
    for t in readers + [writer]:
        t.join()
    assert seen_partial == []
    assert len(store) == len(keys)


# Result and access end are independent.


def test_result_can_come_before_access_end(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D)
    op = driver.get(reader.rows)
    assert statuses(driver, op) == [RowStatus.PENDING, RowStatus.PENDING]

    driver.complete(op, [0])
    assert statuses(driver, op) == [RowStatus.READY, RowStatus.PENDING]
    assert reader.read_row(0) == b"aaaaaaaa"
    assert reader.read_row(1) == b"-" * SLOT

    driver.fail(op, "timeout", [1])
    results = driver.poll(op)
    assert concluded(results)
    assert results[1].status is RowStatus.FAULT and results[1].fault == "timeout"
    assert not driver.access_ended(op)

    driver.end_access(op)
    assert driver.access_ended(op)
    assert driver.poll(op) == results


def test_complete_accepts_a_one_shot_iterator(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"], fill=0x2D)
    op = driver.get(reader.rows)
    driver.complete(op, iter([0]))
    assert statuses(driver, op) == [RowStatus.READY]
    assert reader.read_row(0) == b"aaaaaaaa"


def test_access_can_end_before_the_result_is_published(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"], fill=0x2D)
    op = driver.get(reader.rows)
    driver.transfer(op)
    driver.end_access(op)
    assert driver.access_ended(op)
    assert statuses(driver, op) == [RowStatus.PENDING]
    driver.publish_results(op)
    assert statuses(driver, op) == [RowStatus.READY]
    assert reader.read_row(0) == b"aaaaaaaa"


def test_a_late_write_lands_only_while_access_is_open(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D)
    op = driver.get(reader.rows)
    driver.complete(op, [0])
    driver.fail(op, "timeout", [1])
    results = driver.poll(op)

    driver.write_late(op, 1, b"bbbbbbbb")
    assert reader.read_row(1) == b"bbbbbbbb"
    assert driver.poll(op) == results
    assert not driver.access_ended(op)

    with pytest.raises(RuntimeError, match="ready, not faulted"):
        driver.write_late(op, 0, b"cccccccc")
    assert reader.read_row(0) == b"aaaaaaaa"

    driver.end_access(op)
    with pytest.raises(RuntimeError, match="ended"):
        driver.write_late(op, 1, b"cccccccc")
    with pytest.raises(RuntimeError, match="ended"):
        driver.transfer(op, [1])
    assert reader.read_row(1) == b"bbbbbbbb"
    assert reader.intact()


def test_a_cancelled_row_cannot_be_written_late(store: MemoryStore) -> None:
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"], fill=0x2D)
    op = driver.get(reader.rows)
    driver.cancel(op)
    assert statuses(driver, op) == [RowStatus.CANCELLED]
    with pytest.raises(RuntimeError, match="cancelled, not faulted"):
        driver.write_late(op, 0, b"aaaaaaaa")
    assert reader.read_row(0) == b"-" * SLOT


def test_timeout_is_a_fault_not_a_miss(store: MemoryStore) -> None:
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["C"])
    op = driver.get(reader.rows)
    driver.fail(op, "timeout")
    (result,) = driver.poll(op)
    assert result.status is RowStatus.FAULT
    assert result.fault == "timeout"


class FailingStore(MemoryStore):
    """Raises ``StoreError`` on the n-th commit, as a store that drops a connection would."""

    def __init__(self, fail_on_commit: int) -> None:
        super().__init__()
        self._fail_on = fail_on_commit
        self._commits = 0

    def commit(self, key: str, data: bytes) -> bool:
        self._commits += 1
        if self._commits == self._fail_on:
            raise StoreError("connection dropped")
        return super().commit(key, data)


def test_a_store_error_mid_operation_becomes_a_row_fault_and_keeps_the_handle() -> None:
    store = FailingStore(fail_on_commit=2)
    writer = Endpoint(MemoryDriver(store), ["A", "B", "C"])
    for i, data in enumerate((b"aaaaaaaa", b"bbbbbbbb", b"cccccccc")):
        writer.fill_row(i, data)
    op = writer.driver.put(writer.rows)
    results = writer.driver.poll(op)
    assert [r.status for r in results] == [RowStatus.READY, RowStatus.FAULT, RowStatus.READY]
    assert results[1].fault == "connection dropped"
    assert writer.driver.access_ended(op)
    assert store.committed_payload(writer.rows[0].key) == b"aaaaaaaa"
    assert not store.contains(writer.rows[1].key)
    assert store.committed_payload(writer.rows[2].key) == b"cccccccc"
    assert writer.driver.drain(0.0) is True


class GatedStore(MemoryStore):
    """Holds every commit until the test opens the gate, and says when one is waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.waiting = threading.Event()

    def commit(self, key: str, data: bytes) -> bool:
        self.waiting.set()
        self.gate.wait()
        return super().commit(key, data)


def test_queries_do_not_wait_on_another_operation_in_the_store() -> None:
    store = GatedStore()
    driver = MemoryDriver(store, controlled=True)
    done = Endpoint(driver, ["A"])
    done.fill_row(0, b"aaaaaaaa")
    finished = driver.get(done.rows)
    driver.fail(finished, "timeout")

    slow = Endpoint(driver, ["B"])
    slow.fill_row(0, b"bbbbbbbb")
    blocked = driver.put(slow.rows)
    worker = threading.Thread(target=driver.transfer, args=(blocked,), daemon=True)
    worker.start()
    answered = threading.Event()
    seen: dict[str, object] = {}
    failure: list[Exception] = []

    def query() -> None:
        # Runs off the main thread: if a query waits on the store, the main thread can still
        # open the gate and the test fails instead of hanging.
        try:
            seen["finished"] = statuses(driver, finished)
            seen["blocked"] = statuses(driver, blocked)
            seen["access_ended"] = driver.access_ended(blocked)
            try:
                driver.end_access(blocked)
            except RuntimeError as error:
                seen["end_access"] = str(error)
            driver.cancel(blocked)
            seen["after_cancel"] = statuses(driver, blocked)
        except Exception as error:  # surfaced on the main thread below
            failure.append(error)
        finally:
            answered.set()

    querier = threading.Thread(target=query, daemon=True)
    try:
        assert store.waiting.wait(5.0), "the worker never reached the store"
        # The worker is inside the store with its row taken; these must answer at once.
        querier.start()
        assert answered.wait(5.0), "queries waited on an operation that is inside the store"
    finally:
        store.gate.set()
        worker.join(5.0)
        if querier.ident is not None:  # started; joining an unstarted thread raises
            querier.join(5.0)
    assert not worker.is_alive() and not querier.is_alive()
    if failure:
        raise AssertionError("a query raised") from failure[0]
    assert seen["finished"] == [RowStatus.FAULT]
    assert seen["blocked"] == [RowStatus.PENDING]
    assert seen["access_ended"] is False
    assert seen.get("end_access") == "a transfer is still in progress"
    assert seen["after_cancel"] == [RowStatus.PENDING]
    driver.publish_results(blocked)
    assert statuses(driver, blocked) == [RowStatus.READY]
    assert store.committed_payload(slow.rows[0].key) == b"bbbbbbbb"


# Cancellation, probe, drain.


def test_cancel_concludes_unstarted_rows_and_spares_started_ones(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A", "B"], fill=0x2D)
    op = driver.get(reader.rows)
    driver.transfer(op, [0])
    driver.cancel(op)
    assert statuses(driver, op) == [RowStatus.PENDING, RowStatus.CANCELLED]
    driver.publish_results(op, [0])
    assert statuses(driver, op) == [RowStatus.READY, RowStatus.CANCELLED]
    assert reader.read_row(1) == b"-" * SLOT
    assert not driver.access_ended(op)


def test_an_ineffective_cancel_changes_nothing(store: MemoryStore) -> None:
    driver = MemoryDriver(store, controlled=True)
    driver.set_cancel_effective(False)
    reader = Endpoint(driver, ["A"])
    op = driver.get(reader.rows)
    driver.cancel(op)
    assert statuses(driver, op) == [RowStatus.PENDING]


def test_probe_answers_hit_miss_or_unknown(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store)
    keys = [encode_key("t", NAMES["A"]), encode_key("t", NAMES["C"])]
    assert driver.probe(keys, 1.0) == (ProbeStatus.HIT, ProbeStatus.MISS)
    driver.set_probe_unknown(True)
    assert driver.probe(keys, 1.0) == (ProbeStatus.UNKNOWN, ProbeStatus.UNKNOWN)


def test_drain_waits_for_access_end_and_refuses_new_work(store: MemoryStore) -> None:
    publish(store, {"A": b"aaaaaaaa"})
    driver = MemoryDriver(store, controlled=True)
    reader = Endpoint(driver, ["A"])
    op = driver.get(reader.rows)
    assert driver.drain(0.01) is False
    assert statuses(driver, op) == [RowStatus.CANCELLED]
    with pytest.raises(ValueError, match="draining"):
        driver.get(reader.rows)

    release = threading.Timer(0.05, driver.end_access, args=(op,))
    release.start()
    try:
        assert driver.drain(5.0) is True
    finally:
        release.join()
    assert driver.open_ops() == []


# Registration.


def test_register_checks_geometry_and_overlap_and_unregister_is_idempotent(
    store: MemoryStore,
) -> None:
    driver = MemoryDriver(store)
    buffer = HostBuffer(64)
    with pytest.raises(ValueError, match="layout needs"):
        driver.register(buffer.address, 64, RegionLayout(SLOT, 9))
    handle = driver.register(buffer.address, 64, RegionLayout(SLOT, 8))
    with pytest.raises(ValueError, match="overlaps"):
        driver.register(buffer.address + 8, 8, RegionLayout(8, 1))
    driver.unregister(handle)
    driver.unregister(handle)
    assert driver.live_registrations() == []
    driver.register(buffer.address + 8, 8, RegionLayout(8, 1))


def test_a_registration_the_store_may_hold_keeps_its_handle(store: MemoryStore) -> None:
    driver = MemoryDriver(store)
    buffer = HostBuffer(64)
    driver.fail_next_register("store unreachable", escaped=False)
    with pytest.raises(RegistrationError) as clean:
        driver.register(buffer.address, 64, RegionLayout(SLOT, 8))
    assert clean.value.handle is None
    assert driver.live_registrations() == []

    driver.fail_next_register("timed out after the request", escaped=True)
    with pytest.raises(RegistrationError) as ambiguous:
        driver.register(buffer.address, 64, RegionLayout(SLOT, 8))
    handle = ambiguous.value.handle
    assert handle is not None
    assert driver.live_registrations() == [handle]

    driver.fail_next_unregister(handle, "busy")
    with pytest.raises(StoreError, match="busy"):
        driver.unregister(handle)
    assert driver.live_registrations() == [handle]
    driver.unregister(handle)
    assert driver.live_registrations() == []


def test_memory_driver_has_every_member_of_the_driver_protocol(store: MemoryStore) -> None:
    driver: StoreDriver = MemoryDriver(store)
    assert isinstance(driver, StoreDriver)


def test_automatic_mode_follows_the_same_protocol(store: MemoryStore) -> None:
    writer = Endpoint(MemoryDriver(store), ["A"])
    writer.fill_row(0, b"aaaaaaaa")
    op = writer.driver.put(writer.rows)
    assert concluded(writer.driver.poll(op))
    assert writer.driver.access_ended(op)
    assert writer.driver.drain(0.0) is True
    assert writer.driver.open_ops() == []
