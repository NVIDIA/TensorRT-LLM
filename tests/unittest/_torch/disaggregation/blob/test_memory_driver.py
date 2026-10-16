# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-memory driver: the shared conformance checks, its controlled mode, and proof that every
check fails against a driver that breaks exactly the rule it guards."""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable, Sequence

import pytest
from driver_conformance import (
    SLOT,
    UNRELEASED_ENDPOINTS,
    DriverConformance,
    Endpoint,
    Harness,
    key_of,
    statuses,
)
from host_memory import HostBuffer
from memory_store import (
    DEFAULT_CAPABILITIES,
    MAX_PROBES_IN_FLIGHT,
    Kind,
    MemoryDriver,
    MemoryOp,
    MemoryStore,
    StoredObject,
)
from object_trailer import encode_trailer

from tensorrt_llm._torch.disaggregation.blob.store import (
    MISS,
    DriverCapabilities,
    DriverRejected,
    Hint,
    OpHandle,
    ProbeStatus,
    RegionHandle,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowResult,
    RowStatus,
    StoreDriver,
    StoreError,
    concluded,
    faulted,
)

pytestmark = pytest.mark.cpu_only

KEY_A, KEY_B, KEY_C = key_of("A"), key_of("B"), key_of("C")
TIMEOUT = 5.0


# Stores that misbehave on purpose.


class FailingStore(MemoryStore):
    """Raises ``StoreError`` on the n-th commit, as a store that drops a connection would."""

    def __init__(self, fail_on_commit: int) -> None:
        super().__init__()
        self._fail_on = fail_on_commit
        self._commits = 0

    def commit(self, key: str, payload: bytes, trailer: bytes):  # type: ignore[no-untyped-def]
        self._commits += 1
        if self._commits == self._fail_on:
            raise StoreError("connection dropped")
        return super().commit(key, payload, trailer)


class TimeoutStore(MemoryStore):
    """Reports a timeout for reads of ``key``, as an SDK whose wait expired would."""

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    def get(self, key: str) -> StoredObject | None:
        if key == self._key:
            raise StoreError("timeout")
        return super().get(key)


class CrashingStore(MemoryStore):
    """Raises something that is not a ``StoreError`` from inside a commit, as a binding bug would."""

    def commit(self, key: str, payload: bytes, trailer: bytes):  # type: ignore[no-untyped-def]
        raise RuntimeError("binding returned garbage")


class GatedStore(MemoryStore):
    """Holds reads and writes of ``held`` keys (every key when None) while the gate is closed,
    and says when one is waiting."""

    def __init__(self, held: set[str] | None = None) -> None:
        super().__init__()
        self.held = held
        self.gate = threading.Event()
        self.gate.set()
        self.waiting = threading.Event()

    def _hold(self, key: str) -> None:
        if (self.held is None or key in self.held) and not self.gate.is_set():
            self.waiting.set()
            self.gate.wait()

    def commit(self, key: str, payload: bytes, trailer: bytes):  # type: ignore[no-untyped-def]
        self._hold(key)
        return super().commit(key, payload, trailer)

    def get(self, key: str) -> StoredObject | None:
        self._hold(key)
        return super().get(key)

    def presence(self, key: str) -> ProbeStatus:
        self._hold(key)
        return super().presence(key)


def _run(target: Callable[..., object], *args: object) -> threading.Thread:
    worker = threading.Thread(target=target, args=args, daemon=True)
    worker.start()
    return worker


class _HintFreeSignatures(MemoryDriver):
    """A driver written before hints existed: ``get`` and ``probe`` take no hint. With
    ``hints`` undeclared the backend never passes one, so it stays a valid driver."""

    def get(self, rows: Sequence[RowIO]) -> MemoryOp:  # type: ignore[override]
        return super().get(rows)

    def probe(self, keys: Sequence[str], budget_s: float) -> tuple[ProbeStatus, ...]:  # type: ignore[override]
        return super().probe(keys, budget_s)


class TestMemoryDriverConformance(DriverConformance):
    driver_cls: type[MemoryDriver] = MemoryDriver
    capabilities = DEFAULT_CAPABILITIES

    def make_driver(self) -> StoreDriver:
        return self.driver_cls(MemoryStore(), capabilities=self.capabilities)

    def make_blocked_probe(self) -> tuple[StoreDriver, str, Callable[[], None]]:
        store = GatedStore(held={KEY_A})
        store.gate.clear()
        return self.driver_cls(store, capabilities=self.capabilities), KEY_A, store.gate.set


class TestHintFreeDriverConformance(TestMemoryDriverConformance):
    driver_cls = _HintFreeSignatures


class TestMemoryDriver:
    """Behaviour only the in-memory driver's controls or stores can stage."""

    driver_cls: type[MemoryDriver] = MemoryDriver

    def make(self, store: MemoryStore | None = None, **kwargs: object) -> MemoryDriver:
        store = store if store is not None else MemoryStore()
        return self.driver_cls(store, **kwargs)  # type: ignore[arg-type]

    def _publish(
        self, driver: MemoryDriver, rows: dict[str, bytes], *, keep_registered: bool = False
    ) -> Endpoint:
        writer = Endpoint(driver, list(rows))
        for i, data in enumerate(rows.values()):
            writer.fill_row(i, data)
        op = driver.put(writer.rows())
        assert statuses(driver, op) == [RowStatus.READY] * len(rows)
        if not keep_registered:
            writer.unregister()
        return writer

    def _shutdown(self, driver: MemoryDriver) -> None:
        assert driver.drain(TIMEOUT) is True
        for registration in driver.live_registrations():
            driver.unregister(registration)
        driver.close()

    # The store and integrity.

    def test_put_keeps_a_private_copy_with_a_trailer(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa"})
        stored = store.committed_object(KEY_A)
        assert stored.payload == b"aaaaaaaa" and stored.trailer
        assert len(store) == 1

    def test_a_corrupted_object_is_a_fault_and_never_lands(self) -> None:
        store = MemoryStore()
        driver = self.make(store)
        self._publish(driver, {"A": b"aaaaaaaa"})
        store.corrupt(KEY_A, 3)
        reader = Endpoint(driver, ["A"], fill=0x2D)
        (result,) = driver.poll(driver.get(reader.rows()))
        assert result.status is RowStatus.FAULT
        assert result.fault == "payload checksum mismatch"
        assert reader.read_row(0) == b"-" * SLOT

    def test_a_different_object_under_the_same_key_is_diagnosed(self) -> None:
        store = MemoryStore()
        driver = self.make(store)
        self._publish(driver, {"A": b"aaaaaaaa"})
        other = Endpoint(driver, ["A"])
        other.fill_row(0, b"zzzzzzzz")
        (result,) = driver.poll(driver.put(other.rows()))
        assert result.status is RowStatus.FAULT
        assert store.committed_payload(KEY_A) == b"aaaaaaaa"
        assert len(store) == 1

    def test_a_retry_that_fails_in_the_store_leaves_the_committed_object_whole(self) -> None:
        store = FailingStore(fail_on_commit=2)
        driver = self.make(store)
        self._publish(driver, {"A": b"aaaaaaaa"})
        retry = Endpoint(driver, ["A"])
        retry.fill_row(0, b"aaaaaaaa")
        (result,) = driver.poll(driver.put(retry.rows()))
        assert result.status is RowStatus.FAULT and result.fault == "connection dropped"
        reader = Endpoint(driver, ["A"], fill=0x2D)
        (got,) = driver.poll(driver.get(reader.rows()))
        assert got.status is RowStatus.READY
        assert reader.read_row(0) == b"aaaaaaaa"

    def test_readers_never_see_a_partial_object(self) -> None:
        store = MemoryStore()
        pattern_a, pattern_b = b"a" * 4096, b"b" * 4096
        keys = [f"k{i}" for i in range(200)]
        stop = threading.Event()
        seen_partial = []

        def write() -> None:
            for i, key in enumerate(keys):
                store.commit(key, pattern_a if i % 2 == 0 else pattern_b, b"t")
            stop.set()

        def read() -> None:
            while not stop.is_set():
                for key in keys:
                    stored = store.get(key)
                    if stored is not None and stored.payload not in (pattern_a, pattern_b):
                        seen_partial.append(key)

        workers = [_run(read), _run(read), _run(write)]
        for worker in workers:
            worker.join()
        assert seen_partial == []
        assert len(store) == len(keys)

    # Statuses that are not faults.

    def test_a_key_held_by_an_unfinished_write_is_not_ready_for_get_put_and_probe(self) -> None:
        store = MemoryStore()
        driver = self.make(store)
        store.begin_write(KEY_A)
        reader = Endpoint(driver, ["A", "C"], fill=0x2D)
        results = driver.poll(driver.get(reader.rows()))
        assert [r.status for r in results] == [RowStatus.NOT_READY, RowStatus.MISS]
        assert reader.read_row(0) == b"-" * SLOT
        writer = Endpoint(driver, ["A"])
        writer.fill_row(0, b"aaaaaaaa")
        assert statuses(driver, driver.put(writer.rows())) == [RowStatus.NOT_READY]
        assert driver.probe([KEY_A, KEY_C], TIMEOUT) == (ProbeStatus.NOT_READY, ProbeStatus.MISS)
        store.end_write(KEY_A)
        assert statuses(driver, driver.put(writer.rows())) == [RowStatus.READY]
        assert statuses(driver, driver.get(reader.rows())) == [RowStatus.READY, RowStatus.MISS]
        assert reader.read_row(0) == b"aaaaaaaa"

    def test_a_row_the_store_has_no_room_for_is_declined_not_faulted(self) -> None:
        driver = self.make(MemoryStore(capacity=1))
        writer = Endpoint(driver, ["A", "B"])
        writer.fill_row(0, b"aaaaaaaa")
        writer.fill_row(1, b"bbbbbbbb")
        op = driver.put(writer.rows())
        assert statuses(driver, op) == [RowStatus.READY, RowStatus.DECLINED]
        assert driver.access_ended(op)
        reader = Endpoint(driver, ["A", "B"], fill=0x2D)
        assert statuses(driver, driver.get(reader.rows())) == [RowStatus.READY, RowStatus.MISS]

    # Result and access end are independent.

    def test_result_can_come_before_access_end(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
        driver = self.make(store, controlled=True)
        reader = Endpoint(driver, ["A", "B"], fill=0x2D)
        op = driver.get(reader.rows())
        assert statuses(driver, op) == [RowStatus.PENDING, RowStatus.PENDING]

        driver.complete(op, [0])
        assert statuses(driver, op) == [RowStatus.READY, RowStatus.PENDING]
        assert reader.read_row(0) == b"aaaaaaaa"
        assert reader.read_row(1) == b"-" * SLOT

        driver.fail(op, "timeout", [1])
        results = driver.poll(op)
        assert concluded(results)
        assert results[1] == faulted("timeout")
        assert not driver.access_ended(op)

        driver.end_access(op)
        assert driver.access_ended(op)
        assert driver.poll(op) == results

    def test_complete_accepts_a_one_shot_iterator(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa"})
        driver = self.make(store, controlled=True)
        reader = Endpoint(driver, ["A"], fill=0x2D)
        op = driver.get(reader.rows())
        driver.complete(op, iter([0]))
        assert statuses(driver, op) == [RowStatus.READY]
        assert reader.read_row(0) == b"aaaaaaaa"

    def test_access_can_end_before_the_result_is_published(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa"})
        driver = self.make(store, controlled=True)
        reader = Endpoint(driver, ["A"], fill=0x2D)
        op = driver.get(reader.rows())
        driver.transfer(op)
        driver.end_access(op)
        assert driver.access_ended(op)
        assert statuses(driver, op) == [RowStatus.PENDING]
        driver.publish_results(op)
        assert statuses(driver, op) == [RowStatus.READY]
        assert reader.read_row(0) == b"aaaaaaaa"

    def test_a_late_write_lands_only_while_access_is_open(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
        driver = self.make(store, controlled=True)
        reader = Endpoint(driver, ["A", "B"], fill=0x2D)
        op = driver.get(reader.rows())
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

    def test_a_cancelled_row_cannot_be_written_late(self) -> None:
        driver = self.make(controlled=True)
        reader = Endpoint(driver, ["A"], fill=0x2D)
        op = driver.get(reader.rows())
        driver.cancel(op)
        assert statuses(driver, op) == [RowStatus.CANCELLED]
        with pytest.raises(RuntimeError):
            driver.write_late(op, 0, b"aaaaaaaa")
        assert reader.read_row(0) == b"-" * SLOT

    # Store errors.

    def test_a_store_timeout_is_a_fault_not_a_miss(self) -> None:
        driver = self.make(TimeoutStore(KEY_A))
        reader = Endpoint(driver, ["A", "C"], fill=0x2D)
        results = driver.poll(driver.get(reader.rows()))
        assert [r.status for r in results] == [RowStatus.FAULT, RowStatus.MISS]
        assert results[0].fault == "timeout"
        assert reader.read_row(0) == b"-" * SLOT

    def test_a_store_error_mid_operation_becomes_a_row_fault_and_keeps_the_handle(self) -> None:
        store = FailingStore(fail_on_commit=2)
        driver = self.make(store)
        writer = Endpoint(driver, ["A", "B", "C"])
        for i, data in enumerate((b"aaaaaaaa", b"bbbbbbbb", b"cccccccc")):
            writer.fill_row(i, data)
        op = driver.put(writer.rows())
        results = driver.poll(op)
        assert [r.status for r in results] == [RowStatus.READY, RowStatus.FAULT, RowStatus.READY]
        assert results[1].fault == "connection dropped"
        assert driver.access_ended(op)
        assert store.committed_payload(KEY_A) == b"aaaaaaaa"
        assert not store.contains(KEY_B)
        assert store.committed_payload(KEY_C) == b"cccccccc"

    def test_a_raise_after_the_operation_is_recorded_becomes_a_row_fault(self) -> None:
        driver = self.make(CrashingStore())
        writer = Endpoint(driver, ["A"])
        writer.fill_row(0, b"aaaaaaaa")
        try:
            op = driver.put(writer.rows())
        except Exception as error:  # noqa: BLE001 - the check is that nothing escapes
            pytest.fail(f"put raised after the operation was recorded: {error!r}")
        (result,) = driver.poll(op)
        assert result.status is RowStatus.FAULT
        assert result.fault == "RuntimeError: binding returned garbage"
        assert driver.access_ended(op)
        assert driver.drain(TIMEOUT) is True

    # Concurrency.

    def test_queries_do_not_wait_on_an_operation_inside_the_store(self) -> None:
        store = GatedStore()
        driver = self.make(store, controlled=True)
        done = Endpoint(driver, ["A"])
        finished = driver.get(done.rows())
        driver.fail(finished, "timeout")

        slow = Endpoint(driver, ["B"])
        slow.fill_row(0, b"bbbbbbbb")
        blocked = driver.put(slow.rows())
        store.gate.clear()
        worker = _run(driver.transfer, blocked)
        answered = threading.Event()
        seen: dict[str, object] = {}
        failure: list[Exception] = []

        def query() -> None:
            # Off the main thread: if a query waits on the store, the main thread can still open
            # the gate and the test fails instead of hanging.
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

        querier = None
        try:
            assert store.waiting.wait(TIMEOUT), "the worker never reached the store"
            querier = _run(query)
            assert answered.wait(TIMEOUT), "queries waited on an operation inside the store"
        finally:
            store.gate.set()
            worker.join(TIMEOUT)
            if querier is not None:
                querier.join(TIMEOUT)
        assert not worker.is_alive()
        if failure:
            raise AssertionError("a query raised") from failure[0]
        assert seen["finished"] == [RowStatus.FAULT]
        assert seen["blocked"] == [RowStatus.PENDING]
        assert seen["access_ended"] is False
        assert seen.get("end_access") == "a transfer is still in progress"
        assert seen["after_cancel"] == [RowStatus.PENDING]
        driver.publish_results(blocked)
        assert statuses(driver, blocked) == [RowStatus.READY]
        assert store.committed_payload(KEY_B) == b"bbbbbbbb"

    def test_a_blocked_get_does_not_delay_another_get(self) -> None:
        store = GatedStore(held={KEY_A})
        driver = self.make(store)
        self._publish(driver, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
        slow = Endpoint(driver, ["A"], fill=0x2D)
        quick = Endpoint(driver, ["B"], fill=0x2D)
        store.gate.clear()
        outcome: dict[str, list[RowStatus]] = {}

        def fetch(name: str, endpoint: Endpoint) -> None:
            outcome[name] = statuses(driver, driver.get(endpoint.rows()))

        slow_worker = _run(fetch, "slow", slow)
        try:
            assert store.waiting.wait(TIMEOUT), "the slow get never reached the store"
            quick_worker = _run(fetch, "quick", quick)
            quick_worker.join(TIMEOUT)
            assert not quick_worker.is_alive(), "a get waited for another get blocked in the store"
            assert outcome["quick"] == [RowStatus.READY]
            assert quick.read_row(0) == b"bbbbbbbb"
            assert "slow" not in outcome
        finally:
            store.gate.set()
            slow_worker.join(TIMEOUT)
        assert outcome["slow"] == [RowStatus.READY]

    # Cancel.

    def test_cancel_concludes_unstarted_rows_and_spares_started_ones(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
        driver = self.make(store, controlled=True)
        reader = Endpoint(driver, ["A", "B"], fill=0x2D)
        op = driver.get(reader.rows())
        driver.transfer(op, [0])
        driver.cancel(op)
        assert statuses(driver, op) == [RowStatus.PENDING, RowStatus.CANCELLED]
        driver.publish_results(op, [0])
        assert statuses(driver, op) == [RowStatus.READY, RowStatus.CANCELLED]
        assert reader.read_row(1) == b"-" * SLOT
        assert not driver.access_ended(op)

    def test_cancelling_everything_before_it_started_ends_access(self) -> None:
        driver = self.make(controlled=True)
        reader = Endpoint(driver, ["A", "B"])
        op = driver.get(reader.rows())
        driver.cancel(op)
        assert statuses(driver, op) == [RowStatus.CANCELLED, RowStatus.CANCELLED]
        assert driver.access_ended(op)

    def test_an_ineffective_cancel_changes_nothing(self) -> None:
        driver = self.make(controlled=True)
        driver.set_cancel_effective(False)
        reader = Endpoint(driver, ["A"])
        op = driver.get(reader.rows())
        driver.cancel(op)
        assert statuses(driver, op) == [RowStatus.PENDING]
        assert not driver.access_ended(op)

    # Probe.

    def test_probe_answers_unknown_when_the_store_does_not_answer_in_time(self) -> None:
        store = GatedStore(held={KEY_A})
        driver = self.make(store)
        self._publish(driver, {"B": b"bbbbbbbb"})
        keys = [KEY_B, KEY_A, KEY_C]
        store.gate.clear()
        answer: list[tuple[ProbeStatus, ...]] = []
        elapsed: list[float] = []

        def probe() -> None:
            started = time.monotonic()
            answer.append(driver.probe(keys, 0.05))
            elapsed.append(time.monotonic() - started)

        prober = _run(probe)
        prober.join(1.0)
        try:
            assert not prober.is_alive(), "probe waited for the store past its budget"
            # Keys asked before the blocked one were answered; the rest are unknown, not MISS.
            assert answer == [(ProbeStatus.HIT, ProbeStatus.UNKNOWN, ProbeStatus.UNKNOWN)]
            assert elapsed[0] < 0.5
            assert driver.probes_in_flight() == 1
            assert driver.drain(0.05) is False  # the probe is still inside the store
        finally:
            store.gate.set()
            prober.join(TIMEOUT)
            driver.join_probes(TIMEOUT)
        assert driver.probe(keys, TIMEOUT) == (ProbeStatus.UNKNOWN,) * 3  # draining now
        self._shutdown(driver)

    def test_probes_in_flight_are_capped(self) -> None:
        store = GatedStore(held={KEY_A})
        driver = self.make(store)
        store.gate.clear()
        try:
            for _ in range(MAX_PROBES_IN_FLIGHT):
                assert driver.probe([KEY_A], 0.0) == (ProbeStatus.UNKNOWN,)
            assert driver.probes_in_flight() == MAX_PROBES_IN_FLIGHT
            started = time.monotonic()
            assert driver.probe([KEY_A], 0.2) == (ProbeStatus.UNKNOWN,)
            assert time.monotonic() - started < 0.1  # answered at once, no new worker
            assert driver.probes_in_flight() == MAX_PROBES_IN_FLIGHT
        finally:
            store.gate.set()
            driver.join_probes(TIMEOUT)
        self._shutdown(driver)
        assert driver.probes_in_flight() == 0

    def test_memory_driver_rejects_a_hint_it_cannot_read_before_any_effect(self) -> None:
        """The backend rejects such hints before dispatch; this driver also refuses them itself,
        so a backend bug cannot make it read with a hint it silently dropped."""
        driver = self.make()
        self._publish(driver, {"A": b"aaaaaaaa"})
        reader = Endpoint(driver, ["A"], fill=0x2D)
        with pytest.raises(DriverRejected):
            driver.get(reader.rows(), {"source": "worker-7"})
        assert reader.read_row(0) == b"-" * SLOT
        with pytest.raises(DriverRejected):
            driver.probe([KEY_A], TIMEOUT, {"source": "worker-7"})
        assert statuses(driver, driver.get(reader.rows())) == [RowStatus.READY]
        assert driver.hints_seen == []

    def test_a_hint_reaches_a_driver_that_declares_hints_unchanged(self) -> None:
        hinted = DriverCapabilities(
            get=True, put=True, cancel=True, probe=True, max_rows_per_op=None, hints=True
        )
        driver = self.make(capabilities=hinted)
        self._publish(driver, {"A": b"aaaaaaaa"})
        reader = Endpoint(driver, ["A"], fill=0x2D)
        hint = {"source": "worker-7"}
        op = driver.get(reader.rows(), hint)
        assert statuses(driver, op) == [RowStatus.READY]
        assert reader.read_row(0) == b"aaaaaaaa"
        assert driver.probe([KEY_A], TIMEOUT, hint) == (ProbeStatus.HIT,)
        assert driver.hints_seen == [hint, hint]
        assert all(seen is hint for seen in driver.hints_seen)  # the same object, not a copy
        assert statuses(driver, driver.get(reader.rows())) == [RowStatus.READY]
        assert len(driver.hints_seen) == 2  # no hint, nothing recorded

    def test_probe_unknown_switch(self) -> None:
        driver = self.make()
        self._publish(driver, {"A": b"aaaaaaaa"})
        driver.set_probe_unknown(True)
        assert driver.probe([KEY_A, KEY_C], TIMEOUT) == (ProbeStatus.UNKNOWN,) * 2

    def test_a_drain_racing_a_submission_leaves_a_cancelled_operation_with_its_handle(
        self,
    ) -> None:
        store = MemoryStore()
        racing = type(
            "Racing", (self.driver_cls,), {"_before_execute": lambda self, op: self.drain(0.0)}
        )
        driver = racing(store)
        writer = Endpoint(driver, ["A"])
        writer.fill_row(0, b"aaaaaaaa")
        try:
            op = driver.put(writer.rows())
        except Exception as error:  # noqa: BLE001 - the check is that the handle comes back
            pytest.fail(f"put raised instead of returning its handle: {error!r}")
        assert statuses(driver, op) == [RowStatus.CANCELLED]
        assert driver.access_ended(op)
        assert len(store) == 0
        assert driver.drain(TIMEOUT) is True

    def test_an_object_shorter_than_its_trailer_claims_is_a_fault_and_never_lands(self) -> None:
        store = MemoryStore()
        driver = self.make(store)
        store.overwrite(KEY_A, b"aaaa", encode_trailer(KEY_A.encode(), SLOT))  # claims 8 bytes
        reader = Endpoint(driver, ["A"], fill=0x2D)
        (result,) = driver.poll(driver.get(reader.rows()))
        assert result.status is RowStatus.FAULT
        assert result.fault == f"object is 4 bytes, expected {SLOT}"
        assert reader.read_row(0) == b"-" * SLOT

    # The conformance harness.

    def test_harness_keeps_a_buffer_whose_registration_escaped(self) -> None:
        driver = self.make()
        h = Harness(driver, TIMEOUT)
        driver.fail_next_register("timed out after the request", escaped=True)
        with pytest.raises(RegistrationError):
            h.endpoint(["A"])
        assert len(h.endpoints) == 1 and h.endpoints[0].handle is not None
        assert h.cleanup() == []
        assert driver.live_registrations() == []

    def test_harness_retains_an_endpoint_it_could_not_unregister_and_retries(self) -> None:
        driver = self.make()
        h = Harness(driver, TIMEOUT)
        endpoint = h.endpoint(["A"])
        assert endpoint.handle is not None
        driver.fail_next_unregister(endpoint.handle, "busy")
        problems = h.cleanup()
        assert problems and not h.done
        assert endpoint in UNRELEASED_ENDPOINTS
        assert driver.live_registrations() == [endpoint.handle]
        assert h.cleanup() == [] and h.done
        assert driver.live_registrations() == []

    # Registration and shutdown.

    def test_unregister_is_refused_while_an_operation_may_access_the_region(self) -> None:
        driver = self.make(controlled=True)
        reader = Endpoint(driver, ["A"])
        other = Endpoint(driver, ["B"])
        op = driver.get(reader.rows())
        with pytest.raises(DriverRejected):
            reader.unregister()
        assert reader.handle in driver.live_registrations()
        other.unregister()  # another region is not held by this operation
        driver.transfer(op)
        driver.publish_results(op)
        with pytest.raises(DriverRejected):
            reader.unregister()  # a result is not access-end evidence
        driver.end_access(op)
        reader.unregister()
        assert driver.live_registrations() == []

    def test_drain_waits_for_a_started_transfer(self) -> None:
        store = GatedStore(held={KEY_A})
        driver = self.make(store)
        self._publish(driver, {"A": b"aaaaaaaa"})
        reader = Endpoint(driver, ["A"], fill=0x2D)
        store.gate.clear()
        worker = _run(driver.get, reader.rows())
        try:
            assert store.waiting.wait(TIMEOUT), "the get never reached the store"
            started = time.monotonic()
            assert driver.drain(0.05) is False
            assert time.monotonic() - started < 1.0
            with pytest.raises(DriverRejected, match="draining"):
                driver.get(reader.rows())
        finally:
            store.gate.set()
            worker.join(TIMEOUT)
        assert driver.drain(TIMEOUT) is True
        assert reader.read_row(0) == b"aaaaaaaa"

    def test_an_unprovable_operation_lets_drain_return_false_at_once(self) -> None:
        driver = self.make(controlled=True)
        reader = Endpoint(driver, ["A"])
        op = driver.get(reader.rows())
        driver.transfer(op)
        driver.publish_results(op)
        driver.mark_unprovable(op)
        assert driver.access_unprovable(op)
        assert not driver.access_ended(op)

        started = time.monotonic()
        assert driver.drain(TIMEOUT) is False
        assert time.monotonic() - started < 1.0
        with pytest.raises(DriverRejected):
            reader.unregister()
        with pytest.raises(DriverRejected):
            driver.close()

        driver.end_access(op)  # later evidence, for example from the store's own teardown
        assert driver.drain(TIMEOUT) is True
        reader.unregister()
        driver.close()

    def test_drain_returns_false_at_once_even_while_another_operation_is_still_provable(
        self,
    ) -> None:
        driver = self.make(controlled=True)
        lost = Endpoint(driver, ["A"])
        pending = Endpoint(driver, ["B"])
        unprovable = driver.get(lost.rows())
        driver.transfer(unprovable)
        driver.publish_results(unprovable)
        driver.mark_unprovable(unprovable)
        provable = driver.get(pending.rows())
        driver.transfer(provable)  # started, so drain cannot cancel it; its end is still to come

        started = time.monotonic()
        assert driver.drain(1.0) is False
        assert time.monotonic() - started < 0.5
        driver.end_access(provable)
        assert driver.drain(0.05) is False  # the unprovable one still stands
        driver.end_access(unprovable)
        assert driver.drain(TIMEOUT) is True

    def test_register_checks_geometry_and_overlap(self) -> None:
        driver = self.make()
        buffer = HostBuffer(64)
        with pytest.raises(DriverRejected, match="layout needs"):
            driver.register(buffer.address, 64, RegionLayout(SLOT, 9))
        with pytest.raises(DriverRejected):
            driver.register(buffer.address, 0, RegionLayout(SLOT, 1))
        handle = driver.register(buffer.address, 64, RegionLayout(SLOT, 8))
        with pytest.raises(DriverRejected, match="overlaps"):
            driver.register(buffer.address + 8, 8, RegionLayout(8, 1))
        driver.unregister(handle)
        assert driver.live_registrations() == []

    def test_a_registration_the_store_may_hold_keeps_its_handle(self) -> None:
        driver = self.make()
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

    def test_calls_after_close_are_programming_errors(self) -> None:
        driver = self.make()
        writer = self._publish(driver, {"A": b"aaaaaaaa"}, keep_registered=True)
        op = driver.get(writer.rows())
        assert driver.drain(TIMEOUT) is True
        writer.unregister()
        driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            driver.poll(op)
        with pytest.raises(RuntimeError, match="closed"):
            driver.drain(0.0)
        with pytest.raises(RuntimeError, match="closed"):
            driver.probe([KEY_A], 0.0)
        with pytest.raises(DriverRejected, match="closed"):
            driver.register(writer.payload.address, SLOT, RegionLayout(SLOT, 1))

    def test_memory_driver_is_a_store_driver(self) -> None:
        driver: StoreDriver = self.make()
        assert isinstance(driver, StoreDriver)


# Drivers that break exactly one rule. Each proves that the check guarding the rule fails
# against it, so a weakened check is caught here before a real driver meets it.


class _AcceptsDuplicateKeys(MemoryDriver):
    def _check_unique_keys(self, rows: Sequence[RowIO]) -> None:
        pass


class _TouchesMemoryBeforeChecking(MemoryDriver):
    def get(self, rows: Sequence[RowIO], hint: Hint = None) -> MemoryOp:
        for io in rows:
            ctypes.memset(io.payload.address, 0, io.payload.nbytes)
        return super().get(rows, hint)


class _MissTouchesTheDestination(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        result = super()._get_row(io)
        if result.status is RowStatus.MISS:
            ctypes.memset(io.payload.address, 0, io.payload.nbytes)
        return result


class _CallsAWrongSizeAMiss(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        result = super()._get_row(io)
        return MISS if result.status is RowStatus.FAULT else result


class _OverwritesExistingObjects(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        payload = ctypes.string_at(io.payload.address, io.payload.nbytes)
        self.store.overwrite(io.key, payload, self._trailer_for(io, payload))
        return super()._put_row(io)


class _IgnoresTheRowLimit(MemoryDriver):
    def _check_row_limit(self, rows: Sequence[RowIO]) -> None:
        if not rows:
            raise DriverRejected("an operation needs at least one row")


class _IgnoresTheDirection(MemoryDriver):
    def _check_direction(self, kind: Kind) -> None:
        pass


class _NeverEndsAccess(MemoryDriver):
    def _execute(self, op: MemoryOp) -> None:
        super()._execute(op)
        with self._lock:
            op.access_ended = False  # the evidence is lost again


class _ChangesItsCapabilities(MemoryDriver):
    _flip = False

    @property
    def capabilities(self) -> DriverCapabilities:
        self._flip = not self._flip
        return DriverCapabilities(
            get=True, put=True, cancel=self._flip, probe=True, max_rows_per_op=None
        )


class _ProbeTakesItsTime(MemoryDriver):
    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        time.sleep(budget_s + 0.8)
        return super().probe(keys, budget_s, hint)


class _KeepsAcceptingWhileDraining(MemoryDriver):
    def _check_not_draining(self) -> None:
        pass


class _ClosesAnyTime(MemoryDriver):
    def _check_closable(self) -> None:
        pass


class _UnregisterKeepsTheRange(MemoryDriver):
    def unregister(self, handle: RegionHandle) -> None:
        pass


class _UnregistersBusyRegions(MemoryDriver):
    def _check_region_idle(self, registration: object) -> None:
        pass


class _CallsATimeoutAMiss(MemoryDriver):
    def _move(self, kind: Kind, io: RowIO) -> RowResult:
        try:
            return super()._move(kind, io)
        finally:
            pass

    def _get_row(self, io: RowIO) -> RowResult:
        try:
            return super()._get_row(io)
        except StoreError:
            return MISS


class _LetsRaisesEscape(MemoryDriver):
    def _move(self, kind: Kind, io: RowIO) -> RowResult:
        try:
            return self._put_row(io) if kind == "put" else self._get_row(io)
        except StoreError as error:
            return faulted(str(error))


class _SkipsTheIntegrityCheck(MemoryDriver):
    def _verify(self, io: RowIO, stored: StoredObject) -> str | None:
        return None


class _CallsNotReadyAFault(MemoryDriver):
    def _get_row(self, io: RowIO) -> RowResult:
        result = super()._get_row(io)
        return faulted("not ready") if result.status is RowStatus.NOT_READY else result


class _CallsDeclinedAFault(MemoryDriver):
    def _put_row(self, io: RowIO) -> RowResult:
        result = super()._put_row(io)
        return faulted("no space") if result.status is RowStatus.DECLINED else result


class _ProbeIgnoresTheBudget(MemoryDriver):
    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        return tuple(self.store.presence(key) for key in keys)


class _ProbesWithoutLimit(MemoryDriver):
    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        with self._lock:
            self._probes = self._probes[: MAX_PROBES_IN_FLIGHT - 1]
        return super().probe(keys, budget_s, hint)


class _ReportsAccessEndedEarly(MemoryDriver):
    def access_ended(self, op: OpHandle) -> bool:
        return True


class _AcceptsHintsItCannotRead(MemoryDriver):
    def _check_hint(self, hint: Hint) -> None:
        pass


class _ExecutesWithTheControls(MemoryDriver):
    def _execute(self, op: MemoryOp) -> None:
        self.transfer(op)
        self.publish_results(op)
        self.end_access(op)


class _DrainWaitsForEveryone(MemoryDriver):
    def _drain_can_return(self) -> bool:
        return self._quiet()


LIMITED = DriverCapabilities(get=True, put=True, cancel=True, probe=True, max_rows_per_op=1)
GET_ONLY = DriverCapabilities(get=True, put=False, cancel=True, probe=True, max_rows_per_op=None)

# rule -> (driver breaking it, the capabilities to run with, the conformance check that must fail)
BROKEN_CONFORMANCE = {
    "a key appears once per operation": (
        _AcceptsDuplicateKeys,
        DEFAULT_CAPABILITIES,
        "test_a_key_appears_at_most_once_per_operation",
    ),
    "rejection happens before any effect": (
        _TouchesMemoryBeforeChecking,
        DEFAULT_CAPABILITIES,
        "test_rows_outside_a_live_registration_are_rejected_before_any_effect",
    ),
    "a miss leaves the destination alone": (
        _MissTouchesTheDestination,
        DEFAULT_CAPABILITIES,
        "test_get_reports_hit_hit_miss_and_leaves_the_miss_untouched",
    ),
    "a wrong size is a fault": (
        _CallsAWrongSizeAMiss,
        DEFAULT_CAPABILITIES,
        "test_an_object_of_the_wrong_size_is_a_fault_not_a_miss",
    ),
    "a put never damages an existing object": (
        _OverwritesExistingObjects,
        DEFAULT_CAPABILITIES,
        "test_a_put_under_an_existing_key_leaves_the_object_whole",
    ),
    "the row limit is enforced": (
        _IgnoresTheRowLimit,
        LIMITED,
        "test_rows_beyond_the_declared_limit_are_rejected",
    ),
    "an undeclared direction is rejected": (
        _IgnoresTheDirection,
        GET_ONLY,
        "test_an_undeclared_direction_is_rejected",
    ),
    "access ends with evidence": (
        _NeverEndsAccess,
        DEFAULT_CAPABILITIES,
        "test_results_are_stable_and_access_ends_with_evidence",
    ),
    "capabilities are fixed": (
        _ChangesItsCapabilities,
        DEFAULT_CAPABILITIES,
        "test_capabilities_do_not_change",
    ),
    "probe returns within its budget": (
        _ProbeTakesItsTime,
        DEFAULT_CAPABILITIES,
        "test_probe_returns_within_its_budget",
    ),
    "probe answers unknown for a blocked key": (
        _ProbeIgnoresTheBudget,
        DEFAULT_CAPABILITIES,
        "test_probe_answers_unknown_for_a_key_the_store_does_not_answer_in_time",
    ),
    "drain is one-way": (
        _KeepsAcceptingWhileDraining,
        DEFAULT_CAPABILITIES,
        "test_drain_is_one_way",
    ),
    "close needs a true drain": (
        _ClosesAnyTime,
        DEFAULT_CAPABILITIES,
        "test_close_needs_a_true_drain_and_no_live_registrations",
    ),
    "unregister frees the range": (
        _UnregisterKeepsTheRange,
        DEFAULT_CAPABILITIES,
        "test_unregister_is_idempotent_and_frees_the_range",
    ),
}

# rule -> (driver breaking it, the memory-driver check that must fail)
BROKEN_MEMORY = {
    "unregister needs an idle region": (
        _UnregistersBusyRegions,
        "test_unregister_is_refused_while_an_operation_may_access_the_region",
    ),
    "a store timeout is a fault": (
        _CallsATimeoutAMiss,
        "test_a_store_timeout_is_a_fault_not_a_miss",
    ),
    "a recorded operation never raises": (
        _LetsRaisesEscape,
        "test_a_raise_after_the_operation_is_recorded_becomes_a_row_fault",
    ),
    "integrity is verified on every get": (
        _SkipsTheIntegrityCheck,
        "test_a_corrupted_object_is_a_fault_and_never_lands",
    ),
    "an unfinished write is not ready": (
        _CallsNotReadyAFault,
        "test_a_key_held_by_an_unfinished_write_is_not_ready_for_get_put_and_probe",
    ),
    "no room is declined": (
        _CallsDeclinedAFault,
        "test_a_row_the_store_has_no_room_for_is_declined_not_faulted",
    ),
    "an unreadable hint is refused": (
        _AcceptsHintsItCannotRead,
        "test_memory_driver_rejects_a_hint_it_cannot_read_before_any_effect",
    ),
    "a submission racing a drain keeps its handle": (
        _ExecutesWithTheControls,
        "test_a_drain_racing_a_submission_leaves_a_cancelled_operation_with_its_handle",
    ),
    "probes in flight are capped": (
        _ProbesWithoutLimit,
        "test_probes_in_flight_are_capped",
    ),
    "access does not end early": (
        _ReportsAccessEndedEarly,
        "test_queries_do_not_wait_on_an_operation_inside_the_store",
    ),
    "drain returns at once on an unprovable operation": (
        _DrainWaitsForEveryone,
        "test_drain_returns_false_at_once_even_while_another_operation_is_still_provable",
    ),
}

CHECK_FAILED = (AssertionError, pytest.fail.Exception)


def _conformance(driver_cls: type[MemoryDriver], caps: DriverCapabilities) -> DriverConformance:
    suite = type(
        "Suite",
        (TestMemoryDriverConformance,),
        {"driver_cls": driver_cls, "capabilities": caps, "timeout": 0.3},
    )
    return suite()  # type: ignore[no-any-return]


@pytest.mark.parametrize("rule", sorted(BROKEN_CONFORMANCE))
def test_each_conformance_check_passes_the_driver_and_fails_the_one_breaking_its_rule(
    rule: str,
) -> None:
    broken, capabilities, check = BROKEN_CONFORMANCE[rule]
    getattr(_conformance(MemoryDriver, capabilities), check)()
    with pytest.raises(CHECK_FAILED):
        getattr(_conformance(broken, capabilities), check)()


@pytest.mark.parametrize("rule", sorted(BROKEN_MEMORY))
def test_each_memory_driver_check_fails_the_driver_breaking_its_rule(rule: str) -> None:
    broken, check = BROKEN_MEMORY[rule]
    suite = type("Suite", (TestMemoryDriver,), {"driver_cls": broken})()
    with pytest.raises(CHECK_FAILED):
        getattr(suite, check)()


def test_each_broken_driver_breaks_one_rule_and_names_an_existing_check() -> None:
    conformance = [cls for cls, _, _ in BROKEN_CONFORMANCE.values()]
    memory = [cls for cls, _ in BROKEN_MEMORY.values()]
    assert len(set(conformance + memory)) == len(conformance) + len(memory)
    checks = {check for _, _, check in BROKEN_CONFORMANCE.values()}
    assert checks <= {name for name in dir(DriverConformance) if name.startswith("test_")}
    checks = {check for _, check in BROKEN_MEMORY.values()}
    assert checks <= {name for name in dir(TestMemoryDriver) if name.startswith("test_")}
