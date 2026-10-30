# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-memory driver: the shared conformance checks run several ways, its controlled mode, its
batch-registering variant, and proof that every check fails against a driver that breaks exactly
the rule it guards."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    DriverRejected,
    Hint,
    OnChange,
    OpHandle,
    ProbeStatus,
    RegionHandle,
    RegionLayout,
    RegionSpec,
    RegistrationError,
    RowBatch,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreDriver,
    StoreError,
    concluded,
    faulted,
)
from tensorrt_llm._torch.disaggregation.testing.broken_drivers import (
    BROKEN_ASYNC_CONFORMANCE,
    BROKEN_CONFORMANCE,
    BROKEN_TOGETHER_CONFORMANCE,
)
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import (
    OFF_THREAD_NAME,
    SLOT,
    UNRELEASED_ENDPOINTS,
    UNRELEASED_MEMORY,
    Allocation,
    DriverConformance,
    Endpoint,
    Harness,
    Unblock,
    key_of,
    specs_of,
    statuses,
    wait_until,
)
from tensorrt_llm._torch.disaggregation.testing.host_memory import (
    GUARD,
    HostBuffer,
    Memory,
    MemoryWindow,
)
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    DEFAULT_CAPABILITIES,
    MAX_PROBES_IN_FLIGHT,
    Commit,
    Kind,
    MemoryDriver,
    MemoryOp,
    MemoryStore,
    StoredObject,
    TogetherMemoryDriver,
)
from tensorrt_llm._torch.disaggregation.testing.object_trailer import encode_trailer

# An exception that escapes a thread fails the test.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

PREFIX = "m"  # the prefix the controlled-mode endpoints below use
KEY_A, KEY_B, KEY_C = (key_of(label, PREFIX) for label in "ABC")
STAGED_KEY = key_of("staged", "stage")
TIMEOUT = 5.0


# Stores that misbehave on purpose.


class FailingStore(MemoryStore):
    """Raises ``StoreError`` on the n-th commit, as a store that drops a connection would."""

    def __init__(self, fail_on_commit: int) -> None:
        super().__init__()
        self._fail_on = fail_on_commit
        self._commits = 0

    def commit(self, key: str, payload: bytes, trailer: bytes) -> Commit:
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
    """Raises something that is not a ``StoreError`` from inside every commit and read, as a
    binding bug would."""

    def commit(self, key: str, payload: bytes, trailer: bytes) -> Commit:
        raise RuntimeError("binding returned garbage")

    def get(self, key: str) -> StoredObject | None:
        raise RuntimeError("binding returned garbage")


class GatedStore(MemoryStore):
    """Holds reads, writes and queries of ``held`` keys (every key when None) while the gate is
    closed, and says when one is waiting."""

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

    def commit(self, key: str, payload: bytes, trailer: bytes) -> Commit:
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


def endpoint(
    driver: StoreDriver, labels: Sequence[str], *, fill: int = 0, slot: int = SLOT
) -> Endpoint:
    """A registered endpoint for the controlled-mode tests, outside any harness."""
    made = Endpoint(
        driver, labels, HostBuffer(slot * len(labels), fill), None, prefix=PREFIX, slot=slot
    )
    made.register()
    return made


# The conformance suite, four ways.


class TestMemoryDriverConformance(DriverConformance):
    """The suite plus every scenario the in-memory store can stage."""

    driver_cls: type[MemoryDriver] = MemoryDriver
    capabilities = DEFAULT_CAPABILITIES
    background = False  # True: put/get return once recorded and a thread executes them
    # Automatic mode runs each operation inside put/get, which the protocol forbids; the checks
    # that hold a put or a get inside the store skip it, and TestAsyncMemoryDriverConformance runs
    # them.
    returns_before_the_store_answers = False

    def _driver(self, store: MemoryStore | None = None) -> MemoryDriver:
        # An empty store is falsy (it has a length), so test for None explicitly.
        return self.driver_cls(
            store if store is not None else MemoryStore(),
            background=self.background,
            capabilities=self.capabilities,
        )

    def make_driver(self) -> StoreDriver:
        return self._driver()

    def make_blocked_probe(self) -> tuple[StoreDriver, str, Unblock]:
        store = GatedStore(held={STAGED_KEY})
        store.gate.clear()
        return self._driver(store), STAGED_KEY, store.gate.set

    def make_blocked_get(self) -> tuple[StoreDriver, str, threading.Event | None, Unblock]:
        store = GatedStore(held={STAGED_KEY})
        store.gate.clear()
        return self._driver(store), STAGED_KEY, store.waiting, store.gate.set

    def make_blocked_put(self) -> tuple[StoreDriver, str, threading.Event | None, Unblock]:
        store = GatedStore(held={STAGED_KEY})
        store.gate.clear()
        return self._driver(store), STAGED_KEY, store.waiting, store.gate.set

    def make_held_key(self) -> tuple[StoreDriver, str, Unblock]:
        store = MemoryStore()
        store.begin_write(STAGED_KEY)
        return self._driver(store), STAGED_KEY, lambda: store.end_write(STAGED_KEY)

    def make_full_store(self) -> StoreDriver:
        return self._driver(MemoryStore(capacity=1))

    def make_crash_after_record(self) -> StoreDriver:
        return self._driver(CrashingStore())

    def make_driver_leaving_operations_unstarted(self) -> StoreDriver:
        return self.driver_cls(MemoryStore(), controlled=True, capabilities=self.capabilities)

    def make_results_before_access_end(
        self,
    ) -> tuple[StoreDriver, Callable[[OpHandle], None], Callable[[OpHandle], None]]:
        driver = self.driver_cls(MemoryStore(), controlled=True, capabilities=self.capabilities)
        return driver, driver.complete, driver.end_access

    def make_failing_presence(self) -> tuple[StoreDriver, str]:
        store = MemoryStore()
        store.fail_next_presence("connection dropped")
        return self._driver(store), STAGED_KEY

    def make_failing_get(self) -> tuple[StoreDriver, str]:
        return self._driver(TimeoutStore(STAGED_KEY)), STAGED_KEY

    def make_unprovable_get(self) -> tuple[StoreDriver, Callable[[OpHandle], None]]:
        driver = self.driver_cls(MemoryStore(), controlled=True, capabilities=self.capabilities)

        def lose_the_evidence(op: OpHandle) -> None:
            assert isinstance(op, MemoryOp)
            driver.transfer(op)
            driver.publish_results(op)
            driver.mark_unprovable(op)

        return driver, lose_the_evidence


class _HintFreeSignatures(MemoryDriver):
    """A driver written before hints existed: ``get`` and ``probe`` take no hint. With
    ``hints`` undeclared the backend never passes one, so it stays a valid driver."""

    def get(self, rows: RowBatch, *, on_change=None) -> MemoryOp:  # type: ignore[override]
        return super().get(rows, on_change=on_change)

    def probe(self, keys: Sequence[str], budget_s: float) -> tuple[ProbeStatus, ...]:  # type: ignore[override]
        return super().probe(keys, budget_s)


class TestHintFreeDriverConformance(TestMemoryDriverConformance):
    driver_cls = _HintFreeSignatures


class TestAsyncMemoryDriverConformance(TestMemoryDriverConformance):
    """The same driver returning handles before the store answers, as a real SDK does."""

    background = True
    returns_before_the_store_answers = True


SHARED_STORE = MemoryStore()


class TestSharedStoreConformance(DriverConformance):
    """Fresh clients of one persistent store: every check finds the previous checks' objects."""

    returns_before_the_store_answers = False  # the in-memory driver's automatic mode

    def make_driver(self) -> StoreDriver:
        return MemoryDriver(SHARED_STORE)


class TestPoolDriverConformance(DriverConformance):
    """A driver that registered its memory once, when built; endpoints are carved out of it."""

    registers_on_demand = False
    returns_before_the_store_answers = False  # the in-memory driver's automatic mode
    POOL_BYTES = 1 << 16

    def pool_driver(self) -> MemoryDriver:
        return MemoryDriver(MemoryStore())

    def make_driver(self) -> StoreDriver:
        driver = self.pool_driver()
        self.pool = HostBuffer(self.POOL_BYTES)
        self.handle: RegionHandle = driver.register(
            self.pool.address, self.pool.nbytes, self.pool_layout()
        )
        self.next_offset = 0
        return driver

    def pool_layout(self) -> RegionLayout:
        return RegionLayout(4, self.POOL_BYTES // 4, "pool")

    def allocate(self, nbytes: int, fill: int, memory_kind: MemoryKind) -> Allocation:
        start = self.next_offset + GUARD  # host memory stands in for either kind in-process
        self.next_offset = start + nbytes + GUARD
        assert self.next_offset <= self.POOL_BYTES, "the pool is too small for this check"
        return MemoryWindow(self.pool.address + start, nbytes, fill, owner=self.pool), self.handle

    def shutdown(self, driver: StoreDriver) -> None:
        driver.unregister(self.handle)
        driver.close()


class TestTogetherDriverConformance(TestPoolDriverConformance):
    """The pool registered as one batch through ``RegistersRegionsTogether``, plus the batch
    scenarios the in-memory driver can stage."""

    registers_together = True
    driver_cls: type[TogetherMemoryDriver] = TogetherMemoryDriver
    capabilities = DEFAULT_CAPABILITIES

    def pool_driver(self) -> TogetherMemoryDriver:
        return self.driver_cls(MemoryStore(), capabilities=self.capabilities)

    def make_driver(self) -> StoreDriver:
        driver = self.pool_driver()
        self.pool = HostBuffer(self.POOL_BYTES)
        spec = RegionSpec(self.pool.address, self.pool.nbytes, self.pool_layout())
        self.handle = driver.register_regions([spec])
        self.next_offset = 0
        return driver

    def shutdown(self, driver: StoreDriver) -> None:
        assert isinstance(driver, TogetherMemoryDriver)
        driver.unregister_regions(self.handle)
        driver.close()

    def make_together_registration(self) -> tuple[StoreDriver, list[Memory]]:
        return self.pool_driver(), [HostBuffer(2 * SLOT), HostBuffer(2 * SLOT)]

    def make_failing_together_registrations(self) -> list[tuple[StoreDriver, list[Memory]]]:
        staged: list[tuple[StoreDriver, list[Memory]]] = []
        for escaped in (False, True):
            driver = self.pool_driver()
            driver.fail_next_register_regions("store refused", escaped=escaped)
            staged.append((driver, [HostBuffer(2 * SLOT), HostBuffer(2 * SLOT)]))
        return staged

    def make_busy_together_registration(
        self,
    ) -> tuple[StoreDriver, list[Memory], str, threading.Event, Unblock]:
        store = GatedStore(held={STAGED_KEY})
        store.gate.clear()
        driver = self.driver_cls(store, capabilities=self.capabilities)
        memory: list[Memory] = [HostBuffer(2 * SLOT), HostBuffer(2 * SLOT)]
        return driver, memory, STAGED_KEY, store.waiting, store.gate.set


class TestTogetherMemoryDriver:
    """What only the batch-registering in-memory driver's controls can stage."""

    def _driver(self, **kwargs: object) -> TogetherMemoryDriver:
        return TogetherMemoryDriver(MemoryStore(), **kwargs)  # type: ignore[arg-type]

    def test_single_region_calls_are_rejected(self) -> None:
        driver = self._driver()
        memory = HostBuffer(SLOT)
        with pytest.raises(DriverRejected):
            driver.register(memory.address, SLOT, RegionLayout(SLOT, 1))
        handle = driver.register_regions(specs_of([memory]))
        with pytest.raises(DriverRejected):
            driver.unregister(handle)
        assert driver.live_batches() == [handle]
        driver.unregister_regions(handle)
        assert driver.drain(TIMEOUT) is True
        driver.close()

    def test_a_batch_is_checked_whole_before_anything_is_registered(self) -> None:
        driver = self._driver()
        memory = HostBuffer(2 * SLOT)
        inside = RegionSpec(memory.address + SLOT, SLOT, RegionLayout(SLOT, 1))
        with pytest.raises(DriverRejected):
            driver.register_regions([*specs_of([memory]), inside])  # overlap within the batch
        with pytest.raises(DriverRejected):
            driver.register_regions([])
        assert driver.live_registrations() == [] and driver.live_batches() == []
        host_only = DriverCapabilities(
            get=True, put=True, cancel=False, probe=False, max_rows_per_op=None
        )
        driver = self._driver(capabilities=host_only)
        device = RegionSpec(memory.address, SLOT, RegionLayout(SLOT, 1, "kv", MemoryKind.DEVICE))
        with pytest.raises(DriverRejected):
            driver.register_regions([device])
        assert driver.live_registrations() == [] and driver.live_batches() == []

    def test_a_busy_batch_is_not_revoked(self) -> None:
        driver = self._driver(controlled=True)
        memory = HostBuffer(SLOT)
        handle = driver.register_regions(specs_of([memory]))
        memory.write(b"a" * SLOT, 0)
        op = driver.put(RowBatch.of([RowIO(KEY_A, (Span(memory.address, SLOT),))]))
        with pytest.raises(DriverRejected):
            driver.unregister_regions(handle)
        driver.transfer(op)
        driver.publish_results(op)
        driver.end_access(op)
        driver.unregister_regions(handle)
        assert driver.drain(TIMEOUT) is True
        driver.close()

    @pytest.mark.parametrize("escaped", [True, False])
    def test_a_failed_batch_hands_out_its_handle_only_when_something_may_be_registered(
        self, escaped: bool
    ) -> None:
        driver = self._driver()
        memory = HostBuffer(2 * SLOT)
        driver.fail_next_register_regions("store refused", escaped=escaped)
        with pytest.raises(RegistrationError) as caught:
            driver.register_regions(specs_of([memory]))
        if escaped:
            assert caught.value.handle is not None
            assert len(driver.live_registrations()) == 1
            driver.unregister_regions(caught.value.handle)
        else:
            assert caught.value.handle is None
        assert driver.live_registrations() == []
        assert driver.drain(TIMEOUT) is True
        driver.close()

    def test_a_failed_revocation_keeps_the_batch_live_for_a_retry(self) -> None:
        driver = self._driver()
        memory = HostBuffer(SLOT)
        handle = driver.register_regions(specs_of([memory]))
        driver.fail_next_unregister_regions("rpc timeout")
        with pytest.raises(StoreError):
            driver.unregister_regions(handle)
        assert driver.live_batches() == [handle]
        assert driver.drain(TIMEOUT) is True
        with pytest.raises(DriverRejected):
            driver.close()  # a registration is still live
        driver.unregister_regions(handle)
        driver.close()


class TestMemoryDriver:
    """Behaviour only the in-memory driver's controls or stores can stage."""

    driver_cls: type[MemoryDriver] = MemoryDriver

    def make(self, store: MemoryStore | None = None, **kwargs: object) -> MemoryDriver:
        store = store if store is not None else MemoryStore()
        return self.driver_cls(store, **kwargs)  # type: ignore[arg-type]

    def _publish(
        self, driver: MemoryDriver, rows: dict[str, bytes], *, keep_registered: bool = False
    ) -> Endpoint:
        writer = endpoint(driver, list(rows))
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
        reader = endpoint(driver, ["A"], fill=0x2D)
        (result,) = driver.poll(driver.get(reader.rows()))
        assert result.status is RowStatus.FAULT
        assert result.fault == "payload checksum mismatch"
        assert reader.read_row(0) == b"-" * SLOT

    def test_an_object_shorter_than_its_trailer_claims_is_a_fault_and_never_lands(self) -> None:
        store = MemoryStore()
        driver = self.make(store)
        store.overwrite(KEY_A, b"aaaa", encode_trailer(KEY_A.encode(), SLOT))  # claims 8 bytes
        reader = endpoint(driver, ["A"], fill=0x2D)
        (result,) = driver.poll(driver.get(reader.rows()))
        assert result.status is RowStatus.FAULT
        assert result.fault == f"object is 4 bytes, expected {SLOT}"
        assert reader.read_row(0) == b"-" * SLOT

    def test_a_different_object_under_the_same_key_is_diagnosed(self) -> None:
        store = MemoryStore()
        driver = self.make(store)
        self._publish(driver, {"A": b"aaaaaaaa"})
        other = endpoint(driver, ["A"])
        other.fill_row(0, b"zzzzzzzz")
        (result,) = driver.poll(driver.put(other.rows()))
        assert result.status is RowStatus.FAULT
        assert store.committed_payload(KEY_A) == b"aaaaaaaa"
        assert len(store) == 1

    def test_a_retry_that_fails_in_the_store_leaves_the_committed_object_whole(self) -> None:
        store = FailingStore(fail_on_commit=2)
        driver = self.make(store)
        self._publish(driver, {"A": b"aaaaaaaa"})
        retry = endpoint(driver, ["A"])
        retry.fill_row(0, b"aaaaaaaa")
        (result,) = driver.poll(driver.put(retry.rows()))
        assert result.status is RowStatus.FAULT and result.fault == "connection dropped"
        reader = endpoint(driver, ["A"], fill=0x2D)
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

    # Result and access end are independent.

    def test_result_can_come_before_access_end(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
        driver = self.make(store, controlled=True)
        reader = endpoint(driver, ["A", "B"], fill=0x2D)
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
        reader = endpoint(driver, ["A"], fill=0x2D)
        op = driver.get(reader.rows())
        driver.complete(op, iter([0]))
        assert statuses(driver, op) == [RowStatus.READY]
        assert reader.read_row(0) == b"aaaaaaaa"

    def test_access_can_end_before_the_result_is_published(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa"})
        driver = self.make(store, controlled=True)
        reader = endpoint(driver, ["A"], fill=0x2D)
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
        reader = endpoint(driver, ["A", "B"], fill=0x2D)
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
        reader = endpoint(driver, ["A"], fill=0x2D)
        op = driver.get(reader.rows())
        driver.cancel(op)
        assert statuses(driver, op) == [RowStatus.CANCELLED]
        with pytest.raises(RuntimeError):
            driver.write_late(op, 0, b"aaaaaaaa")
        assert reader.read_row(0) == b"-" * SLOT

    # Store errors.

    def test_a_store_timeout_is_a_fault_not_a_miss(self) -> None:
        driver = self.make(TimeoutStore(KEY_A))
        reader = endpoint(driver, ["A", "C"], fill=0x2D)
        results = driver.poll(driver.get(reader.rows()))
        assert [r.status for r in results] == [RowStatus.FAULT, RowStatus.MISS]
        assert results[0].fault == "timeout"
        assert reader.read_row(0) == b"-" * SLOT

    def test_a_store_error_mid_operation_becomes_a_row_fault_and_keeps_the_handle(self) -> None:
        store = FailingStore(fail_on_commit=2)
        driver = self.make(store)
        writer = endpoint(driver, ["A", "B", "C"])
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

    # Concurrency.

    def test_queries_do_not_wait_on_an_operation_inside_the_store(self) -> None:
        store = GatedStore()
        driver = self.make(store, controlled=True)
        done = endpoint(driver, ["A"])
        finished = driver.get(done.rows())
        driver.fail(finished, "timeout")

        slow = endpoint(driver, ["B"])
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

    # Cancel.

    def test_cancel_concludes_unstarted_rows_and_spares_started_ones(self) -> None:
        store = MemoryStore()
        self._publish(self.make(store), {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
        driver = self.make(store, controlled=True)
        reader = endpoint(driver, ["A", "B"], fill=0x2D)
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
        reader = endpoint(driver, ["A", "B"])
        op = driver.get(reader.rows())
        driver.cancel(op)
        assert statuses(driver, op) == [RowStatus.CANCELLED, RowStatus.CANCELLED]
        assert driver.access_ended(op)

    def test_an_ineffective_cancel_changes_nothing(self) -> None:
        driver = self.make(controlled=True)
        driver.set_cancel_effective(False)
        reader = endpoint(driver, ["A"])
        op = driver.get(reader.rows())
        driver.cancel(op)
        assert statuses(driver, op) == [RowStatus.PENDING]
        assert not driver.access_ended(op)

    # Probe.

    def test_a_blocked_probe_holds_drain_until_the_store_answers(self) -> None:
        store = GatedStore(held={KEY_A})
        base = self.driver_cls

        class BudgetEndsAtTheGate(base):  # type: ignore[misc,valid-type]
            # The wait ends once the worker is stuck at the gate, not after a wall-clock budget:
            # by then B is answered and A, C are not, whatever the scheduler did.
            def _wait_probe(self, finished: threading.Event, budget_s: float) -> bool:
                assert store.waiting.wait(TIMEOUT), "the probe worker never reached the gate"
                return False  # as if the budget had run out

        driver = BudgetEndsAtTheGate(store)
        self._publish(driver, {"B": b"bbbbbbbb"})
        store.gate.clear()
        try:
            assert driver.probe([KEY_B, KEY_A, KEY_C], 0.05) == (
                ProbeStatus.HIT,
                ProbeStatus.UNKNOWN,
                ProbeStatus.UNKNOWN,
            )  # keys before the blocked one were answered; the rest are unknown, not MISS
            assert driver.probes_in_flight() == 1
            assert driver.drain(0.05) is False  # the probe is still inside the store
        finally:
            store.gate.set()
            driver.join_probes(TIMEOUT)
        assert driver.probe([KEY_A], TIMEOUT) == (ProbeStatus.UNKNOWN,)  # draining now
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

    def test_probe_unknown_switch(self) -> None:
        driver = self.make()
        self._publish(driver, {"A": b"aaaaaaaa"})
        driver.set_probe_unknown(True)
        assert driver.probe([KEY_A, KEY_C], TIMEOUT) == (ProbeStatus.UNKNOWN,) * 2

    # Hints and spans the backend must not hand this driver.

    def test_memory_driver_rejects_a_hint_it_cannot_read_before_any_effect(self) -> None:
        """The backend rejects such hints before dispatch; this driver also refuses them itself,
        so a backend bug cannot make it read with a hint it silently dropped."""
        driver = self.make()
        self._publish(driver, {"A": b"aaaaaaaa"})
        reader = endpoint(driver, ["A"], fill=0x2D)
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
        reader = endpoint(driver, ["A"], fill=0x2D)
        hint = {"source": "worker-7"}
        op = driver.get(reader.rows(), hint)
        assert statuses(driver, op) == [RowStatus.READY]
        assert reader.read_row(0) == b"aaaaaaaa"
        assert driver.probe([KEY_A], TIMEOUT, hint) == (ProbeStatus.HIT,)
        assert driver.hints_seen == [hint, hint]
        assert all(seen is hint for seen in driver.hints_seen)  # the same object, not a copy
        assert statuses(driver, driver.get(reader.rows())) == [RowStatus.READY]
        assert len(driver.hints_seen) == 2  # no hint, nothing recorded

    def test_memory_driver_rejects_overlapping_spans_within_one_get(self) -> None:
        """The caller guarantees disjoint spans; this driver checks anyway, for the same reason
        it checks hints."""
        driver = self.make()
        self._publish(driver, {"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
        reader = endpoint(driver, ["A", "B"], fill=0x2D)
        rows = [
            RowIO(KEY_A, (reader.span(0),)),
            RowIO(KEY_B, (Span(reader.span(0).address + 4, SLOT),)),  # overlaps row A's span
        ]
        with pytest.raises(DriverRejected):
            driver.get(RowBatch.of(rows))
        assert reader.read_row(0) == reader.read_row(1) == b"-" * SLOT
        assert statuses(driver, driver.get(reader.rows())) == [RowStatus.READY, RowStatus.READY]

    def test_a_single_span_driver_rejects_a_multi_span_row(self) -> None:
        single = DriverCapabilities(
            get=True, put=True, cancel=True, probe=True, max_rows_per_op=None, absent_is_miss=True
        )
        driver = self.make(capabilities=single)
        writer = endpoint(driver, ["A", "B"])
        with pytest.raises(DriverRejected):
            driver.put(RowBatch.of([RowIO(KEY_A, (writer.span(0), writer.span(1)))]))
        assert statuses(driver, driver.put(writer.rows())) == [RowStatus.READY, RowStatus.READY]

    # Submission and drain.

    def test_a_drain_racing_a_submission_leaves_a_cancelled_operation_with_its_handle(
        self,
    ) -> None:
        store = MemoryStore()
        racing = type(
            "Racing", (self.driver_cls,), {"_before_execute": lambda self, op: self.drain(0.0)}
        )
        driver = racing(store)
        writer = endpoint(driver, ["A"])
        writer.fill_row(0, b"aaaaaaaa")
        try:
            op = driver.put(writer.rows())
        except Exception as error:  # noqa: BLE001 - the check is that the handle comes back
            pytest.fail(f"put raised instead of returning its handle: {error!r}")
        assert statuses(driver, op) == [RowStatus.CANCELLED]
        assert driver.access_ended(op)
        assert len(store) == 0
        assert driver.drain(TIMEOUT) is True

    def test_a_worker_that_cannot_start_leaves_a_faulted_operation_with_its_handle(self) -> None:
        """Background mode records the operation before its thread starts. When the thread cannot
        start, nothing ran: every row is a fault, access is over, and the driver goes on."""
        store = MemoryStore()
        driver = self.make(store, background=True)
        writer = endpoint(driver, ["A", "B"])
        writer.fill_row(0, b"aaaaaaaa")
        writer.fill_row(1, b"bbbbbbbb")

        def refuse(thread: threading.Thread) -> None:
            raise RuntimeError("can't start new thread")

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(threading.Thread, "start", refuse)
            try:
                op = driver.put(writer.rows())
            except Exception as error:  # noqa: BLE001 - the check is that the handle comes back
                pytest.fail(f"put raised instead of returning its handle: {error!r}")
        results = driver.poll(op)
        assert [r.status for r in results] == [RowStatus.FAULT, RowStatus.FAULT]
        assert results[0].fault == "no worker thread: can't start new thread"
        assert driver.access_ended(op)
        assert len(store) == 0
        later = driver.put(writer.rows())  # nothing stale stands in the way of the next thread
        wait_until(lambda: concluded(driver.poll(later)), "the next operation", TIMEOUT)
        assert statuses(driver, later) == [RowStatus.READY, RowStatus.READY]
        self._shutdown(driver)

    def test_drain_waits_for_the_last_on_change_of_every_operation(self) -> None:
        """A worker counts as finished only once it told of its operation: a drain that returned
        True earlier would let the owner close the driver while the callback still queries it."""
        driver = self.make(MemoryStore(), background=True)
        reader = endpoint(driver, ["A"])
        entered, release = threading.Event(), threading.Event()
        seen: list[tuple[list[RowStatus], bool]] = []

        def look(op: OpHandle) -> None:
            entered.set()
            assert release.wait(TIMEOUT)
            seen.append((statuses(driver, op), driver.access_ended(op)))

        driver.get(reader.rows(), on_change=look)
        assert entered.wait(TIMEOUT)
        assert driver.drain(0) is False  # the worker is still telling of its operation
        release.set()
        assert driver.drain(TIMEOUT) is True
        assert seen == [([RowStatus.MISS], True)]
        self._shutdown(driver)

    def test_unregister_is_refused_while_an_operation_may_access_the_region(self) -> None:
        driver = self.make(controlled=True)
        reader = endpoint(driver, ["A"])
        other = endpoint(driver, ["B"])
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
        reader = endpoint(driver, ["A"], fill=0x2D)
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
        reader = endpoint(driver, ["A"])
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
        lost = endpoint(driver, ["A"])
        pending = endpoint(driver, ["B"])
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

    # Registration and shutdown.

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
            driver.register(writer.memory.address, SLOT, RegionLayout(SLOT, 1))

    def test_memory_driver_is_a_store_driver(self) -> None:
        driver: StoreDriver = self.make()
        assert isinstance(driver, StoreDriver)

    # The conformance harness itself.

    def _harness(self, driver: MemoryDriver) -> Harness:
        return Harness(TestMemoryDriverConformance(), driver, TIMEOUT)

    def test_harness_keeps_a_buffer_whose_registration_escaped(self) -> None:
        driver = self.make()
        h = self._harness(driver)
        driver.fail_next_register("timed out after the request", escaped=True)
        with pytest.raises(RegistrationError):
            h.endpoint(["A"])
        assert len(h.endpoints) == 1 and h.endpoints[0].handle is not None
        assert h.cleanup() == []
        assert driver.live_registrations() == []

    def test_harness_retains_an_endpoint_it_could_not_unregister_and_retries(self) -> None:
        driver = self.make()
        h = self._harness(driver)
        made = h.endpoint(["A"])
        assert made.handle is not None
        driver.fail_next_unregister(made.handle, "busy")
        problems = h.cleanup()
        assert problems and not h.done
        assert made in UNRELEASED_ENDPOINTS
        assert driver.live_registrations() == [made.handle]
        assert h.cleanup() == [] and h.done
        assert driver.live_registrations() == []

    def test_harness_retains_endpoints_when_drain_does_not_finish(self) -> None:
        driver = type("DrainSaysNo", (self.driver_cls,), {"drain": lambda self, budget_s: False})(
            MemoryStore()
        )
        h = self._harness(driver)
        made = h.endpoint(["A"])
        problems = h.cleanup()
        assert problems == ["drain did not return True"] and not h.done
        assert made in UNRELEASED_ENDPOINTS

    def test_retained_pool_memory_outlives_its_suite(self) -> None:
        """A pool endpoint retained for an operation that never proved its end keeps the pool
        itself alive, not just an address into it."""

        class ControlledPool(TestPoolDriverConformance):
            def pool_driver(self) -> MemoryDriver:
                return MemoryDriver(MemoryStore(), controlled=True)

        suite = ControlledPool()
        driver = suite.make_driver()
        assert isinstance(driver, MemoryDriver)
        pool_alive = weakref.ref(suite.pool)
        h = Harness(suite, driver, TIMEOUT)
        made = h.endpoint(["A"])
        op = driver.get(made.rows())
        h.ops.append(op)
        driver.transfer(op)
        driver.publish_results(op)
        driver.mark_unprovable(op)
        assert h.cleanup() == []
        assert made in UNRELEASED_ENDPOINTS
        del suite, h, made
        gc.collect()
        assert pool_alive() is not None, "the pool was freed while the driver may still touch it"
        assert driver.live_registrations()

    def test_cleanup_retry_releases_pool_memory_once(self) -> None:
        class DrainFailsOnce(MemoryDriver):
            drains = 0

            def drain(self, budget_s: float) -> bool:
                self.drains += 1
                return super().drain(budget_s) and self.drains > 1

        class CountingPool(TestPoolDriverConformance):
            releases: list[object]

            def pool_driver(self) -> MemoryDriver:
                return DrainFailsOnce(MemoryStore())

            def release(self, memory: object) -> None:
                self.releases.append(memory)

        suite = CountingPool()
        suite.releases = []
        driver = suite.make_driver()
        h = Harness(suite, driver, TIMEOUT)
        made = h.endpoint(["A"])
        assert h.cleanup() == ["drain did not return True"]
        assert suite.releases == [made.memory]
        assert made in UNRELEASED_ENDPOINTS
        assert h.cleanup() == [] and h.done
        assert suite.releases == [made.memory]  # once, not twice

    def test_harness_retains_endpoints_of_an_unprovable_operation_without_complaint(self) -> None:
        driver = self.make(controlled=True)
        h = self._harness(driver)
        reader = h.endpoint(["A"])
        op = driver.get(reader.rows())
        h.ops.append(op)
        driver.transfer(op)
        driver.publish_results(op)
        driver.mark_unprovable(op)
        assert h.cleanup() == []
        assert reader in UNRELEASED_ENDPOINTS and not h.done
        assert reader.handle in driver.live_registrations()


# Drivers that break one rule only the in-memory tests can observe.


class _SkipsTheIntegrityCheck(MemoryDriver):
    def _verify(self, io: RowIO, stored: StoredObject) -> str | None:
        return None


class _ProbesWithoutLimit(MemoryDriver):
    def _probe_slot_free(self) -> bool:
        return True  # the workers stay on the books, so they are still joined


class _ReportsAccessEndedEarly(MemoryDriver):
    def access_ended(self, op: OpHandle) -> bool:
        return True


class _AcceptsHintsItCannotRead(MemoryDriver):
    def _check_hint(self, hint: Hint) -> None:
        pass


class _AcceptsOverlappingSpans(MemoryDriver):
    def _check_disjoint(self, kind: Kind, rows: Sequence[RowIO]) -> None:
        pass


class _TakesAnyRowShape(MemoryDriver):
    def _check_span_count(self, rows: Sequence[RowIO]) -> None:
        pass


class _ExecutesWithTheControls(MemoryDriver):
    def _execute(self, op: MemoryOp) -> None:
        self.transfer(op)
        self.publish_results(op)
        self.end_access(op)


class _LeavesAnUnstartedOperationPending(MemoryDriver):
    def _fail_unstarted(self, op: MemoryOp, reason: str) -> None:
        pass


# rule -> (driver breaking it, the memory-driver check that must fail)
BROKEN_MEMORY: dict[str, tuple[type[MemoryDriver], str]] = {
    "integrity is verified on every get": (
        _SkipsTheIntegrityCheck,
        "test_a_corrupted_object_is_a_fault_and_never_lands",
    ),
    "probes in flight are capped": (
        _ProbesWithoutLimit,
        "test_probes_in_flight_are_capped",
    ),
    "access does not end early": (
        _ReportsAccessEndedEarly,
        "test_queries_do_not_wait_on_an_operation_inside_the_store",
    ),
    "an unreadable hint is refused": (
        _AcceptsHintsItCannotRead,
        "test_memory_driver_rejects_a_hint_it_cannot_read_before_any_effect",
    ),
    "overlapping spans are refused": (
        _AcceptsOverlappingSpans,
        "test_memory_driver_rejects_overlapping_spans_within_one_get",
    ),
    "an undeclared multi-span row is refused": (
        _TakesAnyRowShape,
        "test_a_single_span_driver_rejects_a_multi_span_row",
    ),
    "a submission racing a drain keeps its handle": (
        _ExecutesWithTheControls,
        "test_a_drain_racing_a_submission_leaves_a_cancelled_operation_with_its_handle",
    ),
    "a worker that cannot start concludes its operation": (
        _LeavesAnUnstartedOperationPending,
        "test_a_worker_that_cannot_start_leaves_a_faulted_operation_with_its_handle",
    ),
}

CHECK_FAILED = (AssertionError, pytest.fail.Exception)


class UnexpectedSkip(Exception):
    """A paired check skipped; not an expected failure, so no ``pytest.raises`` swallows it."""


def _run_check(suite: object, check: str) -> None:
    """Runs one check; a skip is a failure here, because a pairing that skips proves nothing."""
    try:
        getattr(suite, check)()
    except pytest.skip.Exception as skipped:
        raise UnexpectedSkip(f"{check} skipped instead of running: {skipped}") from None


def _conformance(
    driver_cls: type[MemoryDriver],
    caps: DriverCapabilities,
    base: type[DriverConformance] = TestMemoryDriverConformance,
) -> DriverConformance:
    suite = type("Suite", (base,), {"driver_cls": driver_cls, "capabilities": caps, "timeout": 0.3})
    return suite()  # type: ignore[no-any-return]


@pytest.mark.parametrize("rule", sorted(BROKEN_CONFORMANCE))
def test_each_conformance_check_passes_the_driver_and_fails_the_one_breaking_its_rule(
    rule: str,
) -> None:
    broken, capabilities, check = BROKEN_CONFORMANCE[rule]
    _run_check(_conformance(MemoryDriver, capabilities), check)
    with pytest.raises(CHECK_FAILED):
        _run_check(_conformance(broken, capabilities), check)


@pytest.mark.parametrize("rule", sorted(BROKEN_ASYNC_CONFORMANCE))
def test_each_async_conformance_check_passes_the_driver_and_fails_the_one_breaking_its_rule(
    rule: str,
) -> None:
    broken, capabilities, check = BROKEN_ASYNC_CONFORMANCE[rule]
    asynchronous = TestAsyncMemoryDriverConformance
    _run_check(_conformance(MemoryDriver, capabilities, asynchronous), check)
    with pytest.raises(CHECK_FAILED):
        _run_check(_conformance(broken, capabilities, asynchronous), check)
    # A broken driver may claim an access ended while its own thread still runs it.
    wait_until(
        lambda: not any(t.name == "memory-driver-op" for t in threading.enumerate()),
        "the broken driver's threads to finish",
        TIMEOUT,
    )


@pytest.mark.parametrize("rule", sorted(BROKEN_TOGETHER_CONFORMANCE))
def test_each_together_conformance_check_passes_the_driver_and_fails_the_one_breaking_its_rule(
    rule: str,
) -> None:
    broken, capabilities, check = BROKEN_TOGETHER_CONFORMANCE[rule]
    together = TestTogetherDriverConformance
    _run_check(_conformance(TogetherMemoryDriver, capabilities, together), check)
    with pytest.raises(CHECK_FAILED):
        _run_check(_conformance(broken, capabilities, together), check)


@pytest.mark.parametrize("rule", sorted(BROKEN_MEMORY))
def test_each_memory_driver_check_fails_the_driver_breaking_its_rule(rule: str) -> None:
    broken, check = BROKEN_MEMORY[rule]
    suite = type("Suite", (TestMemoryDriver,), {"driver_cls": broken})()
    with pytest.raises(CHECK_FAILED):
        _run_check(suite, check)


def test_a_skip_on_the_broken_side_is_not_an_expected_failure() -> None:
    no_probe = DriverCapabilities(
        get=True, put=True, cancel=True, probe=False, max_rows_per_op=None
    )
    suite = _conformance(MemoryDriver, no_probe)
    with pytest.raises(UnexpectedSkip):
        with pytest.raises(CHECK_FAILED):  # the shape the pairing test uses; must not catch it
            _run_check(suite, "test_probe_reports_hit_and_miss")


class _CannotTellAbsenceFromFailure(TogetherMemoryDriver):
    """A driver whose store cannot tell absence from failure: a get of a key the store does not
    hold concludes FAULT and leaves its access unprovable, which the contract allows; every other
    operation ends its access as usual. ``gets`` records the statuses of every get."""

    gets: list[list[RowStatus]] = []

    def __init__(self, store: MemoryStore, **options: object) -> None:
        super().__init__(store, controlled=True, **options)

    def _get_row(self, io: RowIO) -> RowResult:
        result = super()._get_row(io)
        if result.status is RowStatus.MISS:
            raise StoreError("the store cannot tell absence from failure")
        return result

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> MemoryOp:
        op = super().put(rows, on_change=on_change)
        self.complete(op)
        self.end_access(op)
        return op

    def get(
        self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None
    ) -> MemoryOp:
        op = super().get(rows, hint, on_change=on_change)
        self.complete(op)
        self.gets.append(statuses(self, op))
        if RowStatus.FAULT in self.gets[-1]:
            self.mark_unprovable(op)
        else:
            self.end_access(op)
        return op


@pytest.mark.parametrize(
    ("capabilities", "check", "region"),
    [
        pytest.param(
            replace(DEFAULT_CAPABILITIES, put=False, absent_is_miss=False),
            "test_regions_registered_together_are_revoked_together_and_the_range_freed",
            0,
            id="get-only probe",
        ),
        pytest.param(
            replace(DEFAULT_CAPABILITIES, put=False, absent_is_miss=False),
            "test_a_batch_is_not_revoked_while_any_of_its_regions_is_busy",
            0,
            id="get-only probe of a busy batch",
        ),
        pytest.param(
            replace(DEFAULT_CAPABILITIES, absent_is_miss=False),
            "test_a_batch_is_not_revoked_while_any_of_its_regions_is_busy",
            1,
            id="stalled get",
        ),
    ],
)
def test_a_row_left_unprovable_skips_the_batch_check_and_retains_the_memory(
    capabilities: DriverCapabilities, check: str, region: int
) -> None:
    """The batch checks move a row through each region to prove it registered, and the busy check
    waits for its stalled get; a store that leaves such a row's access unprovable ends the check
    as a skip naming the region, before any revocation, with the batch memory retained. A get-only
    driver's probe is a get of a fresh key, which faults at once; with put the probes succeed and
    the stalled get, of a key nobody holds, faults once released."""
    _CannotTellAbsenceFromFailure.gets = []
    suite = _conformance(_CannotTellAbsenceFromFailure, capabilities, TestTogetherDriverConformance)
    retained = len(UNRELEASED_MEMORY)
    with pytest.raises(pytest.skip.Exception, match=f"region {region}: .*unprovable"):
        getattr(suite, check)()
    assert len(UNRELEASED_MEMORY) == retained + 2  # both regions of the batch
    gets = _CannotTellAbsenceFromFailure.gets
    assert gets and all(got == [RowStatus.FAULT] for got in gets), gets


def test_every_check_has_a_broken_driver_and_each_driver_one_pairing() -> None:
    pairings = [
        *BROKEN_CONFORMANCE.values(),
        *BROKEN_ASYNC_CONFORMANCE.values(),
        *BROKEN_TOGETHER_CONFORMANCE.values(),
    ]
    conformance = [cls for cls, _, _ in pairings]
    memory = [cls for cls, _ in BROKEN_MEMORY.values()]
    assert len(set(conformance + memory)) == len(conformance) + len(memory)
    checks = {check for _, _, check in pairings}
    # A check no broken driver fails is one nothing proves can fail.
    assert checks == {name for name in dir(DriverConformance) if name.startswith("test_")}
    checks = {check for _, check in BROKEN_MEMORY.values()}
    assert checks <= {name for name in dir(TestMemoryDriver) if name.startswith("test_")}


# The suite's memory hooks.


def test_a_suite_overriding_only_allocate_is_asked_for_the_total_as_before() -> None:
    asked: list[tuple[int, int, MemoryKind]] = []

    class Suite(TestMemoryDriverConformance):
        def allocate(self, nbytes: int, fill: int, memory_kind: MemoryKind) -> Allocation:
            asked.append((nbytes, fill, memory_kind))
            return super().allocate(nbytes, fill, memory_kind)

    with Suite().harness() as h:
        h.endpoint(["A", "B"], slot=4, fill=0x2D)
    assert asked == [(8, 0x2D, MemoryKind.HOST)]


def test_a_suite_overriding_allocate_rows_learns_the_row_size_and_count() -> None:
    """Two endpoints of the same total, laid out differently: the hook tells them apart, which
    a store whose pools each hold one row size needs to pick the pool."""
    asked: list[tuple[int, int, int, MemoryKind]] = []

    class Suite(TestMemoryDriverConformance):
        def allocate_rows(
            self, slot_bytes: int, rows: int, fill: int, memory_kind: MemoryKind
        ) -> Allocation:
            asked.append((slot_bytes, rows, fill, memory_kind))
            return super().allocate_rows(slot_bytes, rows, fill, memory_kind)

    with Suite().harness() as h:
        h.endpoint(["A", "B"], slot=4)
        h.endpoint(["C"], slot=8)
    assert asked == [(4, 2, 0, MemoryKind.HOST), (8, 1, 0, MemoryKind.HOST)]


def test_the_zero_budget_drain_check_closes_a_staged_driver_that_signals_nothing() -> None:
    """A blocked-get hook without a signal still staged a driver; the check runs its second half
    on that driver and closes it, rather than leaving it open beside a fresh one."""
    closed: list[MemoryDriver] = []
    staged: list[MemoryDriver] = []

    class RecordsItsClose(MemoryDriver):
        def close(self) -> None:
            closed.append(self)
            super().close()

    class Suite(TestMemoryDriverConformance):
        driver_cls = RecordsItsClose

        def make_blocked_get(self) -> tuple[StoreDriver, str, threading.Event | None, Unblock]:
            store = GatedStore(held={STAGED_KEY})
            store.gate.clear()
            driver = self._driver(store)
            staged.append(driver)
            return driver, STAGED_KEY, None, store.gate.set

    Suite().test_drain_polled_without_a_budget_turns_true_once_nothing_runs()
    assert closed == staged and len(staged) == 1


class _TakesStraySpansAndNeverProvesItsEnd(MemoryDriver):
    """Takes a row with a span in no registration, then can never prove that operation's access
    ended: the worst a driver can do with memory the suite does not otherwise retain. Rows inside
    the registrations move as usual."""

    def _check_registered(self, rows: Sequence[RowIO]) -> None:
        pass

    def _check_extents(self, rows: Sequence[RowIO]) -> None:
        pass

    def _execute(self, op: MemoryOp) -> None:
        if all(self._registered(span) for row in op.rows for span in row.io.spans):
            super()._execute(op)
        else:
            self.mark_unprovable(op)


class _RaisesOnAStraySpan(_TakesStraySpansAndNeverProvesItsEnd):
    """Lets a raise that is not ``DriverRejected`` out of a put or get of a row with a span in no
    registration, after the operation was recorded: the memory may be in use all the same."""

    def _execute(self, op: MemoryOp) -> None:
        super()._execute(op)
        if op.unprovable:
            raise RuntimeError("the binding failed with the row in flight")


@pytest.mark.parametrize(
    ("driver_cls", "outcome"),
    [(_TakesStraySpansAndNeverProvesItsEnd, CHECK_FAILED), (_RaisesOnAStraySpan, RuntimeError)],
)
@pytest.mark.parametrize(
    "check",
    [
        "test_a_row_outside_every_registration_is_rejected_before_any_effect",
        "test_every_span_of_a_row_is_checked_against_the_registrations",
    ],
)
def test_a_stray_span_a_driver_took_is_retained_when_its_rejection_check_fails(
    check: str,
    driver_cls: type[MemoryDriver],
    outcome: type[BaseException] | tuple[type[BaseException], ...],
) -> None:
    """A driver that took the row fails the check; one that raised something other than
    ``DriverRejected`` lets that raise through. Either way the stray memory the operation may
    still touch is kept alive, as the endpoints of an unprovable operation are."""
    suite = _conformance(driver_cls, DEFAULT_CAPABILITIES)
    retained = len(UNRELEASED_MEMORY)
    with pytest.raises(outcome):
        _run_check(suite, check)
    assert len(UNRELEASED_MEMORY) == retained + 1


def test_a_drain_that_never_returns_fails_the_zero_budget_check_and_is_not_called_again() -> None:
    """The check fails at the harness timeout, and the harness then asks nothing more of the
    driver: its usual cleanup would drain again and hang the run. The stuck call is released
    only once the check is over, so no thread is left behind."""
    release = threading.Event()
    budgets: list[float] = []

    class DrainNeverReturns(MemoryDriver):
        def drain(self, budget_s: float) -> bool:
            budgets.append(budget_s)
            release.wait(TIMEOUT)
            return super().drain(budget_s)

    suite = _conformance(DrainNeverReturns, DEFAULT_CAPABILITIES)
    failures: list[BaseException] = []

    def check() -> None:
        try:
            suite.test_drain_polled_without_a_budget_turns_true_once_nothing_runs()
        except BaseException as error:  # noqa: BLE001 - judged below
            failures.append(error)

    checker = threading.Thread(target=check, daemon=True)
    checker.start()
    checker.join(TIMEOUT)
    finished = not checker.is_alive()
    release.set()  # lets the stuck drain and, if cleanup drained again, the check return
    checker.join(TIMEOUT)
    wait_until(
        lambda: not any(t.name == OFF_THREAD_NAME for t in threading.enumerate()),
        "the released drain call to return",
        TIMEOUT,
    )
    assert finished, "the check hung in its own cleanup"
    assert failures and "did not return" in str(failures[0]), failures
    assert budgets == [0.0], "cleanup drained a driver whose drain had not returned"


class TestMemoryDriverOnChange:
    @pytest.mark.parametrize("stop", ["cancel", "drain"])
    def test_a_cancel_or_drain_shows_before_on_change_is_called(self, stop: str) -> None:
        driver = MemoryDriver(MemoryStore(), controlled=True)
        reader = endpoint(driver, ["A"])
        seen: list[tuple[list[RowStatus], bool]] = []
        op = driver.get(
            reader.rows(),
            on_change=lambda op: seen.append((statuses(driver, op), driver.access_ended(op))),
        )
        if stop == "cancel":
            driver.cancel(op)
        else:
            assert driver.drain(TIMEOUT) is True
        assert seen == [([RowStatus.CANCELLED], True)]
