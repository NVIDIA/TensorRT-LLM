# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mooncake driver over a fake SDK that moves real bytes.

Pins the mapping of SDK 0.3.13 answers to row results, the put session bookkeeping (every session
the driver starts is ended, revoked or left open; only rows written whole are ended), the access-end
rules (a put scatter that returns ends access, whatever it answers; a ``batch_get_into`` ends it
unless a row answers TRANSFER_FAIL; a call that raises or miscounts does not, and its access is
unprovable), registration, drain and close, the parts and their scheduling (parts cut by rows
and bytes, gets first with a waiting put after GETS_BEFORE_PUT_IS_DUE gets, puts leaving one
worker to gets, a fixed pool of workers, cancel and drain ending the rows no worker took,
``on_change`` once an operation is done), and a backend round trip over the fake SDK. The shared
``driver_conformance`` checks run over the fake SDK too.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter
from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import pytest
from cpu_fixtures import (
    PROMPT_S,
    WAIT_S,
    CpuRegion,
    CpuView,
    Endpoint,
    FakeHold,
    Rows,
    WakeCounter,
    object_key,
    store_key,
    wait_until,
)
from fake_mooncake import FakeMooncakeStore, FakePool, KeyGatedClient

from tensorrt_llm._torch.disaggregation.base.backend import Attempt, Backend, Capabilities
from tensorrt_llm._torch.disaggregation.blob import mooncake as mooncake_module
from tensorrt_llm._torch.disaggregation.blob import store as store_module
from tensorrt_llm._torch.disaggregation.blob.backend import BlobBackend, BlobConfig, RegionBinding
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    DriverRejected,
    RegionLayout,
    RegistrationError,
    RowBatch,
    RowIO,
    RowStatus,
    Span,
    StoreDriver,
    StoreError,
    concluded,
)
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import DriverConformance
from tensorrt_llm._torch.disaggregation.testing.host_memory import HostBuffer

# The driver's workers run until it drains; ``driver_workers_return`` drains every driver a test
# left open and waits for its workers, instead of the thread-leak check.
# An exception that escapes a thread fails the test; a test that ends a thread on purpose
# requests the ``thread_escapes`` fixture.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.threadleak(enabled=False),
    pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning"),
]

SLOT = 8
PUT_START = "batch_put_session_start"
PUT_RANGES = "batch_put_from_multi_buffer_ranges"
PUT_END = "batch_put_session_end"
PUT_REVOKE = "batch_put_session_revoke"
GET = "batch_get_into"
DATA_CALL = {"put": PUT_RANGES, "get": GET}
METADATA_CALL_FAILED = "existence check: METADATA_CALL_FAILED(-100000)"


@pytest.fixture(autouse=True)
def driver_workers_return(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    drivers = []
    original_init = MooncakeDriver.__init__

    def tracked_init(self, *args, **kwargs) -> None:
        drivers.append(self)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(MooncakeDriver, "__init__", tracked_init)
    yield
    for driver in drivers:
        try:
            driver.drain(0.0)
        except Exception:  # closed already, or a test's scripted failure
            pass
    wait_until(
        lambda: not any(t.name.startswith("mooncake-driver-") for t in threading.enumerate()),
        "the driver's workers to return",
    )


def key(name: str) -> str:
    return f"t/v1/{name}"


class Bench:
    """A ``MooncakeDriver`` over a fake SDK, with ``slots`` registered host slots; ``tuning`` goes
    to the driver."""

    def __init__(self, slots: int = 4, pool: FakePool | None = None, **tuning: int) -> None:
        self.fake = FakeMooncakeStore(pool)
        self.driver = MooncakeDriver(self.fake, **tuning)
        self.buffer = HostBuffer(SLOT * slots, fill=0x2D)
        layout = RegionLayout(SLOT, slots)
        self.handle = self.driver.register(self.buffer.address, self.buffer.nbytes, layout)

    def rows(self, *names: str) -> RowBatch:
        return RowBatch.of(
            [RowIO(key(n), (self.buffer.span(i * SLOT, SLOT),)) for i, n in enumerate(names)]
        )

    def store(self, **objects: bytes) -> None:
        for name, data in objects.items():
            self.fake.pool.objects[key(name)] = data

    def write(self, slot: int, data: bytes) -> None:
        self.buffer.write(data, slot * SLOT)

    def read(self, slot: int) -> bytes:
        return self.buffer.read(slot * SLOT, SLOT)

    def run(self, direction: str, *names: str, on_change=None):
        """The results and access end of a put or get of ``names`` once it settled; ``last`` is its
        handle."""
        rows = self.rows(*names)
        if direction == "put":
            self.last = self.driver.put(rows, on_change=on_change)
        else:
            self.last = self.driver.get(rows, on_change=on_change)
        return self.settle(self.last)

    def settle(self, op):
        """``op``'s results and access end, once its results concluded and its access ended or
        can never be proven to."""
        driver = self.driver
        wait_until(
            lambda: (
                concluded(driver.poll(op))
                and (driver.access_ended(op) or driver.access_unprovable(op))
            ),
            "the operation to settle",
        )
        return driver.poll(op), driver.access_ended(op)

    def fill_sources(self, direction: str, *names: str) -> None:
        """Gives a put of ``names`` its slots' bytes, or a get its objects in the pool."""
        for slot, name in enumerate(names):
            data = name.encode() * SLOT
            self.write(slot, data) if direction == "put" else self.store(**{name: data})

    def keys_of(self, method: str) -> list[str]:
        return [k for call in self.fake.calls_to(method) for k in call.args[0]]


def summary(results) -> list[tuple[str, str | None]]:
    return [(result.status.value, result.fault) for result in results]


def test_the_driver_is_a_store_driver_with_mooncake_capabilities() -> None:
    bench = Bench()
    assert isinstance(bench.driver, StoreDriver)
    assert bench.driver.capabilities == DriverCapabilities(
        get=True,
        put=True,
        cancel=True,
        probe=False,
        max_rows_per_op=None,
        hints=False,
        absent_is_miss=True,
        device_memory=False,
        multi_span=False,
    )


class DriverLeavingOperationsUnstarted(MooncakeDriver):
    """A Mooncake driver whose workers never take a part."""

    def _take(self):
        return None


class TestMooncakeDriverConformance(DriverConformance):
    """The shared driver checks over the fake SDK, with every scenario it can stage. Probe is not
    declared, so the blocked and failing presence queries are not staged."""

    def make_driver(self) -> StoreDriver:
        # A part per row, so every operation of more than one row runs in several parts.
        return MooncakeDriver(FakeMooncakeStore(), rows_per_call=1)

    def make_blocked_get(self):
        sdk = KeyGatedClient(FakeMooncakeStore(), GET, key("blocked"))
        return MooncakeDriver(sdk), key("blocked"), sdk.entered, sdk.release.set

    def make_blocked_put(self):
        sdk = KeyGatedClient(FakeMooncakeStore(), PUT_RANGES, key("blocked"))
        return MooncakeDriver(sdk), key("blocked"), sdk.entered, sdk.release.set

    def make_held_key(self):
        pool = FakePool()
        writer = FakeMooncakeStore(pool)  # another client, still writing the key
        assert writer.batch_put_session_start([key("held")], [SLOT]) == [0]

        def release() -> None:
            writer.batch_put_session_revoke([key("held")])

        return MooncakeDriver(FakeMooncakeStore(pool)), key("held"), release

    def make_full_store(self):
        fake = FakeMooncakeStore()
        # Room for one more object as the check uses it: its one two-row put finds room for the
        # first only.
        fake.script(PUT_START, [None, -200])
        return MooncakeDriver(fake)

    def make_crash_after_record(self):
        fake = FakeMooncakeStore()
        fake.script(PUT_START, RuntimeError("the SDK crashed"))
        fake.script(GET, RuntimeError("the SDK crashed"))
        return MooncakeDriver(fake)

    def make_driver_leaving_operations_unstarted(self):
        return DriverLeavingOperationsUnstarted(FakeMooncakeStore())

    def make_failing_get(self):
        fake = FakeMooncakeStore()
        fake.script(GET, [-900])  # the next read fails with RPC_FAIL; the check reads the key once
        return MooncakeDriver(fake), key("failing")

    def make_unprovable_get(self):
        fake = FakeMooncakeStore()
        fake.script(GET, [-800])  # a transfer future gave up while its transfer may still run
        return MooncakeDriver(fake), lambda op: None  # the scripted answer concludes it


# Parts and their scheduling.


PART_LIMITS = {
    # name: (rows_per_call, bytes_per_call, rows in each part)
    "bytes": (10, 2 * SLOT + 1, [2] * 5),
    "rows": (4, 10 * SLOT, [4, 4, 2]),
    "whichever_comes_first": (3, 2 * SLOT, [2] * 5),
    "a_row_over_bytes_per_call_goes_alone": (10, 1, [1] * 10),
    "no_cut": (10, 10 * SLOT, [10]),
}


@pytest.mark.parametrize("case", PART_LIMITS)
def test_an_operation_of_any_size_runs_in_parts_cut_by_rows_and_bytes(case: str) -> None:
    rows_per_call, bytes_per_call, expected = PART_LIMITS[case]
    bench = Bench(slots=10, workers=1, rows_per_call=rows_per_call, bytes_per_call=bytes_per_call)
    names = list("ABCDEFGHIJ")  # one byte each: fill_sources repeats it over the slot
    bench.fill_sources("put", *names)
    results, access_ended = bench.run("put", *names)
    assert summary(results) == [("ready", None)] * 10 and access_ended
    assert [len(call.args[0]) for call in bench.fake.calls_to(PUT_RANGES)] == expected
    assert all(bench.fake.pool.objects[key(name)] == name.encode() * SLOT for name in names)


def test_the_driver_refuses_tuning_that_is_not_a_positive_int() -> None:
    for name in ("workers", "rows_per_call", "bytes_per_call"):
        for value in (0, True, 1.5):
            with pytest.raises(ValueError, match=f"{name} must be a positive int"):
                MooncakeDriver(FakeMooncakeStore(), **{name: value})


def data_calls(bench: Bench) -> list[str]:
    """``get`` or ``put`` for each SDK call that moves rows or starts to, in call order: a
    ``batch_get_into``, or the session start of a put."""
    names = {GET: "get", PUT_START: "put"}
    return [names[call.method] for call in bench.fake.calls if call.method in names]


def test_a_waiting_put_goes_next_once_it_is_due() -> None:
    bench = Bench(slots=3 * mooncake_module.GETS_BEFORE_PUT_IS_DUE + 3, workers=1)
    fetched = [f"F{i}" for i in range(3 * mooncake_module.GETS_BEFORE_PUT_IS_DUE + 1)]
    bench.store(**{name: bytes(SLOT) for name in fetched})
    gate = bench.fake.block(GET)
    ops = [bench.driver.get(bench.rows(fetched[0]))]
    assert gate.entered.wait(WAIT_S)  # the only worker is inside the SDK
    # Each row is an operation of its own; the slots are reused, as no assertion reads them.
    ops += [bench.driver.get(bench.rows(name)) for name in fetched[1:]]
    ops += [bench.driver.put(bench.rows(name)) for name in ("P1", "P2")]
    gate.release.set()
    for op in ops:
        bench.settle(op)
    # The blocked get was taken before any put waited, so it does not count toward one's turn.
    gets_then_a_put = ["get"] * mooncake_module.GETS_BEFORE_PUT_IS_DUE + ["put"]
    expected = ["get", *gets_then_a_put * 2]
    assert data_calls(bench)[: len(expected)] == expected


def test_only_gets_taken_while_a_put_waits_count_toward_its_turn() -> None:
    bench = Bench(slots=2, workers=1)
    bench.store(**{f"F{i}": bytes(SLOT) for i in range(mooncake_module.GETS_BEFORE_PUT_IS_DUE + 2)})
    for i in range(mooncake_module.GETS_BEFORE_PUT_IS_DUE):  # taken with no put waiting
        bench.run("get", f"F{i}")
    gate = bench.fake.block(GET)
    held = bench.driver.get(bench.rows(f"F{mooncake_module.GETS_BEFORE_PUT_IS_DUE}"))
    assert gate.entered.wait(WAIT_S)
    put = bench.driver.put(bench.rows("P"))
    late = bench.driver.get(bench.rows(f"F{mooncake_module.GETS_BEFORE_PUT_IS_DUE + 1}"))
    gate.release.set()
    for op in (held, put, late):
        bench.settle(op)
    gets = ["get"] * (mooncake_module.GETS_BEFORE_PUT_IS_DUE + 2)
    assert data_calls(bench) == [*gets, "put"]  # the late get goes first


def test_a_cancelled_put_leaves_its_turn_to_no_later_put() -> None:
    due = mooncake_module.GETS_BEFORE_PUT_IS_DUE
    bench = Bench(slots=4, workers=1)
    fetched = [f"F{i}" for i in range(due + 2)]
    bench.store(**{name: bytes(SLOT) for name in fetched})
    bench.fill_sources("put", "P", "Q")
    gates = [bench.fake.block(GET) for _ in range(due + 1)]
    ops = [bench.driver.get(bench.rows(fetched[0]))]
    assert gates[0].entered.wait(WAIT_S)  # taken before any put waited
    put = bench.driver.put(bench.rows("P"))
    ops += [bench.driver.get(bench.rows(name)) for name in fetched[1 : due + 1]]
    for gate in gates[:due]:
        gate.release.set()
    assert gates[due].entered.wait(WAIT_S)  # the gets ahead of P made it due
    bench.driver.cancel(put)
    calls_before = len(data_calls(bench))
    rows = bench.rows("Q", fetched[due + 1])
    ops += [bench.driver.put(rows[:1]), bench.driver.get(rows[1:])]
    gates[due].release.set()
    for op in ops:
        bench.settle(op)
    # No get went ahead of Q, which started waiting with no other put queued: gets go first.
    assert data_calls(bench)[calls_before:] == ["get", "put"]


def test_puts_leave_one_worker_to_gets() -> None:
    bench = Bench(slots=3, workers=2)
    bench.fill_sources("put", "P1", "P2")
    bench.store(F=b"ffffffff")
    gate = bench.fake.block(PUT_RANGES)
    rows = bench.rows("P1", "P2", "F")
    p1, p2 = bench.driver.put(rows[:1]), bench.driver.put(rows[1:2])
    try:
        assert gate.entered.wait(WAIT_S)
        time.sleep(PROMPT_S)  # P2 would have taken the other worker by now
        assert bench.keys_of(PUT_START) == [key("P1")]
        f = bench.driver.get(rows[2:])
        assert summary(bench.settle(f)[0]) == [("ready", None)]
        time.sleep(PROMPT_S)  # the worker F freed would have taken P2 by now
        assert bench.keys_of(PUT_START) == [key("P1")]
    finally:
        gate.release.set()
    for op in (p1, p2):
        assert summary(bench.settle(op)[0]) == [("ready", None)]


class OnChangeRecorder:
    """An ``on_change`` that keeps the handle it is called with and what ``poll`` then shows."""

    def __init__(self, driver: MooncakeDriver) -> None:
        self.driver = driver
        self.handles: list = []
        self.shown: list[list[tuple]] = []

    def __call__(self, op) -> None:
        self.handles.append(op)
        self.shown.append(summary(self.driver.poll(op)))

    def shown_for(self, op) -> list[list[tuple]]:
        """What ``poll`` showed at each call, each of which named ``op``."""
        assert all(handle is op for handle in self.handles)
        return self.shown


def test_a_cancel_ends_only_the_rows_no_worker_took() -> None:
    bench = Bench(workers=1, bytes_per_call=SLOT)
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb", C=b"cccccccc")
    gate = bench.fake.block(GET)
    recorder = OnChangeRecorder(bench.driver)
    op = bench.driver.get(bench.rows("A", "B", "C"), on_change=recorder)
    try:
        assert gate.entered.wait(WAIT_S)
        bench.driver.cancel(op)
        shown = summary(bench.driver.poll(op))
        assert shown == [("pending", None), ("cancelled", None), ("cancelled", None)]
        assert not bench.driver.access_ended(op)  # A's part is still in the SDK
        assert recorder.shown == []  # A has not concluded, so there is nothing to act on yet
    finally:
        gate.release.set()
    results, access_ended = bench.settle(op)
    assert summary(results) == [("ready", None), ("cancelled", None), ("cancelled", None)]
    assert access_ended and bench.keys_of(GET) == [key("A")]
    wait_until(lambda: recorder.shown, "on_change once A's part ended")
    assert recorder.shown_for(op) == [summary(results)]
    assert bench.read(1) == bench.read(2) == b"-" * SLOT
    bench.driver.cancel(op)  # nothing left to cancel
    assert bench.driver.drain(WAIT_S) is True


def test_drain_cancels_the_rows_no_worker_took() -> None:
    bench = Bench(workers=1, bytes_per_call=SLOT)
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb")
    gate = bench.fake.block(GET)
    op = bench.driver.get(bench.rows("A", "B"))
    try:
        assert gate.entered.wait(WAIT_S)
        assert bench.driver.drain(0.0) is False  # A's part is still in the SDK
    finally:
        gate.release.set()
    results, access_ended = bench.settle(op)
    assert summary(results) == [("ready", None), ("cancelled", None)] and access_ended
    assert bench.keys_of(GET) == [key("A")]
    assert bench.driver.drain(WAIT_S) is True


def test_a_part_whose_access_never_ends_leaves_the_rest_of_its_operation_to_run() -> None:
    bench = Bench(workers=1, bytes_per_call=SLOT)  # a part per row
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb")
    bench.fake.script(GET, [-800])  # A's transfer may still run
    results, access_ended = bench.run("get", "A", "B")
    assert [status for status, _ in summary(results)] == ["fault", "ready"]
    assert not access_ended and bench.driver.access_unprovable(bench.last)
    assert bench.keys_of(GET) == [key("A"), key("B")]


def test_an_on_change_that_raises_leaves_the_driver_serving() -> None:
    bench = Bench(workers=1)
    bench.store(A=b"aaaaaaaa")
    raised = []

    def raise_once(*_) -> None:
        if not raised:
            raised.append(True)
            raise KeyError("boom")

    for _ in range(2):  # the second runs only if the first left the worker serving
        results, access_ended = bench.run("get", "A", on_change=raise_once)
        assert summary(results) == [("ready", None)] and access_ended
    assert raised and bench.driver.drain(WAIT_S) is True


def test_a_warning_that_raises_leaves_the_part_as_it_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(workers=1)

    def broken(*_) -> None:
        raise KeyError("a logging bug")

    monkeypatch.setattr(mooncake_module, "_warn", broken)
    bench.fake.script(GET, [-703])  # NOT_READY: a skipped row, which the part warns of
    results, access_ended = bench.run("get", "A")
    assert summary(results) == [("not_ready", None)] and access_ended
    assert bench.driver.drain(WAIT_S) is True


def test_the_driver_calls_on_change_once_the_operation_is_done() -> None:
    bench = Bench(workers=1, bytes_per_call=SLOT)  # a part per row, run one after the other
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb")
    recorder = OnChangeRecorder(bench.driver)
    op = bench.driver.get(bench.rows("A", "B"), on_change=recorder)
    bench.settle(op)
    wait_until(lambda: recorder.shown, "on_change once B's part ended")
    # Not after A's part: a caller does nothing with some rows concluded.
    assert recorder.shown_for(op) == [[("ready", None), ("ready", None)]]


def get_into_slot(bench: Bench, name: str, slot: int) -> RowBatch:
    """One get row of ``name`` read into ``slot``."""
    return RowBatch.of([RowIO(key(name), (bench.buffer.span(slot * SLOT, SLOT),))])


def test_a_get_goes_ahead_of_a_due_put_that_has_no_worker_to_run_on() -> None:
    bench = Bench(slots=3, workers=2)  # puts occupy at most one worker
    bench.fill_sources("put", "P1", "P2")
    names = [f"F{i}" for i in range(mooncake_module.GETS_BEFORE_PUT_IS_DUE + 2)]
    bench.store(**{name: bytes(SLOT) for name in names})
    gate = bench.fake.block(PUT_RANGES)
    rows = bench.rows("P1", "P2")
    p1, p2 = bench.driver.put(rows[:1]), bench.driver.put(rows[1:])
    try:
        assert gate.entered.wait(WAIT_S)  # P1 holds the put worker; P2 waits for it
        # Past GETS_BEFORE_PUT_IS_DUE gets P2 is due, but it still cannot run: the gets go on.
        for name in names:
            op = bench.driver.get(get_into_slot(bench, name, 2))
            assert summary(bench.settle(op)[0]) == [("ready", None)]
    finally:
        gate.release.set()
    for op in (p1, p2):
        assert summary(bench.settle(op)[0]) == [("ready", None)]


def test_gets_that_go_ahead_of_a_put_waiting_for_a_put_worker_count_toward_its_turn() -> None:
    n = mooncake_module.GETS_BEFORE_PUT_IS_DUE
    bench = Bench(slots=3, workers=2)  # puts occupy at most one worker
    bench.fill_sources("put", "P1", "P2")
    names = [f"F{i}" for i in range(n + 2)]
    bench.store(**{name: bytes(SLOT) for name in names})
    put_gate, get_gate = bench.fake.block(PUT_RANGES), None
    rows = bench.rows("P1", "P2")
    p1, p2 = bench.driver.put(rows[:1]), bench.driver.put(rows[1:])
    try:
        assert put_gate.entered.wait(WAIT_S)  # P1 holds the put worker; P2 waits for it
        for name in names[:n]:  # each goes ahead of P2, on the other worker
            bench.settle(bench.driver.get(get_into_slot(bench, name, 2)))
        get_gate = bench.fake.block(GET)
        held = bench.driver.get(get_into_slot(bench, names[n], 2))
        assert get_gate.entered.wait(WAIT_S)  # the other worker is busy too
        late = bench.driver.get(get_into_slot(bench, names[n + 1], 2))
        put_gate.release.set()  # P1's worker is the only one free: P2's turn came, so P2 first
        for op in (p1, p2, late):
            bench.settle(op)
        assert data_calls(bench)[-2:] == ["put", "get"]
    finally:
        put_gate.release.set()
        if get_gate is not None:
            get_gate.release.set()
    bench.settle(held)


def test_worker_threads_never_keep_the_process_alive() -> None:
    bench = Bench()
    workers = [t for t in threading.enumerate() if t.name.startswith("mooncake-driver-")]
    assert workers and all(thread.daemon for thread in workers)
    assert bench.driver.drain(WAIT_S) is True


def test_drain_waits_for_every_worker_thread_to_exit() -> None:
    bench = Bench(workers=1)
    bench.store(A=b"aaaaaaaa")
    in_on_change, leave_on_change = threading.Event(), threading.Event()

    def held_on_change(_op) -> None:
        in_on_change.set()
        leave_on_change.wait(WAIT_S)

    op = bench.driver.get(bench.rows("A"), on_change=held_on_change)
    assert in_on_change.wait(WAIT_S)  # the part ended; its worker thread is inside on_change
    try:
        assert bench.driver.access_ended(op)
        assert bench.driver.drain(0.05) is False  # the worker thread has not exited
        threading.Timer(0.1, leave_on_change.set).start()
        started = time.monotonic()
        assert bench.driver.drain(WAIT_S) is True
        assert time.monotonic() - started < WAIT_S / 2  # woken as the thread exited
    finally:
        leave_on_change.set()


def test_the_conformance_suite_runs_its_checks_of_a_get_held_inside_the_sdk() -> None:
    # put and get return before the SDK answers, so those checks must run rather than skip.
    assert TestMooncakeDriverConformance.returns_before_the_store_answers


def test_drain_calls_on_change_for_the_operations_it_ended_before_it_waits() -> None:
    bench = Bench(workers=1)
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb")
    gate = bench.fake.block(GET)
    rows = bench.rows("A", "B")
    running = bench.driver.get(rows[:1])
    assert gate.entered.wait(WAIT_S)
    recorder = OnChangeRecorder(bench.driver)
    queued = bench.driver.get(rows[1:], on_change=recorder)
    drained: list[bool] = []
    drainer = threading.Thread(target=lambda: drained.append(bench.driver.drain(WAIT_S)))
    drainer.start()
    try:
        wait_until(lambda: recorder.shown, "on_change once B was cancelled")
        assert recorder.shown_for(queued) == [[("cancelled", None)]]
        assert not drained  # still waiting for A's part, which is in the SDK
    finally:
        gate.release.set()
        drainer.join(WAIT_S)
    assert drained == [True]
    assert summary(bench.settle(running)[0]) == [("ready", None)]


def test_drain_waits_out_a_budget_of_any_size() -> None:
    bench = Bench(workers=1)
    bench.store(A=b"aaaaaaaa")
    gate = bench.fake.block(GET)
    op = bench.driver.get(bench.rows("A"))
    assert gate.entered.wait(WAIT_S)
    threading.Timer(PROMPT_S, gate.release.set).start()
    assert bench.driver.drain(math.inf) is True  # once A's part ended
    assert summary(bench.settle(op)[0]) == [("ready", None)]


@pytest.fixture
def thread_escapes() -> Iterator[list[BaseException]]:
    """Catches the exceptions that end threads while a test that ends one on purpose runs, and
    checks afterwards that each was a ``SystemExit``. The hook is put back before the drivers are
    drained, so an escape there still reaches pytest and fails the test."""
    escapes: list[BaseException] = []
    pytest_hook = threading.excepthook
    threading.excepthook = lambda args: escapes.append(args.exc_value)
    try:
        yield escapes
    finally:
        threading.excepthook = pytest_hook
    assert all(isinstance(escape, SystemExit) for escape in escapes), escapes


def wait_for_an_escape(escapes: list[BaseException]) -> None:
    """Waits until a thread ended by an exception: a thread runs its excepthook only after the
    driver saw it leave, so a test that ends a thread waits for this before it ends."""
    wait_until(lambda: escapes, "a thread to end by an exception")


def test_a_part_whose_worker_thread_stops_faults_and_the_other_workers_go_on(
    thread_escapes: list[BaseException],
) -> None:
    bench = Bench(workers=2, bytes_per_call=SLOT)  # a part per row
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb")
    bench.fake.script(GET, SystemExit())  # escapes every handler: A's worker thread ends
    results, access_ended = bench.run("get", "A", "B")
    assert summary(results) == [
        ("fault", "the worker stopped before the row was decided"),
        ("ready", None),  # on the other worker
    ]
    assert not access_ended and bench.driver.access_unprovable(bench.last)
    wait_for_an_escape(thread_escapes)


def test_once_no_worker_is_left_queued_rows_fault_and_operations_are_rejected(
    thread_escapes: list[BaseException],
) -> None:
    bench = Bench(workers=1, bytes_per_call=SLOT)  # a part per row
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb")
    bench.fake.script(GET, SystemExit())  # A's part ends the only worker thread
    recorder = OnChangeRecorder(bench.driver)
    op = bench.driver.get(bench.rows("A", "B"), on_change=recorder)
    wait_until(lambda: len(recorder.shown) == 2, "A's end, then B's")
    assert recorder.shown_for(op)[-1] == [
        ("fault", "the worker stopped before the row was decided"),
        ("fault", "no worker thread is left to run the row"),  # B waited for a worker in vain
    ]
    assert bench.driver.access_unprovable(op)  # A's SDK call may still touch its slot
    with pytest.raises(DriverRejected, match="no worker thread is left"):
        bench.driver.get(bench.rows("A"))
    wait_for_an_escape(thread_escapes)


def test_a_worker_that_cannot_start_stops_the_ones_already_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = threading.Thread.start
    started: list[threading.Thread] = []

    def refuse_the_second(thread: threading.Thread) -> None:
        if thread.name.startswith("mooncake-driver-"):
            if started:
                raise RuntimeError("can't start new thread")
            started.append(thread)
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", refuse_the_second)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        MooncakeDriver(FakeMooncakeStore(), workers=2)
    (first,) = started
    first.join(WAIT_S)
    assert not first.is_alive()


def test_drain_wakes_as_a_worker_thread_leaves_on_an_escape(
    thread_escapes: list[BaseException],
) -> None:
    bench = Bench(workers=1)
    bench.store(A=b"aaaaaaaa")
    leave = threading.Event()

    def tell_then_escape(*_) -> None:
        leave.wait(WAIT_S)
        raise SystemExit  # past every handler: the thread exits after its part ended

    results, access_ended = bench.run("get", "A", on_change=tell_then_escape)
    assert summary(results) == [("ready", None)] and access_ended
    threading.Timer(0.1, leave.set).start()
    started = time.monotonic()
    assert bench.driver.drain(WAIT_S) is True
    assert time.monotonic() - started < WAIT_S / 2  # woken as the thread left, not at the budget
    wait_for_an_escape(thread_escapes)


# Answers to row results.


def skipped_counts(bench: Bench) -> dict[str, int]:
    return {name: n for name, n in bench.driver.stats().items() if name.startswith("rows_") and n}


GET_ANSWERS = {
    # name: (batch_get_into's answer (None: real), status, fault, the stat counting a skipped row)
    "object_size": (None, "ready", None, None),  # unscripted
    "short_object": (5, "fault", "object is 5 bytes, expected 8", None),
    "OBJECT_NOT_FOUND": (-704, "miss", None, None),
    "REPLICA_IS_NOT_READY": (-703, "not_ready", None, "rows_not_ready_no_complete_replica"),
    "INVALID_REPLICA": (-702, "fault", "batch_get_into: INVALID_REPLICA(-702)", None),
    "INVALID_PARAMS": (-600, "fault", "batch_get_into: INVALID_PARAMS(-600)", None),
    "RPC_FAIL": (-900, "fault", "batch_get_into: RPC_FAIL(-900)", None),
    "TRANSFER_FAIL": (-800, "fault", "batch_get_into: TRANSFER_FAIL(-800)", None),
    "unknown_code": (-9999, "fault", "batch_get_into: code(-9999)", None),
}


@pytest.mark.parametrize("case", GET_ANSWERS)
def test_get_answers_map_to_row_statuses(case: str) -> None:
    answer, status, fault, counted = GET_ANSWERS[case]
    bench = Bench()
    bench.store(A=b"aaaaaaaa")
    bench.fake.script(GET, [answer])
    results, access_ended = bench.run("get", "A")
    assert summary(results) == [(status, fault)]
    assert skipped_counts(bench) == ({counted: 1} if counted else {})
    assert (bench.read(0) == b"aaaaaaaa") == (status == "ready")
    assert access_ended is (answer != -800)
    assert bench.driver.access_unprovable(bench.last) is (answer == -800)


PUT_START_ANSWERS = {
    # name: (the session start's answer, status, fault, the stat counting a skipped row)
    "NO_AVAILABLE_HANDLE": (-200, "declined", None, "rows_declined_pool_full"),
    # This driver holds no session of the key: not a session of its own, but, say, a cachelib
    # master's object cap.
    "INVALID_PARAMS": (-600, "fault", "put session start: INVALID_PARAMS(-600)", None),
    "INVALID_REPLICA": (-702, "fault", "put session start: INVALID_REPLICA(-702)", None),
    "RPC_FAIL": (-900, "fault", "put session start: RPC_FAIL(-900)", None),
    "unknown_code": (-9999, "fault", "put session start: code(-9999)", None),
}


@pytest.mark.parametrize("case", PUT_START_ANSWERS)
def test_put_session_start_answers_map_to_row_statuses(case: str) -> None:
    answer, status, fault, counted = PUT_START_ANSWERS[case]
    bench = Bench()
    bench.fake.script(PUT_START, [answer])
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [(status, fault)]
    assert skipped_counts(bench) == ({counted: 1} if counted else {})
    assert access_ended
    (start,) = bench.fake.calls_to(PUT_START)
    assert len(start.args) == 2 and start.kwargs == {}  # never a replicate config
    for method in (PUT_RANGES, PUT_END, PUT_REVOKE, "batch_is_exist"):
        assert bench.fake.calls_to(method) == []  # a session it did not start is not its own


PUT_WRITE_ANSWERS = {
    # case: (the scatter's answer, the end's answer (None: real), status, fault, the session's end:
    #        ended, revoked, left_for_close or left_to_master)
    "written": (None, None, "ready", None, "ended"),
    "short_write": (5, None, "fault", "put ranges: moved 5 of 8 bytes", "revoked"),
    # A TCP target may still write into the object's space.
    "transfer_fail": (-800, None, "fault", "put ranges: TRANSFER_FAIL(-800)", "left_to_master"),
    "session_lost": (-600, None, "fault", "put ranges: INVALID_PARAMS(-600)", "revoked"),
    "end_refused": (None, -704, "fault", "put session end: OBJECT_NOT_FOUND(-704)", "revoked"),
    "end_not_owner": (None, -601, "fault", "put session end: ILLEGAL_CLIENT(-601)", "revoked"),
    # The end may still reach the master.
    "end_timed_out": (None, -901, "fault", "put session end: RPC_TIMEOUT(-901)", "left_for_close"),
}


@pytest.mark.parametrize("case", PUT_WRITE_ANSWERS)
def test_put_ranges_and_end_answers_map_to_row_statuses(case: str) -> None:
    ranges, end, status, fault, session = PUT_WRITE_ANSWERS[case]
    bench = Bench()
    bench.write(0, b"aaaaaaaa")
    bench.fake.script(PUT_RANGES, [ranges])
    bench.fake.script(PUT_END, [end])
    bench.fake.script(PUT_END, [end])  # the retry of an unanswered end
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [(status, fault)]
    assert access_ended
    assert (bench.fake.pool.objects.get(key("A")) == b"aaaaaaaa") == (status == "ready")
    assert bench.keys_of(PUT_REVOKE) == ([key("A")] if session == "revoked" else [])
    stats = bench.driver.stats()
    assert stats["put_sessions_left_for_close"] == (session == "left_for_close")
    assert stats["put_sessions_left_to_master"] == (session == "left_to_master")


EXISTENCE_ANSWERS = {
    # batch_is_exist's step: the results of the two rows that already existed
    "committed_and_not": ([1, 0], [("ready", None), ("not_ready", None)]),
    "check_failed": ([1, -900], [("ready", None), ("fault", "existence check: RPC_FAIL(-900)")]),
    # A metadata call touches no row memory: its raise fails only its rows.
    "check_raised": (RuntimeError("master gone"), [("fault", METADATA_CALL_FAILED)] * 2),
}


@pytest.mark.parametrize("case", EXISTENCE_ANSWERS)
def test_only_rows_that_already_exist_are_checked_for_existence(case: str) -> None:
    exist, expected = EXISTENCE_ANSWERS[case]
    bench = Bench()
    # A starts, B and C exist (OBJECT_ALREADY_EXISTS), D finds the pool full.
    bench.fake.script(PUT_START, [None, -705, -705, -200])
    bench.fake.script("batch_is_exist", exist)
    results, access_ended = bench.run("put", "A", "B", "C", "D")
    assert summary(results) == [("ready", None), *expected, ("declined", None)]
    assert bench.keys_of("batch_is_exist") == [key("B"), key("C")]
    assert access_ended


# Session bookkeeping.


def reply_lost(answers):
    """The end committed every object and dropped the sessions, but the reply never came."""
    raise RuntimeError("reply lost")


class SessionScenario(NamedTuple):
    """A put of rows A, B and C with some SDK calls scripted, and what must come of its sessions."""

    steps: dict  # method -> one FakeMooncakeStore.script step per call, in call order
    rows: str  # each row ready (R) or faulted (F)
    access_ends: bool = True
    revoked: str = ""  # rows whose session the put revokes, or tries to
    left_for_close: str = ""  # rows whose session is left open for close to revoke
    left_to_master: str = ""  # rows whose session is left open for the master alone to free


SESSION_SCENARIOS = {
    "all_written": SessionScenario({}, "RRR"),
    # A TCP target may still write into B's object space: only the master may free it.
    "transfer_fail": SessionScenario({PUT_RANGES: [[None, -800, None]]}, "RFR", left_to_master="B"),
    "short_write": SessionScenario({PUT_RANGES: [[None, 5, None]]}, "RFR", revoked="B"),
    "end_retried": SessionScenario({PUT_END: [[-900, None, None], [None]]}, "RRR"),
    # Ends that may still reach the master.
    "end_fails_twice": SessionScenario(
        {PUT_END: [[-901, None, None], [-900]]}, "FRR", left_for_close="A"
    ),
    "end_retry_refused": SessionScenario(
        {PUT_END: [[-901, None, None], [-704]]}, "FRR", left_for_close="A"
    ),
    "end_retry_raises": SessionScenario(
        {PUT_END: [[-901, None, None], RuntimeError("x")]}, "FRR", left_for_close="A"
    ),
    "end_raises": SessionScenario(
        {PUT_END: [RuntimeError("rpc lost")]}, "FFF", left_for_close="ABC"
    ),
    "end_committed_then_raised": SessionScenario(
        {PUT_END: [reply_lost]}, "FFF", left_for_close="ABC"
    ),
    "end_and_revoke_fail": SessionScenario(
        {PUT_END: [[-704, None, None]], PUT_REVOKE: [[-900]]},
        "FRR",
        revoked="A",
        left_for_close="A",
    ),
    # The SDK may have opened any of the sessions.
    "start_raises": SessionScenario(
        {PUT_START: [RuntimeError("rpc lost")]}, "FFF", left_for_close="ABC"
    ),
    "start_miscounts": SessionScenario(
        {PUT_START: [lambda answers: answers[:-1]]}, "FFF", left_for_close="ABC"
    ),
    # The scatter may still be running: nothing is ended or revoked.
    "ranges_raise": SessionScenario(
        {PUT_RANGES: [RuntimeError("bad pointer")]}, "FFF", access_ends=False, left_to_master="ABC"
    ),
    "ranges_miscount": SessionScenario(
        {PUT_RANGES: [lambda answers: answers[:-1]]}, "FFF", access_ends=False, left_to_master="ABC"
    ),
}


@pytest.mark.parametrize("name", SESSION_SCENARIOS)
def test_every_started_put_session_is_ended_revoked_or_left_open(name: str) -> None:
    scenario = SESSION_SCENARIOS[name]
    bench = Bench()
    for slot, data in enumerate((b"aaaaaaaa", b"bbbbbbbb", b"cccccccc")):
        bench.write(slot, data)
    for method, method_steps in scenario.steps.items():
        for step in method_steps:
            bench.fake.script(method, step)
    results, access_ended = bench.run("put", "A", "B", "C")
    assert "".join(r.status.value[0].upper() for r in results) == scenario.rows
    assert access_ended is scenario.access_ends

    assert bench.keys_of(PUT_REVOKE) == [key(n) for n in scenario.revoked]
    settled = Counter(bench.fake.keys_answered(PUT_END, 0) + bench.keys_of(PUT_REVOKE))
    assert set(settled.values()) <= {1}, f"a session ended or revoked twice: {settled}"
    for_close = {key(n) for n in scenario.left_for_close}
    to_master = {key(n) for n in scenario.left_to_master}
    stats = bench.driver.stats()
    assert stats["put_sessions_left_for_close"] == len(for_close)
    assert stats["put_sessions_left_to_master"] == len(to_master)
    # Every session the SDK still holds is left open; the SDK may have dropped one the driver
    # could not see end, as when an end landed but its reply did not.
    assert bench.fake.put_sessions <= for_close | to_master
    # Only a row written whole may be ended: its object becomes readable as it stands.
    written = bench.fake.keys_answered(PUT_RANGES, SLOT)
    assert set(bench.keys_of(PUT_END)) <= set(written)
    ready = [key(n) for n, r in zip("ABC", results) if r.status is RowStatus.READY]
    assert set(ready) == set(bench.fake.keys_answered(PUT_END, 0))
    # Close revokes the sessions left for it, never those left to the master; drain revokes none.
    if scenario.access_ends:
        revocations = len(bench.keys_of(PUT_REVOKE))
        assert bench.driver.drain(WAIT_S) is True
        assert len(bench.keys_of(PUT_REVOKE)) == revocations
        bench.driver.unregister(bench.handle)
        bench.driver.close()
        assert bench.driver.stats()["put_sessions_left_for_close"] == 0
        assert bench.driver.stats()["put_sessions_left_to_master"] == len(to_master)
        assert bench.fake.put_sessions == to_master
        assert not to_master & set(bench.keys_of(PUT_REVOKE))


def test_sessions_left_open_are_revoked_only_in_close() -> None:
    bench = Bench()
    bench.write(0, b"aaaaaaaa")
    bench.fake.script(PUT_END, [-704])
    bench.fake.script(PUT_REVOKE, [-900])
    bench.run("put", "A")
    assert bench.driver.stats()["put_sessions_left_for_close"] == 1
    bench.write(1, b"bbbbbbbb")
    bench.store(C=b"cccccccc")
    bench.run("put", "X", "B")
    bench.run("get", "C")
    assert bench.keys_of(PUT_REVOKE) == [key("A")]  # only its own first try
    results, _ = bench.run("put", "A")  # the driver holds A's session: it starts no other
    assert summary(results) == [("declined", None)]  # and no complete object exists
    assert bench.keys_of(PUT_START).count(key("A")) == 1

    assert bench.driver.drain(WAIT_S) is True  # sessions left open hold no lender memory
    assert bench.keys_of(PUT_REVOKE) == [key("A")]
    bench.driver.unregister(bench.handle)
    bench.fake.script(PUT_REVOKE, [-900])
    bench.driver.close()
    assert bench.keys_of(PUT_REVOKE) == [key("A")] * 2
    assert bench.driver.stats()["put_sessions_left_for_close"] == 1  # the master frees it later
    assert len(bench.fake.calls_to("close")) == 1


def test_drain_returns_at_once_while_a_revocation_would_block() -> None:
    bench = Bench()
    bench.fill_sources("put", "A")
    bench.fake.script(PUT_END, [-901])
    bench.fake.script(PUT_END, [-901])  # the end and its retry go unanswered
    bench.run("put", "A")
    assert bench.driver.stats()["put_sessions_left_for_close"] == 1
    gate = bench.fake.block(PUT_REVOKE)
    started = time.monotonic()
    assert bench.driver.drain(WAIT_S) is True  # once the idle workers exit
    assert time.monotonic() - started < PROMPT_S
    assert not gate.entered.is_set()
    gate.release.set()
    bench.driver.unregister(bench.handle)
    bench.driver.close()
    assert bench.keys_of(PUT_REVOKE) == [key("A")]
    assert bench.driver.stats()["put_sessions_left_for_close"] == 0
    assert bench.fake.put_sessions == set()


def test_close_closes_the_client_even_when_the_revocation_raises() -> None:
    bench = Bench()
    bench.fill_sources("put", "A")
    bench.fake.script(PUT_END, [-901])
    bench.fake.script(PUT_END, [-901])  # the end and its retry go unanswered
    bench.run("put", "A")
    assert bench.driver.drain(WAIT_S) is True
    bench.driver.unregister(bench.handle)
    bench.fake.script(PUT_REVOKE, SystemExit())  # escapes every handler
    with pytest.raises(SystemExit):
        bench.driver.close()
    assert len(bench.fake.calls_to("close")) == 1
    bench.driver.close()  # and is done
    assert len(bench.fake.calls_to("close")) == 1


def test_a_put_whose_worker_thread_exits_leaves_its_session_open_for_close(
    thread_escapes: list[BaseException],
) -> None:
    bench = Bench()
    bench.write(0, b"aaaaaaaa")
    bench.fake.script(PUT_END, SystemExit())  # escapes every handler: the worker thread ends
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [("fault", "the worker stopped before the row was decided")]
    assert not access_ended
    assert key("A") in bench.fake.put_sessions
    assert bench.driver.stats()["put_sessions_left_for_close"] == 1
    results, _ = bench.run("put", "A")  # the driver holds A's session: it starts no other
    assert summary(results) == [("declined", None)]
    assert bench.keys_of(PUT_START).count(key("A")) == 1
    wait_for_an_escape(thread_escapes)


def test_a_key_another_put_of_this_driver_holds_is_not_ready_without_asking_the_sdk() -> None:
    bench = Bench()
    bench.fill_sources("put", "A", "B")
    gate = bench.fake.block(PUT_RANGES)
    first = bench.driver.put(bench.rows("A"))
    try:
        assert gate.entered.wait(WAIT_S)  # A's session is open, its scatter held
        results, _ = bench.settle(bench.driver.put(bench.rows("A", "B")))
        assert summary(results) == [("not_ready", None), ("ready", None)]
        results, _ = bench.settle(bench.driver.put(bench.rows("A")))  # nothing left to start
        assert summary(results) == [("not_ready", None)]
        assert [call.args[0] for call in bench.fake.calls_to(PUT_START)] == [[key("A")], [key("B")]]
    finally:
        gate.release.set()
    bench.settle(first)
    assert bench.fake.pool.objects[key("A")] == b"A" * SLOT
    assert skipped_counts(bench) == {"rows_not_ready_held_by_own_put": 2}
    results, _ = bench.run("put", "A")  # A's put ended, so the key is free again
    assert bench.keys_of(PUT_START).count(key("A")) == 2
    assert summary(results) == [("ready", None)]  # it exists already, committed


@pytest.mark.parametrize("left", ["for_close", "to_master"])
def test_a_key_whose_session_was_left_open_is_ready_once_its_object_exists(left: str) -> None:
    bench = Bench()
    bench.fill_sources("put", "A")
    if left == "to_master":
        bench.fake.script(PUT_RANGES, [-800])
    else:  # the end and its retry go unanswered
        bench.fake.script(PUT_END, [-901])
        bench.fake.script(PUT_END, [-901])
    bench.run("put", "A")
    assert bench.driver.stats()[f"put_sessions_left_{left}"] == 1
    results, _ = bench.run("put", "A")
    assert summary(results) == [("declined", None)]  # no complete object, and none of its own
    assert skipped_counts(bench) == {"rows_declined_held_by_session_left_open": 1}
    # The master freed the unfinished put, and another client then wrote the key whole.
    bench.fake.pool.reserved.pop(key("A"))
    bench.store(A=b"a" * SLOT)
    results, _ = bench.run("put", "A")
    assert summary(results) == [("ready", None)]  # whoever wrote it
    assert bench.keys_of(PUT_START) == [key("A")]  # its session was never started again
    assert bench.keys_of("batch_is_exist") == [key("A")] * 2


def test_close_needs_a_drain_that_returned_true() -> None:
    bench = Bench()
    bench.fill_sources("put", "A")
    gate = bench.fake.block(PUT_RANGES)
    put = bench.driver.put(bench.rows("A"))
    try:
        assert gate.entered.wait(WAIT_S)
        assert bench.driver.drain(0.0) is False  # the put is still running
    finally:
        gate.release.set()
    bench.settle(put)
    bench.driver.unregister(bench.handle)
    with pytest.raises(DriverRejected, match="drain that returned True"):
        bench.driver.close()
    assert bench.fake.calls_to("close") == []
    assert bench.driver.drain(WAIT_S) is True
    bench.driver.close()
    assert len(bench.fake.calls_to("close")) == 1


def test_an_unprovable_put_leaves_its_session_to_the_master() -> None:
    bench = Bench()
    bench.fake.script(PUT_RANGES, RuntimeError("bad pointer"))
    _, access_ended = bench.run("put", "A")
    assert not access_ended
    assert bench.driver.drain(0.0) is False
    assert bench.fake.calls_to(PUT_REVOKE) == []
    assert bench.driver.stats()["put_sessions_left_to_master"] == 1


LEASE_EXPIRED_CASES = {
    # the first read's answers, the reread's, the results, whether access ended
    "reread_ready": ([None, -707, None], [None], "RRR", True),
    "expired_again": ([None, -707, None], [-707], "RFR", True),
    "first_read_failed": ([None, -800, -707], [None], "RFR", False),
    "reread_failed": ([None, None, -707], [-800], "RRF", False),
    # The first read already wrote the slot, so the reread's answer cannot make it a MISS or
    # NOT_READY.
    "reread_not_found": ([None, -707, None], [-704], "RFR", True),
    "reread_not_ready": ([None, -707, None], [-703], "RFR", True),
}


@pytest.mark.parametrize("case", LEASE_EXPIRED_CASES)
def test_rows_whose_lease_expired_are_reread_once_and_both_reads_must_end_access(case: str) -> None:
    first, reread, statuses, ended = LEASE_EXPIRED_CASES[case]
    bench = Bench()
    bench.fill_sources("get", "A", "B", "C")
    bench.fake.script(GET, first)
    bench.fake.script(GET, reread)
    results, access_ended = bench.run("get", "A", "B", "C")
    assert "".join(r.status.value[0].upper() for r in results) == statuses
    expired = [key(name) for name, answer in zip("ABC", first) if answer == -707]
    assert [call.args[0] for call in bench.fake.calls_to(GET)] == [[key(n) for n in "ABC"], expired]
    assert access_ended is ended
    assert bench.driver.drain(WAIT_S) is ended  # False at once when unprovable


def test_an_object_being_written_is_not_ready_for_put_and_get(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(mooncake_module.logger, "warning", warnings.append)
    pool = FakePool()
    writer = FakeMooncakeStore(pool)
    assert writer.batch_put_session_start([key("K")], [SLOT]) == [0]  # another client is writing
    bench = Bench(pool=pool)
    results, access_ended = bench.run("get", "K")
    assert summary(results) == [("not_ready", None)] and access_ended
    results, access_ended = bench.run("put", "K")
    assert summary(results) == [("not_ready", None)] and access_ended
    stats = bench.driver.stats()
    assert stats["rows_not_ready_no_complete_replica"] == stats["rows_not_ready_uncommitted"] == 1
    # Each part logs once it ended, a moment after its operation settled.
    for reason in ("no_complete_replica", "uncommitted"):
        wait_until(
            lambda: any(key("K") in line and f"not_ready ({reason})" in line for line in warnings),
            f"the {reason} warning",
        )


# Access end.


FAILURES = {
    "raises": RuntimeError("rpc lost"),
    "raises_value_error": ValueError("bad pointer"),
    "short": lambda answers: answers[:-1],
    "long": lambda answers: [*answers, 0],
    "empty": lambda answers: [],
}


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("call", ["put", "get", "get_reread"])
def test_access_stays_open_when_a_data_call_raises_or_miscounts(call: str, failure: str) -> None:
    bench = Bench()
    direction = "put" if call == "put" else "get"
    bench.fill_sources(direction, "A", "B")
    if call == "get_reread":
        bench.fake.script(GET, [None, -707])
    bench.fake.script(DATA_CALL[direction], FAILURES[failure])
    results, access_ended = bench.run(direction, "A", "B")  # nothing escapes the driver
    assert not access_ended and bench.driver.access_unprovable(bench.last)
    unproven = results[1:] if call == "get_reread" else results
    assert all(r.status is RowStatus.FAULT and " raised " in r.fault for r in unproven)
    if call == "get_reread":
        assert results[0].status is RowStatus.READY  # read whole the first time
    assert bench.driver.stats()["ops_unprovable"] == 1
    assert bench.driver.drain(0.0) is False


@pytest.mark.parametrize("direction", ["put", "get"])
def test_a_fault_past_the_sdk_calls_keeps_access_open(direction: str, monkeypatch) -> None:
    def broken(*_):
        raise ValueError("mapping bug")

    bench = Bench()
    if direction == "get":
        monkeypatch.setattr(mooncake_module, "_result_of_get", broken)
        bench.fake.script(GET, [5])  # a row read short goes past the all-whole shortcut
    else:
        monkeypatch.setattr(MooncakeDriver, "_end_sessions", broken)
    bench.fill_sources(direction, "A")
    results, access_ended = bench.run(direction, "A")
    assert summary(results) == [("fault", f"{direction} failed: ValueError('mapping bug')")]
    assert not access_ended
    # A put's session may still be open: the driver keeps holding the key.
    assert bench.driver.stats()["put_sessions_left_for_close"] == (direction == "put")


# Admission, registration, concurrency.


def test_bad_rows_are_rejected_before_the_sdk() -> None:
    bench = Bench()
    with pytest.raises(DriverRejected, match="at most"):
        bench.driver.put(RowBatch.of([RowIO(key("A"), (Span(bench.buffer.address, 1 << 31),))]))
    two_spans = (bench.buffer.span(0, SLOT), bench.buffer.span(2 * SLOT, SLOT))
    with pytest.raises(DriverRejected, match="one span"):
        bench.driver.put(RowBatch.of([RowIO(key("A"), two_spans)]))
    assert [c for c in bench.fake.calls if c.method.startswith("batch_")] == []


def test_a_batch_whose_extent_leaves_its_registration_is_rejected_before_the_sdk() -> None:
    bench = Bench()
    other = HostBuffer(SLOT)  # a second registration, apart from the first
    bench.driver.register(other.address, SLOT, RegionLayout(SLOT, 1))
    first, _ = bench.rows("A", "B")
    with pytest.raises(DriverRejected, match="outside every live registration"):
        bench.driver.get(RowBatch.of([first, RowIO(key("C"), (other.span(),))]))  # gap: no row
    assert [c for c in bench.fake.calls if c.method.startswith("batch_")] == []
    results, access_ended = bench.run("get", "A", "B")  # one registration holds them
    assert summary(results) == [("miss", None), ("miss", None)] and access_ended


def test_an_empty_batch_is_rejected_before_the_sdk() -> None:
    bench = Bench()
    empty = RowBatch.of_slots(
        "k/", np.zeros((0, 2), np.uint8), np.zeros(0, np.int64), SLOT, bench.buffer.span()
    )
    with pytest.raises(DriverRejected, match="at least one row"):
        bench.driver.get(empty)
    assert bench.driver.drain(WAIT_S) is True  # nothing was left in flight


def test_a_part_whose_sdk_lists_cannot_be_built_faults_and_ends_its_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = Bench()

    def broken(*_) -> list[str]:
        raise ValueError("a key bug")

    monkeypatch.setattr(RowBatch, "keys", broken)
    results, access_ended = bench.run("get", "A")
    assert summary(results) == [
        ("fault", "get failed before any SDK call: ValueError('a key bug')")
    ]
    assert access_ended and not bench.driver.access_unprovable(bench.last)
    assert bench.fake.calls_to(GET) == []
    assert bench.driver.drain(WAIT_S) is True  # no operation turned unprovable


def test_register_and_unregister_follow_the_protocol() -> None:
    fake = FakeMooncakeStore()
    driver = MooncakeDriver(fake)
    region, other = HostBuffer(64), HostBuffer(64)
    handle = driver.register(region.address, 64, RegionLayout(SLOT, 8))
    with pytest.raises(DriverRejected, match="layout needs"):
        driver.register(other.address, SLOT, RegionLayout(SLOT, 2))
    assert len(fake.calls_to("register_buffer")) == 1

    fake.script("register_buffer", [-1])
    with pytest.raises(RegistrationError, match="INTERNAL_ERROR") as refused:
        driver.register(other.address, 64, RegionLayout(SLOT, 8))
    assert refused.value.handle is None
    fake.script("register_buffer", RuntimeError("rpc lost"))
    with pytest.raises(RegistrationError, match="rpc lost") as unknown:
        driver.register(other.address, 64, RegionLayout(SLOT, 8))
    assert unknown.value.handle is not None
    with pytest.raises(StoreError):  # the SDK never kept it, so it cannot revoke it either
        driver.unregister(unknown.value.handle)

    def revocations() -> int:
        return [call.args for call in fake.calls_to("unregister_buffer")].count((region.address,))

    fake.script("unregister_buffer", [-1])
    with pytest.raises(StoreError, match="INTERNAL_ERROR"):
        driver.unregister(handle)
    with pytest.raises(DriverRejected, match="overlaps"):  # the failed revocation left it live
        driver.register(region.address + SLOT, SLOT, RegionLayout(SLOT, 1))
    driver.unregister(handle)  # and retryable
    assert revocations() == 2
    driver.unregister(handle)  # and is idempotent once it succeeded
    assert revocations() == 2


def wait_until_draining(driver: MooncakeDriver) -> None:
    """Returns once a drain waits: admission, under the drain's lock, refuses a stray row as
    draining rather than as outside every region. Nothing reaches the SDK."""
    stray = HostBuffer(SLOT)
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        with pytest.raises(DriverRejected) as refused:
            driver.get(RowBatch.of([RowIO(key("probe"), (stray.span(),))]))
        if "draining" in str(refused.value):
            return
    pytest.fail("no drain started")


def test_unregister_is_refused_and_drain_waits_while_a_put_runs() -> None:
    bench = Bench()
    bench.fill_sources("put", "A")
    gate = bench.fake.block(PUT_RANGES)
    putter = threading.Thread(target=lambda: bench.run("put", "A"))
    drained: list[bool] = []
    drainer = threading.Thread(target=lambda: drained.append(bench.driver.drain(WAIT_S)))
    putter.start()
    try:
        assert gate.entered.wait(WAIT_S)  # inside the scatter
        with pytest.raises(DriverRejected, match="may still access"):
            bench.driver.unregister(bench.handle)
        drainer.start()
        wait_until_draining(bench.driver)
        assert drainer.is_alive()  # still running, not unprovable: the drain waits
    finally:
        released = time.monotonic()
        gate.release.set()
        putter.join(WAIT_S)
    drainer.join(WAIT_S)
    assert drained == [True]
    assert time.monotonic() - released < PROMPT_S  # the put's end woke the drain
    bench.driver.unregister(bench.handle)
    assert len(bench.fake.calls_to("unregister_buffer")) == 1


def test_an_unprovable_operation_pins_its_region_and_fails_drain_at_once() -> None:
    bench = Bench()
    bench.fill_sources("get", "A", "B")
    bench.fake.script(GET, [-800])
    bench.run("get", "A")
    gate = bench.fake.block(GET)
    blocked = bench.driver.get(bench.rows("A", "B")[1:])
    try:
        assert gate.entered.wait(WAIT_S)  # another get is still running
        started = time.monotonic()
        assert bench.driver.drain(WAIT_S) is False  # waits for no running call
        assert time.monotonic() - started < PROMPT_S
    finally:
        gate.release.set()
    bench.settle(blocked)
    with pytest.raises(DriverRejected, match="may still access"):
        bench.driver.unregister(bench.handle)
    with pytest.raises(DriverRejected, match="drain"):
        bench.driver.close()
    assert bench.fake.calls_to("close") == []


def test_close_needs_a_drain_closes_the_sdk_client_once_and_then_queries_raise() -> None:
    bench = Bench()
    bench.fill_sources("put", "A")
    bench.run("put", "A")
    with pytest.raises(DriverRejected, match="drain"):
        bench.driver.close()
    assert bench.driver.drain(WAIT_S) is True
    bench.driver.unregister(bench.handle)
    bench.driver.close()
    bench.driver.close()  # idempotent
    assert len(bench.fake.calls_to("close")) == 1
    with pytest.raises(RuntimeError, match="closed"):
        bench.driver.poll(bench.last)
    with pytest.raises(RuntimeError, match="closed"):
        bench.driver.drain(0.0)


def test_hints_and_foreign_handles_are_programming_errors() -> None:
    bench, other = Bench(), Bench()
    with pytest.raises(DriverRejected, match="hints"):
        bench.driver.get(bench.rows("A"), "a peer")
    other.fill_sources("put", "A")
    other.run("put", "A")
    for query in ("poll", "access_ended", "access_unprovable", "cancel"):
        with pytest.raises(ValueError, match="does not belong"):
            getattr(bench.driver, query)(other.last)
    with pytest.raises(ValueError, match="does not belong"):
        bench.driver.unregister(other.handle)
    assert [c for c in bench.fake.calls if c.method.startswith("batch_")] == []


def test_shutdown_ends_the_worker_threads_even_with_an_unprovable_get() -> None:
    pool = FakePool()
    fake = FakeMooncakeStore(pool)
    reader = Endpoint(MooncakeDriver(fake), ["A"])
    pool.objects[store_key("A")] = b"aaaaaaaa"
    fake.script(GET, [-800])  # TRANSFER_FAIL: the read may still run, so its access is unproven
    attempt = reader.backend.fetch(reader.view("A"), None)
    reader.wait_outcome(attempt)
    assert reader.backend.shutdown(WAIT_S) == frozenset({attempt.label, "region:0"})
    driver = reader.driver.inner
    wait_until(lambda: driver._workers_alive == 0, "the driver's worker threads to exit")


def test_backend_round_trip_over_the_fake_store() -> None:
    pool = FakePool()  # two clients of one master
    writer = Endpoint(MooncakeDriver(FakeMooncakeStore(pool)), ["A", "B"])
    reader = Endpoint(MooncakeDriver(FakeMooncakeStore(pool)), ["A", "B", "C", "D"], fill=0x2D)
    assert isinstance(writer.backend, Backend)
    assert writer.backend.capabilities == Capabilities(fetch=True, publish=True, cancel=True)
    writer.write("A", b"aaaaaaaa")
    writer.write("B", b"bbbbbbbb")
    pool.objects[store_key("D")] = b"short"

    published = writer.backend.publish(writer.view("A", "B"), None)
    assert isinstance(published, Attempt)
    assert writer.wait_outcome(published).delivered_masks[0].tolist() == [True, True]
    assert pool.objects[store_key("A")] == b"aaaaaaaa"
    writer.wait_quiesced(published)

    view = reader.view("A", "B", "C")
    fetched = reader.backend.fetch(view, None)
    outcome = reader.wait_outcome(fetched)
    outcome.check_mask_shapes(view)
    assert outcome.fault is None
    assert outcome.delivered_masks[0].tolist() == [True, True, False]
    assert [reader.read(name) for name in "ABC"] == [b"aaaaaaaa", b"bbbbbbbb", b"-" * SLOT]
    reader.wait_quiesced(fetched)

    wrong = reader.backend.fetch(reader.view("D"), None)
    outcome = reader.wait_outcome(wrong)
    assert outcome.delivered_masks[0].tolist() == [False]
    assert "object is 5 bytes, expected 8" in outcome.fault
    reader.wait_quiesced(wrong)
    assert reader.region.buffer.canaries_intact()
    writer.close()
    reader.close()


@pytest.mark.parametrize("direction", ["publish", "fetch"])
def test_a_view_naming_one_key_twice_goes_in_two_calls_and_never_faults(direction: str) -> None:
    """The backend cuts a group at a repeated object key, so no operation names a key twice. With
    one worker the second put finds the object the first one committed; running alongside it, it
    would find the key held by this driver's put and be NOT_READY, never a fault."""
    fake = FakeMooncakeStore()
    slots = ["S0", "S1", "S2"]
    endpoint = Endpoint(MooncakeDriver(fake, workers=1), slots)
    keys = "ABA"
    payload = {name: name.encode() * SLOT for name in set(keys)}
    view = CpuView(
        [Rows(0, np.stack([object_key(name) for name in keys]), endpoint.region.slots(0, 1, 2), 0)]
    )
    if direction == "publish":
        for slot, name in zip(slots, keys):
            endpoint.write(slot, payload[name])
        attempt = endpoint.backend.publish(view, None)
    else:
        for name, data in payload.items():
            fake.pool.objects[store_key(name)] = data
        attempt = endpoint.backend.fetch(view, None)
    outcome = endpoint.wait_outcome(attempt)
    assert outcome.fault is None
    assert outcome.delivered_masks[0].tolist() == [True, True, True]
    first_call = PUT_START if direction == "publish" else GET
    assert [call.args[0] for call in fake.calls_to(first_call)] == [
        [store_key("A"), store_key("B")],
        [store_key("A")],
    ]
    if direction == "publish":
        assert {name: fake.pool.objects[store_key(name)] for name in "AB"} == payload
    else:
        assert [endpoint.read(slot) for slot in slots] == [payload[name] for name in keys]
    endpoint.wait_quiesced(attempt)
    endpoint.close()


@pytest.fixture
def work_per_row_or_thread(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Every key or row a ``RowBatch`` builds and every thread started, each with the thread that
    did it."""
    done: list[tuple[str, int]] = []
    keys_iter, key_at = store_module._HexKeys.__iter__, store_module._HexKeys.__getitem__
    row_at, thread_start = RowBatch.__getitem__, threading.Thread.start

    def record(what: str) -> None:
        done.append((what, threading.get_ident()))

    def every_key(self):
        record("keys")
        return keys_iter(self)

    def one_key(self, index):
        if not isinstance(index, slice):
            record("key")
        return key_at(self, index)

    def one_row(self, index):
        if not isinstance(index, slice):
            record("row")
        return row_at(self, index)

    def start(self):
        record("thread")
        return thread_start(self)

    monkeypatch.setattr(store_module._HexKeys, "__iter__", every_key)
    monkeypatch.setattr(store_module._HexKeys, "__getitem__", one_key)
    monkeypatch.setattr(RowBatch, "__getitem__", one_row)
    monkeypatch.setattr(threading.Thread, "start", start)
    return done


def test_fetch_and_publish_build_no_key_or_row_and_start_no_thread_on_the_calling_thread(
    work_per_row_or_thread: list[tuple[str, int]],
) -> None:
    # The calling thread is the executor loop's: it builds no store key or RowIO and starts no
    # thread; keys and rows are built on the driver's threads. Built without the test endpoint,
    # whose driver spy reads every row it is handed.
    rows = 64
    region = CpuRegion(SLOT, 2 * rows)
    binding = RegionBinding((region,), MooncakeDriver(FakeMooncakeStore()), FakeHold())
    binding.register()
    wake = WakeCounter()
    backend = BlobBackend(binding, wake, BlobConfig(key_prefix="t"))
    keys = np.stack([object_key(f"R{i}") for i in range(rows)])
    for i in range(rows):
        region.write(i, f"R{i}".encode().ljust(SLOT, b"."))
    offered = CpuView([Rows(0, keys, region.slots(*range(rows)), 0)])
    wanted = CpuView([Rows(0, keys.copy(), region.slots(*range(rows, 2 * rows)), 0)])
    caller = threading.get_ident()
    for submit, view in ((backend.publish, offered), (backend.fetch, wanted)):
        work_per_row_or_thread.clear()
        attempt = submit(view, None)
        assert [what for what, thread in work_per_row_or_thread if thread == caller] == []
        wake.wait_for(lambda: attempt.poll() is not None and attempt.quiesced(), "the attempt")
        assert attempt.poll().delivered_masks[0].all()
    assert region.read(rows + 7) == b"R7......"
    assert backend.shutdown(WAIT_S) == frozenset()
