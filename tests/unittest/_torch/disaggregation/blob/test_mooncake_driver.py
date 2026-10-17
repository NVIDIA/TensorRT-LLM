# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Mooncake driver over a fake SDK that moves real bytes.

Pins the mapping of SDK 0.3.13 answers to row results, the put session bookkeeping (every session
the driver starts is ended, revoked or parked; only rows written whole are ended), the access-end
rules (a put scatter that returns ends access, whatever it answers; a ``batch_get_into`` ends it
unless a row answers TRANSFER_FAIL; a call that raises or miscounts does not, and its access is
unprovable), registration, drain and close, and a backend round trip over the fake SDK. The shared
``driver_conformance`` checks run over the fake SDK too.
"""

from __future__ import annotations

import threading
import time
from collections import Counter

import pytest
from cpu_fixtures import WAIT_S, Endpoint, store_key
from driver_conformance import DriverConformance
from fake_mooncake import FakeMooncakeStore, FakePool
from host_memory import HostBuffer

from tensorrt_llm._torch.disaggregation.base.backend import Attempt, Backend, Capabilities
from tensorrt_llm._torch.disaggregation.blob import mooncake as mooncake_module
from tensorrt_llm._torch.disaggregation.blob.mooncake import MooncakeDriver
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    DriverRejected,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowStatus,
    Span,
    StoreDriver,
    StoreError,
)

pytestmark = pytest.mark.cpu_only

SLOT = 8
PROMPT_S = 0.5
PUT_START = "batch_put_session_start"
PUT_RANGES = "batch_put_from_multi_buffer_ranges"
PUT_END = "batch_put_session_end"
PUT_REVOKE = "batch_put_session_revoke"
GET = "batch_get_into"
DATA_CALL = {"put": PUT_RANGES, "get": GET}
CALL_FAILED = "existence check: CALL_FAILED(-100000)"
SHORT_OBJECT_LANDS = (
    "batch_get_into reads an object shorter than its buffer into the buffer and answers its "
    "size (real_client.cpp batch_get_into_internal, client_buffer.cpp allocateSlices): the row "
    "is a FAULT and access ends, but the slot's head holds the short object"
)


def key(name: str) -> str:
    return f"t/v1/{name}"


class Bench:
    """A ``MooncakeDriver`` over a fake SDK, with ``slots`` registered host slots."""

    def __init__(self, slots: int = 4, pool: FakePool | None = None) -> None:
        self.fake = FakeMooncakeStore(pool)
        self.driver = MooncakeDriver(self.fake)
        self.buffer = HostBuffer(SLOT * slots, fill=0x2D)
        layout = RegionLayout(SLOT, slots)
        self.handle = self.driver.register(self.buffer.address, self.buffer.nbytes, layout)

    def rows(self, *names: str) -> list[RowIO]:
        return [RowIO(key(n), self.buffer.span(i * SLOT, SLOT)) for i, n in enumerate(names)]

    def store(self, **objects: bytes) -> None:
        for name, data in objects.items():
            self.fake.pool.objects[key(name)] = data

    def write(self, slot: int, data: bytes) -> None:
        self.buffer.write(data, slot * SLOT)

    def read(self, slot: int) -> bytes:
        return self.buffer.read(slot * SLOT, SLOT)

    def run(self, direction: str, *names: str):
        """The results and access end of a put or get of ``names``; ``last`` is its handle."""
        rows = self.rows(*names)
        self.last = self.driver.put(rows) if direction == "put" else self.driver.get(rows)
        return self.driver.poll(self.last), self.driver.access_ended(self.last)

    def objects_for(self, direction: str, *names: str) -> None:
        """What a put or get of ``names`` starts from: filled slots, or objects in the pool."""
        for slot, name in enumerate(names):
            data = name.encode() * SLOT
            self.write(slot, data) if direction == "put" else self.store(**{name: data})

    def keys_of(self, method: str) -> list[str]:
        return [k for call in self.fake.calls_to(method) for k in call.args[0]]


def summary(results) -> list[tuple[str, str | None]]:
    return [(result.status.value, result.fault) for result in results]


def test_the_driver_follows_the_driver_protocol() -> None:
    bench = Bench()
    assert isinstance(bench.driver, StoreDriver)
    assert bench.driver.capabilities == DriverCapabilities(
        get=True, put=True, cancel=False, probe=False, max_rows_per_op=None, hints=False
    )


class TestMooncakeDriverConformance(DriverConformance):
    """The shared driver checks, over the fake SDK."""

    def make_driver(self) -> StoreDriver:
        return MooncakeDriver(FakeMooncakeStore())

    @pytest.mark.xfail(strict=True, reason=SHORT_OBJECT_LANDS)
    def test_an_object_of_the_wrong_size_is_a_fault_not_a_miss(self) -> None:
        super().test_an_object_of_the_wrong_size_is_a_fault_not_a_miss()


# Answers to row results.


def untaken_counts(bench: Bench) -> dict[str, int]:
    return {name: n for name, n in bench.driver.stats().items() if name.startswith("rows_") and n}


GET_ANSWERS = [
    # answer, status, fault, where an untaken row is counted
    (None, "ready", None, None),  # unscripted: the object's size
    (5, "fault", "object is 5 bytes, expected 8", None),
    (-704, "miss", None, None),
    (-703, "not_ready", None, "rows_not_ready_being_written"),
    (-702, "fault", "batch_get_into: INVALID_REPLICA(-702)", None),
    (-600, "fault", "batch_get_into: INVALID_PARAMS(-600)", None),
    (-900, "fault", "batch_get_into: RPC_FAIL(-900)", None),
    (-800, "fault", "batch_get_into: TRANSFER_FAIL(-800)", None),
    (-9999, "fault", "batch_get_into: code(-9999)", None),
]


@pytest.mark.parametrize("answer, status, fault, counted", GET_ANSWERS)
def test_get_answers_map_to_row_statuses(answer, status, fault, counted) -> None:
    bench = Bench()
    bench.store(A=b"aaaaaaaa")
    bench.fake.script(GET, [answer])
    results, access_ended = bench.run("get", "A")
    assert summary(results) == [(status, fault)]
    assert untaken_counts(bench) == ({counted: 1} if counted else {})
    assert (bench.read(0) == b"aaaaaaaa") == (status == "ready")
    assert access_ended is (answer != -800)
    assert bench.driver.access_unprovable(bench.last) is (answer == -800)


PUT_START_ANSWERS = [
    (-200, "declined", None, "rows_declined_pool_full"),
    (-600, "not_ready", None, "rows_not_ready_written_here"),
    (-702, "fault", "put session start: INVALID_REPLICA(-702)", None),
    (-900, "fault", "put session start: RPC_FAIL(-900)", None),
    (-9999, "fault", "put session start: code(-9999)", None),
]


@pytest.mark.parametrize("answer, status, fault, counted", PUT_START_ANSWERS)
def test_put_session_start_answers_map_to_row_statuses(answer, status, fault, counted) -> None:
    bench = Bench()
    bench.fake.script(PUT_START, [answer])
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [(status, fault)]
    assert untaken_counts(bench) == ({counted: 1} if counted else {})
    assert access_ended
    (start,) = bench.fake.calls_to(PUT_START)
    assert len(start.args) == 2 and start.kwargs == {}  # never a replicate config
    for method in (PUT_RANGES, PUT_END, PUT_REVOKE, "batch_is_exist"):
        assert bench.fake.calls_to(method) == []  # a session it did not start is not its own


PUT_WRITE_ANSWERS = [
    ("written", None, None, "ready", None),
    ("short_write", 5, None, "fault", "put ranges: moved 5 of 8 bytes"),
    ("transfer_fail", -800, None, "fault", "put ranges: TRANSFER_FAIL(-800)"),
    ("session_lost", -600, None, "fault", "put ranges: INVALID_PARAMS(-600)"),
    ("end_refused", None, -704, "fault", "put session end: OBJECT_NOT_FOUND(-704)"),
    ("end_not_owner", None, -601, "fault", "put session end: ILLEGAL_CLIENT(-601)"),
]


@pytest.mark.parametrize("case, ranges, end, status, fault", PUT_WRITE_ANSWERS)
def test_put_ranges_and_end_answers_map_to_row_statuses(case, ranges, end, status, fault) -> None:
    bench = Bench()
    bench.write(0, b"aaaaaaaa")
    bench.fake.script(PUT_RANGES, [ranges])
    bench.fake.script(PUT_END, [end])
    results, access_ended = bench.run("put", "A")
    assert summary(results) == [(status, fault)]
    assert access_ended
    assert (bench.fake.pool.objects.get(key("A")) == b"aaaaaaaa") == (status == "ready")
    assert bench.keys_of(PUT_REVOKE) == ([] if status == "ready" else [key("A")])


EXISTENCE_ANSWERS = {
    # batch_is_exist's step: the results of the two rows that already existed
    "committed_and_not": ([1, 0], [("ready", None), ("not_ready", None)]),
    "check_failed": ([1, -900], [("ready", None), ("fault", "existence check: RPC_FAIL(-900)")]),
    # A metadata call touches no row memory: its raise fails only its rows.
    "check_raised": (RuntimeError("master gone"), [("fault", CALL_FAILED)] * 2),
}


@pytest.mark.parametrize("case", EXISTENCE_ANSWERS)
def test_only_rows_that_already_exist_are_checked_for_existence(case: str) -> None:
    exist, expected = EXISTENCE_ANSWERS[case]
    bench = Bench()
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


SESSION_SCENARIOS = {
    # name: (steps by method, rows ready (R) or faulted (F), access ends)
    "all_written": ({}, "RRR", True),
    "transfer_fail": ({PUT_RANGES: [[None, -800, None]]}, "RFR", True),
    "short_write": ({PUT_RANGES: [[None, 5, None]]}, "RFR", True),
    "end_retried": ({PUT_END: [[-900, None, None], [None]]}, "RRR", True),
    "end_fails_twice": ({PUT_END: [[-901, None, None], [-900]]}, "FRR", True),
    "end_and_revoke_fail": ({PUT_END: [[-704, None, None]], PUT_REVOKE: [[-900]]}, "FRR", True),
    "end_raises": ({PUT_END: [RuntimeError("rpc lost")]}, "FFF", True),
    "end_committed_then_raised": ({PUT_END: [reply_lost]}, "FFF", True),
    "ranges_raise": ({PUT_RANGES: [RuntimeError("bad pointer")]}, "FFF", False),
    "ranges_miscount": ({PUT_RANGES: [lambda answers: answers[:-1]]}, "FFF", False),
}


@pytest.mark.parametrize("scenario", SESSION_SCENARIOS)
def test_every_started_put_session_is_ended_revoked_or_parked(scenario: str) -> None:
    steps, statuses, ends_access = SESSION_SCENARIOS[scenario]
    bench = Bench()
    for slot, data in enumerate((b"aaaaaaaa", b"bbbbbbbb", b"cccccccc")):
        bench.write(slot, data)
    for method, method_steps in steps.items():
        for step in method_steps:
            bench.fake.script(method, step)
    results, access_ended = bench.run("put", "A", "B", "C")
    assert "".join(r.status.value[0].upper() for r in results) == statuses
    assert access_ended is ends_access

    started = bench.fake.keys_answered(PUT_START, 0)
    revoked = bench.fake.keys_answered(PUT_REVOKE, 0) + bench.fake.keys_answered(PUT_REVOKE, -600)
    settled = Counter(bench.fake.keys_answered(PUT_END, 0) + revoked)  # -600: none left to revoke
    assert set(settled.values()) <= {1}, f"a session ended or revoked twice: {settled}"
    left_open = [k for k in started if k not in settled]
    assert len(left_open) == bench.driver.stats()["put_sessions_parked"]
    assert set(left_open) == bench.fake.put_sessions
    # Only a row written whole may be ended: its object becomes readable as it stands.
    written = bench.fake.keys_answered(PUT_RANGES, SLOT)
    assert set(bench.keys_of(PUT_END)) <= set(written)
    ready = [key(n) for n, r in zip("ABC", results) if r.status is RowStatus.READY]
    assert set(ready) == set(bench.fake.keys_answered(PUT_END, 0))


def test_parked_sessions_are_revoked_only_in_drain() -> None:
    bench = Bench()
    bench.write(0, b"aaaaaaaa")
    bench.fake.script(PUT_END, [-704])
    bench.fake.script(PUT_REVOKE, [-900])
    bench.run("put", "A")
    assert bench.driver.stats()["put_sessions_parked"] == 1
    bench.write(1, b"bbbbbbbb")
    bench.store(C=b"cccccccc")
    bench.run("put", "X", "B")
    bench.run("get", "C")
    assert bench.keys_of(PUT_REVOKE) == [key("A")]  # only its own first try

    bench.fake.script(PUT_REVOKE, [-900])
    assert bench.driver.drain(0.0) is True  # parked sessions hold no lender memory
    assert bench.keys_of(PUT_REVOKE) == [key("A")] * 2
    assert bench.driver.stats()["put_sessions_parked"] == 1
    assert bench.driver.drain(0.0) is True
    assert bench.driver.stats()["put_sessions_parked"] == 0
    assert bench.fake.put_sessions == set()


def test_drain_leaves_sessions_alone_while_access_is_unproven() -> None:
    bench = Bench()
    bench.fake.script(PUT_RANGES, RuntimeError("bad pointer"))
    _, access_ended = bench.run("put", "A")
    assert not access_ended
    assert bench.driver.drain(0.0) is False
    assert bench.fake.calls_to(PUT_REVOKE) == []
    assert bench.driver.stats()["put_sessions_parked"] == 1


LEASE_EXPIRED_CASES = {
    # the first read's answers, the reread's, the results, whether access ended
    "reread_ready": ([None, -707, None], [None], "RRR", True),
    "expired_again": ([None, -707, None], [-707], "RFR", True),
    "first_read_failed": ([None, -800, -707], [None], "RFR", False),
    "reread_failed": ([None, None, -707], [-800], "RRF", False),
}


@pytest.mark.parametrize("case", LEASE_EXPIRED_CASES)
def test_rows_whose_lease_expired_are_reread_once_and_both_reads_must_end_access(case: str) -> None:
    first, reread, statuses, ended = LEASE_EXPIRED_CASES[case]
    bench = Bench()
    bench.objects_for("get", "A", "B", "C")
    bench.fake.script(GET, first)
    bench.fake.script(GET, reread)
    results, access_ended = bench.run("get", "A", "B", "C")
    assert "".join(r.status.value[0].upper() for r in results) == statuses
    expired = [key(name) for name, answer in zip("ABC", first) if answer == -707]
    assert [call.args[0] for call in bench.fake.calls_to(GET)] == [[key(n) for n in "ABC"], expired]
    assert access_ended is ended
    assert bench.driver.drain(0.0) is ended


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
    assert stats["rows_not_ready_being_written"] == stats["rows_not_ready_written_elsewhere"] == 1
    assert any(key("K") in line and "not_ready (being_written)" in line for line in warnings)
    assert any(key("K") in line and "not_ready (written_elsewhere)" in line for line in warnings)


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
    bench.objects_for(direction, "A", "B")
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
        monkeypatch.setattr(mooncake_module, "_get_result", broken)
        bench.fake.script(GET, [5])  # a row read short goes past the all-whole shortcut
    else:
        monkeypatch.setattr(MooncakeDriver, "_end_put_sessions", broken)
    bench.objects_for(direction, "A")
    results, access_ended = bench.run(direction, "A")
    assert summary(results) == [("fault", f"{direction} failed: ValueError('mapping bug')")]
    assert not access_ended


# Admission, registration, concurrency.


def test_bad_rows_are_rejected_before_the_sdk() -> None:
    bench = Bench()
    with pytest.raises(DriverRejected, match="at least one"):
        bench.driver.get([])
    with pytest.raises(DriverRejected, match="at most"):
        bench.driver.put([RowIO(key("A"), Span(bench.buffer.address, 1 << 31))])
    past_the_end = bench.buffer.span(3 * SLOT, 2 * SLOT)  # starts in the last slot
    with pytest.raises(DriverRejected, match="outside"):
        bench.driver.put([RowIO(key("A"), past_the_end)])
    assert [c for c in bench.fake.calls if c.method.startswith("batch_")] == []


def test_rows_in_two_registrations_are_admitted_row_by_row() -> None:
    bench = Bench()
    other = HostBuffer(SLOT)  # a second registration, apart from the first
    bench.driver.register(other.address, SLOT, RegionLayout(SLOT, 1))
    first, _ = bench.rows("A", "B")
    op = bench.driver.get([first, RowIO(key("C"), other.span())])  # the gap is no row's
    assert summary(bench.driver.poll(op)) == [("miss", None), ("miss", None)]
    assert bench.driver.access_ended(op)


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
            driver.get([RowIO(key("probe"), stray.span())])
        if "draining" in str(refused.value):
            return
    pytest.fail("no drain started")


def test_unregister_waits_for_every_operation_on_the_region() -> None:
    bench = Bench()
    bench.objects_for("put", "A")
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


def test_a_blocked_get_delays_no_other_get_and_no_drain() -> None:
    bench = Bench()
    bench.store(A=b"aaaaaaaa", B=b"bbbbbbbb")
    a, b = bench.rows("A", "B")
    gate = bench.fake.block(GET)  # the first get waits inside the SDK
    blocked = threading.Thread(target=lambda: bench.driver.get([a]))
    blocked.start()
    try:
        assert gate.entered.wait(WAIT_S)
        started = time.monotonic()
        op = bench.driver.get([b])
        assert summary(bench.driver.poll(op)) == [("ready", None)]
        assert bench.driver.drain(0.0) is False  # answers at once while a get runs
        assert time.monotonic() - started < PROMPT_S
    finally:
        gate.release.set()
        blocked.join(WAIT_S)
    assert bench.driver.drain(WAIT_S) is True


def test_an_unprovable_operation_pins_its_region_and_fails_drain_at_once() -> None:
    bench = Bench()
    bench.objects_for("get", "A", "B")
    bench.fake.script(GET, [-800])
    bench.run("get", "A")
    gate = bench.fake.block(GET)
    blocked = threading.Thread(target=lambda: bench.driver.get(bench.rows("A", "B")[1:]))
    blocked.start()
    try:
        assert gate.entered.wait(WAIT_S)  # another get is still running
        started = time.monotonic()
        assert bench.driver.drain(WAIT_S) is False  # waits for no running call
        assert time.monotonic() - started < PROMPT_S
    finally:
        gate.release.set()
        blocked.join(WAIT_S)
    with pytest.raises(DriverRejected, match="may still access"):
        bench.driver.unregister(bench.handle)
    with pytest.raises(DriverRejected, match="drain"):
        bench.driver.close()
    assert bench.fake.calls_to("close") == []


def test_close_needs_a_drain_closes_the_sdk_client_once_and_then_queries_raise() -> None:
    bench = Bench()
    bench.objects_for("put", "A")
    bench.run("put", "A")
    with pytest.raises(DriverRejected, match="drain"):
        bench.driver.close()
    assert bench.driver.drain(0.0) is True
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
    other.objects_for("put", "A")
    other.run("put", "A")
    for query in ("poll", "access_ended", "access_unprovable", "cancel"):
        with pytest.raises(ValueError, match="does not belong"):
            getattr(bench.driver, query)(other.last)
    with pytest.raises(ValueError, match="does not belong"):
        bench.driver.unregister(other.handle)
    assert [c for c in bench.fake.calls if c.method.startswith("batch_")] == []


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
