# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What any ``StoreDriver`` owes the backend, checked through the protocol alone.

A driver's test module subclasses :class:`DriverConformance` as ``Test...``, provides
``make_driver`` and, where its store can stage them, the scenario hooks (a blocked query, a key
another writer holds, a store with no room, and so on); a hook that returns ``None`` skips its
check. Every check observes consequences the backend could observe: bytes in registered memory,
row results, access end, rejections without effect. Nothing here uses a driver's own controls,
so the same checks run against the in-memory driver, a real store's driver, and the deliberately
broken drivers that prove each check bites.

Each check runs inside a :class:`Harness` that keeps every endpoint and operation it created
under a key prefix of its own and, when the check is over, waits for access to end, unregisters
what it registered, drains and shuts the driver down. Memory whose access never provably ended,
or that the driver still holds, is kept alive instead of being freed.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import threading
import time
from collections.abc import Callable, Generator, Sequence

import pytest

from tensorrt_llm._torch.disaggregation.blob.keys import encode_key
from tensorrt_llm._torch.disaggregation.blob.store import (
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
    Span,
    StoreDriver,
    concluded,
)

from .host_memory import HostBuffer, Memory

SLOT = 8

# Lender names are 54 bytes and start with a 16-byte scope digest shared by every name of a
# deployment. A key built from a truncated name collides across these.
SCOPE = b"scope" + bytes(11)
NAME_BYTES = 54


def name_of(label: str) -> bytes:
    return SCOPE + hashlib.blake2b(label.encode(), digest_size=NAME_BYTES - len(SCOPE)).digest()


NAMES = {label: name_of(label) for label in "ABC"}


def key_of(label: str, prefix: str = "t") -> str:
    return encode_key(prefix, name_of(label))


def wait_until(predicate: Callable[[], bool], what: str, timeout: float) -> None:
    """Polls ``predicate`` until true; fails after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.001)


def statuses(driver: StoreDriver, op: OpHandle) -> list[RowStatus]:
    return [result.status for result in driver.poll(op)]


Allocation = tuple[Memory, "RegionHandle | None"]
Unblock = Callable[[], None]


class Endpoint:
    """One side of a transfer: one region with a slot per label, under a key prefix."""

    def __init__(
        self,
        driver: StoreDriver,
        labels: Sequence[str],
        memory: Memory,
        handle: RegionHandle | None,
        *,
        prefix: str,
        slot: int = SLOT,
    ) -> None:
        self.driver = driver
        self.labels = list(labels)
        self.memory = memory
        self.prefix = prefix
        self.slot = slot
        self.handle = handle
        # A handle given up front belongs to whoever allocated the memory; this endpoint only
        # unregisters what it registered itself.
        self.owns_registration = handle is None
        self.released = False  # unregistered or given back; a cleanup retry skips it

    def register(self) -> None:
        """Registers the region. When the store may have kept it despite an error, the handle
        is recorded all the same, so the region can still be unregistered later."""
        layout = RegionLayout(self.slot, len(self.labels), "kv")
        try:
            self.handle = self.driver.register(self.memory.address, self.memory.nbytes, layout)
        except RegistrationError as error:
            self.handle = error.handle
            raise

    def key(self, label: str) -> str:
        return key_of(label, self.prefix)

    def rows(self) -> list[RowIO]:
        return [RowIO(self.key(label), (self.span(i),)) for i, label in enumerate(self.labels)]

    def span_at(self, offset: int, nbytes: int) -> Span:
        """Any range of the region, slot boundaries or not."""
        return Span(self.memory.address + offset, nbytes)

    def span(self, index: int, nbytes: int | None = None) -> Span:
        return Span(
            self.memory.address + index * self.slot, self.slot if nbytes is None else nbytes
        )

    def fill_row(self, index: int, data: bytes) -> None:
        self.memory.write(data, index * self.slot)

    def read_row(self, index: int) -> bytes:
        return self.memory.read(index * self.slot, self.slot)

    def intact(self) -> bool:
        return self.memory.canaries_intact()

    def unregister(self) -> None:
        if self.owns_registration and self.handle is not None:
            self.driver.unregister(self.handle)


# Endpoints whose memory a driver may still touch or still holds registered when a check ends.
# Never freed.
UNRELEASED_ENDPOINTS: list[Endpoint] = []


class Harness:
    """One driver under test plus everything a check created with it."""

    def __init__(self, suite: DriverConformance, driver: StoreDriver, timeout: float) -> None:
        self.suite = suite
        self.driver = driver
        self.timeout = timeout
        self.prefix = "t" + secrets.token_hex(4)  # keys of one check never meet another's
        self.endpoints: list[Endpoint] = []
        self.ops: list[OpHandle] = []
        self.done = False

    # Creating things the harness will clean up.

    def key(self, label: str) -> str:
        return key_of(label, self.prefix)

    def endpoint(self, labels: Sequence[str], *, fill: int = 0, slot: int = SLOT) -> Endpoint:
        memory, handle = self.suite.allocate(slot * len(labels), fill)
        endpoint = Endpoint(self.driver, labels, memory, handle, prefix=self.prefix, slot=slot)
        self.endpoints.append(endpoint)  # kept before the store is asked, whatever register does
        if handle is None:
            endpoint.register()
        return endpoint

    def put(self, rows: Sequence[RowIO]) -> OpHandle:
        op = self.driver.put(rows)
        self.ops.append(op)
        return op

    def get(self, rows: Sequence[RowIO], hint: Hint = None) -> OpHandle:
        # Without a hint the call keeps its old shape, as the backend's does.
        op = self.driver.get(rows) if hint is None else self.driver.get(rows, hint)
        self.ops.append(op)
        return op

    def publish(self, rows: dict[str, bytes], slot: int = SLOT) -> Endpoint:
        """Puts ``rows`` (label to payload) and returns the writer endpoint."""
        writer = self.endpoint(list(rows), slot=slot)
        for i, data in enumerate(rows.values()):
            writer.fill_row(i, data)
        op = self.put(writer.rows())
        assert [r.status for r in self.settle(op)] == [RowStatus.READY] * len(rows)
        self.settle_access(op)
        return writer

    # Waiting and asserting.

    def settle(self, op: OpHandle) -> tuple[RowResult, ...]:
        """Waits for every row's result, not for access end."""
        wait_until(
            lambda: concluded(self.driver.poll(op)), "the operation to conclude", self.timeout
        )
        return self.driver.poll(op)

    def settle_access(self, op: OpHandle) -> None:
        wait_until(lambda: self.driver.access_ended(op), "access to end", self.timeout)

    def assert_absent(self, result: RowResult, endpoint: Endpoint, index: int) -> None:
        """What a get of a key nobody published must look like: MISS with the slot untouched
        for a driver that can tell absence apart, otherwise anything but READY."""
        if self.driver.capabilities.absent_is_miss:
            assert result.status is RowStatus.MISS, result
            assert endpoint.read_row(index) == bytes([0x2D]) * endpoint.slot
        else:
            assert result.status is not RowStatus.READY, result

    def needs(
        self,
        *,
        put: bool = False,
        get: bool = False,
        probe: bool = False,
        cancel: bool = False,
        multi_span: bool = False,
        registers: bool = False,
        rows: int = 1,
    ) -> None:
        """Skips the check when the driver does not declare, or the suite cannot stage, what
        the check needs."""
        caps = self.driver.capabilities
        if put and not caps.put:
            pytest.skip("this driver declares no put")
        if get and not caps.get:
            pytest.skip("this driver declares no get")
        if probe and not caps.probe:
            pytest.skip("this driver declares no probe")
        if cancel and not caps.cancel:
            pytest.skip("this driver declares no cancel")
        if multi_span and not caps.multi_span:
            pytest.skip("this driver takes rows of one span")
        if registers and not self.suite.registers_on_demand:
            pytest.skip("this driver's memory is registered up front")
        if caps.max_rows_per_op is not None and rows > caps.max_rows_per_op:
            pytest.skip(f"this driver takes at most {caps.max_rows_per_op} rows per operation")

    # Cleanup.

    def cleanup(self) -> list[str]:
        """Ends everything the check created, in order; returns what did not go to plan.

        Whenever cleanup stops short of shutting the driver down, every endpoint is retained for
        good: memory the driver may still touch is never freed. An operation whose end the driver
        declares unprovable is allowed by the contract and reported as no problem; one that
        neither ends nor says so is. A cleanup that did not finish may be called again.
        """
        if self.done:
            return []
        problems: list[str] = []
        unprovable = False
        for op in self.ops:
            try:
                wait_until(
                    lambda: self.driver.access_ended(op) or self._unprovable(op),
                    "access to end",
                    self.timeout,
                )
                if not self.driver.access_ended(op):
                    unprovable = True
            except AssertionError as error:
                problems.append(str(error))
            except Exception as error:  # noqa: BLE001 - every cleanup step is reported
                problems.append(f"access end: {error!r}")
        if problems or unprovable:
            self._retain()
            return problems
        for endpoint in self.endpoints:
            if endpoint.released:
                continue
            try:
                if endpoint.owns_registration:
                    endpoint.unregister()
                else:
                    self.suite.release(endpoint.memory)
                endpoint.released = True
            except Exception as error:  # noqa: BLE001 - every cleanup step is reported
                problems.append(f"unregister: {error!r}")
        if problems:
            self._retain()
            return problems
        try:
            if self.driver.drain(self.timeout):
                self.suite.shutdown(self.driver)
            else:
                problems.append("drain did not return True")
        except Exception as error:  # noqa: BLE001 - every cleanup step is reported
            problems.append(f"drain/shutdown: {error!r}")
        if problems:
            self._retain()
            return problems
        self.done = True
        return []

    def _unprovable(self, op: OpHandle) -> bool:
        try:
            return bool(self.driver.access_unprovable(op))
        except Exception:  # noqa: BLE001 - a raising query is not evidence either way
            return False

    def _retain(self) -> None:
        for endpoint in self.endpoints:
            if endpoint not in UNRELEASED_ENDPOINTS:
                UNRELEASED_ENDPOINTS.append(endpoint)


class DriverConformance:
    """Contract checks for one driver. Subclass as ``Test...``, implement ``make_driver`` and
    whichever scenario hooks the driver's store can stage."""

    timeout = 5.0
    registers_on_demand = True  # endpoints register fresh memory; a pool driver sets False

    def make_driver(self) -> StoreDriver:
        raise NotImplementedError

    # Memory and lifetime hooks.

    def allocate(self, nbytes: int, fill: int) -> Allocation:
        """Memory for one endpoint and, when the driver registered it up front, its handle.

        The memory object keeps its backing allocation alive for as long as it exists: a check
        that cannot prove access ended retains the endpoint, and nothing else, for good.
        """
        return HostBuffer(nbytes, fill), None

    def release(self, memory: Memory) -> None:
        """Gives back memory that ``allocate`` handed out with a handle; the suite never
        unregisters such memory itself."""

    def shutdown(self, driver: StoreDriver) -> None:
        """Ends the driver once it is drained; a pool driver unregisters its pool first."""
        driver.close()

    # Scenario hooks; ``None`` skips the check that needs it.

    def make_blocked_probe(self) -> tuple[StoreDriver, str, Unblock] | None:
        """A driver whose store stalls every query of the returned key until the callable runs."""
        return None

    def make_blocked_get(self) -> tuple[StoreDriver, str, threading.Event | None, Unblock] | None:
        """A driver whose store stalls every get of the returned key until the callable runs,
        with an event set once a get is stalled, or ``None`` for no such signal."""
        return None

    def make_held_key(self) -> tuple[StoreDriver, str, Unblock] | None:
        """A driver whose store holds the returned key as another writer's unfinished object
        until the callable runs."""
        return None

    def make_full_store(self) -> StoreDriver | None:
        """A driver whose store has room for exactly one more object."""
        return None

    def make_crash_after_record(self) -> StoreDriver | None:
        """A driver whose store raises something that is not a store error from inside a put."""
        return None

    def make_failing_presence(self) -> tuple[StoreDriver, str] | None:
        """A driver whose store fails the next presence query of the returned key."""
        return None

    @contextlib.contextmanager
    def harness(self, driver: StoreDriver | None = None) -> Generator[Harness, None, None]:
        """A fresh driver (or ``driver``) with cleanup afterwards. A failing check keeps its own
        error; cleanup trouble fails only a check that passed."""
        h = Harness(self, driver if driver is not None else self.make_driver(), self.timeout)
        try:
            yield h
        except BaseException:
            h.cleanup()
            raise
        problems = h.cleanup()
        assert not problems, "cleanup after the check: " + "; ".join(problems)

    # Bytes and results.

    def test_put_then_get_moves_whole_rows_and_never_aliases_the_source(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, rows=2)
            writer = h.publish({"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
            writer.fill_row(0, b"xxxxxxxx")  # the source changes after the put concluded

            reader = h.endpoint(["A", "B"], fill=0x2D)
            results = h.settle(h.get(reader.rows()))
            assert [r.status for r in results] == [RowStatus.READY, RowStatus.READY]
            assert reader.read_row(0) == b"aaaaaaaa"
            assert reader.read_row(1) == b"bbbbbbbb"
            assert reader.intact() and writer.intact()

    def test_get_reports_hit_hit_miss_and_leaves_the_miss_untouched(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, rows=3)
            h.publish({"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
            reader = h.endpoint(["A", "B", "C"], fill=0x2D)
            results = h.settle(h.get(reader.rows()))
            assert [r.status for r in results[:2]] == [RowStatus.READY, RowStatus.READY]
            h.assert_absent(results[2], reader, 2)
            assert reader.intact()

    def test_an_object_larger_than_its_row_is_a_fault_not_a_truncated_ready(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True)
            h.publish({"A": b"aaaaaaaa"}, slot=8)
            reader = h.endpoint(["A"], slot=4, fill=0x2D)
            (result,) = h.settle(h.get(reader.rows()))
            assert result.status is RowStatus.FAULT, result
            assert reader.intact()  # nothing lands outside the row

    def test_an_object_smaller_than_its_row_is_a_fault(self) -> None:
        """A store may answer this with an error that leaves access unprovable; the harness
        accepts that and retains the memory."""
        with self.harness() as h:
            h.needs(put=True, get=True)
            h.publish({"A": b"aaaa"}, slot=4)
            reader = h.endpoint(["A"], slot=8, fill=0x2D)
            (result,) = h.settle(h.get(reader.rows()))
            assert result.status is RowStatus.FAULT, result
            assert reader.intact()

    def test_a_duplicate_put_of_a_committed_object_is_ready(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True)
            h.publish({"A": b"aaaaaaaa"})
            h.publish({"A": b"aaaaaaaa"})  # a second writer, same content
            reader = h.endpoint(["A"])
            (result,) = h.settle(h.get(reader.rows()))
            assert result.status is RowStatus.READY
            assert reader.read_row(0) == b"aaaaaaaa"

    def test_a_put_under_an_existing_key_leaves_the_object_whole(self) -> None:
        """Whatever a driver answers a writer bringing other bytes, the object readers get is
        the one that was committed."""
        with self.harness() as h:
            h.needs(put=True, get=True)
            h.publish({"A": b"aaaaaaaa"})
            other = h.endpoint(["A"])
            other.fill_row(0, b"zzzzzzzz")
            (result,) = h.settle(h.put(other.rows()))
            assert result.status in (RowStatus.READY, RowStatus.FAULT)
            reader = h.endpoint(["A"], fill=0x2D)
            (got,) = h.settle(h.get(reader.rows()))
            assert got.status is RowStatus.READY
            assert reader.read_row(0) == b"aaaaaaaa"

    def test_an_operation_leaves_the_rest_of_its_region_alone(self) -> None:
        """A row is the only part of its region an operation may touch; holding back only a
        stuck operation's own slots depends on it."""
        with self.harness() as h:
            h.needs(put=True, get=True)
            h.publish({"B": b"bbbbbbbb"})
            reader = h.endpoint(["A", "B", "C"], fill=0x2D)
            op = h.get([reader.rows()[1]])
            (result,) = h.settle(op)
            assert result.status is RowStatus.READY
            h.settle_access(op)
            assert reader.read_row(1) == b"bbbbbbbb"
            assert reader.read_row(0) == b"-" * SLOT and reader.read_row(2) == b"-" * SLOT
            assert reader.intact()

    def test_a_multi_span_row_is_stored_and_fetched_back_to_back(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, multi_span=True)
            writer = h.endpoint(["A", "B", "C"], slot=4)
            writer.fill_row(0, b"aaaa")
            writer.fill_row(1, b"----")
            writer.fill_row(2, b"cccc")
            row = RowIO(h.key("M"), (writer.span(0), writer.span(2)))  # 8 bytes, slots 0 and 2
            (put,) = h.settle(h.put([row]))
            assert put.status is RowStatus.READY

            whole = h.endpoint(["M"], slot=8, fill=0x2D)
            (got,) = h.settle(h.get(whole.rows()))
            assert got.status is RowStatus.READY
            assert whole.read_row(0) == b"aaaacccc"

            parts = h.endpoint(["X", "Y", "Z"], slot=4, fill=0x2D)
            back = RowIO(h.key("M"), (parts.span(2), parts.span(0)))  # given order, not address
            (got,) = h.settle(h.get([back]))
            assert got.status is RowStatus.READY
            assert parts.read_row(2) == b"aaaa" and parts.read_row(0) == b"cccc"
            assert parts.read_row(1) == b"----"
            assert writer.intact() and whole.intact() and parts.intact()

    def test_a_row_may_cover_parts_of_slots_across_regions(self) -> None:
        """Spans need not be whole slots or share a region, and put and get may split a row
        differently; only the bytes in the spans move."""
        with self.harness() as h:
            h.needs(put=True, get=True, multi_span=True)
            left = h.endpoint(["L"], slot=8, fill=0x2D)
            right = h.endpoint(["R"], slot=8, fill=0x2D)
            left.fill_row(0, b"-xyz----")
            right.fill_row(0, b"--UVWXY-")
            row = RowIO(h.key("P"), (left.span_at(1, 3), right.span_at(2, 5)))  # "xyz" + "UVWXY"
            (put,) = h.settle(h.put([row]))
            assert put.status is RowStatus.READY

            whole = h.endpoint(["P"], slot=8, fill=0x2D)
            (got,) = h.settle(h.get(whole.rows()))
            assert got.status is RowStatus.READY
            assert whole.read_row(0) == b"xyzUVWXY"

            split = h.endpoint(["S"], slot=16, fill=0x2D)
            back = RowIO(h.key("P"), (split.span_at(10, 5), split.span_at(3, 3)))  # 5 then 3
            (got,) = h.settle(h.get([back]))
            assert got.status is RowStatus.READY
            assert split.read_row(0) == b"---WXY----xyzUV-"  # the first 5 bytes land first
            assert left.intact() and right.intact() and whole.intact() and split.intact()

    # Rejections leave no trace.

    def test_rows_outside_a_live_registration_are_rejected_before_any_effect(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, registers=True)
            inside = h.endpoint(["A"])
            stray = HostBuffer(SLOT, fill=0x2D)
            h.publish({"A": b"aaaaaaaa"})

            with pytest.raises(DriverRejected):
                h.get([RowIO(h.key("A"), (stray.span(),))])
            assert stray.read() == b"-" * SLOT and stray.canaries_intact()

            inside.fill_row(0, b"bbbbbbbb")
            inside.unregister()
            with pytest.raises(DriverRejected):
                h.put([RowIO(h.key("B"), (inside.span(0),))])
            reader = h.endpoint(["B"], fill=0x2D)
            (result,) = h.settle(h.get(reader.rows()))
            h.assert_absent(result, reader, 0)

    def test_a_key_appears_at_most_once_per_operation(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, rows=2)
            writer = h.endpoint(["A", "A"])
            writer.fill_row(0, b"aaaaaaaa")
            writer.fill_row(1, b"bbbbbbbb")
            with pytest.raises(DriverRejected):
                h.put(writer.rows())
            reader = h.endpoint(["A", "A"], fill=0x2D)
            with pytest.raises(DriverRejected):
                h.get(reader.rows())
            single = h.endpoint(["A"], fill=0x2D)
            (result,) = h.settle(h.get(single.rows()))
            h.assert_absent(result, single, 0)
            assert reader.read_row(0) == reader.read_row(1) == b"-" * SLOT

    def test_rows_beyond_the_declared_limit_are_rejected(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True)
            limit = h.driver.capabilities.max_rows_per_op
            if limit is None or limit >= 3:
                pytest.skip("this driver declares no row limit this check can exceed")
            writer = h.endpoint(["A", "B", "C"][: limit + 1])
            with pytest.raises(DriverRejected):
                h.put(writer.rows())
            reader = h.endpoint(["A"], fill=0x2D)
            (result,) = h.settle(h.get(reader.rows()))
            h.assert_absent(result, reader, 0)

    def test_an_undeclared_direction_is_rejected(self) -> None:
        with self.harness() as h:
            caps = h.driver.capabilities
            if caps.get and caps.put:
                pytest.skip("this driver declares both directions")
            endpoint = h.endpoint(["A"])
            endpoint.fill_row(0, b"aaaaaaaa")
            with pytest.raises(DriverRejected):
                if caps.put:
                    h.get(endpoint.rows())
                else:
                    h.put(endpoint.rows())
            assert endpoint.read_row(0) == b"aaaaaaaa"

    # Results, access and capabilities over time.

    def test_results_are_stable_and_access_ends_with_evidence(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, rows=2)
            h.publish({"A": b"aaaaaaaa"})
            reader = h.endpoint(["A", "C"])
            op = h.get(reader.rows())
            results = h.settle(op)
            assert h.driver.poll(op) == results
            h.settle_access(op)
            assert h.driver.access_ended(op)
            assert not h.driver.access_unprovable(op)
            assert h.driver.poll(op) == results

    def test_cancel_after_conclusion_changes_nothing(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, cancel=True)
            h.publish({"A": b"aaaaaaaa"})
            reader = h.endpoint(["A"], fill=0x2D)
            op = h.get(reader.rows())
            results = h.settle(op)
            h.driver.cancel(op)
            assert h.driver.poll(op) == results
            assert reader.read_row(0) == b"aaaaaaaa"

    def test_capabilities_do_not_change(self) -> None:
        with self.harness() as h:
            h.needs(put=True)
            before = h.driver.capabilities
            h.publish({"A": b"aaaaaaaa"})
            assert h.driver.capabilities == before
            assert h.driver.capabilities == before

    def test_a_raise_after_the_operation_is_recorded_becomes_a_row_fault(self) -> None:
        driver = self.make_crash_after_record()
        if driver is None:
            pytest.skip("this driver cannot stage a store that raises mid-operation")
        with self.harness(driver) as h:
            h.needs(put=True)
            writer = h.endpoint(["A"])
            writer.fill_row(0, b"aaaaaaaa")
            try:
                op = h.put(writer.rows())
            except Exception as error:  # noqa: BLE001 - the check is that nothing escapes
                pytest.fail(f"put raised after the operation was recorded: {error!r}")
            (result,) = h.settle(op)
            assert result.status is RowStatus.FAULT, result

    def test_a_key_held_by_an_unfinished_write_is_not_ready_for_get_put_and_probe(self) -> None:
        staged = self.make_held_key()
        if staged is None:
            pytest.skip("this driver cannot stage a key another writer holds")
        driver, key, release = staged
        with self.harness(driver) as h:
            h.needs(put=True, get=True)
            early = h.endpoint(["H"], fill=0x2D)
            writer = h.endpoint(["H"])
            writer.fill_row(0, b"hhhhhhhh")
            write = [RowIO(key, (writer.span(0),))]
            try:
                (got,) = h.settle(h.get([RowIO(key, (early.span(0),))]))
                assert got.status is RowStatus.NOT_READY, got
                assert early.read_row(0) == b"-" * SLOT
                (put,) = h.settle(h.put(write))
                assert put.status is RowStatus.NOT_READY, put
                if driver.capabilities.probe:
                    assert driver.probe([key], h.timeout) == (ProbeStatus.NOT_READY,)
            finally:
                release()
            (put,) = h.settle(h.put(write))
            assert put.status is RowStatus.READY
            # Fresh memory: the spans of the early get stay its own until its access provably ended.
            later = h.endpoint(["H"], fill=0x2D)
            (got,) = h.settle(h.get([RowIO(key, (later.span(0),))]))
            assert got.status is RowStatus.READY
            assert later.read_row(0) == b"hhhhhhhh"

    def test_a_row_with_no_room_is_declined_while_the_others_go_on(self) -> None:
        driver = self.make_full_store()
        if driver is None:
            pytest.skip("this driver cannot stage a store with room for one more object")
        with self.harness(driver) as h:
            h.needs(put=True, get=True, rows=2)
            writer = h.endpoint(["A", "B"])
            writer.fill_row(0, b"aaaaaaaa")
            writer.fill_row(1, b"bbbbbbbb")
            op = h.put(writer.rows())
            results = h.settle(op)
            assert sorted((r.status for r in results), key=lambda s: s.value) == [
                RowStatus.DECLINED,
                RowStatus.READY,
            ], results
            h.settle_access(op)
            stored = [r.status for r in results].index(RowStatus.READY)
            reader = h.endpoint(["A", "B"], fill=0x2D)
            got = h.settle(h.get(reader.rows()))
            assert got[stored].status is RowStatus.READY
            assert reader.read_row(stored) == writer.read_row(stored)
            assert got[1 - stored].status is not RowStatus.READY

    def test_a_blocked_get_delays_neither_another_get_nor_the_non_blocking_calls(self) -> None:
        staged = self.make_blocked_get()
        if staged is None:
            pytest.skip("this driver cannot stage a get blocked in the store")
        driver, blocked_key, blocked, unblock = staged
        with self.harness(driver) as h:
            h.needs(put=True, get=True)
            h.publish({"B": b"bbbbbbbb"})
            slow = h.endpoint(["S"], fill=0x2D)
            quick = h.endpoint(["B"], fill=0x2D)
            seen: dict[str, object] = {}
            failures: list[BaseException] = []

            def slow_get() -> None:
                try:
                    h.get([RowIO(blocked_key, (slow.span(0),))])
                except BaseException as error:  # noqa: BLE001 - reported from the main thread
                    failures.append(error)

            def quick_calls() -> None:
                try:
                    op = h.get(quick.rows())
                    seen["quick"] = h.settle(op)
                    seen["ended"] = h.driver.access_ended(op)
                    seen["drain"] = h.driver.drain(0.05)
                except BaseException as error:  # noqa: BLE001 - reported from the main thread
                    failures.append(error)

            # Both off this thread, so a driver that serializes fails instead of hanging.
            slow_thread = threading.Thread(target=slow_get, daemon=True)
            quick_thread = threading.Thread(target=quick_calls, daemon=True)
            slow_thread.start()
            try:
                if blocked is not None:
                    assert blocked.wait(h.timeout), "the slow get never reached the store"
                else:
                    time.sleep(0.05)
                quick_thread.start()
                quick_thread.join(h.timeout)
                assert not quick_thread.is_alive(), (
                    "another get or a non-blocking call waited behind a get blocked in the store"
                )
                if failures:
                    raise AssertionError(f"a call raised: {failures[0]!r}") from failures[0]
                quick_results = seen["quick"]
                assert isinstance(quick_results, tuple)
                assert [r.status for r in quick_results] == [RowStatus.READY]
                assert quick.read_row(0) == b"bbbbbbbb"
                assert seen["drain"] is False  # the slow get is still inside the store
            finally:
                unblock()
                slow_thread.join(h.timeout)
                if quick_thread.ident is not None:
                    quick_thread.join(h.timeout)
            assert not slow_thread.is_alive() and not quick_thread.is_alive()
            if failures:
                raise AssertionError(f"a call raised: {failures[0]!r}") from failures[0]

    # Probe.

    def test_probe_reports_hit_and_miss(self) -> None:
        with self.harness() as h:
            h.needs(put=True, probe=True)
            h.publish({"A": b"aaaaaaaa"})
            assert h.driver.probe([h.key("A"), h.key("C")], self.timeout) == (
                ProbeStatus.HIT,
                ProbeStatus.MISS,
            )

    def test_probe_returns_within_its_budget(self) -> None:
        with self.harness() as h:
            h.needs(probe=True)
            keys = [h.key(label) for label in "ABC"]
            started = time.monotonic()
            answer = h.driver.probe(keys, 0.05)
            elapsed = time.monotonic() - started
            assert elapsed < 0.05 + 0.5, f"probe took {elapsed:.2f}s on a 0.05s budget"
            assert len(answer) == 3 and set(answer) <= set(ProbeStatus)

    def test_probe_answers_unknown_for_a_key_the_store_does_not_answer_in_time(self) -> None:
        staged = self.make_blocked_probe()
        if staged is None:
            pytest.skip("this driver cannot stage a blocked query")
        driver, blocked_key, unblock = staged
        with self.harness(driver) as h:
            h.needs(probe=True)
            answer: list[tuple[ProbeStatus, ...]] = []
            # Off this thread, so a probe that waits for the store fails instead of hanging.
            prober = threading.Thread(
                target=lambda: answer.append(driver.probe([blocked_key], 0.05)), daemon=True
            )
            prober.start()
            prober.join(0.05 + 0.5)
            try:
                assert not prober.is_alive(), "probe waited for the store past its budget"
                assert answer == [(ProbeStatus.UNKNOWN,)]
            finally:
                unblock()
                prober.join(h.timeout)

    def test_probe_raises_on_a_store_error_rather_than_answering(self) -> None:
        staged = self.make_failing_presence()
        if staged is None:
            pytest.skip("this driver cannot stage a failing presence query")
        driver, key = staged
        with self.harness(driver) as h:
            h.needs(probe=True)
            try:
                answer = driver.probe([key], h.timeout)
            except DriverRejected as error:
                pytest.fail(f"a store error was reported as a rejection: {error!r}")
            except Exception:  # noqa: BLE001 - any store error will do
                return
            pytest.fail(f"probe answered {answer} on a store error")

    # Registration, drain and close.

    def test_unregister_is_idempotent_and_frees_the_range(self) -> None:
        with self.harness() as h:
            h.needs(registers=True)
            endpoint = h.endpoint(["A"])
            with pytest.raises(DriverRejected):
                h.driver.register(endpoint.memory.address, SLOT, RegionLayout(SLOT, 1))
            endpoint.unregister()
            endpoint.unregister()
            try:
                handle = h.driver.register(endpoint.memory.address, SLOT, RegionLayout(SLOT, 1))
            except DriverRejected as error:
                pytest.fail(f"the range stayed registered: {error}")
            h.driver.unregister(handle)

    def test_drain_is_one_way(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True)
            writer = h.publish({"A": b"aaaaaaaa"})
            assert h.driver.drain(self.timeout) is True
            with pytest.raises(DriverRejected):
                h.put(writer.rows())
            with pytest.raises(DriverRejected):
                h.get(writer.rows())
            assert h.driver.drain(self.timeout) is True
            assert writer.read_row(0) == b"aaaaaaaa"

    def test_close_needs_a_true_drain_and_no_live_registrations(self) -> None:
        with self.harness() as h:
            h.needs(put=True, registers=True)
            writer = h.publish({"A": b"aaaaaaaa"})
            with pytest.raises(DriverRejected):
                h.driver.close()
            assert h.driver.drain(self.timeout) is True
            with pytest.raises(DriverRejected):
                h.driver.close()
            writer.unregister()
            h.driver.close()
            h.driver.close()
            h.done = True  # closed by the check itself
            with pytest.raises(DriverRejected):
                h.put(writer.rows())
