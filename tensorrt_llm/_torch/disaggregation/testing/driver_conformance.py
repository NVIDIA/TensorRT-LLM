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

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.keys import encode_key
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverRejected,
    Hint,
    OpHandle,
    ProbeStatus,
    RegionHandle,
    RegionLayout,
    RegionSpec,
    RegistersRegionsTogether,
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
ROW_LIMIT_CHECKED = 64  # a row limit this high is taken on trust, so the check stays small

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
        memory_kind: MemoryKind = MemoryKind.HOST,
    ) -> None:
        self.driver = driver
        self.labels = list(labels)
        self.memory = memory
        self.prefix = prefix
        self.slot = slot
        self.memory_kind = memory_kind
        self.handle = handle
        # A handle given up front belongs to whoever allocated the memory; this endpoint only
        # unregisters what it registered itself.
        self.owns_registration = handle is None
        self.released = False  # unregistered or given back; a cleanup retry skips it

    def register(self) -> None:
        """Registers the region. When the store may have kept it despite an error, the handle
        is recorded all the same, so the region can still be unregistered later."""
        layout = RegionLayout(self.slot, len(self.labels), "kv", self.memory_kind)
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

# Memory a batch registration may still hold when a check ends. Never freed.
UNRELEASED_MEMORY: list[Memory] = []


def _revoke_again(driver: RegistersRegionsTogether, handle: RegionHandle) -> None:
    """A second ``unregister_regions`` of a revoked batch; any raise breaks idempotency."""
    try:
        driver.unregister_regions(handle)
    except Exception as error:  # noqa: BLE001 - the rule allows no raise at all
        pytest.fail(f"a second revocation of the batch raised {error!r}")


def _prove_registered(driver: StoreDriver, memory: Memory, index: int, timeout: float) -> None:
    """A row inside ``memory`` moves and its access ends: the region is registered. A probe by
    registration could not tell, since a driver that takes one batch at a time refuses a new one
    whatever the region's state."""
    row = RowIO(key_of(f"R{index}", "t" + secrets.token_hex(4)), (Span(memory.address, SLOT),))
    try:
        if driver.capabilities.put:
            memory.write(b"a" * SLOT, 0)
            op = driver.put([row])
        else:
            op = driver.get([row])
    except DriverRejected as error:
        pytest.fail(f"region {index} of the batch takes no row: {error}")
    wait_until(lambda: concluded(driver.poll(op)), "the operation to conclude", timeout)
    if driver.capabilities.put:
        assert statuses(driver, op) == [RowStatus.READY]
    wait_until(lambda: driver.access_ended(op), "access to end", timeout)


def specs_of(memory: Sequence[Memory], slot: int = SLOT) -> list[RegionSpec]:
    """One region per memory, laid out in slots of ``slot`` bytes, for ``register_regions``."""
    return [
        RegionSpec(m.address, m.nbytes, RegionLayout(slot, m.nbytes // slot, "kv")) for m in memory
    ]


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

    def endpoint(
        self,
        labels: Sequence[str],
        *,
        fill: int = 0,
        slot: int = SLOT,
        memory_kind: MemoryKind = MemoryKind.HOST,
    ) -> Endpoint:
        memory, handle = self.suite.allocate(slot * len(labels), fill, memory_kind)
        endpoint = Endpoint(
            self.driver,
            labels,
            memory,
            handle,
            prefix=self.prefix,
            slot=slot,
            memory_kind=memory_kind,
        )
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
        """Puts ``rows`` (label to payload) and returns the writer endpoint, its source bytes
        checked once access ended: a put reads its rows and never writes them."""
        writer = self.endpoint(list(rows), slot=slot)
        for i, data in enumerate(rows.values()):
            writer.fill_row(i, data)
        op = self.put(writer.rows())
        assert [r.status for r in self.settle(op)] == [RowStatus.READY] * len(rows)
        self.settle_access(op)
        for i, data in enumerate(rows.values()):
            assert writer.memory.read(i * slot, len(data)) == data, "a put changed its source"
        assert writer.intact()
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
        """What a get of a key nobody published must look like: MISS, or FAULT only for a driver
        without ``absent_is_miss`` whose store could not tell absence from failure. Whichever
        driver answers MISS, the slot is untouched."""
        allowed = [RowStatus.MISS]
        if not self.driver.capabilities.absent_is_miss:
            allowed.append(RowStatus.FAULT)
        assert result.status in allowed, result
        if result.status is RowStatus.MISS:
            assert endpoint.read_row(index) == bytes([0x2D]) * endpoint.slot

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
    whichever scenario hooks the driver's store can stage.

    The checks of a get prepare their objects through the driver's own put, so a driver that
    declares no put skips them and passes the suite without one get being checked; its gets
    count as covered only once its objects are staged some other way.
    """

    timeout = 5.0
    registers_on_demand = True  # endpoints register fresh memory; a pool driver sets False
    # The driver implements RegistersRegionsTogether and, with registers_on_demand False, the
    # handle allocate hands out is the batch its memory was registered as; the suite then revokes
    # through unregister_regions, never through unregister.
    registers_together = False
    handles_before_completion = False  # put/get return their handle before the store answers

    def make_driver(self) -> StoreDriver:
        raise NotImplementedError

    # Memory and lifetime hooks.

    def allocate(self, nbytes: int, fill: int, memory_kind: MemoryKind) -> Allocation:
        """Memory of the given kind for one endpoint and, when the driver registered it up
        front, its handle.

        The memory object keeps its backing allocation alive for as long as it exists: a check
        that cannot prove access ended retains the endpoint, and nothing else, for good. This
        default stands in for either kind with host memory, which is what the in-memory driver
        expects; a driver that really registers device memory allocates it here.
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
        with an event set once a get is stalled, or ``None`` for no such signal. A stalled get is
        inside the store: its row has started, and its access cannot end before the callable
        runs."""
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

    def make_failing_get(self) -> tuple[StoreDriver, str] | None:
        """A driver whose store fails every read of the returned key while a put of it works."""
        return None

    def make_unprovable_get(self) -> tuple[StoreDriver, Callable[[OpHandle], None]] | None:
        """A driver and a callable that, given the handle of a get just submitted, concludes
        it while the driver can never prove that its access ended."""
        return None

    def make_together_registration(self) -> tuple[StoreDriver, list[Memory]] | None:
        """A fresh driver implementing ``RegistersRegionsTogether`` with no batch live, and
        unregistered memory for one batch of at least two regions."""
        return None

    def make_failing_together_registrations(self) -> list[tuple[StoreDriver, list[Memory]]]:
        """Fresh drivers whose next ``register_regions`` of the returned memory fails after the
        store was asked, one per failure shape the store can stage; empty when it can stage
        none."""
        return []

    def make_busy_together_registration(
        self,
    ) -> tuple[StoreDriver, list[Memory], str, threading.Event, Unblock] | None:
        """As ``make_together_registration``, plus a key whose get stalls in the store until the
        callable runs, and an event set once a get of it is stalled. The event is required: the
        check decides nothing by waiting; a suite that cannot give the signal returns ``None``."""
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

            kind = (
                MemoryKind.DEVICE
                if h.driver.capabilities.device_memory and self.registers_on_demand
                else MemoryKind.HOST
            )
            parts = h.endpoint(["X", "Y", "Z"], slot=4, fill=0x2D, memory_kind=kind)
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

    def test_a_larger_object_is_not_truncated_into_a_multi_span_row(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, multi_span=True)
            h.publish({"A": b"aaaaaaaa"})
            parts = h.endpoint(["X", "Y"], slot=3, fill=0x2D)
            row = RowIO(h.key("A"), (parts.span(0), parts.span(1)))  # 6 bytes for an 8-byte object
            (got,) = h.settle(h.get([row]))
            assert got.status is RowStatus.FAULT, got
            assert parts.intact()

    def test_a_failed_store_read_is_a_fault_not_a_miss(self) -> None:
        """A store whose reads fail must not look empty, or it would never be isolated."""
        staged = self.make_failing_get()
        if staged is None:
            pytest.skip("this driver cannot stage a failing read")
        driver, key = staged
        with self.harness(driver) as h:
            h.needs(put=True, get=True)
            writer = h.endpoint(["F"])
            writer.fill_row(0, b"ffffffff")
            (put,) = h.settle(h.put([RowIO(key, (writer.span(0),))]))
            assert put.status is RowStatus.READY
            reader = h.endpoint(["F"], fill=0x2D)
            (got,) = h.settle(h.get([RowIO(key, (reader.span(0),))]))
            assert got.status is RowStatus.FAULT, got
            assert reader.intact()

    # Rejections leave no trace.

    def test_a_row_outside_every_registration_is_rejected_before_any_effect(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True)
            h.publish({"A": b"aaaaaaaa"})
            stray = HostBuffer(SLOT, fill=0x2D)
            with pytest.raises(DriverRejected):
                h.get([RowIO(h.key("A"), (stray.span(),))])
            assert stray.read() == b"-" * SLOT and stray.canaries_intact()
            with pytest.raises(DriverRejected):
                h.put([RowIO(h.key("S"), (stray.span(),))])
            reader = h.endpoint(["S"], fill=0x2D)
            (result,) = h.settle(h.get(reader.rows()))
            h.assert_absent(result, reader, 0)

    def test_a_span_leaving_or_outliving_its_registration_is_rejected(self) -> None:
        with self.harness() as h:
            h.needs(put=True, get=True, registers=True)
            h.publish({"A": b"aaaaaaaa"})
            inside = h.endpoint(["A"], fill=0x2D)
            crossing = Span(inside.memory.address + SLOT // 2, SLOT)  # starts inside, ends past
            with pytest.raises(DriverRejected):
                h.get([RowIO(h.key("A"), (crossing,))])
            assert inside.read_row(0) == b"-" * SLOT and inside.intact()

            inside.fill_row(0, b"bbbbbbbb")
            inside.unregister()
            with pytest.raises(DriverRejected):
                h.put([RowIO(h.key("B"), (inside.span(0),))])
            reader = h.endpoint(["B"], fill=0x2D)
            (result,) = h.settle(h.get(reader.rows()))
            h.assert_absent(result, reader, 0)

    def test_a_device_region_is_rejected_unless_device_memory_is_declared(self) -> None:
        with self.harness() as h:
            h.needs(registers=True)
            if h.driver.capabilities.device_memory:
                pytest.skip("this driver takes device regions")
            memory = HostBuffer(SLOT, fill=0x2D)  # stands in for a device region in-process
            device = RegionLayout(SLOT, 1, "kv", MemoryKind.DEVICE)
            with pytest.raises(DriverRejected):
                h.driver.register(memory.address, SLOT, device)
            handle = h.driver.register(memory.address, SLOT, RegionLayout(SLOT, 1, "kv"))
            h.driver.unregister(handle)

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
            if limit is None:
                pytest.skip("this driver declares no row limit")
            if limit >= ROW_LIMIT_CHECKED:
                pytest.skip(f"a limit of {limit} rows is above what this check allocates")
            writer = h.endpoint([f"R{i}" for i in range(limit + 1)])
            with pytest.raises(DriverRejected):
                h.put(writer.rows())
            reader = h.endpoint(["R0"], fill=0x2D)
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
        """Both objects exist: a get of an absent key may conclude FAULT for a driver without
        ``absent_is_miss``, and a faulted row may leave access unprovable, which is allowed."""
        with self.harness() as h:
            h.needs(put=True, get=True, rows=2)
            h.publish({"A": b"aaaaaaaa", "B": b"bbbbbbbb"})
            reader = h.endpoint(["A", "B"])
            op = h.get(reader.rows())
            results = h.settle(op)
            assert [r.status for r in results] == [RowStatus.READY, RowStatus.READY]
            assert h.driver.poll(op) == results
            h.settle_access(op)
            assert h.driver.access_ended(op)
            assert not h.driver.access_unprovable(op)
            assert h.driver.poll(op) == results

    def test_an_unprovable_get_makes_drain_return_at_once_and_keeps_its_region(self) -> None:
        staged = self.make_unprovable_get()
        if staged is None:
            pytest.skip("this driver cannot stage an operation whose end it cannot prove")
        driver, lose_the_evidence = staged
        with self.harness(driver) as h:
            h.needs(get=True)
            reader = h.endpoint(["U"], fill=0x2D)
            op = h.get(reader.rows())
            lose_the_evidence(op)
            h.settle(op)
            wait_until(
                lambda: driver.access_unprovable(op), "the end to be declared unprovable", h.timeout
            )
            assert not driver.access_ended(op)
            started = time.monotonic()
            assert driver.drain(h.timeout) is False
            assert time.monotonic() - started < h.timeout / 2, (
                "drain waited out its budget on an operation that cannot prove its end"
            )
            if self.registers_on_demand:
                with pytest.raises(DriverRejected):
                    reader.unregister()

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

    def test_cancel_of_a_get_inside_the_store_neither_waits_nor_lies(self) -> None:
        """Cancel is best effort, so only what it must not do is checked: wait for the store,
        call a row the store already took CANCELLED, or end access while that row is inside the
        store. The hook guarantees that access cannot end before the gate opens, so the last
        holds whatever result the row carries. A driver that hands out the handle only once the
        store answered cannot be asked."""
        if not self.handles_before_completion:
            pytest.skip("this driver returns the handle of a get only once the store answered")
        staged = self.make_blocked_get()
        if staged is None:
            pytest.skip("this driver cannot stage a get blocked in the store")
        driver, blocked_key, blocked, unblock = staged
        with self.harness(driver) as h:
            h.needs(get=True, cancel=True)
            if blocked is None:
                pytest.skip("this driver's blocked get gives no signal that it reached the store")
            reader = h.endpoint(["S"], fill=0x2D)
            handed: dict[str, OpHandle] = {}
            handle_back = threading.Event()
            returned = threading.Event()
            failures: list[BaseException] = []

            def submit() -> None:
                try:
                    handed["op"] = h.get([RowIO(blocked_key, (reader.span(0),))])
                except BaseException as error:  # noqa: BLE001 - reported from the main thread
                    failures.append(error)
                finally:
                    handle_back.set()

            def cancel() -> None:
                driver.cancel(handed["op"])
                returned.set()

            # Both off this thread, so a get or a cancel that waits for the store fails instead
            # of hanging; the gate opens on every exit.
            submitter = threading.Thread(target=submit, daemon=True)
            canceller = threading.Thread(target=cancel, daemon=True)
            try:
                submitter.start()
                assert handle_back.wait(h.timeout), "the get kept its handle"
                if failures:
                    raise AssertionError(f"get raised: {failures[0]!r}") from failures[0]
                op = handed["op"]
                assert blocked.wait(h.timeout), "the get never reached the store"
                canceller.start()
                assert returned.wait(h.timeout), "cancel waited for the store"
                (during,) = driver.poll(op)
                assert during.status is not RowStatus.CANCELLED, (
                    "a row the store had taken was called cancelled"
                )
                assert not driver.access_ended(op), "access ended with the row in the store"
            finally:
                unblock()
                for thread in (submitter, canceller):
                    if thread.ident is not None:  # started
                        thread.join(h.timeout)
            assert not submitter.is_alive() and not canceller.is_alive()
            (after,) = h.settle(op)
            if during.concluded:
                assert after == during

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

            handle_back = threading.Event()

            def slow_get() -> None:
                try:
                    seen["slow_op"] = h.get([RowIO(blocked_key, (slow.span(0),))])
                    handle_back.set()
                except BaseException as error:  # noqa: BLE001 - reported from the main thread
                    failures.append(error)

            def quick_calls() -> None:
                try:
                    op = h.get(quick.rows())
                    seen["quick"] = h.settle(op)
                    seen["ended"] = h.driver.access_ended(op)
                    # A driver that returns the handle before the store answers is asked about
                    # the stalled operation itself; the answers must come at once. A suite that
                    # declares such a driver waits for the handle, so this branch always runs.
                    if self.handles_before_completion:
                        assert handle_back.wait(h.timeout), "the slow get kept its handle"
                    if handle_back.is_set():
                        slow_op = seen["slow_op"]
                        seen["slow_poll"] = h.driver.poll(slow_op)
                        seen["slow_ended"] = h.driver.access_ended(slow_op)
                        seen["slow_unprovable"] = h.driver.access_unprovable(slow_op)
                        seen["slow_checked"] = True
                    if self.registers_on_demand:
                        try:
                            slow.unregister()
                        except DriverRejected:
                            seen["unregister"] = "refused"
                        else:
                            seen["unregister"] = "accepted"
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
                if self.registers_on_demand:
                    assert seen["unregister"] == "refused", "a busy region was unregistered"
                if self.handles_before_completion:
                    assert seen.get("slow_checked"), "the stalled get was never asked about"
                if seen.get("slow_checked"):
                    assert seen["slow_ended"] is False and seen["slow_unprovable"] is False
                    slow_results = seen["slow_poll"]
                    assert isinstance(slow_results, tuple)
                    assert all(r.status is not RowStatus.READY for r in slow_results)
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

    def test_unregister_is_idempotent(self) -> None:
        """Also for a driver whose memory was registered up front: its one registration, or the
        batch it was registered as, is revoked twice through the call that made it, and the
        suite's shutdown hook must cope with that."""
        with self.harness() as h:
            endpoint = h.endpoint(["A"])
            assert endpoint.handle is not None
            assert h.driver.drain(self.timeout) is True
            revoke = h.driver.unregister
            if self.registers_together and not self.registers_on_demand:
                assert isinstance(h.driver, RegistersRegionsTogether)
                revoke = h.driver.unregister_regions
            revoke(endpoint.handle)
            revoke(endpoint.handle)
            endpoint.released = True  # the harness must not give pool memory back twice

    def test_unregister_frees_the_range_for_a_new_registration(self) -> None:
        with self.harness() as h:
            h.needs(registers=True)
            endpoint = h.endpoint(["A"])
            with pytest.raises(DriverRejected):
                h.driver.register(endpoint.memory.address, SLOT, RegionLayout(SLOT, 1))
            endpoint.unregister()
            try:
                handle = h.driver.register(endpoint.memory.address, SLOT, RegionLayout(SLOT, 1))
            except DriverRejected as error:
                pytest.fail(f"the range stayed registered: {error}")
            h.driver.unregister(handle)

    def test_regions_registered_together_are_revoked_together_and_the_range_freed(self) -> None:
        """Through the optional ``RegistersRegionsTogether``: every region of a batch is live once
        registered, each overlaps a new registration until the batch is revoked, revocation is
        idempotent, and the whole range is free after."""
        staged = self.make_together_registration()
        if staged is None:
            pytest.skip("this suite stages no batch registration")
        driver, memory = staged
        assert isinstance(driver, RegistersRegionsTogether)
        assert len(memory) >= 2, "a batch check needs at least two regions"
        specs = specs_of(memory)
        released = False
        try:
            handle = driver.register_regions(specs)
            for index, spec in enumerate(specs):
                try:
                    stray = driver.register_regions([spec])
                except DriverRejected:
                    continue  # the region is live: it overlaps a new registration
                driver.unregister_regions(stray)
                pytest.fail(f"region {index} of the batch was not registered")
            for index, region in enumerate(memory):  # the registrations are real: rows move
                _prove_registered(driver, region, index, self.timeout)
            driver.unregister_regions(handle)
            _revoke_again(driver, handle)
            try:
                again = driver.register_regions(specs)
            except DriverRejected as error:
                pytest.fail(f"the revoked batch kept its range: {error}")
            driver.unregister_regions(again)
            assert driver.drain(self.timeout) is True
            driver.close()
            released = True
        finally:
            if not released:
                UNRELEASED_MEMORY.extend(memory)

    def test_a_batch_is_not_revoked_while_any_of_its_regions_is_busy(self) -> None:
        """``unregister_regions`` refuses while an operation may still touch any region of the
        batch and leaves every region registered, the idle ones included; once access ended it
        succeeds, idempotently."""
        staged = self.make_busy_together_registration()
        if staged is None:
            pytest.skip("this suite stages no batch with a stalled get")
        driver, memory, key, blocked, unblock = staged
        assert isinstance(driver, RegistersRegionsTogether)
        assert len(memory) >= 2, "a batch check needs at least two regions"
        if not driver.capabilities.get:
            pytest.skip("this driver declares no get")
        specs = specs_of(memory)
        slow: dict[str, OpHandle] = {}
        row = RowIO(key, (Span(memory[-1].address, SLOT),))  # the last region is the busy one
        getter = threading.Thread(target=lambda: slow.update(op=driver.get([row])), daemon=True)
        refusal: list[BaseException | None] = []

        def revoke_while_busy() -> None:
            try:
                driver.unregister_regions(handle)
            except BaseException as error:  # noqa: BLE001 - judged on the main thread
                refusal.append(error)
            else:
                refusal.append(None)

        # Off this thread: a revocation that waits for the transfer would wait for the gate this
        # thread holds, and must be a failure rather than a hang.
        revoker = threading.Thread(target=revoke_while_busy, daemon=True)
        released = False
        try:
            handle = driver.register_regions(specs)
            getter.start()  # a blocking driver's get returns only once the store lets it
            assert blocked.wait(self.timeout), "the get never reached the store"
            revoker.start()
            revoker.join(self.timeout)
            assert not revoker.is_alive(), "unregister_regions waited for the transfer"
            assert isinstance(refusal[0], DriverRejected), (
                f"a busy batch was revoked: {refusal[0]!r}"
            )
            for index, region in enumerate(memory[:-1]):  # the idle regions stayed registered
                _prove_registered(driver, region, index, self.timeout)
            unblock()
            getter.join(self.timeout)
            assert not getter.is_alive(), "the stalled get never returned"
            wait_until(lambda: driver.access_ended(slow["op"]), "access to end", self.timeout)
            driver.unregister_regions(handle)
            _revoke_again(driver, handle)
            assert driver.drain(self.timeout) is True
            driver.close()
            released = True
        finally:
            unblock()
            for worker in (getter, revoker):
                if worker.ident is not None:  # started: let it leave before the check ends
                    worker.join(self.timeout)
            if not released:
                UNRELEASED_MEMORY.extend(memory)

    def test_a_failed_batch_registration_is_revocable_or_left_nothing_behind(self) -> None:
        """A ``register_regions`` that fails after the store was asked either certifies that
        nothing of the batch is registered, so the same memory registers again at once, or
        hands out a handle that revokes whatever is, idempotently. A failure of any other shape
        is allowed and leaves the memory protected; nothing more is checked for it."""
        staged = self.make_failing_together_registrations()
        if not staged:
            pytest.skip("this suite stages no failing batch registration")
        for driver, memory in staged:
            assert isinstance(driver, RegistersRegionsTogether)
            specs = specs_of(memory)
            released = False
            try:
                try:
                    driver.register_regions(specs)
                except DriverRejected:
                    handle = None
                except RegistrationError as error:
                    handle = error.handle
                except Exception:  # noqa: BLE001 - some of the batch may be registered for good
                    continue
                else:
                    pytest.fail("the staged registration did not fail")
                if handle is not None:
                    driver.unregister_regions(handle)
                    _revoke_again(driver, handle)
                try:
                    again = driver.register_regions(specs)
                except DriverRejected as error:
                    pytest.fail(f"something of the failed batch stayed registered: {error}")
                driver.unregister_regions(again)
                assert driver.drain(self.timeout) is True
                driver.close()
                released = True
            finally:
                if not released:
                    UNRELEASED_MEMORY.extend(memory)

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
        """For a driver whose memory was registered up front, the suite's shutdown hook does
        the unregistering; the order it has to keep is the same."""
        with self.harness() as h:
            h.needs(put=True)
            writer = h.publish({"A": b"aaaaaaaa"})
            with pytest.raises(DriverRejected):
                h.driver.close()
            assert h.driver.drain(self.timeout) is True
            with pytest.raises(DriverRejected):
                h.driver.close()
            if self.registers_on_demand:
                writer.unregister()
                h.driver.close()
            else:
                self.shutdown(h.driver)
            h.driver.close()  # idempotent
            h.done = True  # closed by the check itself
            with pytest.raises(DriverRejected):
                h.put(writer.rows())
