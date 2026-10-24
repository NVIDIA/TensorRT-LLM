# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU stand-ins for the lender side of a blob backend, and ways for a test to steer a driver.

``Endpoint`` is one side of a transfer: host slots in one registered region, a driver, a region
binding, a backend and the waits a test lease holder needs. ``KeyGatedStore`` holds store calls
until the test opens its gate. ``DriverSpy`` forwards to a real driver, records every call with its
thread, and can hold a method on an event or make it raise. Every wait has an upper bound.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from tensorrt_llm._torch.disaggregation.base.backend import MemoryKind, Outcome
from tensorrt_llm._torch.disaggregation.blob.backend import (
    BlobAttempt,
    BlobBackend,
    BlobConfig,
    RegionBinding,
)
from tensorrt_llm._torch.disaggregation.blob.keys import encode_key
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    Hint,
    OnChange,
    OpHandle,
    RegionHandle,
    RegionLayout,
    RegistrationError,
    RowBatch,
    RowIO,
    RowResult,
    StoreDriver,
)
from tensorrt_llm._torch.disaggregation.testing.host_memory import HostBuffer
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    Commit,
    MemoryDriver,
    MemoryStore,
    StoredObject,
)
from tensorrt_llm._torch.disaggregation.testing.object_trailer import Checksum, encode_trailer

WAIT_S = 5.0
"""Upper bound of every wait in these tests; reached only when the code under test is wrong."""

HELD_CALL_S = 10 * WAIT_S
"""How long a held call waits for its gate: past any test wait, so a gate left shut fails the test's
own wait first, yet a failed test never strands the thread."""

PROMPT_S = 0.5
"""A call that must not wait on the driver returns well within this."""

KEY_PREFIX = "t"
SLOT_BYTES = 8
OBJECT_KEY_WIDTH = 16


def object_key(name: str) -> np.ndarray:
    """The lender-style opaque object key of a test row name."""
    digest = hashlib.sha256(name.encode()).digest()[:OBJECT_KEY_WIDTH]
    return np.frombuffer(digest, dtype=np.uint8)


def store_key(name: str) -> str:
    """The store key the backend derives for a test row name."""
    return encode_key(KEY_PREFIX, object_key(name).tobytes())


def commit_object(store: MemoryStore, name: str, data: bytes) -> Commit:
    """Commits ``data`` straight into ``store`` under the key the backend derives for ``name``,
    with the trailer ``MemoryDriver`` keeps beside each object."""
    key = store_key(name)
    trailer = encode_trailer(key.encode(), len(data), checksum=Checksum.CRC32, payload=data)
    return store.commit(key, data, trailer)


def masks(outcome: Outcome) -> list[list[bool]]:
    return [mask.tolist() for mask in outcome.delivered_masks]


def wait_until(predicate: Callable[[], bool], what: str, timeout: float = WAIT_S) -> None:
    """Checks ``predicate`` every millisecond until it holds; fails the test after ``timeout``."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.001)


class _Sealable:
    """Raises on any attribute read once sealed, to prove nothing reads it any more."""

    _sealed = False

    def seal(self) -> None:
        object.__setattr__(self, "_sealed", True)

    def __getattribute__(self, name: str):
        if name != "_sealed" and object.__getattribute__(self, "_sealed"):
            raise AssertionError(f"{type(self).__name__}.{name} read after the view was sealed")
        return object.__getattribute__(self, name)


class CpuRegion:
    """``slot_count`` slots of ``slot_bytes`` in host memory: a ``SlotRegion``. ``memory_kind`` is
    what the region declares; DEVICE stands for device memory in-process."""

    def __init__(
        self,
        slot_bytes: int,
        slot_count: int,
        fill: int = 0,
        layout_label: str = "kv",
        memory_kind: MemoryKind = MemoryKind.HOST,
    ) -> None:
        self.buffer = HostBuffer(slot_bytes * max(slot_count, 1), fill)
        self.layout_label = layout_label
        self.memory_kind = memory_kind
        self.address = self.buffer.address
        self.nbytes = self.buffer.nbytes
        self.slot_bytes = slot_bytes
        self.slot_count = slot_count

    def slot(self, index: int) -> int:
        return self.address + index * self.slot_bytes

    def slots(self, *indexes: int) -> np.ndarray:
        """The addresses of slots ``indexes``, as a view's ``addresses``."""
        return np.array([self.slot(index) for index in indexes], dtype=np.int64)

    def read(self, index: int) -> bytes:
        return self.buffer.read(index * self.slot_bytes, self.slot_bytes)

    def write(self, index: int, data: bytes) -> None:
        if len(data) != self.slot_bytes:
            raise ValueError(f"a slot holds {self.slot_bytes} bytes, got {len(data)}")
        self.buffer.write(data, index * self.slot_bytes)


class Rows(_Sealable):
    """One group of a view: a ``LayerGroupRows``. ``rows`` overrides ``len`` for malformed views."""

    def __init__(
        self,
        layer_group_index: int,
        object_keys: np.ndarray | None,
        addresses: np.ndarray | None,
        region_index: int | None,
        rows: int | None = None,
    ) -> None:
        self.layer_group_index = layer_group_index
        self.object_keys = object_keys
        self.addresses = addresses
        self.region_index = region_index
        self.rows = rows

    def __len__(self) -> int:
        if self.rows is not None:
            return self.rows
        return 0 if self.addresses is None else len(self.addresses)


class CpuView(_Sealable):
    """A ``View`` over ``Rows`` groups; ``seal`` makes any later read of it or its groups raise."""

    def __init__(self, groups: Iterable[Rows]) -> None:
        self.groups = tuple(groups)

    def row_masks(self, fill_value: bool = False) -> tuple[np.ndarray, ...]:
        return tuple(np.full(len(group), bool(fill_value)) for group in self.groups)

    def seal(self) -> None:
        for group in self.groups:
            group.seal()
        super().seal()


class FakeHold:
    """A lender region hold that records each release and its thread."""

    def __init__(self) -> None:
        self.release_threads: list[int] = []

    def release(self) -> None:
        self.release_threads.append(threading.get_ident())

    @property
    def released(self) -> bool:
        return bool(self.release_threads)


class WakeCounter:
    """A ``WakeFn`` that counts calls; tests wait on it for state the backend published."""

    def __init__(self, on_wake: Callable[[], None] | None = None) -> None:
        self.count = 0
        self.on_wake = on_wake
        self._woken = threading.Condition()

    def __call__(self) -> None:
        if self.on_wake is not None:
            self.on_wake()
        with self._woken:
            self.count += 1
            self._woken.notify_all()

    def wait_for(self, predicate: Callable[[], bool], what: str) -> None:
        """Waits for ``predicate``, re-checked on every wake. State that lands without a wake times
        the test out only when it lands after the wait began; a test that pins a wake counts
        the wakes itself."""
        with self._woken:
            assert self._woken.wait_for(predicate, WAIT_S), f"no wake brought {what}"


class KeyGatedStore(MemoryStore):
    """Holds ``get``s and/or ``commit``s of the chosen keys (all by default) until the test sets
    ``gate``; ``waiting`` counts the calls that reached it."""

    def __init__(
        self, *, gets: bool = False, commits: bool = False, keys: Iterable[str] | None = None
    ) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.waiting = 0
        self._gets = gets
        self._commits = commits
        self._keys = None if keys is None else set(keys)
        self._arrived = threading.Condition()

    def get(self, key: str) -> StoredObject | None:
        if self._gets:
            self._hold(key)
        return super().get(key)

    def commit(self, key: str, payload: bytes, trailer: bytes) -> Commit:
        if self._commits:
            self._hold(key)
        return super().commit(key, payload, trailer)

    def wait_for_waiting(self, count: int) -> None:
        with self._arrived:
            arrived = self._arrived.wait_for(lambda: self.waiting >= count, WAIT_S)
            assert arrived, f"{self.waiting} of {count} store calls reached the gate"

    def _hold(self, key: str) -> None:
        if self.gate.is_set() or (self._keys is not None and key not in self._keys):
            return
        with self._arrived:
            self.waiting += 1
            self._arrived.notify_all()
        self.gate.wait(HELD_CALL_S)


class RollbackFailsDriver(MemoryDriver):
    """Registers the first region with a revocation that will fail, then refuses the second."""

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> RegionHandle:
        if self.live_registrations():
            raise RegistrationError("out of memory regions")
        handle = super().register(address, nbytes, layout)
        self.fail_next_unregister(handle, "device busy")
        return handle


@dataclass(frozen=True)
class Call:
    """One call into a ``DriverSpy``."""

    method: str
    thread: str
    rows: tuple[RowIO, ...] = ()
    op: object = None


class DriverSpy:
    """A ``StoreDriver`` that forwards to ``inner`` and records every call with its thread name.

    ``hold(method)`` makes calls of ``method`` wait until the returned event is set;
    ``fail(method, error, times)`` makes the next ``times`` calls raise ``error`` instead;
    ``answer(method, value, times)`` makes them return ``value`` without reaching the driver.
    ``ops`` lists every operation ``put``/``get`` returned, with its rows; ``tell_changed(op)``
    calls the ``on_change`` that came with it again, as a driver may.
    """

    def __init__(self, inner: StoreDriver) -> None:
        self.inner = inner
        self.calls: list[Call] = []
        self.ops: list[tuple[tuple[RowIO, ...], OpHandle]] = []
        self._on_changes: dict[OpHandle, OnChange | None] = {}
        self._changed = threading.Condition()
        self._holds: dict[str, threading.Event] = {}
        self._failures: dict[str, list] = {}
        self._answers: dict[str, list] = {}

    def hold(self, method: str) -> threading.Event:
        self._holds[method] = threading.Event()
        return self._holds[method]

    def fail(self, method: str, error: Exception, times: int = 1) -> None:
        self._failures[method] = [error, times]

    def answer(self, method: str, value: object, times: int = 1) -> None:
        self._answers[method] = [value, times]

    def op_with(self, key: str) -> OpHandle:
        """The latest operation carrying the row stored under ``key``, once one started."""
        found: list[OpHandle] = []

        def started() -> bool:
            found[:] = [op for rows, op in self.ops if any(row.key == key for row in rows)]
            return bool(found)

        self.wait_for(started, f"an operation for {key}")
        return found[-1]

    def tell_changed(self, op: OpHandle) -> None:
        """Calls ``op``'s ``on_change`` on this thread, as a driver telling of a change would."""
        with self._changed:
            on_change = self._on_changes[op]
        on_change(op)

    def calls_to(self, method: str, op: object = None) -> list[Call]:
        with self._changed:
            return [c for c in self.calls if c.method == method and (op is None or c.op is op)]

    def wait_for(self, predicate: Callable[[], bool], what: str) -> None:
        with self._changed:
            assert self._changed.wait_for(predicate, WAIT_S), f"the driver never saw {what}"

    @property
    def capabilities(self) -> DriverCapabilities:
        return self.inner.capabilities

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> RegionHandle:
        self._enter("register")
        return self.inner.register(address, nbytes, layout)

    def unregister(self, handle: RegionHandle) -> None:
        self._enter("unregister")
        self.inner.unregister(handle)

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> OpHandle:
        self._enter("put", rows=rows)
        return self._started(rows, self.inner.put(rows, on_change=on_change), on_change)

    def get(
        self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None
    ) -> OpHandle:
        self._enter("get", rows=rows)
        if hint is None:
            op = self.inner.get(rows, on_change=on_change)
        else:
            op = self.inner.get(rows, hint, on_change=on_change)
        return self._started(rows, op, on_change)

    def poll(self, op: OpHandle) -> tuple[RowResult, ...]:
        self._enter("poll", op=op)
        scripted = self._scripted_answer("poll")
        return self.inner.poll(op) if scripted is None else scripted[0]

    def access_ended(self, op: OpHandle) -> bool:
        self._enter("access_ended", op=op)
        return self.inner.access_ended(op)

    def access_unprovable(self, op: OpHandle) -> bool:
        self._enter("access_unprovable", op=op)
        return self.inner.access_unprovable(op)

    def cancel(self, op: OpHandle) -> None:
        self._enter("cancel", op=op)
        self.inner.cancel(op)

    def drain(self, budget_s: float) -> bool:
        self._enter("drain")
        scripted = self._scripted_answer("drain")
        return self.inner.drain(budget_s) if scripted is None else scripted[0]

    def close(self) -> None:
        self._enter("close")
        self.inner.close()

    def _scripted_answer(self, method: str) -> list | None:
        with self._changed:
            scripted = self._answers.get(method)
            if scripted is None or scripted[1] <= 0:
                return None
            scripted[1] -= 1
            return scripted

    def _started(self, rows: Sequence[RowIO], op: OpHandle, on_change: OnChange | None) -> OpHandle:
        with self._changed:
            self.ops.append((tuple(rows), op))
            self._on_changes[op] = on_change
            self._changed.notify_all()
        return op

    def _enter(self, method: str, rows: Sequence[RowIO] = (), op: object = None) -> None:
        with self._changed:
            self.calls.append(Call(method, threading.current_thread().name, tuple(rows), op))
            self._changed.notify_all()
            failure = self._failures.get(method)
            if failure is not None and failure[1] > 0:
                failure[1] -= 1
            else:
                failure = None
            hold = self._holds.get(method)
        if hold is not None:
            hold.wait(HELD_CALL_S)
        if failure is not None:
            raise failure[0]


class Endpoint:
    """One side of a transfer: host slots in one registered region, a driver, a backend, waits.

    ``others`` are further regions of the lender, after the endpoint's own (region 0); the binding
    registers those the driver can access. Slot ``i`` belongs to the row named ``names[i]``.
    ``driver`` is a ``DriverSpy`` over the driver passed in: a test steers the one it passed
    (``complete``, ``end_access``) and asks the spy what was called. ``close`` is the clean end
    every test that finishes its transfers reaches: shutdown returns no labels, the hold is released
    on this thread and every backend thread has exited.
    """

    def __init__(
        self,
        driver: StoreDriver,
        names: Sequence[str],
        *,
        slot_bytes: int = SLOT_BYTES,
        fill: int = 0,
        on_wake: Callable[[], None] | None = None,
        others: Sequence[CpuRegion] = (),
    ) -> None:
        self.names = list(names)
        self.region = CpuRegion(slot_bytes, len(self.names), fill)
        self.driver = DriverSpy(driver)
        self.hold = FakeHold()
        self.binding = RegionBinding((self.region, *others), self.driver, self.hold)
        self.binding.register()
        self.wake = WakeCounter(on_wake)
        before = set(threading.enumerate())
        self.backend = BlobBackend(self.binding, self.wake, BlobConfig(key_prefix=KEY_PREFIX))
        self.threads = set(threading.enumerate()) - before

    def rows(self, names: Sequence[str], layer_group_index: int = 0) -> Rows:
        keys = np.stack([object_key(name) for name in names]) if names else _no_keys()
        slots = self.region.slots(*(self.names.index(name) for name in names))
        return Rows(layer_group_index, keys, slots, 0)

    def view(self, *names: str) -> CpuView:
        return CpuView([self.rows(names)])

    def write(self, name: str, data: bytes) -> None:
        self.region.write(self.names.index(name), data)

    def read(self, name: str) -> bytes:
        return self.region.read(self.names.index(name))

    def wait_outcome(self, attempt: BlobAttempt) -> Outcome:
        self.wake.wait_for(lambda: attempt.poll() is not None, f"an outcome for {attempt.label}")
        return attempt.poll()

    def op(self, name: str) -> OpHandle:
        """The latest driver operation carrying the row ``name``, once one started."""
        return self.driver.op_with(store_key(name))

    def wait_quiesced(self, attempt: BlobAttempt) -> None:
        self.wake.wait_for(attempt.quiesced, f"{attempt.label} quiescing")

    def close(self) -> None:
        assert self.backend.stats()["attempts_active"] == 0, "an attempt outlived its transfers"
        labels = self.backend.shutdown(WAIT_S)
        assert labels == frozenset(), f"still accessible after shutdown: {sorted(labels)}"
        self.binding.release_hold()
        assert self.hold.release_threads == [threading.get_ident()]
        for thread in self.threads:
            thread.join(WAIT_S)
        assert not any(thread.is_alive() for thread in self.threads)
        # A background ``MemoryDriver``'s threads count as finished a moment before they exit.
        wait_until(
            lambda: not any(t.name == "memory-driver-op" for t in threading.enumerate()),
            "the driver's operation threads to exit",
        )


def _no_keys() -> np.ndarray:
    return np.zeros((0, OBJECT_KEY_WIDTH), dtype=np.uint8)
