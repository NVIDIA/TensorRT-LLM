# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An in-process store and a driver over it, for tests.

``MemoryStore`` is a dictionary of complete objects that endpoints in one process can share. It
can be given a capacity, and a test can mark a key as held by an unfinished write, corrupt an
object or make a presence query fail. ``MemoryDriver`` really copies bytes between registered
host memory and that store, gathering a row's spans into one object and scattering them back,
and keeps a trailer beside each object that it verifies on every get, the way a real store checks
sizes and checksums natively.

In automatic mode ``put``/``get`` run the operation before returning; in background mode they
start a thread of the driver's own to run it. Neither keeps the protocol's rule that ``put`` and
``get`` only record and queue the work, which a test can afford, and automatic mode waits on the
store, so over a store a test holds, use background mode. In controlled mode the test releases
three things separately per operation: the transfer, the logical results and the end of access,
so it can stage partial arrival, a result before or after access end, a late write after a fault,
an operation whose end can never be proven, and registration failures.

Transfers copy and call the store outside the driver's lock, so queries never wait on the store
and one blocked operation never delays another. Each submission check is its own method, so a
test can subclass the driver and disable exactly one rule.

``TogetherMemoryDriver`` is the same driver registering regions in batches, for the optional
``RegistersRegionsTogether`` protocol; its single-region ``register`` and ``unregister`` are
rejected, as for a store that opens its client with the memory it will use.
"""

from __future__ import annotations

import ctypes
import enum
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

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
    StoreError,
    faulted,
)

from .object_trailer import Checksum, check_trailer, encode_trailer

DEFAULT_CAPABILITIES = DriverCapabilities(
    get=True,
    put=True,
    cancel=True,
    probe=True,
    max_rows_per_op=None,
    absent_is_miss=True,
    device_memory=True,  # a DEVICE region is plain memory in-process; the plumbing is what counts
    multi_span=True,
)
MAX_PROBES_IN_FLIGHT = 4

Kind = Literal["put", "get"]

_PENDING = RowResult(RowStatus.PENDING)
_READY = RowResult(RowStatus.READY)
_MISS = RowResult(RowStatus.MISS)
_NOT_READY = RowResult(RowStatus.NOT_READY)
_DECLINED = RowResult(RowStatus.DECLINED)
_CANCELLED = RowResult(RowStatus.CANCELLED)


class ObjectNotReady(StoreError):
    """The key is held by a write that has not finished."""


class Commit(enum.Enum):
    """How the store answered a commit."""

    STORED = "stored"
    SAME = "same"  # an identical object was already there
    DIFFERENT = "different"  # a different object holds the key; it was kept
    DECLINED = "declined"  # no room
    NOT_READY = "not_ready"  # another write holds the key


@dataclass(frozen=True)
class StoredObject:
    """One committed object: its payload and the trailer the driver keeps beside it."""

    payload: bytes
    trailer: bytes


class MemoryStore:
    """Objects by key. Each object is a private copy, committed whole or not at all.

    Subclass and override ``commit``, ``get`` or ``presence`` to make a test's store block or
    fail; they may raise ``StoreError``, which the driver turns into a row fault or re-raises from
    ``probe``.
    """

    def __init__(self, capacity: int | None = None) -> None:
        self._lock = threading.Lock()
        self._objects: dict[str, StoredObject] = {}
        self._writing: set[str] = set()
        self._presence_failure: str | None = None
        self.capacity = capacity

    def commit(self, key: str, payload: bytes, trailer: bytes) -> Commit:
        new = StoredObject(bytes(payload), bytes(trailer))
        with self._lock:
            if key in self._writing:
                return Commit.NOT_READY
            existing = self._objects.get(key)
            if existing is not None:
                return Commit.SAME if existing == new else Commit.DIFFERENT
            if self.capacity is not None and len(self._objects) >= self.capacity:
                return Commit.DECLINED
            self._objects[key] = new
            return Commit.STORED

    def get(self, key: str) -> StoredObject | None:
        with self._lock:
            if key in self._writing:
                raise ObjectNotReady(key)
            return self._objects.get(key)

    def presence(self, key: str) -> ProbeStatus:
        with self._lock:
            if self._presence_failure is not None:
                reason, self._presence_failure = self._presence_failure, None
                raise StoreError(reason)
            if key in self._writing:
                return ProbeStatus.NOT_READY
            return ProbeStatus.HIT if key in self._objects else ProbeStatus.MISS

    def contains(self, key: str) -> bool:
        return self.presence(key) is ProbeStatus.HIT

    def committed_object(self, key: str) -> StoredObject:
        """The object under ``key``; ``KeyError`` when nothing is committed there."""
        stored = self.get(key)
        if stored is None:
            raise KeyError(f"no committed object under {key!r}")
        return stored

    def committed_payload(self, key: str) -> bytes:
        return self.committed_object(key).payload

    def __len__(self) -> int:
        with self._lock:
            return len(self._objects)

    # Test controls.

    def begin_write(self, key: str) -> None:
        """Hold ``key`` as another writer's unfinished object until ``end_write``."""
        with self._lock:
            self._writing.add(key)

    def end_write(self, key: str) -> None:
        with self._lock:
            self._writing.discard(key)

    def fail_next_presence(self, reason: str) -> None:
        """Make the next presence query raise ``StoreError(reason)``."""
        with self._lock:
            self._presence_failure = reason

    def corrupt(self, key: str, index: int) -> None:
        """Flip one payload byte of the committed object, as a damaged store would."""
        with self._lock:
            stored = self._objects[key]
            payload = bytearray(stored.payload)
            payload[index] ^= 0xFF
            self._objects[key] = StoredObject(bytes(payload), stored.trailer)

    def overwrite(self, key: str, payload: bytes, trailer: bytes) -> None:
        """Replace whatever holds ``key``. Only a deliberately broken driver calls this."""
        with self._lock:
            self._objects[key] = StoredObject(bytes(payload), bytes(trailer))


@dataclass(eq=False)
class _Registration:
    span: Span
    layout: RegionLayout
    live: bool = True


@dataclass(eq=False)
class _Row:
    io: RowIO
    published: RowResult = _PENDING
    staged: RowResult | None = None  # decided but not yet visible through poll
    started: bool = False  # a transfer took the row; cancel and fail leave it alone
    transferred: bool = False


@dataclass(eq=False)
class MemoryOp:
    """One put or get. Tests hold it to drive the controlled mode."""

    kind: Kind
    rows: list[_Row]
    hint: Hint = None
    on_change: OnChange | None = None
    access_ended: bool = False
    unprovable: bool = False


@dataclass(eq=False)
class _Probe:
    worker: threading.Thread
    finished: threading.Event


@dataclass(eq=False)
class _Worker:
    thread: threading.Thread
    finished: threading.Event


def _gather(spans: Sequence[Span]) -> bytes:
    return b"".join(ctypes.string_at(span.address, span.nbytes) for span in spans)


def _scatter(spans: Sequence[Span], payload: bytes) -> None:
    offset = 0
    for span in spans:
        ctypes.memmove(span.address, payload[offset : offset + span.nbytes], span.nbytes)
        offset += span.nbytes


class MemoryDriver:
    """A ``StoreDriver`` over a ``MemoryStore``; see the module docstring for its three modes."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        controlled: bool = False,
        background: bool = False,
        capabilities: DriverCapabilities = DEFAULT_CAPABILITIES,
    ) -> None:
        if controlled and background:
            raise ValueError("a driver is controlled or runs in the background, not both")
        self.store = store
        self.controlled = controlled
        self.background = background  # put/get return once recorded; a thread executes them
        self._workers: list[_Worker] = []
        self._capabilities = capabilities
        self._lock = threading.Condition(threading.RLock())
        self._registrations: list[_Registration] = []
        self._ops: list[MemoryOp] = []
        self._probes: list[_Probe] = []
        self._probe_workers: list[threading.Thread] = []  # every worker ever started
        self._draining = False
        self._closed = False
        self._cancel_effective = True
        self._probe_unknown = False
        self._register_failure: tuple[str, bool] | None = None
        self._unregister_failures: dict[int, str] = {}
        self.hints_seen: list[Hint] = []  # every hint a get or probe was given, in order

    # The StoreDriver protocol.

    @property
    def capabilities(self) -> DriverCapabilities:
        return self._capabilities

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> _Registration:
        with self._lock:
            registration = _Registration(self._validate_region(address, nbytes, layout, ()), layout)
            if self._register_failure is not None:
                reason, escaped = self._register_failure
                self._register_failure = None
                if escaped:
                    # The store may have registered before it answered: hand out the handle.
                    self._registrations.append(registration)
                    raise RegistrationError(reason, registration)
                raise RegistrationError(reason)
            self._registrations.append(registration)
            return registration

    def unregister(self, handle: RegionHandle) -> None:
        with self._lock:
            registration = self._registration(handle)
            if not registration.live:
                return
            self._check_region_idle(registration)
            reason = self._unregister_failures.pop(id(registration), None)
            if reason is not None:
                raise StoreError(reason)
            registration.live = False

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> MemoryOp:
        return self._submit("put", rows, on_change=on_change)

    def get(
        self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None
    ) -> MemoryOp:
        return self._submit("get", rows, hint, on_change)

    def poll(self, op: OpHandle) -> tuple[RowResult, ...]:
        with self._lock:
            return tuple(row.published for row in self._op(op).rows)

    def access_ended(self, op: OpHandle) -> bool:
        with self._lock:
            return self._op(op).access_ended

    def access_unprovable(self, op: OpHandle) -> bool:
        with self._lock:
            return self._op(op).unprovable

    def cancel(self, op: OpHandle) -> None:
        """Cancels the rows not started yet, and tells ``on_change`` only when that changed one."""
        with self._lock:
            op = self._op(op)
            before = ([row.published for row in op.rows], op.access_ended)
            if self._cancel_effective:
                self._cancel_unstarted(op)
            changed = ([row.published for row in op.rows], op.access_ended) != before
        if changed:
            self._call_on_change(op)

    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        with self._lock:
            self._require_not_closed()
            self._check_hint(hint)
            if hint is not None:
                self.hints_seen.append(hint)
        answers: dict[str, ProbeStatus] = {}
        errors: list[Exception] = []
        finished = threading.Event()

        def ask() -> None:
            try:
                for key in keys:
                    answers[key] = self.store.presence(key)
            except Exception as error:  # reported from the caller's thread below
                errors.append(error)
            finally:
                finished.set()
                with self._lock:
                    self._lock.notify_all()

        # The store may block; the budget bounds this call, not the store. Registration and
        # start happen under the lock, so close and drain see every worker.
        with self._lock:
            self._probes = [probe for probe in self._probes if not probe.finished.is_set()]
            if self._draining or self._probe_unknown or not self._probe_slot_free():
                return tuple(ProbeStatus.UNKNOWN for _ in keys)
            worker = threading.Thread(target=ask, name="memory-driver-probe", daemon=True)
            self._probes.append(_Probe(worker, finished))
            self._probe_workers.append(worker)
            worker.start()
        if self._wait_probe(finished, budget_s):
            worker.join()  # it is exiting; leave no thread behind a call that got its answer
        if errors:
            raise errors[0]
        return tuple(answers.get(key, ProbeStatus.UNKNOWN) for key in keys)

    def drain(self, budget_s: float) -> bool:
        with self._lock:
            self._require_not_closed()
            self._start_draining()
            ops_to_cancel = [op for op in self._ops if any(_cancellable(row) for row in op.rows)]
            for op in ops_to_cancel:
                self._cancel_unstarted(op)
        for op in ops_to_cancel:  # the operations this drain changed; outside the lock
            self._call_on_change(op)
        with self._lock:
            self._lock.wait_for(self._drain_can_return, timeout=budget_s)
            return self._quiet()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._check_closable()
            self._closed = True
            probes, self._probes = self._probes, []
            workers, self._workers = self._workers, []
        for probe in probes:  # all finished; join only lets the threads exit
            probe.worker.join()
        for worker in workers:  # all finished too; see _quiet
            worker.thread.join()

    # Controlled mode: the test releases each stage.

    def transfer(self, op: MemoryOp, rows: Iterable[int] | None = None) -> None:
        """Move the bytes of ``rows`` (all by default) and stage their results.

        The copies and store calls run outside the driver's lock; any raise from the store
        becomes that row's fault instead of escaping.
        """
        with self._lock:
            self._require_open(op)
            taken = []
            for index in self._indexes(op, rows):
                row = op.rows[index]
                if row.started or row.published.concluded:
                    raise RuntimeError(f"row {index} was already taken or decided")
                row.started = True
                taken.append(row)
        for row in taken:
            result = self._move(op.kind, row.io)
            with self._lock:
                row.staged = result
                row.transferred = True

    def publish_results(self, op: MemoryOp, rows: Iterable[int] | None = None) -> None:
        """Make the staged results of ``rows`` visible through ``poll``."""
        with self._lock:
            for index in self._indexes(op, rows):
                row = op.rows[index]
                if row.staged is None:
                    raise RuntimeError(f"row {index} has no staged result to publish")
                row.published, row.staged = row.staged, None
        self._call_on_change(op)

    def complete(self, op: MemoryOp, rows: Iterable[int] | None = None) -> None:
        """Transfer ``rows`` and publish their results, leaving access open."""
        indexes = self._indexes(op, rows)  # ``rows`` may be a one-shot iterator
        self.transfer(op, indexes)
        self.publish_results(op, indexes)

    def fail(self, op: MemoryOp, reason: str, rows: Iterable[int] | None = None) -> None:
        """Conclude ``rows`` as FAULT without moving bytes, as a store that reported an error
        would; access stays open because the store may still be writing."""
        with self._lock:
            self._require_open(op)
            for index in self._indexes(op, rows):
                row = op.rows[index]
                if row.started or row.published.concluded:
                    raise RuntimeError(f"row {index} was already taken or decided")
                row.published = faulted(reason)
        self._call_on_change(op)

    def write_late(self, op: MemoryOp, index: int, data: bytes) -> None:
        """Write into a faulted get row's spans after its result, as a transfer that kept going
        would. Refused once access has ended, and for rows that are not FAULT: those must stay
        as reported."""
        with self._lock:
            self._require_open(op)
            if op.kind != "get":
                raise RuntimeError("late writes only happen on fetch destinations")
            row = op.rows[index]
            if row.published.status is not RowStatus.FAULT:
                raise RuntimeError(f"row {index} is {row.published.status.value}, not faulted")
            if len(data) != row.io.nbytes:
                raise ValueError(f"late write of {len(data)} bytes into a {row.io.nbytes}-byte row")
            _scatter(row.io.spans, data)

    def end_access(self, op: MemoryOp) -> None:
        """Declare that the operation will never touch its rows' memory again."""
        with self._lock:
            if any(row.started and not row.transferred for row in op.rows):
                raise RuntimeError("a transfer is still in progress")
            op.access_ended = True
            self._lock.notify_all()
        self._call_on_change(op)

    def mark_unprovable(self, op: MemoryOp) -> None:
        """Declare that the driver will never learn when this operation's access ends, as after a
        store failure that issued nothing to wait on. Access stays open."""
        with self._lock:
            self._require_open(op)
            op.unprovable = True
            self._lock.notify_all()
        self._call_on_change(op)

    def set_cancel_effective(self, effective: bool) -> None:
        self._cancel_effective = effective

    def set_probe_unknown(self, unknown: bool) -> None:
        self._probe_unknown = unknown

    def fail_next_register(self, reason: str, *, escaped: bool) -> None:
        """Make the next ``register`` raise; with ``escaped`` the store keeps the region anyway."""
        self._register_failure = (reason, escaped)

    def fail_next_unregister(self, handle: RegionHandle, reason: str) -> None:
        with self._lock:
            self._unregister_failures[id(self._registration(handle))] = reason

    def live_registrations(self) -> list[_Registration]:
        with self._lock:
            return [r for r in self._registrations if r.live]

    def open_ops(self) -> list[MemoryOp]:
        with self._lock:
            return [op for op in self._ops if not op.access_ended]

    def probes_in_flight(self) -> int:
        with self._lock:
            return sum(1 for probe in self._probes if not probe.finished.is_set())

    def join_probes(self, timeout: float) -> None:
        """Wait for every probe worker ever started to exit, for tests that opened a gate they
        had closed."""
        with self._lock:
            workers = list(self._probe_workers)
        for worker in workers:
            worker.join(timeout)

    # Submission checks, one rule each.

    def _check_direction(self, kind: Kind) -> None:
        if not getattr(self._capabilities, kind):
            raise DriverRejected(f"{kind} is not declared by this driver")

    def _check_not_draining(self) -> None:
        if self._draining:
            raise DriverRejected("the driver is draining")

    def _check_hint(self, hint: Hint) -> None:
        if hint is not None and not self._capabilities.hints:
            raise DriverRejected("this driver reads no hints")

    def _check_row_limit(self, rows: Sequence[RowIO]) -> None:
        limit = self._capabilities.max_rows_per_op
        if limit is not None and len(rows) > limit:
            raise DriverRejected(f"{len(rows)} rows exceed the limit of {limit} per operation")

    def _check_span_count(self, rows: Sequence[RowIO]) -> None:
        if not self._capabilities.multi_span and any(len(io.spans) > 1 for io in rows):
            raise DriverRejected("this driver takes rows of one span")

    def _check_disjoint(self, kind: Kind, rows: Sequence[RowIO]) -> None:
        """The spans of a get never overlap across rows; the caller guarantees it, this driver checks
        it anyway to catch a backend mistake in tests."""
        if kind != "get":
            return
        spans = [span for io in rows for span in io.spans]
        for i, first in enumerate(spans):
            for second in spans[i + 1 :]:
                if first.overlaps(second):
                    raise DriverRejected(f"spans {first} and {second} overlap within one get")

    def _check_region_kind(self, layout: RegionLayout) -> None:
        if layout.memory_kind is MemoryKind.DEVICE and not self._capabilities.device_memory:
            raise DriverRejected("this driver registers host memory only")

    def _validate_region(
        self, address: int, nbytes: int, layout: RegionLayout, taken: Sequence[Span]
    ) -> Span:
        """The span this driver would register for the region, or DriverRejected. ``taken`` are
        the spans of the same batch, which count as live registrations."""
        try:
            span = Span(address, nbytes)
        except ValueError as error:
            raise DriverRejected(str(error)) from None
        if layout.nbytes > nbytes:
            raise DriverRejected(f"layout needs {layout.nbytes} bytes, region has {nbytes}")
        self._check_region_kind(layout)
        self._require_accepting()
        for other in self._registrations:
            if other.live and other.span.overlaps(span):
                raise DriverRejected(f"region {span} overlaps a live registration {other.span}")
        for other_span in taken:
            if other_span.overlaps(span):
                raise DriverRejected(f"region {span} overlaps another of the batch {other_span}")
        return span

    def _check_registered(self, rows: Sequence[RowIO]) -> None:
        for io in rows:
            for span in io.spans:
                if not self._registered(span):
                    raise DriverRejected(f"{span} is outside every live registration")

    def _check_extents(self, rows: Sequence[RowIO]) -> None:
        """A ``RowBatch`` promises each extent inside one registered region, which a driver may
        check in place of every row; rows given as a plain sequence carry no extent."""
        for extent in getattr(rows, "extents", ()):
            if not self._registered(extent):
                raise DriverRejected(f"extent {extent} is outside every live registration")

    def _check_region_idle(self, registration: _Registration) -> None:
        busy = sum(
            1
            for op in self._ops
            if not op.access_ended
            and any(span.within(registration.span) for row in op.rows for span in row.io.spans)
        )
        if busy:
            raise DriverRejected(f"{busy} operation(s) may still access the region")

    def _check_closable(self) -> None:
        if not self._draining or not self._quiet():
            raise DriverRejected("close needs a drain that returned True")
        if any(r.live for r in self._registrations):
            raise DriverRejected("close needs every registration unregistered")

    def _probe_slot_free(self) -> bool:
        return len(self._probes) < MAX_PROBES_IN_FLIGHT

    def _wait_probe(self, finished: threading.Event, budget_s: float) -> bool:
        """Waits for the probe worker up to the budget; True when it finished in time. A test
        that must not depend on scheduling overrides this to end the wait on its own signal."""
        return finished.wait(budget_s)

    def _start_draining(self) -> None:
        self._draining = True

    def _drain_can_return(self) -> bool:
        return self._quiet() or any(op.unprovable and not op.access_ended for op in self._ops)

    def _quiet(self) -> bool:
        """Nothing can still touch memory and no background work is running. An operation's
        access ending is not enough: its worker may still be on its way out."""
        return (
            all(op.access_ended for op in self._ops)
            and all(probe.finished.is_set() for probe in self._probes)
            and all(worker.finished.is_set() for worker in self._workers)
        )

    # Internals.

    def _submit(
        self,
        kind: Kind,
        rows: Sequence[RowIO],
        hint: Hint = None,
        on_change: OnChange | None = None,
    ) -> MemoryOp:
        with self._lock:
            self._require_accepting()
            self._check_direction(kind)
            self._check_not_draining()
            self._check_hint(hint)
            self._check_row_limit(rows)
            self._check_span_count(rows)
            self._check_disjoint(kind, rows)
            self._check_registered(rows)
            self._check_extents(rows)
            op = MemoryOp(kind, [_Row(io) for io in rows], hint, on_change)
            self._ops.append(op)
            if hint is not None:
                self.hints_seen.append(hint)
            if self.background:
                # Recorded, started and registered under one lock, so a drain or close that
                # follows sees a worker that exists and has started.
                finished = threading.Event()
                thread = threading.Thread(
                    target=self._run, args=(op, finished), name="memory-driver-op", daemon=True
                )
                try:
                    thread.start()
                except RuntimeError as error:
                    # The thread never ran, so no row started: the recorded operation concludes
                    # as faults and, having touched nothing, has ended its access.
                    self._fail_unstarted(op, f"no worker thread: {error}")
                else:
                    self._workers.append(_Worker(thread, finished))
        if not self.background and not self.controlled:
            self._run(op, None)
        elif op.access_ended:  # no worker thread: it concluded at once
            self._call_on_change(op)
        return op

    def _run(self, op: MemoryOp, finished: threading.Event | None) -> None:
        try:
            self._before_execute(op)
            self._execute(op)
            # Told before the worker counts as finished: a drain that returned True earlier would
            # let the owner close the driver while the callback still queries it.
            self._call_on_change(op)
        finally:
            if finished is not None:
                finished.set()
                with self._lock:
                    self._lock.notify_all()

    def _before_execute(self, op: MemoryOp) -> None:
        """Between recording an operation and executing it; a test races a drain here."""

    def _execute(self, op: MemoryOp) -> None:
        """Automatic mode: like a blocking SDK, the call returns once every row is decided and
        access is over. Rows a concurrent drain cancelled in between stay as they are."""
        with self._lock:
            taken = [row for row in op.rows if not row.started and not row.published.concluded]
            for row in taken:
                row.started = True
        for row in taken:
            result = self._move(op.kind, row.io)
            with self._lock:
                row.published = result
                row.transferred = True
        with self._lock:
            if not op.access_ended:
                op.access_ended = True
                self._lock.notify_all()

    def _move(self, kind: Kind, io: RowIO) -> RowResult:
        # The operation is recorded by now: nothing may escape as an exception any more.
        try:
            return self._put_row(io) if kind == "put" else self._get_row(io)
        except StoreError as error:
            return faulted(str(error))
        except Exception as error:
            return faulted(f"{type(error).__name__}: {error}")

    def _put_row(self, io: RowIO) -> RowResult:
        payload = _gather(io.spans)
        trailer = self._trailer_for(io, payload)
        outcome = self.store.commit(io.key, payload, trailer)
        if outcome in (Commit.STORED, Commit.SAME):
            return _READY
        if outcome is Commit.DECLINED:
            return _DECLINED
        if outcome is Commit.NOT_READY:
            return _NOT_READY
        # A diagnostic this store can afford; the contract asks no driver to compare bytes.
        return faulted("key exists with different content")

    def _get_row(self, io: RowIO) -> RowResult:
        try:
            stored = self.store.get(io.key)
        except ObjectNotReady:
            return _NOT_READY
        if stored is None:
            return _MISS
        if len(stored.payload) != io.nbytes:
            return faulted(f"object is {len(stored.payload)} bytes, expected {io.nbytes}")
        problem = self._verify(io, stored)
        if problem is not None:
            return faulted(problem)
        _scatter(io.spans, stored.payload)
        return _READY

    def _trailer_for(self, io: RowIO, payload: bytes) -> bytes:
        return encode_trailer(io.key.encode(), io.nbytes, checksum=Checksum.CRC32, payload=payload)

    def _verify(self, io: RowIO, stored: StoredObject) -> str | None:
        """Why the stored object may not be delivered as ``io``, or ``None``."""
        return check_trailer(stored.trailer, io.key.encode(), io.nbytes, stored.payload)

    def _call_on_change(self, op: MemoryOp) -> None:
        """Outside the lock, as ``OnChange`` asks."""
        if op.on_change is not None:
            op.on_change(op)

    def _fail_unstarted(self, op: MemoryOp, reason: str) -> None:
        """Every row FAULT and access over, for an operation none of whose rows started."""
        for row in op.rows:
            row.published = faulted(reason)
        op.access_ended = True
        self._lock.notify_all()

    def _cancel_unstarted(self, op: MemoryOp) -> None:
        for row in op.rows:
            if _cancellable(row):
                row.published = _CANCELLED
        self._settle_untouched(op)

    def _settle_untouched(self, op: MemoryOp) -> None:
        """An operation whose rows were all cancelled before any started never touched memory,
        which is access-end evidence in itself."""
        if not op.access_ended and all(
            row.published.status is RowStatus.CANCELLED and not row.started for row in op.rows
        ):
            op.access_ended = True
            self._lock.notify_all()

    def _registered(self, span: Span) -> bool:
        return any(r.live and span.within(r.span) for r in self._registrations)

    def _registration(self, handle: RegionHandle) -> _Registration:
        self._require_not_closed()
        if not isinstance(handle, _Registration) or handle not in self._registrations:
            raise ValueError("handle does not belong to this driver")
        return handle

    def _op(self, op: OpHandle) -> MemoryOp:
        self._require_not_closed()
        if not isinstance(op, MemoryOp) or op not in self._ops:
            raise ValueError("operation does not belong to this driver")
        return op

    def _require_accepting(self) -> None:
        if self._closed:
            raise DriverRejected("the driver is closed")

    def _require_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("the driver is closed")

    def _require_open(self, op: MemoryOp) -> None:
        if self._op(op).access_ended:
            raise RuntimeError("the operation already ended its access")

    @staticmethod
    def _indexes(op: MemoryOp, rows: Iterable[int] | None) -> list[int]:
        return list(range(len(op.rows))) if rows is None else list(rows)


def _cancellable(row: _Row) -> bool:
    """Neither taken by a transfer nor decided: a cancel or a drain may still cancel it."""
    return not row.started and not row.published.concluded


@dataclass(eq=False)
class _Batch:
    """The handle of one ``register_regions`` call: its registrations, revoked together."""

    registrations: list[_Registration]
    live: bool = True


class TogetherMemoryDriver(MemoryDriver):
    """The in-memory driver as a ``RegistersRegionsTogether``: regions come and go in batches,
    one handle per ``register_regions`` call, and the single-region ``register`` and
    ``unregister`` are rejected. Everything else is the in-memory driver's, controls included."""

    def __init__(self, store: MemoryStore, **options: Any) -> None:
        super().__init__(store, **options)
        self._batches: list[_Batch] = []
        self._batch_failure: tuple[str, bool] | None = None
        self._revocation_failure: str | None = None

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> NoReturn:
        raise DriverRejected("this driver registers regions together; use register_regions")

    def unregister(self, handle: RegionHandle) -> NoReturn:
        raise DriverRejected("this driver revokes regions together; use unregister_regions")

    def register_regions(self, specs: Sequence[RegionSpec]) -> _Batch:
        if not specs:
            raise DriverRejected("a batch registers at least one region")
        with self._lock:
            spans: list[Span] = []
            for spec in specs:  # all checked before any is kept: a rejection escapes nothing
                spans.append(self._validate_region(spec.address, spec.nbytes, spec.layout, spans))
            batch = _Batch(
                [_Registration(span, spec.layout) for span, spec in zip(spans, specs, strict=True)]
            )
            if self._batch_failure is not None:
                reason, escaped = self._batch_failure
                self._batch_failure = None
                if not escaped:
                    raise RegistrationError(reason)
                self._keep(batch)  # the store kept the batch before it answered
                raise RegistrationError(reason, batch)
            self._keep(batch)
            return batch

    def unregister_regions(self, handle: RegionHandle) -> None:
        with self._lock:
            batch = self._batch(handle)
            if not batch.live:
                return
            for registration in batch.registrations:
                self._check_region_idle(registration)
            if self._revocation_failure is not None:
                reason, self._revocation_failure = self._revocation_failure, None
                raise StoreError(reason)
            for registration in batch.registrations:
                registration.live = False
            batch.live = False

    # Controls.

    def fail_next_register_regions(self, reason: str, *, escaped: bool) -> None:
        """Make the next ``register_regions`` raise; with ``escaped`` the store keeps the batch
        anyway and the error carries its handle."""
        self._batch_failure = (reason, escaped)

    def fail_next_unregister_regions(self, reason: str) -> None:
        """Make the next ``unregister_regions`` of a live batch raise, the batch staying live."""
        self._revocation_failure = reason

    def live_batches(self) -> list[_Batch]:
        with self._lock:
            return [batch for batch in self._batches if batch.live]

    # Internals.

    def _keep(self, batch: _Batch) -> None:
        self._registrations.extend(batch.registrations)
        self._batches.append(batch)

    def _batch(self, handle: RegionHandle) -> _Batch:
        self._require_not_closed()
        if not isinstance(handle, _Batch) or handle not in self._batches:
            raise ValueError("handle does not belong to this driver")
        return handle
