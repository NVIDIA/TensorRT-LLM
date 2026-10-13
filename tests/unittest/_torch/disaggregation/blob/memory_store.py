# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An in-process store and a driver over it, for tests.

``MemoryStore`` is a dictionary of complete objects that two endpoints in one process can share.
``MemoryDriver`` really copies bytes between registered host memory and that store. In automatic
mode an operation completes inside ``put``/``get``, like a blocking SDK. In controlled mode the test
releases three things separately per operation: the transfer, the logical results and the end of
access, so it can stage partial arrival, a result before or after access end, a late write after a
fault, and registration failures. No threads are started here. Transfers copy and call the store
outside the driver's lock, so queries never wait on the store; the lock is held around state
changes and around a late write's short copy, which keeps ``end_access`` out until it lands.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from tensorrt_llm._torch.disaggregation.blob.format import TRAILER_BYTES
from tensorrt_llm._torch.disaggregation.blob.store import (
    CANCELLED,
    MISS,
    PENDING,
    READY,
    DriverCapabilities,
    OpHandle,
    ProbeStatus,
    RegionHandle,
    RegionLayout,
    RegistrationError,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreError,
    faulted,
)


class MemoryStore:
    """Objects by key. Each object is a private copy, committed whole or not at all.

    Subclass and override ``commit`` or ``get`` to make a test's store block or fail; both may
    raise ``StoreError``, which the driver turns into a row fault.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._objects: dict[str, bytes] = {}

    def commit(self, key: str, data: bytes) -> bool:
        """Store ``data`` under ``key``. False when a different object already holds the key;
        the existing object is kept. Storing identical bytes again is a no-op success."""
        copy = bytes(data)
        with self._lock:
            existing = self._objects.get(key)
            if existing is None:
                self._objects[key] = copy
                return True
            return existing == copy

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._objects.get(key)

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def committed_object(self, key: str) -> bytes:
        """The whole object under ``key``; ``KeyError`` when nothing is committed there."""
        data = self.get(key)
        if data is None:
            raise KeyError(f"no committed object under {key!r}")
        return data

    def committed_payload(self, key: str, trailer_nbytes: int = TRAILER_BYTES) -> bytes:
        """The payload of the object under ``key``, without its trailer."""
        return self.committed_object(key)[:-trailer_nbytes]

    def __len__(self) -> int:
        with self._lock:
            return len(self._objects)


@dataclass(eq=False)
class _Registration:
    span: Span
    layout: RegionLayout
    live: bool = True


@dataclass(eq=False)
class _Row:
    io: RowIO
    published: RowResult = PENDING
    staged: RowResult | None = None  # decided but not yet visible through poll
    started: bool = False  # a transfer took the row; cancel and fail leave it alone
    transferred: bool = False


@dataclass(eq=False)
class MemoryOp:
    """One put or get. Tests hold it to drive the controlled mode."""

    kind: Literal["put", "get"]
    rows: list[_Row]
    access_ended: bool = False


class MemoryDriver:
    """A ``StoreDriver`` over a ``MemoryStore``; see the module docstring for the two modes."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        controlled: bool = False,
        capabilities: DriverCapabilities = DriverCapabilities(cancel=True, probe=True),
    ) -> None:
        self.store = store
        self.controlled = controlled
        self._capabilities = capabilities
        self._lock = threading.Condition(threading.RLock())
        self._registrations: list[_Registration] = []
        self._ops: list[MemoryOp] = []
        self._draining = False
        self._cancel_effective = True
        self._probe_unknown = False
        self._register_failure: tuple[str, bool] | None = None
        self._unregister_failures: dict[int, str] = {}

    # The StoreDriver protocol.

    @property
    def capabilities(self) -> DriverCapabilities:
        return self._capabilities

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> _Registration:
        span = Span(address, nbytes)
        if layout.nbytes > nbytes:
            raise ValueError(f"layout needs {layout.nbytes} bytes, region has {nbytes}")
        with self._lock:
            for other in self._registrations:
                if other.live and other.span.overlaps(span):
                    raise ValueError(f"region {span} overlaps a live registration {other.span}")
            registration = _Registration(span, layout)
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
            reason = self._unregister_failures.pop(id(registration), None)
            if reason is not None:
                raise StoreError(reason)
            registration.live = False

    def put(self, rows: Sequence[RowIO]) -> MemoryOp:
        return self._submit("put", rows)

    def get(self, rows: Sequence[RowIO]) -> MemoryOp:
        return self._submit("get", rows)

    def poll(self, op: OpHandle) -> tuple[RowResult, ...]:
        with self._lock:
            return tuple(row.published for row in self._op(op).rows)

    def access_ended(self, op: OpHandle) -> bool:
        with self._lock:
            return self._op(op).access_ended

    def cancel(self, op: OpHandle) -> None:
        with self._lock:
            if self._cancel_effective:
                self._cancel_unstarted(self._op(op))

    def probe(self, keys: Sequence[str], budget_s: float) -> tuple[ProbeStatus, ...]:
        if self._probe_unknown:
            return tuple(ProbeStatus.UNKNOWN for _ in keys)
        return tuple(
            ProbeStatus.HIT if self.store.contains(key) else ProbeStatus.MISS for key in keys
        )

    def drain(self, budget_s: float) -> bool:
        with self._lock:
            self._draining = True
            for op in self._ops:
                self._cancel_unstarted(op)
            return self._lock.wait_for(
                lambda: all(op.access_ended for op in self._ops), timeout=budget_s
            )

    # Controlled mode: the test releases each stage.

    def transfer(self, op: MemoryOp, rows: Iterable[int] | None = None) -> None:
        """Move the bytes of ``rows`` (all by default) and stage their results.

        The copies and store calls run outside the driver's lock; a ``StoreError`` from the store
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

    def complete(self, op: MemoryOp, rows: Iterable[int] | None = None) -> None:
        """Transfer ``rows`` and publish their results, leaving access open."""
        indexes = self._indexes(op, rows)  # ``rows`` may be a one-shot iterator
        self.transfer(op, indexes)
        self.publish_results(op, indexes)

    def fail(self, op: MemoryOp, reason: str, rows: Iterable[int] | None = None) -> None:
        """Conclude ``rows`` as FAULT without moving bytes; access stays open."""
        with self._lock:
            self._require_open(op)
            for index in self._indexes(op, rows):
                row = op.rows[index]
                if row.started or row.published.concluded:
                    raise RuntimeError(f"row {index} was already taken or decided")
                row.published = faulted(reason)

    def write_late(self, op: MemoryOp, index: int, data: bytes) -> None:
        """Write into a faulted get row's payload after its result, as a transfer that kept going
        would. Refused once access has ended, and for rows that are READY, MISS, CANCELLED or
        still pending: those must stay as reported."""
        with self._lock:
            self._require_open(op)
            if op.kind != "get":
                raise RuntimeError("late writes only happen on fetch destinations")
            row = op.rows[index]
            if row.published.status is not RowStatus.FAULT:
                raise RuntimeError(f"row {index} is {row.published.status.value}, not faulted")
            payload = row.io.payload
            if len(data) != payload.nbytes:
                raise ValueError(
                    f"late write of {len(data)} bytes into a {payload.nbytes}-byte row"
                )
            ctypes.memmove(payload.address, data, len(data))

    def end_access(self, op: MemoryOp) -> None:
        """Declare that the operation will never touch its rows' memory again."""
        with self._lock:
            if any(row.started and not row.transferred for row in op.rows):
                raise RuntimeError("a transfer is still in progress")
            op.access_ended = True
            self._lock.notify_all()

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

    # Internals.

    def _submit(self, kind: Literal["put", "get"], rows: Sequence[RowIO]) -> MemoryOp:
        if not rows:
            raise ValueError("an operation needs at least one row")
        limit = self._capabilities.max_rows_per_op
        if limit is not None and len(rows) > limit:
            raise ValueError(f"{len(rows)} rows exceed the limit of {limit} per operation")
        with self._lock:
            if self._draining:
                raise ValueError("the driver is draining")
            for io in rows:
                for span in (io.payload, io.trailer):
                    if not self._registered(span):
                        raise ValueError(f"{span} is outside every live registration")
            op = MemoryOp(kind, [_Row(io) for io in rows])
            self._ops.append(op)
        if not self.controlled:
            # Like a blocking SDK: the call returns once every row is decided and access is over.
            self.transfer(op)
            self.publish_results(op)
            self.end_access(op)
        return op

    def _move(self, kind: str, io: RowIO) -> RowResult:
        try:
            return self._put_row(io) if kind == "put" else self._get_row(io)
        except StoreError as error:
            return faulted(str(error))

    def _put_row(self, io: RowIO) -> RowResult:
        data = ctypes.string_at(io.payload.address, io.payload.nbytes)
        data += ctypes.string_at(io.trailer.address, io.trailer.nbytes)
        if self.store.commit(io.key, data):
            return READY
        return faulted("key exists with different content")

    def _get_row(self, io: RowIO) -> RowResult:
        data = self.store.get(io.key)
        if data is None:
            return MISS
        expected = io.payload.nbytes + io.trailer.nbytes
        if len(data) != expected:
            return faulted(f"object is {len(data)} bytes, expected {expected}")
        ctypes.memmove(io.payload.address, data, io.payload.nbytes)
        ctypes.memmove(io.trailer.address, data[io.payload.nbytes :], io.trailer.nbytes)
        return READY

    def _cancel_unstarted(self, op: MemoryOp) -> None:
        for row in op.rows:
            if not row.started and not row.published.concluded:
                row.published = CANCELLED

    def _registered(self, span: Span) -> bool:
        return any(r.live and span.within(r.span) for r in self._registrations)

    def _registration(self, handle: RegionHandle) -> _Registration:
        if not isinstance(handle, _Registration) or handle not in self._registrations:
            raise ValueError("handle does not belong to this driver")
        return handle

    def _op(self, op: OpHandle) -> MemoryOp:
        if not isinstance(op, MemoryOp) or op not in self._ops:
            raise ValueError("operation does not belong to this driver")
        return op

    def _require_open(self, op: MemoryOp) -> None:
        if self._op(op).access_ended:
            raise RuntimeError("the operation already ended its access")

    @staticmethod
    def _indexes(op: MemoryOp, rows: Iterable[int] | None) -> list[int]:
        return list(range(len(op.rows))) if rows is None else list(rows)
