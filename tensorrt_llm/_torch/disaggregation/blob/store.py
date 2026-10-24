# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The private boundary between the blob backend and a byte store.

A driver moves whole rows between registered memory and one store. It answers two questions
separately: what each row's logical result is (:meth:`StoreDriver.poll`), and whether an operation
can still touch the caller's memory (:meth:`StoreDriver.access_ended`). Neither answer implies the
other. Nothing here belongs to the public backend contract in ``base/``, from which only
``MemoryKind`` is shared; the backend translates between the two.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, overload, runtime_checkable

import numpy as np

from ..base.memory import MemoryKind

Hint = Any
"""The caller's hint for where a get or probe should look, handed to the driver unchanged. A hint
is an immutable snapshot: the driver may keep a reference to it for as long as it needs and never
modifies it. The backend passes a hint only to a driver that declares ``hints``; for any other
driver it rejects the submission that carries one, before dispatch, so such a driver is never
called with a hint."""


class DriverRejected(Exception):
    """Raised by a driver on purpose, before it touched the store or the caller's memory.

    The only exception that certifies nothing escaped: the caller may forget the call. It is not a
    :class:`StoreError`, so code that handles store trouble never swallows a rejection. Any other
    raise from ``register``, ``put`` or ``get`` may have escaped, so the caller keeps the memory
    protected; the blob backend, which cannot tell what escaped, holds it for good.
    """


class StoreError(Exception):
    """A store-side failure the driver could not express as a row result."""


class RegistrationError(StoreError):
    """Registering a region failed after the store was asked.

    Args:
        reason: What went wrong.
        handle: The handle to ``unregister`` later when the store may have registered the region
            anyway, or ``None`` when the store certainly did not.
    """

    def __init__(self, reason: str, handle: RegionHandle | None = None) -> None:
        super().__init__(reason)
        self.handle = handle


class RowStatus(enum.Enum):
    """The logical state of one row within an operation.

    READY and FAULT are the only two that carry a verdict. MISS, NOT_READY, DECLINED and CANCELLED
    say that nothing was delivered and nothing went wrong: the backend reports such a row as not
    delivered without a fault, and none of them counts toward isolation.
    """

    PENDING = "pending"
    """Not concluded yet."""

    READY = "ready"
    """For a get: the payload landed whole and intact, and the driver will not write the row again.
    For a put: a complete object is readable under the key by any reader now, whoever wrote it. A
    store's "already exists" counts only once that object is committed."""

    MISS = "miss"
    """For a get: the store reported that it holds no object under the key, and the row's memory
    was not touched. Every driver answers MISS only for a reported absence and FAULT for any
    failure; a driver that declares ``absent_is_miss`` adds that an absent key always concludes
    MISS."""

    NOT_READY = "not_ready"
    """The key is held by a write that has not finished: a get delivered nothing, a put published
    nothing of its own. ``probe`` gives the same answer for such a key."""

    DECLINED = "declined"
    """For a put: the store would not take the row, for space or policy. Nothing was published;
    the other rows of the operation go on."""

    CANCELLED = "cancelled"
    """Never started, because of a cancel or a drain; no byte moved and the row's memory was not
    touched. A row that had started never concludes CANCELLED."""

    FAULT = "fault"
    """Anything else: timeout, size or integrity mismatch, transport or store error. Says nothing
    about the destination's contents."""


@dataclass(frozen=True)
class RowResult:
    """One row's logical result. ``fault`` is set exactly when ``status`` is ``FAULT``."""

    status: RowStatus
    fault: str | None = None

    def __post_init__(self) -> None:
        if (self.status is RowStatus.FAULT) != (self.fault is not None):
            raise ValueError("a fault text is given exactly for FAULT results")

    @property
    def concluded(self) -> bool:
        return self.status is not RowStatus.PENDING


def faulted(reason: str) -> RowResult:
    """A FAULT result carrying ``reason``."""
    return RowResult(RowStatus.FAULT, reason)


def concluded(results: Sequence[RowResult]) -> bool:
    """Whether no row of ``results`` is pending."""
    return all(result.concluded for result in results)


@dataclass(frozen=True)
class Span:
    """A non-empty range ``[address, address + nbytes)`` inside a registered region.

    Whether that region is host or device memory is the region's business, not the span's. A span
    need not be a whole slot.
    """

    address: int
    nbytes: int

    def __post_init__(self) -> None:
        if self.address < 0 or self.nbytes <= 0:
            raise ValueError(f"bad span: address={self.address}, nbytes={self.nbytes}")

    @property
    def end(self) -> int:
        return self.address + self.nbytes

    def overlaps(self, other: Span) -> bool:
        return self.address < other.end and other.address < self.end

    def within(self, other: Span) -> bool:
        return other.address <= self.address and self.end <= other.end


@dataclass(frozen=True)
class RegionLayout:
    """How a registered region is divided: ``slot_count`` slots of ``slot_bytes``, back to back.

    ``label`` is an opaque layout label, equal for regions laid out alike; it is not an identity.
    ``memory_kind`` says whether the region is host or device memory; a transport registers the
    two differently.
    """

    slot_bytes: int
    slot_count: int
    label: str = ""
    memory_kind: MemoryKind = MemoryKind.HOST

    def __post_init__(self) -> None:
        if self.slot_bytes <= 0 or self.slot_count <= 0:
            raise ValueError(
                f"bad layout: slot_bytes={self.slot_bytes}, slot_count={self.slot_count}"
            )

    @property
    def nbytes(self) -> int:
        return self.slot_bytes * self.slot_count


@dataclass(frozen=True)
class RowIO:
    """One row to move: its store key and the spans its payload occupies.

    The stored object is the spans' bytes back to back, in the order given; the key does not
    depend on how a row is split. A row staged on the host has one span. A row lent in place has
    one span per pool of its pool group, in pool order, and may cover only part of a slot when the
    two sides of a transfer are sharded differently. Spans of one row never overlap. A row of more
    than one span reaches only a driver that declares ``multi_span``.
    """

    key: str
    spans: tuple[Span, ...]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("a row needs a store key")
        spans = tuple(self.spans)
        object.__setattr__(self, "spans", spans)
        if not spans:
            raise ValueError(f"row {self.key} has no spans")
        for i, first in enumerate(spans):
            for second in spans[i + 1 :]:
                if first.overlaps(second):
                    raise ValueError(f"row {self.key} has overlapping spans {first} and {second}")

    @property
    def nbytes(self) -> int:
        """The object's size: the spans' bytes added up."""
        return sum(span.nbytes for span in self.spans)


class RowBatch(Sequence[RowIO]):
    """The rows of one ``put`` or ``get``, laid out alike and held as columns.

    Row ``i`` is stored under ``keys()[i]``; its payload is its spans
    ``(addresses[i, j], span_bytes[j])`` back to back, in order. ``addresses`` is
    ``int64 [rows, spans_per_row]``, read only. Every row has the same span sizes, so every row
    holds ``row_bytes``. Span ``j`` of every row lies in ``extents[j]``, and each extent lies in one
    region its maker registered with the driver, so a driver can check a batch's memory once per
    extent instead of once per row.

    A batch keeps arrays and builds nothing per row until a driver asks, so handing one over costs
    the same for any number of rows. A driver whose store calls take lists builds them on the
    thread that makes those calls (``keys()``, ``addresses[:, 0].tolist()``); a driver that works
    row by row reads the batch as a ``Sequence[RowIO]``, one ``RowIO`` per row read. Slicing a
    batch gives the batch of those rows, again without building anything per row.

    Its maker guarantees what a driver would otherwise check row by row: at least one row, no key
    twice, spans of positive size inside their extent that do not overlap within a row, and each
    extent inside one registered region. :meth:`of` checks all but the last, which only the driver
    can; :meth:`of_slots`, for a maker that checked its rows already, checks nothing.
    """

    def __init__(
        self,
        keys: Sequence[str],
        addresses: np.ndarray,
        span_bytes: tuple[int, ...],
        extents: tuple[Span, ...],
    ) -> None:
        self._keys = keys
        self.addresses = addresses
        self.span_bytes = span_bytes
        self.extents = extents

    @classmethod
    def of_slots(
        cls,
        key_stem: str,
        object_keys: np.ndarray,
        addresses: np.ndarray,
        slot_bytes: int,
        extent: Span,
    ) -> RowBatch:
        """Rows that are each one whole slot: row ``i`` is keyed ``key_stem`` followed by
        ``object_keys[i]`` in hex, and its payload is the ``slot_bytes`` at ``addresses[i]``.
        Checks nothing and copies nothing.

        Args:
            key_stem: How every key starts (``keys.key_stem``).
            object_keys: ``uint8 [rows, width > 0]``, no row twice.
            addresses: ``int64 [rows]``, each the start of a slot inside ``extent``.
            slot_bytes: Every row's size.
            extent: The registered region holding every slot, or a span inside it.
        """
        column = addresses[:, None]  # a view of the caller's array, which stays writable
        column.flags.writeable = False
        return cls(_HexKeys(key_stem, object_keys), column, (slot_bytes,), (extent,))

    @classmethod
    def of(cls, rows: Sequence[RowIO]) -> RowBatch:
        """The batch of ``rows``, built row by row and checked. Span ``j``'s extent is the
        smallest span holding every row's span ``j``.

        Raises:
            ValueError: No rows, a key twice, or rows not laid out alike.
        """
        rows = list(rows)
        if not rows:
            raise ValueError("a batch needs at least one row")
        keys = tuple(row.key for row in rows)
        if len(set(keys)) != len(keys):
            raise ValueError("a key appears twice in one batch")
        span_bytes = tuple(span.nbytes for span in rows[0].spans)
        if any(tuple(span.nbytes for span in row.spans) != span_bytes for row in rows):
            raise ValueError("the rows of a batch have the same span sizes, in the same order")
        addresses = np.array([[s.address for s in row.spans] for row in rows], dtype=np.int64)
        addresses.flags.writeable = False
        extents = tuple(
            Span(int(column.min()), int(column.max()) + nbytes - int(column.min()))
            for column, nbytes in zip(addresses.T, span_bytes)
        )
        return cls(keys, addresses, span_bytes, extents)

    @property
    def spans_per_row(self) -> int:
        return len(self.span_bytes)

    @property
    def row_bytes(self) -> int:
        """Every row's size: its spans' bytes added up."""
        return sum(self.span_bytes)

    def keys(self) -> list[str]:
        """Every row's store key, in order."""
        return list(self._keys)

    def __len__(self) -> int:
        return len(self.addresses)

    @overload
    def __getitem__(self, index: int) -> RowIO: ...

    @overload
    def __getitem__(self, index: slice) -> RowBatch: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("a RowBatch slice is contiguous")
            rows = slice(start, stop)
            return RowBatch(self._keys[rows], self.addresses[rows], self.span_bytes, self.extents)
        spans = zip(self.addresses[index].tolist(), self.span_bytes)
        return RowIO(self._keys[index], tuple(Span(address, nbytes) for address, nbytes in spans))


@dataclass(frozen=True)
class _HexKeys(Sequence[str]):
    """Store keys as ``stem`` followed by each row of ``object_keys`` in hex, built only when
    read; iterating builds every key with one hex conversion."""

    stem: str
    object_keys: np.ndarray  # uint8 [rows, width]

    def __len__(self) -> int:
        return len(self.object_keys)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _HexKeys(self.stem, self.object_keys[index])
        return self.stem + self.object_keys[index].tobytes().hex()

    def __iter__(self) -> Iterator[str]:
        stem, width = self.stem, 2 * self.object_keys.shape[1]
        hexed = self.object_keys.tobytes().hex()
        return iter([stem + hexed[i : i + width] for i in range(0, len(hexed), width)])


class RegionHandle(Protocol):
    """Opaque handle of one registration; only the driver that issued it reads it."""


class OpHandle(Protocol):
    """Opaque handle of one put or get; only the driver that issued it reads it."""


OnChange = Callable[[OpHandle], None]
"""What a caller of ``put`` or ``get`` hands the driver to hear of the operation.

The driver calls it with the operation's handle once every row concluded, once its access ended
and once its access turned unprovable, each time after the change shows through ``poll``,
``access_ended`` and ``access_unprovable`` (one call may tell of several). The driver may also
call it at other times, such as after some rows concluded, from any thread, outside its locks,
possibly before the ``put`` or ``get`` that issued the operation returned. The callback may call
the driver's non-blocking queries and ``cancel``; it returns promptly and never raises. A caller
that passes None polls instead."""


class ProbeStatus(enum.Enum):
    """What a probe knows about one key, without moving payload."""

    HIT = "hit"
    MISS = "miss"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DriverCapabilities:
    """What a driver implements. Fixed for the driver's life.

    A field whose default would grant something, such as an unlimited row count, is declared
    explicitly. A capability added later defaults to not supported, so existing drivers need no
    change to stay honest.

    Args:
        get: ``get`` is implemented.
        put: ``put`` is implemented.
        cancel: ``cancel`` does something. Without it the backend never calls ``cancel`` and
            declares no cancel of its own: the calls of a stopped submission run to their end.
        probe: ``probe`` answers. The blob backend never probes, so it fetches without asking
            first either way.
        max_rows_per_op: Rows one ``put``/``get`` accepts, or ``None`` for no limit.
        hints: ``get`` and ``probe`` read a caller's hint; without it the backend rejects a
            submission carrying a hint before dispatch and the driver is called without one.
        absent_is_miss: A get of a key the store does not hold concludes MISS, unless an earlier
            read of that row in the same operation already wrote its memory (a reread). Without
            it the store may not tell absence from failure, and such a get may conclude FAULT.
        device_memory: The driver registers regions whose ``memory_kind`` is DEVICE; without it
            every region handed to it is host memory and it rejects a DEVICE one.
        multi_span: ``put`` and ``get`` take rows of more than one span; without it every row
            handed to it has exactly one.
    """

    get: bool
    put: bool
    cancel: bool
    probe: bool
    max_rows_per_op: int | None
    hints: bool = False
    absent_is_miss: bool = False
    device_memory: bool = False
    multi_span: bool = False

    def __post_init__(self) -> None:
        if not (self.get or self.put):
            raise ValueError("a driver implements at least one of get and put")
        if self.max_rows_per_op is not None and self.max_rows_per_op <= 0:
            raise ValueError("max_rows_per_op is positive or None")


@runtime_checkable
class StoreDriver(Protocol):
    """What the backend asks of a store.

    Threads: ``register``, ``unregister``, ``drain`` and ``close`` are called one at a time.
    ``put`` and ``get`` are called on the executor loop's thread, so they only record the
    operation and queue its work: they never wait on the store or start a thread, take only locks
    that no thread holds while waiting on the store (letting go of the GIL only while such a lock
    is contended), and do no Python work per row; one list built at C speed, such as a pending
    result per row, is the most. A driver whose store calls block runs them on threads of its own,
    and it alone decides how they are batched and in what order gets and puts run.
    ``put``, ``get``, ``cancel`` and ``probe`` may run concurrently with each other and with the
    calls above. ``poll``, ``access_ended`` and ``access_unprovable`` may be called from any
    thread. Locks inside a driver guard its state only and are never held while waiting on the
    store.

    Side effects: :class:`DriverRejected` is the only raise that certifies nothing escaped, and
    ``put``/``get`` raise it only before recording the operation. Once recorded, the call returns
    the handle and every later failure, of any kind, is a row result; the operation holds the
    rows' memory until ``access_ended`` is true. Neither a failure result nor a successful
    ``unregister`` is access-end evidence.

    Memory: the caller keeps the spans of a put unchanged, and lets nothing but the operation itself
    write the spans of a get, until ``access_ended`` is true; a logical result does not lift that. The
    driver never reads or writes outside the spans of the rows it was given.

    Integrity: a READY get row is the whole object, intact, under the right key; a READY put row
    is a complete object any reader can get. How a driver assures this is its own: a store's
    native size checks, hidden unfinished objects and checksums, or data the driver keeps beside
    the object. A failed, declined or cancelled put never damages an existing complete object.

    Non-blocking calls (``poll``, ``access_ended``, ``access_unprovable``, ``cancel``) read or
    update recorded state: they never block, never ask the store and never raise for a handle this
    driver issued. A handle from elsewhere, or any call after ``close``, is a programming error
    and raises.
    """

    @property
    def capabilities(self) -> DriverCapabilities:
        """The same value on every call."""
        ...

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> RegionHandle:
        """Register one region the driver may read and write.

        ``layout.memory_kind`` says whether the region is host or device memory; a DEVICE region
        is handed only to a driver that declares ``device_memory``. The kind is checked at this
        entry: a driver without ``device_memory`` rejects a DEVICE region before the store is
        asked or the memory touched, so the mismatch never reaches a registration or a transfer.

        Raises:
            DriverRejected: Bad geometry, overlap with a live registration, or a DEVICE region
                for a driver without ``device_memory``; nothing escaped.
            RegistrationError: The store refused; ``handle`` is set when the store may have
                registered the region anyway.
        """
        ...

    def unregister(self, handle: RegionHandle) -> None:
        """Revoke a registration.

        Called only once every operation touching the region has ``access_ended``; otherwise
        raises :class:`DriverRejected` and the registration stays live. Idempotent once it
        succeeded; any other raise leaves the registration live and the call retryable.
        """
        ...

    def put(self, rows: RowBatch, *, on_change: OnChange | None = None) -> OpHandle:
        """Publish ``rows``; returns without waiting on the store, usually before any byte moved.
        ``on_change`` is called as ``OnChange`` says.

        The caller guarantees that the rows' spans hold the payload and stay unchanged until
        ``access_ended`` is true, and hands rows of more than one span only to a driver that
        declares ``multi_span``; a driver may treat unexpected ones as a programming error.

        Raises:
            DriverRejected: ``put`` is not declared, the driver is draining or closed, ``rows``
                exceed ``max_rows_per_op``, or an extent lies outside every live registration;
                nothing escaped.
        """
        ...

    def get(
        self, rows: RowBatch, hint: Hint = None, *, on_change: OnChange | None = None
    ) -> OpHandle:
        """Fetch ``rows``, from where ``hint`` says when given; returns without waiting on the
        store, usually before any byte moved. ``on_change`` is called as ``OnChange`` says.

        The caller guarantees that no two spans of one call overlap, within a row or across rows,
        so a driver may write the rows concurrently without checking, and that nothing but this
        operation writes those spans until ``access_ended`` is true. ``hint`` is passed only to a
        driver that declares ``hints``; the backend rejects the submission otherwise, and a driver
        may treat an unexpected hint as a programming error.

        Raises as ``put``.
        """
        ...

    def poll(self, op: OpHandle) -> tuple[RowResult, ...]:
        """One result per row, in submission order. A concluded row never changes."""
        ...

    def access_ended(self, op: OpHandle) -> bool:
        """True once the operation can no longer touch any of its rows' memory, with evidence.

        Never turns false again. Independent of ``poll``. Without evidence the answer is False:
        a timeout, an error or a guess never turns it true.
        """
        ...

    def access_unprovable(self, op: OpHandle) -> bool:
        """True once the driver's own mechanisms can never prove this operation's access ended,
        for example when the store failed before issuing anything the driver could wait on.

        Never turns false again. Not permission to reuse memory: the rows stay protected until
        evidence arrives from elsewhere, for example the store's own teardown; the backend counts
        such operations instead of waiting for them. A driver that cannot tell always answers
        False.
        """
        ...

    def cancel(self, op: OpHandle) -> None:
        """Best effort: rows not started conclude as CANCELLED; started rows may still finish.
        Called only when ``capabilities.cancel`` is set."""
        ...

    def probe(
        self, keys: Sequence[str], budget_s: float, hint: Hint = None
    ) -> tuple[ProbeStatus, ...]:
        """Advisory presence of ``keys``, at the source ``hint`` names when given, answered within
        ``budget_s`` seconds.

        Keys the store did not answer in time are UNKNOWN, never MISS; the call returns within the
        budget whatever the store does, and a driver with no capacity to ask answers UNKNOWN at
        once. A key held by an unfinished write is NOT_READY, as a get would report. Called only
        when ``capabilities.probe`` is set, and with a hint only when ``capabilities.hints`` is.
        Raises on a store error rather than answering MISS.
        """
        ...

    def drain(self, budget_s: float) -> bool:
        """Stop starting work and wait up to ``budget_s`` seconds for every operation to end its
        access and for the driver's own background work, such as outstanding probes, to finish.

        One-way: after the first call the driver rejects new ``put``/``get`` for good, whatever
        the result. Before it returns, every row the driver has not started working on (no
        transfer or store call taken up) concludes CANCELLED, and an operation left with nothing
        running ends its access. Returns False at once when an operation that cannot
        prove its end remains.
        True when nothing can still touch memory and nothing is left running; False means keep
        everything registered and try again. Never forces an end.
        """
        ...

    def close(self) -> None:
        """Release the store client and every resource the driver holds.

        Allowed only after ``drain`` returned True and every registration is unregistered;
        otherwise raises :class:`DriverRejected` and changes nothing. Idempotent.
        """
        ...
