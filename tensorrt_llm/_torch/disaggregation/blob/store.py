# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The private boundary between the blob backend and a byte store.

A driver moves whole rows between registered host memory and one store. It answers two questions
separately: what each row's logical result is (:meth:`StoreDriver.poll`), and whether an operation
can still touch the caller's memory (:meth:`StoreDriver.access_ended`). Neither answer implies the
other. Nothing here belongs to the public backend contract in ``base/``; the backend translates
between the two.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

Hint = Any
"""The caller's hint for where a get or probe should look, handed to the driver unchanged. The
backend passes a hint only to a driver that declares ``hints`` and rejects it before dispatch for
any other, which therefore keeps its hint-free calls as they were."""


class StoreError(Exception):
    """A store-side failure the driver could not express as a row result."""


class DriverRejected(StoreError):
    """Raised by a driver on purpose, before it touched the store or the caller's memory.

    The only exception that certifies nothing escaped: the caller may forget the call. Any other
    raise from ``register``, ``put`` or ``get`` may have escaped, and the caller keeps the memory
    protected until the driver proves otherwise, at the latest through a ``drain`` returning True.
    """


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

    READY and FAULT are the only two that carry a verdict; MISS, NOT_READY and DECLINED say that
    nothing moved and nothing went wrong, so the backend reports the row as not delivered without
    a fault and they never count toward isolation.
    """

    PENDING = "pending"
    """Not concluded yet."""

    READY = "ready"
    """For a get: the payload landed whole and intact, and the driver will not write the row again.
    For a put: a complete object is readable under the key by any reader now, whoever wrote it. A
    store's "already exists" counts only once that object is committed."""

    MISS = "miss"
    """For a get: the store holds no object under the key."""

    NOT_READY = "not_ready"
    """The key is held by a write that has not finished: a get delivered nothing, a put published
    nothing of its own. ``probe`` gives the same answer for such a key."""

    DECLINED = "declined"
    """For a put: the store would not take the row, for space or policy. Nothing was published;
    the other rows of the operation go on."""

    CANCELLED = "cancelled"
    """Never started, because of a cancel or a drain; the row's memory was not touched."""

    FAULT = "fault"
    """Anything else: timeout, size or integrity mismatch, transport or store error."""


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


PENDING = RowResult(RowStatus.PENDING)
READY = RowResult(RowStatus.READY)
MISS = RowResult(RowStatus.MISS)
NOT_READY = RowResult(RowStatus.NOT_READY)
DECLINED = RowResult(RowStatus.DECLINED)
CANCELLED = RowResult(RowStatus.CANCELLED)


def faulted(reason: str) -> RowResult:
    """A FAULT result carrying ``reason``."""
    return RowResult(RowStatus.FAULT, reason)


def concluded(results: Sequence[RowResult]) -> bool:
    """Whether no row of ``results`` is pending."""
    return all(result.concluded for result in results)


@dataclass(frozen=True)
class Span:
    """Host memory ``[address, address + nbytes)``."""

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
    """

    slot_bytes: int
    slot_count: int
    label: str = ""

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
    """One row to move: its store key and where its payload sits.

    The payload is the KV bytes of one staging slot in a registered region; the object a driver
    stores is that payload, in whatever format and with whatever integrity data the driver keeps.
    ``position`` is ``(group, row)`` within the submission, kept so the backend can map results
    back; drivers treat it as opaque.
    """

    key: str
    payload: Span
    position: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("a row needs a store key")


class RegionHandle(Protocol):
    """Opaque handle of one registration; only the driver that issued it reads it."""


class OpHandle(Protocol):
    """Opaque handle of one put or get; only the driver that issued it reads it."""


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
        cancel: ``cancel`` does something; without it the backend only suppresses unstarted work.
        probe: ``probe`` answers; without it the backend fetches blindly.
        max_rows_per_op: Rows one ``put``/``get`` accepts, or ``None`` for no limit.
        hints: ``get`` and ``probe`` read a caller's hint; without it the backend rejects a hint
            before dispatch and the driver is called without one.
    """

    get: bool
    put: bool
    cancel: bool
    probe: bool
    max_rows_per_op: int | None
    hints: bool = False

    def __post_init__(self) -> None:
        if not (self.get or self.put):
            raise ValueError("a driver implements at least one of get and put")
        if self.max_rows_per_op is not None and self.max_rows_per_op <= 0:
            raise ValueError("max_rows_per_op is positive or None")


@runtime_checkable
class StoreDriver(Protocol):
    """What the backend asks of a store.

    Threads: ``register``, ``unregister``, ``drain`` and ``close`` are called one at a time.
    ``put``, ``get``, ``cancel`` and ``probe`` may run concurrently with each other and with those,
    from the backend's own threads; a blocked ``put``/``get`` must not delay another. ``poll``,
    ``access_ended`` and ``access_unprovable`` may be called from any thread. Locks inside a driver
    guard its state only and are never held while waiting on the store.

    Side effects: :class:`DriverRejected` is the only raise that certifies nothing escaped, and
    ``put``/``get`` raise it only before recording the operation. Once recorded, the call returns
    the handle and every later failure, of any kind, is a row result; the operation holds the
    rows' memory until ``access_ended`` is true. Neither a failure result nor a successful
    ``unregister`` is access-end evidence.

    Integrity: a READY get row is the whole object, intact, under the right key; a READY put row
    is a complete object any reader can get. How a driver assures this is its own: a store's
    native size checks, hidden unfinished objects and checksums, or data the driver keeps beside
    the object. A failed, declined or cancelled put never damages an existing complete object.

    Queries (``poll``, ``access_ended``, ``access_unprovable``, ``cancel``) read recorded state:
    they never block, never ask the store and never raise for a handle this driver issued. A handle
    from elsewhere, or any call after ``close``, is a programming error and raises.
    """

    @property
    def capabilities(self) -> DriverCapabilities:
        """The same value on every call."""
        ...

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> RegionHandle:
        """Register one host region the driver may read and write.

        Raises:
            DriverRejected: Bad geometry or overlap with a live registration; nothing escaped.
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

    def put(self, rows: Sequence[RowIO]) -> OpHandle:
        """Publish ``rows``; may return before any byte moved.

        Raises:
            DriverRejected: ``put`` is not declared, the driver is draining or closed, a key
                appears twice in ``rows``, ``rows`` exceed ``max_rows_per_op``, or a payload lies
                outside a live registration; nothing escaped.
        """
        ...

    def get(self, rows: Sequence[RowIO], hint: Hint = None) -> OpHandle:
        """Fetch ``rows``, from where ``hint`` says when given; may return before any byte moved.

        Raises as ``put``. ``hint`` is passed only to a driver that declares ``hints``; the
        backend rejects it before dispatch otherwise, and a driver may treat an unexpected hint
        as a programming error.
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
        the result. Returns False at once when an operation that cannot prove its end remains.
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
