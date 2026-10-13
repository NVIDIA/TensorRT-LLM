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
from typing import Protocol, runtime_checkable


class StoreError(Exception):
    """A store-side failure the driver could not express as a row result."""


class RegistrationError(StoreError):
    """Registering a region failed.

    Args:
        reason: What went wrong.
        handle: The handle to ``unregister`` later when the store may have registered the region
            anyway, or ``None`` when nothing escaped.
    """

    def __init__(self, reason: str, handle: RegionHandle | None = None) -> None:
        super().__init__(reason)
        self.handle = handle


class RowStatus(enum.Enum):
    """The logical state of one row within an operation."""

    PENDING = "pending"
    """Not concluded yet."""

    READY = "ready"
    """For a get: the bytes landed whole and the driver will not write the row again. For a put:
    the object is readable whole under its key."""

    MISS = "miss"
    """For a get: the store holds no object under the key. Not a failure."""

    CANCELLED = "cancelled"
    """Never started, because of a cancel or a drain; the row's memory was not touched."""

    FAULT = "fault"
    """Anything else: timeout, size or format mismatch, transport or store error."""


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
    """One row to move: its store key, where its payload sits and where its trailer sits.

    The payload is the KV bytes of one staging slot; the trailer is the format's fixed tail, in a
    buffer of the backend's own. The two spans never overlap. ``position`` is ``(group, row)``
    within the submission, kept so the backend can map results back; drivers treat it as opaque.
    """

    key: str
    payload: Span
    trailer: Span
    position: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("a row needs a store key")
        if self.payload.overlaps(self.trailer):
            raise ValueError(f"payload and trailer overlap for key {self.key}")


class RegionHandle(Protocol):
    """Opaque handle of one registration; only the driver that issued it reads it."""


class OpHandle(Protocol):
    """Opaque handle of one put or get; only the driver that issued it reads it."""


class ProbeStatus(enum.Enum):
    """What a probe knows about one key, without moving payload."""

    HIT = "hit"
    MISS = "miss"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DriverCapabilities:
    """What a driver implements beyond the required methods.

    Args:
        cancel: ``cancel`` does something; without it the backend only suppresses unstarted work.
        probe: ``probe`` answers; without it the backend fetches blindly.
        max_rows_per_op: Rows one ``put``/``get`` accepts, or ``None`` for no limit.
    """

    cancel: bool = False
    probe: bool = False
    max_rows_per_op: int | None = None


@runtime_checkable
class StoreDriver(Protocol):
    """What the backend asks of a store.

    Threads: ``register``, ``unregister``, ``put``, ``get``, ``cancel`` and ``drain`` are called one
    at a time from the backend's own threads; ``poll`` and ``access_ended`` may be called from any
    thread, concurrently with the rest. A driver over a blocking SDK blocks inside ``put``/``get``;
    the backend keeps those calls off the submission thread.

    Side effects: ``register``, ``put`` and ``get`` raise ``ValueError`` only before anything
    escaped, so the caller may forget the call. Once ``put``/``get`` has returned, every later
    problem is a row result, and the operation holds the rows' memory until ``access_ended`` is
    true. A failure return is not access-end evidence.
    """

    @property
    def capabilities(self) -> DriverCapabilities: ...

    def register(self, address: int, nbytes: int, layout: RegionLayout) -> RegionHandle:
        """Register one host region the driver may read and write.

        Raises:
            ValueError: Bad geometry or overlap with a live registration; nothing escaped.
            RegistrationError: The store refused; ``handle`` is set when the store may have
                registered the region anyway.
        """
        ...

    def unregister(self, handle: RegionHandle) -> None:
        """Revoke a registration. Idempotent once it succeeded; a raise leaves the registration
        live and the call retryable."""
        ...

    def put(self, rows: Sequence[RowIO]) -> OpHandle:
        """Publish ``rows``; may return before any byte moved.

        Raises:
            ValueError: A span lies outside a live registration, or ``rows`` exceed the driver's
                capabilities; nothing escaped.
        """
        ...

    def get(self, rows: Sequence[RowIO]) -> OpHandle:
        """Fetch ``rows``; may return before any byte moved. Raises as ``put``."""
        ...

    def poll(self, op: OpHandle) -> tuple[RowResult, ...]:
        """One result per row, in submission order; non-blocking. A concluded row never changes."""
        ...

    def access_ended(self, op: OpHandle) -> bool:
        """True once the operation can no longer touch any of its rows' memory, with evidence.
        Never turns false again. Independent of ``poll``."""
        ...

    def cancel(self, op: OpHandle) -> None:
        """Best effort: rows not started conclude as CANCELLED; started rows may still finish.
        Called only when ``capabilities.cancel`` is set."""
        ...

    def probe(self, keys: Sequence[str], budget_s: float) -> tuple[ProbeStatus, ...]:
        """Advisory presence of ``keys``, within ``budget_s`` seconds; UNKNOWN when the budget runs
        out. Called only when ``capabilities.probe`` is set. Raises on a store error rather than
        answering MISS."""
        ...

    def drain(self, budget_s: float) -> bool:
        """Stop starting work and wait up to ``budget_s`` seconds for every operation to end its
        access. True when none can still touch memory; False means keep everything registered and
        try again. Never forces an end."""
        ...
