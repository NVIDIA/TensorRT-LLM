# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded node-local streaming for SafeTensors checkpoint payloads.

The node-local rank-zero process is the only storage producer. It fills an
MPI-3 shared-memory double buffer with a bounded host-I/O worker pool while all
local ranks materialize the previously published batch. MPI calls are confined
to the caller thread so the transport does not require ``MPI_THREAD_MULTIPLE``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import threading
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, CancelledError, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BorrowedWeightStorageRetentionError,
    WeightBatch,
    WeightBatchLease,
    WeightBatchStream,
    WeightGroup,
    WeightSegment,
)

_NUM_SHARED_SLOTS = 2
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_DEFAULT_SLOT_BYTES = 256 * 1024 * 1024
_DEFAULT_BUFFER_BUDGET_BYTES = 64 * 1024 * 1024 * 1024
_DEFAULT_READ_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_IO_WORKERS = 16
_SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    "C64": 64,
    "C128": 128,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E5M2": 8,
}


class SharedHostStreamError(RuntimeError):
    """Raised when an active shared-host stream fails collectively."""


class SharedHostStreamUnavailableError(RuntimeError):
    """Raised when strict mode requests an ineligible shared-host stream."""


@dataclass(frozen=True)
class SafeTensorMetadata:
    """Side-effect-free metadata for one tensor in a SafeTensors file."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    source_path: str
    source_offset: int


@dataclass(frozen=True)
class SharedHostStreamPreflight:
    """Ordered SafeTensors metadata parsed before transport selection."""

    tensors: tuple[SafeTensorMetadata, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """Return source keys in deterministic stream order."""
        return tuple(tensor.key for tensor in self.tensors)

    @property
    def checkpoint_nbytes(self) -> int:
        """Return the total tensor payload size."""
        return sum(tensor.nbytes for tensor in self.tensors)


@dataclass(frozen=True)
class SharedHostStreamTelemetry:
    """A snapshot of topology, registration, and stream progress."""

    node_rank: int
    node_size: int
    world_rank: int
    world_size: int
    is_node_producer: bool
    slot_count: int
    configured_slot_bytes: int
    slot_bytes: int
    buffer_budget_bytes: int
    largest_group_nbytes: int
    groups_fitting_single_slot: int
    group_count: int
    io_workers: int
    read_chunk_bytes: int
    host_registered: bool
    all_ranks_host_registered: bool
    host_registration_detail: str
    batches_published: int
    bytes_published: int
    direct_view_groups: int
    direct_view_bytes: int
    staged_groups: int
    staged_bytes: int


@dataclass
class HostMemoryRegistration:
    """Result and lifetime handle for CUDA host-memory registration."""

    registered: bool
    detail: str
    _release: Callable[[], None] | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Release the registration once; a failed release is propagated."""
        if self._closed:
            return
        self._closed = True
        if self._release is not None:
            self._release()


class RangeReader(Protocol):
    """Injectable host-I/O operation used by producer worker threads."""

    def read_into(
        self,
        source_path: str,
        source_offset: int,
        destination: memoryview,
        cancel_event: threading.Event,
    ) -> None:
        """Read exactly ``len(destination)`` bytes into ``destination``."""

    def close(self) -> None:
        """Release reader resources after all worker calls have completed."""


class _Window(Protocol):
    def Lock_all(self) -> None:
        """Open a passive-target access epoch for local loads and stores."""

    def Sync(self) -> None:
        """Synchronize private and public shared-window copies."""

    def Unlock_all(self) -> None:
        """Close the passive-target access epoch."""

    def Free(self) -> None:
        """Collectively free the shared window."""


# Factories are node-collective and must not return ownership on only a subset
# of ranks. The production factory coordinates local setup failures after the
# MPI window has been allocated; ``open_shared_host_weight_stream`` still
# detects a broken custom factory so cleanup fails instead of deadlocking.
WindowFactory = Callable[[Any, int], tuple[_Window, memoryview]]
HostRegistrar = Callable[[memoryview], HostMemoryRegistration]


@dataclass(frozen=True)
class _QuarantinedSharedResources:
    """Transport resources intentionally retained after a lifetime violation."""

    registration: HostMemoryRegistration | None
    arena: memoryview | None
    window: _Window | None


_QUARANTINED_SHARED_RESOURCES: list[_QuarantinedSharedResources] = []
_QUARANTINED_SHARED_RESOURCES_LOCK = threading.Lock()


def _quarantine_shared_resources(
    registration: HostMemoryRegistration | None,
    arena: memoryview | None,
    window: _Window | None,
) -> None:
    """Retain an unsafe-to-free arena for the rest of the process lifetime."""
    with _QUARANTINED_SHARED_RESOURCES_LOCK:
        _QUARANTINED_SHARED_RESOURCES.append(
            _QuarantinedSharedResources(registration, arena, window)
        )


@dataclass(frozen=True)
class _ReadExtent:
    source_path: str
    source_offset: int
    payload_offset: int
    nbytes: int


@dataclass(frozen=True)
class _PlannedBatch:
    batch: WeightBatch
    extents: tuple[_ReadExtent, ...]


@dataclass(frozen=True)
class _SlotLayout:
    configured_slot_bytes: int
    slot_bytes: int
    buffer_budget_bytes: int
    largest_group_nbytes: int
    groups_fitting_single_slot: int
    group_count: int


@dataclass(frozen=True)
class _ControlMessage:
    kind: str
    batch: WeightBatch | None = None
    error_rank: int | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class _RankOutcome:
    phase: str
    rank: int
    descriptor_digest: str | None
    error_rank: int | None
    error_type: str | None
    error_message: str | None
    quarantine_shared_resources: bool


class _ReadCancelled(RuntimeError):
    pass


class _PreadRangeReader:
    """Place file byte ranges directly into writable shared-memory views."""

    def __init__(self) -> None:
        self._descriptors: dict[str, int] = {}
        self._descriptor_lock = threading.Lock()
        self._closed = False

    def _get_descriptor(self, source_path: str) -> int:
        with self._descriptor_lock:
            if self._closed:
                raise RuntimeError("Range reader has already been closed")
            descriptor = self._descriptors.get(source_path)
            if descriptor is None:
                descriptor = os.open(source_path, os.O_RDONLY)
                self._descriptors[source_path] = descriptor
            return descriptor

    def read_into(
        self,
        source_path: str,
        source_offset: int,
        destination: memoryview,
        cancel_event: threading.Event,
    ) -> None:
        if not destination:
            return
        descriptor = self._get_descriptor(source_path)
        completed = 0
        while completed < len(destination):
            if cancel_event.is_set():
                raise _ReadCancelled("shared-host read was cancelled")
            written = os.preadv(descriptor, [destination[completed:]], source_offset + completed)
            if written == 0:
                raise EOFError(
                    f"Unexpected EOF reading {source_path!r} at offset {source_offset + completed}"
                )
            completed += written

    def close(self) -> None:
        """Close every cached descriptor after the worker pool has stopped."""
        with self._descriptor_lock:
            if self._closed:
                return
            self._closed = True
            descriptors = tuple(self._descriptors.values())
            self._descriptors.clear()

        first_error = None
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class _SharedHostBatchLease(WeightBatchLease):
    def __init__(
        self,
        batch: WeightBatch,
        payload: memoryview,
        *,
        direct_buffer_enabled: bool,
    ) -> None:
        self._batch = batch
        self._payload: memoryview | None = payload
        self._direct_buffer_enabled = direct_buffer_enabled

    @property
    def batch(self) -> WeightBatch:
        return self._batch

    def view(self, segment: WeightSegment) -> memoryview:
        if self._payload is None:
            raise RuntimeError("Weight batch lease has already been released")
        if segment not in self._batch.segments:
            raise ValueError("Weight segment does not belong to this batch")
        start = segment.payload_offset
        return self._payload[start : start + segment.nbytes].toreadonly()

    def borrow_direct_buffer(self, segment: WeightSegment) -> memoryview | None:
        """Borrow a writable view whose bytes callers must treat as immutable."""
        if not self._direct_buffer_enabled:
            return None
        if self._payload is None:
            raise RuntimeError("Weight batch lease has already been released")
        if segment not in self._batch.segments:
            raise ValueError("Weight segment does not belong to this batch")
        start = segment.payload_offset
        return self._payload[start : start + segment.nbytes]

    def release(self) -> None:
        if self._payload is not None:
            self._payload.release()
            self._payload = None


class SharedHostWeightStream(WeightBatchStream):
    """Double-buffered node-local SafeTensors stream.

    The caller owns ``node_communicator`` and ``world_communicator``. This
    object owns only its shared window, host registration, and I/O workers.
    """

    def __init__(
        self,
        plans: tuple[_PlannedBatch, ...],
        groups: tuple[WeightGroup, ...],
        node_communicator: Any,
        world_communicator: Any,
        window: _Window,
        arena: memoryview,
        registration: HostMemoryRegistration,
        *,
        window_locked: bool,
        slot_layout: _SlotLayout,
        all_ranks_host_registered: bool,
        io_workers: int,
        read_chunk_bytes: int,
        reader: RangeReader,
        executor: ThreadPoolExecutor | None,
    ) -> None:
        self._plans = plans
        self._groups = groups
        self._node_communicator = node_communicator
        self._world_communicator = world_communicator
        self._window: _Window | None = window
        self._window_locked = window_locked
        self._arena: memoryview | None = arena
        self._registration: HostMemoryRegistration | None = registration
        self._slot_layout = slot_layout
        self._slot_bytes = slot_layout.slot_bytes
        self._all_ranks_host_registered = all_ranks_host_registered
        self._io_workers = io_workers
        self._read_chunk_bytes = read_chunk_bytes
        self._reader = reader
        self._node_rank = node_communicator.Get_rank()
        self._node_size = node_communicator.Get_size()
        self._world_rank = world_communicator.Get_rank()
        self._world_size = world_communicator.Get_size()
        self._owner_thread = threading.get_ident()
        self._executor = executor
        self._cancel_event = threading.Event()
        self._pending_index: int | None = None
        self._pending_futures: tuple[Future[None], ...] = ()
        self._next_index = 0
        self._current: _SharedHostBatchLease | None = None
        self._state = "created"
        self._finalized = False
        self._terminal_error: BaseException | None = None
        self._failure_coordinated = False
        self._quarantine_required = False
        self._batches_published = 0
        self._bytes_published = 0
        self._direct_view_groups = 0
        self._direct_view_bytes = 0
        self._staged_groups = 0
        self._staged_bytes = 0

    @property
    def groups(self) -> tuple[WeightGroup, ...]:
        return self._groups

    @property
    def supports_direct_tensor_views(self) -> bool:
        """Whether every rank can borrow the live CUDA-registered arena."""
        registration = self._registration
        return bool(
            self._all_ranks_host_registered and registration is not None and registration.registered
        )

    def record_materialization(self, *, direct: bool, nbytes: int) -> None:
        """Record one successfully materialized atomic group."""
        self._assert_owner_thread()
        if nbytes < 0:
            raise ValueError("Materialized byte count must be nonnegative")
        if direct:
            self._direct_view_groups += 1
            self._direct_view_bytes += nbytes
        else:
            self._staged_groups += 1
            self._staged_bytes += nbytes

    @property
    def telemetry(self) -> SharedHostStreamTelemetry:
        """Return current transport telemetry."""
        registration = self._registration
        return SharedHostStreamTelemetry(
            node_rank=self._node_rank,
            node_size=self._node_size,
            world_rank=self._world_rank,
            world_size=self._world_size,
            is_node_producer=self._node_rank == 0,
            slot_count=_NUM_SHARED_SLOTS,
            configured_slot_bytes=self._slot_layout.configured_slot_bytes,
            slot_bytes=self._slot_bytes,
            buffer_budget_bytes=self._slot_layout.buffer_budget_bytes,
            largest_group_nbytes=self._slot_layout.largest_group_nbytes,
            groups_fitting_single_slot=(self._slot_layout.groups_fitting_single_slot),
            group_count=self._slot_layout.group_count,
            io_workers=self._io_workers,
            read_chunk_bytes=self._read_chunk_bytes,
            host_registered=(registration.registered if registration is not None else False),
            all_ranks_host_registered=self._all_ranks_host_registered,
            host_registration_detail=(
                registration.detail if registration is not None else "registration released"
            ),
            batches_published=self._batches_published,
            bytes_published=self._bytes_published,
            direct_view_groups=self._direct_view_groups,
            direct_view_bytes=self._direct_view_bytes,
            staged_groups=self._staged_groups,
            staged_bytes=self._staged_bytes,
        )

    def _assert_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("SharedHostWeightStream coordination must run on its creator thread")

    def _schedule(self, index: int) -> None:
        if self._node_rank != 0:
            return
        assert self._executor is not None
        assert self._arena is not None
        if self._pending_index is not None:
            raise RuntimeError("A shared-host fill is already pending")
        plan = self._plans[index]
        slot_start = plan.batch.slot * self._slot_bytes
        futures = []
        destinations = []
        submit_failure = None
        try:
            for extent in plan.extents:
                destination = self._arena[
                    slot_start + extent.payload_offset : slot_start
                    + extent.payload_offset
                    + extent.nbytes
                ]
                destinations.append(destination)
                futures.append(
                    self._executor.submit(
                        self._reader.read_into,
                        extent.source_path,
                        extent.source_offset,
                        destination,
                        self._cancel_event,
                    )
                )
        except BaseException as error:
            # Do not propagate ``error`` itself. A submit implementation may
            # retain its arguments in that exception's traceback, including
            # ``destination`` views into the MPI arena. Convert it to inert
            # strings and raise a fresh error only after every worker and view
            # has been settled.
            submit_failure = (type(error).__name__, str(error))
            self._cancel_event.set()
            for future in futures:
                future.cancel()
            # A submit failure is terminal for this stream. Shutting down here
            # guarantees that accepted or cancelled work items no longer own
            # destination arguments before shared-window teardown begins.
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            wait(futures)
            self._clear_future_traceback_references(futures)
            for destination_view in destinations:
                destination_view.release()
            destinations.clear()
            futures.clear()
            destination = None

        if submit_failure is not None:
            error_type, error_message = submit_failure
            raise SharedHostStreamError(
                f"Failed to schedule a shared-host range read: {error_type}: {error_message}"
            )
        self._pending_index = index
        self._pending_futures = tuple(futures)

    def _wait_for_pending(self, index: int) -> BaseException | None:
        if self._node_rank != 0:
            return None
        if self._pending_index != index:
            raise RuntimeError(f"Expected pending batch {index}, got {self._pending_index}")
        futures = self._pending_futures
        if futures:
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            if any(future.exception() is not None for future in done if not future.cancelled()):
                self._cancel_event.set()
                for future in futures:
                    future.cancel()
                wait(futures)

        first_error = None
        first_cancel = None
        for future in futures:
            try:
                error = future.exception()
            except CancelledError as error:
                if first_cancel is None:
                    first_cancel = error
            else:
                if isinstance(error, _ReadCancelled):
                    if first_cancel is None:
                        first_cancel = error
                elif error is not None and first_error is None:
                    first_error = error

        # A completed Future retains its worker exception and traceback. The
        # reader frame owns the destination memoryview into the MPI arena, so
        # drop those traceback frames before forgetting the consumed futures.
        # Use Future.exception() above rather than Future.result(): re-raising
        # here would append this still-executing frame to the traceback and make
        # traceback.clear_frames() unsafe until after this method returns.
        self._clear_future_traceback_references(futures)
        self._pending_index = None
        self._pending_futures = ()
        return first_error or first_cancel

    @staticmethod
    def _error_fields(
        error: BaseException | None,
        rank: int,
    ) -> tuple[int | None, str | None, str | None]:
        if error is None:
            return None, None, None
        return rank, type(error).__name__, str(error)

    def _world_consensus(
        self,
        phase: str,
        *,
        descriptor_digest: str | None = None,
        error: BaseException | None = None,
        error_rank: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        quarantine_shared_resources = self._quarantine_required or isinstance(
            error, BorrowedWeightStorageRetentionError
        )
        if error is not None:
            error_rank, error_type, error_message = self._error_fields(error, self._world_rank)
        outcome = _RankOutcome(
            phase=phase,
            rank=self._world_rank,
            descriptor_digest=descriptor_digest,
            error_rank=error_rank,
            error_type=error_type,
            error_message=error_message,
            quarantine_shared_resources=quarantine_shared_resources,
        )
        outcomes = self._world_communicator.allgather(outcome)
        if any(candidate.quarantine_shared_resources for candidate in outcomes):
            self._quarantine_required = True
        phases = {candidate.phase for candidate in outcomes}
        if phases != {phase}:
            raise SharedHostStreamError(
                f"Shared-host collective order diverged during {phase}: {sorted(phases)}"
            )
        errors = [candidate for candidate in outcomes if candidate.error_rank is not None]
        if errors:
            selected = min(errors, key=lambda candidate: (candidate.error_rank, candidate.rank))
            error_message = (
                f"Shared-host {phase} failed on world rank "
                f"{selected.error_rank}: {selected.error_type}: "
                f"{selected.error_message}"
            )
            if self._quarantine_required:
                raise BorrowedWeightStorageRetentionError(error_message)
            raise SharedHostStreamError(error_message)
        digests = {candidate.descriptor_digest for candidate in outcomes}
        if len(digests) != 1:
            raise SharedHostStreamError(f"Shared-host batch descriptors diverged during {phase}")

    @staticmethod
    def _control_digest(control: _ControlMessage) -> str:
        return hashlib.sha256(repr(control).encode("utf-8")).hexdigest()

    def _cancel(self) -> None:
        self._cancel_event.set()
        self._state = "aborted"

    @staticmethod
    def _clear_future_traceback_references(
        futures: Sequence[Future[None]],
    ) -> None:
        """Drop completed worker tracebacks before the shared arena is freed.

        ``Future`` retains a worker exception and its traceback. A failed
        range read's frame, in turn, retains the destination memoryview into
        the MPI window. Clear those frames while the window is still valid so
        failure cleanup cannot leave an exported view pointing at freed shared
        memory.
        """
        for future in futures:
            if not future.done() or future.cancelled():
                continue
            try:
                error = future.exception()
            except CancelledError:
                continue
            if error is None or error.__traceback__ is None:
                continue
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None

    def start(self, error: BaseException | None = None) -> None:
        self._assert_owner_thread()
        if self._finalized:
            raise RuntimeError("Shared-host stream has been finalized")
        if self._state != "created":
            raise RuntimeError(f"Shared-host stream cannot start from state {self._state!r}")
        try:
            self._world_consensus("start", error=error)
        except BaseException as coordinated_error:
            self._terminal_error = coordinated_error
            self._failure_coordinated = True
            self._cancel()
            raise
        self._state = "open"

    def begin_next(self) -> WeightBatchLease | None:
        self._assert_owner_thread()
        if self._finalized:
            raise RuntimeError("Shared-host stream has been finalized")
        if self._state == "aborted":
            raise SharedHostStreamError("Shared-host stream has been aborted")
        if self._state == "created":
            raise RuntimeError("Call start() on every rank before requesting a batch")
        if self._current is not None:
            raise RuntimeError("Complete the current batch before requesting another")
        if self._state == "eof":
            return None

        index = self._next_index
        local_control = None
        if self._node_rank == 0:
            try:
                if index < len(self._plans):
                    if self._pending_index is None:
                        self._schedule(index)
                    fill_error = self._wait_for_pending(index)
                    if fill_error is None:
                        assert self._window is not None
                        self._window.Sync()
                        local_control = _ControlMessage(
                            kind="batch", batch=self._plans[index].batch
                        )
                    else:
                        error_rank, error_type, error_message = self._error_fields(
                            fill_error, self._world_rank
                        )
                        local_control = _ControlMessage(
                            kind="error",
                            error_rank=error_rank,
                            error_type=error_type,
                            error_message=error_message,
                        )
                else:
                    local_control = _ControlMessage(kind="eof")
            except BaseException as producer_error:
                error_rank, error_type, error_message = self._error_fields(
                    producer_error, self._world_rank
                )
                local_control = _ControlMessage(
                    kind="error",
                    error_rank=error_rank,
                    error_type=error_type,
                    error_message=error_message,
                )

        control = self._node_communicator.bcast(local_control, root=0)
        local_error = None
        try:
            self._node_communicator.Barrier()
            assert self._window is not None
            self._window.Sync()
        except BaseException as synchronization_error:
            local_error = synchronization_error
        if not isinstance(control, _ControlMessage) and local_error is None:
            local_error = RuntimeError(f"Invalid node-local control message {control!r}")

        if (
            local_error is None
            and isinstance(control, _ControlMessage)
            and control.kind == "batch"
            and self._node_rank == 0
            and index + 1 < len(self._plans)
        ):
            try:
                self._schedule(index + 1)
            except BaseException as scheduling_error:
                local_error = scheduling_error

        control_error_rank = control.error_rank if isinstance(control, _ControlMessage) else None
        control_error_type = control.error_type if isinstance(control, _ControlMessage) else None
        control_error_message = (
            control.error_message if isinstance(control, _ControlMessage) else None
        )
        try:
            self._world_consensus(
                f"publish:{index}",
                descriptor_digest=self._control_digest(control),
                error=local_error,
                error_rank=control_error_rank,
                error_type=control_error_type,
                error_message=control_error_message,
            )
        except BaseException as coordinated_error:
            self._terminal_error = coordinated_error
            self._failure_coordinated = True
            self._cancel()
            raise

        if control.kind == "eof":
            self._state = "eof"
            return None
        if control.kind != "batch" or control.batch is None:
            self._cancel()
            raise SharedHostStreamError(f"Invalid shared-host control message {control!r}")
        expected = self._plans[index].batch
        if control.batch != expected:
            self._cancel()
            raise SharedHostStreamError("Node producer published an unexpected batch descriptor")

        assert self._arena is not None
        batch = control.batch
        slot_start = batch.slot * self._slot_bytes
        payload = self._arena[slot_start : slot_start + batch.payload_nbytes]
        lease = _SharedHostBatchLease(
            batch,
            payload,
            direct_buffer_enabled=self.supports_direct_tensor_views,
        )
        self._current = lease
        self._next_index += 1
        self._batches_published += 1
        self._bytes_published += batch.payload_nbytes
        return lease

    def _finish_current(self, lease: WeightBatchLease | None, error: BaseException | None) -> None:
        validation_error = error
        current = self._current
        if current is None:
            if validation_error is None:
                validation_error = RuntimeError("No shared-host batch is awaiting acknowledgement")
            sequence = self._next_index
        else:
            sequence = current.batch.sequence
            if lease is not None and lease is not current and validation_error is None:
                validation_error = ValueError(
                    "Attempted to acknowledge a lease from another stream or batch"
                )
            try:
                current.release()
            except BaseException as release_error:
                if validation_error is None:
                    validation_error = release_error
            self._current = None
        try:
            self._world_consensus(f"complete:{sequence}", error=validation_error)
        except BaseException as coordinated_error:
            self._terminal_error = coordinated_error
            self._failure_coordinated = True
            self._cancel()
            raise

    def complete(self, lease: WeightBatchLease, error: BaseException | None = None) -> None:
        self._assert_owner_thread()
        if self._finalized:
            raise RuntimeError("Shared-host stream has been finalized")
        if self._state == "aborted":
            raise SharedHostStreamError("Shared-host stream has been aborted")
        self._finish_current(lease, error)

    def abort(self, error: BaseException) -> None:
        self._assert_owner_thread()
        if self._finalized:
            return
        if self._state == "aborted" and self._failure_coordinated:
            return
        if self._terminal_error is None:
            self._terminal_error = error
        if self._state == "aborted":
            return
        if self._current is None:
            self._cancel()
            return
        try:
            self._finish_current(self._current, error)
        except BaseException as coordinated_error:
            self._terminal_error = coordinated_error
            raise
        finally:
            self._cancel()

    def finalize(self, error: BaseException | None = None) -> None:
        self._assert_owner_thread()
        if self._finalized:
            return
        if error is not None and self._terminal_error is None:
            self._terminal_error = error
        deferred_error = None
        if self._current is not None:
            try:
                self.abort(RuntimeError("Cannot finalize with an outstanding batch lease"))
            except BaseException as error:
                deferred_error = error
        elif self._state in ("created", "open"):
            self._terminal_error = self._terminal_error or RuntimeError(
                "Shared-host stream was finalized before reaching EOF"
            )
            self._cancel()

        terminal_error = None if self._failure_coordinated else self._terminal_error
        try:
            self._world_consensus("terminal", error=terminal_error)
        except BaseException as terminal_error:
            deferred_error = deferred_error or terminal_error

        self._cancel_event.set()
        cleanup_error = None
        pending_futures = self._pending_futures
        self._pending_futures = ()
        self._pending_index = None
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except BaseException as shutdown_error:
                cleanup_error = shutdown_error
            self._executor = None
        if pending_futures:
            # ``shutdown(wait=True)`` normally guarantees completion. Keep the
            # explicit wait for the exceptional shutdown path too: no worker
            # may still access a slot when host registration is removed.
            wait(pending_futures)
            try:
                self._clear_future_traceback_references(pending_futures)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        del pending_futures

        close_reader = getattr(self._reader, "close", None)
        if close_reader is not None:
            try:
                close_reader()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

        registration = self._registration
        self._registration = None
        arena = self._arena
        self._arena = None
        window = self._window
        self._window = None
        window_locked = self._window_locked
        self._window_locked = False
        if self._quarantine_required:
            # A tensor escaped a lease on at least one rank. The collective
            # completion consensus propagated that fact to every peer, so all
            # ranks take the same teardown branch. Retaining the registered
            # MPI window is an intentional bounded leak on a failed startup;
            # unregistering or freeing it could invalidate escaped storage.
            _quarantine_shared_resources(registration, arena, window)
        else:
            safe_to_free, resource_cleanup_error = _prepare_shared_window_teardown(
                self._node_communicator,
                registration,
                arena,
                window,
                window_locked=window_locked,
            )
            if cleanup_error is None and resource_cleanup_error is not None:
                cleanup_error = resource_cleanup_error

            if safe_to_free and window is not None:
                try:
                    self._node_communicator.Barrier()
                    window.Free()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error

        self._finalized = True
        try:
            self._world_consensus("cleanup", error=cleanup_error)
        except BaseException as error:
            if deferred_error is None:
                deferred_error = error
        if deferred_error is not None:
            raise deferred_error


def prepare_shared_host_weight_stream(
    weight_files: Sequence[str | os.PathLike[str]],
) -> SharedHostStreamPreflight:
    """Parse ordered SafeTensors metadata without allocating shared resources."""
    if not weight_files:
        raise SharedHostStreamUnavailableError(
            "shared_host_producer requires at least one SafeTensors file"
        )
    tensors = []
    keys = set()
    for path_like in sorted(weight_files, key=lambda path: os.fspath(path)):
        path = Path(path_like)
        file_size = path.stat().st_size
        with path.open("rb") as source:
            length_bytes = source.read(8)
            if len(length_bytes) != 8:
                raise SharedHostStreamUnavailableError(
                    f"Invalid SafeTensors header in {str(path)!r}: missing length"
                )
            header_length = int.from_bytes(length_bytes, "little")
            if header_length > _MAX_SAFETENSORS_HEADER_BYTES:
                raise SharedHostStreamUnavailableError(
                    f"SafeTensors header in {str(path)!r} exceeds "
                    f"{_MAX_SAFETENSORS_HEADER_BYTES} bytes"
                )
            if 8 + header_length > file_size:
                raise SharedHostStreamUnavailableError(
                    f"Invalid SafeTensors header length in {str(path)!r}"
                )
            try:
                header = json.loads(source.read(header_length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SharedHostStreamUnavailableError(
                    f"Invalid SafeTensors JSON in {str(path)!r}: {error}"
                ) from error
        if not isinstance(header, dict):
            raise SharedHostStreamUnavailableError(
                f"SafeTensors header in {str(path)!r} must be an object"
            )
        data_start = 8 + header_length
        file_tensors = []
        file_ranges = []
        for key in sorted(name for name in header if name != "__metadata__"):
            entry = header[key]
            if key in keys:
                raise SharedHostStreamUnavailableError(f"Duplicate SafeTensors key {key!r}")
            if not isinstance(entry, dict):
                raise SharedHostStreamUnavailableError(f"Invalid SafeTensors metadata for {key!r}")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if (
                not isinstance(dtype, str)
                or not isinstance(shape, list)
                or any(not isinstance(value, int) or value < 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) for value in offsets)
            ):
                raise SharedHostStreamUnavailableError(f"Invalid SafeTensors metadata for {key!r}")
            start, end = offsets
            if start < 0 or end < start or data_start + end > file_size:
                raise SharedHostStreamUnavailableError(
                    f"Invalid SafeTensors data range for {key!r}"
                )
            dtype_bits = _SAFETENSORS_DTYPE_BITS.get(dtype)
            if dtype_bits is None:
                raise SharedHostStreamUnavailableError(
                    f"Unsupported SafeTensors dtype {dtype!r} for {key!r}"
                )
            element_count = math.prod(shape)
            expected_nbytes = (element_count * dtype_bits + 7) // 8
            if end - start != expected_nbytes:
                raise SharedHostStreamUnavailableError(
                    f"SafeTensors shape/dtype for {key!r} requires "
                    f"{expected_nbytes} bytes, but its data range has "
                    f"{end - start}"
                )
            keys.add(key)
            file_ranges.append((start, end, key))
            file_tensors.append(
                SafeTensorMetadata(
                    key=key,
                    dtype=dtype,
                    shape=tuple(shape),
                    nbytes=end - start,
                    source_path=str(path),
                    source_offset=data_start + start,
                )
            )
        previous_end = 0
        previous_key = None
        for start, end, key in sorted(file_ranges):
            if start < previous_end:
                raise SharedHostStreamUnavailableError(
                    f"SafeTensors data ranges for {previous_key!r} and "
                    f"{key!r} overlap in {str(path)!r}"
                )
            if start > previous_end:
                raise SharedHostStreamUnavailableError(
                    f"SafeTensors data buffer in {str(path)!r} has an unindexed gap before {key!r}"
                )
            previous_end = end
            previous_key = key
        if previous_end != file_size - data_start:
            raise SharedHostStreamUnavailableError(
                f"SafeTensors data buffer in {str(path)!r} has unindexed trailing bytes"
            )
        tensors.extend(file_tensors)
    if not tensors:
        raise SharedHostStreamUnavailableError(
            "shared_host_producer found no tensors in the checkpoint"
        )
    return SharedHostStreamPreflight(tuple(tensors))


def _validate_groups(
    preflight: SharedHostStreamPreflight,
    group_manifest: Sequence[WeightGroup] | None,
    single_tensor_groups_safe: bool,
) -> tuple[tuple[WeightGroup, ...] | None, str | None]:
    if group_manifest is None:
        if not single_tensor_groups_safe:
            return None, (
                "shared_host_producer requires an atomic weight-group manifest "
                "or an explicit single_tensor_groups_safe declaration"
            )
        groups = tuple(WeightGroup(group_id=f"tensor:{key}", keys=(key,)) for key in preflight.keys)
    else:
        groups = tuple(group_manifest)
        if not groups:
            return None, "shared_host_producer weight-group manifest is empty"
    group_ids = [group.group_id for group in groups]
    if len(set(group_ids)) != len(group_ids):
        return None, "shared_host_producer weight-group IDs must be unique"

    expected = set(preflight.keys)
    assigned = []
    for group in groups:
        assigned.extend(group.keys)
    assigned_set = set(assigned)
    seen = set()
    duplicates = set()
    for key in assigned:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    missing = sorted(expected - assigned_set)
    unknown = sorted(assigned_set - expected)
    if duplicates or missing or unknown:
        return None, (
            "shared_host_producer weight-group manifest must partition source "
            f"keys exactly once; duplicate={sorted(duplicates)}, missing={missing}, "
            f"unknown={unknown}"
        )
    return groups, None


def _dtype_alignment(dtype: str) -> int:
    bits = _SAFETENSORS_DTYPE_BITS[dtype]
    return bits // 8 if bits % 8 == 0 else 1


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _packed_group_nbytes(metadata: dict[str, SafeTensorMetadata], group: WeightGroup) -> int:
    payload_nbytes = 0
    for key in group.keys:
        tensor = metadata[key]
        payload_nbytes = _align_up(payload_nbytes, _dtype_alignment(tensor.dtype))
        payload_nbytes += tensor.nbytes
    return payload_nbytes


def _resolve_slot_layout(
    preflight: SharedHostStreamPreflight,
    groups: tuple[WeightGroup, ...],
    configured_slot_bytes: int,
    buffer_budget_bytes: int,
) -> _SlotLayout:
    """Grow slots to atomic groups without exceeding the two-slot budget."""
    maximum_alignment = max(_dtype_alignment(tensor.dtype) for tensor in preflight.tensors)
    maximum_slot_bytes = buffer_budget_bytes // _NUM_SHARED_SLOTS
    maximum_slot_bytes -= maximum_slot_bytes % maximum_alignment
    if maximum_slot_bytes < configured_slot_bytes:
        raise ValueError(
            "shared_host_producer buffer_budget_bytes must hold two configured "
            f"slots ({_NUM_SHARED_SLOTS * configured_slot_bytes} bytes required)"
        )

    metadata = {tensor.key: tensor for tensor in preflight.tensors}
    group_nbytes = tuple(_packed_group_nbytes(metadata, group) for group in groups)
    largest_group_nbytes = max(group_nbytes, default=0)
    desired_slot_bytes = _align_up(
        max(configured_slot_bytes, largest_group_nbytes), maximum_alignment
    )
    slot_bytes = min(desired_slot_bytes, maximum_slot_bytes)
    return _SlotLayout(
        configured_slot_bytes=configured_slot_bytes,
        slot_bytes=slot_bytes,
        buffer_budget_bytes=buffer_budget_bytes,
        largest_group_nbytes=largest_group_nbytes,
        groups_fitting_single_slot=sum(size <= slot_bytes for size in group_nbytes),
        group_count=len(groups),
    )


def get_shared_host_stream_ineligibility_reason(
    preflight: SharedHostStreamPreflight,
    *,
    group_manifest: Sequence[WeightGroup] | None = None,
    single_tensor_groups_safe: bool = False,
    node_communicator: Any | None = None,
    world_communicator: Any | None = None,
    slot_bytes: int = _DEFAULT_SLOT_BYTES,
    buffer_budget_bytes: int = _DEFAULT_BUFFER_BUDGET_BYTES,
    io_workers: int = _DEFAULT_IO_WORKERS,
    read_chunk_bytes: int = _DEFAULT_READ_CHUNK_BYTES,
) -> str | None:
    """Return a precise, pre-mutation shared-host eligibility reason."""
    if not preflight.tensors:
        return "shared_host_producer requires non-empty SafeTensors metadata"
    if node_communicator is None:
        return "shared_host_producer requires a node-local communicator"
    if world_communicator is None:
        return "shared_host_producer requires a world communicator"
    if slot_bytes <= 0:
        return "shared_host_producer slot_bytes must be positive"
    if buffer_budget_bytes <= 0:
        return "shared_host_producer buffer_budget_bytes must be positive"
    if io_workers <= 0:
        return "shared_host_producer io_workers must be positive"
    if read_chunk_bytes <= 0:
        return "shared_host_producer read_chunk_bytes must be positive"
    maximum_alignment = max(_dtype_alignment(tensor.dtype) for tensor in preflight.tensors)
    if slot_bytes % maximum_alignment != 0:
        return (
            "shared_host_producer slot_bytes must be aligned to the largest "
            f"tensor item size ({maximum_alignment} bytes)"
        )
    groups, reason = _validate_groups(preflight, group_manifest, single_tensor_groups_safe)
    if reason is not None:
        return reason
    assert groups is not None
    try:
        _resolve_slot_layout(preflight, groups, slot_bytes, buffer_budget_bytes)
    except ValueError as error:
        return str(error)
    return None


def _plan_batches(
    preflight: SharedHostStreamPreflight,
    groups: tuple[WeightGroup, ...],
    slot_bytes: int,
    read_chunk_bytes: int,
) -> tuple[_PlannedBatch, ...]:
    metadata = {tensor.key: tensor for tensor in preflight.tensors}
    plans = []
    sequence = 0
    for group in groups:
        group_batches: list[tuple[tuple[WeightSegment, ...], tuple[_ReadExtent, ...], int]] = []
        segments = []
        extents = []
        payload_nbytes = 0
        for key in group.keys:
            tensor = metadata[key]
            alignment = _dtype_alignment(tensor.dtype)
            aligned_offset = ((payload_nbytes + alignment - 1) // alignment) * alignment
            if aligned_offset == slot_bytes and payload_nbytes != 0:
                group_batches.append((tuple(segments), tuple(extents), payload_nbytes))
                segments = []
                extents = []
                payload_nbytes = 0
                aligned_offset = 0
            payload_nbytes = aligned_offset
            if tensor.nbytes == 0:
                segments.append(
                    WeightSegment(
                        key=key,
                        dtype=tensor.dtype,
                        shape=tensor.shape,
                        tensor_nbytes=0,
                        tensor_offset=0,
                        payload_offset=payload_nbytes,
                        nbytes=0,
                    )
                )
                continue
            tensor_offset = 0
            while tensor_offset < tensor.nbytes:
                available = slot_bytes - payload_nbytes
                if available == 0:
                    group_batches.append((tuple(segments), tuple(extents), payload_nbytes))
                    segments = []
                    extents = []
                    payload_nbytes = 0
                    available = slot_bytes
                length = min(available, tensor.nbytes - tensor_offset)
                if length < tensor.nbytes - tensor_offset:
                    length -= length % alignment
                if length == 0:
                    group_batches.append((tuple(segments), tuple(extents), payload_nbytes))
                    segments = []
                    extents = []
                    payload_nbytes = 0
                    continue
                segment_offset = payload_nbytes
                segments.append(
                    WeightSegment(
                        key=key,
                        dtype=tensor.dtype,
                        shape=tensor.shape,
                        tensor_nbytes=tensor.nbytes,
                        tensor_offset=tensor_offset,
                        payload_offset=segment_offset,
                        nbytes=length,
                    )
                )
                read_offset = 0
                while read_offset < length:
                    read_length = min(read_chunk_bytes, length - read_offset)
                    extents.append(
                        _ReadExtent(
                            source_path=tensor.source_path,
                            source_offset=(tensor.source_offset + tensor_offset + read_offset),
                            payload_offset=segment_offset + read_offset,
                            nbytes=read_length,
                        )
                    )
                    read_offset += read_length
                tensor_offset += length
                payload_nbytes += length
        if segments:
            group_batches.append((tuple(segments), tuple(extents), payload_nbytes))
        for group_batch_index, (batch_segments, batch_extents, batch_nbytes) in enumerate(
            group_batches
        ):
            batch = WeightBatch(
                sequence=sequence,
                slot=sequence % _NUM_SHARED_SLOTS,
                group_id=group.group_id,
                group_keys=group.keys,
                group_complete=group_batch_index == len(group_batches) - 1,
                segments=batch_segments,
                payload_nbytes=batch_nbytes,
            )
            plans.append(_PlannedBatch(batch=batch, extents=batch_extents))
            sequence += 1
    return tuple(plans)


def _allocate_mpi_shared_window(
    node_communicator: Any, total_bytes: int
) -> tuple[_Window, memoryview]:
    from mpi4py import MPI

    local_bytes = total_bytes if node_communicator.Get_rank() == 0 else 0
    window = MPI.Win.Allocate_shared(local_bytes, 1, comm=node_communicator)
    arena = None
    buffer = None
    setup_error = None
    try:
        buffer, _ = window.Shared_query(0)
        raw_arena = memoryview(buffer).cast("B")
        try:
            arena_nbytes = len(raw_arena)
            if arena_nbytes < total_bytes:
                raise RuntimeError(
                    f"MPI shared window has {arena_nbytes} bytes, expected {total_bytes}"
                )
            arena = raw_arena[:total_bytes]
        finally:
            raw_arena.release()
    except BaseException as error:
        setup_error = error

    setup_state = (
        node_communicator.Get_rank(),
        None if setup_error is None else type(setup_error).__name__,
        None if setup_error is None else str(setup_error),
    )
    setup_states = node_communicator.allgather(setup_state)
    setup_failures = [state for state in setup_states if state[1] is not None]
    if setup_failures:
        if arena is not None:
            arena.release()
        buffer = None
        # Allocate_shared succeeded on every node-local rank, so every rank
        # owns the same collective window even when Shared_query failed only
        # locally. Free it collectively before propagating one stable error.
        window.Free()
        rank, error_type, error_message = min(setup_failures)
        raise RuntimeError(
            "MPI shared-window setup failed on node-local rank "
            f"{rank}: {error_type}: {error_message}"
        ) from setup_error
    assert arena is not None
    return window, arena


def _cuda_result_code(result: Any) -> int:
    if isinstance(result, tuple):
        result = result[0]
    return int(result)


def _try_cuda_host_registration(arena: memoryview) -> HostMemoryRegistration:
    """Best-effort register the shared arena in the process CUDA context."""
    try:
        import torch

        if not torch.cuda.is_available():
            return HostMemoryRegistration(False, "CUDA is unavailable")
        runtime = torch.cuda.cudart()
        address = ctypes.addressof(ctypes.c_char.from_buffer(arena))
        result = runtime.cudaHostRegister(address, len(arena), 0)
        code = _cuda_result_code(result)
        if code != 0:
            return HostMemoryRegistration(False, f"cudaHostRegister returned error code {code}")

        def unregister() -> None:
            unregister_code = _cuda_result_code(runtime.cudaHostUnregister(address))
            if unregister_code != 0:
                raise RuntimeError(f"cudaHostUnregister returned error code {unregister_code}")

        return HostMemoryRegistration(True, "registered with cudaHostRegister", unregister)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        return HostMemoryRegistration(False, f"CUDA host registration unavailable: {error}")


def _preopen_fingerprint(
    preflight: SharedHostStreamPreflight,
    groups: tuple[WeightGroup, ...],
    plans: tuple[_PlannedBatch, ...],
    slot_layout: _SlotLayout,
) -> str:
    descriptor = (
        tuple(
            (tensor.key, tensor.dtype, tensor.shape, tensor.nbytes) for tensor in preflight.tensors
        ),
        groups,
        tuple(plan.batch for plan in plans),
        slot_layout,
    )
    return hashlib.sha256(repr(descriptor).encode("utf-8")).hexdigest()


def _establish_shared_window_epoch(
    node_communicator: Any,
    window: _Window | None,
    setup_error: BaseException | None,
) -> tuple[bool, BaseException | None, bool]:
    """Open and coordinate one passive-target epoch on every node rank."""
    window_locked = False
    epoch_uncertain = False
    if setup_error is None:
        assert window is not None
        try:
            window.Lock_all()
            window_locked = True
        except BaseException as error:
            setup_error = error
            # MPI does not guarantee the local epoch state after an error.
            epoch_uncertain = True

    rank = node_communicator.Get_rank()
    local_state = (
        rank,
        window is not None,
        window_locked,
        epoch_uncertain,
        None if setup_error is None else type(setup_error).__name__,
        None if setup_error is None else str(setup_error),
    )
    try:
        states = node_communicator.allgather(local_state)
    except BaseException as error:
        return window_locked, error, window is not None

    ranks_with_windows = [rank for rank, owns_window, _, _, _, _ in states if owns_window]
    locked_ranks = [rank for rank, owns_window, locked, _, _, _ in states if owns_window and locked]
    failures = [state for state in states if state[4] is not None]
    quarantine_required = (
        any(state[3] for state in states)
        or (bool(ranks_with_windows) and len(ranks_with_windows) != len(states))
        or (bool(locked_ranks) and locked_ranks != ranks_with_windows)
    )
    if ranks_with_windows and len(ranks_with_windows) != len(states):
        setup_error = RuntimeError(
            "Shared-window factory returned ownership on only node-local "
            f"ranks {ranks_with_windows}"
        )
    elif failures:
        failure_rank, _, _, _, error_type, error_message = min(failures, key=lambda state: state[0])
        setup_error = RuntimeError(
            "Shared-window epoch setup failed on node-local rank "
            f"{failure_rank}: {error_type}: {error_message}"
        )
    elif locked_ranks != ranks_with_windows:
        setup_error = RuntimeError(
            "Shared-window epoch did not open on every node-local window owner"
        )
        quarantine_required = True
    else:
        setup_error = None
    return window_locked, setup_error, quarantine_required


def _cleanup_setup_resources(
    node_communicator: Any,
    registration: HostMemoryRegistration | None,
    arena: memoryview | None,
    window: _Window | None,
    *,
    window_locked: bool,
) -> BaseException | None:
    """Release every setup resource while retaining the first cleanup error."""
    safe_to_free, cleanup_error = _prepare_shared_window_teardown(
        node_communicator,
        registration,
        arena,
        window,
        window_locked=window_locked,
    )
    if safe_to_free and window is not None:
        try:
            window.Free()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
    return cleanup_error


def _first_world_error(
    world_communicator: Any,
    error: BaseException | None,
) -> tuple[int, str, str] | None:
    """Return one deterministic rank error after every world rank reports."""
    state = (
        world_communicator.Get_rank(),
        None if error is None else type(error).__name__,
        None if error is None else str(error),
    )
    states = world_communicator.allgather(state)
    failures = [candidate for candidate in states if candidate[1] is not None]
    if not failures:
        return None
    rank, error_type, error_message = min(failures, key=lambda candidate: candidate[0])
    assert error_type is not None
    assert error_message is not None
    return rank, error_type, error_message


def _prepare_shared_window_teardown(
    node_communicator: Any,
    registration: HostMemoryRegistration | None,
    arena: memoryview | None,
    window: _Window | None,
    *,
    window_locked: bool,
) -> tuple[bool, BaseException | None]:
    """Coordinate whether a node-local shared window is safe to free.

    Host registration and arena views are process-local, but the MPI window
    must be freed collectively. If either process-local release fails, every
    rank on the node retains its remaining handles and skips ``Win.Free``.
    """
    local_error = None
    if registration is not None:
        try:
            registration.close()
        except BaseException as error:
            local_error = error
    if arena is not None:
        try:
            arena.release()
        except BaseException as error:
            if local_error is None:
                local_error = error

    rank = node_communicator.Get_rank()
    local_state = (
        rank,
        window is not None,
        window_locked,
        None if local_error is None else type(local_error).__name__,
        None if local_error is None else str(local_error),
    )
    try:
        states = node_communicator.allgather(local_state)
    except BaseException as error:
        _quarantine_shared_resources(registration, arena, window)
        return False, error

    resource_failures = [state for state in states if state[3] is not None]
    ranks_with_windows = [
        candidate_rank for candidate_rank, has_window, _, _, _ in states if has_window
    ]
    locked_ranks = [
        candidate_rank
        for candidate_rank, has_window, locked, _, _ in states
        if has_window and locked
    ]
    cleanup_error = None
    if resource_failures:
        failure_rank, _, _, error_type, error_message = min(
            resource_failures, key=lambda state: state[0]
        )
        cleanup_error = RuntimeError(
            "Shared-host resource teardown failed on node-local rank "
            f"{failure_rank}: {error_type}: {error_message}; "
            "skipping collective Win.Free"
        )
    elif ranks_with_windows and len(ranks_with_windows) != len(states):
        cleanup_error = RuntimeError(
            "Shared-window factory returned ownership on only node-local "
            f"ranks {ranks_with_windows}; skipping collective Win.Free"
        )
    elif locked_ranks and locked_ranks != ranks_with_windows:
        cleanup_error = RuntimeError(
            "Shared-window epoch is active on only node-local ranks "
            f"{locked_ranks}; skipping collective Win.Free"
        )

    if cleanup_error is not None:
        _quarantine_shared_resources(registration, arena, window)
        return False, cleanup_error

    unlock_error = None
    if window is not None and window_locked:
        try:
            window.Unlock_all()
        except BaseException as error:
            unlock_error = error
    unlock_state = (
        rank,
        None if unlock_error is None else type(unlock_error).__name__,
        None if unlock_error is None else str(unlock_error),
    )
    try:
        unlock_states = node_communicator.allgather(unlock_state)
    except BaseException as error:
        _quarantine_shared_resources(registration, arena, window)
        return False, error

    unlock_failures = [state for state in unlock_states if state[1] is not None]
    if unlock_failures:
        failure_rank, error_type, error_message = min(unlock_failures, key=lambda state: state[0])
        cleanup_error = RuntimeError(
            "Shared-window epoch teardown failed on node-local rank "
            f"{failure_rank}: {error_type}: {error_message}; "
            "skipping collective Win.Free"
        )
        _quarantine_shared_resources(registration, arena, window)
        return False, cleanup_error
    return True, None


def open_shared_host_weight_stream(
    preflight: SharedHostStreamPreflight,
    node_communicator: Any,
    world_communicator: Any,
    *,
    group_manifest: Sequence[WeightGroup] | None = None,
    single_tensor_groups_safe: bool = False,
    slot_bytes: int = _DEFAULT_SLOT_BYTES,
    buffer_budget_bytes: int = _DEFAULT_BUFFER_BUDGET_BYTES,
    io_workers: int = _DEFAULT_IO_WORKERS,
    read_chunk_bytes: int = _DEFAULT_READ_CHUNK_BYTES,
    strict: bool = False,
    strict_host_registration: bool = False,
    window_factory: WindowFactory | None = None,
    reader: RangeReader | None = None,
    host_registrar: HostRegistrar | None = None,
) -> SharedHostWeightStream | None:
    """Open a bounded shared-host stream after policy selection.

    Soft mode returns ``None`` for preflight ineligibility. Strict mode raises
    :class:`SharedHostStreamUnavailableError`. Failures after the shared
    transport starts are never converted to a legacy fallback.
    """
    try:
        reason = get_shared_host_stream_ineligibility_reason(
            preflight,
            group_manifest=group_manifest,
            single_tensor_groups_safe=single_tensor_groups_safe,
            node_communicator=node_communicator,
            world_communicator=world_communicator,
            slot_bytes=slot_bytes,
            buffer_budget_bytes=buffer_budget_bytes,
            io_workers=io_workers,
            read_chunk_bytes=read_chunk_bytes,
        )
    except BaseException as error:
        reason = f"shared_host_producer eligibility check failed: {type(error).__name__}: {error}"
    if world_communicator is not None:
        reasons = world_communicator.allgather((world_communicator.Get_rank(), reason))
        reason = next(
            (
                candidate_reason
                for _, candidate_reason in sorted(reasons)
                if candidate_reason is not None
            ),
            None,
        )
    if reason is not None:
        if strict:
            raise SharedHostStreamUnavailableError(reason)
        return None
    groups = None
    plans = None
    slot_layout = None
    fingerprint = None
    planning_error = None
    try:
        groups, _ = _validate_groups(preflight, group_manifest, single_tensor_groups_safe)
        assert groups is not None
        slot_layout = _resolve_slot_layout(preflight, groups, slot_bytes, buffer_budget_bytes)
        plans = _plan_batches(preflight, groups, slot_layout.slot_bytes, read_chunk_bytes)
        fingerprint = _preopen_fingerprint(preflight, groups, plans, slot_layout)
    except BaseException as error:
        planning_error = error
    planning_state = (
        world_communicator.Get_rank(),
        (None if planning_error is None else type(planning_error).__name__),
        (None if planning_error is None else str(planning_error)),
        fingerprint,
    )
    planning_states = world_communicator.allgather(planning_state)
    planning_failures = [state for state in planning_states if state[1] is not None]
    if planning_failures:
        rank, error_type, error_message, _ = min(planning_failures)
        reason = (
            f"shared_host_producer planning failed on world rank {rank}: "
            f"{error_type}: {error_message}"
        )
        if strict:
            raise SharedHostStreamUnavailableError(reason) from planning_error
        return None
    fingerprints = {state[3] for state in planning_states}
    if len(fingerprints) != 1:
        reason = "shared_host_producer metadata or group manifests differ across ranks"
        if strict:
            raise SharedHostStreamUnavailableError(reason)
        return None
    assert groups is not None
    assert plans is not None
    assert slot_layout is not None

    factory = window_factory or _allocate_mpi_shared_window
    total_bytes = _NUM_SHARED_SLOTS * slot_layout.slot_bytes
    window = None
    arena = None
    registration = None
    window_locked = False
    setup_quarantine_required = False
    setup_error = None
    try:
        window, arena = factory(node_communicator, total_bytes)
        if len(arena) < total_bytes:
            raise RuntimeError(f"Shared arena has {len(arena)} bytes, expected {total_bytes}")
    except BaseException as error:
        setup_error = error

    window_locked, setup_error, setup_quarantine_required = _establish_shared_window_epoch(
        node_communicator, window, setup_error
    )
    if setup_error is None:
        assert arena is not None
        try:
            registrar = host_registrar or _try_cuda_host_registration
            registration = registrar(arena)
        except BaseException as error:
            setup_error = error

    local_registration_state = (
        node_communicator.Get_rank(),
        (registration.registered if registration is not None else False),
        (registration.detail if registration is not None else "registration did not run"),
    )
    local_registration_states = None
    try:
        local_registration_states = node_communicator.allgather(local_registration_state)
    except BaseException as error:
        if setup_error is None:
            setup_error = error

    setup_state = (
        world_communicator.Get_rank(),
        (None if setup_error is None else type(setup_error).__name__),
        (None if setup_error is None else str(setup_error)),
        (registration.registered if registration is not None else False),
        (registration.detail if registration is not None else "registration did not run"),
    )
    setup_states = world_communicator.allgather(setup_state)
    failed_states = [state for state in setup_states if state[1] is not None]
    registration_failures = [state for state in setup_states if not state[3]]
    local_registration_failures = (
        [state for state in local_registration_states if not state[1]]
        if local_registration_states is not None
        else [local_registration_state]
    )
    setup_reason = None
    if failed_states:
        rank, error_type, error_message, _, _ = min(failed_states)
        setup_reason = (
            f"shared_host_producer setup failed on world rank {rank}: {error_type}: {error_message}"
        )
    elif strict_host_registration and registration_failures:
        failures = ", ".join(f"rank {state[0]}: {state[4]}" for state in registration_failures)
        setup_reason = "CUDA host registration is required but unavailable: " + failures

    if setup_reason is not None:
        if setup_quarantine_required:
            _quarantine_shared_resources(registration, arena, window)
            cleanup_error = RuntimeError(
                "Shared-window epoch setup was quarantined; skipping "
                "Win.Unlock_all and collective Win.Free"
            )
        else:
            cleanup_error = _cleanup_setup_resources(
                node_communicator,
                registration,
                arena,
                window,
                window_locked=window_locked,
            )
        cleanup_failure = _first_world_error(world_communicator, cleanup_error)
        if cleanup_failure is not None:
            rank, error_type, error_message = cleanup_failure
            raise SharedHostStreamError(
                f"{setup_reason}; cleanup failed on world rank {rank}: "
                f"{error_type}: {error_message}"
            ) from setup_error
        if strict or strict_host_registration:
            raise SharedHostStreamUnavailableError(setup_reason) from setup_error
        return None

    assert window is not None
    assert arena is not None
    assert registration is not None
    assert window_locked
    range_reader = reader or _PreadRangeReader()
    executor = None
    stream = None
    construction_error = None
    try:
        if node_communicator.Get_rank() == 0:
            executor = ThreadPoolExecutor(
                max_workers=io_workers,
                thread_name_prefix="trtllm-shared-host-reader",
            )
        stream = SharedHostWeightStream(
            plans,
            groups,
            node_communicator,
            world_communicator,
            window,
            arena,
            registration,
            window_locked=window_locked,
            slot_layout=slot_layout,
            # Direct host-to-device views require registration on every rank
            # sharing this arena, not on unrelated nodes. A failure on one
            # node therefore stages that node without disabling direct views
            # on independently registered peers elsewhere in the world.
            all_ranks_host_registered=not local_registration_failures,
            io_workers=io_workers,
            read_chunk_bytes=read_chunk_bytes,
            reader=range_reader,
            executor=executor,
        )
    except BaseException as error:
        construction_error = error

    construction_failure = _first_world_error(world_communicator, construction_error)
    if construction_failure is None:
        assert stream is not None
        return stream

    rank, error_type, error_message = construction_failure
    construction_reason = (
        "shared_host_producer construction failed on world rank "
        f"{rank}: {error_type}: {error_message}"
    )
    cleanup_error = None
    if executor is not None:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            cleanup_error = error
        executor = None
        if stream is not None:
            stream._executor = None
    close_reader = getattr(range_reader, "close", None)
    if close_reader is not None:
        try:
            close_reader()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error

    resource_cleanup_error = _cleanup_setup_resources(
        node_communicator,
        registration,
        arena,
        window,
        window_locked=window_locked,
    )
    if cleanup_error is None:
        cleanup_error = resource_cleanup_error
    cleanup_failure = _first_world_error(world_communicator, cleanup_error)
    if cleanup_failure is not None:
        cleanup_rank, cleanup_type, cleanup_message = cleanup_failure
        raise SharedHostStreamError(
            f"{construction_reason}; cleanup failed on world rank "
            f"{cleanup_rank}: {cleanup_type}: {cleanup_message}"
        ) from construction_error
    raise SharedHostStreamError(construction_reason) from construction_error
