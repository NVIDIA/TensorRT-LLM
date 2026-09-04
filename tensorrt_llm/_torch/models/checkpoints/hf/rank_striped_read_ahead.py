# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private host read-ahead primitives for Hugging Face SafeTensors."""

import os
import threading
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import psutil

# Work-assignment granularity across ranks; one extent stays with one issuer.
_CHUNK_SIZE = 256 * 1024 * 1024
# I/O and cancellation granularity within an extent; one pread reads this much.
_READ_SIZE = 8 * 1024 * 1024
# Maximum reader concurrency contributed by one node-local rank.
_WORKERS_PER_RANK = 16
# Maximum aggregate reader concurrency across the node-local load group.
_WORKERS_PER_LOAD_GROUP = 64
# Experimental rolling mode: cap the sum of process-local issuer leads over a
# node-local load group. A lead is that issuer's completed reads minus the
# consumption reported by its local foreground loader. Replicated ranks may
# report the same source tensor independently, so this is deliberately not a
# unique-source-byte or Linux page-cache-residency bound. A later shared
# WeightLoadPlan can give source extents unique credit ownership.
_ROLLING_INITIAL_LEAD_BYTES_PER_LOAD_GROUP = 16 * 1024 * 1024 * 1024
_PENDING_EXTENTS_PER_WORKER = 2
_HOST_MEMORY_HEADROOM_BYTES = 16 * 1024 * 1024 * 1024
_HOST_MEMORY_HEADROOM_FRACTION = 0.1


@dataclass(frozen=True)
class ReadAheadExtent:
    path: str
    offset: int
    length: int


@dataclass(frozen=True)
class ReadAheadPlan:
    extents: tuple[ReadAheadExtent, ...]
    workers: int
    initial_issuer_lead_bytes: int | None = None

    @property
    def assigned_bytes(self) -> int:
        return sum(extent.length for extent in self.extents)


@dataclass(frozen=True)
class ReadAheadProgress:
    issued_bytes: int
    consumed_bytes: int
    consumed_items: int
    sized_items: int
    max_issuer_lead_bytes: int
    credit_wait_seconds: float
    submitted_extents: int
    completed_extents: int
    partial_extents: int
    cancelled_extents: int
    max_pending_extents: int

    @property
    def consumption_coverage(self) -> float:
        if self.consumed_items == 0:
            return 0.0
        return self.sized_items / self.consumed_items


def coordinate_error(communicator, phase: str, error: BaseException | None) -> RuntimeError | None:
    """Return the first rank error after one active-group consensus."""
    message = None if error is None else f"{type(error).__name__}: {error}"
    messages = (
        [message]
        if communicator is None or communicator.Get_size() == 1
        else communicator.allgather(message)
    )
    for rank, rank_error in enumerate(messages):
        if rank_error is not None:
            return RuntimeError(f"Rank {rank} failed during {phase}: {rank_error}")
    return None


def distribute_worker_budget(local_size: int) -> tuple[int, ...]:
    """Distribute an exact issuer budget as evenly as integer workers allow."""
    if local_size <= 0:
        raise ValueError("local_size must be positive")
    budget = min(_WORKERS_PER_LOAD_GROUP, local_size * _WORKERS_PER_RANK)
    workers, remainder = divmod(budget, local_size)
    return tuple(workers + (rank < remainder) for rank in range(local_size))


def distribute_initial_lead_budget(
    initial_lead_bytes: int, local_size: int, extent_count: int
) -> tuple[int, ...]:
    """Distribute one load group's initial lead across active issuers."""
    if initial_lead_bytes < 0:
        raise ValueError("initial_lead_bytes must be nonnegative")
    if extent_count < 0:
        raise ValueError("extent_count must be nonnegative")
    worker_counts = distribute_worker_budget(local_size)
    issuer_ranks = [rank for rank, workers in enumerate(worker_counts) if workers > 0]
    # Each active issuer needs at least one normal I/O reservation; otherwise
    # tiny per-rank quotas fragment every pread. A sub-chunk checkpoint still
    # uses one issuer with its exact smaller budget.
    budgeted_issuers = (
        0
        if initial_lead_bytes == 0
        else max(1, initial_lead_bytes // min(_READ_SIZE, initial_lead_bytes))
    )
    issuer_ranks = issuer_ranks[: min(extent_count, budgeted_issuers)]
    budgets = [0] * local_size
    if not issuer_ranks:
        return tuple(budgets)
    per_issuer, remainder = divmod(initial_lead_bytes, len(issuer_ranks))
    for index, rank in enumerate(issuer_ranks):
        budgets[rank] = per_issuer + (index < remainder)
    return tuple(budgets)


def rolling_issuer_lead_admission(
    checkpoint_bytes: int, available_bytes: int
) -> tuple[bool, int, int]:
    """Return admission, headroom, and load-group initial issuer lead."""
    headroom = max(
        _HOST_MEMORY_HEADROOM_BYTES, int(available_bytes * _HOST_MEMORY_HEADROOM_FRACTION)
    )
    usable_bytes = max(0, available_bytes - headroom)
    initial_lead_bytes = min(
        checkpoint_bytes,
        usable_bytes,
        _ROLLING_INITIAL_LEAD_BYTES_PER_LOAD_GROUP,
    )
    return initial_lead_bytes > 0, headroom, initial_lead_bytes


def _resolve_extent_order(
    source_extents: Sequence[ReadAheadExtent],
    ordered_extents: Sequence[ReadAheadExtent] | None,
) -> list[ReadAheadExtent]:
    """Apply the phase-one ordering hook without changing read coverage.

    The catalog/plan stack will replace this compiler with arbitrary,
    page-aligned/coalesced tensor ranges. Until then, only a full permutation
    of the existing file chunks is accepted: no destination filtering or
    selective skipping is implied by this experimental policy.
    """
    if ordered_extents is None:
        return list(source_extents)
    if Counter(ordered_extents) != Counter(source_extents):
        raise ValueError("ordered_extents must be a complete permutation of checkpoint extents")
    return list(ordered_extents)


def build_local_plan(
    files: Sequence[tuple[str, int]],
    local_rank: int,
    local_size: int,
    *,
    issuer_group_initial_lead_bytes: int | None = None,
    ordered_extents: Sequence[ReadAheadExtent] | None = None,
) -> ReadAheadPlan:
    """Assign every checkpoint extent to exactly one node-local issuer."""
    if not 0 <= local_rank < local_size:
        raise ValueError("local rank and size must describe a valid group")

    extents = []
    for path, file_size in sorted(files):
        if file_size < 0:
            raise ValueError("checkpoint file sizes must be nonnegative")
        for offset in range(0, file_size, _CHUNK_SIZE):
            extents.append(ReadAheadExtent(path, offset, min(_CHUNK_SIZE, file_size - offset)))

    extents = _resolve_extent_order(extents, ordered_extents)

    worker_counts = distribute_worker_budget(local_size)
    issuer_ranks = [rank for rank, workers in enumerate(worker_counts) if workers > 0]
    local_budgets = None
    if issuer_group_initial_lead_bytes is not None:
        local_budgets = distribute_initial_lead_budget(
            issuer_group_initial_lead_bytes, local_size, len(extents)
        )
        issuer_ranks = [rank for rank in issuer_ranks if local_budgets[rank] > 0]
    if not extents or local_rank not in issuer_ranks:
        return ReadAheadPlan(
            (),
            0,
            0 if issuer_group_initial_lead_bytes is not None else None,
        )

    issuer_index = issuer_ranks.index(local_rank)
    local_extents = tuple(extents[issuer_index :: len(issuer_ranks)])
    workers = min(worker_counts[local_rank], len(local_extents))
    local_budget = None if local_budgets is None else local_budgets[local_rank]
    return ReadAheadPlan(local_extents, workers, local_budget)


def _safe_relative_path(path: str) -> Path:
    return Path(*[part for part in Path(path.lstrip("/")).parts if part not in (".", "..")])


def _cgroup_paths() -> tuple[Path | None, Path | None]:
    unified_path = None
    memory_path = None
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except (OSError, UnicodeError):
        return unified_path, memory_path

    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, relative_path = fields
        if hierarchy == "0" and not controllers:
            unified_path = _safe_relative_path(relative_path)
        if "memory" in controllers.split(","):
            memory_path = _safe_relative_path(relative_path)
    return unified_path, memory_path


def _walk_to_mount_root(path: Path, mount_root: Path):
    current = path
    while current == mount_root or mount_root in current.parents:
        yield current
        if current == mount_root:
            return
        current = current.parent


def _read_cgroup_value(path: Path, *, unlimited_threshold: int | None = None) -> int | None:
    try:
        value = path.read_text().strip()
        if value == "max":
            return None
        parsed = int(value)
    except (OSError, UnicodeError, ValueError):
        return None
    if unlimited_threshold is not None and parsed >= unlimited_threshold:
        return None
    return parsed


def cgroup_available_host_memory() -> int | None:
    """Return the tightest remaining cgroup limit, including ancestors."""
    unified_path, memory_path = _cgroup_paths()
    candidates: list[tuple[Path, Path, bool]] = []

    unified_root = Path("/sys/fs/cgroup")
    if unified_path is not None:
        candidates.append((unified_root / unified_path, unified_root, True))
    candidates.append((unified_root, unified_root, True))

    memory_root = Path("/sys/fs/cgroup/memory")
    if memory_path is not None:
        candidates.append((memory_root / memory_path, memory_root, False))
    # A cgroup namespace may expose the process cgroup directly at the mount
    # root even when /proc/self/cgroup reports its host-relative path.
    candidates.append((memory_root, memory_root, False))

    available = []
    seen = set()
    for leaf, mount_root, unified in candidates:
        for path in _walk_to_mount_root(leaf, mount_root):
            key = (path, unified)
            if key in seen:
                continue
            seen.add(key)
            if unified:
                current = _read_cgroup_value(path / "memory.current")
                limits = (
                    _read_cgroup_value(path / "memory.max"),
                    _read_cgroup_value(path / "memory.high"),
                )
            else:
                current = _read_cgroup_value(path / "memory.usage_in_bytes")
                limits = (
                    _read_cgroup_value(path / "memory.limit_in_bytes", unlimited_threshold=1 << 60),
                )
            if current is None:
                continue
            available.extend(max(0, limit - current) for limit in limits if limit is not None)

    return min(available) if available else None


def effective_available_host_memory() -> int:
    host_available = psutil.virtual_memory().available
    cgroup_available = cgroup_available_host_memory()
    return host_available if cgroup_available is None else min(host_available, cgroup_available)


def memory_admission(checkpoint_bytes: int, available_bytes: int) -> tuple[bool, int]:
    """Reserve startup headroom before admitting a full-file read-ahead."""
    headroom = max(
        _HOST_MEMORY_HEADROOM_BYTES, int(available_bytes * _HOST_MEMORY_HEADROOM_FRACTION)
    )
    return checkpoint_bytes <= max(0, available_bytes - headroom), headroom


def close_node_communicator(node_communicator) -> None:
    if node_communicator is not None:
        node_communicator.Free()


class _RollingReadAheadCredits:
    """Bound one issuer's reads relative to its local consumption report."""

    def __init__(self, initial_lead_bytes: int, assigned_bytes: int) -> None:
        if initial_lead_bytes <= 0:
            raise ValueError("initial_lead_bytes must be positive")
        self._initial_lead_bytes = initial_lead_bytes
        self._assigned_bytes = assigned_bytes
        self._issued_bytes = 0
        self._inflight_bytes = 0
        self._consumed_bytes = 0
        self._consumed_items = 0
        self._sized_items = 0
        self._max_issuer_lead_bytes = 0
        self._credit_wait_seconds = 0.0
        self._cancelled = False
        self._condition = threading.Condition()

    def acquire(self, requested_bytes: int) -> int:
        """Reserve up to ``requested_bytes``, blocking until credit exists."""
        wait_started = None
        with self._condition:
            while True:
                if self._cancelled:
                    if wait_started is not None:
                        self._credit_wait_seconds += time.monotonic() - wait_started
                    return 0
                remaining = self._assigned_bytes - self._issued_bytes - self._inflight_bytes
                if remaining <= 0:
                    if wait_started is not None:
                        self._credit_wait_seconds += time.monotonic() - wait_started
                    return 0
                available = (
                    self._initial_lead_bytes
                    + self._consumed_bytes
                    - self._issued_bytes
                    - self._inflight_bytes
                )
                if available > 0:
                    if wait_started is not None:
                        self._credit_wait_seconds += time.monotonic() - wait_started
                    granted = min(requested_bytes, remaining, available)
                    self._inflight_bytes += granted
                    return granted
                if wait_started is None:
                    wait_started = time.monotonic()
                self._condition.wait()

    def complete(self, reserved_bytes: int, issued_bytes: int) -> None:
        """Commit actual bytes from one read and refund a short read."""
        if not 0 <= issued_bytes <= reserved_bytes:
            raise ValueError("issued_bytes must be within the reservation")
        with self._condition:
            self._inflight_bytes -= reserved_bytes
            self._issued_bytes += issued_bytes
            self._max_issuer_lead_bytes = max(
                self._max_issuer_lead_bytes,
                self._issued_bytes - self._consumed_bytes,
            )
            self._condition.notify_all()

    def report_consumed(self, consumed_bytes: int, consumed_items: int, sized_items: int) -> None:
        """Advance foreground progress and wake blocked readers."""
        if consumed_bytes < 0 or not 0 <= sized_items <= consumed_items:
            raise ValueError("invalid consumption report")
        with self._condition:
            if self._cancelled:
                return
            self._consumed_bytes += consumed_bytes
            self._consumed_items += consumed_items
            self._sized_items += sized_items
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def snapshot(
        self,
        *,
        submitted_extents: int,
        completed_extents: int,
        partial_extents: int,
        cancelled_extents: int,
        max_pending_extents: int,
    ) -> ReadAheadProgress:
        with self._condition:
            return ReadAheadProgress(
                issued_bytes=self._issued_bytes,
                consumed_bytes=self._consumed_bytes,
                consumed_items=self._consumed_items,
                sized_items=self._sized_items,
                max_issuer_lead_bytes=self._max_issuer_lead_bytes,
                credit_wait_seconds=self._credit_wait_seconds,
                submitted_extents=submitted_extents,
                completed_extents=completed_extents,
                partial_extents=partial_extents,
                cancelled_extents=cancelled_extents,
                max_pending_extents=max_pending_extents,
            )


class RankStripedReadAheadSession:
    """Own background POSIX reads and their node-local communicator."""

    def __init__(self, active_communicator, node_communicator, plan: ReadAheadPlan) -> None:
        self._active_communicator = active_communicator
        self._node_communicator = node_communicator
        self._plan = plan
        self._cancel = threading.Event()
        self._read_release = threading.Event()
        self._thread: threading.Thread | None = None
        self._file_descriptors: dict[str, int] = {}
        self._read_error: BaseException | None = None
        self._closed = False
        self._started = False
        self._enabled = True
        self._submitted_extents = 0
        self._completed_extents = 0
        self._partial_extents = 0
        self._cancelled_extents = 0
        self._max_pending_extents = 0
        self._credits = (
            None
            if plan.initial_issuer_lead_bytes is None or not plan.extents
            else _RollingReadAheadCredits(plan.initial_issuer_lead_bytes, plan.assigned_bytes)
        )

        try:
            for path in {extent.path for extent in plan.extents}:
                self._file_descriptors[path] = os.open(path, os.O_RDONLY)
        except Exception:
            self._close_file_descriptors()
            raise

    def start(self, *, defer_reads: bool = False) -> "RankStripedReadAheadSession":
        if self._started:
            return self
        if not defer_reads:
            self._read_release.set()
        if not self._plan.extents:
            self._started = True
            return self
        self._thread = threading.Thread(
            target=self._run, name="trtllm-rank-striped-read-ahead", daemon=True
        )
        try:
            self._thread.start()
            self._started = True
        except Exception:
            self._thread = None
            self._close_file_descriptors()
            raise
        return self

    def release_reads(self) -> None:
        """Release prepared workers only after activation consensus."""
        self._read_release.set()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._started

    @property
    def active_communicator(self):
        return self._active_communicator

    def disable(self) -> None:
        """Mark a collectively rejected activation as advisory-native."""
        self._enabled = False

    def _read_extent(self, extent: ReadAheadExtent) -> int:
        file_descriptor = self._file_descriptors[extent.path]
        offset = extent.offset
        remaining = extent.length
        completed = 0
        while remaining > 0 and not self._cancel.is_set():
            requested_size = min(remaining, _READ_SIZE)
            read_size = (
                requested_size if self._credits is None else self._credits.acquire(requested_size)
            )
            if read_size == 0:
                break
            try:
                data = os.pread(file_descriptor, read_size, offset)
            except OSError:
                if self._credits is not None:
                    self._credits.complete(read_size, 0)
                raise
            bytes_read = len(data)
            if self._credits is not None:
                self._credits.complete(read_size, bytes_read)
            if not data:
                raise OSError(
                    f"Unexpected EOF while reading {extent.path} at byte offset {offset}."
                )
            offset += bytes_read
            remaining -= bytes_read
            completed += bytes_read
        return completed

    def report_consumed(self, consumed_bytes: int, consumed_items: int, sized_items: int) -> None:
        """Receive process-local progress; source bytes are not deduplicated."""
        if self._credits is not None:
            self._credits.report_consumed(consumed_bytes, consumed_items, sized_items)

    @property
    def progress(self) -> ReadAheadProgress | None:
        if self._credits is None:
            return None
        return self._credits.snapshot(
            submitted_extents=self._submitted_extents,
            completed_extents=self._completed_extents,
            partial_extents=self._partial_extents,
            cancelled_extents=self._cancelled_extents,
            max_pending_extents=self._max_pending_extents,
        )

    def _request_cancel(self) -> None:
        self._cancel.set()
        # Wake a prepared worker that is still behind the activation gate. It
        # observes cancellation before issuing any payload I/O.
        self._read_release.set()
        if self._credits is not None:
            self._credits.cancel()

    def _submit_bounded(
        self,
        executor: ThreadPoolExecutor,
        extent_iterator: Iterator[ReadAheadExtent],
        pending: dict[Future[int], ReadAheadExtent],
    ) -> bool:
        pending_limit = max(1, self._plan.workers * _PENDING_EXTENTS_PER_WORKER)
        exhausted = False
        while len(pending) < pending_limit and not self._cancel.is_set():
            try:
                extent = next(extent_iterator)
            except StopIteration:
                exhausted = True
                break
            pending[executor.submit(self._read_extent, extent)] = extent
            self._submitted_extents += 1
            self._max_pending_extents = max(self._max_pending_extents, len(pending))
        return exhausted

    def _record_future(self, future: Future[int], extent: ReadAheadExtent) -> None:
        try:
            completed_bytes = future.result()
        except CancelledError:
            self._cancelled_extents += 1
            return
        if completed_bytes == extent.length:
            self._completed_extents += 1
        else:
            self._partial_extents += 1

    def _run(self) -> None:
        try:
            self._read_release.wait()
            if self._cancel.is_set():
                return
            with ThreadPoolExecutor(max_workers=self._plan.workers) as executor:
                if self._credits is None:
                    futures = [
                        executor.submit(self._read_extent, extent) for extent in self._plan.extents
                    ]
                    try:
                        for future in as_completed(futures):
                            if self._cancel.is_set():
                                for pending_future in futures:
                                    pending_future.cancel()
                                break
                            future.result()
                    except Exception:
                        self._request_cancel()
                        for future in futures:
                            future.cancel()
                        raise
                    return

                extent_iterator = iter(self._plan.extents)
                pending: dict[Future[int], ReadAheadExtent] = {}
                exhausted = self._submit_bounded(executor, extent_iterator, pending)
                try:
                    while pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            extent = pending.pop(future)
                            self._record_future(future, extent)
                        if self._cancel.is_set():
                            for future in pending:
                                future.cancel()
                            for future, extent in pending.items():
                                self._record_future(future, extent)
                            pending.clear()
                            break
                        if not exhausted:
                            exhausted = self._submit_bounded(executor, extent_iterator, pending)
                except Exception:
                    self._request_cancel()
                    for future in pending:
                        future.cancel()
                    raise
        except Exception as error:
            self._read_error = error

    def _close_file_descriptors(self) -> BaseException | None:
        first_error = None
        for file_descriptor in self._file_descriptors.values():
            try:
                os.close(file_descriptor)
            except OSError as error:
                if first_error is None:
                    first_error = error
        self._file_descriptors.clear()
        return first_error

    def _close_resources(self) -> BaseException | None:
        if self._closed:
            return None
        self._closed = True
        first_error = self._close_file_descriptors()
        try:
            close_node_communicator(self._node_communicator)
        except Exception as error:
            if first_error is None:
                first_error = error
        return first_error

    def cancel_and_close(self) -> BaseException | None:
        if self._closed:
            return None
        first_error = self.cancel_reads()
        try:
            close_node_communicator(self._node_communicator)
        except Exception as error:
            if first_error is None:
                first_error = error
        self._closed = True
        return first_error

    def cancel_reads(self) -> BaseException | None:
        """Stop background work while leaving the node communicator usable."""
        self._request_cancel()
        if self._thread is not None:
            self._thread.join()
        return self._close_file_descriptors()

    def finish(self, body_error: BaseException | None) -> RuntimeError | None:
        """Stop advisory work once every rank finishes materialization."""
        coordinated_body_error = coordinate_error(
            self._active_communicator, "rank-striped model materialization", body_error
        )

        # Once the slowest rank has materialized its weights, further reads
        # cannot improve first-token latency. Bound the remaining I/O volume
        # to the currently executing small pread on each worker; synchronous
        # storage latency itself is not cancellable here.
        self._request_cancel()
        if self._thread is not None:
            self._thread.join()

        coordinated_read_error = coordinate_error(
            self._active_communicator, "rank-striped background read-ahead", self._read_error
        )
        cleanup_error = self._close_resources()
        coordinated_cleanup_error = coordinate_error(
            self._active_communicator, "rank-striped cleanup", cleanup_error
        )

        if coordinated_body_error is not None:
            raise coordinated_body_error
        if coordinated_cleanup_error is not None:
            raise coordinated_cleanup_error
        return coordinated_read_error
