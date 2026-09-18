# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private host read-ahead primitives for Hugging Face SafeTensors."""

import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Sequence

import psutil

from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import CheckpointCatalog
from tensorrt_llm._torch.models.checkpoints.weight_load_plan import (
    WeightDemand,
    WeightLoadOrderConfidence,
    WeightLoadPlan,
)

# Work-assignment granularity across ranks; one extent stays with one issuer.
_CHUNK_SIZE = 256 * 1024 * 1024
# I/O and cancellation granularity within an extent; one pread reads this much.
_READ_SIZE = 8 * 1024 * 1024
# Maximum reader concurrency contributed by one node-local rank.
_WORKERS_PER_RANK = 16
# Maximum aggregate reader concurrency across the node-local load group.
_WORKERS_PER_LOAD_GROUP = 64
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

    @property
    def assigned_bytes(self) -> int:
        return sum(extent.length for extent in self.extents)


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


def _build_source_extents(
    files: Sequence[tuple[str, int]],
) -> list[ReadAheadExtent]:
    extents = []
    for path, file_size in sorted(files):
        if file_size < 0:
            raise ValueError("checkpoint file sizes must be nonnegative")
        for offset in range(0, file_size, _CHUNK_SIZE):
            extents.append(
                ReadAheadExtent(
                    path,
                    offset,
                    min(_CHUNK_SIZE, file_size - offset),
                )
            )
    return extents


def _topological_demands(plan: WeightLoadPlan) -> tuple[WeightDemand, ...]:
    """Return a stable priority-aware topological order for one rank plan."""
    demand_by_id = {demand.group_id: demand for demand in plan.demands}
    successors = {group_id: [] for group_id in demand_by_id}
    remaining_predecessors = {}
    for demand in plan.demands:
        remaining_predecessors[demand.group_id] = len(demand.predecessors)
        for predecessor in demand.predecessors:
            successors[predecessor].append(demand.group_id)

    ready = [
        (
            demand.priority,
            demand.group_id,
        )
        for demand in plan.demands
        if not demand.predecessors
    ]
    ready.sort()
    ordered = []
    while ready:
        _, group_id = heappop(ready)
        demand = demand_by_id[group_id]
        ordered.append(demand)
        for successor in successors[group_id]:
            remaining_predecessors[successor] -= 1
            if remaining_predecessors[successor] == 0:
                successor_demand = demand_by_id[successor]
                heappush(
                    ready,
                    (
                        successor_demand.priority,
                        successor,
                    ),
                )

    if len(ordered) != len(plan.demands):
        raise ValueError("weight demand predecessor graph must be acyclic")
    return tuple(ordered)


def compile_weight_plan_extent_order(
    catalog: CheckpointCatalog,
    plans: Sequence[WeightLoadPlan],
    files: Sequence[tuple[str, int]],
) -> tuple[ReadAheadExtent, ...]:
    """Prioritize whole source chunks without changing read coverage.

    The earliest ordering-qualified demand for a tensor across node-local rank
    plans wins. Tensor ranges only reprioritize the existing fixed-size source
    chunks; chunks are never split or omitted, and physically ordered unmatched
    chunks form the tail. Conservative plans are consumed through
    ``described_tensor_names`` rather than the selective-I/O-only
    ``selected_tensor_names`` property. If every plan has opaque ordering, the
    source's physical chunk order is preserved.
    """
    if not catalog.has_complete_byte_ranges:
        raise ValueError("checkpoint catalog must expose complete byte ranges")
    if not plans:
        raise ValueError("at least one weight load plan is required")

    paths_by_object_id = {}
    sizes_by_object_id = {}
    for path, size_bytes in files:
        object_id = os.path.basename(path)
        if object_id in paths_by_object_id:
            raise ValueError(f"checkpoint paths have duplicate basename object ID {object_id!r}")
        paths_by_object_id[object_id] = path
        sizes_by_object_id[object_id] = size_bytes

    catalog_sizes = {obj.object_id: obj.size_bytes for obj in catalog.objects}
    missing_objects = catalog_sizes.keys() - paths_by_object_id.keys()
    unexpected_objects = paths_by_object_id.keys() - catalog_sizes.keys()
    if missing_objects or unexpected_objects:
        raise ValueError(
            "checkpoint catalog and source files resolve different objects: "
            f"missing={sorted(missing_objects)}, "
            f"unexpected={sorted(unexpected_objects)}"
        )
    mismatched_sizes = {
        object_id: (catalog_sizes[object_id], sizes_by_object_id[object_id])
        for object_id in catalog_sizes
        if catalog_sizes[object_id] != sizes_by_object_id[object_id]
    }
    if mismatched_sizes:
        raise ValueError(
            f"checkpoint catalog object sizes do not match source files: {mismatched_sizes}"
        )

    ordered_plans = tuple(sorted(plans, key=lambda plan: plan.rank))
    if len({plan.rank for plan in ordered_plans}) != len(ordered_plans):
        raise ValueError("node-local weight load plans must have unique ranks")
    world_sizes = {plan.world_size for plan in ordered_plans}
    if len(world_sizes) != 1:
        raise ValueError("node-local weight load plans must agree on world_size")
    for plan in ordered_plans:
        plan.validate_against(catalog)

    # Dependencies establish an earliest read-ahead stage. Equal-priority,
    # independent groups share a stage: their native consumers may execute
    # concurrently, so preserve physical locality instead of inventing a
    # tensor-name ordering.
    tensor_priorities: dict[str, int] = {}
    for plan in ordered_plans:
        if plan.ordering is WeightLoadOrderConfidence.OPAQUE:
            continue
        group_priorities: dict[str, int] = {}
        for demand in _topological_demands(plan):
            priority = max(
                demand.priority,
                max(
                    (group_priorities[p] + 1 for p in demand.predecessors),
                    default=demand.priority,
                ),
            )
            group_priorities[demand.group_id] = priority
            for tensor_name in demand.source_names:
                tensor_priorities[tensor_name] = min(
                    priority, tensor_priorities.get(tensor_name, priority)
                )

    source_extents = _build_source_extents(files)
    chunk_priorities: dict[ReadAheadExtent, int] = {}
    chunks_by_object_id = {object_id: [] for object_id in paths_by_object_id}
    for extent in source_extents:
        chunks_by_object_id[os.path.basename(extent.path)].append(extent)

    for tensor_name, priority in tensor_priorities.items():
        for tensor_extent in catalog.get_tensor(tensor_name).extents:
            if tensor_extent.length_bytes == 0:
                continue
            chunks = chunks_by_object_id[tensor_extent.object_id]
            first_chunk = tensor_extent.offset_bytes // _CHUNK_SIZE
            final_chunk = (tensor_extent.end_offset_bytes - 1) // _CHUNK_SIZE
            for chunk in chunks[first_chunk : final_chunk + 1]:
                chunk_priorities[chunk] = min(priority, chunk_priorities.get(chunk, priority))

    tail_priority = max(chunk_priorities.values(), default=0) + 1
    # Python's stable sort preserves physical order within each priority and
    # among all unmatched chunks. Every original chunk remains present once.
    return tuple(
        sorted(
            source_extents,
            key=lambda extent: chunk_priorities.get(extent, tail_priority),
        )
    )


def build_local_plan(
    files: Sequence[tuple[str, int]],
    local_rank: int,
    local_size: int,
    *,
    ordered_extents: Sequence[ReadAheadExtent] | None = None,
) -> ReadAheadPlan:
    """Assign every checkpoint extent to exactly one node-local issuer."""
    if not 0 <= local_rank < local_size:
        raise ValueError("local rank and size must describe a valid group")

    extents = _build_source_extents(files)
    if ordered_extents is not None:
        if Counter(ordered_extents) != Counter(extents):
            raise ValueError("ordered_extents must be a complete permutation of checkpoint extents")
        extents = list(ordered_extents)

    worker_counts = distribute_worker_budget(local_size)
    issuer_ranks = [rank for rank, workers in enumerate(worker_counts) if workers > 0]
    if not extents or local_rank not in issuer_ranks:
        return ReadAheadPlan((), 0)

    issuer_index = issuer_ranks.index(local_rank)
    local_extents = tuple(extents[issuer_index :: len(issuer_ranks)])
    workers = min(worker_counts[local_rank], len(local_extents))
    return ReadAheadPlan(local_extents, workers)


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

        try:
            for path in dict.fromkeys(extent.path for extent in plan.extents):
                self._file_descriptors[path] = os.open(path, os.O_RDONLY)
        except Exception:
            self._close_file_descriptors()
            raise

    def start(self, *, defer_reads: bool = False) -> "RankStripedReadAheadSession":
        if not defer_reads:
            self._read_release.set()
        if not self._plan.extents:
            return self
        self._thread = threading.Thread(
            target=self._run, name="trtllm-rank-striped-read-ahead", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            self._close_file_descriptors()
            raise
        return self

    def release_reads(self) -> None:
        """Release prepared workers only after activation consensus."""
        self._read_release.set()

    def _read_extent(self, extent: ReadAheadExtent) -> int:
        file_descriptor = self._file_descriptors[extent.path]
        offset = extent.offset
        remaining = extent.length
        completed = 0
        while remaining > 0 and not self._cancel.is_set():
            read_size = min(remaining, _READ_SIZE)
            data = os.pread(file_descriptor, read_size, offset)
            if not data:
                raise OSError(
                    f"Unexpected EOF while reading {extent.path} at byte offset {offset}."
                )
            bytes_read = len(data)
            offset += bytes_read
            remaining -= bytes_read
            completed += bytes_read
        return completed

    def _request_cancel(self) -> None:
        self._cancel.set()
        # Wake a prepared worker that is still behind the activation gate. It
        # observes cancellation before issuing any payload I/O.
        self._read_release.set()

    def _run(self) -> None:
        try:
            self._read_release.wait()
            if self._cancel.is_set():
                return
            with ThreadPoolExecutor(max_workers=self._plan.workers) as executor:
                futures = [
                    executor.submit(self._read_extent, extent) for extent in self._plan.extents
                ]
                try:
                    for future in as_completed(futures):
                        if self._cancel.is_set():
                            for pending in futures:
                                pending.cancel()
                            break
                        future.result()
                except Exception:
                    self._request_cancel()
                    for future in futures:
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
