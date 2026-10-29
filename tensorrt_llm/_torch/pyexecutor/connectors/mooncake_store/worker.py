# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Worker side of the Mooncake store KV cache connector.

One worker per rank owns a `MooncakeDistributedStore` handle and moves pages
between that pool and its own GPU KV cache. It is also the only place that
knows how a page is addressed and how a key is spelled, which is why the
leader, colocated with rank 0's worker, asks it to run prefix lookups.

Loads must land before the request computes: the runtime has already counted
those tokens, so a failed load is a wrong answer rather than a slow one. A
confirmed prefix reservation loads on a loader thread while its request waits
outside the batch (`SYNC_LOAD_ENV` loads it in the executor's step instead).

Saves are asynchronous and gated on a CUDA event, since the pages are complete
only once the forward pass that wrote them has retired and blocking the
executor loop on an RDMA write is the cost the store exists to avoid. The
scheduler reports such a request as saving asynchronously, which keeps its
pages pinned until `get_finished` says the writes landed. A save that fails is
logged and dropped: a write that did not land costs a future cache miss and
nothing else, so the request retires either way. Only a save or loader thread
failing to start is fatal, because that would strand every request behind pages
nothing will release or loads nothing will run.

A capacity-only worker opens its handle and stops there, with no layout, no
buffer registration and no save or loader threads, so a node can lend host memory to the
pool without an HCA that can pin GPU pages.
"""

import os
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttnLifeCycle

from ..kv_cache_connector import KvCacheConnectorWorker
from ..kv_cache_layout import KvCacheLayout
from .addressing import PageAddressing
from .config import CONFIG_PATH_ENV, MooncakeStoreConnectorConfig, local_address, pool_config
from .keys import KeyNamespace
from .ledger import record_segment
from .metadata import MooncakeStoreMetadata, PrefixLoadTransfers, RequestTransfers
from .staging import (
    MAX_STAGING_BUFFER_BYTES,
    HostStagingPool,
    describe_batch_for_get,
    plan_slot_geometry,
    stage_batch_for_put,
    unstage_batch_after_get,
)
from .staging import sync_stream as _sync_stream
from .validation import validate_layout, validate_llm_args, validate_node_budget

__all__ = ["MooncakeStoreConnectorWorker", "PrefixHit", "resolve_local_worker"]

#: Set by the worker's constructor so the leader, which the executor builds in
#: the same process on rank 0, can reach the store handle without a second
#: connection. `py_executor_creator` constructs the two concurrently, so the
#: leader waits on `_LOCAL_WORKER_READY`.
_LOCAL_WORKER: Optional["MooncakeStoreConnectorWorker"] = None
_LOCAL_WORKER_READY = threading.Event()

#: Seconds `shutdown` gives the save thread, and each loader thread, to drain its queue and return.
#: Sized for a backlog of pages over a congested fabric rather than a healthy
#: one, since what follows the wait is only safe once the thread has stopped.
SAVE_DRAIN_TIMEOUT = 30.0

#: Seconds between the loader, save and lookup statistics reports.
LOAD_STATS_INTERVAL = 30.0

#: Asynchronous prefix-load threads per loading rank, each with a staging pool
#: and stream of its own.
LOAD_THREADS_ENV = "TRTLLM_MOONCAKE_STORE_LOAD_THREADS"
DEFAULT_LOAD_THREADS = 2


#: Set to 1 for reserved prefixes to load synchronously, in the executor's step that computes the rest of the prompt,
#: instead of on loader threads; read by the scheduler, and here to start no loader threads. Asynchronous loads (the
#: default) park the request until its pages are on the device while the others keep running, so a load costs that
#: request time instead of the batch.
SYNC_LOAD_ENV = "TRTLLM_MOONCAKE_STORE_SYNC_LOAD"

#: Whether a lookup that finds no window within the rewind looks further back for an
#: earlier prompt end whose window is stored, instead of reporting a miss (on by default; 0 turns it off).
WINDOW_FALLBACK_ENV = "TRTLLM_MOONCAKE_STORE_WINDOW_FALLBACK"

#: Keys per existence-probe call, 0 for one call. A lookup over a long prefix probes one key per block per rank, which
#: the master answers on one RPC thread while the scheduler waits; split, the calls run concurrently on the master's
#: thread pool.
PROBE_CHUNK_ENV = "TRTLLM_MOONCAKE_STORE_PROBE_CHUNK"
DEFAULT_PROBE_CHUNK = 4096
#: Concurrent probe calls per lookup.
PROBE_THREADS = 4

#: Prompts that computed at least this many tokens list every full-attention block again at their last chunk, so the
#: save path refreshes the chain's head and rewrites what the pool evicted during the prefill. 0 turns it off. Read by
#: the scheduler.
RESAVE_MIN_TOKENS_ENV = "TRTLLM_MOONCAKE_STORE_RESAVE_MIN_TOKENS"


def _load_thread_count() -> int:
    count = int(os.environ.get(LOAD_THREADS_ENV, DEFAULT_LOAD_THREADS))
    if count < 1:
        raise ValueError(f"{LOAD_THREADS_ENV} must be at least 1, got {count}")
    return count


def resolve_local_worker(timeout: float = 60.0) -> "MooncakeStoreConnectorWorker":
    """The worker living in this process, once it has been constructed.

    Args:
        timeout: Seconds to wait. Exceeding it means construction failed.
    """
    if not _LOCAL_WORKER_READY.wait(timeout):
        raise RuntimeError(
            "The mooncake-store leader could not find a worker in its process. "
            "The leader only runs on rank 0, where the executor also builds a "
            "worker, so this means worker construction failed."
        )
    assert _LOCAL_WORKER is not None
    return _LOCAL_WORKER


def _open_store(config: MooncakeStoreConnectorConfig):
    """Connect to the Mooncake master and return a live store handle.

    Returns:
        The store handle, and the host the segment is registered under.
    """
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:
        raise ImportError(
            "The mooncake-store connector needs the Mooncake Python bindings, "
            "which `tensorrt-llm` pulls in as `mooncake-transfer-engine-cuda13`, "
            "so reinstall that package if this environment dropped it. The C++ "
            "transfer engine built into the container is a different component "
            "and does not provide MooncakeDistributedStore."
        ) from exc

    store = MooncakeDistributedStore()
    hostname = config.local_hostname or local_address(config.master_server_address)
    setup_kwargs = {}
    if config.tenant_id:
        setup_kwargs["tenant_id"] = config.tenant_id
    status = store.setup(
        hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if status != 0:
        raise RuntimeError(
            f"MooncakeDistributedStore.setup failed with status {status} "
            f"(master={config.master_server_address!r}, "
            f"metadata={config.metadata_server!r}, protocol={config.protocol!r}, "
            f"global_segment_size={config.global_segment_size}). The master "
            "must already be accepting connections; the protocol and device "
            "must be usable from this host; and this node must have the "
            "segment's worth of memory to spare once every rank on it has "
            f"claimed one. Check the config named by {CONFIG_PATH_ENV}."
        )
    return store, hostname


def _interleaved_keys(
    namespaces: Sequence[KeyNamespace], block_hashes: Sequence[bytes]
) -> List[str]:
    """Every namespace's key of every block, block by block in namespace order.

    Built per namespace from its fixed prefix, then interleaved: a lookup over a long prompt builds one key per block
    per rank on the scheduler's path, and calling `KeyNamespace.key` per key costs several times as much.
    """
    if not namespaces or not block_hashes:
        return []
    hexes = [block_hash.hex() for block_hash in block_hashes]
    columns = []
    for namespace in namespaces:
        prefix = namespace.prefix + "/"
        columns.append([prefix + digest for digest in hexes])
    if len(columns) == 1:
        return columns[0]
    return [key for row in zip(*columns) for key in row]


def _batched(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _stream_handle(stream: Optional[torch.cuda.Stream]) -> int:
    """The raw CUDA stream handle behind a torch stream.

    `None` maps to 0, the default stream, which is what the runtime passes when
    it has no stream of its own to offer.
    """
    if stream is None:
        return 0
    return int(stream.cuda_stream)


class _LoadStats:
    """The loader thread's running totals, reported once per `LOAD_STATS_INTERVAL`."""

    def __init__(self):
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self.since = now
        self.loads = self.pages = self.loaded_bytes = 0
        self.busy = self.waited = self.max_wait = 0.0

    def add(self, pages: int, loaded_bytes: int, wait: float, busy: float) -> Optional[str]:
        """Count one load; past the interval, return the report and start a new one."""
        self.loads += 1
        self.pages += pages
        self.loaded_bytes += loaded_bytes
        self.busy += busy
        self.waited += wait
        self.max_wait = max(self.max_wait, wait)
        now = time.monotonic()
        elapsed = now - self.since
        if elapsed < LOAD_STATS_INTERVAL:
            return None
        report = (
            f"{self.loads} loads, {self.pages} pages, {self.loaded_bytes / (1 << 30):.2f} GiB "
            f"in {elapsed:.0f}s ({self.busy:.1f}s busy, "
            f"{self.loaded_bytes / (1 << 30) / max(self.busy, 1e-9):.1f} GiB/s while busy); "
            f"queue wait mean {self.waited / self.loads:.3f}s max {self.max_wait:.3f}s"
        )
        self._reset(now)
        return report


class _SaveStats:
    """The save thread's running totals, reported once per `LOAD_STATS_INTERVAL`."""

    def __init__(self):
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self.since = now
        self.batches = self.written = self.present = self.written_bytes = 0
        self.busy = self.waited = self.max_wait = 0.0

    def add(
        self, written: int, present: int, written_bytes: int, wait: float, busy: float
    ) -> Optional[str]:
        """Count one queued save; past the interval, return the report and start a new one."""
        self.batches += 1
        self.written += written
        self.present += present
        self.written_bytes += written_bytes
        self.busy += busy
        self.waited += wait
        self.max_wait = max(self.max_wait, wait)
        now = time.monotonic()
        elapsed = now - self.since
        if elapsed < LOAD_STATS_INTERVAL:
            return None
        report = (
            f"{self.batches} batches, {self.written} pages written ({self.written_bytes / (1 << 30):.2f} GiB), "
            f"{self.present} already stored, in {elapsed:.0f}s ({self.busy:.1f}s busy); "
            f"queue wait mean {self.waited / self.batches:.3f}s max {self.max_wait:.3f}s"
        )
        self._reset(now)
        return report


class _LookupStats:
    """Lookup outcomes on the leader, reported once per `LOAD_STATS_INTERVAL`.

    full: the window at the full-attention hit's end is stored. rewound: one within the rewind is. recovered: an
    earlier end found by the fallback. window_miss: full-attention blocks but no usable window. fa_miss: not even
    the first full-attention block.
    """

    OUTCOMES = ("full", "rewound", "recovered", "window_miss", "fa_miss")

    def __init__(self):
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self.since = now
        self.counts = dict.fromkeys(self.OUTCOMES, 0)
        self.hit_blocks = self.lost_blocks = 0
        self.probe_seconds = 0.0
        self.max_probe = 0.0

    def add(
        self, outcome: str, hit_blocks: int, lost_blocks: int, probe_seconds: float
    ) -> Optional[str]:
        self.counts[outcome] += 1
        self.hit_blocks += hit_blocks
        self.lost_blocks += lost_blocks
        self.probe_seconds += probe_seconds
        self.max_probe = max(self.max_probe, probe_seconds)
        now = time.monotonic()
        elapsed = now - self.since
        if elapsed < LOAD_STATS_INTERVAL:
            return None
        total = sum(self.counts.values())
        report = (
            f"{total} lookups ("
            + ", ".join(f"{name} {count}" for name, count in self.counts.items())
            + f"), {self.hit_blocks} blocks hit, {self.lost_blocks} blocks behind the full-attention hit lost, "
            f"probe mean {self.probe_seconds / max(1, total) * 1e3:.1f}ms max {self.max_probe * 1e3:.1f}ms "
            f"in {elapsed:.0f}s"
        )
        self._reset(now)
        return report


@dataclass(frozen=True)
class PrefixHit:
    """How far a request can resume from the store, and what backs it."""

    #: Leading candidate blocks the store can replay.
    blocks: int
    #: Per sliding-window group, the first block from which the store holds
    #: every page up to the hit's end. A replay the runtime cuts short reads its
    #: window from further back, which is sound only down to this block.
    window_present_from: Dict[int, int] = field(default_factory=dict)


def _live_blocks(sinks: int, stale_end: int, lo: int, hi: int) -> List[int]:
    """Block ordinals in ``[lo, hi)`` a window still reads: its sinks and its tail."""
    return [*range(lo, min(sinks, hi)), *range(max(lo, sinks, stale_end), hi)]


class MooncakeStoreConnectorWorker(KvCacheConnectorWorker):
    """Moves KV pages between this rank's GPU cache and the Mooncake pool."""

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.resolve(llm_args)
        # Where this rank records the segment it mounts. `None` when the
        # deployment named no directory, which only makes the run
        # unreportable; see `ledger.record_segment`.
        pool = pool_config(llm_args)
        self._run_dir = pool.run_dir if pool is not None else None
        self._rank = mpi_rank()
        self._world_size = mpi_world_size()
        self._model_key = self._config.resolve_model_key(llm_args.model)

        self._addressing: Optional[PageAddressing] = None
        # Namespaces for this rank, used for both directions of transfer.
        self._namespaces: Dict[int, KeyNamespace] = {}
        # The same namespaces for every rank. A prefix is only reusable when all
        # shards of it are present, so a lookup has to ask about all of them.
        self._peer_namespaces: Dict[int, Tuple[KeyNamespace, ...]] = {}
        # The cache's own life cycle per sliding-window group, which decides
        # the blocks such a group still reads at a given history length.
        self._life_cycles: Dict[int, AttnLifeCycle] = {}

        # Checked before the segment is mounted, since the kernel answers an
        # unaffordable total by killing the process rather than by failing the
        # allocation.
        # No loader threads, and no staging pools for them, when reserved prefixes load synchronously.
        self._num_load_threads = (
            _load_thread_count()
            if self._config.role.loads and os.environ.get(SYNC_LOAD_ENV, "0") != "1"
            else 0
        )
        # A sliding-window layout is served through prefix reservations only; see register_kv_cache_layout.
        self._prefix_aware_scheduling = bool(
            getattr(
                getattr(llm_args, "scheduler_config", None), "enable_prefix_aware_scheduling", True
            )
        )
        validate_node_budget(self._config, load_threads=self._num_load_threads)

        self._store, self._segment_host = _open_store(self._config)
        record_segment(
            self._run_dir,
            host=self._segment_host,
            rank=self._rank,
            segment_size=self._config.global_segment_size,
            role=self._config.role.value,
            model_key=self._model_key,
        )

        self._save_queue: "Queue[Optional[Tuple[torch.cuda.Event, List[RequestTransfers]]]]" = (
            Queue()
        )
        self._save_thread: Optional[threading.Thread] = None
        #: Set once the save thread has finished its device and stream setup,
        #: whether or not that succeeded.
        self._save_started = threading.Event()
        self._save_lock = threading.Lock()
        # Host staging, when the pool cannot register device memory.
        self._load_staging: Optional[HostStagingPool] = None
        self._save_staging: Optional[HostStagingPool] = None
        self._save_stream: Optional[torch.cuda.Stream] = None
        # This rank's device, captured on the executor thread; see _drain_saves.
        self._device_index: Optional[int] = None
        # Pages per store call. Staging narrows this to the slots it can afford.
        self._batch_size = self._config.transfer_batch_size
        # Save submissions still in flight, per request.
        self._outstanding_saves: Dict[int, int] = defaultdict(int)
        # Requests the runtime has told us are done producing KV. Their pages
        # stay pinned until we report them back through `get_finished`.
        self._closed_requests: Set[int] = set()
        # Why the save thread never reached its transfer loop, read back by
        # `register_kv_cache_layout`. A failure inside the loop is dropped
        # there instead.
        self._save_startup_error: Optional[BaseException] = None
        # Asynchronous prefix loads run on threads of their own, each with its
        # own staging slots and stream, while their requests wait outside the
        # batch. The threads share one queue.
        self._load_queue: "Queue[Optional[Tuple[int, RequestTransfers, Optional[torch.cuda.Event], float]]]" = Queue()
        self._load_threads: List[threading.Thread] = []
        self._load_startup_error: Optional[BaseException] = None
        self._async_load_stagings: List[HostStagingPool] = []
        self._window_fallback = os.environ.get(WINDOW_FALLBACK_ENV, "1") != "0"
        self._lookup_stats = _LookupStats()
        self._probe_chunk = int(os.environ.get(PROBE_CHUNK_ENV, DEFAULT_PROBE_CHUNK))
        # Lookups run on the leader's scheduler, so only a loading role probes.
        self._probe_pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=PROBE_THREADS, thread_name_prefix=f"mooncake-store-probe-{self._rank}"
            )
            if self._config.role.loads and self._probe_chunk > 0
            else None
        )
        self._load_lock = threading.Lock()
        # Reservations whose pages landed, until `get_finished_prefix_loads`.
        self._finished_prefix_loads: List[int] = []
        # The loader thread's first failure, raised on the executor thread.
        self._load_error: Optional[BaseException] = None

        global _LOCAL_WORKER
        _LOCAL_WORKER = self
        _LOCAL_WORKER_READY.set()

        logger.warning(
            f"mooncake-store worker rank {self._rank}/{self._world_size} ready "
            f"(role={self._config.role.value}, model_key={self._model_key}, "
            f"master={self._config.master_server_address}, "
            f"segment={self._config.global_segment_size / (1 << 30):.1f} GiB "
            f"on {self._segment_host})"
        )
        if self.capacity_only:
            # Said plainly, since every other sign of a working connector is
            # absent by design and otherwise looks like a broken deployment.
            logger.warning(
                f"mooncake-store worker rank {self._rank}: capacity-only. Its "
                "memory is in the pool and its KV cache is not: no lookups, no "
                "loads, no saves, and no staging buffers pinned for them."
            )

    # ---- registration ----

    @property
    def capacity_only(self) -> bool:
        """Whether this rank lends memory to the pool and transfers nothing."""
        return self._config.capacity_only

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        """Reject the V1 single-pool registration, unless capacity-only.

        Raises:
            NotImplementedError: Unless capacity-only. V1 supplies block hashes
                over a single flat block space, while this connector keys pages
                per layer group from a hash chain of its own, so running the V2
                addressing against V1 block ids would mislabel pages.
        """
        if self.capacity_only:
            # Nothing is addressed, so which manager describes the cache does
            # not matter.
            return
        raise NotImplementedError(
            "The mooncake-store connector requires KVCacheManagerV2. Set "
            "kv_cache_config.use_kv_cache_manager_v2=True."
        )

    def register_kv_cache_layout(self, layout: KvCacheLayout) -> None:
        """Register the KV pools with Mooncake and start the save thread."""
        if self._addressing is not None:
            raise RuntimeError("KV cache layout already registered")

        if self.capacity_only:
            # Nothing will ask the store to reach this rank's pages, so the
            # addressing, buffer registration, staging pool and save thread
            # are all skipped.
            logger.warning(
                f"mooncake-store worker rank {self._rank}: capacity-only, so "
                "the KV cache layout is not registered and no page of it is "
                "reachable from the pool."
            )
            return

        validate_layout(layout)
        addressing = PageAddressing(layout)
        # Torch's current device is thread-local, so read it here on the
        # executor thread; the save thread would otherwise see device 0.
        if torch.cuda.is_available():
            self._device_index = torch.cuda.current_device()
        # Host staging is the only transfer path, which is why
        # stage_through_host is forced True upstream rather than read here.
        self._open_staging(addressing)

        self._addressing = addressing
        for layer_group_id in addressing.layer_group_ids:
            bytes_per_page = addressing.bytes_per_page(layer_group_id)
            self._namespaces[layer_group_id] = self._namespace(
                self._rank, layer_group_id, bytes_per_page
            )
            self._peer_namespaces[layer_group_id] = tuple(
                self._namespace(rank, layer_group_id, bytes_per_page)
                for rank in range(self._world_size)
            )
        for group in layout.groups:
            if group.window_size is not None:
                self._life_cycles[group.layer_group_id] = AttnLifeCycle.make(
                    group.window_size, group.num_sink_tokens, layout.tokens_per_block
                )
        if self._life_cycles and self._config.role.loads and not self._prefix_aware_scheduling:
            raise ValueError(
                "The mooncake-store connector serves sliding-window layer groups only through prefix "
                "reservations, which scheduler_config.enable_prefix_aware_scheduling=False turns off. Without "
                "them the runtime can honour a stored prefix only up to a point whose window the lookup never "
                "verified (its fallback when the full offer does not fit), and the load would then read a "
                "window page nothing wrote. Leave enable_prefix_aware_scheduling on."
            )

        if self._config.role.saves:
            self._save_thread = threading.Thread(
                target=self._drain_saves,
                name=f"mooncake-store-save-{self._rank}",
                daemon=True,
            )
            self._save_thread.start()
            # A thread that cannot bind this rank's device has to fail bringup
            # here, or its requests stay pinned on saves nothing will consume.
            self._save_started.wait()
            with self._save_lock:
                startup_error, self._save_startup_error = self._save_startup_error, None
            if startup_error is not None:
                raise RuntimeError(
                    f"mooncake-store: the save thread for rank {self._rank} "
                    "could not start, so this worker would accept pages it "
                    "could never write to the pool."
                ) from startup_error

        if self._config.role.loads:
            started = []
            for index in range(self._num_load_threads):
                # Set once the thread has finished its device and stream setup,
                # whether or not that succeeded.
                event = threading.Event()
                thread = threading.Thread(
                    target=self._drain_loads,
                    args=(index, event),
                    name=f"mooncake-store-load-{self._rank}-{index}",
                    daemon=True,
                )
                thread.start()
                self._load_threads.append(thread)
                started.append(event)
            # As for the save thread: a loader that cannot bind this rank's
            # device would leave every parked request waiting for good.
            for event in started:
                event.wait()
            with self._load_lock:
                startup_error, self._load_startup_error = self._load_startup_error, None
            if startup_error is not None:
                raise RuntimeError(
                    f"mooncake-store: the load thread for rank {self._rank} "
                    "could not start, so asynchronous prefix loads would never land."
                ) from startup_error

        logger.warning(
            f"mooncake-store worker rank {self._rank} registered layout: {addressing.describe()}"
        )

    def _open_staging(self, addressing: PageAddressing) -> None:
        """Allocate and register the pinned slots pages will pass through.

        Only the directions this role drives get a pool, since each one costs
        a pinned allocation of its own.
        """
        max_bytes_per_page = max(
            addressing.bytes_per_page(layer_group_id)
            for layer_group_id in addressing.layer_group_ids
        )
        slot_bytes, num_slots = plan_slot_geometry(
            max_bytes_per_page,
            self._config.transfer_batch_size,
            MAX_STAGING_BUFFER_BYTES,
        )
        if self._config.role.loads:
            self._load_staging = HostStagingPool(
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                store=self._store,
                label="load",
            )
            # Each loader thread's own slots, so none waits on another's or the
            # executor's.
            self._async_load_stagings = [
                HostStagingPool(
                    slot_bytes=slot_bytes,
                    num_slots=num_slots,
                    store=self._store,
                    label=f"async-load-{index}",
                )
                for index in range(self._num_load_threads)
            ]
        if self._config.role.saves:
            self._save_staging = HostStagingPool(
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                store=self._store,
                label="save",
            )
        self._batch_size = min(self._config.transfer_batch_size, num_slots)
        if self._batch_size < self._config.transfer_batch_size:
            logger.warning(
                f"mooncake-store rank {self._rank} reduced its transfer batch from "
                f"{self._config.transfer_batch_size} to {self._batch_size} pages: "
                f"staging {max_bytes_per_page} B pages within the "
                f"{MAX_STAGING_BUFFER_BYTES} B pinned-memory ceiling does not fit "
                f"more. Lower transfer_batch_size to make the reduction explicit."
            )

    def _namespace(self, rank: int, layer_group_id: int, bytes_per_page: int) -> KeyNamespace:
        return KeyNamespace(
            namespace=self._config.namespace,
            model_key=self._model_key,
            rank=rank,
            world_size=self._world_size,
            layer_group_id=layer_group_id,
            tokens_per_block=self._addressing.tokens_per_block,
            bytes_per_page=bytes_per_page,
        )

    # ---- leader-facing lookup ----

    @property
    def config(self) -> MooncakeStoreConnectorConfig:
        """The resolved connector configuration."""
        return self._config

    @property
    def is_registered(self) -> bool:
        """Whether a KV cache layout has been registered yet."""
        return self._addressing is not None

    @property
    def windowed_group_ids(self) -> Tuple[int, ...]:
        """Layer groups with a sliding window, in layout order.

        Such a group keeps only the tail of a sequence, so the scheduler saves
        its pages where prompts end and a lookup has to land on one of those.
        """
        return tuple(self._life_cycles)

    def sink_blocks(self, layer_group_id: int) -> int:
        """Leading blocks a sliding-window group keeps however far it moves."""
        life_cycle = self._life_cycles.get(layer_group_id)
        return 0 if life_cycle is None else int(life_cycle.num_sink_blocks)

    def stale_blocks(self, layer_group_id: int, history_length: int) -> Tuple[int, int]:
        """Block ordinals ``[start, end)`` a group no longer reads at ``history_length``.

        Answered by the cache's own life cycle, as
        `KVCacheManagerV2._stale_block_range` is, so the blocks a lookup asks
        for are the ones the runtime gives pages to. Empty for full attention.
        """
        life_cycle = self._life_cycles.get(layer_group_id)
        if life_cycle is None:
            return 0, 0
        stale = life_cycle.get_stale_range(history_length, self._addressing.tokens_per_block)
        # Read `beg`/`end` rather than unpacking: only the pure-Python
        # `HalfOpenRange` is a tuple subclass, the bound one is not.
        return int(stale.beg), int(stale.end)

    def count_prefix_hit(self, block_hashes: Sequence[bytes]) -> int:
        """How many leading blocks of `block_hashes` are fully present.

        A block counts only when every layer group and every rank has its
        page, since a prefix is replayed as a whole. The scan stops at the
        first incomplete block, a later hit being unusable on its own.

        Args:
            block_hashes: Candidate hashes in block order.

        Returns:
            Length of the usable prefix, in blocks.
        """
        if not block_hashes or self._addressing is None:
            return 0
        return self._count_leading(block_hashes, tuple(self._peer_namespaces))

    def find_prefix_hit(
        self, block_hashes: Sequence[bytes], first_block: int, rewind_tokens: int = 0
    ) -> PrefixHit:
        """The longest stored prefix a request can resume from, windows included.

        Full-attention groups need every block of the prefix. A sliding-window
        group needs only the blocks it still reads where the prefix ends, and
        the store holds those only near the ends of earlier prompts, so the
        search starts at the longest full-attention hit and walks back up to
        `rewind_tokens`, taking the first prompt end whose window is complete.
        With no window this is `count_prefix_hit`.

        When no end within the rewind has its window, the full-attention blocks
        can still reach past the last stored window (a save that has not landed
        yet, or a window evicted on its own). Rather than give up the whole
        prefix, the search then probes the window pages further back and
        resumes at the latest earlier end whose window is complete.

        Args:
            block_hashes: Candidate hashes in block order.
            first_block: Block ordinal of `block_hashes[0]`. Everything before it
                is on device already, window included.
            rewind_tokens: How far back from the full-attention hit to look for
                a prompt end whose window is stored.

        Returns:
            The usable prefix in blocks, and for each sliding-window group the
            first block from which the store holds every page up to its end.
        """
        started = time.monotonic()
        hit, outcome, fa_end = self._find_prefix_hit(block_hashes, first_block, rewind_tokens)
        if self._life_cycles and block_hashes and self._addressing is not None:
            report = self._lookup_stats.add(
                outcome,
                hit.blocks,
                max(0, fa_end - first_block - hit.blocks),
                time.monotonic() - started,
            )
            if report:
                logger.info(f"mooncake-store rank {self._rank} lookups: {report}")
        return hit

    def _find_prefix_hit(
        self, block_hashes: Sequence[bytes], first_block: int, rewind_tokens: int
    ) -> Tuple[PrefixHit, str, int]:
        """`find_prefix_hit`, which of `_LookupStats.OUTCOMES` it came to, and where the full-attention hit ends."""
        if not block_hashes or self._addressing is None:
            return PrefixHit(0), "fa_miss", first_block
        full_groups = tuple(g for g in self._peer_namespaces if g not in self._life_cycles)
        hit = self._count_leading(block_hashes, full_groups) if full_groups else len(block_hashes)
        if not self._life_cycles or hit == 0:
            return PrefixHit(hit), ("full" if hit else "fa_miss"), first_block + hit

        tokens_per_block = self._addressing.tokens_per_block
        end = first_block + hit
        # The search for a stored prompt end goes back at most `rewind_tokens`;
        # past that, only the fallback looks.
        lowest_end = max(first_block + 1, end - -(-rewind_tokens // tokens_per_block))

        # Every block some candidate end reads, probed in one call: the sinks,
        # and the tail from where the lowest end's window starts.
        probed: Dict[int, List[int]] = {}
        for layer_group_id in self._life_cycles:
            _, stale_end = self.stale_blocks(layer_group_id, lowest_end * tokens_per_block)
            probed[layer_group_id] = _live_blocks(
                self.sink_blocks(layer_group_id), stale_end, first_block, end
            )
        stored = self._probe_window_blocks(block_hashes, first_block, probed)
        if stored is None:
            return PrefixHit(0), "window_miss", end

        for candidate in range(end, lowest_end - 1, -1):
            if self._window_complete(stored, first_block, candidate):
                outcome = "full" if candidate == end else "rewound"
                hit_at = PrefixHit(
                    candidate - first_block, self._present_from(stored, first_block, candidate)
                )
                return hit_at, outcome, end

        if not self._window_fallback or lowest_end <= first_block + 1:
            return PrefixHit(0), "window_miss", end

        # The fallback: probe each window group's blocks below what the first
        # call covered, then take the latest earlier end with a complete window.
        below: Dict[int, List[int]] = {}
        for layer_group_id, blocks in probed.items():
            low = max(first_block, self.sink_blocks(layer_group_id))
            first_probed = next((b for b in blocks if b >= low), end)
            below[layer_group_id] = list(range(low, first_probed))
        more = self._probe_window_blocks(block_hashes, first_block, below)
        if more is None:
            return PrefixHit(0), "window_miss", end
        for layer_group_id, blocks in more.items():
            stored[layer_group_id] |= blocks
        # Missing-block prefix sums per group over [first_block, end), so each
        # candidate's window is checked in constant time.
        missing: Dict[int, List[int]] = {}
        for layer_group_id, blocks in stored.items():
            sums = [0]
            for block in range(first_block, end):
                sums.append(sums[-1] + (block not in blocks))
            missing[layer_group_id] = sums
        for candidate in range(lowest_end - 1, first_block, -1):
            history_length = candidate * tokens_per_block
            complete = True
            for layer_group_id, sums in missing.items():
                sinks = self.sink_blocks(layer_group_id)
                tail_from = min(
                    candidate,
                    max(first_block, sinks, self.stale_blocks(layer_group_id, history_length)[1]),
                )
                sink_hi = max(first_block, min(sinks, candidate))
                if (
                    sums[sink_hi - first_block]
                    or sums[candidate - first_block] - sums[tail_from - first_block]
                ):
                    complete = False
                    break
            if complete:
                return (
                    PrefixHit(
                        candidate - first_block, self._present_from(stored, first_block, candidate)
                    ),
                    "recovered",
                    end,
                )
        return PrefixHit(0), "window_miss", end

    def _probe_window_blocks(
        self, block_hashes: Sequence[bytes], first_block: int, blocks_by_group: Dict[int, List[int]]
    ) -> Optional[Dict[int, Set[int]]]:
        """Which of the given window-group blocks every rank holds, or None if the probe failed."""
        keys: List[str] = []
        for layer_group_id, blocks in blocks_by_group.items():
            keys.extend(
                _interleaved_keys(
                    self._peer_namespaces[layer_group_id],
                    [block_hashes[block - first_block] for block in blocks],
                )
            )
        present = self._probe(keys) if keys else []
        if present is None:
            return None
        stored: Dict[int, Set[int]] = {}
        index = 0
        for layer_group_id, blocks in blocks_by_group.items():
            stored[layer_group_id] = set()
            for block in blocks:
                statuses = present[index : index + self._world_size]
                index += self._world_size
                if all(status == 1 for status in statuses):
                    stored[layer_group_id].add(block)
        return stored

    def _window_complete(
        self, stored: Dict[int, Set[int]], first_block: int, candidate: int
    ) -> bool:
        """Whether every window group holds the blocks it reads at `candidate`."""
        history_length = candidate * self._addressing.tokens_per_block
        return all(
            set(
                _live_blocks(
                    self.sink_blocks(layer_group_id),
                    self.stale_blocks(layer_group_id, history_length)[1],
                    first_block,
                    candidate,
                )
            )
            <= stored[layer_group_id]
            for layer_group_id in self._life_cycles
        )

    @staticmethod
    def _present_from(
        stored: Dict[int, Set[int]], first_block: int, candidate: int
    ) -> Dict[int, int]:
        """Per window group, the first block of the stored run that ends at `candidate`."""
        present_from = {}
        for layer_group_id, blocks in stored.items():
            block = candidate
            while block - 1 >= first_block and block - 1 in blocks:
                block -= 1
            present_from[layer_group_id] = block
        return present_from

    def _count_leading(self, block_hashes: Sequence[bytes], group_ids: Sequence[int]) -> int:
        """Leading blocks whose pages every rank holds in each of `group_ids`."""
        namespaces = [
            namespace
            for layer_group_id in group_ids
            for namespace in self._peer_namespaces[layer_group_id]
        ]
        keys = _interleaved_keys(namespaces, block_hashes)
        keys_per_block = len(namespaces)

        present = self._probe(keys)
        if present is None:
            return 0

        # Mooncake reports 1 for present, 0 for absent and a negative value for
        # a failed probe. Anything but a definite 1 is treated as a miss, and
        # the hit ends at the block holding the first one.
        first_miss = next(
            (index for index, status in enumerate(present) if status != 1), len(present)
        )
        return first_miss // keys_per_block

    def _probe(self, keys: List[str]) -> Optional[List[int]]:
        """`batch_is_exist` over `keys`, or None when the probe itself failed.

        Long key lists go out as concurrent chunks of `PROBE_CHUNK_ENV` keys, answered in order.
        """
        chunk = self._probe_chunk
        if self._probe_pool is None or chunk <= 0 or len(keys) <= chunk:
            chunks = [keys]
        else:
            chunks = [keys[start : start + chunk] for start in range(0, len(keys), chunk)]
        try:
            if len(chunks) == 1:
                answers = [self._store.batch_is_exist(chunks[0])]
            else:
                futures = [
                    self._probe_pool.submit(self._store.batch_is_exist, part) for part in chunks
                ]
                answers = [future.result() for future in futures]
        except Exception as exc:
            logger.warning(
                f"mooncake-store lookup failed; treating as a miss: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            return None

        present: List[int] = []
        for part, answer in zip(chunks, answers):
            if len(answer) != len(part):
                logger.warning(
                    f"mooncake-store batch_is_exist returned {len(answer)} results for "
                    f"{len(part)} keys; treating as a miss"
                )
                return None
            present.extend(answer)
        return present

    # ---- load path ----

    def start_load_kv(self, stream: torch.cuda.Stream):
        """Pull this pass's pages into their GPU slots, or hand them to the loader thread.

        Synchronous loads land before the forward pass. An asynchronous prefix
        load's request is parked outside the batch, so its pages go to the
        loader thread and are reported through `get_finished_prefix_loads`.
        """
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None:
            return
        self._raise_load_error()

        now: List[RequestTransfers] = list(metadata.loads)
        finished: List[int] = []
        for prefix_load in getattr(metadata, "prefix_loads", ()):
            if prefix_load.is_async:
                self._queue_async_load(prefix_load, stream)
            else:
                now.append(prefix_load.transfers)
                finished.append(prefix_load.reservation_id)
        if now:
            staging = self._load_staging
            self._load_pages(now, staging, _stream_handle(stream) if staging is not None else 0)
        if finished:
            with self._load_lock:
                self._finished_prefix_loads.extend(finished)

    def get_finished_prefix_loads(self) -> List[int]:
        """Reservations whose pages are on this rank's device since the last call."""
        self._raise_load_error()
        with self._load_lock:
            finished, self._finished_prefix_loads = self._finished_prefix_loads, []
        return finished

    def _queue_async_load(self, prefix_load: PrefixLoadTransfers, stream) -> None:
        """Hand a parked request's pages to the loader thread.

        The pages were just allocated, and their previous owner's last forward
        may still be queued on the executor's stream, so the loader waits for
        an event recorded there before it writes them.
        """
        if not prefix_load.transfers.pages:
            with self._load_lock:
                self._finished_prefix_loads.append(prefix_load.reservation_id)
            return
        ready = None
        if torch.cuda.is_available():
            ready = torch.cuda.Event()
            ready.record(stream)
        self._load_queue.put(
            (prefix_load.reservation_id, prefix_load.transfers, ready, time.monotonic())
        )

    def _raise_load_error(self) -> None:
        """Surface a loader-thread failure on the executor thread.

        A parked request whose pages did not all land can neither run nor be
        reported done, and under tensor parallelism the ranks cannot agree to
        restart it, so the failure is fatal, as on the synchronous path.
        """
        with self._load_lock:
            error = self._load_error
        if error is not None:
            raise RuntimeError(
                f"mooncake-store: an asynchronous prefix load failed on rank {self._rank}"
            ) from error

    def _drain_loads(self, index: int, started: threading.Event) -> None:
        """A loader thread: one parked request at a time, on its own stream and staging."""
        try:
            if self._device_index is not None:
                # A new thread starts on device 0; see `_drain_saves`.
                torch.cuda.set_device(self._device_index)
            staging = self._async_load_stagings[index] if self._async_load_stagings else None
            stream = (
                torch.cuda.Stream() if staging is not None and torch.cuda.is_available() else None
            )
            handle = _stream_handle(stream)
        except Exception as exc:
            logger.error(
                f"mooncake-store load thread failed to start on rank {self._rank}: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._load_lock:
                if self._load_startup_error is None:
                    self._load_startup_error = exc
            return
        finally:
            started.set()
        stats = _LoadStats()
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            reservation_id, transfers, ready, queued_at = item
            began = time.monotonic()
            try:
                if ready is not None:
                    if stream is not None:
                        # The store fills host slots and may run ahead; only
                        # the scatters onto the pages wait for their release.
                        stream.wait_event(ready)
                    else:
                        ready.synchronize()
                loaded_bytes = self._load_pages([transfers], staging, handle)
            except Exception as exc:
                # Thread boundary: the executor thread raises it on its next
                # call, rather than leaving the request parked forever.
                logger.error(
                    f"mooncake-store rank {self._rank} failed the asynchronous load "
                    f"of reservation {reservation_id}: {type(exc).__name__}: {exc}"
                )
                with self._load_lock:
                    if self._load_error is None:
                        self._load_error = exc
                continue
            with self._load_lock:
                self._finished_prefix_loads.append(reservation_id)
            report = stats.add(
                len(transfers.pages), loaded_bytes, began - queued_at, time.monotonic() - began
            )
            if report:
                logger.info(
                    f"mooncake-store rank {self._rank} loader {index}: {report}, "
                    f"{self._load_queue.qsize()} queued"
                )

    def _load_pages(
        self, transfers: Sequence[RequestTransfers], staging: Optional[HostStagingPool], handle: int
    ) -> int:
        """Fetch pages batch by batch and land them on the device before returning.

        Shared by the executor thread (synchronous loads, on its stream) and the
        loader thread (on its own stream and staging pool).

        Returns:
            The bytes loaded.
        """
        keys, addresses, sizes, total_pages = self._resolve(transfers)
        if not keys:
            return 0

        for batch in zip(
            _batched(keys, self._batch_size),
            _batched(addresses, self._batch_size),
            _batched(sizes, self._batch_size),
        ):
            batch_keys, batch_addresses, batch_sizes = batch
            if staging is None:
                target_addresses, target_sizes = list(batch_addresses), list(batch_sizes)
            else:
                target_addresses, target_sizes = describe_batch_for_get(staging, batch_sizes)
            results = self._store.batch_get_into_multi_buffers(
                list(batch_keys), target_addresses, target_sizes
            )
            failed = [
                key
                for key, result in zip(batch_keys, results)
                if not isinstance(result, int) or result < 0
            ]
            if failed or len(results) != len(batch_keys):
                # The runtime already counted these tokens as computed, so a
                # partial load would leave the forward pass reading
                # uninitialized KV and silently producing wrong tokens.
                raise RuntimeError(
                    f"mooncake-store failed to load {len(failed) or len(batch_keys)} of "
                    f"{len(batch_keys)} pages; the affected KV slots were already "
                    f"reported as computed. First failure: {failed[:1]}"
                )
            if staging is not None:
                # Only reached once every page in the batch landed, so no slot
                # holding a failed read is copied over a device page.
                unstage_batch_after_get(staging, batch_addresses, batch_sizes, handle)
                # The next batch reuses the slots and the forward pass reads
                # these pages, so the scatter has to complete before either.
                _sync_stream(handle)

        logger.debug(f"mooncake-store rank {self._rank} loaded {total_pages} pages")
        return sum(sum(page_sizes) for page_sizes in sizes)

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """No-op: loads complete in `start_load_kv`.

        Transfers are whole pages, so every layer of a group lands in one
        store call and nothing is outstanding when the first layer runs.
        """

    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        """No-op: saves are submitted once per pass in `wait_for_save`.

        A page is only complete when every layer of its group has written its
        slice, so there is no correct per-layer submission point.
        """

    # ---- save path ----

    def wait_for_save(self, stream: torch.cuda.Stream):
        """Hand this pass's saves to the background thread, gated on an event."""
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None or not self._config.role.saves or not metadata.saves:
            return

        # The pages are written by kernels still queued on this stream, so the
        # event is the handoff: the thread reads GPU memory only after the
        # pass retires, without the executor loop waiting for it.
        event = torch.cuda.Event()
        event.record(stream)

        with self._save_lock:
            for transfers in metadata.saves:
                self._outstanding_saves[transfers.request_id] += 1
        self._save_queue.put((event, list(metadata.saves), time.monotonic()))

    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Report which requests' saves have landed.

        Args:
            finished_gen_req_ids: Requests that will produce no further KV.
            started_loading_req_ids: Requests loading asynchronously. Always
                empty here, since `get_num_new_matched_tokens` only offers
                synchronous loads; echoed back so the runtime does not wait on
                something that already happened.

        Returns:
            Requests that have finished saving, and requests that have finished
            loading.
        """
        with self._save_lock:
            self._closed_requests.update(finished_gen_req_ids)
            finished_saving = [
                request_id
                for request_id in self._closed_requests
                if self._outstanding_saves.get(request_id, 0) == 0
            ]
            for request_id in finished_saving:
                self._closed_requests.discard(request_id)
                self._outstanding_saves.pop(request_id, None)
        return finished_saving, list(started_loading_req_ids)

    def _drain_saves(self) -> None:
        try:
            # A new thread starts on device 0, so adopt the device captured on
            # the executor thread. Otherwise the stream below belongs to
            # device 0 while the KV pointers belong to the rank's device, and
            # the copy fails with cudaErrorInvalidValue on every rank but 0.
            if self._device_index is not None:
                torch.cuda.set_device(self._device_index)
            if self._save_staging is not None and torch.cuda.is_available():
                # Owned by this thread so the gather never queues behind the
                # executor's work.
                self._save_stream = torch.cuda.Stream()
        except Exception as exc:
            # Fatal, unlike a failure in the transfer loop below: a thread that
            # died here leaves every later save outstanding against nothing, so
            # the requests holding those pages never retire and the worker
            # looks merely slow.
            logger.error(
                f"mooncake-store save thread failed to start on rank {self._rank}: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._save_lock:
                if self._save_startup_error is None:
                    self._save_startup_error = exc
            return
        finally:
            self._save_started.set()
        stats = _SaveStats()
        while True:
            item = self._save_queue.get()
            if item is None:
                return
            event, transfers, queued_at = item
            began = time.monotonic()
            written = present = written_bytes = 0
            try:
                event.synchronize()
                written, present, written_bytes = self._put(transfers)
            except Exception as exc:
                # Broad on purpose: this is the thread boundary. The batch is
                # dropped and the request retired by the `finally` below, the
                # same policy `_put` applies to a negative store status.
                logger.error(
                    f"mooncake-store dropped a save batch on rank {self._rank}: "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
            finally:
                with self._save_lock:
                    for entry in transfers:
                        remaining = self._outstanding_saves.get(entry.request_id, 0) - 1
                        if remaining <= 0:
                            self._outstanding_saves.pop(entry.request_id, None)
                        else:
                            self._outstanding_saves[entry.request_id] = remaining
            report = stats.add(
                written, present, written_bytes, began - queued_at, time.monotonic() - began
            )
            if report:
                logger.info(
                    f"mooncake-store rank {self._rank} saves: {report}, {self._save_queue.qsize()} queued"
                )

    def _put(self, transfers: Sequence[RequestTransfers]) -> Tuple[int, int, int]:
        """Write the pages the pool does not hold yet.

        Returns:
            Pages written, pages already stored, and the bytes written.
        """
        keys, addresses, sizes, _ = self._resolve(transfers)
        written = present_count = written_bytes = 0
        if not keys:
            return written, present_count, written_bytes

        staging = self._save_staging
        handle = _stream_handle(self._save_stream) if staging is not None else 0

        for batch in zip(
            _batched(keys, self._batch_size),
            _batched(addresses, self._batch_size),
            _batched(sizes, self._batch_size),
        ):
            batch_keys, batch_addresses, batch_sizes = batch
            # Skip pages another rank or another instance already wrote. The
            # scheduler holds no store handle, and the answer changes between
            # the time it builds metadata and now.
            present = self._store.batch_is_exist(list(batch_keys))
            pending = [
                index
                for index, status in enumerate(present)
                if status != 1  # absent, or a failed probe we retry as a write
            ]
            present_count += len(batch_keys) - len(pending)
            if not pending:
                continue
            source_addresses = [batch_addresses[i] for i in pending]
            source_sizes = [batch_sizes[i] for i in pending]
            if staging is not None:
                # Gathered after the existence filter, so a page already in the
                # pool costs no copy.
                source_addresses, source_sizes = stage_batch_for_put(
                    staging, source_addresses, source_sizes, handle
                )
                # The store reads the slots on this thread, so fill them first.
                _sync_stream(handle)
            results = self._store.batch_put_from_multi_buffers(
                [batch_keys[i] for i in pending],
                source_addresses,
                source_sizes,
            )
            failures = sum(1 for result in results if not isinstance(result, int) or result < 0)
            written += len(pending) - failures
            written_bytes += sum(
                sum(source_sizes[i])
                for i, result in enumerate(results)
                if isinstance(result, int) and result >= 0
            )
            if failures:
                # A dropped write only costs a future cache miss, which is not
                # worth failing a request that already answered.
                logger.warning(
                    f"mooncake-store rank {self._rank} failed to save {failures} of "
                    f"{len(pending)} pages"
                )
        return written, present_count, written_bytes

    # ---- shared ----

    def _resolve(
        self, transfers: Sequence[RequestTransfers]
    ) -> Tuple[List[str], List[List[int]], List[List[int]], int]:
        """Expand per-request page transfers into parallel store call arguments."""
        if self._addressing is None:
            raise RuntimeError("KV cache layout has not been registered")
        keys: List[str] = []
        addresses: List[List[int]] = []
        sizes: List[List[int]] = []
        pages = 0
        for entry in transfers:
            for page in entry.pages:
                namespace = self._namespaces.get(page.layer_group_id)
                if namespace is None:
                    raise KeyError(
                        f"layer group {page.layer_group_id} is not in the registered "
                        "layout; the scheduler and worker disagree about the model"
                    )
                page_addresses, page_sizes = self._addressing.buffers(
                    page.layer_group_id, page.page_index
                )
                keys.append(namespace.key(page.block_hash))
                addresses.append(page_addresses)
                sizes.append(page_sizes)
                pages += 1
        return keys, addresses, sizes, pages

    def shutdown(self) -> None:
        """Stop the save and load threads, then release what they used. Idempotent."""
        threads = (
            [("save thread", self._save_thread, self._save_queue)] if self._save_thread else []
        )
        threads += [("load thread", thread, self._load_queue) for thread in self._load_threads]
        # One sentinel per thread: a loader stops at the first one it takes,
        # after the loads queued ahead of it.
        for _, _, queue in threads:
            queue.put(None)
        for label, thread, _ in threads:
            thread.join(timeout=SAVE_DRAIN_TIMEOUT)
            if thread.is_alive():
                # The thread reads the KV pools through the store handle and
                # the staging slots below, so releasing either takes the memory
                # out from under a transfer in flight. Leaking both is the safer
                # end: the process is going down anyway, the thread is a daemon
                # and does not hold it open, and a later call retries the join.
                logger.error(
                    f"mooncake-store rank {self._rank}: the {label} "
                    f"did not stop within {SAVE_DRAIN_TIMEOUT:g}s, so the store handle and its "
                    "staging buffers are left in place rather than freed under a "
                    "transfer still reading them."
                )
                return
        self._save_thread = None
        self._load_threads = []
        if self._probe_pool is not None:
            # Lookups run on the scheduler thread, which is done by now.
            self._probe_pool.shutdown(wait=True)
            self._probe_pool = None
        store, self._store = self._store, None
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                logger.warning(
                    f"mooncake-store close failed: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
        # Released only after the store is closed, since it holds registrations
        # against this memory.
        self._load_staging = None
        self._async_load_stagings = []
        self._save_staging = None
        self._save_stream = None
        global _LOCAL_WORKER
        if _LOCAL_WORKER is self:
            _LOCAL_WORKER = None
            _LOCAL_WORKER_READY.clear()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:  # noqa: S110 - interpreter teardown, nothing left to report to
            pass
