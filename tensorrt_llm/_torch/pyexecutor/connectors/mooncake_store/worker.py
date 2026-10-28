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
scheduler adapter in the same process asks it to run prefix lookups: rank 0's
under tensor parallelism, every owner's under attention data parallelism.

A stored page is one attention shard's KV. Under tensor parallelism each rank
keys its own shard and a prefix counts only when every shard is present. Under
attention data parallelism every owner holds all heads, so all owners key their
pages as shard 0 of 1 and share one key space on the pool, while transfers,
completion and the store handle stay owner-local.

Loads are synchronous by default, and the runtime has already counted the
loaded tokens as computed, so a page missing at load time must never reach a
forward pass. A rank that holds whole pages and its own batch hands the
affected request back for a restart without the offered prefix
(`reports_failed_loads`): a single-rank server or an attention-DP owner. Under
tensor parallelism the ranks hold shards of the same page and cannot restart a
request alone, so there a missing page fails the executor.

With `async_load`, the runtime parks a request the pool serves outside the
batch instead, and `async_load_workers` background threads per rank move its
pages: each waits for the KV cache manager's allocation fence on its own
thread, then fetches, and `get_finished` releases the request once its pages
landed. A missing page is handed back or fails the executor as above. Any
exception on a load thread is fatal at the next executor call, and stays so.

Saves are asynchronous and gated on a CUDA event, since the pages are complete
only once the forward pass that wrote them has retired and blocking the
executor loop on an RDMA write is the cost the store exists to avoid. The
scheduler reports such a request as saving asynchronously, which keeps its
pages pinned until `get_finished` says the writes landed. It counts a save as
stored once `wait_for_save` has queued it, so saves bound for a pass that never
ran are emitted again. A save that fails is logged and dropped: a write that
did not land costs a future cache miss and nothing else, so the request retires
either way. Only the save thread failing to start is fatal, because that would
strand every request behind pages nothing will release. With `soft_pin_ttl_ms`
or `group_blocks` set, each put also carries a Mooncake `ReplicateConfig` that
soft-pins its pages or names each page's eviction group.

Next to one-model MTP or EAGLE3 on a disaggregated context server, the draft
layers share the target pool, so they move with every page. Any other drafter,
or draft layers in a separate KV cache, is refused at construction.

A capacity-only worker opens its handle and stops there, with no layout, no
buffer registration and no save thread, so a node can lend host memory to the
pool without an HCA that can pin GPU pages.

With `stats_interval` set, a role that reads the pool logs two info-level lines
per interval: lookups that stopped at a hole the pool evicted from a prefix it
otherwise still holds, and the time each phase of a load took.
"""

import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from queue import Queue
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger

from ..kv_cache_connector import AllocationFence, KvCacheConnectorWorker
from ..kv_cache_layout import KvCacheLayout
from .addressing import PageAddressing
from .config import (
    CONFIG_PATH_ENV,
    MooncakeStoreConnectorConfig,
    local_address,
    local_reuse_backoff,
    pool_config,
)
from .gpudirect import PEERMEM_ENV, allocation_range, format_diagnosis, select_registration_path
from .keys import KeyNamespace
from .ledger import record_segment
from .metadata import MooncakeStoreMetadata, RequestTransfers
from .staging import (
    HostStagingPool,
    describe_batch_for_get,
    plan_slot_geometry,
    stage_batch_for_put,
    unstage_batch_after_get,
)
from .staging import sync_stream as _sync_stream
from .validation import (
    validate_layout,
    validate_llm_args,
    validate_node_budget,
    validate_speculative_config,
)

__all__ = ["MooncakeStoreConnectorWorker", "resolve_local_worker"]

_GIB = 1 << 30

#: Set by the worker's constructor so the scheduler adapter, which the executor
#: builds in the same process (on rank 0 under tensor parallelism, on every
#: owner under attention DP), can reach the store handle without a second
#: connection. `py_executor_creator` constructs the two concurrently, so the
#: adapter waits on `_LOCAL_WORKER_READY`.
_LOCAL_WORKER: Optional["MooncakeStoreConnectorWorker"] = None
_LOCAL_WORKER_READY = threading.Event()

#: Seconds `shutdown` gives the save thread and the load threads, against one
#: deadline, to drain and return. Sized for a backlog of pages over a congested
#: fabric rather than a healthy one, since what follows the wait is only safe
#: once every thread has stopped.
SAVE_DRAIN_TIMEOUT = 30.0

#: One parked request's load, as the load threads take it off their queue: the
#: request id, its page transfers, when it was queued, and the allocation fence
#: bound in the iteration that parked it. The fence is never None; see
#: `MooncakeStoreConnectorWorker._start_async_load`.
_QueuedLoad = Tuple[int, RequestTransfers, float, AllocationFence]


def resolve_local_worker(timeout: float = 60.0) -> "MooncakeStoreConnectorWorker":
    """The worker living in this process, once it has been constructed.

    Args:
        timeout: Seconds to wait. Exceeding it means construction failed.
    """
    if not _LOCAL_WORKER_READY.wait(timeout):
        raise RuntimeError(
            "The mooncake-store scheduler adapter could not find a worker in its "
            "process. Each adapter (rank 0 under tensor parallelism, every owner "
            "under attention DP) is built next to a worker, so this means worker "
            "construction failed."
        )
    assert _LOCAL_WORKER is not None
    return _LOCAL_WORKER


def _open_store(config: MooncakeStoreConnectorConfig):
    """Connect to the Mooncake master and return a live store handle.

    Returns:
        The store handle, and the host the segment is registered under.
    """
    # Every role selects, since the first client in the process fixes the path
    # for all of them.
    if select_registration_path(
        config.protocol,
        registers_device_memory=not (config.capacity_only or config.stage_through_host),
    ):
        logger.warning(
            "mooncake-store: no peer-memory module is loaded and every visible "
            f"device supports dma-buf, so {PEERMEM_ENV}=0 is set to register the "
            "KV pools through dma-buf. Set it explicitly to choose otherwise."
        )
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


def _replicate_config(soft_pin_ttl_ms: Optional[int], group_ids: Optional[Sequence[str]] = None):
    """The Mooncake `ReplicateConfig` for one put call, or None for the store's defaults.

    Args:
        soft_pin_ttl_ms: Soft-pin every page of the call for this long, if set.
        group_ids: One eviction group per key of the call, if set.

    Returns:
        A `mooncake.store.ReplicateConfig`, or None when neither option is set,
        in which case the put keeps its three-argument form.

    Raises:
        ImportError: When the installed bindings predate either option.
    """
    if soft_pin_ttl_ms is None and group_ids is None:
        return None
    try:
        import mooncake.store as mooncake_store

        config = mooncake_store.ReplicateConfig()
        if soft_pin_ttl_ms is not None:
            config.soft_pin_action = mooncake_store.SoftPinAction.ENABLE
            config.soft_pin_ttl_ms = soft_pin_ttl_ms
        if group_ids is not None:
            config.group_ids = list(group_ids)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "mooncake_store.soft_pin_ttl_ms and group_blocks need "
            "mooncake-transfer-engine-cuda13 0.3.13 or newer, whose mooncake.store "
            "exposes SoftPinAction and a ReplicateConfig with soft_pin_action, "
            "soft_pin_ttl_ms and group_ids"
        ) from exc
    return config


#: The parts of a prefix load timed separately, in the order the stats line
#: lists them. The synchronous path runs them as: resolve keys, wait for the
#: allocation fence (zero copy only), describe each batch's destinations, fetch
#: it, then scatter and sync a staged batch. An asynchronous load waits for the
#: fence first, on its load thread, whether zero copy or staged.
_LOAD_PHASES = ("fence", "resolve", "describe", "get", "unstage", "sync")


@contextmanager
def _timed_phase(phases: Optional[Dict[str, List[float]]], name: str):
    """Add the block's wall and thread CPU seconds to `phases[name]`.

    Args:
        phases: Per-phase `[wall, cpu]` totals of one load, or None to time
            nothing, which costs no clock read.
        name: One of `_LOAD_PHASES`.
    """
    if phases is None:
        yield
        return
    wall, cpu = time.perf_counter(), time.thread_time()
    try:
        yield
    finally:
        totals = phases.setdefault(name, [0.0, 0.0])
        totals[0] += time.perf_counter() - wall
        totals[1] += time.thread_time() - cpu


class _LoadStats:
    """Prefix loads, summarised once per `stats_interval`.

    A load is one pass that moved pages: a `start_load_kv` call with pages to
    fetch, or one parked request's asynchronous load. Every method takes the
    lock, because asynchronous loads record from their load threads into the
    same counters the executor thread logs.
    """

    __slots__ = (
        "_lock",
        "loads",
        "pages",
        "batches",
        "loaded_bytes",
        "failed_pages",
        "xfer_sum",
        "xfer_max",
        "waits",
        "wait_sum",
        "wait_max",
        "phase_wall",
        "phase_cpu",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.loads = 0
        self.pages = 0
        self.batches = 0
        self.loaded_bytes = 0
        self.failed_pages = 0
        self.xfer_sum = 0.0
        self.xfer_max = 0.0
        self.waits = 0
        self.wait_sum = 0.0
        self.wait_max = 0.0
        self.phase_wall = dict.fromkeys(_LOAD_PHASES, 0.0)
        self.phase_cpu = dict.fromkeys(_LOAD_PHASES, 0.0)

    def record(
        self,
        xfer_s: float,
        pages: int,
        batches: int,
        loaded_bytes: int,
        phases: Optional[Dict[str, List[float]]] = None,
        *,
        failed_pages: int = 0,
        wait_s: Optional[float] = None,
    ) -> None:
        """Fold one load into the interval.

        Args:
            xfer_s: Seconds from the destinations being ready to the last page
                landing.
            pages: Pages the load asked for.
            batches: Store calls it took.
            loaded_bytes: Bytes that landed.
            phases: Its per-phase `[wall, cpu]` seconds, from `_timed_phase`.
            failed_pages: Pages that did not land.
            wait_s: For an asynchronous load, seconds from being queued to its
                destinations being ready: the queue plus the fence.
        """
        with self._lock:
            self.loads += 1
            self.pages += pages
            self.batches += batches
            self.loaded_bytes += loaded_bytes
            self.failed_pages += failed_pages
            self.xfer_sum += xfer_s
            self.xfer_max = max(self.xfer_max, xfer_s)
            if wait_s is not None:
                self.waits += 1
                self.wait_sum += wait_s
                self.wait_max = max(self.wait_max, wait_s)
            for name, (wall, cpu) in (phases or {}).items():
                self.phase_wall[name] = self.phase_wall.get(name, 0.0) + wall
                self.phase_cpu[name] = self.phase_cpu.get(name, 0.0) + cpu

    def take_summary(self, *, workers: int = 0, queued: int = 0, active: int = 0) -> Optional[str]:
        """The interval's summary, starting a new interval.

        Args:
            workers: Asynchronous load threads, or 0 when loads are synchronous,
                which leaves the asynchronous fields out.
            queued: Asynchronous loads waiting for a load thread right now.
            active: Asynchronous loads a load thread is running right now.

        Returns:
            The summary, or None when nothing loaded and nothing is waiting.
        """
        with self._lock:
            if not (self.loads or queued or active):
                return None
            n = max(1, self.loads)
            phases = " ".join(
                f"{name}={1000 * self.phase_wall[name] / n:.1f}/"
                f"{1000 * self.phase_cpu[name] / n:.1f}"
                for name in _LOAD_PHASES
            )
            waits = max(1, self.waits)
            asynchronous = (
                f"wait_ms mean={1000 * self.wait_sum / waits:.1f} "
                f"max={1000 * self.wait_max:.1f} "
                f"workers={workers} queue={queued} active={active} "
                if workers
                else ""
            )
            summary = (
                f"loads={self.loads} pages={self.pages} batches={self.batches} "
                f"loaded_gib={self.loaded_bytes / _GIB:.2f} "
                f"failed_pages={self.failed_pages} "
                f"transfer_ms mean={1000 * self.xfer_sum / n:.1f} "
                f"max={1000 * self.xfer_max:.1f} "
                f"{asynchronous}"
                f"phase_ms(wall/cpu per load) {phases}"
            )
            self._reset()
        return summary


class _PrefixHoleStats:
    """Prefix lookups that stopped at a hole, summarised once per `stats_interval`.

    A hole is a missing block followed by stored blocks of the same chain: the
    pool evicted a block from the middle of a chain it otherwise still holds.
    The lookup stops at the hole, so the request recomputes every token behind
    it, including the stored ones counted here as holed.
    """

    __slots__ = (
        "lookups",
        "lookup_blocks",
        "hit_blocks",
        "holes",
        "hole_blocks",
        "holed_tokens",
        "max_holed_tokens",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.lookups = 0
        self.lookup_blocks = 0
        self.hit_blocks = 0
        self.holes = 0
        self.hole_blocks = 0
        self.holed_tokens = 0
        self.max_holed_tokens = 0

    def record(
        self,
        num_blocks: int,
        hit_blocks: int,
        stored_after_hole: Sequence[int],
        tokens_per_block: int,
    ) -> None:
        """Account for one lookup.

        Args:
            num_blocks: Blocks looked up.
            hit_blocks: Usable prefix length, which is also the hole's index.
            stored_after_hole: Ascending indices of fully stored blocks after
                the first missing one.
            tokens_per_block: Tokens per block.
        """
        self.lookups += 1
        self.lookup_blocks += num_blocks
        self.hit_blocks += hit_blocks
        if not stored_after_hole:
            return
        self.holes += 1
        span = stored_after_hole[-1] + 1 - hit_blocks
        self.hole_blocks += span - len(stored_after_hole)
        holed_tokens = len(stored_after_hole) * tokens_per_block
        self.holed_tokens += holed_tokens
        self.max_holed_tokens = max(self.max_holed_tokens, holed_tokens)

    def summary(self) -> str:
        return (
            f"lookups={self.lookups} lookup_blocks={self.lookup_blocks} "
            f"hit_blocks={self.hit_blocks} holes={self.holes} "
            f"hole_blocks={self.hole_blocks} holed_tokens={self.holed_tokens} "
            f"max_holed_tokens={self.max_holed_tokens}"
        )


class MooncakeStoreConnectorWorker(KvCacheConnectorWorker):
    """Moves KV pages between this rank's GPU cache and the Mooncake pool."""

    #: Transfers, completion and the store handle are owner-local, and under
    #: attention DP every owner keys its pages as attention shard 0 of 1, so all
    #: owners share one key space on the pool.
    supports_attention_dp = True
    #: A page offered at lookup can be gone by load time, evicted from a full
    #: pool or left behind by a put that never completed. On a single rank or an
    #: attention-DP owner the affected requests are handed back to the executor
    #: to restart (`take_failed_load_requests`). Tensor-parallel ranks hold
    #: shards of one page and cannot restart a request alone, so there a failed
    #: page stays fatal.
    reports_failed_loads = True
    #: Loads and saves move whole pages of every layer group in the registered
    #: layout, which includes one-model draft layers sharing the target pool;
    #: the per-layer hooks are no-ops. `validate_speculative_config` refuses a
    #: drafter whose KV lives anywhere else.
    supports_speculative_prefill = True
    #: Registering pages drops the automatic host tier, and a prefix restore is
    #: chosen only after the final batch, so on a disaggregated server the
    #: pages each admitted request needs to finish have to stay free.
    requires_completion_reservation = True

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.resolve(llm_args)
        # Before the store is opened, so a drafter this connector cannot key
        # fails before a segment is mounted.
        validate_speculative_config(llm_args, self._config)
        loads = self._config.role.loads
        #: Read by `KVCacheManagerV2`, which is built after the connector. An
        #: offer starts at the local match and covers whole blocks, so a role
        #: that reads the pool needs that match to end on a block boundary.
        #: Producer and capacity roles never offer and keep the manager's own
        #: reuse unchanged. Every rank resolves the same role and arguments, so
        #: the ranks of a server agree.
        self.requires_block_aligned_reuse = loads
        #: See `MooncakeStoreConfig.local_reuse_backoff`: "off" takes local
        #: matches whole instead of aligning a backed-off one down.
        self.drops_reuse_match_backoff = loads and local_reuse_backoff(llm_args) == "off"
        # Where this rank records the segment it mounts. `None` when the
        # deployment named no directory, which only makes the run
        # unreportable; see `ledger.record_segment`.
        pool = pool_config(llm_args)
        self._run_dir = pool.run_dir if pool is not None else None
        self._rank = mpi_rank()
        self._world_size = mpi_world_size()
        enable_attention_dp = getattr(llm_args, "enable_attention_dp", False) is True
        # A stored page is one attention shard's KV. Every attention-DP owner
        # holds all heads, so all owners key shard 0 of 1 on the shared pool;
        # under tensor parallelism each rank keeps its own shard. The global
        # rank still names this process in the ledger, thread names and logs.
        if enable_attention_dp:
            self._attention_rank, self._attention_world_size = 0, 1
        else:
            self._attention_rank, self._attention_world_size = self._rank, self._world_size
        self._model_key = self._config.resolve_model_key(llm_args.model)

        self._addressing: Optional[PageAddressing] = None
        # Namespaces for this rank's attention shard, used for both directions
        # of transfer.
        self._namespaces: Dict[int, KeyNamespace] = {}
        # The same namespaces for every attention shard. A prefix is only
        # reusable when all shards of it are present, so a lookup has to ask
        # about all of them: every tensor-parallel rank's, or the single shard
        # every attention-DP owner shares.
        self._peer_namespaces: Dict[int, Tuple[KeyNamespace, ...]] = {}
        # A rank that holds whole pages and its own batch can restart a request
        # whose page went missing: a single-rank server, or an attention-DP
        # owner. Tensor-parallel ranks hold shards of the same page and would
        # have to agree on the restart, which they cannot do here, so a failed
        # load stays fatal there.
        holds_whole_pages = enable_attention_dp or self._world_size == 1
        self._recover_failed_loads = (
            getattr(type(self), "reports_failed_loads", False) is True and holds_whole_pages
        )
        # Requests whose last synchronous load came up short, taken by the
        # executor right after `start_load_kv`. Executor thread only.
        self._failed_load_requests: Set[int] = set()
        # Diagnostics, off unless `stats_interval` is set. Only a role that
        # reads the pool looks prefixes up or loads them, so the others count
        # nothing.
        self._stats_interval = self._config.stats_interval
        counts = self._stats_interval is not None and loads
        self._hole_stats: Optional[_PrefixHoleStats] = _PrefixHoleStats() if counts else None
        self._hole_stats_last = time.monotonic()
        self._load_stats: Optional[_LoadStats] = _LoadStats() if counts else None
        self._load_stats_last = time.monotonic()

        # Asynchronous loads; see `_start_async_load`. Only a role that reads
        # the pool loads at all, so producer and capacity roles start no load
        # thread and pin no load staging.
        self._async_load = self._config.async_load and loads
        self._async_workers = self._config.async_load_workers if self._async_load else 0
        # One shared FIFO: an idle load thread takes the next request, so each
        # request is handled by exactly one thread, in arrival order.
        self._load_queue: "Queue[Optional[_QueuedLoad]]" = Queue()
        self._load_threads: List[threading.Thread] = []
        # Why a load thread never reached its loop, read back by
        # `_start_load_threads`.
        self._load_startup_errors: List[BaseException] = []
        # Set at shutdown, so a load still queued never reaches the store.
        self._load_stop = threading.Event()
        # Guards the asynchronous bookkeeping below.
        self._load_lock = threading.Lock()
        # One staging pool per load thread, under `stage_through_host`.
        self._async_stagings: List[HostStagingPool] = []
        # Staging pools whose unregistration failed, kept referenced so the
        # fabric never reaches memory the allocator has handed out again.
        self._unreleased_staging: List[HostStagingPool] = []
        # Queued or loading, by request.
        self._async_pending: Set[int] = set()
        # Taken by a load thread and loading right now.
        self._async_active: Set[int] = set()
        # Done but not yet reported: request -> whether a page failed to land.
        self._async_done: Dict[int, bool] = {}
        # Requests the runtime said it parked (`started_loading_req_ids`).
        self._async_announced: Set[int] = set()
        # Reported finished with a page missing; see
        # `take_failed_async_load_requests`.
        self._failed_async_load_requests: Set[int] = set()
        # The first exception on a load thread, or the first load this rank
        # could not queue. Sticky: every later executor call raises it, until
        # shutdown; see `_reraise_background_error`.
        self._load_error: Optional[BaseException] = None

        # Transfer state, set before the store is opened so that `shutdown`
        # also runs on a worker whose store never opened.
        self._store = None
        self._save_queue: "Queue[Optional[Tuple[torch.cuda.Event, List[RequestTransfers]]]]" = (
            Queue()
        )
        self._save_thread: Optional[threading.Thread] = None
        #: Set once the save thread has finished its device and stream setup,
        #: whether or not that succeeded.
        self._save_started = threading.Event()
        self._save_lock = threading.Lock()
        # KV pool ranges registered with the store, in registration order.
        self._registered_ranges: List[Tuple[int, int]] = []
        # Host staging, when the pool cannot register device memory.
        self._load_staging: Optional[HostStagingPool] = None
        self._save_staging: Optional[HostStagingPool] = None
        self._save_stream: Optional[torch.cuda.Stream] = None
        # This rank's device, captured on the executor thread; see _drain_saves.
        self._device_index: Optional[int] = None
        # Pages per store call. Staging narrows this to the slots it can afford.
        self._batch_size = self._config.transfer_batch_size
        # Pages per store call on a load thread; its own staging narrows it.
        self._async_batch_size = self._batch_size
        # Save submissions still in flight, per request.
        self._outstanding_saves: Dict[int, int] = defaultdict(int)
        # Requests the runtime has told us are done producing KV. Their pages
        # stay pinned until we report them back through `get_finished`.
        self._closed_requests: Set[int] = set()
        # Requests whose saves the last `wait_for_save` queued. The scheduler
        # adapter takes them at its next `build_connector_meta` and only then
        # counts those blocks as stored; see `take_submitted_saves`. Executor
        # thread only.
        self._submitted_saves: Set[int] = set()
        # Why the save thread never reached its transfer loop, read back by
        # `register_kv_cache_layout`. A failure inside the loop is dropped
        # there instead.
        self._save_startup_error: Optional[BaseException] = None

        # Checked before the segment is mounted, since the kernel answers an
        # unaffordable total by killing the process rather than by failing the
        # allocation.
        validate_node_budget(self._config)

        # Write options exist only for a role that writes, so consumer and
        # capacity roles build nothing and need no particular binding.
        saves = self._config.role.saves
        self._soft_pin_ttl_ms = self._config.soft_pin_ttl_ms if saves else None
        self._group_blocks = self._config.group_blocks and saves
        # Shared by every put; None keeps the store's defaults. Grouped puts
        # need one group per key and build their own per call, and building a
        # probe here proves the binding supports them before a segment is
        # mounted.
        self._put_config = _replicate_config(
            self._soft_pin_ttl_ms, ["probe"] if self._group_blocks else None
        )

        self._store, self._segment_host = _open_store(self._config)
        record_segment(
            self._run_dir,
            host=self._segment_host,
            rank=self._rank,
            segment_size=self._config.global_segment_size,
            role=self._config.role.value,
            model_key=self._model_key,
        )

        global _LOCAL_WORKER
        _LOCAL_WORKER = self
        _LOCAL_WORKER_READY.set()

        logger.warning(
            f"mooncake-store worker rank {self._rank}/{self._world_size} ready "
            f"(attention shard {self._attention_rank}/{self._attention_world_size}, "
            f"role={self._config.role.value}, model_key={self._model_key}, "
            f"master={self._config.master_server_address}, "
            f"segment={self._config.global_segment_size / (1 << 30):.1f} GiB "
            f"on {self._segment_host}"
            f"{f', soft_pin_ttl_ms={self._soft_pin_ttl_ms}' if self._soft_pin_ttl_ms is not None else ''}"
            f"{', group_blocks=1' if self._group_blocks else ''}"
            f"{f', stats_interval={self._stats_interval:g}s' if self._hole_stats is not None else ''}"
            f"{f', async_load workers={self._async_workers}' if self._async_load else ''})"
        )
        if self.capacity_only:
            # Said plainly, since every other sign of a working connector is
            # absent by design and otherwise looks like a broken deployment.
            logger.warning(
                f"mooncake-store worker rank {self._rank}: capacity-only. Its "
                "memory is in the pool and its KV cache is not: no lookups, no "
                "loads, no saves, and no KV registration, so this rank needs no "
                "GPUDirect RDMA."
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
        """Register the KV pools with Mooncake and start the transfer threads.

        Raises:
            RuntimeError: When the save thread or an asynchronous load thread
                cannot start, since a request would then wait on a transfer
                nothing runs.
        """
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
        # Torch's current device is thread-local, so read it here on the
        # executor thread; the save thread would otherwise see device 0.
        if torch.cuda.is_available():
            self._device_index = torch.cuda.current_device()
        if self._config.stage_through_host:
            addressing = PageAddressing(layout)
            self._open_staging(addressing)
        else:
            addressing = PageAddressing(layout, allocation_lookup=allocation_range)
            self._register_pools(addressing)

        self._addressing = addressing
        for layer_group_id in addressing.layer_group_ids:
            bytes_per_page = addressing.bytes_per_page(layer_group_id)
            self._namespaces[layer_group_id] = self._namespace(
                self._attention_rank, layer_group_id, bytes_per_page
            )
            self._peer_namespaces[layer_group_id] = tuple(
                self._namespace(shard, layer_group_id, bytes_per_page)
                for shard in range(self._attention_world_size)
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
        if self._async_load:
            self._start_load_threads()

        logger.warning(
            f"mooncake-store worker rank {self._rank} registered layout: {addressing.describe()}"
            f"{f' ({self._async_workers} asynchronous load workers)' if self._async_load else ''}"
        )

    def _start_load_threads(self) -> None:
        """Start the asynchronous load threads and wait until each has set up.

        Raises:
            RuntimeError: When any of them could not bind this rank's device or
                open its stream, since a parked request would then wait on a
                load nothing runs.
        """
        started: List[threading.Event] = []
        for index in range(self._async_workers):
            ready = threading.Event()
            thread = threading.Thread(
                target=self._drain_loads,
                args=(index, ready),
                name=f"mooncake-store-load-{self._rank}-{index}",
                daemon=True,
            )
            self._load_threads.append(thread)
            started.append(ready)
            thread.start()
        for ready in started:
            ready.wait()
        with self._load_lock:
            errors, self._load_startup_errors = self._load_startup_errors, []
        if errors:
            raise RuntimeError(
                f"mooncake-store: {len(errors)} of {self._async_workers} load thread(s) "
                f"for rank {self._rank} could not start, so a parked request would "
                "wait on a load nothing runs."
            ) from errors[0]

    def _register_pools(self, addressing: PageAddressing) -> None:
        """Register the KV pools with Mooncake, one call per allocation.

        All or nothing: a failed piece unregisters the pieces before it.

        Registered once for the life of the worker, which is valid only while
        the pool memory stays mapped. KV pool rebalance would remap it, so
        `PyExecutor._can_pause_for_rebalance` refuses it under a connector.
        """
        ranges = addressing.registration_ranges()
        logger.info(
            f"mooncake-store rank {self._rank} registering {len(ranges)} range(s) "
            f"covering {sum(end - start for start, end in ranges) / _GIB:.1f} GiB"
        )
        if addressing.unresolved_ranges:
            logger.warning(
                f"mooncake-store rank {self._rank} could not ask the driver which "
                "cuMemCreate allocations back "
                f"{len(addressing.unresolved_ranges)} KV pool range(s), so they "
                "are registered whole. A range spanning several allocations "
                "fails registration below."
            )
        # The report queries the driver per range, so build it only when shown.
        if logger.is_debug_enabled():
            logger.debug(format_diagnosis(ranges, self._rank))
        for start, end in ranges:
            status = self._store.register_buffer(start, end - start)
            if status != 0:
                self._unregister_pools()
                # Collected only on failure, so the common path pays nothing.
                raise RuntimeError(
                    f"MooncakeDistributedStore.register_buffer failed with status "
                    f"{status} for [{start:#x}, {end:#x}). Without registration "
                    "the store cannot read or write these pages. Registering "
                    "device memory needs GPUDirect RDMA, and what that requires "
                    f"depends on {PEERMEM_ENV}. Unless it is 0, Mooncake calls "
                    "ibv_reg_mr, which needs the nvidia_peermem module and KV "
                    "pools allocated with gpuDirectRDMACapable=1; either missing "
                    "fails with EFAULT, as does a range crossing two allocations. "
                    f"With {PEERMEM_ENV}=0 it registers through dma-buf, which "
                    "needs neither and fails a crossing range with EINVAL. Set "
                    "stage_through_host to pass pages through pinned host memory "
                    "instead.\n"
                    f"{format_diagnosis(ranges, self._rank)}"
                )
            self._registered_ranges.append((start, end))

    def _unregister_pools(self) -> None:
        """Hand back every KV pool registration, latest first. Idempotent."""
        while self._registered_ranges:
            start, end = self._registered_ranges.pop()
            status = self._store.unregister_buffer(start)
            if status != 0:
                # The memory is the cache manager's, so there is nothing to
                # keep alive here. Log the leak.
                logger.error(
                    f"mooncake-store rank {self._rank}: unregister_buffer failed "
                    f"with status {status} for [{start:#x}, {end:#x})."
                )

    def _open_staging(self, addressing: PageAddressing) -> None:
        """Allocate and register the pinned slots pages will pass through.

        Only the directions this role drives get a pool, since each one costs
        a pinned allocation of its own. Each direction gets
        `staging_buffer_bytes`. With `async_load` the load direction is the load
        threads' pools, which split that budget, and no synchronous load pool
        is pinned: the node memory check counts one budget per direction.

        Pools are created loads first, then saves. A pool that fails to
        register leaves the ones before it in place for `shutdown` to release.
        """
        max_bytes_per_page = max(
            addressing.bytes_per_page(layer_group_id)
            for layer_group_id in addressing.layer_group_ids
        )
        batch = self._config.transfer_batch_size
        budget = self._config.staging_buffer_bytes
        slot_bytes, num_slots = plan_slot_geometry(max_bytes_per_page, batch, budget)
        loads, saves = self._config.role.loads, self._config.role.saves
        self._warn_raised_to_one_page(
            [
                label
                for label, used in (("load", loads and not self._async_load), ("save", saves))
                if used
            ],
            budget,
            max_bytes_per_page,
        )
        if loads and not self._async_load:
            self._load_staging = HostStagingPool(
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                store=self._store,
                label="load",
            )
        if self._async_load:
            share = budget // self._async_workers
            _, worker_slots = plan_slot_geometry(max_bytes_per_page, batch, share)
            self._warn_raised_to_one_page(
                [f"async-load-{index}" for index in range(self._async_workers)],
                share,
                max_bytes_per_page,
                split=self._async_workers > 1,
            )
            for index in range(self._async_workers):
                self._async_stagings.append(
                    HostStagingPool(
                        slot_bytes=slot_bytes,
                        num_slots=worker_slots,
                        store=self._store,
                        label=f"async-load-{index}",
                    )
                )
            self._async_batch_size = min(batch, worker_slots)
            if worker_slots < min(64, batch):
                logger.warning(
                    f"mooncake-store rank {self._rank}: each of its {self._async_workers} "
                    f"load workers stages only {worker_slots} page(s) of "
                    f"{max_bytes_per_page} B per store call, so per-call overhead "
                    "dominates its loads. Raise staging_buffer_bytes or lower "
                    "async_load_workers."
                )
        if saves:
            self._save_staging = HostStagingPool(
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                store=self._store,
                label="save",
            )
        # Saves, and synchronous loads where they have a pool.
        self._batch_size = min(batch, num_slots)
        if self._batch_size < batch and (saves or self._load_staging is not None):
            logger.warning(
                f"mooncake-store rank {self._rank} reduced its transfer batch from "
                f"{batch} to {self._batch_size} pages: staging {max_bytes_per_page} B "
                f"pages within staging_buffer_bytes={budget} does not fit more. Raise "
                "staging_buffer_bytes or lower transfer_batch_size."
            )

    def _warn_raised_to_one_page(
        self, labels: List[str], budget: int, page_bytes: int, *, split: bool = False
    ) -> None:
        """Say that pools whose budget is below one page pin a page each regardless.

        Args:
            labels: The pools sharing this budget.
            budget: Pinned bytes each of them was given.
            page_bytes: The largest page, which a slot has to hold.
            split: Whether `budget` is one load thread's share of
                `staging_buffer_bytes`, which fewer load threads would raise.
        """
        if not labels or budget >= page_bytes:
            return
        logger.warning(
            f"mooncake-store rank {self._rank}: the {', '.join(labels)} staging "
            f"pool(s) get {budget} B each, below one {page_bytes} B page, so each is "
            "raised to one page and pins more than the node memory check counted "
            "for staging_buffer_bytes. Raise staging_buffer_bytes"
            f"{' or lower async_load_workers' if split else ''}."
        )

    def _namespace(self, shard: int, layer_group_id: int, bytes_per_page: int) -> KeyNamespace:
        """The key namespace of one attention shard's pages of a layer group."""
        return KeyNamespace(
            namespace=self._config.namespace,
            model_key=self._model_key,
            rank=shard,
            world_size=self._attention_world_size,
            layer_group_id=layer_group_id,
            tokens_per_block=self._addressing.tokens_per_block,
            bytes_per_page=bytes_per_page,
        )

    # ---- lookup, for the scheduler adapter in this process ----

    @property
    def config(self) -> MooncakeStoreConnectorConfig:
        """The resolved connector configuration."""
        return self._config

    @property
    def is_registered(self) -> bool:
        """Whether a KV cache layout has been registered yet."""
        return self._addressing is not None

    def count_prefix_hit(self, block_hashes: Sequence[bytes]) -> int:
        """How many leading blocks of `block_hashes` are fully present.

        A block counts only when every layer group and every attention shard
        has its page, since a prefix is replayed as a whole. The scan stops at
        the first incomplete block, a later hit being unusable on its own.

        Args:
            block_hashes: Candidate hashes in block order.

        Returns:
            Length of the usable prefix, in blocks.
        """
        if not block_hashes or self._addressing is None:
            return 0

        keys: List[str] = []
        for block_hash in block_hashes:
            for namespaces in self._peer_namespaces.values():
                keys.extend(namespace.key(block_hash) for namespace in namespaces)
        keys_per_block = len(keys) // len(block_hashes)

        try:
            present = self._store.batch_is_exist(keys)
        except Exception as exc:
            logger.warning(
                f"mooncake-store lookup failed; treating as a miss: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            return 0

        if len(present) != len(keys):
            logger.warning(
                f"mooncake-store batch_is_exist returned {len(present)} results for "
                f"{len(keys)} keys; treating as a miss"
            )
            return 0

        hit_blocks = 0
        for index in range(len(block_hashes)):
            window = present[index * keys_per_block : (index + 1) * keys_per_block]
            # Mooncake reports 1 for present, 0 for absent and a negative value
            # for a failed probe. Anything but a definite 1 is treated as a miss.
            if not all(status == 1 for status in window):
                break
            hit_blocks += 1
        if self._hole_stats is not None:
            # From the same answer, so the diagnostics cost no store call.
            self._record_prefix_hole(present, keys_per_block, len(block_hashes), hit_blocks)
        return hit_blocks

    def _record_prefix_hole(
        self, present: Sequence[int], keys_per_block: int, num_blocks: int, hit_blocks: int
    ) -> None:
        """Count the stored blocks behind the first missing one. Never changes the hit."""
        stored_after_hole = [
            index
            for index in range(hit_blocks + 1, num_blocks)
            if all(
                status == 1
                for status in present[index * keys_per_block : (index + 1) * keys_per_block]
            )
        ]
        self._hole_stats.record(
            num_blocks, hit_blocks, stored_after_hole, self._addressing.tokens_per_block
        )
        self._maybe_log_hole_stats()

    def _maybe_log_hole_stats(self, force: bool = False) -> Optional[str]:
        """Log the prefix-hole summary at most once per `stats_interval`.

        Args:
            force: Log what accumulated even if the interval has not elapsed.

        Returns:
            The line, when one was logged. Nothing is logged when the stats are
            off or no lookup ran since the last line.
        """
        # Read defensively: `shutdown` also runs from `__del__` on a worker
        # whose constructor may not have got this far.
        stats = getattr(self, "_hole_stats", None)
        if stats is None:
            return None
        now = time.monotonic()
        due = force or now - self._hole_stats_last >= self._stats_interval
        if not (due and stats.lookups):
            return None
        line = (
            f"mooncake-store prefix-hole stats: rank {self._rank} "
            f"window_s={now - self._hole_stats_last:.0f} {stats.summary()}"
        )
        stats.reset()
        self._hole_stats_last = now
        logger.info(line)
        return line

    def _maybe_log_load_stats(self, force: bool = False) -> Optional[str]:
        """Log the load summary at most once per `stats_interval`.

        Args:
            force: Log what accumulated even if the interval has not elapsed.

        Returns:
            The line, when one was logged. Nothing is logged when the stats are
            off or nothing loaded since the last line.
        """
        stats = getattr(self, "_load_stats", None)
        if stats is None:
            return None
        now = time.monotonic()
        if not (force or now - self._load_stats_last >= self._stats_interval):
            return None
        if self._async_load:
            with self._load_lock:
                active = len(self._async_active)
                queued = len(self._async_pending) - active
            summary = stats.take_summary(workers=self._async_workers, queued=queued, active=active)
        else:
            summary = stats.take_summary()
        if summary is None:
            return None
        line = (
            f"mooncake-store load stats: rank {self._rank} "
            f"window_s={now - self._load_stats_last:.0f} {summary}"
        )
        self._load_stats_last = now
        logger.info(line)
        return line

    # ---- load path ----

    def start_load_kv(self, stream: torch.cuda.Stream):
        """Pull every scheduled page into its GPU slot before the forward pass.

        The runtime has already counted these tokens as computed. When a page
        is missing and this rank can restart a request alone, the request is
        recorded for `take_failed_load_requests`, its remaining pages are not
        fetched, and its saves leave the bound metadata; the other requests
        load as usual. The bound loads are consumed, so calling this again on
        the same metadata starts nothing.

        Args:
            stream: The executor's stream, on which staged scatters are issued.

        Raises:
            RuntimeError: When a page is missing on a tensor-parallel rank, a
                zero-copy load has no allocation fence to wait on, a staged
                synchronous load arrives on a rank whose loads all go through
                the load threads, or a background load failed.
        """
        # First, and before the empty-metadata return: under attention DP this
        # runs inside the executor's failure vote, and without it the error
        # stops the executor loop.
        self._reraise_background_error()
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None or not metadata.loads:
            return
        loads = list(metadata.loads)
        metadata.loads.clear()

        # Per-phase times of this load, collected only when stats are on.
        phases: Optional[Dict[str, List[float]]] = {} if self._load_stats is not None else None
        with _timed_phase(phases, "resolve"):
            keys, addresses, sizes, total_pages = self._resolve(loads)
        if not keys:
            return
        # `_resolve` emits one key per page in transfer order, so this names the
        # request every key is loaded for.
        owners = [entry.request_id for entry in loads for _ in entry.pages]

        if self._config.stage_through_host and self._load_staging is None:
            raise RuntimeError(
                "mooncake-store: a synchronous load arrived on a rank whose loads all "
                "go through the async_load workers, so it holds no synchronous "
                "staging pool"
            )
        staging = self._load_staging
        handle = _stream_handle(stream) if staging is not None else 0
        if staging is None:
            # The previous pass may still write a recycled target page, e.g. the
            # last token of a request that stopped there. Only the loading
            # requests' pages are waited for, so a load into free pages still
            # overlaps that pass.
            fence = self.get_allocation_fence()
            if fence is None:
                raise RuntimeError(
                    "mooncake-store: a zero-copy load has no allocation fence to "
                    "wait on, so the NIC could write pages earlier GPU work still "
                    "owns. The KV cache manager must offer wait_allocation_fences; "
                    "KVCacheManagerV2 does."
                )
            with _timed_phase(phases, "fence"):
                fence({entry.request_id for entry in loads})

        transfer_start = time.perf_counter() if phases is not None else 0.0
        failed_requests, num_failed_pages, first_failure, loaded_bytes = self._load_pages(
            keys,
            addresses,
            sizes,
            owners,
            staging,
            handle,
            self._batch_size,
            recover=self._recover_failed_loads,
            phases=phases,
        )
        if phases is not None:
            self._load_stats.record(
                time.perf_counter() - transfer_start,
                pages=total_pages,
                batches=-(-len(keys) // self._batch_size),
                loaded_bytes=loaded_bytes,
                phases=phases,
                failed_pages=num_failed_pages,
            )
        if failed_requests:
            # The executor takes these requests out of the batch and restarts
            # them; see `take_failed_load_requests`. Nothing of theirs is
            # computed this pass, so their saves leave the bound metadata now:
            # `wait_for_save` would otherwise read pages the executor has freed,
            # or another request has since filled, and publish them under the
            # failed request's valid keys.
            self.drop_bound_saves(failed_requests)
            self._failed_load_requests.update(failed_requests)
            logger.warning(
                f"mooncake-store rank {self._rank} failed to load {num_failed_pages} page(s) "
                f"for {len(failed_requests)} request(s); they restart without the offered "
                f"prefix. First failure: {first_failure!r}"
            )
        logger.debug(f"mooncake-store rank {self._rank} loaded {total_pages} pages")

    def _load_pages(
        self,
        keys: List[str],
        addresses: List[List[int]],
        sizes: List[List[int]],
        owners: List[int],
        staging: Optional[HostStagingPool],
        handle: int,
        batch_size: int,
        *,
        recover: bool,
        phases: Optional[Dict[str, List[float]]] = None,
    ) -> Tuple[Set[int], int, Optional[str], int]:
        """Fetch pages batch by batch, scattering staged ones onto the device.

        Shared by the synchronous path (executor thread and stream) and the
        load threads (each with its own stream and staging pool). A store call
        is synchronous, and a staged batch's scatter is synced before the next
        batch reuses its slots, so no write into a page is in flight on return.

        Args:
            keys: Store key of every page, in transfer order.
            addresses: Device region addresses of every page.
            sizes: Device region sizes of every page.
            owners: The request each page is loaded for.
            staging: Host slots the pages pass through, or None for zero copy.
            handle: Raw CUDA stream the staged scatters are issued on.
            batch_size: Pages per store call.
            recover: Whether a missing page is handed back rather than raised.
                The pages of a request that already lost one are then skipped,
                since the request restarts and recomputes all of them.
            phases: Per-phase totals the describe, get, unstage and sync
                phases add to, or None to time nothing; see `_timed_phase`.

        Returns:
            The requests that lost a page, how many pages were lost, the first
            missing key, and the bytes that landed.

        Raises:
            RuntimeError: On the first batch with a missing page when `recover`
                is False.
        """
        failed_requests: Set[int] = set()
        first_failure: Optional[str] = None
        num_failed_pages = 0
        loaded_bytes = 0

        for start in range(0, len(keys), batch_size):
            indices = [
                index
                for index in range(start, min(start + batch_size, len(keys)))
                if owners[index] not in failed_requests
            ]
            if not indices:
                continue
            batch_keys = [keys[index] for index in indices]
            batch_addresses = [addresses[index] for index in indices]
            batch_sizes = [sizes[index] for index in indices]
            with _timed_phase(phases, "describe"):
                if staging is None:
                    target_addresses, target_sizes = batch_addresses, batch_sizes
                else:
                    target_addresses, target_sizes = describe_batch_for_get(staging, batch_sizes)
            with _timed_phase(phases, "get"):
                results = self._store.batch_get_into_multi_buffers(
                    batch_keys, target_addresses, target_sizes
                )
            if len(results) != len(batch_keys):
                # Which pages landed is unknowable, so none of them counts.
                results = [-1] * len(batch_keys)
            landed = [
                position
                for position, result in enumerate(results)
                if isinstance(result, int) and result >= 0
            ]
            loaded_bytes += sum(sum(batch_sizes[position]) for position in landed)
            if len(landed) != len(batch_keys):
                landed_set = set(landed)
                for position, index in enumerate(indices):
                    if position in landed_set:
                        continue
                    failed_requests.add(owners[index])
                    num_failed_pages += 1
                    if first_failure is None:
                        first_failure = batch_keys[position]
                if not recover:
                    # The runtime already counted these tokens as computed, so a
                    # partial load would leave the forward pass reading
                    # uninitialized KV and silently producing wrong tokens.
                    raise RuntimeError(
                        f"mooncake-store failed to load {len(batch_keys) - len(landed)} of "
                        f"{len(batch_keys)} pages; the affected KV slots were already "
                        f"reported as computed. First failure: {first_failure!r}"
                    )
            if staging is not None and landed:
                # Only filled slots are scattered, so no slot holding a failed
                # read is copied over a device page.
                with _timed_phase(phases, "unstage"):
                    unstage_batch_after_get(
                        staging, batch_addresses, batch_sizes, handle, only=landed
                    )
                # The next batch reuses the slots and the forward pass reads
                # these pages, so the scatter has to complete before either.
                with _timed_phase(phases, "sync"):
                    _sync_stream(handle)
        return failed_requests, num_failed_pages, first_failure, loaded_bytes

    # ---- asynchronous load path ----

    def bind_connector_meta(self, metadata: object) -> None:
        """Bind this iteration's work and queue the parked requests' loads.

        Every rank binds the same metadata, so every rank queues its own pages
        here, including the tensor-parallel ranks whose scheduler adapter never
        saw the allocation. The asynchronous loads are consumed, so binding the
        same metadata again queues nothing twice. The wait for the destination
        pages happens on the load thread; see `_drain_loads`.

        Never raises: it runs outside the executor's attention-DP failure vote,
        so what cannot be queued is kept for the next `start_load_kv` or
        `get_finished` to raise.

        Args:
            metadata: This iteration's `MooncakeStoreMetadata`.
        """
        super().bind_connector_meta(metadata)
        async_loads = getattr(metadata, "async_loads", None)
        if not async_loads:
            return
        queued = list(async_loads)
        metadata.async_loads = []
        if not self._async_load:
            self._stash_load_error(
                RuntimeError(
                    "mooncake-store received asynchronous loads, but async_load is off on this rank"
                )
            )
            return
        if not self._load_threads:
            self._stash_load_error(
                RuntimeError(
                    "mooncake-store received asynchronous loads before its KV cache "
                    "layout was registered, so no load thread runs them"
                )
            )
            return
        # Bound by `handle_metadata` just before this metadata, for the batch
        # that allocated the destination pages.
        fence = self.get_allocation_fence()
        for transfers in queued:
            self._start_async_load(transfers, fence)

    def _start_async_load(
        self, transfers: RequestTransfers, fence: Optional[AllocationFence]
    ) -> None:
        """Queue one parked request's pages for the load threads.

        The fence bound in the iteration that parked the request travels with
        it, since a later iteration binds another. A parked request keeps its
        allocation, and the fence its pages, until its load is reported, so the
        load thread can still wait on it. A transfer with no pages, for an
        offer the runtime honoured with no tokens, completes at once, so the
        parked request is released at the next `get_finished`.

        Args:
            transfers: The parked request's pages, every layer group's.
            fence: The KV cache manager's allocation fence for that iteration.
        """
        request_id = transfers.request_id
        if transfers.pages and fence is None:
            # Neither the NIC writing zero copy nor a staged scatter on a load
            # thread's stream is ordered behind the allocation stream.
            self._stash_load_error(
                RuntimeError(
                    "mooncake-store: an asynchronous load has no allocation fence to "
                    "wait on, so the NIC could write pages earlier GPU work still "
                    "owns. The KV cache manager must offer wait_allocation_fences; "
                    "KVCacheManagerV2 does."
                )
            )
            with self._load_lock:
                self._async_done[request_id] = True
            return
        with self._load_lock:
            if not transfers.pages:
                self._async_done[request_id] = False
                return
            self._async_pending.add(request_id)
        self._load_queue.put((request_id, transfers, time.monotonic(), fence))

    def _drain_loads(self, index: int, started: threading.Event) -> None:
        """Run parked requests' loads, one at a time, until a None arrives.

        Args:
            index: This thread's position, which selects its staging pool.
            started: Set once the thread has finished its device and stream
                setup, whether or not that succeeded.
        """
        try:
            # A new thread starts on device 0; see `_drain_saves`.
            if self._device_index is not None:
                torch.cuda.set_device(self._device_index)
            staging = self._async_stagings[index] if self._async_stagings else None
            # Owned by this thread for its lifetime, so a staged scatter never
            # queues behind the executor's work or another load's.
            stream = (
                torch.cuda.Stream() if staging is not None and torch.cuda.is_available() else None
            )
            handle = _stream_handle(stream) if staging is not None else 0
        except Exception as exc:
            # Fatal: a thread that died here would leave the requests parked on
            # it waiting for good.
            logger.error(
                f"mooncake-store load thread {index} failed to start on rank "
                f"{self._rank}: {type(exc).__name__}: {exc}"
            )
            with self._load_lock:
                self._load_startup_errors.append(exc)
            return
        finally:
            started.set()
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            request_id, transfers, enqueued_at, fence = item
            if self._load_stop.is_set():
                # Shutting down: a load still queued never reaches the store.
                with self._load_lock:
                    self._async_pending.discard(request_id)
                continue
            self._run_async_load(index, request_id, transfers, enqueued_at, fence, staging, handle)

    def _run_async_load(
        self,
        index: int,
        request_id: int,
        transfers: RequestTransfers,
        enqueued_at: float,
        fence: AllocationFence,
        staging: Optional[HostStagingPool],
        handle: int,
    ) -> None:
        """Wait for one parked request's pages to be free, then fill them.

        A missing page marks the request failed for `get_finished` to hand
        back or raise; any exception is also kept as this worker's background
        error. Either way the request is marked done, so it does not stay
        parked forever.
        """
        phases: Optional[Dict[str, List[float]]] = {} if self._load_stats is not None else None
        with self._load_lock:
            self._async_active.add(request_id)
        started_at = time.monotonic()
        failed = False
        total_pages = batches = num_failed_pages = loaded_bytes = 0
        try:
            # The pages were reserved in the parking iteration, possibly on the
            # KV cache manager's allocation stream, and a recycled page can
            # wait on a forward pass still in flight. That wait happens here,
            # off the executor thread.
            with _timed_phase(phases, "fence"):
                fence({request_id})
            started_at = time.monotonic()
            with _timed_phase(phases, "resolve"):
                keys, addresses, sizes, total_pages = self._resolve([transfers])
            batches = -(-len(keys) // self._async_batch_size)
            # Recovered here whatever the rank: `get_finished` decides whether
            # a failed request is fatal.
            failed_requests, num_failed_pages, first_failure, loaded_bytes = self._load_pages(
                keys,
                addresses,
                sizes,
                [request_id] * len(keys),
                staging,
                handle,
                self._async_batch_size,
                recover=True,
                phases=phases,
            )
            failed = bool(failed_requests)
            if failed:
                logger.warning(
                    f"mooncake-store rank {self._rank} failed to load {num_failed_pages} "
                    f"page(s) for parked request {request_id}. First failure: "
                    f"{first_failure!r}"
                )
            else:
                logger.debug(
                    f"mooncake-store rank {self._rank} load thread {index} loaded "
                    f"{total_pages} pages for parked request {request_id}"
                )
        except Exception as exc:
            # The thread boundary: kept for the executor thread to raise.
            failed = True
            logger.error(
                f"mooncake-store asynchronous load of request {request_id} failed on "
                f"rank {self._rank}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            self._stash_load_error(exc)
        finally:
            finished_at = time.monotonic()
            # Recorded before the request leaves the pending set, so a summary
            # taken once nothing is pending includes it.
            if self._load_stats is not None:
                self._load_stats.record(
                    finished_at - started_at,
                    pages=total_pages,
                    batches=batches,
                    loaded_bytes=loaded_bytes,
                    phases=phases,
                    failed_pages=num_failed_pages,
                    wait_s=started_at - enqueued_at,
                )
            with self._load_lock:
                self._async_active.discard(request_id)
                self._async_pending.discard(request_id)
                self._async_done[request_id] = failed

    def take_failed_async_load_requests(self) -> Set[int]:
        """Parked requests whose last `get_finished` reported a load that came up short.

        The executor drains this in the same `_kv_connector_terminate_requests`
        call that ran `get_finished`, before the next scheduling pass, drops
        their allocation and lets the scheduler admit them again. Every write
        into their pages has completed by then: an id is reported only once its
        load thread is done with it. Ids the runtime no longer holds parked,
        because the request finished or was cancelled meanwhile, are ignored
        there. Cleared on return.

        Returns:
            The request IDs to restart.
        """
        with self._load_lock:
            failed, self._failed_async_load_requests = self._failed_async_load_requests, set()
        return failed

    def _stash_load_error(self, error: BaseException) -> None:
        """Keep the first background load error for the executor thread to raise."""
        with self._load_lock:
            if self._load_error is None:
                self._load_error = error

    def _reraise_background_error(self) -> None:
        """Raise the kept background load error, if any.

        Sticky rather than raised once, so that no later caller, such as a
        second recovery poll or the attention-DP shutdown drain, sees a healthy
        worker. It is never cleared; only shutting the worker down ends the
        calls that raise it. Save errors keep their log-and-drop policy and are
        not raised.

        Raises:
            RuntimeError: Chained from the kept error.
        """
        with self._load_lock:
            error = self._load_error
        if error is not None:
            raise RuntimeError(
                f"mooncake-store rank {self._rank}: a background load failed: "
                f"{type(error).__name__}: {error}"
            ) from error

    def take_failed_load_requests(self) -> Set[int]:
        """Requests whose pages could not all be loaded in the last `start_load_kv`.

        The executor drains this right after `start_load_kv` and before the
        forward pass, takes those requests out of the batch, frees their
        allocation and lets the scheduler admit them again, so the store is
        asked afresh and the lost pages are recomputed. Every write into their
        pages has completed by then; see `_load_pages`. Cleared on return.

        Returns:
            The request IDs to restart.
        """
        failed, self._failed_load_requests = self._failed_load_requests, set()
        return failed

    def drop_bound_saves(self, request_ids: Iterable[int]) -> int:
        """Remove these requests' saves from the metadata bound for this pass.

        For a request taken out of the batch after `build_connector_meta` ran:
        its scheduled tail is never computed, so none of it may be published.
        Edits the bound metadata in place, which is the object `wait_for_save`
        reads, and a request that is not bound is ignored, so repeating a call
        changes nothing.

        Args:
            request_ids: The restarted requests.

        Returns:
            The number of save entries removed.
        """
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None or not metadata.saves:
            return 0
        drop = set(request_ids)
        saves = metadata.saves
        kept = [transfers for transfers in saves if transfers.request_id not in drop]
        removed = len(saves) - len(kept)
        if removed:
            dropped_ids = sorted({transfers.request_id for transfers in saves} & drop)
            saves[:] = kept
            logger.debug(
                f"mooncake-store rank {self._rank} dropped the saves of requests "
                f"{dropped_ids} ({removed} entr{'y' if removed == 1 else 'ies'}) from this pass"
            )
        return removed

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """No-op: loads complete in `start_load_kv`, or before a parked request rejoins.

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
        if metadata is None or not metadata.saves or not self._config.role.saves:
            return

        # The pages are written by kernels still queued on this stream, so the
        # event is the handoff: the thread reads GPU memory only after the
        # pass retires, without the executor loop waiting for it.
        event = torch.cuda.Event()
        event.record(stream)

        with self._save_lock:
            for transfers in metadata.saves:
                self._outstanding_saves[transfers.request_id] += 1
        self._save_queue.put((event, list(metadata.saves)))
        # Replaced rather than accumulated: under tensor parallelism only rank
        # 0's adapter takes these, so the other ranks must not grow the set
        # every pass.
        self._submitted_saves = {transfers.request_id for transfers in metadata.saves}

    def take_submitted_saves(self) -> Set[int]:
        """Requests whose saves the last `wait_for_save` handed to the save thread.

        A pass that was bound but never ran never reaches `wait_for_save`, so
        its saves are not reported and the scheduler adapter emits them again
        instead of counting them as stored. Cleared on return.

        Returns:
            The request ids of the saves the last `wait_for_save` queued.
        """
        submitted, self._submitted_saves = self._submitted_saves, set()
        return submitted

    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Report which requests' saves and asynchronous loads have landed.

        Args:
            finished_gen_req_ids: Requests that will produce no further KV.
            started_loading_req_ids: Requests the runtime parked for an
                asynchronous load. Without `async_load` the scheduler never
                offers one, and an id here is echoed back so the runtime does
                not wait on something that already happened. With it, an id is
                reported only after it was announced here and its load thread
                is done with it: its pages landed, or one was found missing.

        Returns:
            Requests that have finished saving, and requests that have finished
            loading. A finished load that lost a page is also listed for
            `take_failed_async_load_requests`.

        Raises:
            RuntimeError: When a background load failed, or a parked request
                lost a page on a rank that cannot restart it alone (tensor
                parallelism).
        """
        # Under attention DP this runs inside the executor's failure vote.
        self._reraise_background_error()
        # Runs every iteration on every rank that transfers, so it is where the
        # load summary comes due.
        self._maybe_log_load_stats()
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
        if not self._async_load:
            return finished_saving, list(started_loading_req_ids)

        finished_loading: List[int] = []
        with self._load_lock:
            self._async_announced.update(started_loading_req_ids)
            for request_id in sorted(self._async_announced):
                if request_id in self._async_done:
                    failed = self._async_done.pop(request_id)
                elif request_id in self._async_pending:
                    continue
                else:
                    # Parked by the runtime but never handed to this worker, so
                    # its pages hold nothing usable.
                    logger.warning(
                        f"mooncake-store rank {self._rank}: request {request_id} was "
                        "parked for an asynchronous load that was never started"
                    )
                    failed = True
                if failed and not self._recover_failed_loads:
                    raise RuntimeError(
                        f"mooncake-store rank {self._rank}: the asynchronous load of "
                        f"request {request_id} lost a page, and this deployment cannot "
                        "restart the request alone (tensor parallelism, or a connector "
                        "that does not report failed loads)"
                    )
                self._async_announced.discard(request_id)
                if failed:
                    self._failed_async_load_requests.add(request_id)
                finished_loading.append(request_id)
        return finished_saving, finished_loading

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
        while True:
            item = self._save_queue.get()
            if item is None:
                return
            event, transfers = item
            try:
                event.synchronize()
                self._put(transfers)
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

    def _put(self, transfers: Sequence[RequestTransfers]) -> None:
        keys, addresses, sizes, _ = self._resolve(transfers)
        if not keys:
            return

        staging = self._save_staging
        handle = _stream_handle(self._save_stream) if staging is not None else 0
        groups = self._block_groups(transfers) if self._group_blocks else [None] * len(keys)

        for batch in zip(
            _batched(keys, self._batch_size),
            _batched(addresses, self._batch_size),
            _batched(sizes, self._batch_size),
            _batched(groups, self._batch_size),
        ):
            batch_keys, batch_addresses, batch_sizes, batch_groups = batch
            # Skip pages another rank or another instance already wrote. The
            # scheduler holds no store handle, and the answer changes between
            # the time it builds metadata and now.
            present = self._store.batch_is_exist(list(batch_keys))
            pending = [
                index
                for index, status in enumerate(present)
                if status != 1  # absent, or a failed probe we retry as a write
            ]
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
            put_args = ([batch_keys[i] for i in pending], source_addresses, source_sizes)
            if self._group_blocks:
                # One group per key that survived the existence filter.
                put_args += (
                    _replicate_config(self._soft_pin_ttl_ms, [batch_groups[i] for i in pending]),
                )
            elif self._put_config is not None:
                put_args += (self._put_config,)
            # With neither option set this is the plain three-argument put.
            results = self._store.batch_put_from_multi_buffers(*put_args)
            failures = sum(1 for result in results if not isinstance(result, int) or result < 0)
            if failures:
                # A dropped write only costs a future cache miss, which is not
                # worth failing a request that already answered.
                logger.warning(
                    f"mooncake-store rank {self._rank} failed to save {failures} of "
                    f"{len(pending)} pages"
                )
                if self._soft_pin_ttl_ms is not None and failures == len(pending):
                    logger.warning_once(
                        f"mooncake-store rank {self._rank}: the pool refused every page "
                        f"of a put with soft_pin_ttl_ms={self._soft_pin_ttl_ms}. A "
                        "soft_pin_ttl_ms above the master's --max_kv_soft_pin_ttl (24 h "
                        "by default) makes the master refuse every put.",
                        key="mooncake-store-soft-pin-refused",
                    )

    def _block_groups(self, transfers: Sequence[RequestTransfers]) -> List[str]:
        """The eviction group of every page `_resolve` lists, in the same order.

        One group per block covers every layer group and attention shard of it.
        A block is reusable only while all of those pages survive, so the master
        should evict them together rather than leave orphans that hold space and
        still miss.
        """
        return [
            self._namespaces[page.layer_group_id].group(page.block_hash)
            for entry in transfers
            for page in entry.pages
        ]

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
        """Stop the transfer threads, then release what they used. Idempotent."""
        # What accumulated since the last line would otherwise be lost.
        self._maybe_log_hole_stats(force=True)
        self._maybe_log_load_stats(force=True)
        # A load still queued is skipped rather than started.
        self._load_stop.set()
        save_thread = self._save_thread
        threads = ([save_thread] if save_thread is not None else []) + list(self._load_threads)
        # Every thread is signalled before any is joined, so they drain
        # together against one deadline. A sentinel resent by a retry is
        # harmless.
        if save_thread is not None:
            self._save_queue.put(None)
        for _ in self._load_threads:
            self._load_queue.put(None)
        deadline = time.monotonic() + SAVE_DRAIN_TIMEOUT
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            # These threads use the KV pools through the store handle and the
            # staging slots, so releasing either takes the memory out from
            # under a transfer in flight. Leaking both is the safer end: the
            # process is going down anyway, the threads are daemons and do not
            # hold it open, and a later call retries the join.
            logger.error(
                f"mooncake-store rank {self._rank}: {', '.join(alive)} did not stop "
                f"within {SAVE_DRAIN_TIMEOUT:g}s, so the store handle and its staging "
                "buffers are left in place rather than freed under a transfer still "
                "using them."
            )
            return
        self._save_thread = None
        self._load_threads = []
        if self._store is not None:
            # Before the store closes and the cache manager frees the pools,
            # so the fabric holds no registration over memory handed out again.
            self._unregister_pools()
            self._close_staging()
        store, self._store = self._store, None
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                logger.warning(
                    f"mooncake-store close failed: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
        # Dropped only after the store is closed. A pool whose registration did
        # not come back stays referenced in `_unreleased_staging`.
        self._load_staging = None
        self._save_staging = None
        self._async_stagings = []
        self._save_stream = None
        global _LOCAL_WORKER
        if _LOCAL_WORKER is self:
            _LOCAL_WORKER = None
            _LOCAL_WORKER_READY.clear()

    def _close_staging(self) -> None:
        """Hand back every staging pool's registration before the store closes.

        A pool whose unregistration fails is kept in `_unreleased_staging`, so
        its buffer outlives a registration the fabric may still use.
        """
        for pool in (self._load_staging, self._save_staging, *self._async_stagings):
            if pool is None:
                continue
            try:
                pool.close()
            except RuntimeError as exc:
                logger.error(f"mooncake-store rank {self._rank}: {exc}")
                self._unreleased_staging.append(pool)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:  # noqa: S110 - interpreter teardown, nothing left to report to
            pass
