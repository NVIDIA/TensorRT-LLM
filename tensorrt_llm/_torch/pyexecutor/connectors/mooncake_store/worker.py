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
knows how a page is addressed and how a key is spelled. Each scheduler adapter
asks its process-local worker to run prefix lookups rather than rebuilding that
knowledge: every owner has an adapter under ADP, while TP has one on rank 0.

Loads are synchronous by default: the runtime has already told the scheduler
those tokens are computed, so a failed load is a wrong answer rather than a
slow one. A page offered at lookup can still be gone by the time it is loaded,
so a rank that owns whole pages hands the affected requests back to the
executor to restart rather than failing the server.

With `async_load` the scheduler answers asynchronously instead. The runtime
parks the request outside the batch, a load thread pulls its pages on a stream
of its own, and the request rejoins the batch once `get_finished` reports them
landed. That takes the transfer out of the executor iteration, which under
attention DP every owner waits on.

Saves are asynchronous and gated on a CUDA event, since the pages are complete
only once the forward pass that wrote them has retired and blocking the
executor loop on an RDMA write is the cost the store exists to avoid. The
scheduler reports such a request as saving asynchronously, which keeps its
pages pinned until `get_finished` says the writes landed.

A capacity-only worker opens its handle and stops there, with no layout, no
buffer registration and no save thread, so a node can lend host memory to the
pool without an HCA that can pin GPU pages.
"""

import os
import threading
import time
import traceback
from collections import defaultdict
from queue import Queue
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger

from ..kv_cache_connector import KvCacheConnectorWorker
from ..kv_cache_layout import KvCacheLayout
from .addressing import PageAddressing
from .config import CONFIG_PATH_ENV, MooncakeStoreConnectorConfig, pool_config
from .gpudirect import REGISTRATION_DEBUG_ENV, format_diagnosis
from .keys import KeyNamespace
from .ledger import record_segment
from .metadata import MooncakeStoreMetadata, RequestTransfers
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

__all__ = ["MooncakeStoreConnectorWorker", "resolve_local_worker"]

_GIB = 1 << 30

#: Set by the worker's constructor so the scheduler adapter, built in the same
#: process on every ADP owner (rank 0 for TP), can reach the store handle
#: without a second connection. `py_executor_creator` constructs the two
#: concurrently, so the adapter waits on `_LOCAL_WORKER_READY`.
_LOCAL_WORKER: Optional["MooncakeStoreConnectorWorker"] = None
_LOCAL_WORKER_READY = threading.Event()

#: Seconds `shutdown` gives a background thread to drain its queue and return.
#: Sized for a backlog of pages over a congested fabric rather than a healthy
#: one, since what follows the wait is only safe once the thread has stopped.
DRAIN_TIMEOUT = 30.0

#: Seconds between asynchronous loader summaries, per rank.
ASYNC_STATS_INTERVAL = 30.0

#: Staging slots a load worker keeps however many workers share the budget.
#: Below this the per-batch overhead starts to dominate the transfer.
MIN_ASYNC_STAGING_SLOTS = 64


def resolve_local_worker(timeout: float = 60.0) -> "MooncakeStoreConnectorWorker":
    """The worker living in this process, once it has been constructed.

    Args:
        timeout: Seconds to wait. Exceeding it means construction failed.
    """
    if not _LOCAL_WORKER_READY.wait(timeout):
        raise RuntimeError(
            "The mooncake-store leader could not find a worker in its process. "
            "The executor builds a worker alongside each scheduler adapter, "
            "so this means worker construction failed."
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
            "The mooncake-store connector needs the Mooncake Python bindings "
            "(`pip install mooncake-transfer-engine`). The C++ transfer engine "
            "built into the container is a different component and does not "
            "provide MooncakeDistributedStore."
        ) from exc

    store = MooncakeDistributedStore()
    hostname = config.local_hostname or _default_hostname()
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


def _default_hostname() -> str:
    import socket

    return socket.gethostbyname(socket.gethostname())


def _batched(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class _AsyncLoadStats:
    """Counters for the asynchronous loader, summarised once per interval."""

    __slots__ = (
        "completed",
        "wait_sum",
        "wait_max",
        "transfer_sum",
        "transfer_max",
        "failed_pages",
        "loaded_bytes",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.completed = 0
        self.wait_sum = 0.0
        self.wait_max = 0.0
        self.transfer_sum = 0.0
        self.transfer_max = 0.0
        self.failed_pages = 0
        self.loaded_bytes = 0

    def record(self, wait: float, transfer: float, failed_pages: int, loaded_bytes: int) -> None:
        self.completed += 1
        self.wait_sum += wait
        self.wait_max = max(self.wait_max, wait)
        self.transfer_sum += transfer
        self.transfer_max = max(self.transfer_max, transfer)
        self.failed_pages += failed_pages
        self.loaded_bytes += loaded_bytes

    def summary(self, queued: int, active: int) -> str:
        """One line covering the interval. Queue depth and active loads are live."""
        divisor = max(1, self.completed)
        return (
            f"completed={self.completed} "
            f"wait_ms mean={1000 * self.wait_sum / divisor:.1f} "
            f"max={1000 * self.wait_max:.1f} "
            f"transfer_ms mean={1000 * self.transfer_sum / divisor:.1f} "
            f"max={1000 * self.transfer_max:.1f} "
            f"queue={queued} active={active} failed_pages={self.failed_pages} "
            f"loaded_gib={self.loaded_bytes / _GIB:.2f}"
        )


def _stream_handle(stream: Optional[torch.cuda.Stream]) -> int:
    """The raw CUDA stream handle behind a torch stream.

    `None` maps to 0, the default stream, which is what the runtime passes when
    it has no stream of its own to offer.
    """
    if stream is None:
        return 0
    return int(stream.cuda_stream)


class MooncakeStoreConnectorWorker(KvCacheConnectorWorker):
    """Moves KV pages between this rank's GPU cache and the Mooncake pool."""

    supports_attention_dp = True

    def __init__(self, llm_args: TorchLlmArgs) -> None:
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
        # Each ADP owner holds a complete attention cache. Reusable content is
        # keyed by attention sharding, while rank and client lifetime stay
        # local.
        self._attention_rank = 0 if llm_args.enable_attention_dp else self._rank
        self._attention_world_size = 1 if llm_args.enable_attention_dp else self._world_size
        self._model_key = self._config.resolve_model_key(llm_args.model)

        self._addressing: Optional[PageAddressing] = None
        #: Namespaces for this attention shard, shared by compatible ADP owners.
        self._namespaces: Dict[int, KeyNamespace] = {}
        # A TP hit requires every attention shard. ADP has one complete shard,
        # so neither content identity nor lookup depends on unrelated owners.
        self._peer_namespaces: Dict[int, Tuple[KeyNamespace, ...]] = {}

        # An attention-DP owner, and a single-rank server, holds whole pages
        # and its own batch, so it can decide a restart alone. Under tensor
        # parallelism the ranks hold shards of the same page and would have to
        # agree, which they cannot do here, so a failed load stays fatal.
        self._recover_failed_loads = bool(llm_args.enable_attention_dp) or self._world_size == 1
        self._async_load = bool(self._config.async_load) and self._config.role.loads
        if self._async_load and not self._recover_failed_loads:
            raise NotImplementedError(
                "mooncake_store.async_load needs attention DP or a single rank: a "
                "parked request is loaded and reported by the rank that owns its "
                "pages, and under tensor parallelism the ranks hold shards of the "
                "same page and would have to agree on completion and on restarts."
            )

        # Checked before the segment is mounted, since the kernel answers an
        # unaffordable total by killing the process rather than by failing the
        # allocation.
        validate_node_budget(self._config)

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
        self._save_error: Optional[BaseException] = None
        self._failed_load_requests: Set[int] = set()

        self._async_workers = (
            max(1, int(self._config.async_load_workers)) if self._async_load else 0
        )
        # One queue for every worker, so a request is handled by exactly one of
        # them and starts in arrival order while several transfer at once.
        self._load_queue: "Queue[Optional[Tuple[int, RequestTransfers, float]]]" = Queue()
        self._load_threads: List[threading.Thread] = []
        self._load_lock = threading.Lock()
        self._load_streams: List[Optional[torch.cuda.Stream]] = []
        self._async_stagings: List[HostStagingPool] = []
        #: Pages per store call on the asynchronous path, narrowed by its staging.
        self._async_batch_size = self._batch_size
        #: Queued or transferring.
        self._async_pending: Set[int] = set()
        #: Taken by a worker and transferring right now.
        self._async_active: Set[int] = set()
        #: Landed but not yet reported, mapped to whether a page was missing.
        self._async_done: Dict[int, bool] = {}
        #: Requests the runtime has told us it parked.
        self._async_announced: Set[int] = set()
        self._failed_async_load_requests: Set[int] = set()
        self._async_stats = _AsyncLoadStats()
        self._async_stats_last = time.monotonic()
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
        if self._config.stage_through_host:
            self._open_staging(addressing)
        else:
            ranges = addressing.registration_ranges()
            boundary = addressing.mapping_bytes
            logger.info(
                f"mooncake-store rank {self._rank} registering {len(ranges)} range(s) "
                f"covering {sum(end - start for start, end in ranges) / _GIB:.1f} GiB, "
                f"pool mapping boundary {boundary if boundary else 'unknown'}"
            )
            if boundary and not all(addressing.mapping_origins):
                logger.warning(
                    f"mooncake-store rank {self._rank} could not read the base of "
                    "every KV pool reservation from the driver, so the mapping "
                    "boundaries of those pools are taken to be the multiples of "
                    f"{boundary}. A pool whose reservation is not aligned to that "
                    "will fail registration below."
                )
            # One line per range, so only on request.
            if os.getenv(REGISTRATION_DEBUG_ENV):
                logger.info(format_diagnosis(ranges, self._rank, boundary))
            for start, end in ranges:
                status = self._store.register_buffer(start, end - start)
                if status != 0:
                    # Collected only on failure, so the common path pays nothing.
                    raise RuntimeError(
                        f"MooncakeDistributedStore.register_buffer failed with status "
                        f"{status} for [{start:#x}, {end:#x}). Without registration "
                        "the store cannot read or write these pages. Registering "
                        "device memory needs GPUDirect RDMA, either nvidia_peermem "
                        "(Mooncake's default, selected unless WITH_NVIDIA_PEERMEM=0) "
                        "or dma-buf. Set stage_through_host to pass pages through "
                        "pinned host memory instead.\n"
                        f"{format_diagnosis(ranges, self._rank, boundary)}"
                    )

        self._addressing = addressing
        for layer_group_id in addressing.layer_group_ids:
            bytes_per_page = addressing.bytes_per_page(layer_group_id)
            self._namespaces[layer_group_id] = self._namespace(
                self._attention_rank, layer_group_id, bytes_per_page
            )
            self._peer_namespaces[layer_group_id] = tuple(
                self._namespace(rank, layer_group_id, bytes_per_page)
                for rank in range(self._attention_world_size)
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
                startup_error, self._save_error = self._save_error, None
            if startup_error is not None:
                raise RuntimeError(
                    f"mooncake-store: the save thread for rank {self._rank} "
                    "could not start, so this worker would accept pages it "
                    "could never write to the pool."
                ) from startup_error

        for index in range(self._async_workers):
            thread = threading.Thread(
                target=self._drain_loads,
                args=(index,),
                name=f"mooncake-store-load-{self._rank}-{index}",
                daemon=True,
            )
            thread.start()
            self._load_threads.append(thread)

        logger.warning(
            f"mooncake-store worker rank {self._rank} registered layout: "
            f"{addressing.describe()}"
            + (f", {self._async_workers} asynchronous load worker(s)" if self._async_load else "")
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
        if self._config.role.saves:
            self._save_staging = HostStagingPool(
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                store=self._store,
                label="save",
            )
        if self._async_load:
            # A pool per load thread, so no two transfers wait on each other
            # for a slot and none waits on the synchronous path. One worker
            # keeps the synchronous geometry exactly; several divide the
            # ceiling between them, down to MIN_ASYNC_STAGING_SLOTS each.
            if self._async_workers == 1:
                worker_slots = num_slots
            else:
                _, worker_slots = plan_slot_geometry(
                    max_bytes_per_page,
                    self._config.transfer_batch_size,
                    MAX_STAGING_BUFFER_BYTES // self._async_workers,
                )
                worker_slots = max(
                    worker_slots,
                    min(MIN_ASYNC_STAGING_SLOTS, self._config.transfer_batch_size),
                )
            self._async_stagings = [
                HostStagingPool(
                    slot_bytes=slot_bytes,
                    num_slots=worker_slots,
                    store=self._store,
                    label=f"async-load-{index}",
                )
                for index in range(self._async_workers)
            ]
            self._async_batch_size = min(self._config.transfer_batch_size, worker_slots)
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
            world_size=self._attention_world_size,
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

    def count_prefix_hit(self, block_hashes: Sequence[bytes]) -> int:
        """How many leading blocks of `block_hashes` are fully present.

        A block counts only when every layer group and attention shard has its
        page. ADP owners share the one unsharded representation; TP requires
        all shards, a prefix being replayed as a whole. The scan stops at the
        first incomplete block, a later hit being unusable on its own.

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
        return hit_blocks

    # ---- load path ----

    def start_load_kv(self, stream: torch.cuda.Stream):
        """Pull every scheduled page into its GPU slot before the forward pass."""
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None or not metadata.loads:
            return
        self._reraise_background_error()

        keys, addresses, sizes, total_pages = self._resolve(metadata.loads)
        if not keys:
            return
        # `_resolve` emits one key per page in transfer order, so this is the
        # owning request of every key.
        owners = [entry.request_id for entry in metadata.loads for _ in entry.pages]

        staging = self._load_staging
        handle = _stream_handle(stream) if staging is not None else 0
        failed_requests, num_failed_pages, first_failure, _ = self._load_pages(
            keys, addresses, sizes, owners, staging, handle, self._batch_size
        )

        if failed_requests:
            # These requests leave the batch without computing anything, so
            # their saves go too; see `drop_bound_saves`.
            self.drop_bound_saves(failed_requests)
            self._failed_load_requests.update(failed_requests)
            logger.warning(
                f"mooncake-store rank {self._rank} failed to load {num_failed_pages} page(s) "
                f"for {len(failed_requests)} request(s); they are restarted without the "
                f"offered prefix. First failure: {first_failure!r}"
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
    ) -> Tuple[Set[int], int, Optional[str], int]:
        """Fetch pages batch by batch, scattering staged ones onto the device.

        Shared by the synchronous path, on the executor thread and stream, and
        by the loader threads, each on its own stream and staging pool.

        Returns:
            The requests that lost a page, how many pages were lost, the first
            missing key, and the bytes that landed. When failures are not
            recoverable on this rank the first one raises instead.
        """
        failed_requests: Set[int] = set()
        first_failure: Optional[str] = None
        num_failed_pages = 0
        loaded_bytes = 0

        for start in range(0, len(keys), batch_size):
            # Pages of a request that already lost one are not loaded: the
            # request is restarted and recomputes all of them.
            indices = [
                i
                for i in range(start, min(start + batch_size, len(keys)))
                if owners[i] not in failed_requests
            ]
            if not indices:
                continue
            batch_keys = [keys[i] for i in indices]
            batch_addresses = [addresses[i] for i in indices]
            batch_sizes = [sizes[i] for i in indices]
            if staging is None:
                target_addresses, target_sizes = list(batch_addresses), list(batch_sizes)
            else:
                target_addresses, target_sizes = describe_batch_for_get(staging, batch_sizes)
            results = self._store.batch_get_into_multi_buffers(
                batch_keys, target_addresses, target_sizes
            )
            if len(results) != len(batch_keys):
                results = [-1] * len(batch_keys)
            landed = [
                j for j, result in enumerate(results) if isinstance(result, int) and result >= 0
            ]
            loaded_bytes += sum(sum(batch_sizes[j]) for j in landed)
            if len(landed) != len(batch_keys):
                missing = set(range(len(batch_keys))) - set(landed)
                for j in sorted(missing):
                    failed_requests.add(owners[indices[j]])
                    num_failed_pages += 1
                    if first_failure is None:
                        first_failure = batch_keys[j]
                if not self._recover_failed_loads:
                    # The runtime already counted these tokens as computed, so a
                    # partial load would leave the forward pass reading
                    # uninitialized KV and silently producing wrong tokens.
                    raise RuntimeError(
                        f"mooncake-store failed to load {len(missing)} of "
                        f"{len(batch_keys)} pages; the affected KV slots were already "
                        f"reported as computed. First failure: {first_failure!r}"
                    )
            if staging is not None and landed:
                # Only the slots that were filled are scattered, so a failed
                # read is never copied over a device page.
                unstage_batch_after_get(staging, batch_addresses, batch_sizes, handle, only=landed)
                # The next batch reuses the slots and the forward pass reads
                # these pages, so the scatter has to complete before either.
                _sync_stream(handle)
        return failed_requests, num_failed_pages, first_failure, loaded_bytes

    def take_failed_load_requests(self) -> Set[int]:
        """Requests whose pages could not all be loaded in the last `start_load_kv`.

        The executor drops their allocation and lets the scheduler admit them
        again; the store is asked afresh and the lost pages are simply
        recomputed. Cleared on return.
        """
        failed, self._failed_load_requests = self._failed_load_requests, set()
        return failed

    # ---- asynchronous load path ----

    def start_async_load(self, request_id: int, transfers: RequestTransfers) -> None:
        """Queue a parked request's offered pages for the load threads.

        Called by the scheduler adapter as soon as the runtime has reported
        the pages that will hold the prefix. An empty transfer completes at
        once, which releases a request the runtime parked for an offer it
        could not then fit.
        """
        if not self._async_load:
            raise RuntimeError("mooncake-store asynchronous loads are not enabled")
        self._reraise_background_error()
        with self._load_lock:
            if not transfers.pages:
                self._async_done[request_id] = False
                return
            self._async_pending.add(request_id)
        self._load_queue.put((request_id, transfers, time.monotonic()))

    def take_failed_async_load_requests(self) -> Set[int]:
        """Parked requests already reported by `get_finished` that lost a page.

        Handled like a synchronous failure, one iteration later: the executor
        drops the allocation and the scheduler admits the request again.
        Cleared on return.
        """
        with self._load_lock:
            failed, self._failed_async_load_requests = self._failed_async_load_requests, set()
        return failed

    def _drain_loads(self, index: int) -> None:
        # The same device adoption as the save thread: a stream created here
        # has to belong to this rank's device, not to the thread's default.
        if self._device_index is not None:
            torch.cuda.set_device(self._device_index)
        staging = self._async_stagings[index] if self._async_stagings else None
        stream = torch.cuda.Stream() if staging is not None and torch.cuda.is_available() else None
        with self._load_lock:
            while len(self._load_streams) <= index:
                self._load_streams.append(None)
            self._load_streams[index] = stream
        handle = _stream_handle(stream) if staging is not None else 0

        while True:
            item = self._load_queue.get()
            if item is None:
                return
            request_id, transfers, enqueued_at = item
            started_at = time.monotonic()
            with self._load_lock:
                self._async_active.add(request_id)
            failed = False
            num_failed_pages = 0
            loaded_bytes = 0
            try:
                keys, addresses, sizes, total_pages = self._resolve([transfers])
                owners = [request_id] * len(keys)
                failed_requests, num_failed_pages, first_failure, loaded_bytes = self._load_pages(
                    keys, addresses, sizes, owners, staging, handle, self._async_batch_size
                )
                failed = bool(failed_requests)
                if failed:
                    logger.warning(
                        f"mooncake-store rank {self._rank} failed to load {num_failed_pages} "
                        f"page(s) for parked request {request_id}; it is restarted without "
                        f"the offered prefix. First failure: {first_failure!r}"
                    )
                else:
                    logger.debug(
                        f"mooncake-store rank {self._rank} worker {index} loaded "
                        f"{total_pages} pages for parked request {request_id}"
                    )
            except Exception as exc:
                # Broad on purpose: this is the thread boundary. The request is
                # restarted rather than left parked on a load that will never
                # report, and the error is re-raised on the executor thread.
                failed = True
                logger.error(
                    f"mooncake-store asynchronous load failed on rank {self._rank}: "
                    f"{type(exc).__name__}: {exc}"
                )
                with self._load_lock:
                    if self._load_error is None:
                        self._load_error = exc
            finally:
                finished_at = time.monotonic()
                with self._load_lock:
                    self._async_active.discard(request_id)
                    self._async_pending.discard(request_id)
                    self._async_done[request_id] = failed
                    self._async_stats.record(
                        started_at - enqueued_at,
                        finished_at - started_at,
                        num_failed_pages,
                        loaded_bytes,
                    )

    def _maybe_log_async_stats(self) -> Optional[str]:
        """Emit the loader summary at most once per interval, and return it.

        Nothing is emitted while the loader is idle and has completed nothing
        since the last line.
        """
        now = time.monotonic()
        with self._load_lock:
            due = now - self._async_stats_last >= ASYNC_STATS_INTERVAL
            busy = self._async_stats.completed or self._async_pending
            if not (due and busy):
                return None
            queued = len(self._async_pending) - len(self._async_active)
            line = (
                f"mooncake-store async-load stats: rank {self._rank} "
                f"workers {self._async_workers} "
                + self._async_stats.summary(queued, len(self._async_active))
            )
            self._async_stats.reset()
            self._async_stats_last = now
        logger.info(line)
        return line

    def drop_bound_saves(self, request_ids: Iterable[int]) -> int:
        """Remove these requests' saves from the metadata bound for this pass.

        A request that leaves the batch after `build_connector_meta` ran
        computes none of its scheduled tail, so none of it may be published.
        Its pages are freed on the way out and another request may already have
        filled them, so a save left bound would publish those bytes under this
        request's valid keys. Edited in place, which is the object
        `wait_for_save` reads.

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
                f"{dropped_ids} ({removed} entries) from this pass"
            )
        return removed

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
        if metadata is None or not metadata.saves or not self._config.role.saves:
            return
        self._reraise_background_error()

        # The pages are written by kernels still queued on this stream, so the
        # event is the handoff: the thread reads GPU memory only after the
        # pass retires, without the executor loop waiting for it.
        event = torch.cuda.Event()
        event.record(stream)

        with self._save_lock:
            for transfers in metadata.saves:
                self._outstanding_saves[transfers.request_id] += 1
        self._save_queue.put((event, list(metadata.saves)))

    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Report which requests' saves and asynchronous loads have landed.

        Args:
            finished_gen_req_ids: Requests that will produce no further KV.
            started_loading_req_ids: Requests the runtime parked for an
                asynchronous load. Without `async_load` the scheduler offers
                none, and anything here is echoed back so the runtime does not
                wait on something that already happened. With it, a request is
                reported only once a load thread has put its pages in place.

        Returns:
            Requests that have finished saving, and requests that have finished
            loading.
        """
        self._reraise_background_error()
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

        self._maybe_log_async_stats()
        finished_loading: List[int] = []
        with self._load_lock:
            self._async_announced.update(started_loading_req_ids)
            for request_id in list(self._async_announced):
                if request_id in self._async_done:
                    failed = self._async_done.pop(request_id)
                elif request_id in self._async_pending:
                    continue
                else:
                    # Parked by the runtime but never handed to this worker,
                    # so the adapter found no page to load into. Nothing
                    # usable is behind the skipped prefix, so restart it.
                    logger.warning(
                        f"mooncake-store rank {self._rank}: request {request_id} was "
                        "parked for an asynchronous load that was never started; "
                        "restarting it"
                    )
                    failed = True
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
            # The same thread boundary and handoff as the transfer loop below.
            # A thread that died here would leave every later save outstanding
            # against nothing, so the requests holding those pages would never
            # retire and the worker would look merely slow.
            logger.error(
                f"mooncake-store save thread failed to start on rank {self._rank}: "
                f"{type(exc).__name__}: {exc}"
            )
            with self._save_lock:
                if self._save_error is None:
                    self._save_error = exc
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
                # Broad on purpose: this is the thread boundary. Anything that
                # escapes here would be lost, so it is stashed and re-raised on
                # the executor thread at the next connector call.
                logger.error(
                    f"mooncake-store save failed on rank {self._rank}: {type(exc).__name__}: {exc}"
                )
                with self._save_lock:
                    if self._save_error is None:
                        self._save_error = exc
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
            if failures:
                # A dropped write only costs a future cache miss, which is not
                # worth failing a request that already answered.
                logger.warning(
                    f"mooncake-store rank {self._rank} failed to save {failures} of "
                    f"{len(pending)} pages"
                )

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

    def _reraise_save_error(self) -> None:
        with self._save_lock:
            error = self._save_error
            self._save_error = None
        if error is not None:
            raise RuntimeError("mooncake-store background save failed") from error

    def _reraise_background_error(self) -> None:
        """Surface whatever a save or load thread could not report itself."""
        self._reraise_save_error()
        with self._load_lock:
            error = self._load_error
            self._load_error = None
        if error is not None:
            raise RuntimeError("mooncake-store background load failed") from error

    def shutdown(self) -> None:
        """Stop the background threads, then release what they were reading.

        Idempotent.
        """
        for _ in self._load_threads:
            self._load_queue.put(None)
        for thread in self._load_threads:
            thread.join(timeout=DRAIN_TIMEOUT)
            if thread.is_alive():
                # The same reasoning as the save thread below: a loader writes
                # into the KV pools through the store handle and its staging
                # slots, so both are left in place rather than freed under it.
                # The threads are kept so that a later call retries the join.
                logger.error(
                    f"mooncake-store rank {self._rank}: a load thread did not stop "
                    f"within {DRAIN_TIMEOUT:g}s, so the store handle and its "
                    "staging buffers are left in place rather than freed under a "
                    "transfer still writing them."
                )
                return
        self._load_threads = []
        thread = self._save_thread
        if thread is not None:
            self._save_queue.put(None)
            thread.join(timeout=DRAIN_TIMEOUT)
            if thread.is_alive():
                # The thread reads the KV pools through the store handle and
                # the staging slots below, so releasing either takes the memory
                # out from under a transfer in flight. Leaking both is the safer
                # end: the process is going down anyway, the thread is a daemon
                # and does not hold it open, and a later call retries the join.
                logger.error(
                    f"mooncake-store rank {self._rank}: the save thread did not stop "
                    f"within {DRAIN_TIMEOUT:g}s, so the store handle and its "
                    "staging buffers are left in place rather than freed under a "
                    "transfer still reading them."
                )
                return
            self._save_thread = None
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
        self._save_staging = None
        self._async_stagings = []
        self._save_stream = None
        self._load_streams = []
        global _LOCAL_WORKER
        if _LOCAL_WORKER is self:
            _LOCAL_WORKER = None
            _LOCAL_WORKER_READY.clear()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:  # noqa: S110 - interpreter teardown, nothing left to report to
            pass
