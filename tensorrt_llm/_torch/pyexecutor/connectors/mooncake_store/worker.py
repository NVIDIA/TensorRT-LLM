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

Loads are synchronous, and the runtime has already counted the loaded tokens as
computed, so a page missing at load time must never reach a forward pass. A
rank that holds whole pages and its own batch hands the affected request back
for a restart without the offered prefix (`reports_failed_loads`): a single-rank
server or an attention-DP owner. Under tensor parallelism the ranks hold shards
of the same page and cannot restart a request alone, so there a missing page
fails the executor.

Saves are asynchronous and gated on a CUDA event, since the pages are complete
only once the forward pass that wrote them has retired and blocking the
executor loop on an RDMA write is the cost the store exists to avoid. The
scheduler reports such a request as saving asynchronously, which keeps its
pages pinned until `get_finished` says the writes landed. It counts a save as
stored once `wait_for_save` has queued it, so saves bound for a pass that never
ran are emitted again. A save that fails is logged and dropped: a write that
did not land costs a future cache miss and nothing else, so the request retires
either way. Only the save thread failing to start is fatal, because that would
strand every request behind pages nothing will release.

A capacity-only worker opens its handle and stops there, with no layout, no
buffer registration and no save thread, so a node can lend host memory to the
pool without an HCA that can pin GPU pages.
"""

import threading
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
from .config import CONFIG_PATH_ENV, MooncakeStoreConnectorConfig, local_address, pool_config
from .gpudirect import PEERMEM_ENV, allocation_range, format_diagnosis, select_registration_path
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

#: Set by the worker's constructor so the scheduler adapter, which the executor
#: builds in the same process (on rank 0 under tensor parallelism, on every
#: owner under attention DP), can reach the store handle without a second
#: connection. `py_executor_creator` constructs the two concurrently, so the
#: adapter waits on `_LOCAL_WORKER_READY`.
_LOCAL_WORKER: Optional["MooncakeStoreConnectorWorker"] = None
_LOCAL_WORKER_READY = threading.Event()

#: Seconds `shutdown` gives the save thread to drain its queue and return.
#: Sized for a backlog of pages over a congested fabric rather than a healthy
#: one, since what follows the wait is only safe once the thread has stopped.
SAVE_DRAIN_TIMEOUT = 30.0


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
    #: Registering pages drops the automatic host tier, and a prefix restore is
    #: chosen only after the final batch, so on a disaggregated server the
    #: pages each admitted request needs to finish have to stay free.
    requires_completion_reservation = True

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

        global _LOCAL_WORKER
        _LOCAL_WORKER = self
        _LOCAL_WORKER_READY.set()

        logger.warning(
            f"mooncake-store worker rank {self._rank}/{self._world_size} ready "
            f"(attention shard {self._attention_rank}/{self._attention_world_size}, "
            f"role={self._config.role.value}, model_key={self._model_key}, "
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

        logger.warning(
            f"mooncake-store worker rank {self._rank} registered layout: {addressing.describe()}"
        )

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
        self._batch_size = min(self._config.transfer_batch_size, num_slots)
        if self._batch_size < self._config.transfer_batch_size:
            logger.warning(
                f"mooncake-store rank {self._rank} reduced its transfer batch from "
                f"{self._config.transfer_batch_size} to {self._batch_size} pages: "
                f"staging {max_bytes_per_page} B pages within the "
                f"{MAX_STAGING_BUFFER_BYTES} B pinned-memory ceiling does not fit "
                f"more. Lower transfer_batch_size to make the reduction explicit."
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
        return hit_blocks

    # ---- load path ----

    def start_load_kv(self, stream: torch.cuda.Stream):
        """Pull every scheduled page into its GPU slot before the forward pass.

        The runtime has already counted these tokens as computed. When a page
        is missing and this rank can restart a request alone, the request is
        recorded for `take_failed_load_requests`, its remaining pages are not
        fetched, and its saves leave the bound metadata; the other requests
        load as usual.

        Args:
            stream: The executor's stream, on which staged scatters are issued.

        Raises:
            RuntimeError: When a page is missing on a tensor-parallel rank, or
                a zero-copy load has no allocation fence to wait on.
        """
        metadata: Optional[MooncakeStoreMetadata] = self.get_connector_meta()
        if metadata is None or not metadata.loads:
            return

        keys, addresses, sizes, total_pages = self._resolve(metadata.loads)
        if not keys:
            return
        # `_resolve` emits one key per page in transfer order, so this names the
        # request every key is loaded for.
        owners = [entry.request_id for entry in metadata.loads for _ in entry.pages]

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
            fence({entry.request_id for entry in metadata.loads})

        failed_requests, num_failed_pages, first_failure, _ = self._load_pages(
            keys,
            addresses,
            sizes,
            owners,
            staging,
            handle,
            self._batch_size,
            recover=self._recover_failed_loads,
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
    ) -> Tuple[Set[int], int, Optional[str], int]:
        """Fetch pages batch by batch, scattering staged ones onto the device.

        A store call is synchronous, and a staged batch's scatter is synced
        before the next batch reuses its slots, so no write into a page is in
        flight on return.

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
            if staging is None:
                target_addresses, target_sizes = batch_addresses, batch_sizes
            else:
                target_addresses, target_sizes = describe_batch_for_get(staging, batch_sizes)
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
                unstage_batch_after_get(staging, batch_addresses, batch_sizes, handle, only=landed)
                # The next batch reuses the slots and the forward pass reads
                # these pages, so the scatter has to complete before either.
                _sync_stream(handle)
        return failed_requests, num_failed_pages, first_failure, loaded_bytes

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

    def shutdown(self) -> None:
        """Stop the save thread, then release what it was reading. Idempotent."""
        thread = self._save_thread
        if thread is not None:
            self._save_queue.put(None)
            thread.join(timeout=SAVE_DRAIN_TIMEOUT)
            if thread.is_alive():
                # The thread reads the KV pools through the store handle and
                # the staging slots below, so releasing either takes the memory
                # out from under a transfer in flight. Leaking both is the safer
                # end: the process is going down anyway, the thread is a daemon
                # and does not hold it open, and a later call retries the join.
                logger.error(
                    f"mooncake-store rank {self._rank}: the save thread did not stop "
                    f"within {SAVE_DRAIN_TIMEOUT:g}s, so the store handle and its "
                    "staging buffers are left in place rather than freed under a "
                    "transfer still reading them."
                )
                return
            self._save_thread = None
        if self._store is not None:
            # Before the store closes and the cache manager frees the pools,
            # so the fabric holds no registration over memory handed out again.
            self._unregister_pools()
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
