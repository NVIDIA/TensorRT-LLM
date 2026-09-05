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
between that pool and its own GPU KV cache. It is also the only place that knows
how a page is addressed and how a key is spelled, which is why the leader,
colocated with rank 0's worker in the same process, asks it to run prefix
lookups instead of rebuilding that knowledge.

Loads are synchronous: the runtime has already told the scheduler those tokens
are computed, so the bytes must be in place before the forward pass reads them,
and a failed load is a wrong answer rather than a slow one.

Saves are asynchronous and gated on a CUDA event. The pages are only complete
once the forward pass that wrote them has retired, and blocking the executor
loop on an RDMA write is exactly the cost the store is supposed to avoid. The
scheduler reports such a request as saving asynchronously, which keeps its pages
pinned until `get_finished` says the writes landed.
"""

import threading
import traceback
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger

from ..kv_cache_connector import KvCacheConnectorWorker
from ..kv_cache_layout import KvCacheLayout
from .addressing import PageAddressing
from .config import CONFIG_PATH_ENV, MooncakeStoreConnectorConfig
from .keys import KeyNamespace
from .metadata import MooncakeStoreMetadata, RequestTransfers
from .staging import (
    HostStagingPool,
    describe_batch_for_get,
    plan_slot_geometry,
    stage_batch_for_put,
    unstage_batch_after_get,
)
from .staging import sync_stream as _sync_stream
from .validation import validate_layout, validate_llm_args

__all__ = ["MooncakeStoreConnectorWorker", "resolve_local_worker"]

#: Set by the worker's constructor so the leader, which the executor builds in
#: the same process on rank 0, can reach the store handle without a second
#: connection or an out-of-band channel. See `py_executor_creator`, which
#: constructs scheduler and worker concurrently for exactly this kind of
#: mutual dependency.
_LOCAL_WORKER: Optional["MooncakeStoreConnectorWorker"] = None
_LOCAL_WORKER_READY = threading.Event()


def resolve_local_worker(timeout: float = 60.0) -> "MooncakeStoreConnectorWorker":
    """The worker living in this process, once it has been constructed.

    Args:
        timeout: Seconds to wait. Construction is concurrent with the leader's,
            so a short wait is expected; exceeding it means the worker failed.

    Returns:
        The process-local worker.
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
    """Connect to the Mooncake master and return a live store handle."""
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
            f"metadata={config.metadata_server!r}, protocol={config.protocol!r}). "
            f"Check the config named by {CONFIG_PATH_ENV}."
        )
    return store


def _default_hostname() -> str:
    import socket

    return socket.gethostbyname(socket.gethostname())


def _batched(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _stream_handle(stream) -> int:
    """The raw CUDA stream handle behind a torch stream, or a handle as given.

    `None` maps to 0, the default stream, which is what the runtime passes when
    it has no stream of its own to offer.
    """
    if stream is None:
        return 0
    return int(getattr(stream, "cuda_stream", stream))


class MooncakeStoreConnectorWorker(KvCacheConnectorWorker):
    """Moves KV pages between this rank's GPU cache and the Mooncake pool."""

    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)

        validate_llm_args(llm_args)
        self._config = MooncakeStoreConnectorConfig.from_env()
        self._rank = mpi_rank()
        self._world_size = mpi_world_size()
        self._model_key = self._config.resolve_model_key(llm_args.model)

        self._addressing: Optional[PageAddressing] = None
        # Namespaces for this rank, used for both directions of transfer.
        self._namespaces: Dict[int, KeyNamespace] = {}
        # The same namespaces for every rank. A prefix is only reusable when all
        # shards of it are present, so a lookup has to ask about all of them.
        self._peer_namespaces: Dict[int, Tuple[KeyNamespace, ...]] = {}

        self._store = _open_store(self._config)

        self._save_queue: "Queue[Optional[Tuple[torch.cuda.Event, List[RequestTransfers]]]]" = (
            Queue()
        )
        self._save_thread: Optional[threading.Thread] = None
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

        global _LOCAL_WORKER
        _LOCAL_WORKER = self
        _LOCAL_WORKER_READY.set()

        logger.info(
            f"mooncake-store worker rank {self._rank}/{self._world_size} ready "
            f"(role={self._config.role.value}, model_key={self._model_key}, "
            f"master={self._config.master_server_address})"
        )

    # ---- registration ----

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        """Reject the V1 single-pool registration.

        Raises:
            NotImplementedError: Always. Identity here is a hash chain the
                connector computes itself, keyed per layer group, and the V1
                manager supplies real block hashes over a single flat block
                space instead. Running the V2 addressing against V1 block ids
                would silently mislabel pages, so V1 is refused rather than
                approximated.
        """
        raise NotImplementedError(
            "The mooncake-store connector requires KVCacheManagerV2. Set "
            "kv_cache_config.use_kv_cache_manager_v2=True."
        )

    def register_kv_cache_layout(self, layout: KvCacheLayout) -> None:
        """Register the KV pools with Mooncake and start the save thread."""
        if self._addressing is not None:
            raise RuntimeError("KV cache layout already registered")

        validate_layout(layout)
        addressing = PageAddressing(layout)
        # Torch's current device is thread-local, so read it here on the
        # executor thread; the save thread would otherwise see device 0.
        if torch.cuda.is_available():
            self._device_index = torch.cuda.current_device()
        if self._config.stage_through_host:
            self._open_staging(addressing)
        else:
            for start, end in addressing.registration_ranges():
                status = self._store.register_buffer(start, end - start)
                if status != 0:
                    raise RuntimeError(
                        f"MooncakeDistributedStore.register_buffer failed with status "
                        f"{status} for [{start:#x}, {end:#x}). Without registration "
                        "the store cannot read or write these pages. Registering "
                        "device memory needs GPUDirect RDMA (nvidia_peermem or "
                        "dma-buf); where that is unavailable, set "
                        "stage_through_host to pass pages through pinned host "
                        "memory instead."
                    )

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

        if self._config.role.saves:
            self._save_thread = threading.Thread(
                target=self._drain_saves,
                name=f"mooncake-store-save-{self._rank}",
                daemon=True,
            )
            self._save_thread.start()

        logger.info(
            f"mooncake-store worker rank {self._rank} registered layout: {addressing.describe()}"
        )

    def _open_staging(self, addressing: PageAddressing) -> None:
        """Allocate and register the pinned slots pages will pass through.

        Only the directions this role drives get a pool, since each one costs a
        pinned allocation of its own. The GPU pools are left unregistered,
        which is the point of the mode.
        """
        max_bytes_per_page = max(
            addressing.bytes_per_page(layer_group_id)
            for layer_group_id in addressing.layer_group_ids
        )
        slot_bytes, num_slots = plan_slot_geometry(
            max_bytes_per_page,
            self._config.transfer_batch_size,
            self._config.staging_buffer_bytes,
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
                f"staging {max_bytes_per_page} B pages within "
                f"{self._config.staging_buffer_bytes} B does not fit more. Raise "
                f"staging_buffer_bytes to restore the configured batch size."
            )

    def _namespace(self, rank: int, layer_group_id: int, bytes_per_page: int) -> KeyNamespace:
        return KeyNamespace(
            cache_prefix=self._config.cache_prefix,
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

    def count_prefix_hit(self, block_hashes: Sequence[bytes]) -> int:
        """How many leading blocks of `block_hashes` are fully present.

        A block counts only when every layer group and every rank has its page,
        because a prefix is replayed as a whole. The scan stops at the first
        incomplete block: the runtime consumes a prefix, so a later hit is not
        usable on its own.

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
        self._reraise_save_error()

        keys, addresses, sizes, total_pages = self._resolve(metadata.loads)
        if not keys:
            return

        staging = self._load_staging
        handle = _stream_handle(stream) if staging is not None else 0

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
                # partial load leaves the forward pass reading uninitialized KV
                # and silently producing wrong tokens. Fail loudly instead.
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

    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """No-op: loads complete in `start_load_kv`.

        Transfers are whole pages, so a page's bytes for every layer in a group
        land in one store call rather than layer by layer. There is nothing left
        outstanding by the time the first layer runs.
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
        self._reraise_save_error()

        # The pages are written by kernels still queued on this stream. The event
        # is the handoff: the thread reads GPU memory only after the pass retires,
        # and the executor loop is not blocked waiting for that.
        event = torch.cuda.Event()
        event.record(stream)

        with self._save_lock:
            for transfers in metadata.saves:
                self._outstanding_saves[transfers.request_id] += 1
        self._save_queue.put((event, list(metadata.saves)))

    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Report which requests' saves have landed.

        Args:
            finished_gen_req_ids: Requests that will produce no further KV.
            started_loading_req_ids: Requests loading asynchronously. Always
                empty here, since `get_num_new_matched_tokens` only ever
                offers synchronous loads; echoed back so the runtime does not
                wait on something that already happened.

        Returns:
            Requests that have finished saving, and requests that have finished
            loading.
        """
        self._reraise_save_error()
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
        # A new thread starts on device 0, so adopt the device captured on the
        # executor thread. Otherwise a stream created below belongs to device 0
        # while the KV pointers belong to the rank's device, and the copy fails
        # with cudaErrorInvalidValue on every rank except 0.
        if self._device_index is not None:
            torch.cuda.set_device(self._device_index)
        if self._save_staging is not None and torch.cuda.is_available():
            # Owned by this thread so the gather never queues behind the
            # executor's work, and created after set_device so it lands on the
            # rank's device.
            self._save_stream = torch.cuda.Stream()
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
            # scheduler cannot know this: it holds no store handle, and the
            # answer changes between the time it builds metadata and now.
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
                # A dropped write only costs a future cache miss, so it is worth
                # a warning rather than failing a request that already answered.
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

    def shutdown(self) -> None:
        """Stop the save thread and release the store handle. Idempotent."""
        thread, self._save_thread = self._save_thread, None
        if thread is not None:
            self._save_queue.put(None)
            thread.join(timeout=30.0)
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
