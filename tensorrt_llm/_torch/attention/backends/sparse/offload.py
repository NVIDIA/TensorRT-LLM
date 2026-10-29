# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse KV cache offload shared by sparse attention models.

With offload, the history a sparse attention layer selects from lives in host
memory, and each decode step fetches the selected units into per-request rows
of a GPU arena. This module holds what every offloading model shares:

- the per-layer descriptors and the attention metadata's persistent state;
- ``SparseOffloadCacheMixin``, the KV cache manager side of a decode step:
  publishing the native Batch, preparing write rows, fetching selections and
  advancing history after the forward.

A model's cache manager mixes ``SparseOffloadCacheMixin`` in before
``KVCacheManagerV2``, marks its offloaded buffers sparse, and describes each
offloaded layer with ``_build_sparse_offload_descriptor``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Collection, Iterable

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    nvtx_range_debug,
    prefer_pinned,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    AttentionLayerConfig,
    Batch,
    BatchRow,
    BufferId,
    CudaStream,
    DataRole,
    LayerGroupId,
    LayerId,
    _KVCache,
)

from .offload_kernels import select_sparse_history_units

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor import llm_request
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig

# Attention metadata holding a SparseOffloadState: the eager one, and the CUDA
# graph metadata, which share theirs through the graph buffer pool.
SPARSE_OFFLOAD_STATE_COPIES = 2

# The sparse MLA FMHA reads the NVFP4 scratch as its aux KV pool and can read up to
# one KV tile (128 rows) past the last row its indices name (https://nvbugs/6796296),
# so the scratch carries a zeroed tail of that many rows.
NVFP4_SCRATCH_TAIL_ROWS = 128


@dataclass(frozen=True, slots=True)
class SparseOffloadPaddingRegion:
    """Unused buffer bytes across arena leases, initialized during manager construction."""

    base_address: int
    bytes_per_lease: int
    lease_stride: int


@dataclass(frozen=True, slots=True)
class SparseOffloadLayerDescriptor:
    """All sparse buffers of one model layer, in KVCM's required fetch order.

    ``unit_tokens`` original tokens make one selection unit. Fetch returns a
    separate physical index for each buffer, relative to that buffer's
    ``get_sparse_cache_base_address()``. These indices already include the
    request and buffer placement; no page scaling or layer offset is needed.
    Descriptor construction validates matching data/scale pool counts and
    buffer positions, including padding. Their index columns are equal, so a
    writer and a gather consume the data column for both pools while retaining
    their respective base addresses and byte strides.
    ``arena_row_counts`` covers each complete pool region across all leases
    and buffers, including padding; it is independent of host-page capacity.

    ``selection_owner`` is the model layer that resolves this layer's
    selection, where a model runs one indexer across several layers. ``None``
    means the layer selects for itself, which is the ordinary case. A layer
    naming another is relying on being handed that layer's selection verbatim;
    nothing checks that, and a different selection would silently resolve
    against the owner's residency.
    """

    buffer_ids: tuple[BufferId, ...]
    arena_base_addresses: tuple[int, ...]
    arena_row_counts: tuple[int, ...]
    data_buffer_index: int
    scale_buffer_index: int | None
    group_id: LayerGroupId
    unit_tokens: int
    padding_regions: tuple[SparseOffloadPaddingRegion, ...] = ()
    selection_owner: int | None = None


@dataclass(kw_only=True)
class SparseOffloadState:
    """Fixed-capacity storage owned by one serial attention metadata instance.

    Index and count tensors are int32. Selections are [B, K], fetch outputs are
    [B, F, K], write destinations are [B, F], the active count is [1] and the
    Batch row map is [B]. B is the native Batch capacity, including when an
    attention graph uses fewer query rows. Storage is shared by layers with the
    same buffer count F. Reuse must follow the previous layer's attention on the
    consuming stream.

    NVFP4 additionally uses persistent FP8 scratch [B, K, head_dim] for the
    dequantized rows consumed by attention. Arena allocations remain owned by
    KVCM; the cache manager retains non-owning tensor views sized by arena
    geometry. ``scatter_start_positions`` is initialized to zero once and
    supplies the one-row-page view a writer can address the arena through.

    ``history_lengths`` is a zero-copy view of Batch's published token counts,
    bound outside capture. Batch owns the page tables and watermarks; metadata
    does not maintain another copy. A request keeps its Batch row while it stays
    scheduled, so rows need not follow the input order or be contiguous.
    ``batch_rows`` maps each request's input position to its row, and positions
    past the live requests to -1; it is refreshed outside capture each step.
    Selections, fetch output, write destinations and watermarks are indexed by
    Batch row; per-request inputs and attention outputs by input position.

    ``prepared`` means preparation was enqueued successfully; stream ordering
    establishes device readiness. ``is_prefill`` is set by host batch validation
    before execution, never inferred from a device count inside a layer.
    ``write_layer`` tracks which layer enqueued the shared write mapping during
    eager execution or capture. Graph replay uses the captured GPU operations;
    this host marker does not determine device readiness.

    An in-flight selection is masked out of the history fetch. Its physical
    index must come from the separately prepared write mapping when assembling
    attention inputs.
    """

    layers: dict[int, SparseOffloadLayerDescriptor]
    active_request_count: torch.Tensor
    batch_rows: torch.Tensor
    selected_history_units: torch.Tensor
    fetched_indices: dict[int, torch.Tensor]
    write_indices: dict[int, torch.Tensor]
    scatter_start_positions: torch.Tensor
    #: Which slice of ``fetched_indices`` and ``write_indices`` each of a layer's buffers
    #: reads. Buffers whose arena indices coincide share one, and the slices no buffer
    #: maps to are never written, so a buffer must be read at its slot rather than at its
    #: own position. Filled on first use, the earliest the native Batch exists to ask.
    out_slots: dict[int, tuple[int, ...]] = field(default_factory=dict)
    nvfp4_scratch: torch.Tensor | None = None
    batch: Batch | None = None
    history_lengths: torch.Tensor | None = None
    prepared: bool = False
    is_prefill: bool = False
    write_layer: int | None = None

    @staticmethod
    def device_bytes(
        *,
        max_batch_size: int,
        topk: int,
        buffer_counts: Collection[int],
        nvfp4_head_dim: int | None,
    ) -> int:
        """Return the device bytes one instance allocates.

        Mirrors the allocations of ``allocate_sparse_offload_state``.

        Args:
            max_batch_size: Rows B, the native Batch capacity.
            topk: Units one row selects, K.
            buffer_counts: The distinct sparse buffer counts F of the layers.
            nvfp4_head_dim: Head dim of the dequantized rows with NVFP4, else None.
        """
        # The active count, the selections, the scatter start positions and the
        # Batch row map, then the fetch outputs and write destinations of each
        # buffer count.
        index_entries = 1 + max_batch_size * topk + 2 * max_batch_size
        index_entries += sum(max_batch_size * count * (topk + 1) for count in buffer_counts)
        nvfp4_bytes = 0
        if nvfp4_head_dim is not None:
            nvfp4_rows = max_batch_size * topk + NVFP4_SCRATCH_TAIL_ROWS
            nvfp4_bytes = nvfp4_rows * nvfp4_head_dim * torch.float8_e4m3fn.itemsize
        return index_entries * torch.int32.itemsize + nvfp4_bytes


def allocate_sparse_offload_state(
    allocate: Callable[[str, tuple[int, ...], torch.dtype], torch.Tensor],
    layers: dict[int, SparseOffloadLayerDescriptor],
    *,
    max_batch_size: int,
    topk: int,
    nvfp4_head_dim: int | None = None,
) -> SparseOffloadState:
    """Allocate an attention metadata's persistent offload state.

    Args:
        allocate: Returns an uninitialized tensor of a name, shape and dtype;
            the metadata's graph-buffer allocator.
        layers: The manager's descriptor of each offloaded local layer.
        max_batch_size: Rows B, the native Batch capacity.
        topk: Units one row selects.
        nvfp4_head_dim: Head dim of the NVFP4 gather's dequantized rows, else None.
    """

    def index_tensor(name: str, shape: tuple[int, ...], value: int) -> torch.Tensor:
        tensor = allocate(name, shape, torch.int32)
        tensor.fill_(value)
        return tensor

    def nvfp4_scratch(head_dim: int) -> torch.Tensor:
        rows = max_batch_size * topk
        flat = allocate(
            "nvfp4_scratch", (rows + NVFP4_SCRATCH_TAIL_ROWS, head_dim), torch.float8_e4m3fn
        )
        flat[rows:].zero_()
        return flat[:rows].view(max_batch_size, topk, head_dim)

    buffer_counts = sorted({len(descriptor.buffer_ids) for descriptor in layers.values()})
    return SparseOffloadState(
        layers=layers,
        active_request_count=index_tensor("active_request_count", (1,), 0),
        batch_rows=index_tensor("batch_rows", (max_batch_size,), -1),
        selected_history_units=index_tensor(
            "selected_history_units", (max_batch_size, topk), BAD_PAGE_INDEX
        ),
        fetched_indices={
            count: index_tensor(
                f"fetched_indices_{count}", (max_batch_size, count, topk), BAD_PAGE_INDEX
            )
            for count in buffer_counts
        },
        write_indices={
            count: index_tensor(f"write_indices_{count}", (max_batch_size, count), BAD_PAGE_INDEX)
            for count in buffer_counts
        },
        scatter_start_positions=index_tensor("scatter_start_positions", (max_batch_size,), 0),
        nvfp4_scratch=nvfp4_scratch(nvfp4_head_dim) if nvfp4_head_dim is not None else None,
    )


def get_sparse_offload_metadata_layers(metadata) -> dict[int, SparseOffloadLayerDescriptor]:
    """Validate an attention metadata's fixed offload configuration and return its layers.

    Offload runs single-beam and non-speculative, so per-batch beam and draft
    values start at defaults; the manager's configured limits are checked too,
    before any offload buffer is allocated. History policies are bound after
    manager construction, before metadata initialization, so they are checked
    once here rather than on each batch. An empty result means this pipeline
    stage has no offloaded layer.
    """
    manager = metadata.kv_cache_manager
    if (
        metadata.beam_width != 1
        or manager.max_beam_width != 1
        or metadata.max_draft_tokens
        or manager.max_draft_len
        or manager.max_total_draft_tokens
        or metadata.draft_kv_cache_manager is not None
    ):
        raise NotImplementedError("Sparse offload requires single-beam, non-speculative execution")
    manager._validate_sparse_history_policy()
    layers = manager.get_sparse_offload_descriptors()
    if layers and (
        metadata.sparse_mla_topk <= 0 or metadata.sparse_mla_topk != manager._sparse_offload_topk
    ):
        raise ValueError("Sparse offload metadata must match the configured positive top-k")
    return layers


@dataclass(frozen=True, slots=True)
class _SparseArenaLayout:
    """Where every sparse buffer sits in its pool's arena region."""

    # Coalesced-buffer position of each buffer, which is its pool index in its group.
    pool_indices: dict[BufferId, int]
    layer_configs: dict[LayerId, AttentionLayerConfig]
    # (buffers in the pool region, this buffer's position among them).
    positions: dict[BufferId, tuple[int, int]]
    max_units_by_group: dict[LayerGroupId, int]
    max_leases: int


@dataclass(slots=True)
class _SparseBatchMember:
    """A request seated in the native Batch."""

    cache: _KVCache
    # The row add_kv_cache returned. The runtime never moves a seated member, so the
    # row holds while the request stays scheduled, and with it the GPU cache lease.
    row: BatchRow
    # Page-table columns the row may have written since it joined. A cache
    # writes page indices only below its block count, so the host snapshot
    # copies no further than this.
    blocks: int


class SparseOffloadCacheMixin:
    """KV cache manager side of sparse KV cache offload, model-agnostic.

    Mix in before ``KVCacheManagerV2``. Call ``_init_sparse_offload`` before the
    base constructor; it reads the selection width while building the config.

    A request's sparse history advances right after each forward, since one
    query token per request writes one unit. Between ``prepare_resources`` and
    the post-forward update, the scheduled requests are members of a native
    Batch, which owns their host page-index rows; every path that mutates a
    request's cache releases the Batch first.

    Batch rows are slots. A request keeps its row, and with it the GPU cache
    lease, for as long as it stays scheduled; a departing request frees its row
    and an arriving one takes the lowest free row. Attention finds each request's
    row through ``SparseOffloadState.batch_rows`` rather than by input position.
    """

    _enable_kv_cache_offload: bool = False
    _sparse_offload_topk: int | None = None
    _sparse_batch: Batch | None = None
    _sparse_batch_prepared: bool = False

    def _init_sparse_offload(self, *, enabled: bool, topk: int | None) -> None:
        self._enable_kv_cache_offload = enabled
        self._sparse_offload_topk = topk
        self._sparse_batch = None
        self._sparse_batch_members: dict[int, _SparseBatchMember] = {}
        self._sparse_batch_prepared = False

    def _sparse_offload_nvfp4_head_dim(self) -> int | None:
        """Head dim of the dequantized rows an NVFP4 gather keeps per selection.

        None when no layer dequantizes into offload scratch.
        """
        return None

    def get_sparse_offload_scratch_bytes(self, config: KVCacheManagerConfig) -> int:
        """Return the device bytes of the attention metadata's sparse offload state.

        Counts a copy for the CUDA graph metadata even when graphs are off.
        """
        buffer_counts = {
            sum(buffer.is_sparse for buffer in layer.buffers)
            for layer in config.layers
            if getattr(layer, "sparse_cache", None) is not None
        }
        if not buffer_counts:
            return 0
        return SPARSE_OFFLOAD_STATE_COPIES * SparseOffloadState.device_bytes(
            max_batch_size=self.max_batch_size,
            topk=self._sparse_offload_topk,
            buffer_counts=buffer_counts,
            nvfp4_head_dim=self._sparse_offload_nvfp4_head_dim(),
        )

    def _has_sparse_offload_layers(self) -> bool:
        """Whether a local layer fetches through a sparse cache."""
        return self._enable_kv_cache_offload and any(
            getattr(layer, "sparse_cache", None) is not None
            and any(buffer.is_sparse for buffer in layer.buffers)
            for layer in self.impl.init_config.layers
        )

    # ------------------------------------------------------------------
    # Batch lifecycle
    # ------------------------------------------------------------------

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        self._sparse_batch_prepared = False
        if self._sparse_batch is not None:
            self._sparse_batch.wait_published()
        if self._enable_kv_cache_offload and any(
            not request.is_disagg_generation_init_state
            for request in scheduled_batch.context_requests
        ):
            # An offloading worker admits one query token per request per step, so it serves
            # the decode phase and runs no context batch. Disaggregated gen-init requests are
            # the exception only because they run no forward: they are resized and their KV
            # transferred before they join the Batch. Rejected here rather than in the
            # compressor, which sees this as a bare context count once the model is already
            # writing.
            raise NotImplementedError(
                "Sparse offload does not support prefill; it serves generation only"
            )
        super().prepare_resources(scheduled_batch)
        if not self._enable_kv_cache_offload:
            return
        requests = scheduled_batch.context_requests + scheduled_batch.generation_requests
        # Disaggregated gen-init requests are prepared before their KV transfer
        # starts and run no forward this step, so they never join the Batch. A
        # batch holding only those publishes nothing.
        forward_request_ids = [
            request.py_request_id
            for request in requests
            if not request.is_disagg_generation_init_state
        ]
        if forward_request_ids or not requests:
            self.prepare_sparse_batch(forward_request_ids)

    def _ensure_sparse_batch(self) -> Batch:
        """The Batch the sparse caches are seated in, created on first use.

        It owns the cell arenas, so its rows are what they are sized for and its
        base addresses are what a fetch's indices are relative to. Both are fixed
        once it exists, which is why a single Batch serves the manager's lifetime.
        """
        if self._sparse_batch is None:
            self._sparse_batch = Batch(
                self.impl,
                CudaStream(self._stream.cuda_stream),
                # The fetch and the write-back read converted indices, so every sparse
                # buffer needs a table. Their shape follows from the pool layout.
                index_tables=self.impl.sparse_index_table_specs(),
                max_batch_size=self.max_batch_size,
                max_blocks_per_seq=self.max_blocks_per_seq,
            )
        return self._sparse_batch

    def _sparse_arena_base(self, layer_id: int, role: DataRole) -> int:
        """Where this buffer's arena region starts, for a consumer of fetched units.

        The arenas belong to the Batch that seats the caches, so the address is
        asked of it.
        """
        return self._ensure_sparse_batch().get_sparse_cache_base_address(layer_id, role)

    def prepare_sparse_batch(self, request_ids: list[int]) -> None:
        """Publish scheduler or warmup requests, or join an already published batch.

        Attention metadata calls this outside capture with its final request order.
        Graph padding may append dummy requests; the live requests must be the
        scheduled ones, in any order. Warmup requests can reach this without
        prepare_resources().
        """
        if self._sparse_batch_prepared:
            self.get_sparse_batch(request_ids)
            torch.cuda.current_stream(self._stream.device).wait_stream(self._stream)
        else:
            self._prepare_sparse_batch(request_ids)

    def _prepare_sparse_batch(self, request_ids: list[int]) -> None:
        """Publish the scheduled caches before a forward.

        Continuing requests keep their Batch row, membership and sparse GPU residency
        across steps, whatever their input order; departing requests free their rows
        and arriving ones take the lowest free rows. Members write their page indices
        into Batch staging; the runtime's host tables, which dense consumers read,
        receive a snapshot after scheduling.
        """
        if not self._has_sparse_offload_layers():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Sparse Batch publication must run outside CUDA graph capture")
        if len(request_ids) > self.max_batch_size:
            raise ValueError("Scheduled requests exceed sparse Batch capacity")
        caches = {}
        padding = False
        for request_id in request_ids:
            if request_id == CUDA_GRAPH_DUMMY_REQUEST_ID:
                padding = True
                continue
            if padding:
                raise ValueError("CUDA-graph dummy requests must form a trailing suffix")
            if request_id in caches:
                raise ValueError("Sparse Batch requires unique request IDs")
            cache = self.kv_cache_map.get(request_id)
            if cache is None or not cache.is_active:
                raise ValueError("Sparse Batch requires active scheduled request caches")
            if cache.beam_width != 1:
                raise ValueError("Sparse Batch supports one beam per request")
            if cache.cuda_stream != self._stream.cuda_stream:
                raise RuntimeError("Sparse Batch requires the manager's cache stream")
            if not 0 <= cache.history_length <= cache.capacity:
                raise ValueError("Invalid KVCM history frontier")
            if cache.num_blocks > self.max_blocks_per_seq:
                raise ValueError("Request allocation exceeds sparse Batch page-table capacity")
            caches[request_id] = cache

        batch = self._ensure_sparse_batch()
        try:
            batch.wait_published()
            self._release_sparse_batch(
                request_id for request_id in self._sparse_batch_members if request_id not in caches
            )
            for request_id, cache in caches.items():
                member = self._sparse_batch_members.get(request_id)
                if member is None:
                    self._sparse_batch_members[request_id] = _SparseBatchMember(
                        cache=cache, row=batch.add_kv_cache(cache), blocks=cache.num_blocks
                    )
                else:
                    member.blocks = max(member.blocks, cache.num_blocks)
            self._refresh_sparse_page_index_bufs()
            stream = torch.cuda.current_stream(self._stream.device)
            # Reusing the device mirror follows the previous forward. Publish on
            # the cache stream so metadata preparation can also join it there.
            self._stream.wait_stream(stream)
            batch.publish()
            stream.wait_stream(self._stream)
            self._sparse_batch_prepared = True
        finally:
            if not self._sparse_batch_prepared:
                self._release_sparse_batch()

    def get_sparse_batch(self, request_ids: list[int]) -> Batch:
        """Return the published Batch after checking the attention metadata's requests.

        The live requests must be the Batch's members, each once, in any order: each
        keeps its own row, which ``prepare_sparse_offload`` maps from the input
        position. CUDA graph padding is a trailing suffix of dummy requests, which map
        to no row. Fetch inputs must use Batch.max_batch_size rows, even when a smaller
        attention profile is executing.
        """
        if self._sparse_batch is None or not self._sparse_batch_prepared:
            raise RuntimeError("Sparse Batch must be prepared before model execution")
        members = self._sparse_batch_members
        live_ids = request_ids[: len(members)]
        if (
            len(request_ids) > self._sparse_batch.max_batch_size
            or len(live_ids) != len(members)
            or set(live_ids) != members.keys()
            or any(req != CUDA_GRAPH_DUMMY_REQUEST_ID for req in request_ids[len(members) :])
        ):
            raise ValueError("Attention requests do not match the published sparse Batch")
        return self._sparse_batch

    def _refresh_sparse_page_index_bufs(self) -> None:
        """Snapshot Batch-owned page indices into the runtime's host tables for dense consumers.

        Copies each member's row up to the widest block count any member has had since
        joining. Past that, both tables hold BAD_PAGE_INDEX for every member: a cache
        writes only below its block count, and its binding filled the rest.
        """
        if not self._sparse_batch_members:
            return
        assert self._sparse_batch is not None
        members = sorted(self._sparse_batch_members.items(), key=lambda item: item[1].row)
        width = max(member.blocks for _, member in members)
        if width == 0:
            return
        slots = torch.tensor(
            [
                self.index_mapper.get_index(request_id) * self.max_beam_width
                for request_id, _ in members
            ],
            dtype=torch.long,
            device="cpu",
        )
        first, last = members[0][1].row, members[-1][1].row
        # Members without free rows between them are one slice; otherwise gather them.
        rows = (
            None
            if last - first + 1 == len(members)
            else torch.tensor([member.row for _, member in members], dtype=torch.long)
        )
        for pool_id in range(self.num_pools):
            table = torch.from_dlpack(self._sparse_batch.coalesced_page_table(pool_id))[:, :width]
            self.host_kv_cache_block_offsets[pool_id, :, 0, :width].index_copy_(
                0, slots, table[first : last + 1] if rows is None else table.index_select(0, rows)
            )

    def _release_sparse_batch(self, request_ids: Iterable[int] | None = None) -> None:
        """Detach departing requests while retaining other members' GPU cache leases."""
        if self._sparse_batch is None:
            return
        self._sparse_batch.wait_published()
        request_ids = list(self._sparse_batch_members if request_ids is None else request_ids)
        for request_id in request_ids:
            member = self._sparse_batch_members.get(request_id)
            if member is None:
                continue
            self._sparse_batch_prepared = False
            self._sparse_batch.remove_kv_cache(member.cache)
            del self._sparse_batch_members[request_id]
            # Removal restores the cache's internal indices. Rebind the runtime's
            # host table so subsequent resizes also reach the dense consumers.
            self._set_page_index_bufs(request_id, member.cache)

    def suspend_request(self, req: llm_request.LlmRequest) -> None:
        self._release_sparse_batch([req.py_request_id])
        self._order_sparse_history_update([req])
        super().suspend_request(req)

    def free_resources(self, request: llm_request.LlmRequest, pin_on_release: bool = False) -> None:
        self._release_sparse_batch([request.py_request_id])
        self._order_sparse_history_update([request])
        super().free_resources(request, pin_on_release)

    def release_index_slot(self, request_id: int) -> None:
        self._release_sparse_batch([request_id])
        super().release_index_slot(request_id)

    def shutdown(self) -> None:
        self._release_sparse_batch()
        self._sparse_batch = None
        super().shutdown()

    def _validate_sparse_history_policy(self) -> None:
        """Check fixed history policies during attention metadata initialization."""
        if (
            self.enable_block_reuse
            or self.kv_compression_manages_history
            or self._has_cp_helix
            or self.is_draft
        ):
            raise NotImplementedError(
                "Sparse offload requires block reuse, KV compression, CP Helix, and draft caches disabled"
            )

    def _order_sparse_history_update(self, requests: Iterable[llm_request.LlmRequest]) -> None:
        """Order writeback after all model producers and attention readers.

        The executor orders its calling stream after model forward.
        Join it onto the cache stream before flushing completed units or resizing
        request caches. prepare_sparse_batch() supplies the reverse dependency
        for the next forward.
        """
        if not self._enable_kv_cache_offload:
            return
        active = False
        for request in requests:
            cache = self.kv_cache_map.get(request.py_request_id)
            if cache is None or not cache.is_active:
                continue
            if cache.cuda_stream != self._stream.cuda_stream:
                raise RuntimeError("Sparse history updates require the manager's cache stream")
            active = True
        if active:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Sparse history updates must run outside CUDA graph capture")
            self._stream.wait_stream(torch.cuda.current_stream(self._stream.device))

    def update_context_resources(self, scheduled_batch: ScheduledRequests) -> None:
        self._release_sparse_batch(
            request.py_request_id for request in scheduled_batch.context_requests
        )
        self._order_sparse_history_update(scheduled_batch.context_requests)
        super().update_context_resources(scheduled_batch)

    def try_allocate_generation(self, req: llm_request.LlmRequest) -> bool:
        """Reserve a decode request's next input token unless it is already reserved.

        With offload, history advances right after each forward, so at scheduling
        time a cache holding capacity beyond its history has already reserved this
        step. Attention-DP padding dummies are created that way. Growing such a
        cache again would exceed the sparse cache's one-token bound between
        capacity and history. No growth is recorded, so a skipped step reverts
        nothing.
        """
        if self._enable_kv_cache_offload:
            if self._sparse_batch is not None:
                self._sparse_batch.wait_published()
            cache = self.kv_cache_map.get(req.py_request_id)
            if cache is not None and cache.is_active and cache.capacity > cache.history_length:
                return True
        return super().try_allocate_generation(req)

    def revert_allocate_generation(self, req: llm_request.LlmRequest) -> None:
        if self._sparse_batch is not None:
            self._sparse_batch.wait_published()
        super().revert_allocate_generation(req)

    def update_forward_resources(
        self, scheduled_batch: ScheduledRequests, *, wait_for_input_copy: Callable[[], None]
    ) -> None:
        """Advance sparse decode history from executed inputs before the next schedule.

        Offload decode validates one input token per request, with speculation
        disabled. That input's KV is produced by this forward; the sampled
        output has no KV yet. Queue writeback after the forward on the cache
        stream, independently of when the executor consumes sampling results.
        """
        if not self._enable_kv_cache_offload or not scheduled_batch.generation_requests:
            return
        wait_for_input_copy()
        self._order_sparse_history_update(scheduled_batch.generation_requests)
        # Before any history moves: the write-back reads
        # the converted indices the Batch published, so it has to run while the members are
        # still seated. A row that is not advancing repeats its own history, which asks for
        # nothing.
        batch = self._sparse_batch
        if batch is not None:
            batch.wait_published()
            histories = [0] * batch.max_batch_size
            for member in self._sparse_batch_members.values():
                histories[member.row] = member.cache.history_length
            for request in scheduled_batch.generation_requests:
                member = self._sparse_batch_members.get(request.py_request_id)
                if member is not None and member.cache.is_active:
                    histories[member.row] = member.cache.history_length + 1
            batch.flush_sparse_units(histories)

        # The flush watermark belongs to the attached Batch row. Keep membership
        # until resize has checked it and advanced the request's history.
        for request in scheduled_batch.generation_requests:
            request_id = request.py_request_id
            cache = self.kv_cache_map.get(request_id)
            if cache is None or not cache.is_active:
                continue
            history_length = cache.history_length + 1
            if history_length > cache.capacity:
                raise ValueError(f"Sparse decode request {request_id} has no reserved input token")
            if not cache.resize(None, history_length):
                raise ValueError(
                    f"Failed to advance sparse KV history for request {request_id} "
                    f"to {history_length} tokens after model forward"
                )
            self._allocated_draft_lens.pop(request_id, None)
            # This step's allocation is settled here rather than in update_resources.
            inflight = self._inflight_draft_lens.get(request_id)
            if inflight:
                inflight.pop(0)
                if not inflight:
                    del self._inflight_draft_lens[request_id]
        self._sparse_batch_prepared = False

    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata=None,
        kv_cache_dtype_byte_size: float | None = None,
    ) -> None:
        if self._enable_kv_cache_offload:
            # Sparse decode has no speculative slots to rewind. History follows
            # forward completion; delayed sampling updates must neither rewind it
            # nor detach the Batch prepared for a later forward.
            return
        super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

    # ------------------------------------------------------------------
    # Arena descriptors
    # ------------------------------------------------------------------

    def _sparse_arena_layout(self) -> _SparseArenaLayout:
        """Re-derive KVCM's arena walk: configuration layer order, then declaration
        order within each pool. Host coalesced-buffer order is not used here."""
        pool_indices = {
            buffer_id: pool_index
            for group in self.impl.pool_group_descs
            for variant in group.slot_desc.variants
            for pool_index, coalesced in enumerate(variant.coalesced_buffers)
            for buffer_id in coalesced.buffer_ids
        }
        layer_configs = {
            layer.layer_id: layer
            for layer in self.impl.init_config.layers
            if isinstance(layer, AttentionLayerConfig)
        }
        sparse_pool_buffers: dict[tuple[LayerGroupId, int], list[BufferId]] = defaultdict(list)
        max_units_by_group: dict[LayerGroupId, int] = defaultdict(int)
        for layer_config in layer_configs.values():
            cache = layer_config.sparse_cache
            if cache is None:
                continue
            group_id = self.impl.get_layer_group_id(layer_config.layer_id)
            for buffer in layer_config.buffers:
                if buffer.is_sparse:
                    buffer_id = BufferId(layer_config.layer_id, buffer.role)
                    sparse_pool_buffers[group_id, pool_indices[buffer_id]].append(buffer_id)
                    max_units_by_group[group_id] = max(
                        max_units_by_group[group_id], cache.max_cached_units
                    )
        positions = {
            buffer_id: (len(buffers), position)
            for buffers in sparse_pool_buffers.values()
            for position, buffer_id in enumerate(buffers)
        }
        # The Batch owns the arenas and sizes them for its rows, so the lease count is
        # its capacity. A constraint describes a batch the manager must be able to hold,
        # not the arena's extent.
        max_leases = self.max_batch_size
        return _SparseArenaLayout(
            pool_indices=pool_indices,
            layer_configs=layer_configs,
            positions=positions,
            max_units_by_group=dict(max_units_by_group),
            max_leases=max_leases,
        )

    def _build_sparse_offload_descriptor(
        self,
        layout: _SparseArenaLayout,
        layer_id: LayerId,
        data_role: DataRole,
        *,
        unit_tokens: int,
        scale_role: DataRole | None = None,
        selection_owner: int | None = None,
        name: str,
    ) -> SparseOffloadLayerDescriptor:
        """Describe one layer's complete sparse fetch buffer set.

        KVCM orders buffers by pool index, retaining declaration order within
        each pool. Pool addresses and page-index scales describe host storage;
        they do not transform the physical arena indices returned by fetch.

        Args:
            layout: The manager's arena layout.
            layer_id: The KVCM layer holding the data buffer.
            data_role: The buffer attention reads, by arena row.
            unit_tokens: Original tokens one selection unit covers.
            scale_role: A block-scale buffer that must share the data's arena rows.
            selection_owner: The model layer that resolves this layer's selection,
                or None where the layer selects for itself. Must agree with the
                layer's configured ``selection_id``, since that is what decides
                whether the manager gives the layers one resident set.
            name: The model layer, for messages.
        """
        if not self.impl.is_sparse(layer_id, data_role):
            raise ValueError(f"{name}'s {data_role} buffer must be sparse")
        units_per_page = self.impl.get_sparse_cache_units_per_page(layer_id)
        if units_per_page <= 0 or self.tokens_per_block != units_per_page * unit_tokens:
            raise ValueError(
                f"{name}: a page of {self.tokens_per_block} tokens must hold whole "
                f"selection units of {unit_tokens} tokens"
            )
        # A layer may hold a dense buffer beside the one it fetches through, so take
        # only the sparse buffers: the dense ones are not fetched through the cache.
        layer_config = layout.layer_configs[layer_id]
        cache = layer_config.sparse_cache
        if (
            cache is None
            or cache.max_cached_units <= 0
            or cache.num_selected_blocks != self._sparse_offload_topk
        ):
            raise ValueError(
                "A sparse layer must configure the runtime's selection width and cache capacity"
            )
        if (cache.selection_id is None) != (selection_owner is None):
            raise ValueError(
                f"{name}: a layer shares a selection exactly when it names a selection_id; "
                f"got selection_id={cache.selection_id} and owner={selection_owner}"
            )
        buffers = [
            BufferId(layer_id, buffer.role) for buffer in layer_config.buffers if buffer.is_sparse
        ]
        data_id = BufferId(layer_id, data_role)
        if data_id not in buffers:
            raise ValueError(f"{name}'s sparse buffers must include {data_role}")
        buffers.sort(key=layout.pool_indices.__getitem__)
        scale_id = BufferId(layer_id, scale_role) if scale_role is not None else None
        if scale_id is not None:
            if scale_id not in buffers:
                raise ValueError(f"{name} requires a sparse block-scale buffer")
            data_layout = layout.positions[data_id]
            scale_layout = layout.positions[scale_id]
            if data_layout != scale_layout:
                raise ValueError(
                    f"{name}: the sparse data and scale arenas must have matching buffer "
                    "counts and positions, including padding, to share physical indices; "
                    f"got {data_layout} and {scale_layout}."
                )
        # A sparse buffer's size is a whole page; its arena region holds units.
        unit_bytes = {buffer.role: buffer.size // units_per_page for buffer in layer_config.buffers}
        group_id = self.impl.get_layer_group_id(layer_id)
        max_units = layout.max_units_by_group[group_id]
        reserved_units = (max_units + units_per_page - 1) // units_per_page * units_per_page
        arena_bases = tuple(
            self._sparse_arena_base(buffer.layer_id, buffer.role) for buffer in buffers
        )
        padding_regions = []
        for buffer, base in zip(buffers, arena_bases):
            if buffer == data_id or (scale_id is not None and buffer == scale_id):
                continue
            pool_count, position = layout.positions[buffer]
            buffer_bytes = reserved_units * unit_bytes[buffer.role]
            padding_regions.append(
                SparseOffloadPaddingRegion(
                    base_address=base + position * buffer_bytes,
                    bytes_per_lease=buffer_bytes,
                    lease_stride=pool_count * buffer_bytes,
                )
            )
        return SparseOffloadLayerDescriptor(
            buffer_ids=tuple(buffers),
            arena_base_addresses=arena_bases,
            arena_row_counts=tuple(
                layout.max_leases * layout.positions[buffer][0] * reserved_units
                for buffer in buffers
            ),
            data_buffer_index=buffers.index(data_id),
            scale_buffer_index=buffers.index(scale_id) if scale_id is not None else None,
            group_id=group_id,
            unit_tokens=unit_tokens,
            padding_regions=tuple(padding_regions),
            selection_owner=selection_owner,
        )

    def _initialize_sparse_arena_padding(
        self, layers: dict[int, SparseOffloadLayerDescriptor]
    ) -> None:
        """Initialize unused roles once, before requests can acquire arena leases.

        The arena's per-pool lease strides follow its configuration, independently
        of host page capacities. Views cover only padding buffers; real KV and
        scale buffers are initialized by their producers. KVCM owns these allocations.
        """
        regions = [region for layer in layers.values() for region in layer.padding_regions]
        if not regions:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Sparse arena padding must be initialized before graph capture")
        max_leases = self.max_batch_size
        with torch.cuda.device(self._stream.device), torch.cuda.stream(self._stream):
            for region in regions:
                convert_to_torch_tensor(
                    TensorWrapper(
                        region.base_address,
                        torch.uint8,
                        (max_leases, region.bytes_per_lease),
                        strides=(region.lease_stride, 1),
                    )
                ).zero_()
        # Construction completes before any producer or captured graph can use the arena.
        self._stream.synchronize()

    # ------------------------------------------------------------------
    # Per-step offload operations
    # ------------------------------------------------------------------

    @nvtx_range_debug("prepare_sparse_offload")
    def prepare_sparse_offload(
        self,
        state: SparseOffloadState,
        request_ids: list[int],
        *,
        beam_width: int,
    ) -> None:
        """Bind the published Batch and refresh the per-step mapping outside capture.

        Scheduling publishes the page tables and watermarks by Batch row. This maps
        each request in ``request_ids``, the attention metadata's input order, to its
        row, and sets the active count. The native Batch and its device views persist
        across steps and graph profiles; only trailing dummy requests may extend the
        scheduled requests, and they map to no row. This prepares fetch metadata, not
        write destinations.
        """
        state.prepared = False
        state.write_layer = None
        if not self._enable_kv_cache_offload or beam_width != 1:
            raise ValueError("Sparse offload preparation requires offload enabled and beam width 1")
        with torch.cuda.device(state.selected_history_units.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Sparse offload metadata must be prepared outside graph capture")
            batch = self.get_sparse_batch(request_ids)
            if batch.max_batch_size != state.selected_history_units.shape[0]:
                raise ValueError("Sparse fetch buffers must use the native Batch capacity")
            if state.batch is not None and state.batch is not batch:
                raise RuntimeError(
                    "Sparse Batch allocation must remain stable across graph replays"
                )
            torch.cuda.current_stream().wait_stream(self._stream)
            if state.batch is None:
                state.history_lengths = torch.from_dlpack(batch.history_lengths())
                state.batch = batch
            live = len(self._sparse_batch_members)
            rows = [self._sparse_batch_members[request_id].row for request_id in request_ids[:live]]
            rows += [-1] * (batch.max_batch_size - live)
            # Pinned memory returns to the caching host allocator only after the copy runs.
            state.batch_rows.copy_(
                torch.tensor(rows, dtype=torch.int32, pin_memory=prefer_pinned()),
                non_blocking=True,
            )
            state.active_request_count.fill_(live)
            state.selected_history_units.fill_(BAD_PAGE_INDEX)
            for out in state.fetched_indices.values():
                out.fill_(BAD_PAGE_INDEX)
            for out in state.write_indices.values():
                out.fill_(BAD_PAGE_INDEX)
        state.prepared = True

    def prepare_sparse_write_indices(
        self, state: SparseOffloadState, layer_idx: int
    ) -> torch.Tensor:
        """Reserve this layer's in-flight arena rows on the current stream.

        Return int32[B, F] in Batch row order, with columns assigned by
        ``Batch.sparse_out_slots`` and cached in ``state.out_slots[layer_idx]``.
        Buffers with matching arena indices share a column; unused columns are
        untouched. A used column gets -1 when its row has no unit in flight.
        A writer addressing requests by input position
        reads request i's entries at Batch row ``state.batch_rows[i]``, for example
        through ``gather_sparse_batch_rows``. The caller must join Batch publication
        before this call. Reservation must finish before either the writer or
        the history fetch; they may run concurrently because fetch excludes the
        protected in-flight row. Both must finish before attention or a history
        update. No KV is fetched here.

        Data and scale buffers use the same column when their arena indices coincide;
        the output-slot mapping follows the arena geometry and requires no device read.
        """
        if (
            not self._enable_kv_cache_offload
            or not state.prepared
            or state.is_prefill
            or state.batch is None
        ):
            raise RuntimeError("Sparse writes require prepared decode metadata")
        descriptor = state.layers[layer_idx]
        out = state.write_indices[len(descriptor.buffer_ids)]
        state.write_layer = None
        with torch.cuda.device(out.device):
            if layer_idx not in state.out_slots:
                state.out_slots[layer_idx] = tuple(
                    state.batch.sparse_out_slots(list(descriptor.buffer_ids))
                )
            state.batch.prepare_sparse_write_indices(
                buffers=descriptor.buffer_ids,
                out=out,
                stream=CudaStream(torch.cuda.current_stream().cuda_stream),
            )
        state.write_layer = layer_idx
        return out

    @nvtx_range_debug("fetch_sparse_indices")
    def fetch_sparse_indices(
        self,
        state: SparseOffloadState,
        layer_idx: int,
        topk_indices: torch.Tensor,
        request_indices: torch.Tensor,
        unit_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Fetch one layer's history into compact per-buffer arena indices.

        Inputs are CUDA int32: top-k unit ordinals [Q, K], each query's request
        by input position [Q], and selectable unit counts [R] by input position.
        Live queries name distinct requests, one query per request; the prepared
        ``state.batch_rows`` maps each to its Batch row. Q and R may be smaller
        than Batch.max_batch_size.

        Return int32[B, F, K] in Batch row order and descriptor.buffer_ids order.
        Each index is relative to that buffer's sparse arena base. Top-k slots
        keep their order; padding and in-flight selections produce -1. Attention
        must supply the in-flight write index separately, without treating this
        output as an ordinal-indexed page table or rescaling its indices.

        The caller joins selection producers and write reservation before this
        call, and joins the fetch before consuming its output on another stream.
        Output storage is reused by subsequent layers after attention consumes it.
        """
        if (
            not self._enable_kv_cache_offload
            or not state.prepared
            or state.is_prefill
            or state.batch is None
            or state.history_lengths is None
        ):
            raise RuntimeError("Sparse fetch requires prepared decode metadata")
        descriptor = state.layers[layer_idx]
        with torch.cuda.device(state.selected_history_units.device):
            select_sparse_history_units(
                topk_indices,
                request_indices,
                state.batch_rows,
                state.active_request_count,
                unit_lengths,
                state.history_lengths,
                state.selected_history_units,
                unit_tokens=descriptor.unit_tokens,
            )
            out = state.fetched_indices[len(descriptor.buffer_ids)]
            if layer_idx not in state.out_slots:
                state.out_slots[layer_idx] = tuple(
                    state.batch.sparse_out_slots(list(descriptor.buffer_ids))
                )
            # Fetch even for an empty selection so every output slot is refreshed
            # on graph replay. Native Batch supplies page tables and watermarks.
            stream = CudaStream(torch.cuda.current_stream().cuda_stream)
            owner = descriptor.selection_owner
            if owner is None:
                state.batch.fetch_sparse_pages(
                    buffers=descriptor.buffer_ids,
                    selected=state.selected_history_units,
                    out=out,
                    stream=stream,
                )
            else:
                # The layers an indexer selects for share one resident set, so the
                # selection is resolved once -- by the layer that owns the indexer --
                # and each layer then copies those units into its own cells. Resolving
                # per layer instead would have the later resolves read the first's
                # admissions as hits and leave their cells unfilled.
                if owner == layer_idx:
                    state.batch.resolve_sparse_selection(
                        buffers=descriptor.buffer_ids,
                        selected=state.selected_history_units,
                        stream=stream,
                    )
                state.batch.fetch_resolved_pages(
                    buffers=descriptor.buffer_ids,
                    selected=state.selected_history_units,
                    out=out,
                    stream=stream,
                )
        return out
