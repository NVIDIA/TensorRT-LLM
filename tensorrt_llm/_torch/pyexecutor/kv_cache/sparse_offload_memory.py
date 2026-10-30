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

"""Memory accounting for sparse attention KV cache offload.

With offload, the KV history a sparse attention layer selects from lives in
host-backed pages, and each request leases a GPU cache of selected units from a
sparse cell arena.

The native manager puts a sparse buffer's pages in a host-backed pool of the
same pool group as its life cycle's dense buffers. A slot spans every pool of
its group, so the host-backed pools get the slot count the device quota buys
for the dense ones, and nothing else bounds them. The host memory offload takes
is therefore set by the device quota and by how that quota is split across life
cycles. The arena is a separate device allocation outside every tier's quota.

Everything here is derived from the layer configs a cache manager declares, so
it is the same for every model. A model marks each offloaded buffer
``is_sparse``, gives its layer a fetch path from :func:`sparse_cache_config`,
and reports the device bytes of its other buffers through
``KVCacheManagerV2.get_request_device_bytes``.

The module has three parts: mirrors of the native layout, which must change
when ``cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2`` does; the sizing
policy; and the capacity report and host admission.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SparseCacheConfig

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        BufferConfig,
        KVCacheManagerConfig,
        LayerConfig,
    )

_GIB = float(1 << 30)
_MIB = float(1 << 20)

# ---------------------------------------------------------------------------
# Native layout: SparseCellArena and leaseLayout in sparseCache.cpp, the Batch in
# batch.cpp, pool and life-cycle formation in storage/config.cpp and
# lifeCycleRegistry.cpp, cacheTierGranularity in storageManager.cpp, and HostMem
# in utils/hostMem.cpp.
# ---------------------------------------------------------------------------

# SparseCellArena rounds every region to this many bytes.
SPARSE_ARENA_ALIGNMENT = 256
# Every arena and Batch index array holds 32-bit entries (int32_t or uint32_t).
_INDEX_BYTES = 4
# A Batch row holds two watermarks, history length and capacity.
_BATCH_WATERMARKS_PER_ROW = 2
# Pinned host memory is mapped in whole pages of this many bytes.
_HOST_PAGE_BYTES = 4096
# A decode step writes one query token per request, which completes at most one
# unit, so one unit per layer is in flight (_sparseCacheMaxInFlightUnits).
_SPARSE_MAX_FLUSH_UNITS = 1


def _index_array_bytes(num_entries: int) -> int:
    """Return the aligned bytes of an arena array of ``num_entries`` 32-bit entries."""
    return pad_up(_INDEX_BYTES * num_entries, SPARSE_ARENA_ALIGNMENT)


def sparse_units_per_page(unit_tokens: int | None, tokens_per_block: int) -> int:
    """Return the selection units one page holds.

    Args:
        unit_tokens: Tokens one unit covers, or None when a unit is a whole page.
        tokens_per_block: Tokens per page.
    """
    return tokens_per_block // (unit_tokens or tokens_per_block)


@dataclass(frozen=True, slots=True)
class SparseArenaShape:
    """GPU cell arena of one sparse life cycle.

    Attributes:
        num_layers: Layers with a sparse fetch path.
        num_selections: Selections those layers resolve between them, which is how
            many metadata rows a lease carries. One per layer unless layers name a
            shared ``selection_id``, in which case they share a row.
        cells_per_buffer: Cells each fetched sparse buffer gets per lease.
        units_per_cell: Selection units in one cell.
        cell_bytes: Bytes of one cell of each fetched sparse buffer, in layer order.
    """

    num_layers: int
    num_selections: int
    cells_per_buffer: int
    units_per_cell: int
    cell_bytes: tuple[int, ...]

    @property
    def entries(self) -> int:
        """Cache entries of one lease: one per cached unit of each selection.

        Per selection, not per layer: layers sharing a selection hold the same units
        in mirrored cells, so one resident set serves them all.
        """
        return self.num_selections * self.cells_per_buffer * self.units_per_cell

    @property
    def lease_metadata_bytes(self) -> int:
        """Bytes of one lease's residency and eviction state.

        One word per entry: the unit it holds and when that unit was last used.

        The reservation only. A step's work lists belong to the Batch, not to a lease:
        they are sized by the batch and reused by whichever requests are seated.
        """
        return _index_array_bytes(self.entries)

    @property
    def lease_bytes(self) -> int:
        """GPU bytes one lease adds to the arena."""
        return self.cells_per_buffer * sum(self.cell_bytes) + self.lease_metadata_bytes

    def arena_bytes(self, max_leases: int) -> int:
        """Return the GPU bytes of an arena sized for ``max_leases`` leases."""
        # One region per pool. Buffers of one cell size share a pool, so a region
        # holds every lease's cells of every buffer of that size.
        buffers_by_cell_bytes: dict[int, int] = defaultdict(int)
        for cell_bytes in self.cell_bytes:
            buffers_by_cell_bytes[cell_bytes] += 1
        cell_regions = sum(
            pad_up(
                max_leases * num_buffers * self.cells_per_buffer * cell_bytes,
                SPARSE_ARENA_ALIGNMENT,
            )
            for cell_bytes, num_buffers in buffers_by_cell_bytes.items()
        )
        num_buffers = len(self.cell_bytes)
        shared_arrays = (
            num_buffers,  # layer of each buffer
            num_buffers,  # slot of each buffer in its pool
            self.num_layers,  # metadata row each layer resolves within
        )
        shared = sum(_index_array_bytes(n) for n in shared_arrays)
        return cell_regions + shared + max_leases * self.lease_metadata_bytes


def _fetched_sparse_buffers(layer: LayerConfig) -> list[BufferConfig]:
    """Return the sparse buffers of ``layer`` that are read through a sparse cache."""
    if getattr(layer, "sparse_cache", None) is None:
        return []
    return [buffer for buffer in layer.buffers if buffer.is_sparse]


def _fetch_path_life_cycles(
    layers: Sequence[LayerConfig], tokens_per_block: int
) -> list[list[AttentionLayerConfig]]:
    """Return the layers with a sparse fetch path, grouped by life cycle."""
    # Layers share a life cycle when their window and sink pages match.
    groups: dict[tuple[int | None, int], list[AttentionLayerConfig]] = defaultdict(list)
    for layer in layers:
        if _fetched_sparse_buffers(layer):
            sink_blocks = ceil_div(layer.num_sink_tokens or 0, tokens_per_block)
            groups[layer.sliding_window_size, sink_blocks].append(layer)
    return list(groups.values())


def sparse_arena_shapes(
    layers: Sequence[LayerConfig], *, tokens_per_block: int
) -> list[SparseArenaShape]:
    """Return the arena of every life cycle whose layers have a sparse fetch path.

    Args:
        layers: Every layer config the manager allocates.
        tokens_per_block: Tokens per page.

    Raises:
        ValueError: Layers sharing a life cycle declare different
            ``sparse_cache`` values. They share one arena, so the native manager
            rejects them too.
    """
    shapes = []
    for grouped_layers in _fetch_path_life_cycles(layers, tokens_per_block):
        geometries = {
            (
                layer.sparse_cache.num_selected_blocks,
                layer.sparse_cache.max_cached_units,
                sparse_units_per_page(layer.sparse_cache.unit_tokens, tokens_per_block),
            )
            for layer in grouped_layers
        }
        if len(geometries) > 1:
            raise ValueError(
                "Layers sharing a life cycle must declare the same sparse_cache; they share "
                f"one arena. Got (selected, cached, units per page) {sorted(geometries)}."
            )
        ((_, max_cached_units, units_per_page),) = geometries
        # An unset selection_id reads as the layer's own, so a layer that names
        # nothing keeps a row to itself and only an explicit id merges two.
        selections = {
            layer.sparse_cache.selection_id
            if layer.sparse_cache.selection_id is not None
            else layer.layer_id
            for layer in grouped_layers
        }
        shapes.append(
            SparseArenaShape(
                num_layers=len(grouped_layers),
                num_selections=len(selections),
                cells_per_buffer=ceil_div(max_cached_units, units_per_page),
                units_per_cell=units_per_page,
                # A cell holds one page of one buffer, and a buffer's size is one page.
                cell_bytes=tuple(
                    buffer.size
                    for layer in grouped_layers
                    for buffer in _fetched_sparse_buffers(layer)
                ),
            )
        )
    return shapes


def sparse_arena_bytes(config: KVCacheManagerConfig, max_leases: int) -> int:
    """Return the GPU bytes of every sparse arena the Batch allocates.

    ``max_leases`` is the Batch's row count, which is what the arenas are sized
    for: a lease is taken when a cache joins and returned when it leaves.

    The layers alone decide whether there is an arena at all, so nothing else is
    read until they say there is one: a config with no sparse fetch path allocates
    nothing, and need not describe its block size.
    """
    if not any(
        getattr(layer, "sparse_cache", None) is not None
        and any(buffer.is_sparse for buffer in layer.buffers)
        for layer in config.layers
    ):
        return 0
    shapes = sparse_arena_shapes(config.layers, tokens_per_block=config.tokens_per_block)
    if not shapes or max_leases <= 0:
        return 0
    return sum(shape.arena_bytes(max_leases) for shape in shapes)


def sparse_history_bytes_per_block(layers: Sequence[LayerConfig]) -> int:
    """Return the host bytes one request's offloaded history takes per page of tokens.

    Every sparse buffer keeps its history in host-backed pages, and a buffer's
    size is one page, so this is the summed size of the sparse buffers. A
    windowed sparse layer is counted as if it kept its whole history.
    """
    return sum(buffer.size for layer in layers for buffer in layer.buffers if buffer.is_sparse)


@dataclass(frozen=True)
class LifeCyclePage:
    """What one page of a life cycle costs, split by where the native manager puts it.

    Attributes:
        window_size: Sliding window in tokens, or None when the life cycle keeps
            all of its history.
        num_sink_tokens: Tokens kept from the start of the sequence beside the
            window.
        is_state: Whether the life cycle holds one slot of state per request,
            as SSM layers do, rather than pages of tokens.
        device_bytes: Bytes of its dense buffers, which the device quota pays for.
        host_bytes: Bytes of its sparse buffers, which live in pinned host memory.
        device_pool_bytes: Slot bytes of each device-backed pool, largest first.
            The native manager gives a life cycle's dense buffers of one page
            size a pool of their own, so these sum to ``device_bytes``.
    """

    window_size: int | None
    num_sink_tokens: int
    is_state: bool
    device_bytes: int
    host_bytes: int
    device_pool_bytes: tuple[int, ...]

    def pages_per_request(
        self, *, capacity: int, capacity_headroom: int, tokens_per_block: int
    ) -> int:
        """Return the most pages one decoding request holds.

        Args:
            capacity: Tokens the request's cache holds, its history included.
            capacity_headroom: Tokens a decode step reserves above the history.
            tokens_per_block: Tokens per page.
        """
        if self.is_state:
            return 1
        pages = ceil_div(capacity, tokens_per_block)
        if self.window_size is None:
            return pages
        # The live interval spans window + headroom - 1 tokens, at any page offset.
        window_pages = ceil_div(self.window_size + capacity_headroom - 2, tokens_per_block) + 1
        sink_pages = ceil_div(self.num_sink_tokens, tokens_per_block)
        return min(pages, window_pages + sink_pages)


def life_cycle_pages(layers: Sequence[LayerConfig], tokens_per_block: int) -> list[LifeCyclePage]:
    """Return the life cycles ``layers`` form, in the order the native manager numbers them.

    The native manager numbers life cycles by their first layer, which is also
    the order of ``initial_pool_ratio``. Attention layers share a life cycle
    when their window and sink pages match; all SSM layers share one.

    Raises:
        ValueError: A life cycle has sparse buffers but no dense one. The native
            manager sizes a pool group's slots from its device-backed pools, so it
            rejects such a life cycle too.
    """
    keys: list[tuple] = []
    device_pages: dict[tuple, Counter[int]] = {}
    host_bytes: dict[tuple, int] = {}
    pages: dict[tuple, tuple[int | None, int, bool]] = {}
    for layer in layers:
        if isinstance(layer, AttentionLayerConfig):
            window = layer.sliding_window_size
            sink_tokens = layer.num_sink_tokens or 0
            key = ("attention", window, ceil_div(sink_tokens, tokens_per_block))
            shape = (window, sink_tokens, False)
        else:
            key = ("state",)
            shape = (None, 0, True)
        if key not in pages:
            keys.append(key)
            pages[key] = shape
            device_pages[key] = Counter()
            host_bytes[key] = 0
        for buffer in layer.buffers:
            if buffer.is_sparse:
                host_bytes[key] += buffer.size
            elif buffer.size:
                override = buffer.tokens_per_block_override or tokens_per_block
                device_pages[key][buffer.size * (tokens_per_block // override)] += 1
    result = []
    for key in keys:
        device_pool_bytes = tuple(
            sorted((size * count for size, count in device_pages[key].items()), reverse=True)
        )
        if host_bytes[key] and not device_pool_bytes:
            raise ValueError(
                "A life cycle with sparse buffers must also hold a dense buffer: the native "
                "KV cache manager sizes a pool group's slots from its device-backed pools."
            )
        window, sink_tokens, is_state = pages[key]
        result.append(
            LifeCyclePage(
                window_size=window,
                num_sink_tokens=sink_tokens,
                is_state=is_state,
                device_bytes=sum(device_pool_bytes),
                host_bytes=host_bytes[key],
                device_pool_bytes=device_pool_bytes,
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class SparseBatchShape:
    """The native Batch the sparse fetch reads, sized once for every step.

    It holds a page table per life cycle, and two watermarks and one arena slot per row,
    in a device mirror, a pinned host staging copy and a device staging copy, and the
    fetch's work arrays on the device. Every step uploads the host staging copy to the
    device one, then copies the mirror's part of it across.

    Attributes:
        num_life_cycles: Life cycles of the manager; each has a page table.
        max_batch_size: Rows.
        max_blocks_per_seq: Columns of each page table.
        selection_widths: Units one step selects, for each life cycle with a sparse
            fetch path.
        layer_counts: Layers of each of those life cycles. The write-back keeps a work
            list per layer, so this scales that term; a zero would drop it and under-size
            the allocation, which is why it is required rather than defaulted.
        flush_capacities: Units one step can write back per layer, for each of those life
            cycles, from the step's token budget.
    """

    num_life_cycles: int
    max_batch_size: int
    max_blocks_per_seq: int
    selection_widths: tuple[int, ...]
    layer_counts: tuple[int, ...]
    flush_capacities: tuple[int, ...]

    @property
    def block_bytes(self) -> int:
        """Device bytes of the staged block: the page tables, watermarks and leases."""
        entries = self.max_batch_size * (
            self.num_life_cycles * self.max_blocks_per_seq + _BATCH_WATERMARKS_PER_ROW + 1
        )
        return _INDEX_BYTES * entries

    @property
    def mirror_bytes(self) -> int:
        """Device bytes of the mirror: the block's tail, which a captured kernel reads.

        The page tables leading the block are read from its device staging copy by the
        index conversion, so they are not carried on.
        """
        return _INDEX_BYTES * self.max_batch_size * (_BATCH_WATERMARKS_PER_ROW + 1)

    @property
    def staging_bytes(self) -> int:
        """Pinned host bytes of the staging copy of the block."""
        return pad_up(self.block_bytes, _HOST_PAGE_BYTES)

    @property
    def device_bytes(self) -> int:
        """Device bytes of the Batch: the staged block, the mirror of its tail and the
        fetch's work arrays."""
        return self.block_bytes + self.mirror_bytes + self.fetch_scratch_bytes

    @property
    def fetch_scratch_bytes(self) -> int:
        """Device bytes of a step's work arrays, for the fetch and the write-back both.

        Per life cycle and row: one entry per selected unit, a miss count and the unit
        in flight. Then the two compacted work lists, which carry a row, an ordinal and
        a slot per entry plus one total per list: one list for the fetch, sized for every
        row missing its whole selection, and one per layer for the write-back, sized by
        the step's token budget.
        """
        rows = self.max_batch_size
        total = 0
        for width, num_layers, flush_capacity in zip(
            self.selection_widths, self.layer_counts, self.flush_capacities, strict=True
        ):
            total += rows * (width + 1 + _SPARSE_MAX_FLUSH_UNITS)
            total += 3 * rows * width + 1
            total += num_layers * (3 * flush_capacity + 1)
            total += rows * num_layers
        return _INDEX_BYTES * total


def sparse_batch_shape(
    layers: Sequence[LayerConfig],
    *,
    tokens_per_block: int,
    max_batch_size: int,
    max_blocks_per_seq: int,
    max_total_input_tokens: int | None = None,
) -> SparseBatchShape:
    """Return the Batch a manager with ``layers`` steps its requests in.

    Args:
        layers: Every layer config the manager allocates.
        tokens_per_block: Tokens per page.
        max_batch_size: Rows of the Batch.
        max_blocks_per_seq: Pages the longest request can hold.
        max_total_input_tokens: Tokens one step may ingest across every row. It sizes
            the write-back's work lists: a row advancing ``t`` tokens crosses at most
            ``t / unit_tokens + 1`` units. Left unset, a decode step is assumed, where
            each row advances one token and so crosses at most one unit boundary.
    """
    fetch_groups = list(_fetch_path_life_cycles(layers, tokens_per_block))

    def flush_capacity(grouped_layers: Sequence[LayerConfig]) -> int:
        if max_total_input_tokens is None:
            return max_batch_size * _SPARSE_MAX_FLUSH_UNITS
        unit_tokens = grouped_layers[0].sparse_cache.unit_tokens or tokens_per_block
        return max_total_input_tokens // unit_tokens + max_batch_size

    return SparseBatchShape(
        num_life_cycles=len(life_cycle_pages(layers, tokens_per_block)),
        max_batch_size=max_batch_size,
        max_blocks_per_seq=max_blocks_per_seq,
        selection_widths=tuple(
            grouped_layers[0].sparse_cache.num_selected_blocks for grouped_layers in fetch_groups
        ),
        layer_counts=tuple(len(grouped_layers) for grouped_layers in fetch_groups),
        flush_capacities=tuple(flush_capacity(grouped_layers) for grouped_layers in fetch_groups),
    )


# ---------------------------------------------------------------------------
# Sizing policy: the GPU cache per request, the quota split, the host budget and
# the device quota it bounds.
# ---------------------------------------------------------------------------


def sparse_cache_cells(cache_factor: float, num_selected_units: int, units_per_cell: int) -> int:
    """Return the cells one sparse buffer gets in each request's device cache.

    The cache holds ``cache_factor`` selections plus the unit being written,
    rounded up to whole cells. A request has at most one unit in flight, so that
    unit adds one whatever the factor. A factor of 1 is the smallest cache the
    native manager accepts.

    Args:
        cache_factor: Selections the cache holds; at least 1.
        num_selected_units: Units one step selects.
        units_per_cell: Units in one cell, i.e. in one page.
    """
    if cache_factor < 1:
        raise ValueError(f"cache_factor must be at least 1, got {cache_factor}")
    if num_selected_units <= 0 or units_per_cell <= 0:
        raise ValueError(
            "num_selected_units and units_per_cell must be positive, got "
            f"{num_selected_units} and {units_per_cell}"
        )
    # Round before taking the ceiling so float error cannot add a unit.
    selected_units = math.ceil(round(cache_factor * num_selected_units, 6))
    return ceil_div(selected_units + 1, units_per_cell)


def sparse_cache_config(
    *,
    num_selected_units: int,
    unit_tokens: int | None,
    tokens_per_block: int,
    gpu_cache_factor: float,
    selection_id: int | None = None,
) -> SparseCacheConfig:
    """Return a layer's sparse fetch path whose cache holds ``gpu_cache_factor`` selections.

    A selection unit is what one selection names: a token, a compressed row or
    pool of tokens, or a whole page. Each request caches selected units in GPU
    cells of one page each. The cache holds ``gpu_cache_factor`` selections plus
    the unit being written, rounded up to whole cells.

    Args:
        num_selected_units: Units one decode step selects for one request. When
            heads select separately, this is the size of their union.
        unit_tokens: Tokens one unit covers, or None when a unit is a whole page.
            Must divide ``tokens_per_block``.
        tokens_per_block: Tokens per page.
        gpu_cache_factor: Selections the cache holds; at least 1.
        selection_id: Names the selection this layer resolves, where several layers
            are handed one. Layers naming the same id share a resident set and
            resolve once between them, so the id must be shared by exactly the
            layers an indexer selects for, and those layers must agree on every
            other argument here. None, the default, means the layer selects for
            itself; a layer that names nothing never shares.
    """
    if unit_tokens is not None and (unit_tokens <= 0 or tokens_per_block % unit_tokens):
        raise ValueError(
            f"unit_tokens ({unit_tokens}) must divide tokens_per_block ({tokens_per_block})"
        )
    units_per_page = sparse_units_per_page(unit_tokens, tokens_per_block)
    cells = sparse_cache_cells(gpu_cache_factor, num_selected_units, units_per_page)
    return SparseCacheConfig(
        num_selected_blocks=num_selected_units,
        max_cached_units=cells * units_per_page,
        # Resolved rather than passed through: the config names a unit size, and a whole
        # page is one of them. None is this helper's convenience, not the config's.
        unit_tokens=tokens_per_block // units_per_page,
        selection_id=selection_id,
    )


def sparse_offload_pool_ratio(
    life_cycles: Sequence[LifeCyclePage],
    *,
    capacity: int,
    capacity_headroom: int,
    tokens_per_block: int,
) -> list[float]:
    """Return the device quota's split across life cycles for decoding requests.

    Each life cycle's share is the device bytes its pages take in a request of
    ``capacity`` tokens, so the quota buys the same number of such requests in
    every life cycle. Host-backed pools cost the quota nothing and get no share.

    Args:
        life_cycles: The manager's life cycles, in native order.
        capacity: Tokens one request's cache holds.
        capacity_headroom: Tokens a decode step reserves above the history.
        tokens_per_block: Tokens per page.

    Returns:
        One positive share per life cycle, summing to 1; the order and meaning of
        ``KVCacheManagerConfig.initial_pool_ratio``.
    """
    # Device and host bytes together. The manager's pool ratio counts every pool, host-backed
    # ones included -- it is projected onto cold levels, where one pool holds dense and sparse
    # buffers together, and a rebalance compares it against a layout measured the same way.
    # Weighing device bytes alone here would hand the manager a ratio in one currency and
    # measure the result in another.
    weights = [
        (life_cycle.device_bytes + life_cycle.host_bytes)
        * life_cycle.pages_per_request(
            capacity=capacity,
            capacity_headroom=capacity_headroom,
            tokens_per_block=tokens_per_block,
        )
        for life_cycle in life_cycles
    ]
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"Every life cycle needs bytes to share the quota, got {weights}")
    total = sum(weights)
    return [weight / total for weight in weights]


def plan_host_history_bytes(
    *,
    host_history_size: int | None,
    max_batch_size: int,
    host_bytes_per_request: int,
    pinnable_host_bytes: float,
    staging_bytes: int,
) -> int:
    """Return the host bytes the host-backed history pools may take.

    Offload also pins the sparse Batch's staging buffers, so they come out of
    the same host memory. An explicit ``host_history_size`` covers both.
    Otherwise the pools hold ``max_batch_size`` requests, capped so that pools
    and staging fit in the host memory the rank may pin.

    Args:
        host_history_size: The configured size, or None to size automatically.
        max_batch_size: Requests the automatic budget holds.
        host_bytes_per_request: Host history bytes one request holds.
        pinnable_host_bytes: Host bytes automatic sizing may pin for offload;
            ``inf`` when unknown.
        staging_bytes: Pinned host bytes of the sparse Batch's staging buffers.

    Raises:
        ValueError: No host memory is left for the pools.
    """
    if host_history_size is not None:
        budget = host_history_size - staging_bytes
        if budget <= 0:
            raise ValueError(
                "sparse_attention_config.kv_cache_offload_config.host_history_size "
                f"({host_history_size / _MIB:.1f}MiB) leaves no host memory for offloaded "
                f"history: the sparse Batch's staging buffers take {staging_bytes / _MIB:.1f}MiB "
                "of it."
            )
        return budget
    if not host_bytes_per_request:
        # This rank's layers offload no history.
        return 0
    budget = min(max_batch_size * host_bytes_per_request, pinnable_host_bytes - staging_bytes)
    if budget <= 0:
        raise ValueError(
            "Sizing sparse KV cache offload automatically leaves no host memory for offloaded "
            f"history: this rank may pin {max(0.0, pinnable_host_bytes) / _MIB:.1f}MiB, and the "
            f"sparse Batch's staging buffers take {staging_bytes / _MIB:.1f}MiB. Set "
            "sparse_attention_config.kv_cache_offload_config.host_history_size, or lower "
            "kv_cache_config.host_cache_size."
        )
    return int(budget)


# ---------------------------------------------------------------------------
# Capacity report and host admission.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseOffloadCapacity:
    """How many requests of one length a worker with sparse KV cache offload holds.

    Attributes:
        num_tokens: Tokens per request.
        max_batch_size: The configured batch limit.
        gpu_pool_bytes: Device bytes of the KV pools.
        host_pool_bytes: Host bytes of the host-backed pools the manager allocated.
        host_budget_bytes: Host bytes offloaded history may take.
        gpu_bytes_per_request: Device pool bytes one request holds.
        fixed_gpu_bytes_per_request: The part of ``gpu_bytes_per_request`` that
            does not grow with length, such as sliding windows.
        host_bytes_per_request: Host history bytes one request holds.
        arena_bytes: Device bytes of the sparse cache arenas.
        num_leases: Leases each arena holds.
        lease_bytes: Device bytes of one lease, summed over the arenas.
        batch_buffer_bytes: Device bytes of the buffers sized for
            ``max_batch_size`` rows: the Batch and the model's offload scratch.
        by_limit: Requests that fit by each limit: ``max_batch_size``, ``host``
            and ``gpu``, in the order that breaks ties.
        recommended_max_batch_size: The largest batch the device budget and the
            host budget hold.
    """

    num_tokens: int
    max_batch_size: int
    gpu_pool_bytes: int
    host_pool_bytes: int
    host_budget_bytes: int
    gpu_bytes_per_request: int
    fixed_gpu_bytes_per_request: int
    host_bytes_per_request: int
    arena_bytes: int
    num_leases: int
    lease_bytes: int
    batch_buffer_bytes: int
    by_limit: dict[str, int] = field(compare=False)
    recommended_max_batch_size: int

    @property
    def limited_by(self) -> str:
        """The limit that admits the fewest requests."""
        return min(self.by_limit, key=self.by_limit.__getitem__)

    @property
    def fits(self) -> int:
        """Requests that fit."""
        return self.by_limit[self.limited_by]

    def describe(self) -> str:
        """Return a one-line report for the log."""
        return (
            f"Sparse KV cache offload capacity at {self.num_tokens} tokens per request: "
            f"GPU pools {self.gpu_pool_bytes / _GIB:.2f}GiB at "
            f"{self.gpu_bytes_per_request / _MIB:.1f}MiB per request "
            f"({self.fixed_gpu_bytes_per_request / _MIB:.1f}MiB fixed); arena "
            f"{self.arena_bytes / _GIB:.2f}GiB for {self.num_leases} leases of "
            f"{self.lease_bytes / _MIB:.2f}MiB; offload buffers "
            f"{self.batch_buffer_bytes / _MIB:.1f}MiB for {self.max_batch_size} rows; "
            f"host-backed history pools "
            f"{self.host_pool_bytes / _GIB:.2f}GiB at "
            f"{self.host_bytes_per_request / _MIB:.1f}MiB per request, host budget "
            f"{self.host_budget_bytes / _GIB:.2f}GiB. Requests: {self.by_limit['gpu']} by "
            f"GPU, {self.by_limit['host']} by host, max_batch_size {self.max_batch_size}; "
            f"{self.fits} fit, limited by {self.limited_by}. A max_batch_size of "
            f"{self.recommended_max_batch_size} fits the GPU budget and the host budget."
        )


def solve_sparse_offload_capacity(
    *,
    num_tokens: int,
    max_batch_size: int,
    gpu_pool_bytes: int,
    host_pool_bytes: int,
    host_budget_bytes: int,
    gpu_bytes_per_request: int,
    fixed_gpu_bytes_per_request: int,
    host_bytes_per_request: int,
    arenas: Sequence[SparseArenaShape],
    num_leases: int,
    batch_buffer_bytes: int,
) -> SparseOffloadCapacity:
    """Return how many requests of ``num_tokens`` tokens fit, and the batch size that fits.

    Args:
        num_tokens: Tokens per request.
        max_batch_size: The configured batch limit, the rows of the Batch.
        gpu_pool_bytes: Device bytes of the KV pools.
        host_pool_bytes: Host bytes of the host-backed pools the manager allocated.
        host_budget_bytes: Host bytes offloaded history may take.
        gpu_bytes_per_request: Device pool bytes one request holds.
        fixed_gpu_bytes_per_request: The part of ``gpu_bytes_per_request`` that
            does not grow with length.
        host_bytes_per_request: Host history bytes one request holds.
        arenas: The arena of each sparse life cycle.
        num_leases: Leases each arena holds. A cache holds a lease only while it
            is seated in the Batch, so this is ``max_batch_size``; any beyond it
            are padding.
        batch_buffer_bytes: Device bytes of the buffers sized for
            ``max_batch_size`` rows, reserved apart from ``gpu_pool_bytes``.
    """
    gpu_bytes_per_request = max(1, gpu_bytes_per_request)
    host_bytes_per_request = max(1, host_bytes_per_request)
    arena_bytes = sum(arena.arena_bytes(num_leases) for arena in arenas)
    lease_bytes = sum(arena.lease_bytes for arena in arenas)
    by_limit = {
        "max_batch_size": max_batch_size,
        "host": host_pool_bytes // host_bytes_per_request,
        "gpu": gpu_pool_bytes // gpu_bytes_per_request,
    }
    # The device budget pays for the pools, the arena and the buffers sized for
    # max_batch_size rows. The arena's shared tables and padding leases are
    # fixed; every batch slot adds a request's pages, the one lease its seated
    # cache holds and its rows of those buffers.
    padding_leases = max(0, num_leases - max_batch_size)
    fixed_arena_bytes = sum(arena.arena_bytes(padding_leases) for arena in arenas)
    batch_bytes_per_slot = ceil_div(batch_buffer_bytes, max(1, max_batch_size))
    recommended = min(
        max(0, gpu_pool_bytes + arena_bytes + batch_buffer_bytes - fixed_arena_bytes)
        // (gpu_bytes_per_request + lease_bytes + batch_bytes_per_slot),
        host_budget_bytes // host_bytes_per_request,
    )
    return SparseOffloadCapacity(
        num_tokens=num_tokens,
        max_batch_size=max_batch_size,
        gpu_pool_bytes=gpu_pool_bytes,
        host_pool_bytes=host_pool_bytes,
        host_budget_bytes=host_budget_bytes,
        gpu_bytes_per_request=gpu_bytes_per_request,
        fixed_gpu_bytes_per_request=fixed_gpu_bytes_per_request,
        host_bytes_per_request=host_bytes_per_request,
        arena_bytes=arena_bytes,
        num_leases=num_leases,
        lease_bytes=lease_bytes,
        batch_buffer_bytes=batch_buffer_bytes,
        by_limit=by_limit,
        recommended_max_batch_size=recommended,
    )


class HostHistoryReservations:
    """Host history pages reserved for admitted requests through their last token.

    A request whose history outgrows the host-backed pool while it decodes has
    nowhere to go, so admission reserves every page a request can reach.
    """

    def __init__(self, capacity_blocks: int) -> None:
        """Track reservations against a pool of ``capacity_blocks`` pages."""
        self.capacity_blocks = capacity_blocks
        self.reserved_blocks = 0
        self._blocks_by_request: dict[int, int] = {}

    def can_reserve(self, request_id: int, num_blocks: int) -> bool:
        """Return whether ``num_blocks`` pages fit for ``request_id``, replacing its reservation."""
        held = self._blocks_by_request.get(request_id, 0)
        return self.reserved_blocks - held + num_blocks <= self.capacity_blocks

    def reserve(self, request_id: int, num_blocks: int) -> None:
        """Reserve ``num_blocks`` pages for ``request_id``, replacing its reservation."""
        held = self._blocks_by_request.get(request_id, 0)
        self._blocks_by_request[request_id] = num_blocks
        self.reserved_blocks += num_blocks - held

    def release(self, request_id: int) -> None:
        """Release the pages reserved for ``request_id``, if any."""
        self.reserved_blocks -= self._blocks_by_request.pop(request_id, 0)
