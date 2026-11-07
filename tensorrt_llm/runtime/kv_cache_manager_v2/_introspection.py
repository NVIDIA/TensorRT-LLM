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

"""Internal KVCacheManagerV2 surface: no stability promise, may change with the C++ side.

Everything here forwards to the native ``_introspection`` submodule of the bindings. The
indirection exists so callers import one Python path rather than reaching into
``tensorrt_llm.bindings`` directly, and so return values are normalised to plain Python
containers. Most hooks are white-box helpers for tests and accuracy harnesses, but a few
back production internals -- the stats report and the disaggregated bounce buffer -- so
treat this as internal API rather than test-only scaffolding.
"""

from __future__ import annotations

import sys
from typing import Any, NamedTuple


def _cpp_introspection_module() -> Any | None:
    package = sys.modules.get(__package__)
    if package is None:
        return None
    return getattr(package, "_cpp_introspection", None)


def _cpp() -> Any:
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is None:
        raise RuntimeError(
            "KVCacheManagerV2 introspection requires the native _introspection module, "
            "which is missing from this build of tensorrt_llm.bindings"
        )
    return cpp_introspection


#: CUDA virtual-memory primitives, forwarded rather than wrapped: they are classes the
#: caller constructs, so there is nothing to normalise. Used by the native disaggregated
#: bounce buffer to reserve one contiguous fabric region.
_FORWARDED_TYPES = ("PooledPhysMemAllocator", "VirtMem")


def __getattr__(name: str) -> Any:
    # Resolved on first access, not at import, so importing this module never depends on
    # the native submodule being present -- the same contract the hooks below follow.
    if name in _FORWARDED_TYPES:
        return getattr(_cpp(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def poison_for_testing(reason: str) -> None:
    """Set the poison latch directly, to exercise the refusal paths."""
    _cpp().poison_for_testing(reason)


def create_test_padding_cold_page_codec(cold_page_bytes_by_layer: dict[int, int]) -> Any:
    """Create the private native padding codec used by cold-tier end-to-end tests."""
    return _cpp().create_test_padding_cold_page_codec(cold_page_bytes_by_layer)


def drop_optional_pages(kv_cache: Any, end: int, groups: list[int] | None = None) -> None:
    """Test-only eviction injection, preserving other groups and live readers."""
    _cpp().drop_optional_pages(kv_cache, end, groups)


def make_test_block(
    manager: Any,
    tokens: Any,
    coverage_per_lc: list[int],
    parent: Any = None,
    reuse_scope: Any = None,
) -> Any:
    """Build a real block with real GPU pages at the requested lifecycle coverage."""
    return _cpp().make_test_block(manager, list(tokens), coverage_per_lc, parent, reuse_scope)


def close_test_block(block: Any) -> None:
    """Unlink and release a test block pages before its manager shuts down."""
    block.close()


def reattach_test_block(
    manager: Any, block: Any, parent: Any = None, reuse_scope: Any = None
) -> None:
    """Reattach an orphan through the native deduplication and page-adoption path."""
    _cpp().reattach_test_block(manager, block, parent, reuse_scope)


def set_test_block_page_cache_level(block: Any, life_cycle_id: int, cache_level: int) -> None:
    """Set synthetic page residency for attribution tests."""
    _cpp().set_test_block_page_cache_level(block, life_cycle_id, cache_level)


def test_block_key(block: Any) -> bytes:
    """Return a test block real radix-tree key."""
    return bytes(_cpp().test_block_key(block))


def event_manager_add_stored_block(event_manager: Any, block: Any) -> None:
    """Derive and enqueue stored events from a real test block."""
    _cpp().event_manager_add_stored_block(event_manager, block)


def event_manager_add_stored_life_cycle(event_manager: Any, block: Any, life_cycle_id: int) -> None:
    """Derive and enqueue one lifecycle stored event from a real test block."""
    _cpp().event_manager_add_stored_life_cycle(event_manager, block, life_cycle_id)


def streaming_event_sink_add_stored_block(event_sink: Any, block: Any) -> None:
    """Feed a real test block to the native streaming sink."""
    _cpp().streaming_event_sink_add_stored_block(event_sink, block)


def streaming_event_sink_add_stored_life_cycle(
    event_sink: Any, block: Any, life_cycle_id: int
) -> None:
    """Feed one lifecycle of a real test block to the native streaming sink."""
    _cpp().streaming_event_sink_add_stored_life_cycle(event_sink, block, life_cycle_id)


def streaming_event_sink_add_removed_block(event_sink: Any, block: Any) -> None:
    """Remove a real test block from the native streaming sink."""
    _cpp().streaming_event_sink_add_removed_block(event_sink, block)


def streaming_event_sink_add_removed_life_cycle(
    event_sink: Any, block: Any, life_cycle_id: int
) -> None:
    """Remove one lifecycle of a real test block from the native streaming sink."""
    _cpp().streaming_event_sink_add_removed_life_cycle(event_sink, block, life_cycle_id)


def active_page_stats(kv_cache: Any) -> tuple[list[int], list[int]]:
    """Return active pages and unscheduled evictable active pages by cache level."""
    counts, unscheduled_evictable = _cpp().active_page_stats(kv_cache)
    return list(counts), list(unscheduled_evictable)


def committed_page_is_linked(kv_cache: Any, ordinal: int, lc_id: int) -> bool | None:
    """Whether the sequence's page at ``(ordinal, lc_id)`` still points at a tree block.

    ``None`` when the slot is empty or holds an uncommitted page. Test hook: in C++ the
    back-pointer is raw, so a page left pointing at a block that dies first is read after
    free, and freed-but-mapped memory reads back plausibly enough that only a sanitizer
    build catches the fault itself.
    """
    return _cpp().committed_page_is_linked(kv_cache, ordinal, lc_id)


class HostMemPlacement(NamedTuple):
    """Where a host-tier allocation's pages landed, and under which NUMA policy."""

    gpu_numa_node: int
    #: None where the policy could not be read back, which also means mbind did
    #: not apply and the placement came from first touch.
    strict_binding: bool | None
    #: Resident pages per NUMA node.
    node_page_counts: dict[int, int]


def probe_host_mem_placement(
    size: int, allow_remote_numa_fallback: bool
) -> HostMemPlacement | None:
    """Allocate host-tier memory the way the host tier would and report its placement.

    None where the question does not apply: a platform whose selected backing is
    not mmap, no libnuma, or a single NUMA node.
    """
    placement = _cpp().probe_host_mem_placement(size, allow_remote_numa_fallback)
    if placement is None:
        return None
    return HostMemPlacement(
        gpu_numa_node=placement.gpu_numa_node,
        strict_binding=placement.strict_binding,
        node_page_counts=dict(placement.node_page_counts),
    )


def all_tree_pages_droppable(manager: Any) -> bool:
    """Return whether every page reachable from the radix tree is droppable."""
    return bool(_cpp().all_tree_pages_droppable(manager))


def is_commit_allowed(kv_cache: Any) -> bool:
    """Return whether the KV cache still allows token commits."""
    return bool(_cpp().is_commit_allowed(kv_cache))


def current_gpu_ratio(manager: Any) -> list[float]:
    """Return the current GPU pool-group ratio list."""
    return list(_cpp().current_gpu_ratio(manager))


def set_num_sampled_kv_caches(manager: Any, value: int) -> None:
    """Set the sampled-KV-cache counter that gates auto-tuner rebalancing."""
    _cpp().set_num_sampled_kv_caches(manager, value)


def set_last_adjustment_time(manager: Any, value: float) -> None:
    """Set the auto-tuner's last-adjustment timestamp."""
    _cpp().set_last_adjustment_time(manager, value)


def set_target_ratio_list_gpu(manager: Any, ratios: list[float]) -> None:
    """Set the auto-tuner's target GPU pool-group ratio list."""
    _cpp().set_target_ratio_list_gpu(manager, list(ratios))


def force_rebalance_precondition(manager: Any, skew: float = 2.0) -> None:
    """Force the V2 auto-tuner to do real pool-resize work on the next rebalance.

    Bypasses the sample-count / cooldown gates and perturbs the target GPU
    ratio so it differs from the current ratio by more than the auto-tuner's
    adjustment threshold. Requires a model with >=2 pool groups (e.g. a VSWA
    model) and raises ``ValueError`` otherwise, so a future model change can't
    silently turn a dependent test into a no-op. White-box hook intended for
    accuracy tests, not production code.
    """
    current = current_gpu_ratio(manager)
    if len(current) < 2:
        raise ValueError(
            f"force_rebalance_precondition requires >=2 pool groups; got {len(current)}. "
            "Check that VSWA is actually configured for this model."
        )
    set_num_sampled_kv_caches(manager, 2001)
    set_last_adjustment_time(manager, 0.0)
    skewed = [current[0] * skew] + current[1:]
    total = sum(skewed)
    set_target_ratio_list_gpu(manager, [x / total for x in skewed])


def storage_utilization(manager: Any, cache_level: int = 0) -> list[float]:
    """Return storage utilization by pool group for a cache level."""
    return list(_cpp().storage_utilization(manager, cache_level))


def sparse_cache_resident_ordinals(kv_cache: Any, lc_id: int) -> list[int]:
    """Resident unit ordinal per sparse cache entry, laid out [buffer][entry].

    A free entry holds the bad-ordinal sentinel, which is -1, the same value as
    BAD_PAGE_INDEX. Empty when the life cycle reserves no cache.
    """
    return list(_cpp().sparse_cache_resident_ordinals(kv_cache, lc_id))


def sparse_cache_base_cell(kv_cache: Any, lc_id: int, pool: int = 0) -> int | None:
    """First cell of this request's lease in one pool's arena region, or None if it holds none.

    Identifies the lease the way the page list used to identify the reservation, now that a
    lease is one contiguous run of cells rather than scattered slots. Per pool, because the
    arena mirrors the hot pool group: buffers of different sizes sit in regions of their own.
    """
    cell = _cpp().sparse_cache_base_cell(kv_cache, lc_id, pool)
    return None if cell is None else int(cell)


def sparse_flush_counts(kv_cache: Any, lc_id: int) -> list[int]:
    """Units the last write-back moved for this cache's row, per layer.

    -1 where no write-back has run for that layer, which distinguishes "never flushed"
    from "flushed and found nothing resident".

    The lists belong to the Batch rather than the request -- one scan covers every row --
    so this reads the cache's own row out of it, and an unbatched cache has none.
    """
    return list(_cpp().sparse_flush_counts(kv_cache, lc_id))


def batch_page_table(batch: Any, layer_group_id: int) -> list[int]:
    """A Batch's published device page table, row-major [max_batch][max_blocks_per_seq].

    Reads the device copy, so it shows what the last publish() put there rather than what
    the staging rows currently hold. Synchronises.
    """
    return list(_cpp().batch_page_table(batch, layer_group_id))


def batch_index_table_slot(batch: Any, layer_id: int, role: str) -> tuple[int, int]:
    """``(table, entry)`` serving one buffer.

    A consumer names a table by the spec it asked for -- :meth:`Batch.index_table_id` --
    and slices it at the buffer's position in that spec, which it already knows. This is
    the same answer reached from the buffer, and exists so a test can compare the two.
    """
    return tuple(_cpp().batch_index_table_slot(batch, layer_id, role))


def batch_index_table_base(batch: Any, layer_id: int, role: str) -> int:
    """What a converted index for one buffer is added to.

    Buffers sharing a pool share it, their own offsets being already in the entries.
    """
    return _cpp().batch_index_table_base(batch, layer_id, role)


def batch_index_table(batch: Any, table_id: int) -> list[int]:
    """A Batch's converted index table, row-major [max_batch][entries][blocks_per_seq].

    Reads the device copy, so it shows what the last publish() converted rather than what
    the raw staging holds. Synchronises.
    """
    return list(_cpp().batch_index_table(batch, table_id))


def batch_gpu_coalesced_page_table(batch: Any, layer_group_id: int) -> Any:
    """One layer group's coalesced page table on the device, as a DLPack array.

    ``int32[max_batch, max_blocks_per_seq]``. A view rather than a copy, so unlike the
    other hooks here it neither synchronises nor returns a plain list: callers check the
    device-array contract and that the pointer is stable across resizes. The indices a
    consumer actually reads come from ``Batch.coalesced_page_table()``, on the host.
    """
    return _cpp().batch_gpu_coalesced_page_table(batch, layer_group_id)


def batch_sparse_entries_per_row(batch: Any, layer_group_id: int) -> list[int]:
    """Metadata entries per layer of the lease each row holds.

    Zero where the row holds none, so this is non-zero exactly where a member holds a
    reservation for that layer group. Synchronises.
    """
    return list(_cpp().batch_sparse_entries_per_row(batch, layer_group_id))


def batch_watermarks(batch: Any) -> tuple[list[int], list[int]]:
    """A Batch's published per-row (history_length, capacity), in tokens.

    Reads the device copy, so it shows what the last publish() put there. Synchronises.
    """
    history, capacity = _cpp().batch_watermarks(batch)
    return list(history), list(capacity)


class SparseResolveResult(NamedTuple):
    """State and outputs of one sparse resolve launch."""

    resident_ordinal: list[int]
    lru_order: list[int]
    out_slot: list[int]
    #: The compacted work list, one run per layer, of which the first ``miss_total[layer]``
    #: are valid. Entry order is whatever the CTAs raced to, so compare as a set.
    miss_row: list[int]
    miss_ordinal: list[int]
    miss_slot: list[int]
    miss_total: list[int]
    miss_count: list[int]
    in_flight_slot: list[int]


def sparse_unit_copy(
    host_pool: bytes | list[int],
    host_page_stride: int,
    page_table: list[int],
    device_arena: bytes | list[int],
    base_cell: int,
    num_blocks: int,
    cells_per_buffer: int,
    buffer_bytes: int,
    num_chunks: int,
    chunk_bytes: int,
    chunk_stride: int,
    unit_stride: int,
    expansion: int,
    miss_ordinal: list[int],
    miss_slot: list[int],
    miss_count: list[int],
    selection_width: int,
    buffer_layer: list[int],
    write_back: bool = False,
    grid_ctas: int = 1,
) -> list[int]:
    """Move units between a caller-supplied cold pool and cache entries.

    Exercises the copy's address arithmetic against a known byte pattern. Returns the
    destination pool: the cache pool when fetching, the cold pool when writing back.

    ``page_table`` holds converted indices, [num_buffers][num_blocks]: one entry per
    buffer, each counting whole buffers rather than coalesced slots, so a buffer's offset
    within its page is already in the value and ``host_page_stride`` is one buffer.

    ``buffer_layer`` maps each buffer to the work-list row it reads, so buffers of one
    layer share a list; its length is the number of buffers copied.

    ``grid_ctas`` is how many CTAs walk the compacted list. Only throughput depends on it,
    so a result must not. Synchronises.
    """
    return list(
        _cpp().sparse_unit_copy(
            list(host_pool),
            host_page_stride,
            page_table,
            list(device_arena),
            base_cell,
            num_blocks,
            cells_per_buffer,
            buffer_bytes,
            num_chunks,
            chunk_bytes,
            chunk_stride,
            unit_stride,
            expansion,
            miss_ordinal,
            miss_slot,
            miss_count,
            selection_width,
            buffer_layer,
            write_back,
            grid_ctas,
        )
    )


class SparseFlushList(NamedTuple):
    """Entries holding units a watermark move has just passed."""

    #: The compacted list, one run per layer, of which the first ``total[layer]`` are
    #: valid. Entry order is whatever the CTAs raced to, so compare as a set.
    row: list[int]
    ordinal: list[int]
    slot: list[int]
    total: list[int]
    count: list[int]


def sparse_flush_list(
    resident_ordinal: list[int],
    num_layers: int,
    entries_per_layer: int,
    from_units: int,
    to_units: int,
    max_flush_units: int,
) -> SparseFlushList:
    """Collect the entries whose units fall in ``[from_units, to_units)``.

    These are the units the watermark has just made selectable, so they must reach their
    host pages before any later fetch can read them. Synchronises.
    """
    row, ordinal, slot, total, count = _cpp().sparse_flush_list(
        resident_ordinal,
        num_layers,
        entries_per_layer,
        from_units,
        to_units,
        max_flush_units,
    )
    return SparseFlushList(list(row), list(ordinal), list(slot), list(total), list(count))


def sparse_resolve(
    resident_ordinal: list[int],
    lru_order: list[int],
    num_layers: int,
    entries_per_layer: int,
    selected: list[int],
    selection_width: int,
    history_units: int | list[int],
    capacity_units: int | list[int] = 0,
    max_in_flight_units: int = 0,
    inactive_rows: list[int] | None = None,
) -> SparseResolveResult:
    """Run the sparse resolve kernel over caller-supplied cache state.

    Issues one launch per layer, as a step does: a layer's selection is not known until
    the previous layer has run. Cache state is therefore [row][layer][entry], following
    the view each row owns, while the selection and every output are [layer][row][width]
    so that each launch sees a contiguous slice of its own.

    A negative selection entry is padding. ``inactive_rows`` get a default view, standing
    in for a row holding no reservation. Ordinals in ``[history_units, capacity_units)`` are in
    flight: they are made resident before the selection takes any victim, are never
    chosen as victims themselves, and never enter the miss list because their host pages
    hold nothing to fetch. Synchronises.
    """
    # One row unless the caller passes per-row lists; the kernel is batched either way.
    history = [history_units] if isinstance(history_units, int) else list(history_units)
    if isinstance(capacity_units, int):
        capacity = [max(capacity_units, h) for h in history]
    else:
        capacity = list(capacity_units)

    (
        resident,
        lru,
        out,
        miss_row,
        miss_ord,
        miss_slot,
        miss_total,
        miss_count,
        in_flight,
    ) = _cpp().sparse_resolve(
        resident_ordinal,
        lru_order,
        num_layers,
        entries_per_layer,
        selected,
        selection_width,
        history,
        capacity,
        max_in_flight_units,
        list(inactive_rows or []),
    )
    return SparseResolveResult(
        list(resident),
        list(lru),
        list(out),
        list(miss_row),
        list(miss_ord),
        list(miss_slot),
        list(miss_total),
        list(miss_count),
        list(in_flight),
    )


def grains_for_slots(
    num_slots: int,
    slot_size_list: list[int],
    granularity: int,
    pools: list[int] | None = None,
) -> int:
    """Return the grain count required for a pool group slot count.

    ``pools`` names which of ``slot_size_list``'s pools to count, defaulting to all of
    them -- which is what a caller with a single budget means.
    """
    return int(_cpp().grains_for_slots(num_slots, slot_size_list, granularity, pools))


def grains_to_slots(
    pg_grains: int,
    slot_size_list: list[int],
    granularity: int,
    pools: list[int] | None = None,
) -> tuple[int, int]:
    """Return (slot count, consumed grains) for a pool group grain budget."""
    slots, used = _cpp().grains_to_slots(pg_grains, slot_size_list, granularity, pools)
    return int(slots), int(used)


def ratio_to_slot_count_list(
    quota: int,
    slot_size_lists: list[list[int]],
    ratio_list: list[float],
    granularity: int,
    min_slots: list[int],
    second_quota: int | None = None,
    second_granularity: int | None = None,
    second_pools: list[list[int]] | None = None,
) -> list[int]:
    """Return slot counts by pool group for a quota and ratio list.

    ``slot_size_lists`` holds every pool's size. A second medium's budget is given by
    ``second_quota`` with its own ``second_granularity`` and ``second_pools``; the quota
    pays for every pool that budget does not name, so the two cannot overlap or leave a
    pool unpaid.
    """
    return list(
        _cpp().ratio_to_slot_count_list(
            quota,
            slot_size_lists,
            ratio_list,
            granularity,
            min_slots,
            second_quota,
            second_granularity,
            second_pools,
        )
    )


def attention_life_cycle_ids(manager: Any) -> list[int]:
    """Return the lifecycle ids of all attention lifecycles, in order."""
    return list(_cpp().attention_life_cycle_ids(manager))


def swa_life_cycle_ids(manager: Any) -> list[int]:
    """Return the lifecycle ids of attention lifecycles that use a sliding window."""
    return list(_cpp().swa_life_cycle_ids(manager))


def ssm_life_cycle_id(manager: Any) -> int | None:
    """Return the SSM lifecycle id, or None if there is no SSM lifecycle."""
    return _cpp().ssm_life_cycle_id(manager)


def reuse_match_pages(
    manager: Any,
    reuse_scope: Any,
    tokens: Any,
    lc_id: int,
    enable_partial: bool = False,
) -> tuple[int, list[tuple[int, int | None] | None]]:
    """Match ``tokens`` against the radix tree and report reusable pages per block.

    Returns ``(num_tokens, pages)`` where ``pages[i]`` is ``None`` when block ``i``
    holds no page for lifecycle ``lc_id``, otherwise ``(slot_id, num_tokens_in_block)``.
    See ``CommittedPage.num_tokens_in_block`` for how attention and SSM life cycles
    interpret the recorded token count.
    """
    num_tokens, raw_pages = _cpp().reuse_match_pages(
        manager, reuse_scope, list(tokens), lc_id, enable_partial
    )
    pages: list[tuple[int, int | None] | None] = []
    for entry in raw_pages:
        if entry is None:
            pages.append(None)
        else:
            slot_id, num_tokens_in_block = entry
            pages.append((slot_id, None if num_tokens_in_block < 0 else num_tokens_in_block))
    return num_tokens, pages


def reuse_match_planned_drop_counts(
    manager: Any,
    reuse_scope: Any,
    tokens: Any,
    lc_id: int,
    enable_partial: bool = False,
) -> tuple[int, list[int | None]]:
    """Match ``tokens`` and report each matched block's ``planned_drop_count`` for lifecycle ``lc_id``.

    Returns ``(num_tokens, counts)`` where ``counts[i]`` is ``None`` when block ``i`` holds no page
    for lifecycle ``lc_id``, otherwise the matched page's ``planned_drop_count``.
    """
    return _cpp().reuse_match_planned_drop_counts(
        manager, reuse_scope, list(tokens), lc_id, enable_partial
    )


def compute_slots_for_batch(
    manager: Any,
    batch: Any,
    tokens_per_block: int,
    swa_scratch_reuse: Any = None,
) -> list[int]:
    """Return the minimum per-pool-group slot counts to support ``batch``."""
    return list(_cpp().compute_slots_for_batch(manager, batch, tokens_per_block, swa_scratch_reuse))


def tier_allocation_granularity(tier: Any, quota: int) -> int:
    """Return the unit a ``tier`` of ``quota`` bytes allocates its pools in."""
    return _cpp().tier_allocation_granularity(tier, quota)
