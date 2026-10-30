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

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import (
    HostHistoryReservations,
    HostHistoryShape,
    LifeCyclePage,
    SparseArenaShape,
    SparseBatchShape,
    host_history_shape,
    life_cycle_pages,
    plan_host_history_bytes,
    solve_sparse_offload_capacity,
    sparse_arena_bytes,
    sparse_arena_shapes,
    sparse_batch_shape,
    sparse_cache_cells,
    sparse_cache_config,
    sparse_history_bytes_per_block,
    sparse_history_floor_bytes,
    sparse_units_per_page,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BatchDesc,
    BufferConfig,
    CacheTier,
    DataRole,
    GpuCacheTierConfig,
    KVCacheDesc,
    KVCacheManagerConfig,
    LayerId,
    SparseCacheConfig,
    SsmLayerConfig,
    _introspection,
)

TOKENS_PER_BLOCK = 128
# A V4-Flash NVFP4 compressed row covers four tokens: 288 data bytes and 36 scale bytes.
V4_FLASH_HOST_BYTES_PER_BLOCK = 21 * 32 * (288 + 36)
# The data and the scales of its 21 CSA layers each take a host-backed pool.
V4_FLASH_HOST_HISTORY = HostHistoryShape(
    num_pools=2, bytes_per_block=V4_FLASH_HOST_BYTES_PER_BLOCK, max_util_for_resume=0.95
)
V4_FLASH_ARENA = SparseArenaShape(
    num_layers=21,
    num_selections=21,
    cells_per_buffer=33,
    units_per_cell=32,
    cell_bytes=(32 * 288, 32 * 36) * 21,
)
# 256 rows of up to 1M tokens: a table of 8,196 pages for each of three life
# cycles, a converted index table for each of the history's two sparse buffers,
# and one fetch of 512 rows.
V4_FLASH_BATCH = SparseBatchShape(
    num_life_cycles=3,
    max_batch_size=256,
    max_blocks_per_seq=8196,
    selection_widths=(512,),
    layer_counts=(2,),
    flush_capacities=(256,),
    index_table_entries=2,
)
# The Batch and DeepSeek-V4's attention metadata offload state, of which the eager
# and the CUDA graph metadata each hold one: 64 MiB of dequantized NVFP4 rows and
# 1.5 MiB of indices.
V4_FLASH_BATCH_BUFFER_BYTES = V4_FLASH_BATCH.device_bytes + 2 * 68_685_828


def _layer(
    layer_id: int,
    *buffers: tuple[str, int, bool],
    window: int | None = None,
    sparse_cache: SparseCacheConfig | None = None,
) -> AttentionLayerConfig:
    return AttentionLayerConfig(
        layer_id=LayerId(layer_id),
        buffers=[
            BufferConfig(role=DataRole(role), size=size, is_sparse=is_sparse)
            for role, size, is_sparse in buffers
        ],
        sliding_window_size=window,
        sparse_cache=sparse_cache,
    )


def _sparse_layer(
    layer_id: int,
    max_cached_units: int = 33 * 32,
    *,
    fetched: bool = True,
    selection_id: int | None = None,
) -> AttentionLayerConfig:
    """A V4-Flash CSA layer: offloaded NVFP4 rows beside the GPU-resident fp4 indexer.

    One compressed row covers four tokens, 32 rows to a page.
    """
    return _layer(
        layer_id,
        ("data", 32 * 288, True),
        ("scale", 32 * 36, True),
        ("indexer", 32 * 68, False),
        sparse_cache=SparseCacheConfig(
            num_selected_blocks=512,
            max_cached_units=max_cached_units,
            unit_tokens=4,
            selection_id=selection_id,
        )
        if fetched
        else None,
    )


def _config(layers, constraints) -> KVCacheManagerConfig:
    return KVCacheManagerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        cache_tiers=[GpuCacheTierConfig(quota=1 << 30)],
        layers=layers,
        constraints=constraints,
    )


def _v4_flash_like_layers() -> list:
    return [
        _layer(0, ("swa", 65_536, False), window=128),
        _sparse_layer(1),
        # A ratio-128 layer keeps its one compressed row per page on the GPU.
        _layer(2, ("data", 288, False), ("scale", 36, False)),
        AttentionLayerConfig(
            layer_id=LayerId(3),
            buffers=[
                BufferConfig(role=DataRole("state"), size=16_384),
                # A page of eight-token blocks expands to 16 of them.
                BufferConfig(role=DataRole("score"), size=25, tokens_per_block_override=8),
            ],
            sliding_window_size=8,
        ),
        _sparse_layer(4),
    ]


# Native layout


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("unit_tokens", "units_per_page"), [(4, 32), (1, 128), (TOKENS_PER_BLOCK, 1), (None, 1)]
)
def test_units_per_page(unit_tokens: int | None, units_per_page: int) -> None:
    assert sparse_units_per_page(unit_tokens, TOKENS_PER_BLOCK) == units_per_page


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("cells", "lease_bytes"),
    [
        # 21 layers x cells x (9216 B data + 1152 B scale), plus two int32 arrays
        # per cached row. A lease holds the reservation and nothing of the step: the
        # work lists belong to the Batch, sized by the batch rather than by whoever
        # is seated.
        (17, 3_793_024),
        (32, 7_139_328),
        (33, 7_362_688),
    ],
)
def test_lease_matches_native_layout(cells: int, lease_bytes: int) -> None:
    shape = SparseArenaShape(
        num_layers=21,
        num_selections=21,
        cells_per_buffer=cells,
        units_per_cell=32,
        cell_bytes=(9216, 1152) * 21,
    )

    assert shape.lease_bytes == lease_bytes
    # Shared tables and region alignment are paid once, not per lease.
    assert shape.arena_bytes(513) - shape.arena_bytes(512) == pytest.approx(lease_bytes, abs=512)


@pytest.mark.cpu_only
def test_arena_follows_the_batch_rows() -> None:
    layers = [_layer(0, ("key", 65_536, False), window=128)]
    layers += [_sparse_layer(i) for i in range(1, 22)]
    config = _config(
        layers,
        [
            BatchDesc([KVCacheDesc(capacity=4096, history_length=4095)] * 6),
            BatchDesc([KVCacheDesc(capacity=0, history_length=0)] * 513),
        ],
    )

    (shape,) = sparse_arena_shapes(config.layers, tokens_per_block=TOKENS_PER_BLOCK)

    assert shape == V4_FLASH_ARENA
    assert shape.lease_bytes == 7_362_688
    assert sparse_arena_bytes(config, 513) == shape.arena_bytes(513)


@pytest.mark.cpu_only
def test_layers_of_one_arena_must_agree_on_the_cache() -> None:
    layers = [_sparse_layer(0, 17 * 32), _sparse_layer(1, 33 * 32)]

    with pytest.raises(ValueError, match="same sparse_cache"):
        sparse_arena_shapes(layers, tokens_per_block=TOKENS_PER_BLOCK)


@pytest.mark.cpu_only
def test_layers_without_a_fetch_path_join_no_arena() -> None:
    # A layer whose life cycle has an arena, but which reads nothing through it.
    layers = [_sparse_layer(0), _sparse_layer(1, 17 * 32, fetched=False)]

    (shape,) = sparse_arena_shapes(layers, tokens_per_block=TOKENS_PER_BLOCK)

    assert shape.num_layers == 1
    assert shape.cell_bytes == (32 * 288, 32 * 36)


@pytest.mark.cpu_only
def test_layers_sharing_a_selection_share_a_metadata_row() -> None:
    """The arena's metadata is sized by selections, so sharing one shrinks it."""
    apart = sparse_arena_shapes(
        [_sparse_layer(0), _sparse_layer(1)], tokens_per_block=TOKENS_PER_BLOCK
    )
    together = sparse_arena_shapes(
        [_sparse_layer(0), _sparse_layer(1, selection_id=0)], tokens_per_block=TOKENS_PER_BLOCK
    )

    (apart_shape,) = apart
    (shared_shape,) = together
    assert (apart_shape.num_layers, apart_shape.num_selections) == (2, 2)
    assert (shared_shape.num_layers, shared_shape.num_selections) == (2, 1)
    # The cells are per layer and unchanged; only the residency and recency rows merge.
    assert shared_shape.cell_bytes == apart_shape.cell_bytes
    assert shared_shape.entries == apart_shape.entries // 2
    assert shared_shape.lease_bytes < apart_shape.lease_bytes


@pytest.mark.cpu_only
def test_layers_with_different_windows_get_separate_arenas() -> None:
    windowed = _sparse_layer(1)
    windowed.sliding_window_size = 4096

    shapes = sparse_arena_shapes([_sparse_layer(0), windowed], tokens_per_block=TOKENS_PER_BLOCK)

    assert [shape.num_layers for shape in shapes] == [1, 1]


@pytest.mark.cpu_only
def test_no_arena_without_a_fetch_path_or_leases() -> None:
    lease_constraint = [BatchDesc([KVCacheDesc(capacity=0, history_length=0)] * 4)]

    assert (
        sparse_arena_bytes(_config([_layer(0, ("key", 65_536, False))], lease_constraint), 4) == 0
    )
    # A sparse buffer without a fetch path has no cache to lease.
    unfetched = _sparse_layer(0, fetched=False)
    assert sparse_arena_bytes(_config([unfetched], lease_constraint), 4) == 0
    # No rows to seat means no leases, whatever the layers declare.
    assert sparse_arena_bytes(_config([_sparse_layer(0)], []), 0) == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("quota", "granule_bytes"),
    [
        (512 << 20, 2 << 20),
        (1 << 30, 2 << 20),
        (3 << 30, 4 << 20),
        (16 << 30, 32 << 20),
        (1 << 40, 32 << 20),
    ],
)
def test_gpu_pool_granule_follows_the_quota(quota: int, granule_bytes: int) -> None:
    assert _introspection.tier_allocation_granularity(CacheTier.GPU_MEM, quota) == granule_bytes


@pytest.mark.cpu_only
def test_host_history_holds_a_page_of_every_sparse_buffer() -> None:
    layers = [_sparse_layer(layer_id) for layer_id in range(21)]
    layers.append(_layer(21, ("key", 8704, False)))

    assert sparse_history_bytes_per_block(layers) == V4_FLASH_HOST_BYTES_PER_BLOCK
    # A sparse buffer without a fetch path still keeps its history on the host.
    assert (
        sparse_history_bytes_per_block([_layer(0, ("key", 4096, True), ("value", 4096, False))])
        == 4096
    )


@pytest.mark.cpu_only
def test_life_cycles_follow_native_numbering_and_split_bytes_by_memory() -> None:
    life_cycles = life_cycle_pages(_v4_flash_like_layers(), TOKENS_PER_BLOCK)

    assert life_cycles == [
        LifeCyclePage(
            window_size=128,
            num_sink_tokens=0,
            is_state=False,
            device_bytes=65_536,
            host_bytes=0,
            device_pool_bytes=(65_536,),
        ),
        # Both compressed layers and the ratio-128 layer share the no-window life
        # cycle, and the two indexers of one page size share a pool.
        LifeCyclePage(
            window_size=None,
            num_sink_tokens=0,
            is_state=False,
            device_bytes=2 * 2176 + 288 + 36,
            host_bytes=2 * (9216 + 1152),
            device_pool_bytes=(2 * 2176, 288, 36),
        ),
        LifeCyclePage(
            window_size=8,
            num_sink_tokens=0,
            is_state=False,
            device_bytes=16_384 + 400,
            host_bytes=0,
            device_pool_bytes=(16_384, 400),
        ),
    ]
    ssm = SsmLayerConfig(
        layer_id=LayerId(5), buffers=[BufferConfig(role=DataRole("ssm"), size=4096)]
    )
    (state,) = life_cycle_pages([ssm], TOKENS_PER_BLOCK)
    assert state.is_state and state.device_bytes == 4096
    assert state.device_pool_bytes == (4096,)


@pytest.mark.cpu_only
def test_life_cycle_with_only_sparse_buffers_is_rejected() -> None:
    with pytest.raises(ValueError, match="must also hold a dense buffer"):
        life_cycle_pages([_layer(0, ("history", 4096, True))], TOKENS_PER_BLOCK)


@pytest.mark.cpu_only
def test_batch_matches_native_layout() -> None:
    # Three int32 page tables of 8,196 pages, two watermarks per row, and one arena slot
    # per row, which serves every life cycle.
    assert V4_FLASH_BATCH.block_bytes == 4 * 256 * (3 * 8196 + 2 + 1)
    assert V4_FLASH_BATCH.block_bytes == 25_181_184
    # The mirror carries only the block's tail: what a captured kernel reads.
    assert V4_FLASH_BATCH.mirror_bytes == 4 * 256 * (2 + 1) == 3_072
    # The staging copy is mapped in whole 4 KiB pages.
    assert V4_FLASH_BATCH.staging_bytes == 25_182_208
    # Per row: a slot per selected unit, a miss count and the unit in flight. Then the
    # two compacted work lists -- a row, an ordinal and a slot per entry, plus a total
    # per list -- one for the fetch, sized for every row missing its whole selection,
    # and one per layer for the write-back, sized by the step's token budget.
    assert V4_FLASH_BATCH.fetch_scratch_bytes == 4 * (
        256 * (512 + 2) + (3 * 256 * 512 + 1) + 2 * (3 * 256 + 1) + 256 * 2
    )
    assert V4_FLASH_BATCH.fetch_scratch_bytes == 2_107_404
    # The fetch and the write-back read converted indices: a table per sparse buffer,
    # shared by the life cycle's layers since nothing substitutes a scratch slot there.
    assert V4_FLASH_BATCH.index_table_bytes == 4 * 2 * 256 * 8196 == 16_785_408
    # The device holds the staged block a publish fills ahead of the step, the mirror of
    # its tail that a captured kernel reads, the converted tables and the work arrays.
    assert V4_FLASH_BATCH.device_bytes == 25_181_184 + 3_072 + 16_785_408 + 2_107_404 == 44_077_068


@pytest.mark.cpu_only
def test_the_token_budget_sizes_the_write_back_list() -> None:
    """A row advancing t tokens crosses at most t/unit_tokens + 1 units."""
    layers = _v4_flash_like_layers()
    shape = sparse_batch_shape(
        layers,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_batch_size=256,
        max_blocks_per_seq=8196,
        max_total_input_tokens=8192,
    )

    # Four tokens to a unit, and one partial unit per row at the start of its move.
    assert shape.flush_capacities == (8192 // 4 + 256,)
    # The bound is an aggregate: it does not multiply by rows, which is what makes it
    # affordable. Per row it would have to allow for one row taking the whole budget.
    assert shape.flush_capacities[0] * 2 < 256 * (8192 // 4 + 1)
    assert shape.fetch_scratch_bytes > V4_FLASH_BATCH.fetch_scratch_bytes

    # Left unset it assumes a decode step: one token per row, so one unit boundary.
    assert V4_FLASH_BATCH.flush_capacities == (256,)


@pytest.mark.cpu_only
def test_batch_has_a_page_table_per_life_cycle_and_a_fetch_per_sparse_one() -> None:
    layers = _v4_flash_like_layers()

    assert (
        sparse_batch_shape(
            layers, tokens_per_block=TOKENS_PER_BLOCK, max_batch_size=256, max_blocks_per_seq=8196
        )
        == V4_FLASH_BATCH
    )
    # Layers without a fetch path still get page tables, but no fetch work arrays.
    layers = [_layer(0, ("swa", 4096, False), window=128), _sparse_layer(1, fetched=False)]
    batch = sparse_batch_shape(
        layers, tokens_per_block=TOKENS_PER_BLOCK, max_batch_size=4, max_blocks_per_seq=8
    )
    assert batch.num_life_cycles == 2
    assert batch.selection_widths == ()
    assert batch.fetch_scratch_bytes == 0


# Sizing policy


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("cache_factor", "num_selected_blocks", "units_per_cell", "cells"),
    [
        # The smallest cache holds 512 selected rows plus the row being written.
        (1.0, 512, 32, 17),
        # The factor scales the selection; the row being written stays one row.
        (1.5, 512, 32, 25),
        (2.0, 512, 32, 33),
        (3.0, 512, 32, 49),
        # A selection that does not fill whole cells rounds up once, at the end.
        (2.0, 500, 32, 32),
        (1.0, 512, 64, 9),
        (1.0, 1, 32, 1),
        # Float error from a computed factor does not add a unit.
        (0.1 * 3 / 0.3, 512, 32, 17),
    ],
)
def test_sparse_cache_cells_scale_the_selection(
    cache_factor: float, num_selected_blocks: int, units_per_cell: int, cells: int
) -> None:
    assert sparse_cache_cells(cache_factor, num_selected_blocks, units_per_cell) == cells


@pytest.mark.cpu_only
def test_sparse_cache_cells_reject_less_than_one_selection() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        sparse_cache_cells(0.5, 512, 32)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("unit_tokens", "num_selected_units", "units_per_page", "cells"),
    [
        # Two selections of 512 rows plus the one being written fill 33 cells of 32 rows.
        (4, 512, 32, 33),
        # Two selections of 2048 tokens plus one fill 33 cells of 128 tokens.
        (1, 2048, 128, 33),
        # Two selections of 128 pools of 16 tokens plus one fill 33 cells of 8 pools.
        (16, 128, 8, 33),
        # Two selections of 64 pages plus one, a unit given as a page or left unset.
        (TOKENS_PER_BLOCK, 64, 1, 129),
        (None, 64, 1, 129),
    ],
    ids=["compressed_row", "token", "token_pool", "page", "unset"],
)
def test_sparse_cache_config_sizes_the_cache_for_any_selection_unit(
    unit_tokens: int | None, num_selected_units: int, units_per_page: int, cells: int
) -> None:
    cache = sparse_cache_config(
        num_selected_units=num_selected_units,
        unit_tokens=unit_tokens,
        tokens_per_block=TOKENS_PER_BLOCK,
        gpu_cache_factor=2.0,
    )

    assert cache.num_selected_blocks == num_selected_units
    # The config names a definite unit size: a caller passing none means a whole page,
    # and the helper resolves that rather than storing the absence.
    assert cache.unit_tokens == (TOKENS_PER_BLOCK if unit_tokens is None else unit_tokens)
    # The default cache holds two selections plus the unit being written, in whole cells.
    assert cache.max_cached_units == cells * units_per_page
    (arena,) = sparse_arena_shapes(
        [_layer(0, ("history", 64 * units_per_page, True), ("key", 64, False), sparse_cache=cache)],
        tokens_per_block=TOKENS_PER_BLOCK,
    )
    assert arena.units_per_cell == units_per_page
    assert arena.cells_per_buffer == cells
    assert arena.cell_bytes == (64 * units_per_page,)


@pytest.mark.cpu_only
def test_sparse_cache_config_rejects_a_unit_that_does_not_divide_a_page() -> None:
    with pytest.raises(ValueError, match="must divide tokens_per_block"):
        sparse_cache_config(
            num_selected_units=8,
            unit_tokens=48,
            tokens_per_block=TOKENS_PER_BLOCK,
            gpu_cache_factor=1.0,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("host_history_size", "pinnable_host_bytes", "budget"),
    [
        # An explicit size covers the staging buffers too, whatever automatic
        # sizing may pin.
        (1 << 30, 0.0, (1 << 30) - 4096),
        # Otherwise the pools take what holds the batch...
        (None, float("inf"), 256_000),
        # ...unless they and the staging buffers outgrow what automatic sizing
        # may pin.
        (None, 50_000.0, 50_000 - 4096),
    ],
)
def test_host_budget(
    host_history_size: int | None, pinnable_host_bytes: float, budget: int
) -> None:
    assert (
        plan_host_history_bytes(
            host_history_size=host_history_size,
            batch_bytes=256_000,
            pinnable_host_bytes=pinnable_host_bytes,
            staging_bytes=4096,
        )
        == budget
    )


@pytest.mark.cpu_only
def test_automatic_host_budget_grows_to_the_floor() -> None:
    # The batch would take 256,000 bytes, short of the 300,000 the manager's
    # constraint batches need.
    assert (
        plan_host_history_bytes(
            host_history_size=None,
            batch_bytes=256_000,
            pinnable_host_bytes=float("inf"),
            staging_bytes=4096,
            floor_bytes=300_000,
        )
        == 300_000
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("host_history_size", "pinnable_host_bytes", "match"),
    [
        (200_000, float("inf"), "host_history_size .* less than the 0.3MiB"),
        (None, 200_000.0, "automatically leaves 0.2MiB .* less than the 0.3MiB"),
    ],
    ids=["explicit", "automatic"],
)
def test_host_budget_below_the_floor_is_refused(
    host_history_size: int | None, pinnable_host_bytes: float, match: str
) -> None:
    # The native manager refuses a budget below the floor at construction; refusing it
    # here says what to change.
    with pytest.raises(ValueError, match=match):
        plan_host_history_bytes(
            host_history_size=host_history_size,
            batch_bytes=256_000,
            pinnable_host_bytes=pinnable_host_bytes,
            staging_bytes=4096,
            floor_bytes=300_000,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("host_history_size", "pinnable_host_bytes", "match"),
    [
        (4096, float("inf"), "host_history_size .* leaves no host memory for offloaded history"),
        (None, 4096.0, "automatically leaves no host memory for offloaded history"),
    ],
)
def test_host_budget_must_leave_the_pools_room(
    host_history_size: int | None, pinnable_host_bytes: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        plan_host_history_bytes(
            host_history_size=host_history_size,
            batch_bytes=256_000,
            pinnable_host_bytes=pinnable_host_bytes,
            staging_bytes=4096,
        )


@pytest.mark.cpu_only
def test_host_history_holds_pages_under_the_resume_limit() -> None:
    history = host_history_shape(
        _v4_flash_like_layers(), tokens_per_block=TOKENS_PER_BLOCK, max_util_for_resume=0.95
    )

    # The two sparse layers share a pool of data pages and a pool of scale pages.
    assert history == HostHistoryShape(
        num_pools=2, bytes_per_block=20_736, max_util_for_resume=0.95
    )
    # 256 requests of 32 pages beside two dummy pages: 8,194 pages, which resume keeps
    # in at most 95% of the slots.
    assert history.slots(256 * 32 + 2) == 8626
    # Each pool rounds up to a 2 MiB grain, so a budget holds a grain per pool beside them.
    assert history.history_bytes(256 * 32 + 2) == 8626 * 20_736 + 2 * (2 << 20)
    assert history.slots_in(history.history_bytes(256 * 32 + 2)) == 8626
    # The count inverts the cost: the slots sized for a batch hold all of it, and one
    # slot fewer holds a request less.
    assert history.requests_in(8626, pages_per_request=32, dummy_pages=2) == 256
    assert history.requests_in(8625, pages_per_request=32, dummy_pages=2) == 255
    # Dummies that fill the pools leave no room for a request.
    assert history.requests_in(2, pages_per_request=32, dummy_pages=2) == 0
    # A manager without host-backed pools needs no host budget.
    dense = host_history_shape(
        [_layer(0, ("key", 4096, False))],
        tokens_per_block=TOKENS_PER_BLOCK,
        max_util_for_resume=0.95,
    )
    assert dense.history_bytes(256 * 32 + 2) == 0


@pytest.mark.cpu_only
def test_host_budget_buys_the_slots_the_batch_needs_in_the_native_pools() -> None:
    """The budget is sized by the rule the native allocator divides it by.

    V4-Flash on a decode worker with max_batch_size 128 and requests of 4,096 tokens:
    the data and the scales of 21 CSA layers take two host-backed pools beside the
    GPU-resident indexer keys, and two dummy requests hold a page each.
    """
    device_quota = 100 << 30
    batch_pages = 128 * 32 + 2

    def host_slots(host_budget: int) -> int:
        (slots,) = _introspection.ratio_to_slot_count_list(
            device_quota,
            [[21 * 32 * 288, 21 * 32 * 36, 21 * 32 * 68]],
            [1.0],
            _introspection.tier_allocation_granularity(CacheTier.GPU_MEM, device_quota),
            [1],
            second_quota=host_budget,
            second_granularity=2 << 20,
            second_pools=[[0, 1]],
        )
        return slots

    def requests(slots: int) -> int:
        return V4_FLASH_HOST_HISTORY.requests_in(slots, pages_per_request=32, dummy_pages=2)

    # The bytes of 128 requests alone leave no room for each pool's grain, the dummies'
    # pages or the slots resume keeps free: they buy 4,085 slots, which hold 121.
    old_slots = host_slots(128 * 32 * V4_FLASH_HOST_BYTES_PER_BLOCK)
    assert old_slots == 4085
    assert requests(old_slots) == 121
    budget = V4_FLASH_HOST_HISTORY.history_bytes(batch_pages)
    assert host_slots(budget) >= V4_FLASH_HOST_HISTORY.slots(batch_pages)
    assert requests(host_slots(budget)) == 128


@pytest.mark.cpu_only
def test_host_floor_holds_the_largest_constraint_batch() -> None:
    layers = _v4_flash_like_layers()
    shorter = BatchDesc([KVCacheDesc(capacity=1 << 16, history_length=0)])
    # DeepSeek-V4's constraint: one max_seq_len request beside short decode requests.
    longest = BatchDesc(
        [KVCacheDesc(capacity=1 << 17, history_length=(1 << 17) - 1)]
        + [KVCacheDesc(capacity=1, history_length=0)] * 15
    )

    floor = sparse_history_floor_bytes(
        layers, [shorter, longest], tokens_per_block=TOKENS_PER_BLOCK, max_util_for_resume=0.95
    )

    # The longest request spans 1,024 pages and each short one a page: 1,039 slots, or
    # 1,094 under the 0.95 utilization the manager keeps. Each slot holds a page of both
    # sparse layers' data and scales, and each of those two pools rounds up to a 2 MiB
    # grain.
    assert sparse_history_bytes_per_block(layers) == 2 * (32 * 288 + 32 * 36)
    assert floor == 1094 * 20_736 + 2 * (2 << 20) == 26_879_488
    # A manager without constraints, or without host-backed pools, keeps no floor.
    assert (
        sparse_history_floor_bytes(
            layers, [], tokens_per_block=TOKENS_PER_BLOCK, max_util_for_resume=0.95
        )
        == 0
    )
    assert (
        sparse_history_floor_bytes(
            [_layer(0, ("key", 4096, False))],
            [longest],
            tokens_per_block=TOKENS_PER_BLOCK,
            max_util_for_resume=0.95,
        )
        == 0
    )


@pytest.mark.cpu_only
def test_host_budget_is_empty_on_a_rank_without_offloaded_history() -> None:
    assert (
        plan_host_history_bytes(
            host_history_size=None,
            batch_bytes=0,
            pinnable_host_bytes=0.0,
            staging_bytes=0,
        )
        == 0
    )


# Capacity report and host admission


def _v4_flash_capacity(**overrides):
    # 128K-token requests on a disaggregated decode worker with max_batch_size 256.
    kwargs = dict(
        num_tokens=1 << 17,
        max_batch_size=256,
        gpu_pool_bytes=100 << 30,
        # The host pools hold four requests of 128K tokens beside a dummy's page:
        # 4,313 slots, kept at most 95% full.
        host_pool_bytes=4313 * V4_FLASH_HOST_BYTES_PER_BLOCK,
        host_budget_bytes=50 << 30,
        # Every window spans two pages, plus 52,176 GPU bytes per 128 tokens.
        gpu_bytes_per_request=81_657_856 + 1024 * 52_176,
        fixed_gpu_bytes_per_request=81_657_856,
        host_history=V4_FLASH_HOST_HISTORY,
        host_pool_slots=4313,
        host_pages_per_request=1024,
        host_dummy_pages=1,
        arenas=[V4_FLASH_ARENA],
        # One lease per Batch row.
        num_leases=256,
        batch_buffer_bytes=V4_FLASH_BATCH_BUFFER_BYTES,
    )
    kwargs.update(overrides)
    return solve_sparse_offload_capacity(**kwargs)


@pytest.mark.cpu_only
def test_capacity_names_the_binding_limit_and_the_batch_that_fits() -> None:
    capacity = _v4_flash_capacity()

    assert capacity.by_limit == {"max_batch_size": 256, "host": 4, "gpu": 794}
    assert capacity.limited_by == "host"
    assert capacity.fits == 4
    assert capacity.lease_bytes == 7_362_688
    assert capacity.arena_bytes == V4_FLASH_ARENA.arena_bytes(256)
    assert capacity.host_bytes_per_request == 1024 * V4_FLASH_HOST_BYTES_PER_BLOCK
    # The 50 GiB host budget holds 228 requests of 128K tokens beside a dummy's page,
    # kept at most 95% full; the GPU budget would hold more.
    assert capacity.recommended_max_batch_size == 228
    report = capacity.describe()
    assert "4 by host" in report
    assert "limited by host" in report
    assert "256 leases" in report
    assert "offload buffers 173.0MiB for 256 rows" in report
    assert "A max_batch_size of 228 fits" in report


@pytest.mark.cpu_only
@pytest.mark.parametrize("padding_leases", [0, 1])
def test_capacity_recommendation_charges_a_batch_slot_one_lease(padding_leases: int) -> None:
    # A cache holds a lease only while it is seated in the Batch, so each slot
    # holds one, even on a disaggregated or overlap worker where the index mapper
    # admits two requests per slot.
    lease_bytes = V4_FLASH_ARENA.lease_bytes
    gpu_bytes_per_request = 81_657_856 + 1024 * 52_176
    # Ten batch slots of one lease each, plus any padding leases.
    budget = (
        10 * (gpu_bytes_per_request + lease_bytes)
        + V4_FLASH_ARENA.arena_bytes(padding_leases)
        - V4_FLASH_ARENA.arena_bytes(10 + padding_leases)
    )
    capacity = _v4_flash_capacity(
        max_batch_size=10,
        gpu_pool_bytes=budget,
        host_budget_bytes=1 << 50,
        num_leases=10 + padding_leases,
        batch_buffer_bytes=0,
    )
    assert capacity.recommended_max_batch_size == 10

    short = _v4_flash_capacity(
        max_batch_size=10,
        gpu_pool_bytes=budget - 1,
        host_budget_bytes=1 << 50,
        num_leases=10 + padding_leases,
        batch_buffer_bytes=0,
    )
    assert short.recommended_max_batch_size == 9


@pytest.mark.cpu_only
def test_capacity_recommendation_charges_every_batch_slot_its_rows_of_offload_buffers() -> None:
    gpu_bytes_per_request = 81_657_856 + 1024 * 52_176
    # 1 MiB of offload buffers per row, sized for 16 rows.
    slot_bytes = gpu_bytes_per_request + V4_FLASH_ARENA.lease_bytes + (1 << 20)
    # Device memory for ten batch slots, the arena's shared tables and the
    # offload buffers, of which the pools get what the rest leaves.
    budget = 10 * slot_bytes + V4_FLASH_ARENA.arena_bytes(0)
    gpu_pool_bytes = budget - V4_FLASH_ARENA.arena_bytes(16) - (16 << 20)
    capacity = _v4_flash_capacity(
        max_batch_size=16,
        gpu_pool_bytes=gpu_pool_bytes,
        host_budget_bytes=1 << 50,
        num_leases=16,
        batch_buffer_bytes=16 << 20,
    )

    assert capacity.recommended_max_batch_size == 10


@pytest.mark.cpu_only
def test_capacity_ties_go_to_the_batch_limit_then_host() -> None:
    capacity = _v4_flash_capacity(max_batch_size=4, gpu_pool_bytes=4 * (81_657_856 + 1024 * 52_176))
    assert capacity.by_limit == {"max_batch_size": 4, "host": 4, "gpu": 4}
    assert capacity.limited_by == "max_batch_size"

    capacity = _v4_flash_capacity(gpu_pool_bytes=4 * (81_657_856 + 1024 * 52_176))
    assert capacity.limited_by == "host"


@pytest.mark.cpu_only
def test_host_reservations_replace_and_release() -> None:
    reservations = HostHistoryReservations(capacity_blocks=10)

    assert reservations.can_reserve(1, 6)
    reservations.reserve(1, 6)
    assert not reservations.can_reserve(2, 5)
    # A request's new reservation replaces the one it holds.
    assert reservations.can_reserve(1, 10)
    reservations.reserve(1, 2)
    assert reservations.can_reserve(2, 5)
    reservations.reserve(2, 5)
    assert reservations.reserved_blocks == 7

    reservations.release(1)
    reservations.release(9)
    assert reservations.reserved_blocks == 5
