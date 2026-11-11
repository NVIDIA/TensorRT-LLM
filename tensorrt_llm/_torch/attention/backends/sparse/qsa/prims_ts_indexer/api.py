# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host entry for the paged task-scheduled QSA indexer scorer."""

import functools

import torch

from . import kernel as _k

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
# One CTA fits per SM; a CTA's prologue (page-id staging, Q load, ring fill)
# costs about as much as this many block-tile iterations.
_CTA_FIXED_ITERS = 2


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=1024)
def _pick_n_splits(tiles_bound: int, n_max: int, min_splits: int, num_sms: int) -> int:
    """Split each tile's block range so whole CTA waves do the least serial work."""
    best, best_cost = min_splits, None
    for n_splits in range(max(1, min_splits), n_max + 1):
        waves = -(-tiles_bound * n_splits // num_sms)
        iters = -(-n_max // n_splits)
        cost = waves * (iters + _CTA_FIXED_ITERS)
        if best_cost is None or cost < best_cost:
            best, best_cost = n_splits, cost
        if waves * _CTA_FIXED_ITERS >= best_cost:
            break
    return best


def _aligned(t: torch.Tensor) -> torch.Tensor:
    # Row slices of the caller's packed metadata land on 4 B boundaries; the
    # kernel's vector loads and TMA need 16 B.
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _paged_view(index_cache: torch.Tensor) -> torch.Tensor:
    # The KV pool may coalesce the index cache with K/V, so the page stride is
    # arbitrary; the kernel's TMA map takes it as a dimension. Pages must stay
    # dense inside and 16 B aligned; anything else is copied once.
    if (
        index_cache.stride()[1:] == (_k.HEAD_DIM, _k.HEAD_DIM, 1)
        and index_cache.stride(0) * index_cache.element_size() % 16 == 0
        and index_cache.data_ptr() % 16 == 0
    ):
        return index_cache
    return index_cache.contiguous()


def _ptr(t, dtype):
    import cutlass
    from cutlass.cute.runtime import make_ptr

    return make_ptr(dtype, t.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16)


@functools.cache
def _compiled(blocks_per_page: int, compress_ratio: int):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    dev = torch.device("cuda", torch.cuda.current_device())
    i32 = torch.empty(16, dtype=torch.int32, device=dev)
    f32 = torch.empty(16, dtype=torch.float32, device=dev)
    bf16 = torch.empty(16, dtype=torch.bfloat16, device=dev)
    return cute.compile(
        _k.qsa_index_scores_host,
        _ptr(bf16, cutlass.BFloat16),
        _ptr(bf16, cutlass.BFloat16),
        _ptr(i32, cutlass.Int32),
        _ptr(i32, cutlass.Int32),
        _ptr(i32, cutlass.Int32),
        _ptr(i32, cutlass.Int32),
        _ptr(f32, cutlass.Float32),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        blocks_per_page,
        compress_ratio,
        make_fake_stream(),
        options=_COMPILE_OPTIONS,
    )


def prepare_qsa_index_scores(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    *,
    tokens_per_block: int,
    compress_ratio: int,
    only_visible_blocks: bool = False,
):
    """Build device metadata and return ``(launch, scores)``.

    Rows must be sorted by request. ``launch()`` runs the kernel on the current
    stream without host synchronization and is CUDA-graph capturable; the
    metadata built here depends only on request boundaries, so a captured
    launch stays valid while those are unchanged.
    """
    rows, num_heads, head_dim = q.shape
    if (num_heads, head_dim) != (_k.NUM_HEADS, _k.HEAD_DIM):
        raise ValueError("prims_ts QSA scorer supports exactly 4 heads x 128 dims")
    if compress_ratio < 2 or tokens_per_block % compress_ratio != 0:
        raise ValueError("prims_ts QSA scorer requires compress_ratio >= 2 dividing the page size")
    blocks_per_page = tokens_per_block // compress_ratio
    if blocks_per_page > _k.TILE_N or _k.TILE_N % blocks_per_page != 0:
        raise ValueError("compressed blocks per page must divide 128")
    if (
        index_cache.ndim != 4
        or index_cache.shape[1] != tokens_per_block
        or index_cache.shape[2] != 1
    ):
        raise ValueError("index_cache must be [pages, tokens_per_block, 1, 128]")
    if q.dtype != torch.bfloat16 or index_cache.dtype != torch.bfloat16:
        raise ValueError("prims_ts QSA scorer requires BF16 q and index cache")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise ValueError("block_table must be int32 [requests, max_pages]")
    if request_indices.shape != (rows,) or query_positions.shape != (rows,):
        raise ValueError("request_indices and query_positions must have one entry per row")
    if request_indices.dtype != torch.int32:
        raise ValueError("request_indices must be int32")
    device = q.device
    num_requests = block_table.shape[0]
    max_blocks = block_table.shape[1] * blocks_per_page
    tile_meta = torch.empty(2 * num_requests, dtype=torch.int32, device=device)
    request_indices = _aligned(request_indices)
    tiles_bound = (rows + _k.QUERY_ROWS_PER_TILE - 1) // _k.QUERY_ROWS_PER_TILE + num_requests
    n_max = max(1, (max_blocks + _k.TILE_N - 1) // _k.TILE_N)
    pages_per_tile = _k.TILE_N // blocks_per_page
    min_splits = (n_max * pages_per_tile + _k.PAGE_ID_CAPACITY - 1) // _k.PAGE_ID_CAPACITY
    num_sms = _num_sms(device.index if device.index is not None else torch.cuda.current_device())
    n_splits = _pick_n_splits(tiles_bound, n_max, min_splits, num_sms)
    grid_size = tiles_bound * n_splits
    scores = torch.empty(rows, max_blocks, dtype=torch.float32, device=device)
    positions = _aligned(
        query_positions if query_positions.dtype == torch.int32 else query_positions.to(torch.int32)
    )
    fn = _compiled(blocks_per_page, compress_ratio)
    import cuda.bindings.driver as cuda_driver

    block_table = _aligned(block_table)
    q = _aligned(q)
    index_cache = _paged_view(index_cache)
    # Raw pointers go into the compiled call; these tensors own the storage.
    keepalive = (
        q,
        index_cache,
        block_table,
        request_indices,
        tile_meta,
        positions,
        scores,
    )
    # The TVM-FFI entry takes raw addresses and Python scalars; building
    # cutlass pointer and numeric objects here would cost ~11 us per call.
    args = (
        q.data_ptr(),
        index_cache.data_ptr(),
        block_table.data_ptr(),
        request_indices.data_ptr(),
        tile_meta.data_ptr(),
        positions.data_ptr(),
        scores.data_ptr(),
        num_requests,
        block_table.shape[1],
        max_blocks,
        index_cache.shape[0],
        index_cache.stride(0),
        rows,
        n_splits,
        head_dim**-0.5,
        grid_size,
    )

    def launch():
        if not only_visible_blocks:
            scores.fill_(float("-inf"))
        fn(*args, cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))
        return scores

    launch.keepalive = keepalive
    return launch, scores


def prims_ts_qsa_index_scores(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    *,
    tokens_per_block: int,
    compress_ratio: int,
    only_visible_blocks: bool = False,
) -> torch.Tensor:
    """Score compressed keys for each query row; same contract as the Triton scorer.

    Rows must be ordered by request. Columns at or past each row's causal block
    bound hold ``-inf`` unless ``only_visible_blocks``, which leaves them
    unspecified.
    """
    launch, _ = prepare_qsa_index_scores(
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        tokens_per_block=tokens_per_block,
        compress_ratio=compress_ratio,
        only_visible_blocks=only_visible_blocks,
    )
    return launch()
