# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the layered V2 sparse-KV compaction op (pipelined bf16 kernels)."""

from typing import NamedTuple, Optional

import pytest
import torch

import tensorrt_llm  # noqa: F401  # Register torch.ops.trtllm operators.

_TOKENS_PER_BLOCK = 32
_NUM_KV_HEADS = 2
_BATCH_SIZE = 2
_PAGE_INDEX_DIVISOR = 2

# Profiler probe names: the pipelined bf16 kernels are the only shipped
# path; the retired register-staging kernel must never appear.
_FAST_KERNEL_NAME = "sparseKvCacheCompactV2Bf16PipelineKernel"
_RETIRED_KERNEL_NAME = "updateSparseKvCacheAfterFmha"


def _encode_k_block_offsets(
    page_table: torch.Tensor, page_index_scale: int = _PAGE_INDEX_DIVISOR
) -> torch.Tensor:
    encoded = torch.empty(
        page_table.shape[0],
        2,
        page_table.shape[1],
        dtype=torch.int32,
        device=page_table.device,
    )
    encoded[:, 0] = page_table * page_index_scale
    encoded[:, 1] = encoded[:, 0] + 1
    return encoded[:, 0]


class _DeviceArguments(NamedTuple):
    pool_pointers: torch.Tensor
    source_indices: torch.Tensor
    source_offsets: torch.Tensor
    source_layer_indices: Optional[torch.Tensor]


def _make_pools(
    num_layers: int,
    dtype: torch.dtype,
    head_dim: int,
    page_index_scale: int = _PAGE_INDEX_DIVISOR,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    pages_per_seq = 3
    num_pages = _BATCH_SIZE * pages_per_seq * page_index_scale // _PAGE_INDEX_DIVISOR
    shape = (
        num_pages,
        2,
        _NUM_KV_HEADS,
        _TOKENS_PER_BLOCK,
        head_dim,
    )
    numel = torch.Size(shape).numel()
    pools_cpu = [
        ((torch.arange(numel, dtype=torch.int32) + layer * 37) % 251).reshape(shape).to(dtype)
        for layer in range(num_layers)
    ]
    pools = [pool.cuda() for pool in pools_cpu]
    raw_pages = [[4, 1, 5], [2, 0, 3]]
    assert set(raw_pages[0]).isdisjoint(raw_pages[1])
    raw_page_table = torch.tensor(raw_pages, dtype=torch.int32, device="cuda")
    page_table = _encode_k_block_offsets(raw_page_table, page_index_scale)
    page_tables = [page_table] * num_layers
    assert page_tables[0].stride(0) == 2 * page_tables[0].shape[1]
    return pools_cpu, pools, page_tables


def _device_arguments(
    pools: list[torch.Tensor],
    source_indices: torch.Tensor,
    source_offsets: torch.Tensor,
    source_layer_indices: Optional[torch.Tensor] = None,
) -> _DeviceArguments:
    device = pools[0].device
    return _DeviceArguments(
        pool_pointers=torch.tensor(
            [pool.data_ptr() for pool in pools], dtype=torch.int64, device=device
        ),
        source_indices=source_indices.to(device),
        source_offsets=source_offsets.to(device),
        source_layer_indices=(
            None if source_layer_indices is None else source_layer_indices.to(device)
        ),
    )


def _reference_compact(
    pools: list[torch.Tensor],
    page_tables: list[torch.Tensor],
    source_indices: torch.Tensor,
    source_offsets: torch.Tensor,
    destination_base: "int | list[int]",
    source_layer_indices: Optional[torch.Tensor] = None,
    tokens_per_block: int = _TOKENS_PER_BLOCK,
    batch_size: int = _BATCH_SIZE,
) -> list[torch.Tensor]:
    original = [pool.clone() for pool in pools]
    expected = [pool.clone() for pool in pools]
    for group_layer, (source_pool, destination_pool, page_table) in enumerate(
        zip(original, expected, page_tables)
    ):
        # The kernel decodes K offsets as offset // 2 regardless of the
        # encoder's scale; the reference mirrors the kernel.
        raw_page_table = page_table // _PAGE_INDEX_DIVISOR
        if source_indices.ndim == 2:
            layer_sources = source_indices
        else:
            assert source_layer_indices is not None
            layer_sources = source_indices[int(source_layer_indices[group_layer])]
        for request in range(batch_size):
            begin = int(source_offsets[request])
            end = int(source_offsets[request + 1])
            request_base = (
                destination_base[request]
                if isinstance(destination_base, (list, tuple))
                else destination_base
            )
            for head in range(layer_sources.shape[0]):
                for request_move, global_move in enumerate(range(begin, end)):
                    source_token = int(layer_sources[head, global_move])
                    destination_token = request_base + request_move
                    source_page = int(raw_page_table[request, source_token // tokens_per_block])
                    destination_page = int(
                        raw_page_table[request, destination_token // tokens_per_block]
                    )
                    destination_pool[
                        destination_page,
                        :,
                        head,
                        destination_token % tokens_per_block,
                        :,
                    ] = source_pool[
                        source_page,
                        :,
                        head,
                        source_token % tokens_per_block,
                        :,
                    ]
    return expected


def _compact(
    pools: list[torch.Tensor],
    page_tables: list[torch.Tensor],
    arguments: _DeviceArguments,
    destination_base: "int | list[int]",
    batch_size: int = _BATCH_SIZE,
) -> None:
    # The op takes per-request destination bases; scalar test parameters are
    # broadcast to the batch here. torch.full stays CUDA-graph-capturable.
    if isinstance(destination_base, int):
        destination_bases = torch.full(
            (batch_size,), destination_base, dtype=torch.int32, device="cuda"
        )
    else:
        destination_bases = torch.tensor(destination_base, dtype=torch.int32, device="cuda")
    torch.ops.trtllm.sparse_kv_cache_compact_layers(
        pools,
        arguments.pool_pointers,
        page_tables[0],
        arguments.source_indices,
        arguments.source_offsets,
        destination_bases,
        arguments.source_layer_indices,
    )


_SMALL_ROW = [2, 5, 8, 3, 7, 10]
# The production-shaped fast-geometry matrix below is the byte-exact anchor
# (both head dims, both page sizes, per-request destination bases, 3-D
# per-layer routing, multi-tile ragged moves). These two rows keep the
# op-level contracts it does not pin: the destination-base-0 (prompt 0)
# boundary and the fixed //2 K-offset decode against a scale-4 encoder.
_LAYER_CASES = [
    pytest.param(dict(head_dim=64, dest=0, scale=2), id="bf16_h64_dest0_scale2"),
    pytest.param(dict(head_dim=64, dest=2, scale=4), id="bf16_h64_dest2_scale4"),
]


@pytest.mark.parametrize("case", _LAYER_CASES)
def test_sparse_kv_cache_compact_layers(case):
    pools_cpu, pools, page_tables = _make_pools(3, torch.bfloat16, case["head_dim"], case["scale"])
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor((0, 3, 6), dtype=torch.int32)
    source_row = torch.tensor(_SMALL_ROW, dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    destination_base = case["dest"]
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        source_indices,
        source_offsets,
        destination_base,
    )
    arguments = _device_arguments(pools, source_indices, source_offsets)

    _compact(pools, page_tables, arguments, destination_base)
    torch.cuda.synchronize()

    for actual, reference in zip(pools, expected):
        assert torch.equal(actual.cpu(), reference)


def test_sparse_kv_cache_compact_layers_cuda_graph_replay():
    """Check operation-level capture safety inside an externally captured graph."""
    pools_cpu, pools, page_tables = _make_pools(3, torch.bfloat16, 64)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    source_row = torch.tensor(_SMALL_ROW, dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    replay_row = torch.tensor([3, 6, 9, 2, 5, 8], dtype=torch.int32)
    replay_indices = replay_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    destination_base = 2
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        replay_indices,
        source_offsets,
        destination_base,
    )
    arguments = _device_arguments(pools, source_indices, source_offsets)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _compact(pools, page_tables, arguments, destination_base)

    for pool, initial in zip(pools, pools_cpu):
        pool.copy_(initial)
    arguments.source_indices.copy_(replay_indices)
    graph.replay()
    torch.cuda.synchronize()

    for actual, reference in zip(pools, expected):
        assert torch.equal(actual.cpu(), reference)


# --- Production-shaped geometry for the pipelined bf16 fast path ----------

_FAST_BATCH_SIZE = 3
# Ragged+full tiles, an empty request, and a prologue/epilogue-only tile.
_FAST_MOVE_COUNTS = (71, 0, 29)
# Mixed prompt lengths: none tile- or page-aligned.
_FAST_DESTINATION_BASES = [3, 9, 17]
# Allocation-wide index buffers: padding past the round's total move count
# makes a device-derived stride read padding and fail the byte-compare.
_FAST_SOURCE_PAD = 37
_FAST_IDENTITY_MOVES = 5


class _FastGeometryCase(NamedTuple):
    pools_cpu: list[torch.Tensor]
    pools: list[torch.Tensor]
    page_tables_cpu: list[torch.Tensor]
    page_tables: list[torch.Tensor]
    source_indices: torch.Tensor
    source_offsets: torch.Tensor
    source_layer_indices: Optional[torch.Tensor]
    destination_bases: list[int]
    tokens_per_block: int
    batch_size: int


def _fast_sources_row(
    base: int, count: int, limit: int, generator: torch.Generator
) -> torch.Tensor:
    """Distinct sorted source tokens >= base, so src(i) >= base + i (the op's
    in-place-safety contract). The first few moves are identities (src == dst),
    which the kernel skips storing."""
    if count == 0:
        return torch.empty(0, dtype=torch.int32)
    identity = min(_FAST_IDENTITY_MOVES, count)
    candidates = torch.arange(base + identity, limit, dtype=torch.int32)
    picks = torch.randperm(candidates.numel(), generator=generator)[: count - identity]
    tail = candidates[picks].sort().values
    return torch.cat((torch.arange(base, base + identity, dtype=torch.int32), tail))


def _make_fast_geometry_case(
    head_dim: int,
    tokens_per_block: int,
    dtype: torch.dtype = torch.bfloat16,
    num_layers: int = 2,
    per_layer_sources: bool = False,
) -> _FastGeometryCase:
    batch_size = _FAST_BATCH_SIZE
    pages_per_seq = max(2, 128 // tokens_per_block)
    tokens_per_seq = pages_per_seq * tokens_per_block
    num_pages = batch_size * pages_per_seq
    shape = (num_pages, 2, _NUM_KV_HEADS, tokens_per_block, head_dim)
    numel = torch.Size(shape).numel()
    pools_cpu = [
        ((torch.arange(numel, dtype=torch.int32) + layer * 37) % 251).reshape(shape).to(dtype)
        for layer in range(num_layers)
    ]
    pools = [pool.cuda() for pool in pools_cpu]

    # Deterministic per-geometry inputs so any failure reproduces exactly.
    generator = torch.Generator().manual_seed(20260720 + head_dim * 1000 + tokens_per_block)
    raw_page_table = (
        torch.randperm(num_pages, generator=generator)
        .to(torch.int32)
        .reshape(batch_size, pages_per_seq)
        .cuda()
    )
    page_table = _encode_k_block_offsets(raw_page_table)
    page_tables = [page_table] * num_layers
    assert page_tables[0].stride(0) == 2 * pages_per_seq
    page_tables_cpu = [table.cpu() for table in page_tables]

    offsets = [0]
    for count in _FAST_MOVE_COUNTS:
        offsets.append(offsets[-1] + count)
    source_offsets = torch.tensor(offsets, dtype=torch.int32)
    width = offsets[-1] + _FAST_SOURCE_PAD
    source_layers = 3 if per_layer_sources else 1
    # Padding is a valid token id so a stride bug corrupts output (caught by
    # the byte-compare) instead of faulting.
    rows = torch.full((source_layers, _NUM_KV_HEADS, width), tokens_per_seq - 1, dtype=torch.int32)
    for layer in range(source_layers):
        for head in range(_NUM_KV_HEADS):
            cursor = 0
            for request, count in enumerate(_FAST_MOVE_COUNTS):
                rows[layer, head, cursor : cursor + count] = _fast_sources_row(
                    _FAST_DESTINATION_BASES[request], count, tokens_per_seq, generator
                )
                cursor += count
    if per_layer_sources:
        source_indices = rows.contiguous()
        source_layer_indices = torch.tensor([2, 0], dtype=torch.int32)
    else:
        source_indices = rows[0].contiguous()
        source_layer_indices = None

    return _FastGeometryCase(
        pools_cpu=pools_cpu,
        pools=pools,
        page_tables_cpu=page_tables_cpu,
        page_tables=page_tables,
        source_indices=source_indices,
        source_offsets=source_offsets,
        source_layer_indices=source_layer_indices,
        destination_bases=list(_FAST_DESTINATION_BASES),
        tokens_per_block=tokens_per_block,
        batch_size=batch_size,
    )


def _run_fast_geometry_case(case: _FastGeometryCase) -> list[torch.Tensor]:
    expected = _reference_compact(
        case.pools_cpu,
        case.page_tables_cpu,
        case.source_indices,
        case.source_offsets,
        case.destination_bases,
        case.source_layer_indices,
        tokens_per_block=case.tokens_per_block,
        batch_size=case.batch_size,
    )
    arguments = _device_arguments(
        case.pools, case.source_indices, case.source_offsets, case.source_layer_indices
    )
    _compact(
        case.pools,
        case.page_tables,
        arguments,
        case.destination_bases,
        batch_size=case.batch_size,
    )
    torch.cuda.synchronize()
    return expected


# The full fast-path gate matrix; the per-layer-source row keeps the 3-D
# routing path runnable through the fast kernel.
_FAST_GEOMETRY_MATRIX = [(64, 32), (128, 32), (64, 128), (128, 128)]


@pytest.mark.parametrize(
    "head_dim,tokens_per_block,per_layer",
    [(h, t, False) for h, t in _FAST_GEOMETRY_MATRIX] + [(64, 32, True)],
)
def test_sparse_kv_cache_compact_layers_fast_geometry(head_dim, tokens_per_block, per_layer):
    # Byte-compare against the CPU reference, plus the dispatch probe: the
    # pipelined kernel must actually run (and the retired one must not) --
    # every byte-equality here would still pass on a silent fallback.
    case = _make_fast_geometry_case(head_dim, tokens_per_block, per_layer_sources=per_layer)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        expected = _run_fast_geometry_case(case)
    names = [event.name for event in profiler.events()]
    assert any(_FAST_KERNEL_NAME in name for name in names)
    assert not any(_RETIRED_KERNEL_NAME in name for name in names)
    for actual, reference in zip(case.pools, expected):
        assert torch.equal(actual.cpu(), reference)


# One representative per reject family; no fallback kernel exists, so every
# reject must fail loudly and leave the pools untouched.
@pytest.mark.parametrize(
    "dtype,head_dim,tokens_per_block,flat_with_layer_indices,match",
    [
        pytest.param(torch.float16, 64, 32, False, "bf16|BF16", id="dtype_outside_gate"),
        pytest.param(torch.bfloat16, 256, 32, False, "bf16|BF16", id="head_dim_outside_gate"),
        pytest.param(torch.bfloat16, 64, 16, False, "bf16|BF16", id="page_size_outside_gate"),
        pytest.param(
            torch.bfloat16,
            64,
            32,
            True,
            "require 3-D per-layer source_indices",
            id="flat_source_with_layer_indices",
        ),
    ],
)
def test_sparse_kv_cache_compact_layers_rejects_invalid_launch(
    dtype, head_dim, tokens_per_block, flat_with_layer_indices, match
):
    case = _make_fast_geometry_case(head_dim, tokens_per_block, dtype=dtype)
    source_layer_indices = case.source_layer_indices
    if flat_with_layer_indices:
        source_layer_indices = torch.tensor([0, 0], dtype=torch.int32)
    arguments = _device_arguments(
        case.pools, case.source_indices, case.source_offsets, source_layer_indices
    )
    with pytest.raises((RuntimeError, ValueError), match=match):
        _compact(
            case.pools,
            case.page_tables,
            arguments,
            case.destination_bases,
            batch_size=case.batch_size,
        )
    torch.cuda.synchronize()
    for actual, reference in zip(case.pools, case.pools_cpu):
        assert torch.equal(actual.cpu(), reference)
