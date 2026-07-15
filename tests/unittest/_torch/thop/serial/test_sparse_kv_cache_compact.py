# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the layered V2 adapter over the existing sparse-KV updater."""

from typing import NamedTuple, Optional

import pytest
import torch

import tensorrt_llm  # noqa: F401  # Register torch.ops.trtllm operators.

_TOKENS_PER_BLOCK = 4
_NUM_KV_HEADS = 2
_BATCH_SIZE = 2
_MAX_PAGES_PER_SEQUENCE = 3
_NUM_PAGES = _BATCH_SIZE * _MAX_PAGES_PER_SEQUENCE
_PAGE_INDEX_DIVISOR = 2


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
    num_pages = _NUM_PAGES * page_index_scale // _PAGE_INDEX_DIVISOR
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
    destination_base: int,
    source_layer_indices: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    original = [pool.clone() for pool in pools]
    expected = [pool.clone() for pool in pools]
    for group_layer, (source_pool, destination_pool, page_table) in enumerate(
        zip(original, expected, page_tables)
    ):
        raw_page_table = page_table // _PAGE_INDEX_DIVISOR
        if source_indices.ndim == 2:
            layer_sources = source_indices
        else:
            assert source_layer_indices is not None
            layer_sources = source_indices[int(source_layer_indices[group_layer])]
        for request in range(_BATCH_SIZE):
            begin = int(source_offsets[request])
            end = int(source_offsets[request + 1])
            for head in range(_NUM_KV_HEADS):
                for request_move, global_move in enumerate(range(begin, end)):
                    source_token = int(layer_sources[head, global_move])
                    destination_token = destination_base + request_move
                    source_page = int(raw_page_table[request, source_token // _TOKENS_PER_BLOCK])
                    destination_page = int(
                        raw_page_table[request, destination_token // _TOKENS_PER_BLOCK]
                    )
                    destination_pool[
                        destination_page,
                        :,
                        head,
                        destination_token % _TOKENS_PER_BLOCK,
                        :,
                    ] = source_pool[
                        source_page,
                        :,
                        head,
                        source_token % _TOKENS_PER_BLOCK,
                        :,
                    ]
    return expected


def _compact(
    pools: list[torch.Tensor],
    page_tables: list[torch.Tensor],
    arguments: _DeviceArguments,
    destination_base: int,
) -> None:
    torch.ops.trtllm.sparse_kv_cache_compact_layers(
        pools,
        arguments.pool_pointers,
        page_tables[0],
        arguments.source_indices,
        arguments.source_offsets,
        arguments.source_layer_indices,
        destination_base,
    )


@pytest.mark.parametrize(
    "dtype,head_dim",
    [
        (torch.float16, 16),
        (torch.bfloat16, 32),
        (torch.float32, 64),
        (torch.float16, 128),
        (torch.bfloat16, 256),
        (torch.float32, 256),
    ],
)
@pytest.mark.parametrize(
    "destination_base,page_index_scale",
    [(0, 2), (2, 2), (2, 4)],
)
def test_sparse_kv_cache_compact_layers(dtype, head_dim, destination_base, page_index_scale):
    pools_cpu, pools, page_tables = _make_pools(3, dtype, head_dim, page_index_scale)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    source_row = torch.tensor([2, 5, 8, 3, 7, 10], dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
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


def test_sparse_kv_cache_compact_layers_per_layer_source():
    pools_cpu, pools, page_tables = _make_pools(2, torch.bfloat16, 64)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    source_indices = torch.tensor(
        [
            [[2, 5, 8, 3, 7, 10], [3, 6, 9, 2, 5, 8]],
            [[3, 7, 10, 2, 6, 9], [2, 5, 9, 3, 6, 10]],
            [[4, 7, 9, 3, 6, 8], [3, 5, 8, 4, 7, 10]],
        ],
        dtype=torch.int32,
    )
    source_layer_indices = torch.tensor([2, 0], dtype=torch.int32)
    destination_base = 2
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        source_indices,
        source_offsets,
        destination_base,
        source_layer_indices,
    )
    arguments = _device_arguments(
        pools,
        source_indices,
        source_offsets,
        source_layer_indices,
    )

    _compact(pools, page_tables, arguments, destination_base)
    torch.cuda.synchronize()

    for actual, reference in zip(pools, expected):
        assert torch.equal(actual.cpu(), reference)


def test_sparse_kv_cache_compact_layers_multiple_tiles():
    num_layers = 2
    max_pages_per_sequence = 24
    num_pages = _BATCH_SIZE * max_pages_per_sequence
    shape = (num_pages, 2, _NUM_KV_HEADS, _TOKENS_PER_BLOCK, 64)
    numel = torch.Size(shape).numel()
    pools_cpu = [
        ((torch.arange(numel, dtype=torch.int32) + layer * 37) % 251)
        .reshape(shape)
        .to(torch.bfloat16)
        for layer in range(num_layers)
    ]
    pools = [pool.cuda() for pool in pools_cpu]
    raw_page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").reshape(
        _BATCH_SIZE, max_pages_per_sequence
    )
    page_table = _encode_k_block_offsets(raw_page_table)
    page_tables = [page_table] * num_layers
    assert page_tables[0].stride(0) == 2 * max_pages_per_sequence
    page_tables_cpu = [table.cpu() for table in page_tables]
    source_offsets = torch.tensor([0, 40, 75], dtype=torch.int32)
    source_row = torch.cat((torch.arange(40, 80), torch.arange(36, 71))).to(torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    destination_base = 2
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
    """Check operation-level capture safety, not a standalone TriAttention graph."""
    pools_cpu, pools, page_tables = _make_pools(3, torch.bfloat16, 64)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    source_row = torch.tensor([2, 5, 8, 3, 7, 10], dtype=torch.int32)
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
