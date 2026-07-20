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

_FAST_KNOB_ENV = "TLLM_SPARSE_KV_COMPACT_FAST"


@pytest.fixture(autouse=True, params=["0", "1"], ids=["existing_kernel", "fast_kernel"])
def sparse_compact_kernel_knob(request, monkeypatch):
    """Run every test in this module under both kernel selections.

    The C++ dispatcher re-reads TLLM_SPARSE_KV_COMPACT_FAST on every launch
    (temporary A/B knob), so flipping the process environment per test is
    enough; no re-import or subprocess is needed. Cases outside the fast-path
    gate (dtype, head_dim, or page size not covered) take the existing kernel
    under both values and simply run twice.
    """
    monkeypatch.setenv(_FAST_KNOB_ENV, request.param)
    return request.param


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
        # The kernel decodes K offsets as offset // 2 (the V2 2*page+plane
        # encoding) regardless of the scale the table was built with; the
        # reference mirrors the kernel, not the encoder.
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


def test_sparse_kv_cache_compact_layers_per_request_destination_bases():
    # One launch may mix pinned-prompt lengths: each request lands at its own
    # destination base.
    pools_cpu, pools, page_tables = _make_pools(2, torch.bfloat16, 64)
    page_tables_cpu = [page_table.cpu() for page_table in page_tables]
    source_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    source_row = torch.tensor([3, 5, 8, 6, 7, 10], dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    destination_bases = [2, 5]
    expected = _reference_compact(
        pools_cpu,
        page_tables_cpu,
        source_indices,
        source_offsets,
        destination_bases,
    )
    arguments = _device_arguments(pools, source_indices, source_offsets)

    _compact(pools, page_tables, arguments, destination_bases)
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


def test_sparse_kv_cache_compact_layers_rejects_flat_source_with_layer_indices():
    # A flat [kv_heads, total] source with per-layer indices would silently
    # read layer 0 for every launch; the op rejects the combination instead.
    _, pools, page_tables = _make_pools(2, torch.bfloat16, 64)
    source_offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
    source_row = torch.tensor([2, 5, 8, 3, 7, 10], dtype=torch.int32)
    source_indices = source_row.view(1, -1).expand(_NUM_KV_HEADS, -1).contiguous()
    source_layer_indices = torch.tensor([0, 0], dtype=torch.int32)
    arguments = _device_arguments(pools, source_indices, source_offsets, source_layer_indices)

    with pytest.raises(RuntimeError, match="require 3-D per-layer source_indices"):
        _compact(pools, page_tables, arguments, 0)


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


# --- Production-shaped geometry for the pipelined bf16 fast path ----------
#
# The fast path only dispatches for bf16 pools with head_dim 64/128 and
# 32/128-token pages, so the cases below use their own builder instead of the
# 4-token-page fixtures above.

_FAST_BATCH_SIZE = 3
# Per-request move counts: two ragged tiles plus a full one (pipeline steady
# state), an empty request (kernel early-return), and a single ragged tile
# (prologue/epilogue only).
_FAST_MOVE_COUNTS = (71, 0, 29)
# Mixed prompt lengths: none tile- or page-aligned.
_FAST_DESTINATION_BASES = [3, 9, 17]
# The production move-index buffers are allocation-wide: pad the head-plane
# width beyond this round's total move count so a kernel that derives the
# stride on device (instead of honoring the explicit sourceHeadStride) reads
# padding for heads > 0 and fails the byte-compare.
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

    # Deterministic per-geometry inputs: the cross-kernel test rebuilds the
    # identical case for each knob value.
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


# The full fast-path gate matrix. 128-token pages are the geometry the ported
# kernel was written for; 32-token pages and head_dim 128 are this tree's
# production configuration.
_FAST_GEOMETRY_MATRIX = [(64, 32), (128, 32), (64, 128), (128, 128)]


@pytest.mark.parametrize("head_dim,tokens_per_block", _FAST_GEOMETRY_MATRIX)
def test_sparse_kv_cache_compact_layers_fast_geometry(head_dim, tokens_per_block):
    # The autouse knob fixture runs this both through the pipelined fast path
    # and through the existing register-staging kernel.
    case = _make_fast_geometry_case(head_dim, tokens_per_block)
    expected = _run_fast_geometry_case(case)
    for actual, reference in zip(case.pools, expected):
        assert torch.equal(actual.cpu(), reference)


def test_sparse_kv_cache_compact_layers_fast_geometry_per_layer_source():
    case = _make_fast_geometry_case(64, 32, per_layer_sources=True)
    expected = _run_fast_geometry_case(case)
    for actual, reference in zip(case.pools, expected):
        assert torch.equal(actual.cpu(), reference)


@pytest.mark.parametrize(
    "dtype,head_dim,tokens_per_block",
    [
        (torch.float16, 64, 32),  # dtype outside the bf16-only gate
        (torch.bfloat16, 256, 32),  # head_dim outside the gate
        (torch.bfloat16, 64, 16),  # page size outside the gate
    ],
)
def test_sparse_kv_cache_compact_layers_fast_gate_fallback(dtype, head_dim, tokens_per_block):
    # Near-miss geometries must fall back to the existing kernel and stay
    # byte-correct under both knob values.
    case = _make_fast_geometry_case(head_dim, tokens_per_block, dtype=dtype)
    expected = _run_fast_geometry_case(case)
    for actual, reference in zip(case.pools, expected):
        assert torch.equal(actual.cpu(), reference)


@pytest.mark.parametrize("head_dim,tokens_per_block", _FAST_GEOMETRY_MATRIX)
def test_sparse_kv_cache_compact_layers_fast_path_actually_runs(
    head_dim, tokens_per_block, sparse_compact_kernel_knob
):
    # Guard against the fast-path gate silently never firing: every
    # byte-equality test in this module would still pass if the dispatcher
    # fell through to the existing kernel under both knob values. Assert via
    # the profiler that the selected kernel is the one that actually ran,
    # for each of the four static geometry dispatch branches.
    case = _make_fast_geometry_case(head_dim, tokens_per_block)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        _run_fast_geometry_case(case)
    names = [event.name for event in profiler.events()]
    fast_fired = any("sparseKvCacheCompactV2Bf16PipelineKernel" in name for name in names)
    existing_fired = any("updateSparseKvCacheAfterFmha" in name for name in names)
    if sparse_compact_kernel_knob == "1":
        assert fast_fired and not existing_fired
    else:
        assert existing_fired and not fast_fired


@pytest.mark.parametrize("head_dim,tokens_per_block", [(64, 32), (128, 128)])
def test_sparse_kv_cache_compact_layers_fast_matches_existing_kernel(
    head_dim, tokens_per_block, monkeypatch
):
    # The same inputs through both kernel selections must produce
    # byte-identical pools, and both must match the reference.
    outputs = {}
    expected = None
    for knob in ("0", "1"):
        monkeypatch.setenv(_FAST_KNOB_ENV, knob)
        case = _make_fast_geometry_case(head_dim, tokens_per_block)
        expected = _run_fast_geometry_case(case)
        outputs[knob] = [pool.cpu() for pool in case.pools]
    for existing_pool, fast_pool, reference in zip(outputs["0"], outputs["1"], expected):
        assert torch.equal(fast_pool, existing_pool)
        assert torch.equal(fast_pool, reference)
