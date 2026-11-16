# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm107 context-step reduce-scatter overlaps (``kernels/context_sequence_parallel/overlap.py``).

The levers ``context_sp_oproj_overlap``, ``context_sp_tail_pipeline`` and
``context_sp_moe_rs_overlap`` are bitwise exactly where the GEMMs they run on row chunks
compute every row as the whole call does, and NCCL reduces a chunk's elements as the whole
buffer's. These tests pin that on the device they run on:

* one GPU: the BF16 projections the levers chunk -- the output projection over owner-major
  packed rows, the pooler and the unpooler over shard rows -- through the served ``Linear``
  module, bit for bit against the whole call, at the step and chunk counts the levers take;
  and the router-input join (``router_input.join_with_router_input``) into row slices of
  whole-shard buffers, bit for bit against the whole-shard join;
* four ranks (``mpirun -n 4 python -m pytest ...``): ``overlap.reduce_scatter_unpool`` against
  ``unpool(reducescatter(...))``, eager and replayed from a CUDA graph, bit for bit.

The output projection's own chunked reduce-scatter is the built-in
``tests/unittest/_torch/modeling/test_musespark_open_sequence_parallel.py::
test_overlapped_projection_chunks_bitwise`` (four ranks).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_sequence_parallel import (  # noqa: E501
    overlap,
    router_input,
)
from tensorrt_llm._torch.modules.linear import Linear

WORLD = 4
HIDDEN = 8192
EXPERT_HIDDEN = 4096
#: Columns of the output projection's input on one TP4 rank: 32 heads of 64.
OPROJ_IN = 2048


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


def _linear(in_features: int, out_features: int, seed: int) -> Linear:
    torch.manual_seed(seed)
    module = Linear(in_features, out_features, bias=False, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        module.weight.copy_(torch.randn_like(module.weight) / 32)
    return module


@pytest.mark.skipif(not _sm107(), reason="the sm107 context path runs on SM107")
@pytest.mark.parametrize("rows", [8192, 4096])
@pytest.mark.parametrize("chunks", [2, 3, 4])
@torch.no_grad()
def test_the_chunked_projections_are_the_whole_calls_rows(rows: int, chunks: int) -> None:
    shard = rows // WORLD
    bounds = overlap.chunk_bounds(shard, chunks)
    pairs = list(zip(bounds[:-1], bounds[1:]))
    # The output projection: the step's rows, packed owner-major per chunk.
    o_proj = _linear(OPROJ_IN, HIDDEN, 17)
    attn = torch.randn(rows, OPROJ_IN, device="cuda", dtype=torch.bfloat16)
    whole = o_proj(attn)
    for lo, hi in pairs:
        index = torch.cat(
            [torch.arange(r * shard + lo, r * shard + hi, device="cuda") for r in range(WORLD)]
        )
        packed = attn.index_select(0, index).contiguous()
        assert torch.equal(o_proj(packed), whole.index_select(0, index)), (lo, hi)
    # The pooler and the unpooler: the shard's rows, chunk by chunk.
    shapes = ((HIDDEN, EXPERT_HIDDEN, 23), (EXPERT_HIDDEN, HIDDEN, 29))
    for in_features, out_features, seed in shapes:
        module = _linear(in_features, out_features, seed)
        x = torch.randn(shard, in_features, device="cuda", dtype=torch.bfloat16)
        whole = module(x)
        for lo, hi in pairs:
            assert torch.equal(module(x[lo:hi].contiguous()), whole[lo:hi]), (
                in_features,
                out_features,
                lo,
                hi,
            )


@pytest.mark.skipif(not _sm107(), reason="the sm107 context path runs on SM107")
@pytest.mark.parametrize("chunks", [1, 2, 3, 4])
@torch.no_grad()
def test_the_router_input_join_into_row_slices_is_the_whole_join(chunks: int) -> None:
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    torch.manual_seed(31)
    shard = 2048
    join = MuseSparkOpenGatedResidual(HIDDEN, torch.bfloat16).cuda()
    join.gate.copy_(torch.randn_like(join.gate))
    join.cache_derived_state()
    norm = RMSNorm(hidden_size=HIDDEN, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
    norm.weight.copy_(torch.randn_like(norm.weight) / 8)
    residual = torch.randn(shard, HIDDEN, device="cuda", dtype=torch.float32)
    branch = torch.randn(shard, HIDDEN, device="cuda", dtype=torch.bfloat16)
    whole = router_input.join_with_router_input(join, residual, branch, 1e-6, norm)
    assert whole is not None, "the fused join declined a 2048-row shard"
    updated = torch.empty_like(residual)
    next_input = torch.empty_like(branch)
    router = torch.empty_like(residual)
    bounds = overlap.chunk_bounds(shard, chunks)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = slice(lo, hi)
        # The tail pipeline joins a reduced chunk: its own tensor, not a slice of the branch.
        joined = router_input.join_with_router_input(
            join,
            residual[rows],
            branch[rows].clone(),
            1e-6,
            norm,
            None,
            out=(updated[rows], next_input[rows], router[rows]),
        )
        assert joined is not None, (lo, hi)
    for got, want in zip((updated, next_input, router), whole):
        assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunks", [1, 2, 3, 4])
@pytest.mark.parametrize("graph", [False, True])
@torch.no_grad()
def test_the_chunked_experts_reduce_scatter_is_the_whole_one(chunks: int, graph: bool) -> None:
    from tensorrt_llm._torch.distributed import reducescatter
    from tensorrt_llm._torch.models.modeling_musespark_open import _pad_sequence_parallel_tokens
    from tensorrt_llm._utils import mpi_rank, mpi_world_size
    from tensorrt_llm.mapping import Mapping

    if mpi_world_size() != WORLD:
        pytest.skip("Requires four MPI ranks")
    rank = mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=WORLD, rank=rank, tp_size=WORLD, moe_ep_size=WORLD)
    unpool = _linear(EXPERT_HIDDEN, HIDDEN, 41)  # the same weight on every rank
    stream = overlap._stream(high_priority=True)
    torch.manual_seed(1103 + rank)
    for rows in (8192, 8193, 16384):
        combined = _pad_sequence_parallel_tokens(
            torch.randn(rows, EXPERT_HIDDEN, device="cuda", dtype=torch.bfloat16), WORLD
        )
        expected = unpool(reducescatter(combined, mapping, dim=0))
        if not graph:
            actual = overlap.reduce_scatter_unpool(combined, mapping, chunks, stream, unpool)
        else:
            static = combined.clone()
            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):
                overlap.reduce_scatter_unpool(static, mapping, chunks, stream, unpool)
            torch.cuda.current_stream().wait_stream(warm)
            torch.cuda.synchronize()
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                captured = overlap.reduce_scatter_unpool(static, mapping, chunks, stream, unpool)
            static.copy_(torch.randn_like(static))
            cuda_graph.replay()
            static.copy_(combined)
            cuda_graph.replay()
            torch.cuda.synchronize()
            actual = captured
        torch.cuda.synchronize()
        assert actual.shape == (combined.shape[0] // WORLD, HIDDEN)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_the_chunk_bounds_split_every_shard_alike() -> None:
    assert overlap.chunk_bounds(2048, 2) == [0, 1024, 2048]
    assert overlap.chunk_bounds(2049, 2) == [0, 1025, 2049]
    assert overlap.chunk_bounds(2048, 3) == [0, 683, 1366, 2048]
    assert overlap.chunk_bounds(3, 4) == [0, 1, 2, 3]
    assert overlap.chunk_bounds(256, 4) == [0, 64, 128, 192, 256]
