# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 index decode scoring split across two TP ranks (index_tp_split)."""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.index_tp_split import IndexTpSplit
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)


@torch.inference_mode()
def _run_rank(world: int) -> bool:
    # Imported here: the op object does not pickle with this by-value module.
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    # The same inputs on every rank.
    g = torch.Generator(device="cuda").manual_seed(0)
    dql, max_blocks = 4, 72
    seq_lens = torch.tensor([4, 129, 1025, 4097, 9001], dtype=torch.int32, device="cuda")
    block_table = torch.randperm(5 * max_blocks, generator=g, device="cuda").int().view(5, -1)
    idx_q = torch.randn(5 * dql, 4, 128, generator=g, device="cuda").to(torch.float8_e4m3fn)
    cache = torch.randn(5 * max_blocks, 1, 128, 128, generator=g, device="cuda").to(
        torch.float8_e4m3fn
    )
    expected = torch.full((4, max_blocks, 5 * dql), float("nan"), device="cuda")
    cute_dsl_custom_ops.cute_dsl_minimax_m3_index_decode_score(
        idx_q, cache[:, 0], block_table, seq_lens, expected.transpose(1, 2), dql
    )
    mapping = Mapping(world_size=world, rank=rank, tp_size=world)
    split = IndexTpSplit(mapping, range(5), 4, max_blocks, 5 * dql)
    # Every slot set, the wrap from the last layer to the first and set reuse, back to back.
    scores = [torch.full_like(expected[split.heads], float("nan")) for _ in range(15)]
    scores = [
        split.score(i % 5, idx_q, cache, block_table, seq_lens, got, dql)
        for i, got in enumerate(scores)
    ]
    torch.cuda.synchronize()
    return all(
        torch.equal(got.view(torch.int32), expected[split.heads].view(torch.int32))
        for got in scores
    )


@pytest.mark.skip_less_device(2)
def test_minimax_m3_index_tp_split_matches_unsplit() -> None:
    """Two ranks scoring half the index-K rows each give the unsplit scores bit for bit."""
    if get_sm_version() not in (100, 103):
        pytest.skip("the MiniMax-M3 index scorer needs SM100/SM103")
    with MPIPoolExecutor(max_workers=2) as executor:
        assert all(executor.map(_run_rank, [2, 2]))
