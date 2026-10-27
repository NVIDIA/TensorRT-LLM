# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/m3_index_score_t2sc entry (T2SC, dispatch.INDEX_SCORE_T2SC): the TP2 / TP8 MSA index
scorer with its pre-wait schedule, as the TP4 target's copy at its 1 index head per rank.

Every cell is held to the fp32 reference (a few fp32 ulps: the kernel sums the 128 exact products in its k-split order)
and to the in-tree scorer (``cute_dsl_minimax_m3_index_decode_score`` at its own 256-CTA split) bit for bit, which the
pre-wait prefetch, the early tile issue and the cell's split must not change.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.attention.m3_index_score_t2sc import (  # noqa: E501
    PAGE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp4.ops.attention.m3_index_score_t2sc import (  # noqa: E501
    m3_index_score_t2sc as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_index_score requires a CUDA device"

HEADS = 1  # index heads per rank at TP4 (4 index heads over 4 KV groups)
HEAD_DIM = 128


def _inputs(seq_lens, qlen: int, seed: int, dtype=torch.float8_e4m3fn):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    max_blocks = (max(seq_lens) + PAGE - 1) // PAGE
    stride = ((max_blocks + 15) // 16) * 16
    pages = batch * max_blocks
    block_table = (
        torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(batch, max_blocks)
    )
    idx_q = torch.randn(batch * qlen, HEADS, HEAD_DIM, device="cuda", generator=g).to(dtype)
    index_k = torch.randn(pages, PAGE, HEAD_DIM, device="cuda", generator=g).to(dtype)
    seq = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return idx_q, index_k, block_table, seq, stride


def _score(stride: int, total_q: int):
    backing = torch.full(
        (HEADS, stride, total_q), -float("inf"), device="cuda", dtype=torch.float32
    )
    return backing.transpose(1, 2)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    qlen = spec["qlen"]
    idx_q, index_k, block_table, seq, stride = _inputs(
        spec["seq_lens"], qlen, zlib.crc32(cell.why.encode())
    )
    score = _score(stride, idx_q.shape[0])
    assert spec["heads"] == HEADS
    args = (idx_q, index_k, block_table, seq, score, qlen, spec["split_k"], spec["early"])
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    stock = _score(stride, idx_q.shape[0])
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, index_k, block_table, seq, stock, qlen
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(score, ref, rtol=1e-5, atol=1e-4)
    assert torch.equal(score, stock), "the scores differ from the in-tree scorer"


@pytest.mark.parametrize("early", [True, False], ids=["early", "prefetch"])
def test_bf16_index_is_the_in_tree_scorer(early) -> None:
    idx_q, index_k, block_table, seq, stride = _inputs((5000,), 4, 11, dtype=torch.bfloat16)
    score, stock = _score(stride, 4), _score(stride, 4)
    op(idx_q, index_k, block_table, seq, score, 4, 1536, early)
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, index_k, block_table, seq, stock, 4
    )
    assert torch.equal(score, stock)


@pytest.mark.parametrize("qlen", [1, 2, 3, 4])
def test_prestaged_blocks_at_the_stable_edge_are_the_in_tree_scorer(qlen) -> None:
    """Lengths around the edge between pre-staged (all-old) and post-wait first blocks, 1 and 2 requests."""
    for k in (1, 2, 1294):
        for length in (k * PAGE - qlen - 1, k * PAGE - qlen, k * PAGE - qlen + 1, k * PAGE + 1):
            for seq_lens, split in (
                ((length,), 1536),
                ((length,), 1272),
                ((length, max(qlen, length // 3)), 768),
            ):
                idx_q, index_k, block_table, seq, stride = _inputs(seq_lens, qlen, 31 + length)
                score, stock = _score(stride, idx_q.shape[0]), _score(stride, idx_q.shape[0])
                op(idx_q, index_k, block_table, seq, score, 4, split, True)
                torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
                    idx_q, index_k, block_table, seq, stock, qlen
                )
                assert torch.equal(score, stock), (qlen, seq_lens)


def test_guard_refuses_a_request_past_its_block_table() -> None:
    idx_q, index_k, block_table, seq, stride = _inputs((1000,), 1, 5)
    seq = seq + 4 * PAGE
    with pytest.raises(AssertionError, match="past its block-table row"):
        with validating(op):
            op(idx_q, index_k, block_table, seq, _score(stride + 16, 1), 1, 1536, True)
