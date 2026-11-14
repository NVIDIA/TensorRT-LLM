# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# PyTorch oracle ported from vLLM (Apache-2.0), _reference_decode_index_score:
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/tests/kernels/attention/test_minimax_m3.py#L188
"""The MSA index scorer of a sparse decode step: per (index head, query token, KV block) the causal max of Q . K over
the block's 128 index-K positions, with the pre-wait schedule of ``ops/m3_index_score``."""

from __future__ import annotations

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.attention.m3_index_score import (
    M3IndexScoreBase,
)

__all__ = ["m3_index_score"]

PAGE = 128
HEAD_DIM = 128


class M3IndexScore(M3IndexScoreBase):
    """``score[h, t, j]`` (fp32, a transposed view of the [heads, blocks, tokens] selector buffer) = the max over the
    positions ``p`` of block ``j`` of request ``b = t // qlen`` with ``p <= seq_lens[b] - qlen + t % qlen`` of
    ``idx_q[t, h] . index_k[block_table[b, j], p % 128]``, for ``j < ceil(seq_lens[b] / 128)`` (no other entry is
    written). FP8 or BF16 q and index-K, fp32 products and sums.

    ``early`` (the target's kernel at every request count it covers) issues the Q and first K tiles right after the
    grid wait from a pre-wait length hint and prefetches one older block per CTA into L2 before the wait; otherwise
    every older block of the CTA is prefetched. Neither, nor ``split_k``, changes a score.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request, 1 + 3 rows over 165521 tokens (AgentX ISL), 1 index head (TP8): 1536 CTAs, early issue",
            spec=dict(seq_lens=(165521,), qlen=4, heads=1, split_k=1536, early=True),
        ),
        Cell(
            why="one request of 129 tokens: the early tiles of most of the 1536 CTAs lie past the length and drain",
            spec=dict(seq_lens=(129,), qlen=4, heads=1, split_k=1536, early=True),
        ),
        Cell(
            why="one request of 400000 tokens: 3125 blocks, so the 1536 CTAs loop",
            spec=dict(seq_lens=(400000,), qlen=4, heads=1, split_k=1536, early=True),
        ),
        Cell(
            why="one request decoding one row over 65536 tokens",
            spec=dict(seq_lens=(65536,), qlen=1, heads=1, split_k=1536, early=True),
        ),
        Cell(
            why="one request, 1 + 3 rows over 165521 tokens, 1 index head (TP8): 1272 CTAs (one wave), early issue",
            spec=dict(seq_lens=(165521,), qlen=4, heads=1, split_k=1272, early=True),
        ),
        Cell(
            why="one request of 129 tokens at 1272 CTAs: the early tiles of most CTAs lie past the length and drain",
            spec=dict(seq_lens=(129,), qlen=4, heads=1, split_k=1272, early=True),
        ),
        Cell(
            why="one request of 400000 tokens at 1272 CTAs: 3125 blocks, so the CTAs loop",
            spec=dict(seq_lens=(400000,), qlen=4, heads=1, split_k=1272, early=True),
        ),
        Cell(
            why="two requests verifying 4 rows each over 165521 and 20000 tokens: 768 CTAs each, early issue",
            spec=dict(seq_lens=(165521, 20000), qlen=4, heads=1, split_k=768, early=True),
        ),
        Cell(
            why="two requests, one of 300 tokens and one of 262144 (2048 blocks: the 768 CTAs loop)",
            spec=dict(seq_lens=(300, 262144), qlen=4, heads=1, split_k=768, early=True),
        ),
    )

    note = """
    The products of FP8 (or BF16) values are exact in fp32 and the kernel sums 128 of them in its k-split order, so it
    differs from the reference matmul by a few fp32 ulps; it is held to the in-tree scorer bit for bit.
    """


m3_index_score = M3IndexScore()
