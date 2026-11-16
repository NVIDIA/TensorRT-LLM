# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# PyTorch oracle ported from vLLM (Apache-2.0), _reference_decode_index_score:
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/tests/kernels/attention/test_minimax_m3.py#L188
"""The MSA index scorer of a sparse decode step: per (index head, query token, KV block) the causal max of Q . K over
the block's 128 index-K positions, with the pre-wait schedule of ``ops/m3_index_score``."""

from __future__ import annotations

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.m3_index_score import (
    M3IndexScoreBase,
)

__all__ = ["m3_index_score"]

PAGE = 128
HEAD_DIM = 128
#: CTAs per request (each scores blocks split, split + SPLIT_K, ...; the scores do not depend on it).
SPLIT_K = 512


class M3IndexScore(M3IndexScoreBase):
    """``score[h, t, j]`` (fp32, a transposed view of the [heads, blocks, tokens] selector buffer) = the max over the
    positions ``p`` of block ``j`` of request ``b = t // qlen`` with ``p <= seq_lens[b] - qlen + t % qlen`` of
    ``idx_q[t, h] . index_k[block_table[b, j], p % 128]``, for ``j < ceil(seq_lens[b] / 128)`` (no other entry is
    written). FP8 or BF16 q and index-K, fp32 products and sums.

    ``early`` (the target's one-request kernel) issues the Q and first K tiles right after the grid wait from a
    pre-wait length hint and prefetches one older block per CTA into L2 before the wait; otherwise every older block of
    the CTA is prefetched. Neither changes a score.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request verifying 1 + 3 rows over 20000 tokens: the early-issue kernel",
            spec=dict(seq_lens=(20000,), qlen=4, early=True),
        ),
        Cell(
            why="one request, one row, 65536 tokens: early issue, 512 blocks on 512 CTAs",
            spec=dict(seq_lens=(65536,), qlen=1, early=True),
        ),
        Cell(
            why="one request of 129 tokens: the early tiles of most CTAs lie past the length and are drained",
            spec=dict(seq_lens=(129,), qlen=4, early=True),
        ),
        Cell(
            why="one request longer than 512 blocks (the CTAs loop): early issue",
            spec=dict(seq_lens=(70000,), qlen=4, early=True),
        ),
        Cell(
            why="two requests: the prefetching kernel",
            spec=dict(seq_lens=(20000, 3000), qlen=4, early=False),
        ),
        Cell(
            why="eight requests of mixed lengths: the prefetching kernel",
            spec=dict(
                seq_lens=(300, 1500, 2049, 4096, 5000, 7777, 8192, 33001), qlen=4, early=False
            ),
        ),
    )

    note = """
    The products of FP8 (or BF16) values are exact in fp32 and the kernel sums 128 of them in its k-split order, so it
    differs from the reference matmul by a few fp32 ulps; it is held to the in-tree scorer bit for bit.
    """


m3_index_score = M3IndexScore()
