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

#: The wide target's 9..48-request buckets (modeling's INDEX_SCORE_WIDE_SPLITS, repeated here because the ops do not
#: import the dispatch; test_modeling_v2_m3_dispatch checks that every covered request count is certified, so they
#: cannot drift).
_WIDE_SPLITS = ((10, 127), (13, 97), (17, 74), (23, 55), (31, 41), (39, 32), (48, 26))


def _wide_split(requests: int) -> int:
    return next(s for hi, s in _WIDE_SPLITS if requests <= hi)


def _wide_lens(requests: int) -> tuple:
    """Short, mid and edge contexts (one 165521-token request at 12 / 20 / 32 / 46 requests)."""
    base = (129, 4000, 20000, 65536, 300, 1000, 8193, 16384)
    lens = [base[i % len(base)] for i in range(requests)]
    if requests in (12, 20, 32, 46):
        lens[0] = 165521
    return tuple(lens)


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
            why="one request verifying 1 + 3 rows over 165521 tokens (AgentX ISL) at TP2: 1536 CTAs, early issue",
            spec=dict(seq_lens=(165521,), qlen=4, heads=2, split_k=1536, early=True),
        ),
        Cell(
            why="one request of 129 tokens: the early tiles of most of the 1536 CTAs lie past the length and drain",
            spec=dict(seq_lens=(129,), qlen=4, heads=2, split_k=1536, early=True),
        ),
        Cell(
            why="one request of 400000 tokens: 3125 blocks, so the 1536 CTAs loop",
            spec=dict(seq_lens=(400000,), qlen=4, heads=2, split_k=1536, early=True),
        ),
        Cell(
            why="one request decoding one row over 65536 tokens",
            spec=dict(seq_lens=(65536,), qlen=1, heads=2, split_k=1536, early=True),
        ),
        Cell(
            why="two requests verifying 4 rows each over 165521 and 20000 tokens: 768 CTAs each, early issue",
            spec=dict(seq_lens=(165521, 20000), qlen=4, heads=2, split_k=768, early=True),
        ),
        Cell(
            why="two requests, one of 300 tokens and one of 262144 (2048 blocks: the 768 CTAs loop)",
            spec=dict(seq_lens=(300, 262144), qlen=4, heads=2, split_k=768, early=True),
        ),
        Cell(
            why="three requests verifying 4 rows each over 165521, 148968 and 132416 tokens: 424 CTAs each",
            spec=dict(seq_lens=(165521, 148968, 132416), qlen=4, heads=2, split_k=424, early=True),
        ),
        Cell(
            why="three requests, 129 / 323584 / 20000 tokens (2528 blocks: the 424 CTAs loop; the short one drains)",
            spec=dict(seq_lens=(129, 323584, 20000), qlen=4, heads=2, split_k=424, early=True),
        ),
        Cell(
            why="four requests verifying 4 rows each over 165521, 148968, 132416 and 115864 tokens: 318 CTAs each",
            spec=dict(
                seq_lens=(165521, 148968, 132416, 115864), qlen=4, heads=2, split_k=318, early=True
            ),
        ),
        Cell(
            why="four requests, 300 / 400000 / 129 / 65536 tokens (3125 blocks: the 318 CTAs loop)",
            spec=dict(seq_lens=(300, 400000, 129, 65536), qlen=4, heads=2, split_k=318, early=True),
        ),
        Cell(
            why="five requests over 165521 down to 99312 tokens (0.6 x): 254 CTAs each",
            spec=dict(
                seq_lens=(165521, 148968, 132416, 115864, 99312),
                qlen=4,
                heads=2,
                split_k=254,
                early=True,
            ),
        ),
        Cell(
            why="six requests, 129 / 400000 / 300 / 20000 / 65536 / 1000: 212 CTAs (3125 blocks loop; short drain)",
            spec=dict(
                seq_lens=(129, 400000, 300, 20000, 65536, 1000),
                qlen=4,
                heads=2,
                split_k=212,
                early=True,
            ),
        ),
        Cell(
            why="seven requests over 323584 down to 129414 tokens (0.4 x): 181 CTAs each, all loop",
            spec=dict(
                seq_lens=(323584, 291225, 258867, 226508, 194150, 161792, 129433),
                qlen=4,
                heads=2,
                split_k=181,
                early=True,
            ),
        ),
        Cell(
            why="eight requests, 165521 down to 49656 tokens (0.3 x) and one of 129: 159 CTAs each",
            spec=dict(
                seq_lens=(165521, 148968, 132416, 115864, 99312, 82760, 49656, 129),
                qlen=4,
                heads=2,
                split_k=159,
                early=True,
            ),
        ),
    ) + tuple(
        Cell(
            why=f"{r} requests at {_wide_split(r)} CTAs each (W-IDX1): "
            + ("one long request" if r in (12, 20, 32, 46) else "short and mid contexts"),
            spec=dict(seq_lens=_wide_lens(r), qlen=4, heads=2, split_k=_wide_split(r), early=True),
        )
        for r in range(9, 49)
    )

    note = """
    The products of FP8 (or BF16) values are exact in fp32 and the kernel sums 128 of them in its k-split order, so it
    differs from the reference matmul by a few fp32 ulps; it is held to the in-tree scorer bit for bit.
    """


m3_index_score = M3IndexScore()
