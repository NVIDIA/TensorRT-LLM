# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# PyTorch oracle ported from vLLM (Apache-2.0), _reference_decode_index_score:
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/tests/kernels/attention/test_minimax_m3.py#L188
"""The MSA index scorer of a sparse decode step: per (index head, query token, KV block) the causal max of Q . K over
the block's 128 index-K positions, with the pre-wait schedule of ``ops/m3_index_score``."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

__all__ = ["m3_index_score"]

PAGE = 128
HEAD_DIM = 128
#: CTAs per request (each scores blocks split, split + SPLIT_K, ...; the scores do not depend on it).
SPLIT_K = 512


def compiled_scorer(key: tuple):
    """The compiled kernel of ``key`` (dtype, heads, query length, head dim, split_k, early), compiling it on a miss
    (the target compiles every key it can take before warmup: ``layers.prebuild_index_scorer``)."""
    from ..m3_index_score import IndexScoreRunner

    if key not in IndexScoreRunner.kernel_cache:
        IndexScoreRunner._compile(*key)
    return IndexScoreRunner.kernel_cache[key]


class M3IndexScore(OpWrapper):
    """``score[h, t, j]`` (fp32, a transposed view of the [heads, blocks, tokens] selector buffer) = the max over the
    positions ``p`` of block ``j`` of request ``b = t // qlen`` with ``p <= seq_lens[b] - qlen + t % qlen`` of
    ``idx_q[t, h] . index_k[block_table[b, j], p % 128]``, for ``j < ceil(seq_lens[b] / 128)`` (no other entry is
    written). FP8 or BF16 q and index-K, fp32 products and sums.

    ``early`` (the target's one-request kernel) issues the Q and first K tiles right after the grid wait from a
    pre-wait length hint and prefetches one older block per CTA into L2 before the wait; otherwise every older block of
    the CTA is prefetched. Neither changes a score.
    """

    ARCHS = frozenset({Arch.SM_103})

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

    def __call__(
        self,
        idx_q: torch.Tensor,
        index_k: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        score: torch.Tensor,
        max_decode_query_len: int,
        split_k: int,
        early: bool,
    ) -> torch.Tensor:
        _, num_heads, head_dim = idx_q.shape
        key = (idx_q.dtype, num_heads, max_decode_query_len, head_dim, split_k, early)
        compiled_scorer(key)(idx_q, index_k, block_table, score, seq_lens)
        return score

    def reference(
        self,
        idx_q: torch.Tensor,
        index_k: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        score: torch.Tensor,
        max_decode_query_len: int,
        split_k: int,
        early: bool,
    ) -> torch.Tensor:
        """The fp32 [heads, total_q, blocks] scores (``-inf`` where nothing is written). ``score``, ``split_k`` and
        ``early`` do not change them."""
        total_q, heads, _ = idx_q.shape
        qlen = max_decode_query_len
        out = torch.full(
            (heads, total_q, score.shape[2]),
            -float("inf"),
            dtype=torch.float32,
            device=idx_q.device,
        )
        with true_fp32_matmul():
            for b, seq in enumerate(seq_lens.tolist()):
                nblocks = (seq + PAGE - 1) // PAGE
                if nblocks == 0:
                    continue
                t0 = b * qlen
                q = idx_q[t0 : t0 + qlen].float()
                k = index_k[block_table[b, :nblocks].long()].reshape(nblocks * PAGE, -1).float()
                s = torch.einsum("qhd,kd->hqk", q, k)
                q_pos = seq - qlen + torch.arange(qlen, device=idx_q.device)
                k_pos = torch.arange(k.shape[0], device=idx_q.device)
                s.masked_fill_(k_pos[None, :] > q_pos[:, None], -float("inf"))
                out[:, t0 : t0 + qlen, :nblocks] = s.reshape(heads, qlen, nblocks, PAGE).amax(dim=3)
        return out

    def is_valid(
        self,
        idx_q: torch.Tensor,
        index_k: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        score: torch.Tensor,
        max_decode_query_len: int,
        split_k: int,
        early: bool,
    ) -> None:
        # The block-table row and the score row are read and written up to ceil(seq_len / 128) with no bound.
        nblocks = (seq_lens + PAGE - 1) // PAGE
        assert bool((nblocks <= block_table.shape[1]).all()), (
            "a request runs past its block-table row"
        )
        assert bool((nblocks <= score.shape[2]).all()), "a request runs past the score buffer"
        assert idx_q.shape[0] == seq_lens.shape[0] * max_decode_query_len, (
            "idx_q must hold max_decode_query_len rows per request"
        )


m3_index_score = M3IndexScore()
