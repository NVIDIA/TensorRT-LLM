# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge one chunk's partial attention into the running result with their softmax statistics (MLA chunked
prefill)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: merge_op values, per context request.
SKIP, MERGE, COPY = 0, 1, 2


class MergeChunkedAttentionForMla(OpWrapper):
    """In place: for each context request ``b`` and each of its query rows, ``merged_attn`` and
    ``merged_softmax_stats`` become the softmax-weighted combination of the running partial (``merged_*``) and this
    chunk's partial (``temp_*``) when ``merge_op[b] == 1``, this chunk's partial when ``merge_op[b] == 2``, and stay as
    they are when ``merge_op[b] == 0``. A row's statistics are ``(max, sum)`` of its exp-scores, fp32
    ``[rows, num_heads, 2]``; attention rows are ``[rows, num_heads * head_size]``.

    Returns ``merged_attn``. The query rows of request ``b`` are ``cu_q_seq_lens[b]:cu_q_seq_lens[b + 1]``.
    """

    ARCHS = frozenset({Arch.SM_100})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Kimi K3 chunked prefill: 6 heads per rank, v_head_dim 128, three requests with merge ops copy, "
            "merge and skip",
            spec=dict(num_heads=6, head_size=128, q_lens=(50, 30, 9), merge_op=(COPY, MERGE, SKIP)),
        ),
        Cell(
            why="every request merged, uneven rows",
            spec=dict(
                num_heads=6, head_size=128, q_lens=(1, 64, 17), merge_op=(MERGE, MERGE, MERGE)
            ),
        ),
    )

    note = """
    Writes `merged_softmax_stats` too, although the op's schema marks only `merged_attn` mutable. The schedule of
    merge ops per loop is the attention metadata's `merge_op_tensor[loop]` (a request whose chunk is empty skips; its
    first non-empty chunk copies). Rows past `cu_q_seq_lens[num_requests]` are not touched.
    """

    def __call__(
        self,
        merged_attn: torch.Tensor,
        temp_attn: torch.Tensor,
        merged_softmax_stats: torch.Tensor,
        temp_softmax_stats: torch.Tensor,
        num_requests: int,
        cu_q_seq_lens: torch.Tensor,
        max_q_seq_len: int,
        merge_op: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.merge_chunked_attention_for_mla(
            merged_attn,
            temp_attn,
            merged_softmax_stats,
            temp_softmax_stats,
            num_requests,
            cu_q_seq_lens,
            max_q_seq_len,
            merge_op,
            num_heads,
            head_size,
        )
        return merged_attn

    def reference(
        self,
        merged_attn: torch.Tensor,
        temp_attn: torch.Tensor,
        merged_softmax_stats: torch.Tensor,
        temp_softmax_stats: torch.Tensor,
        num_requests: int,
        cu_q_seq_lens: torch.Tensor,
        max_q_seq_len: int,
        merge_op: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> tuple:
        """``(attn, stats)`` after the merge, fp32-computed; the inputs are not modified."""
        attn = merged_attn.float().clone().view(-1, num_heads, head_size)
        stats = merged_softmax_stats.clone()
        tattn = temp_attn.float().view(-1, num_heads, head_size)
        cu = cu_q_seq_lens.tolist()
        ops = merge_op.tolist()
        for b in range(num_requests):
            rows = slice(cu[b], cu[b + 1])
            if ops[b] == COPY:
                attn[rows] = tattn[rows]
                stats[rows] = temp_softmax_stats[rows]
            elif ops[b] == MERGE:
                m0, s0 = stats[rows, :, 0], stats[rows, :, 1]
                m1, s1 = temp_softmax_stats[rows, :, 0], temp_softmax_stats[rows, :, 1]
                m = torch.maximum(m0, m1)
                w0, w1 = s0 * torch.exp(m0 - m), s1 * torch.exp(m1 - m)
                s = w0 + w1
                attn[rows] = (attn[rows] * w0[..., None] + tattn[rows] * w1[..., None]) / s[
                    ..., None
                ]
                stats[rows, :, 0], stats[rows, :, 1] = m, s
        return attn.view_as(merged_attn).to(merged_attn.dtype), stats


merge_chunked_attention_for_mla = MergeChunkedAttentionForMla()
