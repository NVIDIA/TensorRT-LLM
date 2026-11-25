# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py
"""Shared index-score operation; targets supply scheduling/cache policy explicitly."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper, true_fp32_matmul

from ...kernels.attention.index_score_runtime import IndexScoreRunner

PAGE = 128
HEAD_DIM = 128


def compiled_scorer(key: tuple, runner: type[IndexScoreRunner]):
    if key not in runner.kernel_cache:
        runner._compile(*key)
    return runner.kernel_cache[key]


class M3IndexScoreBase(OpWrapper):
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
        *,
        runner: type[IndexScoreRunner],
    ) -> torch.Tensor:
        _, num_heads, head_dim = idx_q.shape
        key = (idx_q.dtype, num_heads, max_decode_query_len, head_dim, split_k, early)
        compiled_scorer(key, runner)(idx_q, index_k, block_table, score, seq_lens)
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
        *,
        runner: type[IndexScoreRunner],
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
        *,
        runner: type[IndexScoreRunner],
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
