# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latent MQA decode over the paged bf16 MLA cache on the CuTe DSL Blackwell kernel, and its per-request split-KV
plan: what the built-in MLA runs Kimi K3's decode-only generation steps on (the stock CuTe DSL MLA FMHA library)."""

from __future__ import annotations

import math
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_mla.split_kv_plan import (  # noqa: E501
    _SPLIT_PLAN_MAX_REQUESTS,
    launch_mla_split_kv_plan,
    plan_mla_split_kv,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

#: Every CuTe DSL tactic's mma_qk_tiler N: the KV tile the softmax runs over.
KV_TILE = 128


def available() -> bool:
    """Whether the CuTe DSL runtime is installed (the op's kernel compiles on first use)."""
    return bool(IS_CUTLASS_DSL_AVAILABLE)


def _runner():
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner


def workspace_bytes(
    num_heads: int, seq_len_q: int, kv_lora_rank: int, max_num_requests: int
) -> int:
    """Bytes the kernel owns at the front of the attention workspace for this geometry."""
    import cutlass

    return _runner().get_max_padded_workspace_size(
        num_heads, seq_len_q, kv_lora_rank, max_num_requests, cutlass.Float32
    )


class MlaSplitKvPlan(OpWrapper):
    """Write into ``plan[:num_requests]`` each request's split-KV count for ``cache_seqs`` (one launch). The longest
    request is cut into ``k`` equal pieces of at least 4 KV tiles (every request then takes ``ceil(tiles / piece)``
    pieces, at most 32) for the ``k`` with the smallest wave-quantized makespan."""

    ARCHS = frozenset({Arch.SM_100})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Kimi K3 decode: 6 heads x 1 token (one 128-row query tile per request), 12 requests",
            spec=dict(
                num_heads=6, seq_len_q=1, kv_lens=(164, 72, 110, 6, 70, 142, 12, 95, 38, 69, 7, 55)
            ),
        ),
        Cell(
            why="DSpark verification: 6 heads x 8 tokens, long and short histories",
            spec=dict(num_heads=6, seq_len_q=8, kv_lens=(40000, 1, 129, 4096)),
        ),
    )

    note = """
    The plan is a function of the KV lengths and the decode geometry only, so a step plans once and every MLA layer
    reads it. `rows_per_request` is the runner's packed query tiles per request
    (`ceil(num_heads * seq_len_q / 128)`), `wave_clusters` its occupancy ceiling over the 2-CTA cluster; both are
    passed in. Up to 1024 requests.
    """

    def __call__(
        self,
        cache_seqs: torch.Tensor,
        plan: torch.Tensor,
        num_requests: int,
        rows_per_request: int,
        wave_clusters: int,
    ) -> torch.Tensor:
        launch_mla_split_kv_plan(
            cache_seqs,
            plan,
            num_requests=num_requests,
            rows_per_request=rows_per_request,
            wave_clusters=wave_clusters,
        )
        return plan[:num_requests]

    def reference(
        self,
        cache_seqs: torch.Tensor,
        plan: torch.Tensor,
        num_requests: int,
        rows_per_request: int,
        wave_clusters: int,
    ) -> torch.Tensor:
        """The host mirror of the plan (``plan`` is not read)."""
        counts = plan_mla_split_kv(
            cache_seqs[:num_requests].tolist(),
            rows_per_request=rows_per_request,
            wave_clusters=wave_clusters,
        )
        return torch.tensor(counts, dtype=torch.int32, device=cache_seqs.device)

    @staticmethod
    def geometry(num_heads: int, seq_len_q: int) -> tuple:
        """``(rows_per_request, wave_clusters)`` of the CuTe DSL runner for this decode geometry."""
        runner = _runner()
        rows = runner.get_num_q_tiles(num_heads, seq_len_q)
        waves = max(runner._get_max_active_blocks() // runner._CLUSTER_SHAPE_MNK[0], 1)
        return rows, waves

    MAX_REQUESTS = _SPLIT_PLAN_MAX_REQUESTS


class CuteDslMlaDecode(OpWrapper):
    """Write ``output`` ``[R * S, num_heads * 512]``: for request ``r``'s query token ``s`` (rows token-major within a
    request) and each head, ``softmax(q . K^T * softmax_scale) V`` over its first ``cache_seqs[r] - S + 1 + s`` cached
    latent rows ``[c_latent 512 | c_rope 64]``, with ``K`` the whole row and ``V`` the latent part. ``q`` is
    ``[R * S, num_heads, 576]``; the cache is the layer's paged bf16 pool ``[pages, 1, page_size, 1, >= 576]``,
    request ``r``'s pages in ``page_table[:, r]`` (``[max_pages, R]`` int32)."""

    ARCHS = frozenset({Arch.SM_100})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Kimi K3 decode beyond the decode kernels' 8 requests: 6 heads, page 64, 12 requests of 1 token",
            spec=dict(
                num_heads=6, seq_len_q=1, kv_lens=(164, 72, 110, 6, 70, 142, 12, 95, 38, 69, 7, 55)
            ),
        ),
        Cell(
            why="12 requests of 2 tokens",
            spec=dict(
                num_heads=6, seq_len_q=2, kv_lens=(166, 74, 112, 8, 72, 144, 14, 97, 40, 71, 9, 57)
            ),
        ),
        Cell(
            why="DSpark verification beyond 8 requests: 12 requests of 8 tokens",
            spec=dict(
                num_heads=6,
                seq_len_q=8,
                kv_lens=(174, 82, 120, 16, 80, 152, 22, 105, 48, 79, 17, 65),
            ),
        ),
        Cell(
            why="the generic path's decode of 3 requests (a decode step the decode kernels decline)",
            spec=dict(num_heads=6, seq_len_q=1, kv_lens=(151, 68, 105)),
        ),
    )

    note = """
    bf16 only, which is all a Kimi K3 latent pool holds. The kernel's tactic (split_kv, persistence) is the
    AutoTuner's per shape; `block_split_kvs` (the `MlaSplitKvPlan` output, or None for the tactic's uniform split)
    sets how many of the tactic's split slots each request uses. `page_table` and `cache_seqs` must keep 16-byte
    aligned pointers (the runner compiles with that assumption); pass a contiguous copy of a misaligned view.
    `workspace` is the front `workspace_bytes(...)` of the attention workspace. Linear-chain causal verification only:
    a tree mask is not expressible.
    """

    def __call__(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        kv_pool: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        max_num_requests: int,
        block_split_kvs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch = cache_seqs.shape[0]
        latent = kv_lora_rank + qk_rope_head_dim
        kv_pages = kv_pool[:, 0, :, 0, :latent]
        q_view = q.view(batch, seq_len_q, num_heads, latent)
        o_view = output.view(batch, seq_len_q, num_heads, kv_lora_rank)
        torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell(
            q_view[..., :kv_lora_rank].permute(2, 3, 1, 0),
            q_view[..., kv_lora_rank:].permute(2, 3, 1, 0),
            kv_pages[..., :kv_lora_rank].permute(1, 2, 0),
            kv_pages[..., kv_lora_rank:].permute(1, 2, 0),
            page_table,
            cache_seqs,
            o_view.permute(2, 3, 1, 0),
            workspace,
            num_heads,
            seq_len_q,
            page_size,
            softmax_scale,
            1.0,
            max_num_requests,
            None,
            None,
            block_split_kvs,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        kv_pool: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        max_num_requests: int,
        block_split_kvs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """fp32 latent MQA over the pages the table names (``output``, ``workspace`` and the plan are not read)."""
        latent = kv_lora_rank + qk_rope_head_dim
        batch = cache_seqs.shape[0]
        q4 = q.view(batch, seq_len_q, num_heads, latent).float()
        out = torch.empty(
            batch, seq_len_q, num_heads, kv_lora_rank, dtype=torch.float32, device=q.device
        )
        lens = cache_seqs.tolist()
        for r in range(batch):
            pages = page_table[: math.ceil(lens[r] / page_size), r].long()
            rows = kv_pool[pages, 0, :, 0, :latent].reshape(-1, latent)[: lens[r]].float()
            for s in range(seq_len_q):
                visible = lens[r] - seq_len_q + 1 + s
                scores = (q4[r, s] @ rows[:visible].t()) * softmax_scale
                out[r, s] = scores.softmax(-1) @ rows[:visible, :kv_lora_rank]
        return out.view(batch * seq_len_q, num_heads * kv_lora_rank).to(q.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        kv_pool: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqs: torch.Tensor,
        workspace: torch.Tensor,
        num_heads: int,
        seq_len_q: int,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        max_num_requests: int,
        block_split_kvs: Optional[torch.Tensor],
    ) -> None:
        for name, t in (("page_table", page_table), ("cache_seqs", cache_seqs)):
            assert t.data_ptr() % 16 == 0, (
                f"{name} must keep a 16-byte aligned pointer; the runner assumes it and reads shifted rows otherwise"
            )
        assert kv_pool.dtype == q.dtype == output.dtype == torch.bfloat16, (
            "bf16 q, cache and output only"
        )
        assert kv_pool.shape[2] == page_size, "page_size must be the pool's page size"
        assert workspace.numel() * workspace.element_size() >= workspace_bytes(
            num_heads, seq_len_q, kv_lora_rank, max_num_requests
        ), "the workspace is smaller than the kernel's split-KV partials; the kernel writes past it"


mla_split_kv_plan = MlaSplitKvPlan()
cute_dsl_mla_decode = CuteDslMlaDecode()
