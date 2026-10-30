# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's MLA decode query path (q_a RMSNorm, q_b projection, k_b absorption) in one launch, which also stores the
step's latent KV rows into the paged latent cache; also the query path alone and the form that stores the rows
densely."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_mla import (
    op as _k3_mla_op,  # noqa: F401 — registers the ops
)


class K3MlaQkv(OpWrapper):
    """Return ``fused_q`` [M, heads * 576] bf16 of the M <= 64 rows of ``ag``, and store each token's cache row
    ``[bf16(rmsnorm(ag[t, 1536:2048]) * w_kv) | ag[t, 2048:2112]]`` into ``pool`` at its request's position."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16 decode without speculation, three requests",
            spec=dict(num_requests=3, tokens=1, heads=6),
        ),
        Cell(
            why="TP16 decode at the maximum eight-request batch",
            spec=dict(num_requests=8, tokens=1, heads=6),
        ),
        Cell(
            why="TP16 eight-token DSpark verification, two requests",
            spec=dict(num_requests=2, tokens=8, heads=6),
        ),
        Cell(
            why="TP16 maximum 64-token verification batch",
            spec=dict(num_requests=8, tokens=8, heads=6),
        ),
        Cell(
            why="TP4 verification with 24 heads and four head groups",
            spec=dict(num_requests=2, tokens=8, heads=24),
        ),
    )

    note = """
    Kimi K3 is NoPE: q_pe and the rope columns are copied, not rotated.
    The fused projection producing ag and the attention/output projection are
    outside this fusion. ag is contiguous bf16 [M, C], C >= 2112 and divisible
    by 8, normally [q_a 1536 | kv_a 512 | rope 64 | gate heads * 128].
    The weights are contiguous bf16: w_qa [1536], w_qb [heads * 192, 1536],
    w_kb [heads, 512, 128], w_kv [512]. The underlying kernel supports SM100
    and SM103; only SM100 has certification.

    k3_mla_q returns the same fused_q bits without storing rows; k3_mla_qkv_out
    stores the same rows densely in bf16 [M, 576]. Query rows are computed in
    chunks of 8, 16 or 32; every eight-token chunk is bit-identical to its own
    call. The op-level test tests/unittest/_torch/modeling_v2/models/kimi_k3_vl/test_k3_mla_q.py
    covers M 1..64 against the unfused RMSNorm/GEMM/bmm chain, 640-element
    interleaved rows, short lengths, refusals, and optional K3_BASE_TRTLLM
    single-request identity. The catalog tests retain the 2e-2 query bound
    per half, 1e-2 latent-row bound and bit-exact rope columns.

    There is no owned state and nothing to re-arm. The caller's pool belongs
    to a SELFKONLY KVCacheManager: one latent head of 576, kv_factor 1,
    64 tokens per block. The manager owns and sizes it, interleaving layers
    by block; page_offset selects the layer's slot. Each call writes only
    its tokens' 576-column rows, skipping negative positions. Calls for
    different layers/positions write disjoint rows; two managers are independent.
    A wrong but valid page_offset silently overwrites another layer. No input
    at this call site identifies the intended layer, so is_valid cannot detect
    that mistake; the negative control compares the manager's expected pools.

    k3_mla_decode_view(attn, metadata, M), computed per layer after prepare(),
    supplies the flat pool, token row_stride (>=576, divisible by 8), strided
    page-table rows kv_cache_block_offsets[pool, num_contexts + i, 0, :],
    lengths kv_lens_cuda_runtime[num_contexts + i], and the layer page_offset.
    The view admits generation-only R <= 8, uniform T <= 8, beam width 1,
    no tree speculation, 64-token pages, bf16 pool, heads divisible by 6,
    kv_lora_rank 512 and qk_rope_head_dim 64; otherwise the target falls back.
    A one-dimensional page table is accepted for a single request.

    Page tables and lengths are host-filled before the step or graph replay;
    they must not be produced by the immediately preceding kernel, because
    this launch reads them before its grid-dependency wait. Weights are TMA'd
    before that wait with EVICT_FIRST; ag is read after it. The step's rows
    must be stored before the same layer's attention reads them after its
    wait, so attention may directly follow this call. trigger_early defaults
    to True: dependents may launch early but wait for the whole grid.

    Invalid shapes, M > 64, M not divisible by R, inconsistent metadata,
    or heads outside 2..24 are rejected before launch without changing the
    pool. seq_len < T intentionally stores only nonnegative positions.
    First calls compile and must occur outside CUDA-graph capture.
    Certification drives three layers over two eager steps and two graph
    replays with rewritten inputs and prepared metadata, checking every
    layer's whole pool against manager block ids; alternating two managers
    also checks isolation. Cached lengths cover 64..4100 and page crossings.
    """

    def __call__(
        self,
        ag: torch.Tensor,
        w_qa: torch.Tensor,
        eps: float,
        w_qb: torch.Tensor,
        w_kb: torch.Tensor,
        w_kv: torch.Tensor,
        kv_eps: float,
        pool: torch.Tensor,
        row_stride: int,
        page_table: torch.Tensor,
        page_offset: int,
        seq_len: torch.Tensor,
        trigger_early: bool = True,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_mla_qkv(
            ag,
            w_qa,
            eps,
            w_qb,
            w_kb,
            w_kv,
            kv_eps,
            pool,
            row_stride,
            page_table,
            page_offset,
            seq_len,
            trigger_early,
        )

    def reference(
        self,
        ag: torch.Tensor,
        w_qa: torch.Tensor,
        eps: float,
        w_qb: torch.Tensor,
        w_kb: torch.Tensor,
        w_kv: torch.Tensor,
        kv_eps: float,
        pool: torch.Tensor,
        row_stride: int,
        page_table: torch.Tensor,
        page_offset: int,
        seq_len: torch.Tensor,
        trigger_early: bool = True,
    ) -> torch.Tensor:
        m, heads = ag.shape[0], w_kb.shape[0]
        x = ag[:, :1536].float()
        qn = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w_qa.float()).bfloat16()
        with true_fp32_matmul():
            q = (qn.float() @ w_qb.float().t()).bfloat16().view(m, heads, 192)
            q_abs = torch.einsum("thd,hcd->thc", q[..., :128].float(), w_kb.float()).bfloat16()
        x = ag[:, 1536:2048].float()
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + kv_eps)
        rows = torch.cat([((x * r) * w_kv.float()).bfloat16(), ag[:, 2048:2112]], dim=-1)
        tokens = m // seq_len.numel()
        table = page_table.reshape(1, -1) if page_table.ndim == 1 else page_table
        positions = seq_len.long()[:, None] - tokens + torch.arange(tokens, device=ag.device)
        requests = torch.arange(seq_len.numel(), device=ag.device)[:, None].expand_as(positions)
        valid = positions >= 0
        positions, requests = positions[valid], requests[valid]
        indices = (table[requests, positions // 64].long() + page_offset) * 64 + positions % 64
        pool.view(-1, row_stride)[indices, :576] = rows[valid.reshape(-1)]
        return torch.cat([q_abs, q[..., 128:]], dim=-1).reshape(m, heads * 576)

    def is_valid(
        self,
        ag: torch.Tensor,
        w_qa: torch.Tensor,
        eps: float,
        w_qb: torch.Tensor,
        w_kb: torch.Tensor,
        w_kv: torch.Tensor,
        kv_eps: float,
        pool: torch.Tensor,
        row_stride: int,
        page_table: torch.Tensor,
        page_offset: int,
        seq_len: torch.Tensor,
        trigger_early: bool = True,
    ) -> None:
        return None


k3_mla_qkv = K3MlaQkv()


def k3_mla_q(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """Return ``fused_q`` alone (no cache row stored): the same bits as ``k3_mla_qkv``'s."""
    return torch.ops.trtllm.k3_mla_q(ag, w_qa, eps, w_qb, w_kb, trigger_early)


def k3_mla_qkv_out(
    ag: torch.Tensor,
    w_qa: torch.Tensor,
    eps: float,
    w_qb: torch.Tensor,
    w_kb: torch.Tensor,
    w_kv: torch.Tensor,
    kv_eps: float,
    kv_out: torch.Tensor,
    trigger_early: bool = True,
) -> torch.Tensor:
    """Return ``fused_q`` and store the cache rows densely into ``kv_out`` [M, 576] instead of the pool."""
    return torch.ops.trtllm.k3_mla_qkv_out(
        ag, w_qa, eps, w_qb, w_kb, w_kv, kv_eps, kv_out, trigger_early
    )
