# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-row work between the sparse MLA input GEMMs and the query projections, in one launch.

For each generation row: the RMSNorms of the q_a and kv_a halves of the fused input GEMM
output, the LayerNorm of the indexer key, and the paged-cache writes of the normalized
latent (``clamp(latent * scale, +-448)`` as E4M3, into the FP8 latent cache) and of the
indexer row ``[k | gate]``. It returns the normalized q_a rows (``q_resid``) for the q_b and
indexer query projections: one launch in place of RMSNorm, RMSNorm, LayerNorm, cat and a
cache append.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _prologue_kernel(
    qkv_ptr,
    idx_ptr,
    qw_ptr,
    kvw_ptr,
    lnw_ptr,
    lnb_ptr,
    q_out_ptr,
    pos_ptr,
    bt_ptr,
    latent_pool_ptr,
    index_pool_ptr,
    scale_ptr,
    rms_eps,
    ln_eps,
    qkv_stride,
    idx_stride,
    bt_stride,
    latent_slot_stride,
    latent_row_stride,
    index_slot_stride,
    index_row_stride,
    tpb,
    ROWS_PER_TABLE: tl.constexpr,
    Q_DIM: tl.constexpr,
    Q_PAD: tl.constexpr,
    KV_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    r = tl.program_id(0)
    # q_a RMSNorm
    qd = tl.arange(0, Q_PAD)
    qm = qd < Q_DIM
    q = tl.load(qkv_ptr + r * qkv_stride + qd, mask=qm, other=0.0).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, axis=0) / Q_DIM + rms_eps)
    q = q * tl.load(qw_ptr + qd, mask=qm, other=0.0).to(tl.float32)
    tl.store(q_out_ptr + r * Q_DIM + qd, q.to(tl.bfloat16), mask=qm)
    # kv_a RMSNorm -> latent cache
    kd = tl.arange(0, KV_DIM)
    kv = tl.load(qkv_ptr + r * qkv_stride + Q_DIM + kd).to(tl.float32)
    kv = kv * tl.rsqrt(tl.sum(kv * kv, axis=0) / KV_DIM + rms_eps)
    kv = (kv * tl.load(kvw_ptr + kd).to(tl.float32)).to(tl.bfloat16)
    tbl = (r // ROWS_PER_TABLE).to(tl.int64)
    pos = tl.load(pos_ptr + r).to(tl.int64)
    page = pos // tpb
    off = pos - page * tpb
    slot = tl.load(bt_ptr + tbl * bt_stride + page).to(tl.int64)
    latent_dst = latent_pool_ptr + slot * latent_slot_stride + off * latent_row_stride + kd
    scaled = kv.to(tl.float32) * tl.load(scale_ptr)
    tl.store(latent_dst, tl.clamp(scaled, -448.0, 448.0).to(latent_pool_ptr.dtype.element_ty))
    # indexer key LayerNorm and gate -> index cache
    hd = tl.arange(0, HEAD_DIM)
    k = tl.load(idx_ptr + r * idx_stride + hd).to(tl.float32)
    mean = tl.sum(k, axis=0) / HEAD_DIM
    diff = k - mean
    var = tl.sum(diff * diff, axis=0) / HEAD_DIM
    k = diff * tl.rsqrt(var + ln_eps) * tl.load(lnw_ptr + hd).to(tl.float32) + tl.load(
        lnb_ptr + hd
    ).to(tl.float32)
    gate = tl.load(idx_ptr + r * idx_stride + HEAD_DIM + hd)
    index_dst = index_pool_ptr + slot * index_slot_stride + off * index_row_stride
    tl.store(index_dst + hd, k.to(tl.bfloat16).to(index_pool_ptr.dtype.element_ty))
    tl.store(index_dst + HEAD_DIM + hd, gate.to(index_pool_ptr.dtype.element_ty))


class MlaPrologue(OpWrapper):
    """``q_resid`` [R, q_lora] from ``qkv`` [R, q_lora + kv_lora] and ``idx`` [R, >= 2 * head_dim]
    (indexer key | gate | ...), writing the latent and indexer rows of position ``positions``
    (``[N, T]``, table ``block_tables[n]``) into the paged pools: the FP8 latent pool
    ``[slots, tokens_per_block, kv_lora]`` and the BF16 indexer pool ``[slots, tokens_per_block,
    >= 2 * head_dim]``. ``scale`` is the FP32 ``[1]`` quantization scale of the latent cache."""

    ARCHS = frozenset({Arch.SM_100})
    # The target's generation rows: T = 1 (plain decode and the MTP draft steps) or 4 (MTP-3
    # verify) rows per request, read through the int64 GLM block tables.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row: an MTP draft step of one request", spec=dict(n=1, t=1)),
        Cell(why="single-token decode of eight requests: one row per table", spec=dict(n=8, t=1)),
        Cell(why="MTP-3 verify of one request", spec=dict(n=1, t=4)),
        Cell(why="MTP-3 verify of eight requests", spec=dict(n=8, t=4)),
    )

    note = """
    `T` (`positions.shape[1]`) is a compile-time constant of the kernel: T = 1 and T = 4 compile
    once each.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        idx: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        rms_eps: float,
        ln_eps: float,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
        latent_pool: torch.Tensor,
        index_pool: torch.Tensor,
        tokens_per_block: int,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        q_dim, kv_dim = q_norm_weight.shape[0], kv_norm_weight.shape[0]
        head_dim = k_norm_weight.shape[0]
        rows = positions.numel()
        q_resid = torch.empty((rows, q_dim), dtype=torch.bfloat16, device=qkv.device)
        _prologue_kernel[(rows,)](
            qkv,
            idx,
            q_norm_weight,
            kv_norm_weight,
            k_norm_weight,
            k_norm_bias,
            q_resid,
            positions,
            block_tables,
            latent_pool,
            index_pool,
            scale,
            rms_eps,
            ln_eps,
            qkv.stride(0),
            idx.stride(0),
            block_tables.stride(0),
            latent_pool.stride(0),
            latent_pool.stride(1),
            index_pool.stride(0),
            index_pool.stride(1),
            tokens_per_block,
            ROWS_PER_TABLE=positions.shape[1],
            Q_DIM=q_dim,
            Q_PAD=triton.next_power_of_2(q_dim),
            KV_DIM=kv_dim,
            HEAD_DIM=head_dim,
            num_warps=4,
        )
        return q_resid

    def reference(
        self,
        qkv: torch.Tensor,
        idx: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        rms_eps: float,
        ln_eps: float,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
        latent_pool: torch.Tensor,
        index_pool: torch.Tensor,
        tokens_per_block: int,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            x = x.float()
            return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps) * w.float()).to(
                torch.bfloat16
            )

        q_dim, head_dim = q_norm_weight.shape[0], k_norm_weight.shape[0]
        q_resid = rms(qkv[:, :q_dim], q_norm_weight)
        latent = rms(qkv[:, q_dim:], kv_norm_weight)
        k = torch.nn.functional.layer_norm(
            idx[:, :head_dim].float(),
            (head_dim,),
            k_norm_weight.float(),
            k_norm_bias.float(),
            ln_eps,
        ).to(torch.bfloat16)
        packed = torch.cat([k, idx[:, head_dim : 2 * head_dim]], dim=-1)
        tokens = positions.shape[1]
        for r in range(positions.numel()):
            pos = int(positions.view(-1)[r])
            slot = int(block_tables[r // tokens, pos // tokens_per_block])
            off = pos % tokens_per_block
            latent_pool[slot, off] = (
                (latent[r].float() * scale).clamp(-448, 448).to(latent_pool.dtype)
            )
            index_pool[slot, off, : 2 * head_dim] = packed[r].to(index_pool.dtype)
        return q_resid

    def is_valid(
        self,
        qkv: torch.Tensor,
        idx: torch.Tensor,
        q_norm_weight: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        rms_eps: float,
        ln_eps: float,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
        latent_pool: torch.Tensor,
        index_pool: torch.Tensor,
        tokens_per_block: int,
        scale: torch.Tensor,
    ) -> None:
        for t in (qkv, idx):
            assert t.dtype == torch.bfloat16 and t.stride(-1) == 1
        kv_dim, head_dim = kv_norm_weight.shape[0], k_norm_weight.shape[0]
        assert qkv.shape[1] == q_norm_weight.shape[0] + kv_dim
        for n in (kv_dim, head_dim):
            assert n & (n - 1) == 0, f"width {n} must be a power of two"
        assert idx.shape[1] >= 2 * head_dim and k_norm_bias.shape == k_norm_weight.shape
        # The kernel reads position r at offset r and the table of row r at row r // T.
        assert positions.dim() == 2 and positions.is_contiguous()
        assert not positions.is_floating_point(), f"positions are integers; got {positions.dtype}"
        assert qkv.shape[0] == idx.shape[0] == positions.numel()
        assert block_tables.dim() == 2 and block_tables.stride(-1) == 1
        assert block_tables.shape[0] >= positions.shape[0]
        # Only the FP8 cache's write (scale, clamp, E4M3) is compiled in.
        assert latent_pool.dtype == torch.float8_e4m3fn, f"latent pool {latent_pool.dtype}"
        assert scale.dtype == torch.float32 and scale.numel() == 1
        assert latent_pool.dim() == 3 and latent_pool.shape[1:] == (tokens_per_block, kv_dim)
        assert latent_pool.stride(-1) == 1
        assert index_pool.dtype == torch.bfloat16 and index_pool.dim() == 3
        assert index_pool.shape[1] == tokens_per_block and index_pool.stride(-1) == 1
        assert index_pool.shape[-1] >= 2 * head_dim and tokens_per_block > 0
        if positions.numel() == 0:
            return
        pos = positions.long()
        # A position past the table's width reads the next table's pages as its own.
        assert int(pos.min()) >= 0
        assert int(pos.max()) < block_tables.shape[1] * tokens_per_block, (
            f"positions up to {int(pos.max())} exceed the tables' "
            f"{block_tables.shape[1] * tokens_per_block} positions"
        )
        # Two rows on one cache position race on the cache row.
        assert bool((pos.sort(dim=1).values.diff(dim=1) != 0).all()), "a position is written twice"
        slots = block_tables[: positions.shape[0]].gather(1, pos // tokens_per_block).long()
        # A slot outside the pools is written, not refused.
        assert int(slots.min()) >= 0 and int(slots.max()) < min(
            latent_pool.shape[0], index_pool.shape[0]
        ), "a table entry addresses a slot past the pools"


mla_prologue = MlaPrologue()
