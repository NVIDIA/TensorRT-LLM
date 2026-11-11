# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-row work between the sparse MLA input GEMMs and the query projections, in one launch.

For each generation row: the RMSNorms of the q_a and kv_a halves of the fused input GEMM
output, the LayerNorm of the indexer key, and the paged-cache writes of the normalized
latent (FP8: ``clamp(latent * scale, +-448)`` as E4M3) and of the indexer row
``[k | gate]``. It returns the normalized q_a rows (``q_resid``) for the q_b and indexer
query projections. The module path runs the same math as RMSNorm, RMSNorm, LayerNorm,
cat and the append kernel (five launches).
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
    FP8: tl.constexpr,
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
    if FP8:
        scaled = kv.to(tl.float32) * tl.load(scale_ptr)
        tl.store(latent_dst, tl.clamp(scaled, -448.0, 448.0).to(latent_pool_ptr.dtype.element_ty))
    else:
        tl.store(latent_dst, kv.to(latent_pool_ptr.dtype.element_ty))
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
    (``[N, T]``, table ``block_tables[n]``) into the paged pools."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP-3 verify of one request", spec=dict(n=1, t=4)),
        Cell(why="MTP-3 verify of eight requests", spec=dict(n=8, t=4)),
    )

    def __call__(
        self,
        qkv,
        idx,
        q_norm_weight,
        kv_norm_weight,
        k_norm_weight,
        k_norm_bias,
        rms_eps,
        ln_eps,
        positions,
        block_tables,
        latent_pool,
        index_pool,
        tokens_per_block,
        scale,
    ):
        q_dim, kv_dim = q_norm_weight.shape[0], kv_norm_weight.shape[0]
        head_dim = k_norm_weight.shape[0]
        rows = positions.numel()
        q_resid = torch.empty((rows, q_dim), dtype=torch.bfloat16, device=qkv.device)
        fp8 = latent_pool.dtype == torch.float8_e4m3fn
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
            scale if fp8 else qkv,
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
            FP8=fp8,
            num_warps=4,
        )
        return q_resid

    def reference(
        self,
        qkv,
        idx,
        q_norm_weight,
        kv_norm_weight,
        k_norm_weight,
        k_norm_bias,
        rms_eps,
        ln_eps,
        positions,
        block_tables,
        latent_pool,
        index_pool,
        tokens_per_block,
        scale,
    ):
        def rms(x, w):
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
            if latent_pool.dtype == torch.float8_e4m3fn:
                latent_pool[slot, off] = (
                    (latent[r].float() * scale).clamp(-448, 448).to(latent_pool.dtype)
                )
            else:
                latent_pool[slot, off] = latent[r].to(latent_pool.dtype)
            index_pool[slot, off, : 2 * head_dim] = packed[r].to(index_pool.dtype)
        return q_resid

    def is_valid(
        self,
        qkv,
        idx,
        q_norm_weight,
        kv_norm_weight,
        k_norm_weight,
        k_norm_bias,
        rms_eps,
        ln_eps,
        positions,
        block_tables,
        latent_pool,
        index_pool,
        tokens_per_block,
        scale,
    ):
        for t in (qkv, idx):
            assert t.dtype == torch.bfloat16 and t.stride(-1) == 1
        kv_dim, head_dim = kv_norm_weight.shape[0], k_norm_weight.shape[0]
        assert qkv.shape[1] == q_norm_weight.shape[0] + kv_dim
        for n in (kv_dim, head_dim):
            assert n & (n - 1) == 0, f"width {n} must be a power of two"
        assert idx.shape[1] >= 2 * head_dim and k_norm_bias.shape == k_norm_weight.shape
        # The kernel reads position r at offset r and the table of row r at row r // T.
        assert positions.dim() == 2 and positions.is_contiguous()
        assert qkv.shape[0] == idx.shape[0] == positions.numel()
        assert block_tables.shape[0] >= positions.shape[0] and block_tables.stride(-1) == 1
        assert latent_pool.dtype in (torch.float8_e4m3fn, torch.bfloat16)
        assert latent_pool.shape[-1] == kv_dim and latent_pool.stride(-1) == 1
        if latent_pool.dtype == torch.float8_e4m3fn:
            assert scale.dtype == torch.float32 and scale.numel() == 1
        assert index_pool.dtype == torch.bfloat16 and index_pool.stride(-1) == 1
        assert index_pool.shape[-1] >= 2 * head_dim and tokens_per_block > 0


mla_prologue = MlaPrologue()
