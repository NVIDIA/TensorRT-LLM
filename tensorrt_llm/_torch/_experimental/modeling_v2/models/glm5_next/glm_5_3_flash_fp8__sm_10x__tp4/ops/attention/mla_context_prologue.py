# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-row work between the sparse MLA input GEMMs and the query projections, for packed
context rows, in one launch.

The context-row counterpart of ``mla_prologue``. For each packed context row ``r`` of a prefill
step: the RMSNorms of the q_a and kv_a halves of the fused input GEMM output, the LayerNorm of
the indexer key, and the paged-cache writes of the normalized latent (FP8: ``clamp(latent *
scale, +-448)`` as E4M3) and of the indexer row ``[k | gate]`` at cache position
``positions[r]`` of block table ``block_tables[request_ids[r]]``. It returns the normalized q_a
rows (``q_resid``) for the q_b and indexer query projections. The module path runs the same math
as RMSNorm, RMSNorm, LayerNorm, cat and the indexed cache writes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _context_prologue_kernel(
    qkv_ptr,
    idx_ptr,
    qw_ptr,
    kvw_ptr,
    lnw_ptr,
    lnb_ptr,
    q_out_ptr,
    pos_ptr,
    req_ptr,
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
    Q_DIM: tl.constexpr,
    Q_PAD: tl.constexpr,
    KV_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP8: tl.constexpr,
):
    # A prefill step packs up to max_num_tokens rows: 64-bit row offsets.
    r = tl.program_id(0).to(tl.int64)
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
    tbl = tl.load(req_ptr + r).to(tl.int64)
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


class MlaContextPrologue(OpWrapper):
    """``q_resid`` BF16 ``[T, q_lora]`` from ``qkv`` ``[T, q_lora + kv_lora]`` and ``idx`` ``[T,
    >= 2 * head_dim]`` (indexer key | gate | ...), writing the latent and indexer rows of
    position ``positions[r]`` (``[T]``) of table ``block_tables[request_ids[r]]`` into the
    paged pools. Only the ``[k | gate]`` columns of the indexer rows are written; the pool-key
    columns are ``kpool_update``'s.
    """

    ARCHS = frozenset({Arch.SM_100})
    # GLM-5.3-Flash: q_lora 1536, kv_lora 512, an indexer key of 128 (the indexer GEMM output
    # is [k | gate | 32 head weights], 288 wide), RMSNorm eps 1e-5, key LayerNorm eps 1e-6,
    # 32-token blocks of an FP8 latent cache. Every rank writes every context row of a step,
    # so the cells are whole steps: (cached, new) per context request, in batch order.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one 13,312-token prompt", spec=dict(context=((0, 13312),))),
        Cell(
            why="two requests: a chunk continuing an unaligned prefix, and a fresh prompt",
            spec=dict(context=((4099, 3000), (0, 1313))),
        ),
        Cell(
            why="a 37-token prompt in a mixed step: table 0 of the step's eight",
            spec=dict(context=((0, 37),), tables=8),
        ),
    )

    note = """
    The block table of row r is `block_tables[request_ids[r]]`, a row of the step's tables in
    batch order: context requests first, so a context request's id is its index among them.

    Rows writing the same cache position race; the module path never packs a position twice.
    The latent pool may be BF16, in which case it receives the BF16 latent; only the FP8 cache
    this target serves is certified.
    """

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
        request_ids,
        block_tables,
        latent_pool,
        index_pool,
        tokens_per_block,
        scale,
    ):
        q_dim, kv_dim = q_norm_weight.shape[0], kv_norm_weight.shape[0]
        head_dim = k_norm_weight.shape[0]
        rows = positions.shape[0]
        q_resid = torch.empty((rows, q_dim), dtype=torch.bfloat16, device=qkv.device)
        if rows == 0:
            return q_resid
        fp8 = latent_pool.dtype == torch.float8_e4m3fn
        _context_prologue_kernel[(rows,)](
            qkv,
            idx,
            q_norm_weight,
            kv_norm_weight,
            k_norm_weight,
            k_norm_bias,
            q_resid,
            positions,
            request_ids,
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
        request_ids,
        block_tables,
        latent_pool,
        index_pool,
        tokens_per_block,
        scale,
    ):
        """FP32 norms rounded once to BF16, then the indexed cache writes, in plain torch."""

        def rms(x, w):
            x = x.float()
            return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps) * w.float()).to(
                torch.bfloat16
            )

        q_dim, head_dim = q_norm_weight.shape[0], k_norm_weight.shape[0]
        q_resid = rms(qkv[:, :q_dim], q_norm_weight)
        if positions.shape[0] == 0:
            return q_resid
        latent = rms(qkv[:, q_dim:], kv_norm_weight)
        k = torch.nn.functional.layer_norm(
            idx[:, :head_dim].float(),
            (head_dim,),
            k_norm_weight.float(),
            k_norm_bias.float(),
            ln_eps,
        ).to(torch.bfloat16)
        packed = torch.cat([k, idx[:, head_dim : 2 * head_dim]], dim=-1)
        pos = positions.long()
        page = block_tables[request_ids.long(), pos // tokens_per_block].long()
        off = pos % tokens_per_block
        if latent_pool.dtype == torch.float8_e4m3fn:
            quantized = (latent.float() * scale).clamp(-448, 448).to(latent_pool.dtype)
            # Indexed writes into FP8 payloads go through their byte views.
            latent_pool.view(torch.uint8)[page, off] = quantized.view(torch.uint8)
        else:
            latent_pool[page, off] = latent.to(latent_pool.dtype)
        index_pool[page, off, : 2 * head_dim] = packed.to(index_pool.dtype)
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
        request_ids,
        block_tables,
        latent_pool,
        index_pool,
        tokens_per_block,
        scale,
    ):
        for t in (qkv, idx):
            assert t.dtype == torch.bfloat16 and t.dim() == 2 and t.stride(-1) == 1
        kv_dim, head_dim = kv_norm_weight.shape[0], k_norm_weight.shape[0]
        assert qkv.shape[1] == q_norm_weight.shape[0] + kv_dim
        for n in (kv_dim, head_dim):
            assert n & (n - 1) == 0, f"width {n} must be a power of two"
        assert idx.shape[1] >= 2 * head_dim and k_norm_bias.shape == k_norm_weight.shape
        for w in (q_norm_weight, kv_norm_weight, k_norm_weight, k_norm_bias):
            assert w.is_contiguous()
        # The kernel reads row r's position and table id at offset r.
        assert positions.dim() == 1 and positions.is_contiguous()
        assert request_ids.shape == positions.shape and request_ids.is_contiguous()
        assert qkv.shape[0] == idx.shape[0] == positions.shape[0]
        assert block_tables.dim() == 2 and block_tables.stride(-1) == 1
        assert latent_pool.dtype in (torch.float8_e4m3fn, torch.bfloat16)
        assert latent_pool.shape[-1] == kv_dim and latent_pool.stride(-1) == 1
        if latent_pool.dtype == torch.float8_e4m3fn:
            assert scale.dtype == torch.float32 and scale.numel() == 1
        assert index_pool.dtype == torch.bfloat16 and index_pool.stride(-1) == 1
        assert index_pool.shape[-1] >= 2 * head_dim and tokens_per_block > 0
        if positions.shape[0] == 0:
            return
        req = request_ids.long()
        assert 0 <= int(req.min()) and int(req.max()) < block_tables.shape[0]
        pos = positions.long()
        assert int(pos.min()) >= 0
        assert int(pos.max()) < block_tables.shape[1] * tokens_per_block, (
            "a position past the table width reads the next table's pages"
        )
        # Two rows on one (table, position) race on the cache row.
        key = req * (block_tables.shape[1] * tokens_per_block) + pos
        assert key.unique().numel() == key.numel(), "a cache position is written twice"


mla_context_prologue = MlaContextPrologue()
