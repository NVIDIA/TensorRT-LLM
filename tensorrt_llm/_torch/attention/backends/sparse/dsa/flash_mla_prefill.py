# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashMLA sparse prefill over a paged per-tensor FP8 (or BF16) MLA cache.

FlashMLA's sparse prefill kernel takes a BF16 KV tensor and runs the absorbed head group padded to 128 heads at
several times the tensor-core rate TRTLLM-Gen reaches on head groups it does not tile natively (Kimi K3.1's 96 heads
under attention DP). For an FP8 cache, this module gathers the layer's latent KV rows of the context requests out of
the paged pool into one compact BF16 buffer (request ``i`` at ``ctx_kv_indptr[i]``, dequantized with the per-tensor
KV scale) and rebases the indexer's request-local top-k onto it.

The attention numerics are FlashMLA's (BF16 queries over BF16 copies of the FP8 KV values); the selected top-k rows
are the same as on the TRTLLM-Gen path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

from .metadata import DSAtrtllmAttentionMetadata

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.mla import MLA

try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

FLASH_MLA_HEADS = 128
_LATENT_COLS = 64  # column tile of the KV gather (576 = 9 tiles)


@triton.jit
def _gather_context_kv_kernel(
    pool_ptr,  # [num_pool_rows, D] fp8 (as uint8) or bf16
    block_table_ptr,  # int32 [num_contexts, bt_stride0]
    kv_indptr_ptr,  # int64 [num_contexts + 1]
    scale_ptr,  # fp32 [1] dequant scale
    out_ptr,  # bf16 [num_rows, D]: rows [row_lo, row_lo + num_rows) of the compact layout
    row_lo,
    num_rows,
    num_contexts,
    bt_stride0,
    page_stride,  # rows per physical page (page_index_scale * tokens_per_block)
    layer_rows,  # layer_offset * tokens_per_block
    TOKENS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
    MAX_CTX: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    o = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    rmask = o < num_rows
    r = o.to(tl.int64) + row_lo
    # request of each row: number of request ends <= row
    c = tl.arange(0, MAX_CTX)
    ends = tl.load(kv_indptr_ptr + 1 + c, mask=c < num_contexts, other=2**62)
    req = tl.sum((ends[None, :] <= r[:, None]).to(tl.int32), axis=1)
    req = tl.minimum(req, num_contexts - 1)
    start = tl.load(kv_indptr_ptr + req, mask=rmask, other=0)
    pos = r - start
    blk = tl.load(
        block_table_ptr + req.to(tl.int64) * bt_stride0 + pos // TOKENS_PER_BLOCK,
        mask=rmask,
        other=0,
    )
    src = blk.to(tl.int64) * page_stride + layer_rows + pos % TOKENS_PER_BLOCK
    m = rmask[:, None] & (d[None, :] < D)
    x = tl.load(pool_ptr + src[:, None] * D + d[None, :], mask=m, other=0)
    if IS_FP8:
        y = x.to(tl.float8e4nv, bitcast=True).to(tl.float32) * tl.load(scale_ptr)
    else:
        y = x.to(tl.float32)
    tl.store(out_ptr + o.to(tl.int64)[:, None] * D + d[None, :], y.to(tl.bfloat16), mask=m)


@triton.jit
def _copy_head_rows_kernel(
    src_ptr,  # bf16 [n, heads, *] (row = one token's head)
    dst_ptr,  # bf16 [n, *, *]
    num_rows,  # n * heads
    heads,
    src_tok_stride,
    src_head_stride,
    dst_tok_stride,
    dst_head_stride,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    m = r < num_rows
    t = (r // heads).to(tl.int64)
    h = (r % heads).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(
        src_ptr + (t * src_tok_stride + h * src_head_stride)[:, None] + d[None, :], mask=m[:, None]
    )
    tl.store(
        dst_ptr + (t * dst_tok_stride + h * dst_head_stride)[:, None] + d[None, :],
        x,
        mask=m[:, None],
    )


def copy_head_rows(src: torch.Tensor, dst: torch.Tensor) -> None:
    """``dst.copy_(src)`` for ``[n, heads, D]`` views whose last dim is contiguous (strided token / head
    dims): one 16-byte-vectorized pass instead of the generic strided copy."""
    n, heads, d = src.shape
    if d & (d - 1) or src.stride(-1) != 1 or dst.stride(-1) != 1:
        dst.copy_(src)
        return
    rows = n * heads
    if rows == 0:
        return
    block_r = 64
    _copy_head_rows_kernel[(triton.cdiv(rows, block_r),)](
        src,
        dst,
        rows,
        heads,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        D=d,
        BLOCK_R=block_r,
    )


@triton.jit
def _rebase_topk_kernel(
    topk_ptr,  # int32 [n, K] request-local positions, < 0 = none
    req_idx_ptr,  # int32 [n] context request of each row
    kv_indptr_ptr,  # int64 [num_contexts + 1]
    out_ptr,  # int32 [n, K] rows of the compact KV buffer (minus row_lo), -1 = none
    row_lo,
    topk_stride,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    km = k < K
    req = tl.load(req_idx_ptr + t)
    start = tl.load(kv_indptr_ptr + req)
    length = tl.load(kv_indptr_ptr + req + 1) - start
    v = tl.load(topk_ptr + t.to(tl.int64) * topk_stride + k, mask=km, other=-1).to(tl.int64)
    ok = (v >= 0) & (v < length)
    out = tl.where(ok, v + start - row_lo, -1)
    tl.store(out_ptr + t.to(tl.int64) * K + k, out.to(tl.int32), mask=km)


def gather_context_kv_bf16(
    mla: "MLA",
    attn_metadata: DSAtrtllmAttentionMetadata,
    row_lo: int = 0,
    row_hi: Optional[int] = None,
) -> torch.Tensor:
    """This layer's latent KV rows ``[row_lo, row_hi)`` of the context requests laid out back to back
    (request ``i`` at ``ctx_kv_indptr[i]``), ``[row_hi - row_lo, 1, head_dim]`` BF16."""
    attn_metadata._ensure_pool_view_cached()
    pool = attn_metadata._cached_pool_view
    if row_hi is None:
        row_hi = attn_metadata.num_ctx_mla_kv_tokens
    num_rows = row_hi - row_lo
    num_contexts = attn_metadata.num_contexts
    head_dim = pool.shape[-1]
    local_layer_idx = mla.mqa.get_local_layer_idx(attn_metadata)
    page_scale, layer_offset = attn_metadata.kv_cache_manager.get_primary_pool_page_index_params(
        local_layer_idx
    )
    tpb = attn_metadata._cached_tokens_per_block
    block_table = attn_metadata._cached_block_table_ctx
    is_fp8 = pool.dtype != torch.bfloat16
    out = torch.empty(num_rows, 1, head_dim, dtype=torch.bfloat16, device=pool.device)
    block_r = 64
    grid = (triton.cdiv(num_rows, block_r), triton.cdiv(head_dim, _LATENT_COLS))
    _gather_context_kv_kernel[grid](
        pool.view(torch.uint8) if is_fp8 else pool,
        block_table,
        attn_metadata.ctx_kv_indptr,
        mla.mqa.kv_scale_quant_orig,
        out,
        row_lo,
        num_rows,
        num_contexts,
        block_table.stride(0),
        page_scale * tpb,
        layer_offset * tpb,
        TOKENS_PER_BLOCK=tpb,
        D=head_dim,
        MAX_CTX=triton.next_power_of_2(max(num_contexts, 2)),
        BLOCK_R=block_r,
        BLOCK_D=_LATENT_COLS,
        IS_FP8=is_fp8,
    )
    return out


def rebase_topk_to_compact_kv(
    topk_local: torch.Tensor,
    req_idx: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    row_lo: int = 0,
) -> torch.Tensor:
    """Request-local top-k ``[n, K]`` -> rows of a ``gather_context_kv_bf16`` buffer that starts at compact row
    ``row_lo``, ``[n, 1, K]`` (-1 = none)."""
    n, k = topk_local.shape
    out = torch.empty(n, 1, k, dtype=torch.int32, device=topk_local.device)
    if n == 0:
        return out
    block_k = 512
    _rebase_topk_kernel[(n, triton.cdiv(k, block_k))](
        topk_local,
        req_idx,
        attn_metadata.ctx_kv_indptr,
        out,
        row_lo,
        topk_local.stride(0),
        K=k,
        BLOCK_K=block_k,
    )
    return out


def flash_mla_sparse_prefill_rows(
    mla: "MLA",
    q_heads: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Absorb, FlashMLA sparse prefill and V projection for ``n`` query rows with all ``H`` heads given.

    q_heads ``[n, H, qk_nope + qk_rope]`` (RoPE applied), w_uk ``[H, kv_lora, qk_nope]``, w_uv
    ``[H, v_head, kv_lora]``, kv ``[rows, 1, kv_lora + qk_rope]`` BF16, indices ``[n, 1, K]``;
    writes ``out`` ``[n, H, v_head]``.
    """
    n, num_heads, _ = q_heads.shape
    kv_lora = mla.kv_lora_rank
    q_nope, q_pe = q_heads.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1)
    # The padding heads stay uninitialized: FlashMLA computes every head independently and their outputs are
    # sliced off below.
    q_concat = q_heads.new_empty(n, FLASH_MLA_HEADS, kv_lora + mla.qk_rope_head_dim)
    mla._bmm_bf16_out(
        q_nope.transpose(0, 1),
        w_uk,
        w_uk.transpose(1, 2),
        q_concat[:, :num_heads, :kv_lora].transpose(0, 1),
    )
    copy_head_rows(q_pe, q_concat[:, :num_heads, kv_lora:])
    attn_out_latent = flash_mla_sparse_fwd(q_concat, kv, indices, mla.softmax_scale)[0]
    mla._bmm_bf16_out(
        attn_out_latent[:, :num_heads, :].transpose(0, 1),
        w_uv,
        w_uv.transpose(1, 2),
        out.transpose(0, 1),
    )


def flash_mla_fp8_prefill_supported(mla: "MLA") -> bool:
    """FlashMLA sparse prefill through the compact BF16 KV gather: SM100 / SM103, BF16 absorb weights,
    per-tensor FP8 or BF16 MLA cache, at most 128 heads per call."""
    return (
        flash_mla_sparse_fwd is not None
        and get_sm_version() in (100, 103)
        and mla.k_b_proj_trans.dtype == torch.bfloat16
        and mla.v_b_proj.dtype == torch.bfloat16
        and mla.kv_cache_dtype != "fp8_ds_mla"
        and not getattr(mla.mqa, "has_fp4_kv_cache", False)
        and mla.mapping.cp_size == 1
        and mla.mqa.indexer is not None
    )


def kv_scale_is_unity(mla: "MLA") -> bool:
    """The paged-KV append of this path (mla_rope_append_paged_kv_assign_q) quantizes FP8 with scale 1; checked
    once per layer (one host read in eager warmup)."""
    cached = getattr(mla, "_shardq_unity_kv_scale", None)
    if cached is None:
        scale = mla.mqa.kv_scale_orig_quant
        cached = (
            bool(float(scale.reshape(-1)[0].item()) == 1.0)
            if torch.is_tensor(scale)
            else scale == 1.0
        )
        mla._shardq_unity_kv_scale = cached
    return cached
