# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused paged-cache scatter for the MiniMax-M3 MSA backend.

One Triton launch writes a layer's new-token main K, main V, and (sparse
layers) index-K into their paged HND caches at the step's write slots.
Writing them separately costs three aten advanced-indexing writes per
layer plus their index preprocessing; at 60 layers per forward step, all
captured into decode CUDA graphs, the launch count dominates the cost.
The kernel derives each token's (page, within-page) split from
out_cache_loc in-register, so it needs no precomputed index tensors.

Sources may be strided row views (slices of the fused QKV projection);
only the innermost [num_heads * head_dim] extent must be contiguous.
Stores cast to the cache dtype, which folds the FP8 KV-cache cast in.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_paged_scatter_kernel(
    k_src,
    v_src,
    idx_src,
    k_cache,
    v_cache,
    idx_cache,
    out_cache_loc,
    k_src_row_stride,
    v_src_row_stride,
    idx_src_row_stride,
    kc_stride_page,
    kc_stride_head,
    kc_stride_tok,
    vc_stride_page,
    vc_stride_head,
    vc_stride_tok,
    ic_stride_page,
    ic_stride_tok,
    tokens_per_block,
    H: tl.constexpr,
    D: tl.constexpr,
    HAS_IDX: tl.constexpr,
):
    # int64 throughout: t * row_stride can exceed 2^31 elements on large
    # eager prefill steps (num_tokens up to max_num_tokens times the fused
    # QKV row stride), and the slot * page-stride products likewise.
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(out_cache_loc + t).to(tl.int64)
    valid = slot >= 0
    page = slot // tokens_per_block
    within = slot % tokens_per_block
    d = tl.arange(0, D)
    for h in tl.static_range(H):
        k_vals = tl.load(k_src + t * k_src_row_stride + h * D + d)
        v_vals = tl.load(v_src + t * v_src_row_stride + h * D + d)
        k_dst = k_cache + page * kc_stride_page + h * kc_stride_head + within * kc_stride_tok + d
        v_dst = v_cache + page * vc_stride_page + h * vc_stride_head + within * vc_stride_tok + d
        tl.store(k_dst, k_vals.to(k_cache.dtype.element_ty), mask=valid)
        tl.store(v_dst, v_vals.to(v_cache.dtype.element_ty), mask=valid)
    if HAS_IDX:
        i_vals = tl.load(idx_src + t * idx_src_row_stride + d)
        i_dst = idx_cache + page * ic_stride_page + within * ic_stride_tok + d
        tl.store(i_dst, i_vals.to(idx_cache.dtype.element_ty), mask=valid)


@triton.jit
def _fused_subpaged_scatter_kernel(
    k_src,
    v_src,
    k_cache,
    v_cache,
    out_cache_loc,
    k_src_row_stride,
    v_src_row_stride,
    kc_stride_page,
    kc_stride_subpage,
    kc_stride_head,
    kc_stride_tok,
    vc_stride_page,
    vc_stride_subpage,
    vc_stride_head,
    vc_stride_tok,
    logical_tokens_per_block,
    physical_tokens_per_block,
    H: tl.constexpr,
    D: tl.constexpr,
):
    """Scatter FP8 K/V into P32 pages inside one logical P128 slot."""
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(out_cache_loc + t).to(tl.int64)
    valid = slot >= 0
    page = slot // logical_tokens_per_block
    logical_within = slot % logical_tokens_per_block
    subpage = logical_within // physical_tokens_per_block
    within = logical_within % physical_tokens_per_block
    d = tl.arange(0, D)
    for h in tl.static_range(H):
        src = t * k_src_row_stride + h * D + d
        k_vals = tl.load(k_src + src)
        v_vals = tl.load(v_src + t * v_src_row_stride + h * D + d)
        k_dst = (
            k_cache
            + page * kc_stride_page
            + subpage * kc_stride_subpage
            + h * kc_stride_head
            + within * kc_stride_tok
            + d
        )
        v_dst = (
            v_cache
            + page * vc_stride_page
            + subpage * vc_stride_subpage
            + h * vc_stride_head
            + within * vc_stride_tok
            + d
        )
        tl.store(k_dst, k_vals.to(k_cache.dtype.element_ty), mask=valid)
        tl.store(v_dst, v_vals.to(v_cache.dtype.element_ty), mask=valid)


@triton.jit
def _fused_nvfp4_paged_scatter_kernel(
    k_data_src,
    v_data_src,
    k_scale_src,
    v_scale_src,
    idx_src,
    k_data_cache,
    v_data_cache,
    k_scale_cache,
    v_scale_cache,
    idx_cache,
    out_cache_loc,
    idx_src_row_stride,
    kdc_stride_page,
    kdc_stride_subpage,
    kdc_stride_head,
    kdc_stride_tok,
    vdc_stride_page,
    vdc_stride_subpage,
    vdc_stride_head,
    vdc_stride_tok,
    ksc_stride_page,
    ksc_stride_subpage,
    ksc_stride_head,
    vsc_stride_page,
    vsc_stride_subpage,
    vsc_stride_head,
    ic_stride_page,
    ic_stride_tok,
    logical_tokens_per_block,
    physical_tokens_per_block,
    H: tl.constexpr,
    D_PACKED: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    IDX_D: tl.constexpr,
    HAS_IDX: tl.constexpr,
):
    """Scatter already-quantized NVFP4 data and its two scale layouts."""
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(out_cache_loc + t).to(tl.int64)
    valid = slot >= 0
    page = slot // logical_tokens_per_block
    logical_within = slot % logical_tokens_per_block
    subpage = logical_within // physical_tokens_per_block
    within = logical_within % physical_tokens_per_block
    d = tl.arange(0, D_PACKED)
    s = tl.arange(0, SCALE_COLS)
    for h in tl.static_range(H):
        src_data_base = (t * H + h) * D_PACKED
        k_data = tl.load(k_data_src + src_data_base + d)
        v_data = tl.load(v_data_src + src_data_base + d)
        k_data_dst = (
            k_data_cache
            + page * kdc_stride_page
            + subpage * kdc_stride_subpage
            + h * kdc_stride_head
            + within * kdc_stride_tok
            + d
        )
        v_data_dst = (
            v_data_cache
            + page * vdc_stride_page
            + subpage * vdc_stride_subpage
            + h * vdc_stride_head
            + within * vdc_stride_tok
            + d
        )
        tl.store(k_data_dst, k_data, mask=valid)
        tl.store(v_data_dst, v_data, mask=valid)

        src_scale_base = (t * H + h) * SCALE_COLS
        k_scale = tl.load(k_scale_src + src_scale_base + s)
        v_scale = tl.load(v_scale_src + src_scale_base + s)
        # K uses token-major linear scale bytes. V uses vLLM's 4x4
        # token-quad order: [token//4, scale_col, token%4].
        k_scale_offset = within * SCALE_COLS + s
        v_scale_offset = (within // 4) * (4 * SCALE_COLS) + 4 * s + (within % 4)
        tl.store(
            k_scale_cache
            + page * ksc_stride_page
            + subpage * ksc_stride_subpage
            + h * ksc_stride_head
            + k_scale_offset,
            k_scale,
            mask=valid,
        )
        tl.store(
            v_scale_cache
            + page * vsc_stride_page
            + subpage * vsc_stride_subpage
            + h * vsc_stride_head
            + v_scale_offset,
            v_scale,
            mask=valid,
        )

    if HAS_IDX:
        i = tl.arange(0, IDX_D)
        i_vals = tl.load(idx_src + t * idx_src_row_stride + i)
        i_dst = idx_cache + page * ic_stride_page + logical_within * ic_stride_tok + i
        tl.store(i_dst, i_vals.to(idx_cache.dtype.element_ty), mask=valid)


def _row_stride_if_fusable(src: torch.Tensor, inner: int) -> Optional[int]:
    """Row stride (elements) if `src` is a [T, inner] row view with contiguous
    rows (e.g. a column slice of the fused QKV projection); None otherwise."""
    if src.dim() != 2 or src.shape[1] != inner or src.stride(1) != 1:
        return None
    return src.stride(0)


def fused_write_layer_caches(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_cache: Optional[torch.Tensor],
    out_cache_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_k: Optional[torch.Tensor],
) -> bool:
    """Fused single-launch write of new-token K/V (+index-K) into paged HND
    caches. Returns False when a layout precondition fails, so the caller can
    keep the legacy per-cache writes.

    `k_cache`/`v_cache` are [num_pages, num_kv_heads, tokens_per_block,
    head_dim] HND views; `idx_cache` is the MQA index-K view with head dim 1.
    `k`/`v` are the layer's new-token values, [T, H*D] or [T, H, D] row views;
    `idx_k` is [T, D] or [T, 1, D].
    """
    if not (k.is_cuda and k_cache.is_cuda):
        return False
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        return False
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        return False
    num_pages, num_heads, tokens_per_block, head_dim = k_cache.shape
    if (head_dim & (head_dim - 1)) != 0:
        return False
    inner = num_heads * head_dim
    k_stride = _row_stride_if_fusable(k, inner)
    v_stride = _row_stride_if_fusable(v, inner)
    if k_stride is None or v_stride is None:
        return False

    has_idx = idx_k is not None
    idx_stride = 0
    ic_stride_page = 0
    ic_stride_tok = 0
    if has_idx:
        if idx_cache is None or idx_cache.dim() != 4 or idx_cache.stride(-1) != 1:
            return False
        if int(idx_cache.shape[1]) != 1 or int(idx_cache.shape[3]) != head_dim:
            return False
        if int(idx_cache.shape[2]) != tokens_per_block:
            return False
        idx_stride = _row_stride_if_fusable(idx_k, head_dim)
        if idx_stride is None:
            return False
        ic_stride_page = idx_cache.stride(0)
        ic_stride_tok = idx_cache.stride(2)

    num_tokens = int(out_cache_loc.shape[0])
    if num_tokens == 0:
        return True

    _fused_paged_scatter_kernel[(num_tokens,)](
        k,
        v,
        idx_k if has_idx else k,  # unused when HAS_IDX=False
        k_cache,
        v_cache,
        idx_cache if has_idx else k_cache,  # unused when HAS_IDX=False
        out_cache_loc,
        k_stride,
        v_stride,
        idx_stride,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        ic_stride_page,
        ic_stride_tok,
        tokens_per_block,
        H=num_heads,
        D=head_dim,
        HAS_IDX=has_idx,
        num_warps=2,
    )
    return True


def fused_write_subpaged_layer_caches(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    """Write ordinary K/V into physical sub-pages of a logical cache block.

    Hybrid M3 stores dense and shared-Eagle K/V as FP8 P32 pages while its
    allocator and request lifecycle remain P128.  ``k_cache``/``v_cache`` are
    ``[logical_slots, pages_per_role, heads, P32, D]`` zero-copy views.
    """
    if not (k.is_cuda and k_cache.is_cuda):
        return False
    if k_cache.dim() != 5 or v_cache.shape != k_cache.shape:
        return False
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        return False
    _num_slots, pages_per_role, num_heads, physical_page, head_dim = k_cache.shape
    if pages_per_role <= 0 or physical_page <= 0 or (head_dim & (head_dim - 1)) != 0:
        return False
    inner = num_heads * head_dim
    k_stride = _row_stride_if_fusable(k, inner)
    v_stride = _row_stride_if_fusable(v, inner)
    if k_stride is None or v_stride is None:
        return False
    num_tokens = int(out_cache_loc.shape[0])
    if num_tokens == 0:
        return True

    _fused_subpaged_scatter_kernel[(num_tokens,)](
        k,
        v,
        k_cache,
        v_cache,
        out_cache_loc,
        k_stride,
        v_stride,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        pages_per_role * physical_page,
        physical_page,
        H=num_heads,
        D=head_dim,
        num_warps=2,
    )
    return True


def fused_write_layer_caches_nvfp4(
    k_data_cache: torch.Tensor,
    v_data_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    idx_cache: Optional[torch.Tensor],
    out_cache_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_k: Optional[torch.Tensor],
    kv_scale_orig_quant: torch.Tensor,
) -> bool:
    """Quantize BF16 K/V and scatter NVFP4 data/scales plus optional index-K.

    ``torch.ops.trtllm.fp4_quantize`` supplies the production E2M1 rounding
    and E4M3 scale encoding.  This module only performs the M3-specific paged
    placement: linear K scale bytes and vLLM-compatible 4x4 V scale bytes.
    """
    if not (k.is_cuda and k_data_cache.is_cuda):
        return False
    caches = (k_data_cache, v_data_cache, k_scale_cache, v_scale_cache)
    if any(cache.dim() not in (4, 5) or cache.stride(-1) != 1 for cache in caches):
        return False
    if any(cache.dim() != k_data_cache.dim() for cache in caches):
        return False
    if k_data_cache.dim() == 4:
        num_pages, num_heads, physical_tokens_per_block, packed_dim = k_data_cache.shape
        pages_per_role = 1
        data_subpage_stride = scale_subpage_stride = 0
    else:
        (
            num_pages,
            pages_per_role,
            num_heads,
            physical_tokens_per_block,
            packed_dim,
        ) = k_data_cache.shape
        data_subpage_stride = k_data_cache.stride(1)
        scale_subpage_stride = k_scale_cache.stride(1)
    if v_data_cache.shape != k_data_cache.shape:
        return False
    logical_tokens_per_block = pages_per_role * physical_tokens_per_block
    logical_dim = packed_dim * 2
    scale_cols = logical_dim // 16
    expected_scale_shape = (
        (num_pages, num_heads, physical_tokens_per_block, scale_cols)
        if pages_per_role == 1
        else (num_pages, pages_per_role, num_heads, physical_tokens_per_block, scale_cols)
    )
    if tuple(k_scale_cache.shape) != expected_scale_shape:
        return False
    if tuple(v_scale_cache.shape) != expected_scale_shape:
        return False
    if physical_tokens_per_block % 4 != 0 or scale_cols % 4 != 0:
        return False

    inner = num_heads * logical_dim
    if _row_stride_if_fusable(k, inner) is None or _row_stride_if_fusable(v, inner) is None:
        return False
    if kv_scale_orig_quant.dtype != torch.float32 or kv_scale_orig_quant.numel() < 3:
        return False

    has_idx = idx_k is not None
    idx_stride = 0
    idx_dim = logical_dim
    ic_stride_page = 0
    ic_stride_tok = 0
    if has_idx:
        if idx_cache is None or idx_cache.dim() != 4 or idx_cache.stride(-1) != 1:
            return False
        if int(idx_cache.shape[1]) != 1 or int(idx_cache.shape[2]) != logical_tokens_per_block:
            return False
        idx_dim = int(idx_cache.shape[3])
        idx_stride = _row_stride_if_fusable(idx_k, idx_dim)
        if idx_stride is None:
            return False
        ic_stride_page = idx_cache.stride(0)
        ic_stride_tok = idx_cache.stride(2)

    num_tokens = int(out_cache_loc.shape[0])
    if num_tokens == 0:
        return True
    k_rows = k.reshape(num_tokens, num_heads, logical_dim).contiguous()
    v_rows = v.reshape(num_tokens, num_heads, logical_dim).contiguous()
    k_data, k_scale = torch.ops.trtllm.fp4_quantize(
        k_rows,
        kv_scale_orig_quant[1:2],
        16,
        False,
        False,
    )
    v_data, v_scale = torch.ops.trtllm.fp4_quantize(
        v_rows,
        kv_scale_orig_quant[2:3],
        16,
        False,
        False,
    )
    # Triton treats these as opaque packed bytes.  Reinterpret custom FP4 and
    # E4M3 containers rather than numerically converting them.
    k_data = k_data.view(torch.uint8)
    v_data = v_data.view(torch.uint8)
    k_scale = k_scale.view(torch.uint8)
    v_scale = v_scale.view(torch.uint8)

    _fused_nvfp4_paged_scatter_kernel[(num_tokens,)](
        k_data,
        v_data,
        k_scale,
        v_scale,
        idx_k if has_idx else k,
        k_data_cache.view(torch.uint8),
        v_data_cache.view(torch.uint8),
        k_scale_cache,
        v_scale_cache,
        idx_cache if has_idx else k_data_cache,
        out_cache_loc,
        idx_stride,
        k_data_cache.stride(0),
        data_subpage_stride,
        k_data_cache.stride(-3),
        k_data_cache.stride(-2),
        v_data_cache.stride(0),
        0 if pages_per_role == 1 else v_data_cache.stride(1),
        v_data_cache.stride(-3),
        v_data_cache.stride(-2),
        k_scale_cache.stride(0),
        scale_subpage_stride,
        k_scale_cache.stride(-3),
        v_scale_cache.stride(0),
        0 if pages_per_role == 1 else v_scale_cache.stride(1),
        v_scale_cache.stride(-3),
        ic_stride_page,
        ic_stride_tok,
        logical_tokens_per_block,
        physical_tokens_per_block,
        H=num_heads,
        D_PACKED=packed_dim,
        SCALE_COLS=scale_cols,
        IDX_D=idx_dim,
        HAS_IDX=has_idx,
        num_warps=2,
    )
    return True


__all__ = [
    "fused_write_layer_caches",
    "fused_write_layer_caches_nvfp4",
    "fused_write_subpaged_layer_caches",
]
