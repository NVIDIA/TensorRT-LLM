# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused paged-cache scatter for the MiniMax-M3 MSA backend.

One Triton launch writes a layer's new-token main K, main V, and (sparse
layers) index-K into their paged HND caches at the step's write slots.
The legacy path costs three aten advanced-indexing writes per layer plus
their index preprocessing; at 60 layers per forward step, all captured
into decode CUDA graphs, the launch count dominates the cost. The kernel
derives each token's (page, within-page) split from ``out_cache_loc``
in-register, so it needs no precomputed index tensors at all.

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
    page = slot // tokens_per_block
    within = slot % tokens_per_block
    d = tl.arange(0, D)
    for h in tl.static_range(H):
        k_vals = tl.load(k_src + t * k_src_row_stride + h * D + d)
        v_vals = tl.load(v_src + t * v_src_row_stride + h * D + d)
        k_dst = k_cache + page * kc_stride_page + h * kc_stride_head + within * kc_stride_tok + d
        v_dst = v_cache + page * vc_stride_page + h * vc_stride_head + within * vc_stride_tok + d
        tl.store(k_dst, k_vals.to(k_cache.dtype.element_ty))
        tl.store(v_dst, v_vals.to(v_cache.dtype.element_ty))
    if HAS_IDX:
        i_vals = tl.load(idx_src + t * idx_src_row_stride + d)
        i_dst = idx_cache + page * ic_stride_page + within * ic_stride_tok + d
        tl.store(i_dst, i_vals.to(idx_cache.dtype.element_ty))


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


__all__ = ["fused_write_layer_caches"]
