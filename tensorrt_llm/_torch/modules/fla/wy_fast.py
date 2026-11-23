# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/wy_fast.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.index import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4, 8]
#         for num_stages in [2, 3, 4]
#     ],
#     key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
# )
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(
            tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta_offsets_0 = (i_t * BT) + tl.arange(0, BT)
    p_beta = (beta + bos * H + i_h) + p_beta_offsets_0.to(tl.int64) * (H)
    p_beta_mask_0 = (p_beta_offsets_0 >= 0) & (p_beta_offsets_0 < (T))
    p_g_offsets_0 = (i_t * BT) + tl.arange(0, BT)
    p_g = (g + (bos * H + i_h)) + p_g_offsets_0.to(tl.int64) * (H)
    p_g_mask_0 = (p_g_offsets_0 >= 0) & (p_g_offsets_0 < (T))
    p_A_offsets_0 = (i_t * BT) + tl.arange(0, BT)
    p_A_offsets_1 = (0) + tl.arange(0, BT)
    p_A = (A + (bos * H + i_h) * BT) + p_A_offsets_0[:, None].to(
        tl.int64) * (H * BT) + p_A_offsets_1[None, :].to(tl.int64) * (1)
    p_A_mask_0 = (p_A_offsets_0 >= 0) & (p_A_offsets_0 < (T))
    p_A_mask_1 = (p_A_offsets_1 >= 0) & (p_A_offsets_1 < (BT))
    b_beta = tl.load(p_beta, mask=p_beta_mask_0, other=0)
    b_A = tl.load(p_A, mask=p_A_mask_0[:, None] & p_A_mask_1[None, :], other=0)
    b_g = tl.exp(tl.load(p_g, mask=p_g_mask_0, other=0))

    for i_v in range(tl.cdiv(V, BV)):
        p_v_offsets_0 = (i_t * BT) + tl.arange(0, BT)
        p_v_offsets_1 = (i_v * BV) + tl.arange(0, BV)
        p_v = (v + (bos * H + i_h) * V) + p_v_offsets_0[:, None].to(
            tl.int64) * (H * V) + p_v_offsets_1[None, :].to(tl.int64) * (1)
        p_v_mask_0 = (p_v_offsets_0 >= 0) & (p_v_offsets_0 < (T))
        p_v_mask_1 = (p_v_offsets_1 >= 0) & (p_v_offsets_1 < (V))
        p_u_offsets_0 = (i_t * BT) + tl.arange(0, BT)
        p_u_offsets_1 = (i_v * BV) + tl.arange(0, BV)
        p_u = (u + (bos * H + i_h) * V) + p_u_offsets_0[:, None].to(
            tl.int64) * (H * V) + p_u_offsets_1[None, :].to(tl.int64) * (1)
        p_u_mask_0 = (p_u_offsets_0 >= 0) & (p_u_offsets_0 < (T))
        p_u_mask_1 = (p_u_offsets_1 >= 0) & (p_u_offsets_1 < (V))
        b_v = tl.load(p_v,
                      mask=p_v_mask_0[:, None] & p_v_mask_1[None, :],
                      other=0)
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u,
                 b_u.to(p_u.dtype.element_ty),
                 mask=p_u_mask_0[:, None] & p_u_mask_1[None, :])

    for i_k in range(tl.cdiv(K, BK)):
        p_k_offsets_0 = (i_t * BT) + tl.arange(0, BT)
        p_k_offsets_1 = (i_k * BK) + tl.arange(0, BK)
        p_k = (k + (bos * Hg + i_h //
                    (H // Hg)) * K) + p_k_offsets_0[:, None].to(tl.int64) * (
                        Hg * K) + p_k_offsets_1[None, :].to(tl.int64) * (1)
        p_k_mask_0 = (p_k_offsets_0 >= 0) & (p_k_offsets_0 < (T))
        p_k_mask_1 = (p_k_offsets_1 >= 0) & (p_k_offsets_1 < (K))
        p_w_offsets_0 = (i_t * BT) + tl.arange(0, BT)
        p_w_offsets_1 = (i_k * BK) + tl.arange(0, BK)
        p_w = (w + (bos * H + i_h) * K) + p_w_offsets_0[:, None].to(
            tl.int64) * (H * K) + p_w_offsets_1[None, :].to(tl.int64) * (1)
        p_w_mask_0 = (p_w_offsets_0 >= 0) & (p_w_offsets_0 < (T))
        p_w_mask_1 = (p_w_offsets_1 >= 0) & (p_w_offsets_1 < (K))
        b_k = tl.load(p_k,
                      mask=p_k_mask_0[:, None] & p_k_mask_1[None, :],
                      other=0)
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w,
                 b_w.to(p_w.dtype.element_ty),
                 mask=p_w_mask_0[:, None] & p_w_mask_1[None, :])


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = (prepare_chunk_indices(cu_seqlens, BT)
                     if cu_seqlens is not None else None)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )
    return w, u


fwd_recompute_w_u = recompute_w_u_fwd
