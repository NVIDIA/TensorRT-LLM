# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/fla/chunk_delta_h.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.index import (prepare_chunk_indices,
                                                   prepare_chunk_offsets)
from tensorrt_llm._torch.modules.fla.op import exp, safe_exp
from tensorrt_llm._torch.modules.fla.utils import is_nvidia_hopper

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]


def _save_autotune_state(args: dict, reset_only: bool = False) -> None:
    """Snapshot the indexed state slots mutated by each autotune trial."""
    if reset_only or args["h0"] is None or args["h0_i"] is None:
        return
    # Match N in chunk_gated_delta_rule_fwd_h and the kernel's
    # state_index = tl.load(h0_i + i_n) for i_n < N.
    cu_seqlens = args["cu_seqlens"]
    num_seqs = (len(cu_seqlens) -
                1 if cu_seqlens is not None else args["k"].shape[0])
    indices = args["h0_i"][:num_seqs].to(torch.long)
    # Only these slots are updated by the indexed kernel. The pool may be
    # much larger than the active batch, so never clone the whole pool.
    saved_state = args["h0"].index_select(0, indices)
    args["_gdn_autotune_state"] = indices, saved_state


def _restore_autotune_state(args: dict, exception: Exception | None) -> None:
    """Restore the trial's indexed slots, including after a kernel exception."""
    saved = args.pop("_gdn_autotune_state", None)
    if saved is not None:
        indices, state = saved
        args["h0"].index_copy_(0, indices, state)


@triton.heuristics({
    "USE_G": lambda args: args["g"] is not None,
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "USE_INDEXED_STATE": lambda args: args["h0_i"] is not None,
    "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=nw, num_stages=ns)
        for nw in NUM_WARPS for ns in [2, 3, 4] for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT", "USE_G"],
    # Custom hooks replace Triton's reset_to_zero/restore_value handling.
    # Back up active slots only: h0 can be much larger than the batch.
    pre_hook=_save_autotune_state,
    post_hook=_restore_autotune_state,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    h0_i,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    stride_h0,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_INDEXED_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute chunk boundary states and optional final state updates."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(
            tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * Hg + i_h // (H // Hg)) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INDEXED_STATE:
        state_index = tl.load(h0_i + i_n).to(tl.int64)
        h0 = h0 + state_index * stride_h0
        ht = h0
    if USE_INITIAL_STATE:
        h0 = h0 + (i_h if USE_INDEXED_STATE else i_nh) * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V
    elif USE_INDEXED_STATE:
        ht = ht + i_h * K * V

    # load initial state
    # Pool layout [slots, HV, V, K], K innermost: each (K, V) tile has K stride 1
    # and V stride K.
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    m_h1 = m_k1[:, None] & m_v[None, :]
    m_h2 = m_k2[:, None] & m_v[None, :]
    m_h3 = m_k3[:, None] & m_v[None, :]
    m_h4 = m_k4[:, None] & m_v[None, :]
    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_k1[:, None] + o_v[None, :] * K
        b_h1 += tl.load(p_h0_1, mask=m_h1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_k2[:, None] + o_v[None, :] * K
            b_h2 += tl.load(p_h0_2, mask=m_h2, other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_k3[:, None] + o_v[None, :] * K
            b_h3 += tl.load(p_h0_3, mask=m_h3, other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_k4[:, None] + o_v[None, :] * K
            b_h4 += tl.load(p_h0_4, mask=m_h4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_64 = tl.cast(i_t, tl.int64)
        o_t = i_t_64 * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_tv = m_t[:, None] & m_v[None, :]
        # h is [B, NT, H, K, V] with V innermost, unlike the pool layout.
        h_chunk = h + i_t_64 * stride_h
        p_h1 = h_chunk + o_k1[:, None] * V + o_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = h_chunk + o_k2[:, None] * V + o_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_h3 = h_chunk + o_k3[:, None] * V + o_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_h4 = h_chunk + o_k4[:, None] * V + o_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_v = v + o_t[:, None] * stride_v + o_v[None, :]
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v_new = -b_v_new + tl.load(p_v, mask=m_tv, other=0.0)

        if SAVE_NEW_VALUE:
            p_v_new = v_new + o_t[:, None] * stride_v + o_v[None, :]
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), mask=m_tv)

        if USE_G:
            last_idx = min((i_t_64 + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + i_h + o_t * H
            b_g = tl.load(p_g, mask=m_t, other=0.0)
            b_v_new = b_v_new * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            b_h4 += tl.dot(b_k, b_v_new)

    # epilogue — write final state back to pool-layout [slots, HV, V, K].
    if STORE_FINAL_STATE or USE_INDEXED_STATE:
        p_ht = ht + o_k1[:, None] + o_v[None, :] * K
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_ht = ht + o_k2[:, None] + o_v[None, :] * K
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_h2)
        if K > 128:
            p_ht = ht + o_k3[:, None] + o_v[None, :] * K
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_h3)
        if K > 192:
            p_ht = ht + o_k4[:, None] + o_v[None, :] * K
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_h4)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    inplace_indexed_state_update: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = (prepare_chunk_indices(cu_seqlens, chunk_size)
                     if cu_seqlens is not None else None)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    use_indexed_state = initial_state is not None and initial_state_indices is not None
    if use_indexed_state and not inplace_indexed_state_update:
        raise ValueError(
            "Indexed chunk state updates require inplace_indexed_state_update=True."
        )
    store_final_state_in_kernel = output_final_state and not use_indexed_state
    # Kernel writes final state in [V, K] layout (K innermost) to match the
    # pool layout. Allocate accordingly so the tensor's shape reflects memory.
    final_state = (k.new_empty(N, H, V, K, dtype=torch.float32)
                   if store_final_state_in_kernel else None)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        h0_i=initial_state_indices,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        stride_h0=initial_state.stride(0) if initial_state is not None else 0,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    if output_final_state and use_indexed_state:
        # The indexed kernel path updates h0 in-place, so returning
        # the final state means gathering those updated slots back out.
        final_state = initial_state.index_select(
            0, initial_state_indices.to(torch.long))
    return h, v_new, final_state
