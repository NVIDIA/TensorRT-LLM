# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inkling Triton attention: paged prefill + decode with a learned relative-bias
``score_mod`` and native sliding window.

Why this exists
---------------
Inkling attention adds a learned per-(query-token, head, relative-distance)
additive bias INSIDE the attention score, and windows local layers separately.
No fused/CUDA-graph-safe TensorRT-LLM backend exposes a ``score_mod`` hook:
``attentionOp.cpp`` disables context FMHA for ``kRELATIVE`` position embedding,
and the trtllm-gen decode kernel rejects a relative bias. So the production
attention path for Inkling is a pair of Triton kernels that apply the bias as an
aux-tensor ``score_mod``, exactly mirroring the SGLang reference
(``sglang/kernels/ops/attention/{score_mod,extend_attention,decode_attention}.py``).

The bias is precomputed on the torch side as a contiguous ``rel_logits`` aux
tensor ``[num_query_tokens, num_heads, rel_extent]`` (``einsum('thd,de->the', r,
proj)`` with the global-layer ``tau`` folded in). The kernels only gather+add:

    rel_dist = q_pos - k_pos
    rel_idx  = clamp(rel_dist, 0, rel_extent - 1)
    bias     = rel_logits[q_idx, head, rel_idx]   if 0 <= rel_dist < rel_extent else 0
    qk      += bias

This keeps ``rel_logits`` a *static-shape* tensor (``num_query_tokens`` == batch
in the decode phase), so the decode kernel is CUDA-graph capturable: the launch
grid ``(batch, num_heads)`` is fixed, and per-request sequence lengths are read
from a GPU tensor inside the kernel (no host sync, no ``.item()``).

Both kernels read the paged KV cache in the ``KVCacheManagerV2`` HND layout
(``[num_pages, num_kv_heads, page_size, head_dim]`` after selecting K or V from
the ``[num_pages, 2, ...]`` pool), addressed through a per-request page table.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# Additive value used to drop a masked key from the softmax. Large enough that
# ``exp(qk - max)`` underflows to 0 in fp32, finite so online-softmax bookkeeping
# never sees a NaN. (float("-inf") would poison the running max on the first,
# fully-masked tile of a windowed row.) Inlined as a literal inside the kernels
# because Triton @jit functions cannot read non-constexpr module globals.
_NEG = tl.constexpr(-1.0e30)


# ---------------------------------------------------------------------------
# Prefill (context) kernel: contiguous varlen Q/K/V, causal + optional window,
# optional relative-bias score_mod. One fresh context has no cached prefix, so
# K/V are read from the packed extend tensors directly.
# ---------------------------------------------------------------------------
@triton.jit
def _inkling_prefill_kernel(
    Q,
    K,
    V,
    O,
    RelLogits,
    cu_seqlens,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_ot,
    stride_oh,
    stride_rt,
    stride_rh,
    kv_group_num,
    rel_extent: tl.constexpr,
    HAS_REL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    seq_start = tl.load(cu_seqlens + cur_seq)
    seq_len = tl.load(cu_seqlens + cur_seq + 1) - seq_start

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < Lk

    q_pos = cur_block_m * BLOCK_M + offs_m  # [BLOCK_M], position within sequence
    mask_m = q_pos < seq_len

    q_ptrs = ((seq_start + q_pos)[:, None] * stride_qt + cur_head * stride_qh +
              offs_d[None, :])
    q = tl.load(Q + q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Causal: query block cur_block_m only attends to keys <= its last row.
    end_n = tl.minimum(seq_len, (cur_block_m + 1) * BLOCK_M)
    # Sliding window: skip whole key tiles older than the window low bound.
    if WINDOW_LEFT >= 0:
        lo = cur_block_m * BLOCK_M - WINDOW_LEFT
        if lo < 0:
            lo = 0
        lo = (lo // BLOCK_N) * BLOCK_N
    else:
        lo = 0

    for start_n in range(lo, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_pos = start_n + offs_n  # [BLOCK_N]
        mask_n = k_pos < seq_len

        k_ptrs = ((seq_start + k_pos)[None, :] * stride_kt +
                  cur_kv_head * stride_kh + offs_d[:, None])
        k = tl.load(K + k_ptrs,
                    mask=mask_n[None, :] & mask_d[:, None],
                    other=0.0)
        qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale  # [BLOCK_M, BLOCK_N]

        if HAS_REL:
            rel_dist = q_pos[:, None] - k_pos[None, :]
            rel_idx = tl.minimum(tl.maximum(rel_dist, 0), rel_extent - 1)
            rel_ptrs = ((seq_start + q_pos)[:, None] * stride_rt +
                        cur_head * stride_rh + rel_idx)
            rel_valid = (rel_dist >= 0) & (rel_dist < rel_extent)
            bias = tl.load(RelLogits + rel_ptrs,
                           mask=mask_m[:, None] & mask_n[None, :] & rel_valid,
                           other=0.0)
            qk += bias

        valid = mask_m[:, None] & mask_n[None, :] & (q_pos[:, None]
                                                     >= k_pos[None, :])
        if WINDOW_LEFT >= 0:
            valid &= (q_pos[:, None] - k_pos[None, :]) <= WINDOW_LEFT
        qk = tl.where(valid, qk, _NEG)

        row_max = tl.max(qk, 1)
        n_e_max = tl.maximum(e_max, row_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        e_sum = e_sum * re_scale + tl.sum(p, 1)

        v_ptrs = ((seq_start + k_pos)[:, None] * stride_vt +
                  cur_kv_head * stride_vh + offs_d[None, :])
        v = tl.load(V + v_ptrs,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0)
        acc = acc * re_scale[:, None] + tl.dot(
            p.to(v.dtype), v, out_dtype=tl.float32)
        e_max = n_e_max

    acc = acc / e_sum[:, None]
    o_ptrs = ((seq_start + q_pos)[:, None] * stride_ot + cur_head * stride_oh +
              offs_d[None, :])
    tl.store(O + o_ptrs,
             acc.to(O.dtype.element_ty),
             mask=mask_m[:, None] & mask_d[None, :])


# ---------------------------------------------------------------------------
# Decode (generation) kernel: one query token per request, paged KV read,
# causal + optional window, optional relative-bias score_mod. CUDA-graph safe:
# static grid (batch, num_heads); seq lengths and the page table are read from
# GPU tensors, no host sync.
# ---------------------------------------------------------------------------
@triton.jit
def _inkling_decode_kernel(
    Q,
    K_Cache,
    V_Cache,
    O,
    RelLogits,
    seq_lens,
    page_table,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_kp,
    stride_kh,
    stride_kt,
    stride_vp,
    stride_vh,
    stride_vt,
    stride_ob,
    stride_oh,
    stride_rb,
    stride_rh,
    stride_ptb,
    kv_group_num,
    page_size: tl.constexpr,
    rel_extent: tl.constexpr,
    HAS_REL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Lk: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    seq_len = tl.load(seq_lens + cur_batch)
    q_pos = seq_len - 1  # decode query sits at the last cached position

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < Lk

    q = tl.load(Q + cur_batch * stride_qb + cur_head * stride_qh + offs_d,
                mask=mask_d,
                other=0.0).to(tl.float32)  # [BLOCK_DMODEL]

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    e_max = -float("inf")
    e_sum = 0.0

    if WINDOW_LEFT >= 0:
        lo = q_pos - WINDOW_LEFT
        if lo < 0:
            lo = 0
        lo = (lo // BLOCK_N) * BLOCK_N
    else:
        lo = 0

    for start_n in range(lo, seq_len, BLOCK_N):
        k_pos = start_n + offs_n  # [BLOCK_N]
        mask_n = k_pos < seq_len

        page_local = k_pos // page_size
        tok_in_page = k_pos % page_size
        page_id = tl.load(page_table + cur_batch * stride_ptb + page_local,
                          mask=mask_n,
                          other=0).to(tl.int64)

        k_ptrs = (page_id[:, None] * stride_kp + cur_kv_head * stride_kh +
                  tok_in_page[:, None] * stride_kt + offs_d[None, :])
        k = tl.load(K_Cache + k_ptrs,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0).to(tl.float32)
        qk = tl.sum(q[None, :] * k, 1) * sm_scale  # [BLOCK_N]

        if HAS_REL:
            rel_dist = q_pos - k_pos
            rel_idx = tl.minimum(tl.maximum(rel_dist, 0), rel_extent - 1)
            rel_ptrs = cur_batch * stride_rb + cur_head * stride_rh + rel_idx
            rel_valid = (rel_dist >= 0) & (rel_dist < rel_extent)
            bias = tl.load(RelLogits + rel_ptrs,
                           mask=mask_n & rel_valid,
                           other=0.0)
            qk += bias

        valid = mask_n & (k_pos <= q_pos)
        if WINDOW_LEFT >= 0:
            valid &= (q_pos - k_pos) <= WINDOW_LEFT
        qk = tl.where(valid, qk, _NEG)

        n_e_max = tl.maximum(e_max, tl.max(qk, 0))
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)  # [BLOCK_N]
        e_sum = e_sum * re_scale + tl.sum(p, 0)

        v_ptrs = (page_id[:, None] * stride_vp + cur_kv_head * stride_vh +
                  tok_in_page[:, None] * stride_vt + offs_d[None, :])
        v = tl.load(V_Cache + v_ptrs,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0).to(tl.float32)
        acc = acc * re_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    o = acc / e_sum
    tl.store(O + cur_batch * stride_ob + cur_head * stride_oh + offs_d,
             o.to(O.dtype.element_ty),
             mask=mask_d)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def _block_dmodel(head_dim: int) -> int:
    return triton.next_power_of_2(head_dim)


def inkling_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    sm_scale: float,
    rel_logits: Optional[torch.Tensor] = None,
    rel_extent: int = 0,
    window_left: int = -1,
) -> torch.Tensor:
    """Context-phase attention over packed varlen Q/K/V.

    Args:
        q: ``[total_tokens, num_heads, head_dim]``
        k, v: ``[total_tokens, num_kv_heads, head_dim]``
        cu_seqlens: ``[batch + 1]`` int32 cumulative token counts.
        max_seqlen: max per-request length (host int; used for the grid).
        sm_scale: softmax scale (``1 / head_dim`` for Inkling).
        rel_logits: ``[total_tokens, num_heads, rel_extent]`` fp32 aux bias, or
            None to skip the score_mod.
        rel_extent: relative-bias extent (profile width).
        window_left: sliding-window radius (inclusive), -1 to disable.

    Returns ``[total_tokens, num_heads, head_dim]`` in q's dtype.
    """
    # The kernels index the head_dim with an implicit stride-1 last axis, so the
    # inputs must be contiguous. ``v`` in particular reaches here non-contiguous:
    # it is the fused-qkv v slice run through the short conv, and (unlike ``k``)
    # never passes through ``apply_qk_norm``'s reshape, so it keeps the qkv row
    # stride. ``.contiguous()`` is a no-op for already-contiguous q/k.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    total_tokens, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    kv_group_num = num_heads // num_kv_heads
    o = torch.empty_like(q)

    has_rel = rel_logits is not None
    if has_rel:
        assert rel_logits.is_contiguous() and rel_logits.shape[-1] == rel_extent
        r_st, r_sh = rel_logits.stride(0), rel_logits.stride(1)
        rel_arg = rel_logits
    else:
        r_st = r_sh = 0
        rel_arg = q  # unused placeholder pointer

    BLOCK_DMODEL = _block_dmodel(head_dim)
    BLOCK_M = 64
    BLOCK_N = 64
    batch = cu_seqlens.shape[0] - 1
    grid = (batch, num_heads, triton.cdiv(max_seqlen, BLOCK_M))

    _inkling_prefill_kernel[grid](
        q,
        k,
        v,
        o,
        rel_arg,
        cu_seqlens,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        r_st,
        r_sh,
        kv_group_num,
        rel_extent=rel_extent if has_rel else 1,
        HAS_REL=has_rel,
        WINDOW_LEFT=window_left,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lk=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return o


def inkling_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    sm_scale: float,
    rel_logits: Optional[torch.Tensor] = None,
    rel_extent: int = 0,
    window_left: int = -1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generation-phase attention: one query per request over paged KV.

    Args:
        q: ``[batch, num_heads, head_dim]``
        k_cache, v_cache: ``[num_pages, num_kv_heads, page_size, head_dim]`` HND
            views (K/V selected from the ``[num_pages, 2, ...]`` pool).
        seq_lens: ``[batch]`` int32 GPU total-KV length per request.
        page_table: ``[batch, max_pages]`` int32 GPU physical page ids.
        page_size: tokens per page.
        sm_scale: softmax scale (``1 / head_dim``).
        rel_logits: ``[batch, num_heads, rel_extent]`` fp32 aux bias, or None.
        rel_extent: relative-bias extent.
        window_left: sliding-window radius (inclusive), -1 to disable.
        out: optional pre-allocated ``[batch, num_heads, head_dim]`` output (for
            CUDA-graph static buffers).

    Returns ``[batch, num_heads, head_dim]`` in q's dtype.
    """
    q = q.contiguous()  # kernel indexes head_dim as the stride-1 axis
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    kv_group_num = num_heads // num_kv_heads
    o = out if out is not None else torch.empty_like(q)

    has_rel = rel_logits is not None
    if has_rel:
        assert rel_logits.is_contiguous() and rel_logits.shape[-1] == rel_extent
        r_sb, r_sh = rel_logits.stride(0), rel_logits.stride(1)
        rel_arg = rel_logits
    else:
        r_sb = r_sh = 0
        rel_arg = q

    BLOCK_DMODEL = _block_dmodel(head_dim)
    BLOCK_N = 64
    grid = (batch, num_heads)

    _inkling_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        o,
        rel_arg,
        seq_lens,
        page_table,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        o.stride(0),
        o.stride(1),
        r_sb,
        r_sh,
        page_table.stride(0),
        kv_group_num,
        page_size=page_size,
        rel_extent=rel_extent if has_rel else 1,
        HAS_REL=has_rel,
        WINDOW_LEFT=window_left,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        Lk=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return o


def build_page_table(block_ids_per_seq, max_pages: int, device) -> torch.Tensor:
    """Pack a ragged ``block_ids_per_seq`` (from
    ``KVCacheManagerV2.get_batch_cache_indices``) into a dense
    ``[batch, max_pages]`` int32 page table, padding short rows with 0 (never
    read: the decode kernel bounds every access by the per-request ``seq_len``).
    """
    batch = len(block_ids_per_seq)
    pt = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    for i, blocks in enumerate(block_ids_per_seq):
        valid = [int(b) for b in blocks if int(b) >= 0]
        if valid:
            pt[i, :len(valid)] = torch.tensor(valid,
                                              dtype=torch.int32,
                                              device=device)
    return pt


def write_kv_cache_hnd(k_cache: torch.Tensor, v_cache: torch.Tensor,
                       new_k: torch.Tensor, new_v: torch.Tensor, block_ids,
                       start_slot: int, page_size: int) -> None:
    """Write ``new_k``/``new_v`` (``[n, num_kv_heads, head_dim]``) for ONE
    request into the paged HND cache starting at logical position ``start_slot``.

    ``k_cache``/``v_cache`` are ``[num_pages, num_kv_heads, page_size,
    head_dim]`` views. ``block_ids`` is the request's physical page list. Used at
    prefill/decode to populate the cache before attention reads it.
    """
    valid_blocks = [int(b) for b in block_ids if int(b) >= 0]
    n = new_k.shape[0]
    written = 0
    while written < n:
        pos = start_slot + written
        page = valid_blocks[pos // page_size]
        off = pos % page_size
        take = min(page_size - off, n - written)
        k_cache[page, :,
                off:off + take, :] = (new_k[written:written + take].transpose(
                    0, 1).to(k_cache.dtype))
        v_cache[page, :,
                off:off + take, :] = (new_v[written:written + take].transpose(
                    0, 1).to(v_cache.dtype))
        written += take
