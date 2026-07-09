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
"""DFlash hybrid-context cross-attention.

Reads the draft context K/V from the target KV cache manager's paged pool
(hybrid mode: draft layers registered as spec layers of the target manager).
No existing kernel fits this op: it needs device-resident sequence lengths
(acceptance-dependent, updated inside CUDA graphs), 32-token pages in the
manager's NHD block layout, fp8 KV, and a dense per-step noise-K/V suffix
that is never written to the cache.

Two execution paths, chosen by batch size at launch (static under CUDA
graph capture, where the batch is padded to a fixed size):

- Small batch: flash-decoding style split-KV. ``(B, NKV)`` CTAs cannot fill
  the GPU, so the context is partitioned across ``S`` CTAs that emit partial
  softmax states (acc, m, l); a merge kernel folds the partials plus the
  dense noise suffix. Per-split ranges are derived from the device-resident
  ctx_len, so the fixed grid stays CUDA-graph safe.
- Large batch: single-pass kernel (one CTA per (request, kv head)) — already
  faster than a dense read at this occupancy, and it skips the partial
  buffers and merge round-trip.

Both paths tile ``BLOCK_N`` context tokens (several pages) per iteration
with a per-token page-id gather, rather than one 32-token page per ``tl.dot``.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attend_ctx_tokens(
    q_tile,  # [R_PAD, D] bf16
    k_cache_ptr,
    v_cache_ptr,
    blk_row_ptr,  # page-id row for this request
    kvh,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    start,
    end,  # token range [start, end) of this CTA
    sm_scale,
    m_i,
    l_i,
    acc,
    R_PAD: tl.constexpr,
    TPB: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Accumulate online-softmax over context tokens [start, end)."""
    d = tl.arange(0, D)
    tok = tl.arange(0, BLOCK_N)
    for n0 in range(start, end, BLOCK_N):
        idx = n0 + tok
        valid = idx < end
        # Per-token page gather: page ids come from the block table row.
        page = tl.load(blk_row_ptr + idx // TPB, mask=valid, other=0).to(tl.int64)
        toff = idx % TPB
        k_ptrs = (
            k_cache_ptr + page[:, None] * stride_kp + toff[:, None] * stride_kt
            + kvh * stride_kh + d[None, :]
        )
        v_ptrs = (
            v_cache_ptr + page[:, None] * stride_vp + toff[:, None] * stride_vt
            + kvh * stride_vh + d[None, :]
        )
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)

        s = tl.dot(q_tile, tl.trans(k)) * sm_scale  # [R_PAD, BLOCK_N]
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # All-(-inf) rows (empty split) keep m == -inf; exp(-inf - -inf) is
        # NaN, so guard the rescale factor.
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
        p_ij = tl.exp(s - m_new[:, None])
        p_ij = tl.where(valid[None, :], p_ij, 0.0)
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p_ij.to(tl.bfloat16), v)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _attend_noise_suffix(
    q_tile,  # [R_PAD, D]
    k_noise_ptr,
    v_noise_ptr,
    b,
    kvh,
    stride_nb,
    stride_nq,
    stride_nh,
    sm_scale,
    m_i,
    l_i,
    acc,
    R_PAD: tl.constexpr,
    Q: tl.constexpr,
    D: tl.constexpr,
    NOISE_PAD: tl.constexpr,
):
    """Fold the dense per-step noise K/V suffix into the softmax state."""
    d = tl.arange(0, D)
    tn = tl.arange(0, NOISE_PAD)
    n_valid = tn < Q
    kn_ptrs = k_noise_ptr + b * stride_nb + tn[:, None] * stride_nq + kvh * stride_nh + d[None, :]
    vn_ptrs = v_noise_ptr + b * stride_nb + tn[:, None] * stride_nq + kvh * stride_nh + d[None, :]
    kn = tl.load(kn_ptrs, mask=n_valid[:, None], other=0.0).to(tl.bfloat16)
    vn = tl.load(vn_ptrs, mask=n_valid[:, None], other=0.0).to(tl.bfloat16)

    s = tl.dot(q_tile, tl.trans(kn)) * sm_scale  # [R_PAD, NOISE_PAD]
    s = tl.where(n_valid[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
    p_n = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p_n, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p_n.to(tl.bfloat16), vn)
    return m_new, l_i, acc


@triton.jit
def _load_q_tile(
    q_ptr, b, kvh, stride_qb, stride_qq, stride_qh,
    Q: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    R_PAD: tl.constexpr,
):
    # Row r -> (head h = kvh*GROUP + r // Q, query qi = r % Q). tl.arange
    # needs a power-of-two span, so rows are padded to R_PAD; padded rows
    # load zeros and are masked again at the output store.
    R: tl.constexpr = Q * GROUP
    r = tl.arange(0, R_PAD)
    row_ok = r < R
    h = kvh * GROUP + r // Q
    qi = r % Q
    d = tl.arange(0, D)
    q_ptrs = q_ptr + b * stride_qb + qi[:, None] * stride_qq + h[:, None] * stride_qh + d[None, :]
    return tl.load(q_ptrs, mask=row_ok[:, None], other=0.0).to(tl.bfloat16)  # [R_PAD, D]


@triton.jit
def _dflash_ctx_attn_kernel(
    q_ptr,  # [B, Q, NH, D]
    k_cache_ptr,  # [pages, TPB, NKV, D]
    v_cache_ptr,  # [pages, TPB, NKV, D]
    blk_ptr,  # [B, W] page indices per request
    ctx_len_ptr,  # [B] valid ctx tokens per request (device)
    k_noise_ptr,  # [B, Q, NKV, D]
    v_noise_ptr,  # [B, Q, NKV, D]
    out_ptr,  # [B, Q, NH, D]
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    stride_bb,
    stride_nb,
    stride_nq,
    stride_nh,
    stride_ob,
    stride_oq,
    stride_oh,
    sm_scale,
    Q: tl.constexpr,  # queries per request (draft block size)
    GROUP: tl.constexpr,  # q heads per kv head
    TPB: tl.constexpr,  # tokens per page
    D: tl.constexpr,  # head dim
    BLOCK_N: tl.constexpr,  # ctx tokens per iteration
    NOISE_PAD: tl.constexpr,  # Q padded to >=16 for tl.dot
    R_PAD: tl.constexpr,  # Q*GROUP padded to a power of two
):
    """Single-pass path: one CTA per (request, kv head)."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    R: tl.constexpr = Q * GROUP

    q_tile = _load_q_tile(q_ptr, b, kvh, stride_qb, stride_qq, stride_qh,
                          Q, GROUP, D, R_PAD)

    m_i = tl.full([R_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([R_PAD], dtype=tl.float32)
    acc = tl.zeros([R_PAD, D], dtype=tl.float32)

    ctx_len = tl.load(ctx_len_ptr + b)
    m_i, l_i, acc = _attend_ctx_tokens(
        q_tile, k_cache_ptr, v_cache_ptr, blk_ptr + b * stride_bb, kvh,
        stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
        0, ctx_len, sm_scale, m_i, l_i, acc, R_PAD, TPB, D, BLOCK_N)

    m_i, l_i, acc = _attend_noise_suffix(
        q_tile, k_noise_ptr, v_noise_ptr, b, kvh, stride_nb, stride_nq,
        stride_nh, sm_scale, m_i, l_i, acc, R_PAD, Q, D, NOISE_PAD)

    out = acc / l_i[:, None]
    r = tl.arange(0, R_PAD)
    row_ok = r < R
    qi = r % Q
    h = kvh * GROUP + r // Q
    d = tl.arange(0, D)
    out_ptrs = (
        out_ptr + b * stride_ob + qi[:, None] * stride_oq + h[:, None] * stride_oh + d[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])


@triton.jit
def _dflash_ctx_attn_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_ptr,
    ctx_len_ptr,
    part_acc_ptr,  # [B, NKV, S, R, D] fp32
    part_m_ptr,  # [B, NKV, S, R] fp32
    part_l_ptr,  # [B, NKV, S, R] fp32
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    stride_bb,
    sm_scale,
    S,  # number of context splits (runtime: varies with eager batch size,
    # keeping it non-constexpr avoids a Triton recompile per batch size)
    NKV: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    TPB: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    R_PAD: tl.constexpr,
):
    """Split phase: CTA (b, kvh, s) covers a slice of the context."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    s_id = tl.program_id(2)

    ctx_len = tl.load(ctx_len_ptr + b)
    # Even token split, rounded to BLOCK_N so slices don't share tiles.
    per_split = tl.cdiv(tl.cdiv(ctx_len, BLOCK_N), S) * BLOCK_N
    start = s_id * per_split
    end = tl.minimum(start + per_split, ctx_len)

    q_tile = _load_q_tile(q_ptr, b, kvh, stride_qb, stride_qq, stride_qh,
                          Q, GROUP, D, R_PAD)

    m_i = tl.full([R_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([R_PAD], dtype=tl.float32)
    acc = tl.zeros([R_PAD, D], dtype=tl.float32)

    m_i, l_i, acc = _attend_ctx_tokens(
        q_tile, k_cache_ptr, v_cache_ptr, blk_ptr + b * stride_bb, kvh,
        stride_kp, stride_kt, stride_kh, stride_vp, stride_vt, stride_vh,
        start, end, sm_scale, m_i, l_i, acc, R_PAD, TPB, D, BLOCK_N)

    # Padded rows are stored too (the partial buffers are R_PAD-strided);
    # the merge kernel discards them at its masked output store.
    r = tl.arange(0, R_PAD)
    d = tl.arange(0, D)
    base = ((b * NKV + kvh) * S + s_id) * R_PAD
    tl.store(part_m_ptr + base + r, m_i)
    tl.store(part_l_ptr + base + r, l_i)
    tl.store(part_acc_ptr + (base + r)[:, None] * D + d[None, :], acc)


@triton.jit
def _dflash_ctx_attn_merge_kernel(
    q_ptr,
    part_acc_ptr,  # [B, NKV, S, R, D]
    part_m_ptr,  # [B, NKV, S, R]
    part_l_ptr,  # [B, NKV, S, R]
    k_noise_ptr,
    v_noise_ptr,
    out_ptr,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_nb,
    stride_nq,
    stride_nh,
    stride_ob,
    stride_oq,
    stride_oh,
    sm_scale,
    S,  # runtime, see split kernel
    NKV: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    NOISE_PAD: tl.constexpr,
    R_PAD: tl.constexpr,
):
    """Merge phase: fold split partials, then the noise suffix."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    R: tl.constexpr = Q * GROUP
    r = tl.arange(0, R_PAD)
    d = tl.arange(0, D)

    m_i = tl.full([R_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([R_PAD], dtype=tl.float32)
    acc = tl.zeros([R_PAD, D], dtype=tl.float32)

    for s_id in range(0, S):
        base = ((b * NKV + kvh) * S + s_id) * R_PAD
        m_s = tl.load(part_m_ptr + base + r)
        l_s = tl.load(part_l_ptr + base + r)
        a_s = tl.load(part_acc_ptr + (base + r)[:, None] * D + d[None, :])

        m_new = tl.maximum(m_i, m_s)
        guard = m_new == float("-inf")
        alpha = tl.where(guard, 1.0, tl.exp(m_i - m_new))
        beta = tl.where(guard | (m_s == float("-inf")), 0.0, tl.exp(m_s - m_new))
        # Empty splits carry l == 0, so they contribute nothing even when
        # beta is defined.
        l_i = l_i * alpha + l_s * beta
        acc = acc * alpha[:, None] + a_s * beta[:, None]
        m_i = m_new

    q_tile = _load_q_tile(q_ptr, b, kvh, stride_qb, stride_qq, stride_qh,
                          Q, GROUP, D, R_PAD)
    m_i, l_i, acc = _attend_noise_suffix(
        q_tile, k_noise_ptr, v_noise_ptr, b, kvh, stride_nb, stride_nq,
        stride_nh, sm_scale, m_i, l_i, acc, R_PAD, Q, D, NOISE_PAD)

    out = acc / l_i[:, None]
    row_ok = r < R
    qi = r % Q
    h = kvh * GROUP + r // Q
    out_ptrs = (
        out_ptr + b * stride_ob + qi[:, None] * stride_oq + h[:, None] * stride_oh + d[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])


# Reused across steps so CUDA graph capture sees stable addresses; keyed by
# (B, NKV, S, R, D, device). Bounded: shapes are fixed per serving config.
_PARTIAL_BUFS = {}


def _get_partial_bufs(B, NKV, S, R, D, device):
    key = (B, NKV, S, R, D, device)
    bufs = _PARTIAL_BUFS.get(key)
    if bufs is None:
        acc = torch.empty(B * NKV * S * R * D, dtype=torch.float32, device=device)
        m = torch.empty(B * NKV * S * R, dtype=torch.float32, device=device)
        l = torch.empty(B * NKV * S * R, dtype=torch.float32, device=device)
        bufs = (acc, m, l)
        _PARTIAL_BUFS[key] = bufs
    return bufs


def _num_splits(B: int, NKV: int) -> int:
    """Pick the context split count from the (static) launch batch size.

    Aim for enough CTAs to occupy the GPU; cap so per-split work stays
    above ~2 tiles. B is padded/static under CUDA graph capture, so the
    resulting grid is capture-safe.
    """
    sm_count = torch.cuda.get_device_properties(
        torch.cuda.current_device()).multi_processor_count
    if B * NKV >= sm_count:
        return 1
    return min(16, max(1, sm_count // (B * NKV)))


def dflash_ctx_paged_attention(
    q: torch.Tensor,  # [B, Q, NH, D] bf16
    k_cache: torch.Tensor,  # [pages, TPB, NKV, D] fp8/bf16 pool view
    v_cache: torch.Tensor,
    block_idx: torch.Tensor,  # [B, W] int32/int64 page ids
    ctx_lens: torch.Tensor,  # [B] int32/int64, device-resident
    k_noise: torch.Tensor,  # [B, Q, NKV, D] bf16
    v_noise: torch.Tensor,
) -> torch.Tensor:
    B, Q, NH, D = q.shape
    _, TPB, NKV, _ = k_cache.shape
    assert NH % NKV == 0
    group = NH // NKV
    # tl.arange spans must be powers of two. Query rows (R = Q*GROUP) are
    # padded below, so any draft length / GQA ratio works; head_dim and the
    # manager page size have no such padding path and must be powers of two
    # (every known draft uses 64/128; the manager default page is 32).
    if D < 16 or (D & (D - 1)) != 0:
        raise ValueError(
            f"DFlash hybrid ctx kernel requires a power-of-two head_dim "
            f">= 16, got {D}.")
    if TPB & (TPB - 1) != 0:
        raise ValueError(
            f"DFlash hybrid ctx kernel requires a power-of-two "
            f"tokens_per_block, got {TPB}.")
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1

    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    noise_pad = max(16, triton.next_power_of_2(Q))
    BLOCK_N = 128
    R = Q * group
    # Pad query rows to a power of two (>= 16 for tl.dot); e.g. a draft
    # length of 5 gives Q = 6 and R = 24 -> R_PAD = 32.
    r_pad = max(16, triton.next_power_of_2(R))

    common_q = (q.stride(0), q.stride(1), q.stride(2))
    common_n = (k_noise.stride(0), k_noise.stride(1), k_noise.stride(2))
    common_o = (out.stride(0), out.stride(1), out.stride(2))

    S = _num_splits(B, NKV)
    if S == 1:
        _dflash_ctx_attn_kernel[(B, NKV)](
            q, k_cache, v_cache, block_idx, ctx_lens, k_noise, v_noise, out,
            *common_q,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_idx.stride(0),
            *common_n,
            *common_o,
            sm_scale,
            Q=Q, GROUP=group, TPB=TPB, D=D, BLOCK_N=BLOCK_N,
            NOISE_PAD=noise_pad, R_PAD=r_pad,
            num_warps=4,
        )
        return out

    part_acc, part_m, part_l = _get_partial_bufs(B, NKV, S, r_pad, D, q.device)
    _dflash_ctx_attn_split_kernel[(B, NKV, S)](
        q, k_cache, v_cache, block_idx, ctx_lens,
        part_acc, part_m, part_l,
        *common_q,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_idx.stride(0),
        sm_scale,
        NKV=NKV, S=S, Q=Q, GROUP=group, TPB=TPB, D=D, BLOCK_N=BLOCK_N,
        R_PAD=r_pad,
        num_warps=4,
    )
    _dflash_ctx_attn_merge_kernel[(B, NKV)](
        q, part_acc, part_m, part_l, k_noise, v_noise, out,
        *common_q,
        *common_n,
        *common_o,
        sm_scale,
        NKV=NKV, S=S, Q=Q, GROUP=group, D=D, NOISE_PAD=noise_pad,
        R_PAD=r_pad,
        num_warps=4,
    )
    return out


def dflash_ctx_paged_attention_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_idx: torch.Tensor,
    ctx_lens: torch.Tensor,
    k_noise: torch.Tensor,
    v_noise: torch.Tensor,
) -> torch.Tensor:
    """Plain-torch reference (slow gather path) for parity tests."""
    B, Q, NH, D = q.shape
    _, TPB, NKV, _ = k_cache.shape
    group = NH // NKV
    out = torch.empty_like(q)
    lens = ctx_lens.tolist()
    for b in range(B):
        n = int(lens[b])
        n_pages = (n + TPB - 1) // TPB
        pages = block_idx[b, :n_pages].long()
        k_ctx = k_cache[pages].to(torch.float32).reshape(-1, NKV, D)[:n]
        v_ctx = v_cache[pages].to(torch.float32).reshape(-1, NKV, D)[:n]
        k = torch.cat([k_ctx, k_noise[b].to(torch.float32)], dim=0)  # [n+Q, NKV, D]
        v = torch.cat([v_ctx, v_noise[b].to(torch.float32)], dim=0)
        qb = q[b].to(torch.float32)  # [Q, NH, D]
        for h in range(NH):
            g = h // group
            s = (qb[:, h] @ k[:, g].T) / math.sqrt(D)  # [Q, n+Q]
            out[b, :, h] = (torch.softmax(s, dim=-1) @ v[:, g]).to(out.dtype)
    return out
