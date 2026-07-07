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
"""

import math

import torch
import triton
import triton.language as tl


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
    NOISE_PAD: tl.constexpr,  # Q padded to >=16 for tl.dot
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)

    # Row r -> (head h = kvh*GROUP + r // Q, query qi = r % Q)
    R: tl.constexpr = Q * GROUP
    r = tl.arange(0, R)
    h = kvh * GROUP + r // Q
    qi = r % Q
    d = tl.arange(0, D)

    q_ptrs = q_ptr + b * stride_qb + qi[:, None] * stride_qq + h[:, None] * stride_qh + d[None, :]
    q_tile = tl.load(q_ptrs).to(tl.bfloat16)  # [R, D]

    m_i = tl.full([R], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([R], dtype=tl.float32)
    acc = tl.zeros([R, D], dtype=tl.float32)

    ctx_len = tl.load(ctx_len_ptr + b)
    n_pages = tl.cdiv(ctx_len, TPB)
    t = tl.arange(0, TPB)

    for p in range(0, n_pages):
        page = tl.load(blk_ptr + b * stride_bb + p).to(tl.int64)
        valid = (p * TPB + t) < ctx_len
        k_ptrs = (
            k_cache_ptr + page * stride_kp + t[:, None] * stride_kt + kvh * stride_kh + d[None, :]
        )
        v_ptrs = (
            v_cache_ptr + page * stride_vp + t[:, None] * stride_vt + kvh * stride_vh + d[None, :]
        )
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)

        s = tl.dot(q_tile, tl.trans(k)) * sm_scale  # [R, TPB]
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p_ij, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p_ij.to(tl.bfloat16), v)
        m_i = m_new

    # Dense noise suffix: Q transient mask/bonus K/V, fully visible.
    tn = tl.arange(0, NOISE_PAD)
    n_valid = tn < Q
    kn_ptrs = k_noise_ptr + b * stride_nb + tn[:, None] * stride_nq + kvh * stride_nh + d[None, :]
    vn_ptrs = v_noise_ptr + b * stride_nb + tn[:, None] * stride_nq + kvh * stride_nh + d[None, :]
    kn = tl.load(kn_ptrs, mask=n_valid[:, None], other=0.0).to(tl.bfloat16)
    vn = tl.load(vn_ptrs, mask=n_valid[:, None], other=0.0).to(tl.bfloat16)

    s = tl.dot(q_tile, tl.trans(kn)) * sm_scale  # [R, NOISE_PAD]
    s = tl.where(n_valid[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.exp(m_i - m_new)
    p_n = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p_n, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p_n.to(tl.bfloat16), vn)

    out = acc / l_i[:, None]
    out_ptrs = (
        out_ptr + b * stride_ob + qi[:, None] * stride_oq + h[:, None] * stride_oh + d[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty))


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
    assert (Q * group) >= 16 and D >= 16, "tl.dot needs tiles >= 16"
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1

    out = torch.empty_like(q)
    grid = (B, NKV)
    _dflash_ctx_attn_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_idx,
        ctx_lens,
        k_noise,
        v_noise,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_idx.stride(0),
        k_noise.stride(0),
        k_noise.stride(1),
        k_noise.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        1.0 / math.sqrt(D),
        Q=Q,
        GROUP=group,
        TPB=TPB,
        D=D,
        NOISE_PAD=max(16, triton.next_power_of_2(Q)),
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
