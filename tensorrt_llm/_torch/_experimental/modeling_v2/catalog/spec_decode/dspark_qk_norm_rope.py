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
# =============================================================================
# DSpark / DFlash drafter block: q / k RMSNorm + partial NeoX RoPE in one launch
# =============================================================================
#
# It replaces, bit for bit, the three launches of DFlashForCausalLM.dflash_forward's flashinfer-RoPE branch at head_dim
# 128: flashinfer_rmsnorm on the q heads, flashinfer_rmsnorm on the k heads (both flashinfer's CuTe RMSNormKernel),
# then flashinfer_apply_rope_with_cos_sin_cache_inplace (NeoX, the cache's leading ROT dims rotated, the rest kept).
# The arithmetic is flashinfer's, made explicit (every op inline PTX, so nothing is contracted or reordered):
#   RMSNorm: 16 threads x 8 contiguous elements per row; each thread sums its x^2 sequentially from 0; the 16 sums are
#     added by the butterfly at offsets 1, 2, 4, 8 (adjacent pairs, then pairs of pairs); mean = sum * (1 / 128);
#     rstd = rsqrt.approx.ftz(mean + eps); y = bf16((x * rstd) * w).
#   RoPE (on the bf16 norm output, cos / sin fp32 from the cache): out_i = fma(x_i, cos_i, -x_{i+R/2} * sin_i) and
#     out_{i+R/2} = fma(x_{i+R/2}, cos_i, x_i * sin_i) for i < R / 2, rounded to bf16.
# Measured against the stock launches: 0 mismatching elements at 4 and 32 rows.
#
# DRAFT_APPEND_FUSE (``qk_norm_rope_append``): the same launch also writes each row's K (the bits it returns) and V
# (copied from qkv) into the layer's paged HND cache [pages, 2, NKV, PS, HD] where flashinfer.page.append_paged_kv_cache
# would: page kv_indices[kv_indptr[b] + pos // PS], slot pos % PS, with b = batch_indices[row] and pos = positions[row].
# The caller then skips that append launch; the cache bits are the same.

import torch
import triton
import triton.language as tl

HEAD_DIM = 128


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _fma(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rsqrt(a):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _pick(v, idx, j):
    """Column j of a [R, C] tile as [R] (exact: one nonzero term)."""
    return tl.sum(tl.where(idx == j, v, 0.0), axis=1)


@triton.jit
def _pair_sum(v, n: tl.constexpr):
    """[n] -> [n / 2]: element k = v[2k] + v[2k + 1]."""
    v2 = tl.reshape(v, (n // 2, 2))
    idx = tl.arange(0, 2)[None, :] + tl.arange(0, n // 2)[:, None] * 0
    return _add(_pick(v2, idx, 0), _pick(v2, idx, 1))


@triton.jit
def _qk_norm_rope_kernel(
    qkv_ptr,
    qkv_stride,
    pos_ptr,
    cs_ptr,
    cs_stride,
    wq_ptr,
    wk_ptr,
    q_out,
    k_out,
    eps_q,
    eps_k,
    cache_ptr,
    cache_page_stride,
    bidx_ptr,
    apos_ptr,
    kvi_ptr,
    kvp_ptr,
    v_col,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    HD: tl.constexpr,
    HALF: tl.constexpr,
    PS: tl.constexpr,
    STORE: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // (HQ + HKV)
    h = pid % (HQ + HKV)
    is_q = h < HQ
    base = qkv_ptr + t * qkv_stride + h * HD
    r16 = tl.arange(0, 16)
    c8 = tl.arange(0, 8)
    x2 = tl.load(base + r16[:, None] * 8 + c8[None, :]).to(tl.float32)
    sq = _mul(x2, x2)
    col = c8[None, :] + r16[:, None] * 0
    s = tl.zeros([16], dtype=tl.float32)
    for j in tl.static_range(8):
        s = _add(s, _pick(sq, col, j))
    s8 = _pair_sum(s, 16)
    s4 = _pair_sum(s8, 8)
    s2 = _pair_sum(s4, 4)
    two = tl.arange(0, 2)
    total = _add(tl.sum(tl.where(two == 0, s2, 0.0)), tl.sum(tl.where(two == 1, s2, 0.0)))
    eps = tl.where(is_q, eps_q, eps_k)
    rstd = _rsqrt(_add(_mul(total, 1.0 / HD), eps))
    d = tl.arange(0, HD)
    x = tl.load(base + d).to(tl.float32)
    w = tl.where(is_q, tl.load(wq_ptr + d), tl.load(wk_ptr + d)).to(tl.float32)
    y = _mul(_mul(x, rstd), w).to(tl.bfloat16)
    pos = tl.load(pos_ptr + t)
    i = tl.arange(0, HALF)
    cos = tl.load(cs_ptr + pos * cs_stride + i)
    sin = tl.load(cs_ptr + pos * cs_stride + HALF + i)
    yf = y.to(tl.float32)
    x1 = tl.sum(tl.where(d[None, :] == i[:, None], yf[None, :], 0.0), axis=1)
    x2r = tl.sum(tl.where(d[None, :] == (i[:, None] + HALF), yf[None, :], 0.0), axis=1)
    o1 = _fma(x1, cos, _mul(-x2r, sin))
    o2 = _fma(x2r, cos, _mul(x1, sin))
    obase = tl.where(is_q, q_out + t * (HQ * HD) + h * HD, k_out + t * (HKV * HD) + (h - HQ) * HD)
    tl.store(obase + d, y, mask=d >= 2 * HALF)
    tl.store(obase + i, o1.to(tl.bfloat16))
    tl.store(obase + HALF + i, o2.to(tl.bfloat16))
    if STORE:
        # K / V heads only: K's bits as stored above, V copied from qkv, at the append's page and slot.
        is_k = h >= HQ
        hk = tl.where(is_k, h - HQ, 0)
        b = tl.load(bidx_ptr + t).to(tl.int64)
        p = tl.load(apos_ptr + t).to(tl.int64)
        page = tl.load(kvi_ptr + tl.load(kvp_ptr + b).to(tl.int64) + p // PS).to(tl.int64)
        slot = p % PS
        kdst = cache_ptr + page * cache_page_stride + (hk * PS + slot) * HD
        vdst = cache_ptr + page * cache_page_stride + ((HKV + hk) * PS + slot) * HD
        tl.store(kdst + d, y, mask=(d >= 2 * HALF) & is_k)
        tl.store(kdst + i, o1.to(tl.bfloat16), mask=is_k)
        tl.store(kdst + HALF + i, o2.to(tl.bfloat16), mask=is_k)
        v = tl.load(qkv_ptr + t * qkv_stride + v_col + hk * HD + d, mask=is_k & (d >= 0), other=0.0)
        tl.store(vdst + d, v, mask=is_k & (d >= 0))


def _plain_rmsnorm(norm) -> bool:
    """The stock q_norm / k_norm takes flashinfer's plain (non-Gemma) RMSNorm on a bf16 input."""
    return (
        getattr(norm, "use_gemma", True) is False
        and not getattr(norm, "use_cuda_tile", False)
        and not getattr(norm, "is_nvfp4", False)
        and not getattr(norm, "return_hp_output", False)
        and getattr(norm, "weight", None) is not None
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.shape == (HEAD_DIM,)
    )


def qk_norm_rope(qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox):
    """q / k RMSNorm + RoPE of the drafter block in one launch: returns new contiguous (q [T, q_size], k [T, kv_size])
    bitwise equal to the stock flashinfer path, or None (nothing computed) for anything it does not reproduce."""
    if not qk_norm_rope_supported(
        qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox
    ):
        return None
    return _launch(qkv, q_size, kv_size, q_norm, k_norm, positions, cos_sin_cache)


def qk_norm_rope_supported(
    qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox
) -> bool:
    """Whether the one-launch q / k RMSNorm + RoPE reproduces the stock path for these operands."""
    from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE

    return bool(
        IS_FLASHINFER_AVAILABLE
        and is_neox
        and head_dim == HEAD_DIM
        and qkv.dtype == torch.bfloat16
        and qkv.dim() == 2
        and qkv.stride(1) == 1
        and qkv.shape[1] >= q_size + kv_size
        and q_size % head_dim == 0
        and kv_size % head_dim == 0
        and positions.dtype == torch.int32
        and positions.dim() == 1
        and positions.shape[0] == qkv.shape[0]
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.stride(1) == 1
        and cos_sin_cache.shape[1] in (16, 32, 64, 128)
        and _plain_rmsnorm(q_norm)
        and _plain_rmsnorm(k_norm)
    )


def _launch(qkv, q_size, kv_size, q_norm, k_norm, positions, cos_sin_cache, append=None):
    T = qkv.shape[0]
    hq, hkv = q_size // HEAD_DIM, kv_size // HEAD_DIM
    q = torch.empty(T, q_size, dtype=torch.bfloat16, device=qkv.device)
    k = torch.empty(T, kv_size, dtype=torch.bfloat16, device=qkv.device)
    if T == 0:
        return q, k
    if append is None:
        cache, bidx, apos, kvi, kvp, ps, store = (
            q,
            positions,
            positions,
            positions,
            positions,
            1,
            False,
        )
    else:
        cache, bidx, apos, kvi, kvp = append
        ps, store = cache.shape[-2], True
    _qk_norm_rope_kernel[(T * (hq + hkv),)](
        qkv,
        qkv.stride(0),
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_norm.weight,
        k_norm.weight,
        q,
        k,
        float(q_norm.variance_epsilon),
        float(k_norm.variance_epsilon),
        cache,
        cache.stride(0) if store else 0,
        bidx,
        apos,
        kvi,
        kvp,
        q_size + kv_size,
        HQ=hq,
        HKV=hkv,
        HD=HEAD_DIM,
        HALF=cos_sin_cache.shape[1] // 2,
        PS=ps,
        STORE=store,
        num_warps=1,
    )
    return q, k


_INT = (torch.int32, torch.int64)


def append_supported(
    qkv, kv_size, paged_kv_cache, batch_indices, positions, kv_indices, kv_indptr
) -> bool:
    """Whether ``qk_norm_rope_append`` reproduces flashinfer.page.append_paged_kv_cache for these operands: a
    contiguous bf16 HND layer cache [pages, 2, NKV, PS, HEAD_DIM] with NKV = kv_size / HEAD_DIM, one append row per qkv
    row, integer index tensors on the same device."""
    c = paged_kv_cache
    T = qkv.shape[0]
    return (
        isinstance(c, torch.Tensor)
        and c.dim() == 5
        and c.dtype == torch.bfloat16
        and c.is_contiguous()
        and c.device == qkv.device
        and c.shape[1] == 2
        and c.shape[2] == kv_size // HEAD_DIM
        and c.shape[4] == HEAD_DIM
        and c.shape[3] >= 1
        and all(
            x.dtype in _INT and x.device == qkv.device and x.dim() == 1 and x.is_contiguous()
            for x in (batch_indices, positions, kv_indices, kv_indptr)
        )
        and batch_indices.shape[0] == T
        and positions.shape[0] == T
    )


def qk_norm_rope_append(qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache, is_neox,
                        paged_kv_cache, append_batch_indices, append_positions, kv_indices, kv_indptr):  # fmt: skip
    """``qk_norm_rope`` that also appends the rows' K / V into ``paged_kv_cache`` exactly as
    flashinfer.page.append_paged_kv_cache(k, v, append_batch_indices, append_positions, paged_kv_cache, kv_indices,
    kv_indptr, ..., kv_layout="HND") would (same bits); None (nothing computed or written) when either part is not
    reproduced. The caller skips its append on a non-None return."""
    if not qk_norm_rope_supported(qkv, q_size, kv_size, head_dim, q_norm, k_norm, positions, cos_sin_cache,
                                  is_neox):  # fmt: skip
        return None
    if not append_supported(qkv, kv_size, paged_kv_cache, append_batch_indices, append_positions, kv_indices,
                            kv_indptr):  # fmt: skip
        return None
    return _launch(qkv, q_size, kv_size, q_norm, k_norm, positions, cos_sin_cache,
                   (paged_kv_cache, append_batch_indices, append_positions, kv_indices, kv_indptr))  # fmt: skip
