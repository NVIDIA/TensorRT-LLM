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
"""One kernel for the token-wise glue between the packed DSA projections and the MXFP4 indexer.

With the packed [q_b_proj | wq_b] GEMM (``forward_dsa_proj``) a position-free FP4 indexer runs, per layer, a
latent-cache concat, two contiguous copies of the packed output's column slices (the MLA query and the indexer
query), two ``fused_cat_fp4`` quantizations (query and key) and the FP32 cast and scale of the head weights:
six small launches, two of them on the aux stream. ``dsa_fp4_decode_prologue`` writes the same tensors in one
launch, bit for bit:

- the MLA query ``q[:, :n_mla]`` contiguous and ``latent_cache = cat([compressed_kv, k_pe])``;
- the indexer query (``n_heads`` rows of 128 per token) and the normed index key as ``fused_cat_fp4`` packs
  them: per 32 values the UE8M0 scale ``2^ceil(log2(max(amax, 1e-12) / 6))`` (exponent bits of the FP32
  ratio, rounded up), each value divided by it and rounded to E2M1 with ties toward zero, two codes per byte
  (even element in the low nibble), the four scale exponents of a row in one little-endian int32;
- ``weights.float() * scale`` (one FP32 product, as PyTorch computes it).

Given the indexer K cache and its slot mappings, the launch also stores the first ``num_tokens`` packed keys and
scale words into the cache as ``indexer_k_cache_scatter_op`` does (4-byte words at the unravelled flat byte
offsets; rows with a negative slot are skipped), so the separate scatter launch can be dropped.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

_HEAD_DIM = 128
_FP4_BLOCK = 32
# The constants of fusedCatFp4.cu as FP32 values (exactly representable as Python floats).
_INV_E2M1_MAX = float(np.float32(1.0) / np.float32(6.0))
_MIN_AMAX = float(np.float32(1.0e-12))
# Index heads quantized per program; the last program of a row handles the key, weights and latent cache.
_HEADS_PER_PROGRAM = 8


@triton.jit
def _e2m1_ties_toward_zero(s):
    ax = tl.minimum(tl.abs(s), 6.0)
    idx = (
        (ax > 0.25).to(tl.uint32)
        + (ax > 0.75).to(tl.uint32)
        + (ax > 1.25).to(tl.uint32)
        + (ax > 1.75).to(tl.uint32)
        + (ax > 2.5).to(tl.uint32)
        + (ax > 3.5).to(tl.uint32)
        + (ax > 5.0).to(tl.uint32)
    )
    sign = ((s < 0.0) & (idx != 0)).to(tl.uint32)
    return idx | (sign << 3)


@triton.jit
def _quantize_rows_fp4(x, ROWS: tl.constexpr, INV_E2M1_MAX: tl.constexpr, MIN_AMAX: tl.constexpr):
    """[ROWS, 128] FP32 -> ([ROWS, 64] packed bytes as uint32, [ROWS] int32 of the four UE8M0 exponents)."""
    xb = tl.reshape(x, (ROWS, 4, 32))
    amax = tl.maximum(tl.max(tl.abs(xb), axis=2), MIN_AMAX)
    bits = (amax * INV_E2M1_MAX).to(tl.uint32, bitcast=True)
    exp_bits = bits & 0x7F800000
    exp_bits = tl.where((bits & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
    # 1 / scale is the power of two with the mirrored exponent, so x * (1 / scale) is the IEEE x / scale.
    inv_scale = ((254 << 23) - exp_bits).to(tl.float32, bitcast=True)
    codes = _e2m1_ties_toward_zero(xb * inv_scale[:, :, None])
    lo, hi = tl.split(tl.reshape(codes, (ROWS, 64, 2)))
    packed = lo | (hi << 4)
    ue8m0 = exp_bits >> 23
    word = tl.sum(ue8m0 << (tl.arange(0, 4) * 8).to(tl.uint32)[None, :], axis=1)
    return packed, word


@triton.jit
def _cache_byte_offset(flat, d1, d2, d3, s0, s1, s2, s3):
    """Byte offset of a flat element index of a [d0, d1, d2, d3] uint8 view with strides s0..s3."""
    i3 = flat % d3
    flat = flat // d3
    i2 = flat % d2
    flat = flat // d2
    i1 = flat % d1
    i0 = flat // d1
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3


@triton.jit
def _dsa_fp4_decode_prologue_kernel(
    pq_ptr,  # bf16 [M, pq_stride]: [MLA q (N_MLA) | indexer q (N_HEADS * 128)]
    pq_stride,
    iq_ptr,  # bf16 [M, iq_stride]: indexer q (Q_SPLIT; pq then holds the MLA q only)
    iq_stride,
    kv_ptr,  # bf16 [M, KV]: normed compressed kv
    kv_stride,
    kpe_ptr,  # bf16 [M, PE]
    kpe_stride,
    k_ptr,  # bf16 [M, 128]: normed index key
    k_stride,
    w_ptr,  # bf16 [M, N_HEADS]: head weights
    w_stride,
    w_scale,  # fp32
    q_out_ptr,  # bf16 [M, N_MLA]
    lc_out_ptr,  # bf16 [M, KV + PE]
    q4_out_ptr,  # int8 [M * N_HEADS, 64]
    qs_out_ptr,  # int32 [M * N_HEADS]
    k4_out_ptr,  # int8 [M, 64]
    ks_out_ptr,  # int32 [M]
    w_out_ptr,  # fp32 [M, N_HEADS]
    cache_ptr,  # uint8 [d0, d1, d2, d3] indexer K cache (SCATTER)
    slot_fp8_ptr,  # int64 flat byte index of each token's packed key (SCATTER)
    slot_scale_ptr,  # int64 flat byte index of each token's scale word (SCATTER)
    num_scatter,
    cd1,
    cd2,
    cd3,
    cs0,
    cs1,
    cs2,
    cs3,
    N_MLA: tl.constexpr,
    N_HEADS: tl.constexpr,
    KV: tl.constexpr,
    PE: tl.constexpr,
    HPP: tl.constexpr,
    MLA_CHUNK: tl.constexpr,
    MLA_BLOCK: tl.constexpr,
    INV_E2M1_MAX: tl.constexpr,
    MIN_AMAX: tl.constexpr,
    SCATTER: tl.constexpr,
    Q_SPLIT: tl.constexpr,
    PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    NUM_Q_PARTS: tl.constexpr = N_HEADS // HPP
    if part < NUM_Q_PARTS:
        if PDL:
            tl.extra.cuda.gdc_wait()
        heads = part * HPP + tl.arange(0, HPP)
        cols = tl.arange(0, 128)
        if Q_SPLIT:
            # The MLA query is already its own contiguous tensor.
            iq_row = iq_ptr + row * iq_stride
        else:
            # MLA query: one chunk of the contiguous copy.
            offs = part * MLA_CHUNK + tl.arange(0, MLA_BLOCK)
            mask = (tl.arange(0, MLA_BLOCK) < MLA_CHUNK) & (offs < N_MLA)
            v = tl.load(pq_ptr + row * pq_stride + offs, mask=mask)
            tl.store(q_out_ptr + row * N_MLA + offs, v, mask=mask)
            iq_row = pq_ptr + row * pq_stride + N_MLA
        # Indexer query: HPP heads of 128 values.
        x = tl.load(iq_row + heads[:, None] * 128 + cols[None, :]).to(tl.float32)
        q_packed, q_word = _quantize_rows_fp4(x, HPP, INV_E2M1_MAX, MIN_AMAX)
        out_rows = row * N_HEADS + heads
        tl.store(
            q4_out_ptr + out_rows[:, None] * 64 + tl.arange(0, 64)[None, :],
            q_packed.to(tl.uint8).to(tl.int8, bitcast=True),
        )
        tl.store(qs_out_ptr + out_rows, q_word.to(tl.int32, bitcast=True))
    else:
        k_cols = tl.arange(0, 128)
        k = tl.load(k_ptr + row * k_stride + k_cols).to(tl.float32)
        k_packed, k_word = _quantize_rows_fp4(tl.reshape(k, (1, 128)), 1, INV_E2M1_MAX, MIN_AMAX)
        tl.store(
            k4_out_ptr + row * 64 + tl.arange(0, 64)[None, :],
            k_packed.to(tl.uint8).to(tl.int8, bitcast=True),
        )
        tl.store(ks_out_ptr + row + tl.arange(0, 1), k_word.to(tl.int32, bitcast=True))
        if SCATTER:
            if row < num_scatter:
                flat_k = tl.load(slot_fp8_ptr + row)
                flat_s = tl.load(slot_scale_ptr + row)
                if (flat_k >= 0) & (flat_s >= 0):
                    # Little-endian 4-byte words of the packed key, one per 4-byte chunk as the scatter op writes them.
                    words = tl.sum(
                        tl.reshape(k_packed, (16, 4))
                        << (tl.arange(0, 4) * 8).to(tl.uint32)[None, :],
                        axis=1,
                    )
                    flat = flat_k + tl.arange(0, 16).to(tl.int64) * 4
                    off = _cache_byte_offset(flat, cd1, cd2, cd3, cs0, cs1, cs2, cs3)
                    tl.store(
                        (cache_ptr + off).to(tl.pointer_type(tl.int32)),
                        words.to(tl.int32, bitcast=True),
                    )
                    off_s = _cache_byte_offset(flat_s, cd1, cd2, cd3, cs0, cs1, cs2, cs3)
                    tl.store(
                        (cache_ptr + off_s + tl.arange(0, 1)).to(tl.pointer_type(tl.int32)),
                        k_word.to(tl.int32, bitcast=True),
                    )
        h = tl.arange(0, N_HEADS)
        w = tl.load(w_ptr + row * w_stride + h).to(tl.float32)
        tl.store(w_out_ptr + row * N_HEADS + h, w * w_scale)
        kc = tl.arange(0, KV)
        tl.store(lc_out_ptr + row * (KV + PE) + kc, tl.load(kv_ptr + row * kv_stride + kc))
        pc = tl.arange(0, PE)
        tl.store(lc_out_ptr + row * (KV + PE) + KV + pc, tl.load(kpe_ptr + row * kpe_stride + pc))


def _rows_ok(t: torch.Tensor, m: int, cols: int) -> bool:
    return (
        t.dim() == 2
        and t.shape[0] == m
        and t.shape[1] == cols
        and t.dtype == torch.bfloat16
        and t.is_cuda
        and t.stride(1) == 1
    )


def dsa_fp4_decode_prologue(
    projected_q: torch.Tensor,
    num_mla_q: int,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    weight_scale: float,
    n_heads: int,
    k_cache_scatter: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = None,
    indexer_q: Optional[torch.Tensor] = None,
    pdl: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """The packed-projection glue of a position-free MXFP4 DSA indexer in one launch, or ``None`` if a shape or
    layout is not served.

    ``projected_q`` is the packed [q_b_proj | wq_b] output ``[M, num_mla_q + n_heads * 128]``; ``compressed_kv``
    the normed kv_a ``[M, kv_lora]``, ``k_pe`` ``[M, rope]``; ``index_k`` the normed index key ``[M, 128]``;
    ``index_weights`` ``[M, n_heads]`` (BF16; rows may be strided). Returns ``(q, latent_cache, q_fp4, q_scale,
    k_fp4, k_scale, weights)`` with ``q`` contiguous ``[M, num_mla_q]``, ``latent_cache`` ``[M, kv_lora + rope]``,
    ``q_fp4`` int8 ``[M * n_heads, 64]``, ``q_scale`` int32 ``[M * n_heads, 1]``, ``k_fp4`` int8 ``[M, 64]``,
    ``k_scale`` int32 ``[M, 1]`` (the ``fused_cat_fp4`` outputs) and ``weights`` FP32 ``[M, n_heads]``.

    ``k_cache_scatter`` = ``(k_cache, slot_mapping_fp8, slot_mapping_scale, num_tokens)`` (the arguments of
    ``indexer_k_cache_scatter_op``: uint8 ``[num_blocks, block_size, 1, per_token_size]`` cache, int64 slot
    mappings) also writes the keys of the first ``num_tokens`` rows into the cache; ``None`` is returned, without
    writing anything, when it is not in that form.

    With ``indexer_q`` ``[M, n_heads * 128]`` (the two-GEMM projection), ``projected_q`` is the contiguous MLA query
    ``[M, num_mla_q]`` alone and is returned as ``q``.

    ``pdl=True`` launches with programmatic dependent launch behind the kernel that writes ``projected_q`` /
    ``indexer_q``: only the query programs wait for it (griddepcontrol.wait); the key program reads the other
    inputs at once, so they must be complete before that kernel launches. There is no early trigger.
    """
    m = projected_q.shape[0] if projected_q.dim() == 2 else -1
    kv, pe = compressed_kv.shape[-1], k_pe.shape[-1]
    split = indexer_q is not None
    if split and not (
        _rows_ok(indexer_q, m, n_heads * _HEAD_DIM)
        and _rows_ok(projected_q, m, num_mla_q)
        and projected_q.is_contiguous()
    ):
        return None
    if (
        m < 1
        or not _rows_ok(projected_q, m, num_mla_q if split else num_mla_q + n_heads * _HEAD_DIM)
        or not _rows_ok(compressed_kv, m, kv)
        or not _rows_ok(k_pe, m, pe)
        or not _rows_ok(index_k, m, _HEAD_DIM)
        or not _rows_ok(index_weights, m, n_heads)
        or n_heads % _HEADS_PER_PROGRAM
        or n_heads & (n_heads - 1)
        or kv & (kv - 1)
        or pe & (pe - 1)
        or num_mla_q % 8
    ):
        return None
    if k_cache_scatter is not None:
        k_cache, slot_fp8, slot_scale, num_scatter = k_cache_scatter
        if not (
            k_cache.dim() == 4
            and k_cache.dtype == torch.uint8
            and k_cache.is_cuda
            and k_cache.shape[2] == 1
            and all(
                t.dtype == torch.int64 and t.dim() == 1 and t.is_contiguous()
                for t in (slot_fp8, slot_scale)
            )
            and 0 <= num_scatter <= min(m, slot_fp8.shape[0], slot_scale.shape[0])
        ):
            return None
        cache_args = (
            k_cache,
            slot_fp8,
            slot_scale,
            int(num_scatter),
            *(int(d) for d in k_cache.shape[1:]),
            *(int(st) for st in k_cache.stride()),
        )
    else:
        cache_args = (None, None, None, 0, 1, 1, 1, 0, 0, 0, 0)
    dev = projected_q.device
    q = projected_q if split else torch.empty((m, num_mla_q), dtype=torch.bfloat16, device=dev)
    latent_cache = torch.empty((m, kv + pe), dtype=torch.bfloat16, device=dev)
    q_fp4 = torch.empty((m * n_heads, _HEAD_DIM // 2), dtype=torch.int8, device=dev)
    q_scale = torch.empty((m * n_heads, 1), dtype=torch.int32, device=dev)
    k_fp4 = torch.empty((m, _HEAD_DIM // 2), dtype=torch.int8, device=dev)
    k_scale = torch.empty((m, 1), dtype=torch.int32, device=dev)
    weights = torch.empty((m, n_heads), dtype=torch.float32, device=dev)
    q_parts = n_heads // _HEADS_PER_PROGRAM
    mla_chunk = triton.cdiv(num_mla_q, q_parts)
    _dsa_fp4_decode_prologue_kernel[(m, q_parts + 1)](
        projected_q,
        projected_q.stride(0),
        indexer_q if split else projected_q,
        indexer_q.stride(0) if split else 0,
        compressed_kv,
        compressed_kv.stride(0),
        k_pe,
        k_pe.stride(0),
        index_k,
        index_k.stride(0),
        index_weights,
        index_weights.stride(0),
        float(np.float32(weight_scale)),
        q,
        latent_cache,
        q_fp4,
        q_scale,
        k_fp4,
        k_scale,
        weights,
        *cache_args,
        N_MLA=num_mla_q,
        N_HEADS=n_heads,
        KV=kv,
        PE=pe,
        HPP=_HEADS_PER_PROGRAM,
        MLA_CHUNK=mla_chunk,
        MLA_BLOCK=triton.next_power_of_2(mla_chunk),
        INV_E2M1_MAX=_INV_E2M1_MAX,
        MIN_AMAX=_MIN_AMAX,
        SCATTER=k_cache_scatter is not None,
        Q_SPLIT=split,
        PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )
    return q, latent_cache, q_fp4, q_scale, k_fp4, k_scale, weights
