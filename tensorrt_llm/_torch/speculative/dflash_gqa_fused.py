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
"""Fused small-op kernels for the GQA DFlash / DSpark drafter (FP8 paged context cache).

The GQA drafter (Qwen3-shaped, e.g. the K3 DSpark drafter) runs each decode step as a
long chain of tiny eager PyTorch ops: the context K/V projection output is RMS-normed,
scaled, rotated, masked, cast to FP8 and scattered into the paged pool with ~40 launches
per step, and every drafter layer spends another ~12 launches between the QKV GEMM and
the attention kernel (q/k norm, RoPE, FP8 cast of q, k and v, page scatter, the folded
query layout).  Two kernels replace them:

* :func:`dflash_gqa_ctx_kv_store` -- the context K/V store of the K+1 target tokens of
  every generation request: k_norm (``F.rms_norm`` then the per-layer weight multiply, as
  ``DFlashForCausalLM.precompute_context_kv``), neox RoPE from the flashinfer cos/sin cache,
  the write mask, the FP8 E4M3 cast and the page scatter, for all layers at once.
* :func:`dflash_gqa_qk_prep` -- one drafter layer's block-token prologue: q_norm / k_norm,
  RoPE, FP8 casts, the K/V page scatter of the block's own tokens and the query already in
  the head-folded layout the block decode kernel reads.

Both are selected by ``TLLM_DFLASH_GQA_FUSED=1`` (context store) and
``TLLM_DFLASH_GQA_FUSED_LAYER=1`` (per-layer kernel), default off, and are CUDA-graph
capturable (shapes are host-static; page / offset operands come from device tensors).
"""

import os

import torch
import triton
import triton.language as tl

GQA_FUSED_ENV_VAR = "TLLM_DFLASH_GQA_FUSED"
GQA_FUSED_LAYER_ENV_VAR = "TLLM_DFLASH_GQA_FUSED_LAYER"
#: ``TLLM_DFLASH_GQA_NORM_ROUND``: 1 = the q/k RMSNorm output is rounded to bf16 before the
#: weight multiply (what the generic torch fallback does), 0 = one rounding after the weight
#: multiply (what the flashinfer RMSNorm kernel does).  Chosen against the module in the unit
#: test; default is set from that result.
GQA_NORM_ROUND_ENV_VAR = "TLLM_DFLASH_GQA_NORM_ROUND"


def gqa_fused_enabled() -> bool:
    """``TLLM_DFLASH_GQA_FUSED=1``: fused context K/V store and bookkeeping for the GQA drafter."""
    return os.environ.get(GQA_FUSED_ENV_VAR, "0") == "1"


def gqa_fused_layer_enabled() -> bool:
    """``TLLM_DFLASH_GQA_FUSED_LAYER=1``: fused per-layer q/k prologue (needs the context fusion)."""
    return os.environ.get(GQA_FUSED_LAYER_ENV_VAR, "0") == "1"


def gqa_norm_round() -> int:
    return int(os.environ.get(GQA_NORM_ROUND_ENV_VAR, "0"))


@triton.jit
def _to_e4m3(x):
    """Saturated E4M3 byte of an fp32 value (round to nearest even), as the uint8 bit pattern."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    return x.to(tl.float8e4nv).to(tl.uint8, bitcast=True)


@triton.jit
def _ctx_kv_store_kernel(
    kv_ptr,
    stride_kv_tok,
    kw_ptr,
    cs_ptr,
    stride_cs,
    pos_ptr,
    mask_ptr,
    pages_ptr,
    offsets_ptr,
    pool_ptrs_ptr,
    stride_pool_page,
    stride_pool_kv,
    stride_pool_head,
    stride_pool_off,
    eps,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    HALF: tl.constexpr,
):
    tok = tl.program_id(0)
    layer = tl.program_id(1)
    h = tl.arange(0, NKV)[:, None]
    d = tl.arange(0, HALF)[None, :]
    base = kv_ptr + tok.to(tl.int64) * stride_kv_tok + layer * (2 * NKV * HD)

    # K: F.rms_norm (no weight; fp32 math, bf16 result), then * per-layer weight (bf16 multiply).
    k1 = tl.load(base + h * HD + d).to(tl.float32)
    k2 = tl.load(base + h * HD + HALF + d).to(tl.float32)
    ss = tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)
    rstd = tl.rsqrt(ss / HD + eps)[:, None]
    n1 = (k1 * rstd).to(tl.bfloat16).to(tl.float32)
    n2 = (k2 * rstd).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(kw_ptr + layer * HD + d).to(tl.float32)
    w2 = tl.load(kw_ptr + layer * HD + HALF + d).to(tl.float32)
    n1 = (n1 * w1).to(tl.bfloat16).to(tl.float32)
    n2 = (n2 * w2).to(tl.bfloat16).to(tl.float32)

    # neox RoPE from the fp32 cos/sin cache ([cos | sin] per position), bf16 result.
    pos = tl.load(pos_ptr + tok).to(tl.int64)
    cos = tl.load(cs_ptr + pos * stride_cs + d)
    sin = tl.load(cs_ptr + pos * stride_cs + HALF + d)
    o1 = (n1 * cos - n2 * sin).to(tl.bfloat16).to(tl.float32)
    o2 = (n2 * cos + n1 * sin).to(tl.bfloat16).to(tl.float32)

    # Write mask (a bf16 multiply by 0 or 1) and the FP8 cast.
    keep = tl.load(mask_ptr + tok).to(tl.float32)
    o1 = (o1 * keep).to(tl.bfloat16).to(tl.float32)
    o2 = (o2 * keep).to(tl.bfloat16).to(tl.float32)
    v1 = tl.load(base + NKV * HD + h * HD + d).to(tl.float32)
    v2 = tl.load(base + NKV * HD + h * HD + HALF + d).to(tl.float32)
    v1 = (v1 * keep).to(tl.bfloat16).to(tl.float32)
    v2 = (v2 * keep).to(tl.bfloat16).to(tl.float32)

    page = tl.load(pages_ptr + tok).to(tl.int64)
    off = tl.load(offsets_ptr + tok).to(tl.int64)
    pool = tl.load(pool_ptrs_ptr + layer).to(tl.pointer_type(tl.uint8))
    dst = pool + page * stride_pool_page + h * stride_pool_head + off * stride_pool_off
    tl.store(dst + d, _to_e4m3(o1))
    tl.store(dst + HALF + d, _to_e4m3(o2))
    dst_v = dst + stride_pool_kv
    tl.store(dst_v + d, _to_e4m3(v1))
    tl.store(dst_v + HALF + d, _to_e4m3(v2))


def dflash_gqa_ctx_kv_store(
    kv_flat: torch.Tensor,
    k_norm_weights: torch.Tensor,
    k_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    pool_ptrs: torch.Tensor,
    pool0: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """Store the context K/V of ``N`` tokens in every layer's FP8 paged pool.

    Args:
        kv_flat: ``[N, L * 2 * nkv * hd]`` bf16 output of the fused context K/V GEMM, laid out
            ``[L][K|V][nkv][hd]`` per row.
        k_norm_weights: ``[L, hd]`` stacked per-layer k_norm weights.
        cos_sin_cache: ``[max_pos, hd]`` fp32 flashinfer cache (cos half, then sin half).
        positions: ``[N]`` RoPE positions.  mask: ``[N]`` int8 write mask (0 stores zeros).
        pages, offsets: ``[N]`` destination page / in-page offset (each layer's pool).
        pool_ptrs: ``[L]`` int64 base addresses of the per-layer pools; pool0: layer 0's
            ``[pages, 2, nkv, page_size, hd]`` FP8 view (all layers share its strides).
    """
    num_tokens = kv_flat.shape[0]
    num_layers = k_norm_weights.shape[0]
    if num_tokens == 0:
        return
    assert kv_flat.stride(-1) == 1 and kv_flat.shape[1] == num_layers * 2 * num_kv_heads * head_dim
    assert pool0.dtype == torch.float8_e4m3fn and pool0.stride(-1) == 1
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.stride(-1) == 1
    assert head_dim % 2 == 0
    _ctx_kv_store_kernel[(num_tokens, num_layers)](
        kv_flat,
        kv_flat.stride(0),
        k_norm_weights,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        positions,
        mask,
        pages,
        offsets,
        pool_ptrs,
        pool0.stride(0),
        pool0.stride(1),
        pool0.stride(2),
        pool0.stride(3),
        float(k_norm_eps),
        NKV=num_kv_heads,
        HD=head_dim,
        HALF=head_dim // 2,
        num_warps=1,
    )


@triton.jit
def _qk_prep_kernel(
    qkv_ptr,
    stride_qkv,
    qw_ptr,
    kw_ptr,
    cs_ptr,
    stride_cs,
    pos_ptr,
    pages_ptr,
    offsets_ptr,
    pool_ptr,
    stride_pool_page,
    stride_pool_kv,
    stride_pool_head,
    stride_pool_off,
    qout_ptr,
    eps,
    BLK: tl.constexpr,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    HALF: tl.constexpr,
    NORM_ROUND: tl.constexpr,
):
    tok = tl.program_id(0)
    b = tok // BLK
    j = tok % BLK
    GROUP: tl.constexpr = NQ // NKV
    d = tl.arange(0, HALF)[None, :]
    base = qkv_ptr + tok.to(tl.int64) * stride_qkv
    pos = tl.load(pos_ptr + tok).to(tl.int64)
    cos = tl.load(cs_ptr + pos * stride_cs + d)
    sin = tl.load(cs_ptr + pos * stride_cs + HALF + d)

    # ---- q: RMSNorm * weight, RoPE, FP8, written head-folded ([B, kvh, j, group, hd]) ----
    hq = tl.arange(0, NQ)[:, None]
    q1 = tl.load(base + hq * HD + d).to(tl.float32)
    q2 = tl.load(base + hq * HD + HALF + d).to(tl.float32)
    rq = tl.rsqrt((tl.sum(q1 * q1, axis=1) + tl.sum(q2 * q2, axis=1)) / HD + eps)[:, None]
    qw1 = tl.load(qw_ptr + d).to(tl.float32)
    qw2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
    if NORM_ROUND:
        qn1 = (((q1 * rq).to(tl.bfloat16).to(tl.float32)) * qw1).to(tl.bfloat16).to(tl.float32)
        qn2 = (((q2 * rq).to(tl.bfloat16).to(tl.float32)) * qw2).to(tl.bfloat16).to(tl.float32)
    else:
        qn1 = (q1 * rq * qw1).to(tl.bfloat16).to(tl.float32)
        qn2 = (q2 * rq * qw2).to(tl.bfloat16).to(tl.float32)
    qo1 = (qn1 * cos - qn2 * sin).to(tl.bfloat16).to(tl.float32)
    qo2 = (qn2 * cos + qn1 * sin).to(tl.bfloat16).to(tl.float32)
    row = b * (NKV * BLK * GROUP) + ((hq // GROUP) * BLK + j) * GROUP + (hq % GROUP)
    qdst = qout_ptr + row.to(tl.int64) * HD
    tl.store(qdst + d, _to_e4m3(qo1))
    tl.store(qdst + HALF + d, _to_e4m3(qo2))

    # ---- k: RMSNorm * weight, RoPE; v: raw; both FP8 into the block's pool slots ----
    hk = tl.arange(0, NKV)[:, None]
    kbase = base + NQ * HD
    k1 = tl.load(kbase + hk * HD + d).to(tl.float32)
    k2 = tl.load(kbase + hk * HD + HALF + d).to(tl.float32)
    rk = tl.rsqrt((tl.sum(k1 * k1, axis=1) + tl.sum(k2 * k2, axis=1)) / HD + eps)[:, None]
    kw1 = tl.load(kw_ptr + d).to(tl.float32)
    kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
    if NORM_ROUND:
        kn1 = (((k1 * rk).to(tl.bfloat16).to(tl.float32)) * kw1).to(tl.bfloat16).to(tl.float32)
        kn2 = (((k2 * rk).to(tl.bfloat16).to(tl.float32)) * kw2).to(tl.bfloat16).to(tl.float32)
    else:
        kn1 = (k1 * rk * kw1).to(tl.bfloat16).to(tl.float32)
        kn2 = (k2 * rk * kw2).to(tl.bfloat16).to(tl.float32)
    ko1 = (kn1 * cos - kn2 * sin).to(tl.bfloat16).to(tl.float32)
    ko2 = (kn2 * cos + kn1 * sin).to(tl.bfloat16).to(tl.float32)
    v1 = tl.load(kbase + NKV * HD + hk * HD + d).to(tl.float32)
    v2 = tl.load(kbase + NKV * HD + hk * HD + HALF + d).to(tl.float32)

    page = tl.load(pages_ptr + tok).to(tl.int64)
    off = tl.load(offsets_ptr + tok).to(tl.int64)
    dst = pool_ptr + page * stride_pool_page + hk * stride_pool_head + off * stride_pool_off
    tl.store(dst + d, _to_e4m3(ko1))
    tl.store(dst + HALF + d, _to_e4m3(ko2))
    dst_v = dst + stride_pool_kv
    tl.store(dst_v + d, _to_e4m3(v1))
    tl.store(dst_v + HALF + d, _to_e4m3(v2))


def dflash_gqa_qk_prep(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    layer_cache: torch.Tensor,
    batch: int,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """One drafter layer's block prologue; returns the FP8 head-folded query.

    ``qkv`` is the ``[B * blk, (nq + 2 * nkv) * hd]`` QKV projection output.  Writes the block's
    K (norm + RoPE) and V into ``layer_cache`` (``[pages, 2, nkv, page_size, hd]`` FP8) at
    ``(pages[t], offsets[t])`` and returns ``q_folded [B, nkv * blk * group, hd]`` FP8, equal to
    ``q_norm -> RoPE -> .to(fp8) -> reshape(B, blk, nkv, group, hd).permute(0, 2, 1, 3, 4)
    .reshape(B, -1, hd)``.
    """
    tokens = batch * block_size
    group = num_heads // num_kv_heads
    q_folded = torch.empty(
        (batch, num_kv_heads * block_size * group, head_dim),
        dtype=torch.float8_e4m3fn,
        device=qkv.device,
    )
    if tokens == 0:
        return q_folded
    assert qkv.stride(-1) == 1 and qkv.shape == (tokens, (num_heads + 2 * num_kv_heads) * head_dim)
    assert layer_cache.dtype == torch.float8_e4m3fn and layer_cache.stride(-1) == 1
    pool_bytes = layer_cache.view(torch.uint8)
    _qk_prep_kernel[(tokens,)](
        qkv,
        qkv.stride(0),
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        positions,
        pages,
        offsets,
        pool_bytes,
        pool_bytes.stride(0),
        pool_bytes.stride(1),
        pool_bytes.stride(2),
        pool_bytes.stride(3),
        q_folded.view(torch.uint8),
        float(eps),
        BLK=block_size,
        NQ=num_heads,
        NKV=num_kv_heads,
        HD=head_dim,
        HALF=head_dim // 2,
        NORM_ROUND=gqa_norm_round(),
        num_warps=1,
    )
    return q_folded
