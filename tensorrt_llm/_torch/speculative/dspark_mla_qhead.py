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
"""The MLA DSpark drafter's query head over an FP8 draft pool in one launch.

q_a_layernorm folds into the q_b GEMM (``W_qb * w`` on the un-normalized q_a,
:func:`fold_rmsnorm_weight`), and :func:`draft_qhead` applies the inverse RMS
and does the block RoPE, the latent RMSNorm, the (owner-masked) FP8 pool store,
the absorption ``q_nope @ W_UK`` and the FP8 query.
Drafter rounding: the norm's scale after the GEMM, FP32 reductions in other
orders than cuBLAS.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .spec_tail_kernels import draft_fp8_saturate


def fold_rmsnorm_weight(weight: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    """``weight * norm_weight[None, :]`` in FP32, rounded to the weight's dtype: the projection of
    ``x * norm_weight`` by ``weight`` is the projection of ``x`` by the result."""
    return (weight.float() * norm_weight.float().view(1, -1)).to(weight.dtype).contiguous()


@triton.jit
def _rms_inverse(x_ptr, stride_xt, t, t_mask, eps, WIDTH: tl.constexpr, BLOCK_K: tl.constexpr):
    """``rsqrt(mean(x[t, :WIDTH]^2) + eps)`` of BF16 rows, FP32, per row of the token block."""
    acc = tl.zeros([t.shape[0]], dtype=tl.float32)
    for k0 in tl.static_range(0, WIDTH, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + t[:, None].to(tl.int64) * stride_xt + k[None, :],
            mask=t_mask[:, None] & (k[None, :] < WIDTH),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * x, axis=1)
    return tl.math.rsqrt(acc / WIDTH + eps)


@triton.jit
def _rope_pairs(x, cos, sin):
    """Rotate the interleaved (real, imag) pairs of ``x`` [BT, 2 * P] (FP32) by ``cos`` / ``sin``
    [BT, P]; each half rounded to BF16, as the block RoPE kernel stores it."""
    re, im = tl.split(tl.reshape(x, [x.shape[0], x.shape[1] // 2, 2]))
    out_re = (re * cos - im * sin).to(tl.bfloat16).to(tl.float32)
    out_im = (im * cos + re * sin).to(tl.bfloat16).to(tl.float32)
    return tl.reshape(tl.join(out_re, out_im), [x.shape[0], x.shape[1]])


@triton.jit
def _qhead_kernel(
    qa_ptr,
    stride_qa,
    kv_ptr,
    stride_kv,
    q_ptr,
    stride_qt,
    stride_qh,
    w_ptr,
    stride_wh,
    stride_wk,
    kv_norm_ptr,
    positions_ptr,
    freqs_ptr,
    out_ptr,
    stride_ot,
    stride_oh,
    pool_ptr,
    stride_pool_page,
    stride_pool_off,
    pages_ptr,
    offsets_ptr,
    owned_ptr,
    num_tokens,
    num_heads,
    q_eps,
    kv_eps,
    Q_LORA: tl.constexpr,
    NOPE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_OWNED: tl.constexpr,
):
    pid = tl.program_id(0)
    half: tl.constexpr = ROPE // 2
    if pid < num_tokens:
        # Block latent of token ``pid``: RMSNorm of the latent, RoPE of k_pe, FP8 into the pool.
        row = pid
        store = True
        if HAS_OWNED:
            store = tl.load(owned_ptr + row) != 0
        if store:
            kv_row = kv_ptr + row.to(tl.int64) * stride_kv
            d = tl.arange(0, LATENT)
            x = tl.load(kv_row + d).to(tl.float32)
            inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / LATENT + kv_eps)
            w = tl.load(kv_norm_ptr + d).to(tl.float32)
            latent = ((x * inv) * w).to(tl.bfloat16).to(tl.float32)
            pos = tl.load(positions_ptr + row)
            p = tl.arange(0, half)
            cos = tl.load(freqs_ptr + pos * ROPE + p * 2)
            sin = tl.load(freqs_ptr + pos * ROPE + p * 2 + 1)
            r = tl.arange(0, ROPE)
            pe = tl.load(kv_row + LATENT + r).to(tl.float32)
            pe = tl.reshape(
                _rope_pairs(tl.reshape(pe, [1, ROPE]), cos[None, :], sin[None, :]), [ROPE]
            )
            page = tl.load(pages_ptr + row)
            off = tl.load(offsets_ptr + row)
            dst = pool_ptr + page * stride_pool_page + off * stride_pool_off
            tl.store(dst + d, draft_fp8_saturate(latent))
            tl.store(dst + LATENT + r, draft_fp8_saturate(pe))
    else:
        q_pid = pid - num_tokens
        col_blocks = LATENT // BLOCK_N + 1
        col_block = q_pid % col_blocks
        head = (q_pid // col_blocks) % num_heads
        t_block = q_pid // (col_blocks * num_heads)
        t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t < num_tokens
        k = tl.arange(0, NOPE)
        n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        is_latent = col_block * BLOCK_N < LATENT
        w = tl.load(
            w_ptr + head * stride_wh + k[:, None] * stride_wk + n[None, :],
            mask=is_latent & (n[None, :] < LATENT),
            other=0.0,
        )
        # q_a's RMSNorm, folded: q = inverse_rms(q_a) * (q_a @ (W_qb * w)^T). The per-row scale
        # is linear, so it is applied after the absorption / RoPE: the q_a loads of the inverse
        # RMS overlap the dot product.
        q_row = q_ptr + t[:, None].to(tl.int64) * stride_qt + head * stride_qh
        if is_latent:
            a = tl.load(q_row + k[None, :], mask=t_mask[:, None], other=0.0)
            acc = tl.dot(a, w)
            inv = _rms_inverse(qa_ptr, stride_qa, t, t_mask, q_eps, Q_LORA, 256)
            latent_vals = (acc * inv[:, None]).to(tl.bfloat16).to(tl.float32)
            tl.store(
                out_ptr + t[:, None] * stride_ot + head * stride_oh + n[None, :],
                draft_fp8_saturate(latent_vals),
                mask=t_mask[:, None],
            )
        else:
            r = tl.arange(0, ROPE)
            pe = tl.load(q_row + NOPE + r[None, :], mask=t_mask[:, None], other=0.0).to(tl.float32)
            inv = _rms_inverse(qa_ptr, stride_qa, t, t_mask, q_eps, Q_LORA, 256)
            pos = tl.load(positions_ptr + t, mask=t_mask, other=0)
            p = tl.arange(0, half)
            cos = tl.load(
                freqs_ptr + pos[:, None] * ROPE + p[None, :] * 2, mask=t_mask[:, None], other=1.0
            )
            sin = tl.load(
                freqs_ptr + pos[:, None] * ROPE + p[None, :] * 2 + 1,
                mask=t_mask[:, None],
                other=0.0,
            )
            pe = _rope_pairs(pe, cos, sin) * inv[:, None]
            pe = pe.to(tl.bfloat16).to(tl.float32)
            tl.store(
                out_ptr + t[:, None] * stride_ot + head * stride_oh + LATENT + r[None, :],
                draft_fp8_saturate(pe),
                mask=t_mask[:, None],
            )


def draft_qhead(
    q_a: torch.Tensor,
    kv: torch.Tensor,
    q: torch.Tensor,
    q_eps: float,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    kv_eps: float,
    positions: torch.Tensor,
    freqs: torch.Tensor,
    pool: torch.Tensor,
    pages: torch.Tensor,
    offsets: torch.Tensor,
    rows_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The decode's FP8 query and the block's FP8 pool rows of one drafter layer from the q_a
    and kv_a rows and the q_b projection of the un-normalized q_a, in one launch.

    Args:
        q_a: BF16 ``[T, q_lora]`` q_a rows, ``kv``: BF16 ``[T, kv_lora_rank + rope]`` kv_a rows
            (unit inner strides; e.g. the two column ranges of the ``[q_a | kv_a]`` GEMM).
        q: BF16 ``[T, H, nope + rope]``: ``q_a @ (W_qb * w_qa)^T`` (``fold_rmsnorm_weight``), i.e.
            q_b of q_a_layernorm's input before its per-row inverse RMS.
        q_eps: q_a_layernorm's epsilon.
        w_uk: BF16 ``[H, nope, kv_lora_rank]`` K absorption.
        kv_norm_weight: BF16 ``[kv_lora_rank]`` kv_a_layernorm weight; ``kv_eps`` its epsilon.
        positions: int64 ``[T]`` RoPE positions; ``freqs`` FP32 ``[P, rope // 2, 2]`` (cos, sin).
        pool: FP8 E4M3 ``[pages, page_size, kv_lora_rank + rope]``; ``pages`` / ``offsets`` int64
            ``[T]`` destinations; ``rows_mask``: optional ``[T]``, rows where it is 0 write nothing.

    Returns:
        FP8 E4M3 ``[T, H, kv_lora_rank + rope]`` query. Same operations as q_a_layernorm, the q_b
        GEMM, the fused block RoPE, the masked scatter, the absorption and the FP8 cast, with the
        q_a norm's per-row scale applied after the GEMM and the FP32 reductions in another order:
        drafter rounding.
    """
    num_tokens, num_heads, q_dim = q.shape
    nope, latent = w_uk.shape[1], w_uk.shape[2]
    rope = q_dim - nope
    q_lora = q_a.shape[1]
    assert q_a.dtype == kv.dtype == q.dtype == w_uk.dtype == kv_norm_weight.dtype == torch.bfloat16
    assert q_a.shape[0] == kv.shape[0] == num_tokens and kv.shape[1] == latent + rope
    assert q_a.stride(1) == 1 and kv.stride(1) == 1
    assert q.stride(-1) == 1 and w_uk.is_contiguous() and kv_norm_weight.is_contiguous()
    assert w_uk.shape[0] == num_heads and kv_norm_weight.numel() == latent
    assert pool.dtype == torch.float8_e4m3fn and pool.shape[-1] == latent + rope
    assert pool.stride(-1) == 1
    assert positions.dtype == pages.dtype == offsets.dtype == torch.int64
    assert positions.numel() == pages.numel() == offsets.numel() == num_tokens
    assert freqs.dtype == torch.float32 and freqs.is_contiguous()
    assert freqs.shape[1:] == (rope // 2, 2)
    out = torch.empty(
        (num_tokens, num_heads, latent + rope), dtype=torch.float8_e4m3fn, device=q.device
    )
    if num_tokens == 0:
        return out
    if rows_mask is not None:
        rows_mask = rows_mask.reshape(-1)
        assert rows_mask.numel() == num_tokens and rows_mask.is_contiguous()
        if rows_mask.dtype == torch.bool:
            rows_mask = rows_mask.view(torch.uint8)
    # 256 columns x 8 warps per program.
    block_n = 256
    warps = 8
    # The kernel spans latent, nope and rope with tl.arange (powers of two) and gives the rope
    # columns one column block after the latent ones.
    for name, dim in (("kv_lora_rank", latent), ("nope", nope), ("rope", rope)):
        assert dim > 0 and dim & (dim - 1) == 0, f"{name} {dim} is not a power of two"
    assert latent % block_n == 0 and rope <= block_n, (latent, rope)
    block_t = 16 if num_tokens <= 16 else 32
    q_programs = triton.cdiv(num_tokens, block_t) * num_heads * (latent // block_n + 1)
    positions = positions.reshape(-1)
    _qhead_kernel[(num_tokens + q_programs,)](
        q_a,
        q_a.stride(0),
        kv,
        kv.stride(0),
        q,
        q.stride(0),
        q.stride(1),
        w_uk,
        w_uk.stride(0),
        w_uk.stride(1),
        kv_norm_weight,
        positions,
        freqs,
        out,
        out.stride(0),
        out.stride(1),
        pool,
        pool.stride(0),
        pool.stride(1),
        pages.reshape(-1),
        offsets.reshape(-1),
        pages if rows_mask is None else rows_mask,
        num_tokens,
        num_heads,
        float(q_eps),
        float(kv_eps),
        Q_LORA=q_lora,
        NOPE=nope,
        LATENT=latent,
        ROPE=rope,
        BLOCK_T=block_t,
        BLOCK_N=block_n,
        HAS_OWNED=rows_mask is not None,
        num_warps=warps,
    )
    return out
