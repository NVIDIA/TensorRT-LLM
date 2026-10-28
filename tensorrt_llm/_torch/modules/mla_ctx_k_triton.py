# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-pass Triton kernel that builds the MLA context-phase K tensor.

K[t, h, :] = concat(k_nope[t, h, :], k_pe[t, :]) is a pure copy, so the result
is bit-exact vs torch.cat((k_nope, k_pe.expand(-1, H, -1)), dim=-1).
Enabled with TRTLLM_MLA_TRITON_CTX_K=1.
"""
import os

import torch
import triton
import triton.language as tl


def is_enabled() -> bool:
    return os.environ.get("TRTLLM_MLA_TRITON_CTX_K", "0") == "1"


@triton.jit
def _mla_concat_k_kernel(
    k_nope_ptr,
    k_pe_ptr,
    out_ptr,
    num_tokens,
    num_heads,
    nope_stride_t,
    nope_stride_h,
    pe_stride_t,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # int64 offsets: tokens * heads * head_dim can exceed 2^31.
    t = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    h = (tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    t_ok = t < num_tokens
    row_ok = t_ok[:, None] & (h[None, :] < num_heads)  # [BT, BH]
    out_row = (t[:, None] * num_heads + h[None, :]) * (D_NOPE + D_ROPE)  # [BT, BH]

    # non-positional part: one copy per (token, head) row
    cn = tl.arange(0, BLOCK_NOPE)
    m_n = row_ok[:, :, None] & (cn < D_NOPE)[None, None, :]
    src_n = (
        t[:, None, None] * nope_stride_t + h[None, :, None] * nope_stride_h + cn[None, None, :]
    )
    vn = tl.load(k_nope_ptr + src_n, mask=m_n, other=0)
    tl.store(out_ptr + out_row[:, :, None] + cn[None, None, :], vn, mask=m_n)

    # rotary part: loaded once per token, written to each head of the tile
    cr = tl.arange(0, BLOCK_ROPE)
    m_pe = t_ok[:, None] & (cr < D_ROPE)[None, :]
    vp = tl.load(k_pe_ptr + t[:, None] * pe_stride_t + cr[None, :], mask=m_pe, other=0)
    m_r = row_ok[:, :, None] & (cr < D_ROPE)[None, None, :]
    vp3 = tl.broadcast_to(vp[:, None, :], [BLOCK_T, BLOCK_H, BLOCK_ROPE])
    tl.store(out_ptr + out_row[:, :, None] + D_NOPE + cr[None, None, :], vp3, mask=m_r)


def mla_concat_k(k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """k_nope: [T, H, d_nope] (last dim contiguous, any t/h strides);
    k_pe: [T, 1, d_rope] or [T, d_rope] (last dim contiguous).
    Returns contiguous [T, H, d_nope + d_rope], same dtype."""
    T, H, d_nope = k_nope.shape
    k_pe = k_pe.reshape(T, -1) if k_pe.dim() == 3 else k_pe
    d_rope = k_pe.shape[-1]
    assert k_nope.dtype == k_pe.dtype and k_nope.stride(-1) == 1 and k_pe.stride(-1) == 1
    out = torch.empty((T, H, d_nope + d_rope), dtype=k_nope.dtype, device=k_nope.device)
    if T == 0:
        return out
    # Copy-only: reinterpret fp8 as uint8 so Triton never does dtype conversion.
    if k_nope.element_size() == 1:
        k_nope, k_pe, out_v = (x.view(torch.uint8) for x in (k_nope, k_pe, out))
    else:
        out_v = out
    block_h = 8
    block_t = 4
    grid = (triton.cdiv(T, block_t), triton.cdiv(H, block_h))
    _mla_concat_k_kernel[grid](
        k_nope,
        k_pe,
        out_v,
        T,
        H,
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe.stride(0),
        D_NOPE=d_nope,
        D_ROPE=d_rope,
        BLOCK_NOPE=triton.next_power_of_2(d_nope),
        BLOCK_ROPE=triton.next_power_of_2(d_rope),
        BLOCK_T=block_t,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out
