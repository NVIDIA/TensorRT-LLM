# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton attention output gate + NVFP4 quantize: ``fp4_quantize(attn * sigmoid(gate))``.

``fused_sigmoid_mul_inplace``'s expression (``attention * sigmoid(gate)``) rounded to
BF16 as stored, then the ``fp4_quantize`` tail of ``quantization/_nvfp4_triton``,
bitwise equal to the two-kernel path; one extra program row zeroes the SF padding
rows. The gate may be a ``[tokens, heads, head_dim]`` strided view (the gate halves of
the interleaved ``[q0, g0, q1, g1, ...]`` projection); the attention output is not
modified. With PDL every access stays behind ``griddepcontrol.wait`` and dependents
are released after the stores.
"""

import functools

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ..quantization._nvfp4_triton import (
    _cvt_e2m1x8,
    _cvt_e4m3_bits,
    _gdc_wait_with_memory_clobber,
    _mul_rn,
    _rcp_approx_ftz,
    _sf_offset,
)

_BLOCK = 512


@triton.jit
def _fused_sigmoid_mul_nvfp4_kernel(
    fp4_ptr,  # int32 view of uint8 [num_tokens, hidden_size / 2]
    sf_ptr,  # uint8 swizzled SF, ceil128(num_tokens) * NUM_K_TILES * 4
    attention_ptr,
    gate_ptr,
    scale_ptr,  # 1-element fp32 global scale (6 / input_scale)
    num_tokens,
    attention_stride_token,
    gate_stride_token,
    gate_stride_head,
    hidden_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """_fused_sigmoid_mul_kernel's expression, rounded to BF16 as stored,
    then fp4_quantize(r, scale, 16) (the SwiGLU+NVFP4 kernel's quantize tail).
    Program row num_tokens zeroes the SF padding rows. With USE_PDL all
    accesses stay behind griddepcontrol.wait and dependents are released
    after the stores."""
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    i = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    NSF: tl.constexpr = BLOCK_SIZE // 16
    sf_cols = j * NSF + tl.arange(0, NSF)
    num_sf_cols: tl.constexpr = hidden_size // 16
    if i < num_tokens:
        offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_size
        head = offsets // head_dim
        dim = offsets - head * head_dim
        attention = tl.load(
            attention_ptr + i * attention_stride_token + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + i * gate_stride_token + head * gate_stride_head + dim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        r = (attention * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)

        h2 = tl.reshape(r, [NSF, 16])
        vmax = tl.max(tl.abs(h2), axis=1)
        s = tl.load(scale_ptr)
        s_v = tl.full([NSF], 0.0, tl.float32) + s
        r6 = _rcp_approx_ftz(tl.full([NSF], 6.0, tl.float32))
        sf_val = _mul_rn(s_v, _mul_rn(vmax, r6))
        sf_u8 = _cvt_e4m3_bits(sf_val).to(tl.uint8)
        sf_f = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        out_scale = tl.where(vmax != 0.0, _rcp_approx_ftz(_mul_rn(sf_f, _rcp_approx_ftz(s_v))), 0.0)
        y = _mul_rn(h2, tl.broadcast_to(out_scale[:, None], [NSF, 16]))
        y8 = tl.reshape(y, [BLOCK_SIZE // 8, 2, 2, 2])
        p, q = tl.split(y8)
        p0, p1 = tl.split(p)
        q0, q1 = tl.split(q)
        p00, p01 = tl.split(p0)
        p10, p11 = tl.split(p1)
        q00, q01 = tl.split(q0)
        q10, q11 = tl.split(q1)
        packed = _cvt_e2m1x8(p00, q00, p10, q10, p01, q01, p11, q11)
        w_offsets = j * (BLOCK_SIZE // 8) + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            fp4_ptr + i * (hidden_size // 8) + w_offsets, packed, mask=w_offsets < hidden_size // 8
        )
        sf_off = _sf_offset(i, sf_cols, NUM_K_TILES)
        tl.store(sf_ptr + sf_off, sf_u8, mask=sf_cols < num_sf_cols)
    else:
        m0 = (num_tokens // 128) * 128
        rows = m0 + tl.arange(0, 128).to(tl.int64)
        off = _sf_offset(rows[:, None], sf_cols[None, :], NUM_K_TILES)
        pmask = (rows >= num_tokens)[:, None] & (sf_cols < num_sf_cols)[None, :]
        tl.store(sf_ptr + off, tl.zeros([128, NSF], tl.uint8), mask=pmask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@functools.lru_cache(maxsize=None)
def _sm_at_least_90(device_index: int) -> bool:
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor >= 90


def sigmoid_mul_nvfp4(
    attention_output: torch.Tensor, gate: torch.Tensor, global_scale: torch.Tensor
):
    """``(fp4 [T, H/2] uint8, swizzled sf uint8)`` of ``attention_output * sigmoid(gate)``;
    ``gate`` ``[T, H]`` or a ``[T, heads, head_dim]`` view with unit inner stride."""
    num_tokens, hidden_size = attention_output.shape
    if gate.dim() == 3:
        head_dim = gate.shape[2]
        gate_stride_token = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        head_dim = hidden_size
        gate_stride_token = gate.stride(0)
        gate_stride_head = hidden_size
    num_k_tiles = (hidden_size // 16 + 3) // 4
    m_pad = (num_tokens + 127) // 128 * 128
    fp4 = torch.empty(
        (num_tokens, hidden_size // 2), dtype=torch.uint8, device=attention_output.device
    )
    sf = torch.empty((m_pad * num_k_tiles * 4,), dtype=torch.uint8, device=attention_output.device)
    grid_rows = num_tokens + (1 if num_tokens % 128 != 0 else 0)
    if grid_rows == 0:
        return fp4, sf
    grid = (grid_rows, triton.cdiv(hidden_size, _BLOCK))
    use_pdl = get_env_enable_pdl() and _sm_at_least_90(attention_output.device.index)
    _fused_sigmoid_mul_nvfp4_kernel[grid](
        fp4.view(torch.int32),
        sf,
        attention_output,
        gate,
        global_scale,
        num_tokens,
        attention_output.stride(0),
        gate_stride_token,
        gate_stride_head,
        hidden_size=hidden_size,
        head_dim=head_dim,
        BLOCK_SIZE=_BLOCK,
        NUM_K_TILES=num_k_tiles,
        USE_PDL=use_pdl,
        num_warps=4,
        launch_pdl=use_pdl,
    )
    return fp4, sf
