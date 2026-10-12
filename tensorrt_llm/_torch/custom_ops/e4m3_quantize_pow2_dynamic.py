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
"""Dynamic per-tensor E4M3 quantization with a power-of-two scale:
``trtllm::e4m3_quantize_pow2_dynamic``.

Over the whole tensor ``x`` (bf16 / fp16 / fp32, contiguous)::

    amax          = max(max|x|, 1e-4)                    # fp32
    s             = 2^ceil(log2(amax * (1 / 448)))       # from the fp32 bits, exact
    out           = e4m3_rne(clamp(x * (1 / s), -448, 448))   # exact scaling by a power of two
    scale_out[0]  = s * scale_mult                       # one fp32 multiply

so ``out * s`` is ``x`` rounded to E4M3's 3-bit mantissa (no value saturates:
``|x| / s <= 448``), and ``scale_out`` can be handed on directly as a fused
dequantization factor (e.g. an attention's ``bmm1_scale`` with
``scale_mult = softmax_scale * kv_scale``). The scale is the same recipe as the
MXFP8 activation quantizer's UE8M0 scale, taken over the whole tensor.

Three launches: a memset of a one-float amax scratch, a grid of block-amax
programs with an atomic max, and a grid of quantizing programs (program 0 also
writes ``scale_out``). Graph-capturable; no host synchronization.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_AMAX_FLOOR = 1e-4
_IN_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _amax_kernel(x_ptr, amax_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + off, mask=off < n, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x), axis=0))


@triton.jit
def _pow2_scale(amax_ptr, floor):
    a = tl.maximum(tl.load(amax_ptr), floor) * (1.0 / 448.0)
    bits = a.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)  # biased exponent of the ceil
    return (e << 23).to(tl.float32, bitcast=True)


@triton.jit
def _quant_kernel(x_ptr, out_ptr, amax_ptr, scale_ptr, n, scale_mult, floor, BLOCK: tl.constexpr):
    s = _pow2_scale(amax_ptr, floor)
    inv = ((254 << 23) - s.to(tl.int32, bitcast=True)).to(tl.float32, bitcast=True)  # exact 1/s
    pid = tl.program_id(0)
    off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + off, mask=off < n, other=0.0).to(tl.float32)
    v = x * inv
    v = tl.where(v != v, v, tl.minimum(tl.maximum(v, -448.0), 448.0))  # NaN stays NaN
    tl.store(out_ptr + off, v.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=off < n)
    if pid == 0:
        tl.store(scale_ptr, s * scale_mult)


@torch.library.custom_op("trtllm::e4m3_quantize_pow2_dynamic", mutates_args=("out", "scale_out"))
def e4m3_quantize_pow2_dynamic(
    x: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    scale_mult: float,
) -> None:
    """Quantize ``x`` to E4M3 into ``out`` with one power-of-two scale ``s`` derived from
    ``max|x|`` on the device, and write ``s * scale_mult`` to ``scale_out[0]`` (fp32)."""
    if not x.is_cuda or x.dtype not in _IN_DTYPES or not x.is_contiguous():
        raise ValueError(
            f"x must be a contiguous bf16/fp16/fp32 CUDA tensor, got {x.dtype} on {x.device}"
        )
    if out.device != x.device or out.dtype != torch.float8_e4m3fn or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous float8_e4m3fn tensor on {x.device}")
    if out.numel() != x.numel():
        raise ValueError(f"out has {out.numel()} elements, x has {x.numel()}")
    if scale_out.device != x.device or scale_out.dtype != torch.float32 or scale_out.numel() < 1:
        raise ValueError(
            f"scale_out must be a float32 CUDA tensor with >= 1 element, got {scale_out.dtype}"
        )
    n = x.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    if n:
        grid = (triton.cdiv(n, _BLOCK),)
        _amax_kernel[grid](x, amax, n, BLOCK=_BLOCK, num_warps=8)
        _quant_kernel[grid](
            x,
            out.view(torch.uint8),
            amax,
            scale_out,
            n,
            float(scale_mult),
            _AMAX_FLOOR,
            BLOCK=_BLOCK,
            num_warps=8,
        )
    else:
        _quant_kernel[(1,)](
            x,
            out.view(torch.uint8),
            amax,
            scale_out,
            0,
            float(scale_mult),
            _AMAX_FLOOR,
            BLOCK=_BLOCK,
            num_warps=8,
        )


@e4m3_quantize_pow2_dynamic.register_fake
def _e4m3_quantize_pow2_dynamic_fake(
    x: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    scale_mult: float,
) -> None:
    return None
