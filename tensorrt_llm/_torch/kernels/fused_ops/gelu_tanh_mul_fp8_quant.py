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
"""Fused gelu_tanh+mul+static FP8 quantize (single Triton kernel).

Equivalent to ``flashinfer_gelu_tanh_and_mul`` followed by
``static_quantize_e4m3_per_tensor`` on a packed [gate | up] input, with
matching numerics (``tanh.approx.f32``, bf16 product, ``fp32(v) * (1 / scale)``
with saturation).
"""

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

from ...flashinfer_utils import get_env_enable_pdl
from ...modules.fla.op import gdc_wait_with_memory_clobber
from .rmsnorm_residual_add import _quant_fp8


@triton.jit
def _gelu_tanh_mul_fp8_kernel(
    x_ptr,  # [M, 2I] bf16 packed [gate | up], row stride SX
    o_ptr,  # [M, I] fp8, contiguous
    qs_ptr,  # [1] fp32 static input scale of the consumer
    M,
    SX,
    IDIM: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    PDL: tl.constexpr,
):
    # The quant scale is not written by the previous kernel: read it before the PDL wait.
    inv = tl.div_rn(1.0, tl.load(qs_ptr))
    if PDL:
        gdc_wait_with_memory_clobber()
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (rows < M)[:, None] & (cols < IDIM)[None, :]
    base = x_ptr + rows[:, None] * SX + cols[None, :]
    gate = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + IDIM, mask=mask, other=0.0).to(tl.float32)

    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [inner], dtype=tl.float32, is_pure=True, pack=1
    )
    v = (gate * (0.5 * (1.0 + t)) * up).to(tl.bfloat16).to(tl.float32)

    tl.store(o_ptr + rows[:, None] * IDIM + cols[None, :], _quant_fp8(v, inv), mask=mask)


def gelu_tanh_mul_fp8_quant(x: torch.Tensor, quant_scale: torch.Tensor) -> torch.Tensor:
    """Fused ``fp8(bf16(gelu_tanh(gate) * up) / quant_scale)``.

    Args:
        x: [M, 2I] bf16 packed [gate | up]; may have a row stride larger than
            its width.
        quant_scale: [1] fp32 static input scale of the consumer Linear.

    Returns:
        [M, I] float8_e4m3fn, contiguous.
    """
    assert x.dim() == 2 and x.stride(-1) == 1 and x.dtype == torch.bfloat16
    assert quant_scale.dtype == torch.float32 and quant_scale.numel() == 1
    m, two_i = x.shape
    assert two_i % 2 == 0
    idim = two_i // 2
    out = torch.empty((m, idim), dtype=torch.float8_e4m3fn, device=x.device)
    if m == 0:
        return out
    # Fixed tiles: one row and 1024 columns per program.
    bm, bk = 1, 1024
    grid = (triton.cdiv(m, bm), triton.cdiv(idim, bk))
    pdl = get_env_enable_pdl() and get_sm_version() >= 90
    _gelu_tanh_mul_fp8_kernel[grid](
        x,
        out,
        quant_scale,
        m,
        x.stride(0),
        IDIM=idim,
        BM=bm,
        BK=bk,
        PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )
    return out
