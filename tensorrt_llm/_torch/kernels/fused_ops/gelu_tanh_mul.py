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
"""Fused ``gelu_tanh(gate) * up`` over separate, row-strided bf16 operands (one Triton kernel).

Equivalent to ``F.gelu(gate, approximate="tanh") * up`` for separate (not packed) operands.
Numerics match the unfused pair: GELU in fp32 rounded to bf16, then the product in fp32 rounded
to bf16.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorrt_llm._utils import get_sm_version

from ...flashinfer_utils import get_env_enable_pdl
from ...modules.fla.op import gdc_wait_with_memory_clobber


@triton.jit
def _gelu_tanh_mul_kernel(
    gate_ptr,  # [M, I] bf16, row stride SG
    up_ptr,  # [M, I] bf16, row stride SU
    out_ptr,  # [M, I] bf16, contiguous
    M,
    SG,
    SU,
    IDIM: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    PDL: tl.constexpr,
):
    if PDL:
        gdc_wait_with_memory_clobber()
    rows = tl.program_id(0) * BM + tl.arange(0, BM).to(tl.int64)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    mask = (rows < M)[:, None] & (cols < IDIM)[None, :]
    x = tl.load(gate_ptr + rows[:, None] * SG + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + rows[:, None] * SU + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    inner = 0.7978845608028654 * (x + 0.044715 * (x * x * x))
    g = (0.5 * x * (1.0 + libdevice.tanh(inner))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + rows[:, None] * IDIM + cols[None, :], (g * up).to(tl.bfloat16), mask=mask)


def gelu_tanh_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``bf16(bf16(gelu_tanh(gate)) * up)``.

    Args:
        gate: [M, I] bf16; may have a row stride larger than its width.
        up: [M, I] bf16; may have a row stride larger than its width.

    Returns:
        [M, I] bf16, contiguous.
    """
    assert gate.dim() == 2 and gate.shape == up.shape
    assert gate.dtype == up.dtype == torch.bfloat16
    assert gate.stride(-1) == 1 and up.stride(-1) == 1
    m, idim = gate.shape
    out = torch.empty((m, idim), dtype=torch.bfloat16, device=gate.device)
    if m == 0:
        return out
    # Fixed tiles (no runtime autotune) so launches stay deterministic under CUDA-graph capture.
    bm, bk = 4, 256
    grid = (triton.cdiv(m, bm), triton.cdiv(idim, bk))
    pdl = get_env_enable_pdl() and get_sm_version() >= 90
    _gelu_tanh_mul_kernel[grid](
        gate,
        up,
        out,
        m,
        gate.stride(0),
        up.stride(0),
        IDIM=idim,
        BM=bm,
        BK=bk,
        PDL=pdl,
        num_warps=4,
        launch_pdl=pdl,
    )
    return out
