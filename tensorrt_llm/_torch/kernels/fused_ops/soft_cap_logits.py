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
"""One-pass FP32 logit epilogue: ``soft_cap * tanh(float(logits) * multiplier / soft_cap)``.

Bit-identical to the unfused PyTorch sequence (float cast, multiply, divide, tanh, multiply).
"""

from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _soft_cap_logits_kernel(
    x_ptr,  # [M, N] low precision, row stride SX
    o_ptr,  # [M, N] fp32, contiguous
    N,
    SX,
    multiplier,
    inv_soft_cap,
    soft_cap,
    SCALE: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * SX + cols, mask=mask, other=0.0).to(tl.float32)
    if SCALE:
        x = x * multiplier
    if CAP:
        x = libdevice.tanh(x * inv_soft_cap) * soft_cap
    tl.store(o_ptr + row * N + cols, x, mask=mask)


def _fp32_reciprocal(value: float) -> float:
    # PyTorch divides a float tensor by a scalar as a multiply by the fp32 reciprocal.
    return float(np.float32(1.0) / np.float32(value))


def soft_cap_logits(
    logits: torch.Tensor, multiplier: float, soft_cap: Optional[float]
) -> torch.Tensor:
    """FP32 ``soft_cap * tanh(logits * multiplier / soft_cap)`` of a 2-D CUDA tensor, in one pass
    (a ``soft_cap`` of None skips the cap). Returns a new contiguous FP32 tensor."""
    assert logits.dim() == 2 and logits.is_cuda and logits.stride(-1) == 1
    m, n = logits.shape
    out = torch.empty((m, n), dtype=torch.float32, device=logits.device)
    if m == 0 or n == 0:
        return out
    block = 2048
    _soft_cap_logits_kernel[(m, triton.cdiv(n, block))](
        logits,
        out,
        n,
        logits.stride(0),
        float(multiplier),
        _fp32_reciprocal(soft_cap) if soft_cap is not None else 1.0,
        float(soft_cap) if soft_cap is not None else 1.0,
        SCALE=multiplier != 1.0,
        CAP=soft_cap is not None,
        BLOCK=block,
        num_warps=8,
    )
    return out
