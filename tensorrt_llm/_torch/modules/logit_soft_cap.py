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
"""A logit multiplier and tanh soft cap in one pass, bitwise the torch chain's FP32 values."""

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_FLT_MIN = tl.constexpr(1.1754943508222875e-38)
_BLOCK = 4096


@triton.jit
def _soft_cap_kernel(
    logits_ptr,
    out_ptr,
    n_cols,
    multiplier,
    inv_soft_cap,
    soft_cap,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(logits_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
    # The torch chain's operations: three separately rounded FP32 products (plain
    # multiplies keep subnormals) and an accurate tanhf. Triton's libdevice tanhf
    # flushes subnormals; tanhf of a subnormal is the input itself.
    t = (x * multiplier) * inv_soft_cap
    t = tl.where(tl.abs(t) < _FLT_MIN, t, libdevice.tanh(t))
    tl.store(out_ptr + row * n_cols + cols, soft_cap * t, mask=mask)


@functools.lru_cache(maxsize=None)
def soft_cap_constants(multiplier: float, soft_cap: float) -> Tuple[float, float, float]:
    """The FP32 scalars ATen applies for ``x * multiplier``, ``x / soft_cap`` and ``soft_cap * t``.

    A Python scalar operand is converted to FP32, and division by a scalar
    multiplies by its FP32 reciprocal computed on the host.
    """
    m = torch.tensor(multiplier, dtype=torch.float32)
    cap = torch.tensor(soft_cap, dtype=torch.float32)
    inv = torch.tensor(1.0, dtype=torch.float32) / cap
    return m.item(), inv.item(), cap.item()


def soft_capped_logits(
    logits: torch.Tensor,
    multiplier: float,
    soft_cap: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``soft_cap * tanh(logits.float() * multiplier / soft_cap)`` as FP32, bitwise.

    On CUDA one kernel reads ``logits`` once and writes the FP32 result once;
    the torch chain spends five full passes (the widening and four elementwise
    kernels) on the same values. ``out`` (FP32, ``logits``' shape, contiguous)
    receives the result and may be ``logits`` itself; otherwise a new tensor is
    returned.
    """
    if out is not None and (
        out.dtype != torch.float32 or out.shape != logits.shape or not out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous FP32 tensor of the logits' shape")
    if not logits.is_cuda or logits.dim() == 0 or not logits.is_contiguous():
        result = soft_cap * torch.tanh(logits.float() * multiplier / soft_cap)
        return result if out is None else out.copy_(result)
    if out is None:
        out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    n_cols = logits.shape[-1]
    n_rows = logits.numel() // n_cols if n_cols else 0
    if n_rows == 0:
        return out
    m, inv, cap = soft_cap_constants(multiplier, soft_cap)
    grid = (n_rows, triton.cdiv(n_cols, _BLOCK))
    _soft_cap_kernel[grid](logits, out, n_cols, m, inv, cap, BLOCK=_BLOCK, num_warps=8)
    return out
