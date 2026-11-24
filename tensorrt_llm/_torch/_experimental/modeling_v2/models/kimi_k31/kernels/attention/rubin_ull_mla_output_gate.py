# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-kernel ``attn_output * gate.sigmoid()`` for the Kimi K3 MLA output gate.

The kernel reproduces the two PyTorch BF16 kernels bit for bit: the sigmoid is
``1 / (1 + exp(-g))`` in FP32 with the CUDA math library's ``expf`` and a
correctly rounded division, rounded to BF16, and the product is the FP32
product of the two BF16 values rounded to BF16 (round to nearest even). The
division and the product keep denormal results (no flush to zero), as the
PyTorch kernels do: ``g`` in about [-103, -87.3] has a denormal sigmoid.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Below this many rows the two PyTorch kernels are kept (decode batches).
SIGMOID_GATE_MIN_TOKENS = 128

_BLOCK = 2048


@triton.jit
def _sigmoid_gate_mul_kernel(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    denom = 1.0 + libdevice.exp(-g)
    # rcp.rn is the correctly rounded 1 / x, i.e. div.rn(1, x).
    gate = tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;", "=f,f", [denom], dtype=tl.float32, is_pure=True, pack=1
    )
    gate = gate.to(tl.bfloat16).to(tl.float32)
    prod = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [x, gate], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(out_ptr + offs, prod.to(tl.bfloat16), mask=mask)


def sigmoid_gate_mul(attn_output: torch.Tensor, gate: torch.Tensor) -> Optional[torch.Tensor]:
    """``attn_output * gate.sigmoid()`` in one kernel, or ``None`` when the inputs are not two contiguous BF16 CUDA
    tensors of one shape with at least ``SIGMOID_GATE_MIN_TOKENS`` rows."""
    if (
        attn_output.dtype is not torch.bfloat16
        or gate.dtype is not torch.bfloat16
        or not attn_output.is_cuda
        or attn_output.shape != gate.shape
        or attn_output.dim() != 2
        or attn_output.shape[0] < SIGMOID_GATE_MIN_TOKENS
        or not attn_output.is_contiguous()
        or not gate.is_contiguous()
    ):
        return None
    out = torch.empty_like(attn_output)
    n = attn_output.numel()
    _sigmoid_gate_mul_kernel[(triton.cdiv(n, _BLOCK),)](
        attn_output, gate, out, n, BLOCK=_BLOCK, num_warps=4
    )
    return out
