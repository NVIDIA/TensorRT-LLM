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

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Below this many rows the two PyTorch kernels are kept by default (decode batches), unless
# enable_small_batches() ran or TRTLLM_K3_MLA_GATE_SMALL_BATCHES=1; callers pass min_tokens to take
# the kernel there regardless.
SIGMOID_GATE_MIN_TOKENS = 128

_SMALL_BATCHES_ENV = os.environ.get("TRTLLM_K3_MLA_GATE_SMALL_BATCHES")
if _SMALL_BATCHES_ENV not in (None, "0", "1"):
    raise ValueError(f"TRTLLM_K3_MLA_GATE_SMALL_BATCHES must be 0 or 1, got {_SMALL_BATCHES_ENV!r}")
_min_tokens = 1 if _SMALL_BATCHES_ENV == "1" else SIGMOID_GATE_MIN_TOKENS


def enable_small_batches() -> None:
    """Make one row the default ``min_tokens`` of :func:`sigmoid_gate_mul` in this process (called
    when a Kimi K3.1 model is built). The kernel gives the same bits as the two PyTorch kernels and
    is faster at decode sizes too; an explicit TRTLLM_K3_MLA_GATE_SMALL_BATCHES keeps its value."""
    global _min_tokens
    if _SMALL_BATCHES_ENV is None:
        _min_tokens = 1


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


@triton.jit
def _sigmoid_gate_mul_rows_kernel(x_ptr, g_ptr, out_ptr, cols, g_row_stride, BLOCK: tl.constexpr):
    # One row per program column-block: the gate is a column slice of a wider GEMM output.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < cols
    g = tl.load(g_ptr + row * g_row_stride + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * cols + offs, mask=mask, other=0.0).to(tl.float32)
    denom = 1.0 + libdevice.exp(-g)
    gate = tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;", "=f,f", [denom], dtype=tl.float32, is_pure=True, pack=1
    )
    gate = gate.to(tl.bfloat16).to(tl.float32)
    prod = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [x, gate], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(out_ptr + row * cols + offs, prod.to(tl.bfloat16), mask=mask)


def sigmoid_gate_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    min_tokens: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """``attn_output * gate.sigmoid()`` in one kernel, or ``None`` when the inputs are not two BF16 CUDA tensors of
    one 2D shape with at least ``min_tokens`` rows, a contiguous ``attn_output`` and a ``gate`` whose rows are
    contiguous (a contiguous tensor, or a column slice of a wider row-major tensor). ``min_tokens=None`` is
    ``SIGMOID_GATE_MIN_TOKENS``, or one row after :func:`enable_small_batches`."""
    if min_tokens is None:
        min_tokens = _min_tokens
    if (
        attn_output.dtype is not torch.bfloat16
        or gate.dtype is not torch.bfloat16
        or not attn_output.is_cuda
        or attn_output.shape != gate.shape
        or attn_output.dim() != 2
        or attn_output.shape[0] < max(min_tokens, 1)
        or not attn_output.is_contiguous()
    ):
        return None
    out = torch.empty_like(attn_output)
    if gate.is_contiguous():
        n = attn_output.numel()
        _sigmoid_gate_mul_kernel[(triton.cdiv(n, _BLOCK),)](
            attn_output, gate, out, n, BLOCK=_BLOCK, num_warps=4
        )
        return out
    rows, cols = attn_output.shape
    if gate.stride(1) != 1 or gate.stride(0) < cols:
        return None
    block = min(_BLOCK, triton.next_power_of_2(cols))
    _sigmoid_gate_mul_rows_kernel[(rows, triton.cdiv(cols, block))](
        attn_output, gate, out, cols, gate.stride(0), BLOCK=block, num_warps=4
    )
    return out
