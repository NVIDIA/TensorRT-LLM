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
"""One kernel for the MLA V up-projection and the Kimi K3 sigmoid output gate at a few rows.

``vup_gate(attn, v_b_proj, gate)`` is ``bmm(attn_h, v_b_proj[h]^T) * sigmoid(gate)`` per head
``h`` for ``attn`` ``[rows, heads * kv_lora_rank]``: one program per (head, block of value
columns) multiplies every row of the head (FP32 accumulation on the tensor cores), rounds the
projection to BF16 and applies the gate with the arithmetic of ``output_gate.sigmoid_gate_mul``
(the sigmoid rounded to BF16, the product the FP32 product of the two BF16 values rounded to
BF16). The weights do not depend on earlier kernels, so a program reads its block of them before
it waits for the attention output (programmatic dependent launch).
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

VUP_GATE_MAX_ROWS = 32


@triton.jit
def _vup_gate_kernel(
    x_ptr,  # bf16 [rows, x_row_stride]: per-head latent outputs, head-major columns
    w_ptr,  # bf16 [heads, DV, DL]
    g_ptr,  # bf16 [rows, g_row_stride]: gate logits, head-major columns
    out_ptr,  # bf16 [rows, out_row_stride]
    rows,
    x_row_stride,
    g_row_stride,
    out_row_stride,
    RP: tl.constexpr,
    DL: tl.constexpr,
    DV: tl.constexpr,
    BV: tl.constexpr,
    HAS_GATE: tl.constexpr,
    PDL: tl.constexpr,
):
    hh = tl.program_id(0)
    vb = tl.program_id(1)
    r = tl.arange(0, RP)
    rmask = r < rows
    dl = tl.arange(0, DL)
    v = vb * BV + tl.arange(0, BV)
    wt = tl.load(w_ptr + (hh * DV + v[:, None]).to(tl.int64) * DL + dl[None, :])  # [BV, DL]
    col = hh * DV + v
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_GATE:
        g = tl.load(
            g_ptr + r[:, None].to(tl.int64) * g_row_stride + col[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
    x = tl.load(
        x_ptr + r[:, None].to(tl.int64) * x_row_stride + hh * DL + dl[None, :],
        mask=rmask[:, None],
        other=0.0,
    )
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    y = tl.dot(x, tl.trans(wt)).to(tl.bfloat16).to(tl.float32)  # [RP, BV]
    if HAS_GATE:
        den = 1.0 + libdevice.exp(-g)
        # rcp.rn is the correctly rounded 1 / x, i.e. div.rn(1, x).
        gate = tl.inline_asm_elementwise(
            "rcp.rn.f32 $0, $1;", "=f,f", [den], dtype=tl.float32, is_pure=True, pack=1
        )
        gate = gate.to(tl.bfloat16).to(tl.float32)
        y = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [y, gate], dtype=tl.float32, is_pure=True, pack=1
        )
    tl.store(
        out_ptr + r[:, None].to(tl.int64) * out_row_stride + col[None, :],
        y.to(tl.bfloat16),
        mask=rmask[:, None],
    )


def vup_gate(
    attn: torch.Tensor,
    v_b_proj: torch.Tensor,
    gate: Optional[torch.Tensor],
    out: Optional[torch.Tensor] = None,
    block_v: int = 32,
    num_warps: int = 4,
    pdl: bool = True,
) -> torch.Tensor:
    """``attn`` ``[rows, heads * kv_lora_rank]`` (BF16, rows of contiguous columns) through
    ``v_b_proj`` ``[heads, v_dim, kv_lora_rank]`` and, with ``gate`` (``[rows, heads * v_dim]``,
    contiguous columns), the sigmoid gate: ``[rows, heads * v_dim]`` BF16, into ``out`` if given."""
    rows = attn.shape[0]
    heads, v_dim, dl = v_b_proj.shape
    assert 0 < rows <= VUP_GATE_MAX_ROWS
    assert dl == triton.next_power_of_2(dl) and v_dim % min(block_v, v_dim) == 0
    assert attn.dtype == torch.bfloat16 and v_b_proj.dtype == torch.bfloat16
    assert attn.dim() == 2 and attn.shape[1] == heads * dl and attn.stride(1) == 1
    assert v_b_proj.is_contiguous()
    if out is None:
        out = torch.empty((rows, heads * v_dim), dtype=torch.bfloat16, device=attn.device)
    assert out.shape == (rows, heads * v_dim) and out.stride(1) == 1
    if gate is not None:
        assert gate.shape == (rows, heads * v_dim) and gate.stride(1) == 1
        assert gate.dtype == torch.bfloat16
    block_v = min(block_v, v_dim)
    _vup_gate_kernel[(heads, v_dim // block_v)](
        attn,
        v_b_proj,
        gate if gate is not None else attn,
        out,
        rows,
        attn.stride(0),
        gate.stride(0) if gate is not None else 0,
        out.stride(0),
        RP=max(16, triton.next_power_of_2(rows)),
        DL=dl,
        DV=v_dim,
        BV=block_v,
        HAS_GATE=gate is not None,
        PDL=pdl,
        num_warps=num_warps,
        launch_pdl=pdl,
    )
    return out
