# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft layer's MoE combination and hyper-connection injection in one Triton launch.

Bitwise the composed `fused_sigmoid_gate_mul_add` + `qwen4_exp_post_inject` path: one fp32
`fma.rn` rounded once to bf16 for the combination, and an exact bf16 x bf16 product for the
injection, rounded once on the store as `addcmul` does.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Columns a program forms.
BLOCK = 512


# The gate logit is one scalar a row: its row stride and address carry no vector width.
@triton.jit(do_not_specialize=["s_gate"], do_not_specialize_on_alignment=["gate"])
def _combine_inject_kernel(
    x,
    routed,
    shared,
    gate,
    inject,
    out,
    s_x,
    s_routed,
    s_shared,
    s_gate,
    s_inject,
    s_out,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_ + tl.arange(0, BLOCK_)
    mask = cols < H
    s = tl.sigmoid(tl.load(gate + row * s_gate).to(tl.float32))
    f = tl.load(routed + row * s_routed + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(shared + row * s_shared + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.fma(s, sh, f).to(routed.dtype.element_ty).to(tl.float32)
    for c in tl.static_range(HC):
        w = tl.load(inject + row * s_inject + c).to(tl.float32)
        xv = tl.load(x + row * s_x + c * H + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + row * s_out + c * H + cols, (xv + b * w).to(out.dtype.element_ty), mask=mask)


def _ckpt(rows: int, gate: str = "strided") -> dict:
    return dict(rows=rows, hc_count=4, hidden=2560, gate=gate)


class MtpMoeCombineInject(OpWrapper):
    """`out = mtp_moe_combine_inject(x, routed, gate_logits, shared, inject)`:
    `out[t, s * H + i] = x[t, s * H + i] + bf16(routed[t, i] + sigmoid(gate_logits[t]) *
    shared[t, i]) * inject[t, s]`, a fresh `[T, hc_count * H]` tensor.

    - `x` `[T, hc_count * H]` bf16, unit column stride (the pre-norm residual, stream-major);
    - `routed`, `shared` `[T, H]` bf16, unit column stride;
    - `gate_logits` `[T, 1]` / `[T]` (any row stride: the column the shared expert's fused GEMM
      leaves); `inject` `[T, hc_count]` bf16, unit column stride.

    No input is written.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="tep4 draft step 1: 1 row", spec=_ckpt(1)),
        Cell(why="batch-4 draft step 0: 12 rows", spec=_ckpt(12)),
        Cell(why="contiguous gate vector, 4 rows", spec=_ckpt(4, "contiguous")),
    )

    note = """
    Source: one Triton kernel (`_combine_inject_kernel`), grid `(T, ceil(H / 512))`, JIT-compiled
    per (H, hc_count, dtypes); the engine's warmup forward compiles it before capture.
    """

    def __call__(self, x, routed, gate_logits, shared, inject) -> torch.Tensor:
        rows, width = x.shape
        hidden = routed.shape[1]
        hc = inject.shape[1]
        out = torch.empty([rows, width], dtype=x.dtype, device=x.device)
        if rows:
            gate = gate_logits.reshape(rows, -1) if gate_logits.dim() == 1 else gate_logits
            _combine_inject_kernel[(rows, triton.cdiv(hidden, BLOCK))](
                x,
                routed,
                shared,
                gate,
                inject,
                out,
                x.stride(0),
                routed.stride(0),
                shared.stride(0),
                gate.stride(0),
                inject.stride(0),
                out.stride(0),
                H=hidden,
                HC=hc,
                BLOCK_=BLOCK,
                num_warps=4,
            )
        return out

    def reference(self, x, routed, gate_logits, shared, inject) -> torch.Tensor:
        """fp32 combination rounded once to bf16, then the fp32 injection rounded once."""
        rows = x.shape[0]
        g = torch.sigmoid(gate_logits.reshape(rows, 1).float())
        b = (routed.float() + g * shared.float()).to(routed.dtype).float()
        streams = x.float().unflatten(-1, (inject.shape[1], routed.shape[1]))
        return (streams + b.unsqueeze(-2) * inject.float().unsqueeze(-1)).flatten(-2).to(x.dtype)

    def is_valid(self, x, routed, gate_logits, shared, inject) -> None:
        rows = x.shape[0]
        assert x.dim() == 2 and routed.dim() == 2 and shared.dim() == 2 and inject.dim() == 2
        assert x.dtype == routed.dtype == shared.dtype == inject.dtype == torch.bfloat16
        hidden, hc = routed.shape[1], inject.shape[1]
        assert x.shape[1] == hc * hidden, (tuple(x.shape), hc, hidden)
        assert routed.shape[0] == shared.shape[0] == inject.shape[0] == rows
        assert tuple(shared.shape) == (rows, hidden)
        for t in (x, routed, shared, inject):
            assert t.stride(1) == 1 or t.shape[1] == 1, (tuple(t.shape), t.stride())
        assert gate_logits.numel() == rows and (gate_logits.dim() == 1 or gate_logits.shape[1] == 1)
        assert x.device == routed.device == shared.device == inject.device == gate_logits.device


mtp_moe_combine_inject = MtpMoeCombineInject()
