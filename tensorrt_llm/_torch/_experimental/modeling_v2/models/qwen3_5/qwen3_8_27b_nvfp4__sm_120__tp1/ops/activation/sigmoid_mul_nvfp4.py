# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention output gate fused with o_proj's NVFP4 input quantize.

Gated attention multiplies its output by ``sigmoid(gate)`` before o_proj. Where
o_proj runs its NVFP4 W4A4 copy (above 32 rows) one Triton kernel computes the
gated output rounded to BF16 and quantizes it with the copy's activation global
scale (``6 / input_scale``), bitwise ``fused_sigmoid_mul_inplace`` + ``fp4_quantize``,
without writing the BF16 product or modifying the attention output. The gate is
read through its strided view into the q/gate-interleaved qkv projection. Kernel
in ``_sigmoid_mul_nvfp4``.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ._sigmoid_mul_nvfp4 import sigmoid_mul_nvfp4 as _launch

_SF_VEC = 16


def _cell(why: str, m: int) -> Cell:
    return Cell(why=why, spec=dict(m=m, heads=24, head_dim=256, qkv_width=14336))


class SigmoidMulNvfp4(OpWrapper):
    """``fp4_quantize(bf16(attention_output * sigmoid(gate)), global_scale, 16)`` with
    swizzled scales: ``(fp4 [T, H/2] uint8, sf uint8 [ceil128(T) * ceil4(H/16)])``,
    padding rows of ``sf`` zero. ``attention_output`` BF16 ``[T, H]``; ``gate`` BF16
    ``[T, H]`` or a ``[T, heads, head_dim]`` view (``heads * head_dim == H``), unit inner
    stride; ``global_scale`` fp32 ``[1]``."""

    ARCHS = frozenset({Arch.SM_120})

    # Qwen3.8-27B gated attention (24 q heads x 256 = 6144) where o_proj runs its W4A4 copy:
    # 33 rows up to max_num_tokens. The gate is the strided view into qkv_proj's output
    # (14336 wide, q and gate interleaved per head).
    CELLS: tuple[Cell, ...] = (
        _cell("T=33: first W4A4 o_proj row", 33),
        _cell("T=64", 64),
        _cell("T=128: no SF padding rows", 128),
        _cell("T=129: 127 SF padding rows", 129),
        _cell("T=2048: a prefill chunk", 2048),
        _cell("T=8192: max_num_tokens", 8192),
    )

    note = """
    Metadata consumed: none. `global_scale` is o_proj's W4A4 copy's `act_global`.

    Preconditions: as above, plus H a multiple of 64 (whole 4-column scale tiles).

    Numerics: exact -- bitwise `fused_sigmoid_mul_inplace` (Triton `attention *
    sigmoid(gate)`, BF16) then `fp4_quantize(.., global_scale, 16)` with swizzled
    scales. Against `reference` (torch's sigmoid) only the BF16 product's last bit can
    differ, and with it a block's scale or a code.

    PDL: launched with PDL when TRTLLM_ENABLE_PDL is on; every access after
    `griddepcontrol.wait`, dependents released after the stores.
    """

    def __call__(
        self, attention_output: torch.Tensor, gate: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch(attention_output, gate, global_scale)

    def reference(
        self, attention_output: torch.Tensor, gate: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``bf16(attention * sigmoid(gate))`` in fp32 with torch's sigmoid, then
        `fp4_quantize`'s native-torch NVFP4 with swizzled scales."""
        t = attention_output.shape[0]
        g = gate.reshape(t, -1).float()
        r = (attention_output.float() * torch.sigmoid(g)).to(torch.bfloat16)
        return fp4_quantize.reference(r, global_scale, _SF_VEC, False, True)

    def is_valid(
        self, attention_output: torch.Tensor, gate: torch.Tensor, global_scale: torch.Tensor
    ) -> None:
        # The kernel addresses both operands by their strides with unit inner stride and
        # takes the gate's geometry from its shape; it checks none of it.
        assert attention_output.dim() == 2 and attention_output.stride(-1) == 1, (
            "attention_output must be [T, H] with unit inner stride"
        )
        assert attention_output.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16, (
            "BF16 operands only"
        )
        t, hidden = attention_output.shape
        assert hidden % 64 == 0, "H must be a multiple of 64 (whole scale tiles)"
        assert gate.stride(-1) == 1, "gate must have unit inner stride"
        if gate.dim() == 3:
            assert gate.shape[0] == t and gate.shape[1] * gate.shape[2] == hidden, (
                "gate view must be [T, heads, head_dim] with heads * head_dim == H"
            )
        else:
            assert gate.dim() == 2 and tuple(gate.shape) == (t, hidden), "gate must be [T, H]"
        assert global_scale.numel() == 1, "global_scale must hold exactly one element"


sigmoid_mul_nvfp4 = SigmoidMulNvfp4()
