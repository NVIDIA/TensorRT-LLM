# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 per-tensor projection: static E4M3 quantize, then cuBLASLt's scaled FP8 GEMM.

``static_quantize_e4m3_per_tensor(x, a)`` followed by ``cublas_scaled_mm(q, w8.T,
a, s)`` -- the checkpoint's own FP8 mode. This target runs it on one slice only:
the K and V rows (``[12288:)``) of the attention qkv_proj above its decode cap, so
the keys and values written to the KV cache keep the checkpoint's FP8 precision
while the query and output-gate columns come from the NVFP4 W4A4 copy (modeling.py
writes this result over those columns of the W4A4 output).
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

_E4M3_MAX = 448.0


def _cell(why: str, m: int) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=2048, k=5120))


class Fp8PerTensorMm(OpWrapper):
    """``bf16((e4m3(x / a) @ w8.T) * a * s)`` for BF16 ``x`` ``[M, K]``, an FP8 weight
    ``w8`` ``[N, K]`` (a contiguous row slice of the stored weight is fine), and fp32
    one-element scales ``input_scale`` (``a``) and ``weight_scale`` (``s``). Returns a
    fresh ``[M, N]`` BF16 tensor."""

    ARCHS = frozenset({Arch.SM_120})

    # Attention qkv_proj K/V rows (2 x 4 kv heads x 256 = 2048 of 14336, hidden 5120) from
    # 65 rows up to max_num_tokens.
    CELLS: tuple[Cell, ...] = (
        _cell("qkv K/V rows M=65: first row above the qkv Marlin cap", 65),
        _cell("qkv K/V rows M=129", 129),
        _cell("qkv K/V rows M=2048: a prefill chunk", 2048),
        _cell("qkv K/V rows M=8192: max_num_tokens", 8192),
    )

    note = """
    Metadata consumed: none. `weight_fp8` is `qkv_proj.weight[12288:]` of the FP8
    checkpoint, `input_scale` / `weight_scale` that projection's static scales.

    Preconditions: `x` BF16 contiguous (the quantize op checks); `weight_fp8` E4M3 with
    unit inner stride and a row stride that is a multiple of 16 (cuBLASLt checks).

    Numerics: the quantize multiplies by `1 / a` in fp32 and rounds to E4M3 with
    saturation; cuBLASLt multiplies E4M3 x E4M3 exactly, accumulates in fp32, applies
    `a * s`, one rounding.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, input_scale)
        return torch.ops.trtllm.cublas_scaled_mm(
            q,
            weight_fp8.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=None,
            out_dtype=torch.bfloat16,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """``e4m3(x * (1 / a))`` (saturated), an fp32 product of the E4M3 values (TF32 off),
        times ``a * s``, one rounding."""
        a = input_scale.float().reshape(())
        q = (x.float() * (1.0 / a)).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        with true_fp32_matmul():
            acc = q.float() @ weight_fp8.float().t()
        return (acc * (a * weight_scale.float().reshape(()))).to(torch.bfloat16)

    def is_valid(
        self,
        x: torch.Tensor,
        weight_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        # The per-tensor quantize reads input_scale[0] only, while cuBLASLt would apply
        # a per-row scale_a of M elements: the two would disagree silently.
        assert input_scale.numel() == 1, "input_scale must be one element (per-tensor)"


fp8_per_tensor_mm = Fp8PerTensorMm()
