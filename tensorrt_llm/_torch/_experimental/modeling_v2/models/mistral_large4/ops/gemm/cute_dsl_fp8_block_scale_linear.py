# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 FP8 linear with K128 quantization and replicated K32 UE8M0 scales."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.gemm.fp8_prequantized_swap_ab_gemm import (  # noqa: E501
    PROJECTIONS,
    _per_element_scale,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.ops.quantization.fp8_quantize_1x128_packed_ue8m0 import (  # noqa: E501
    fp8_quantize_1x128_packed_ue8m0 as quantize,
)


def prepare_weight_scale(packed: torch.Tensor, k: int) -> torch.Tensor:
    """MN-major int32 K128 scales -> R128c4 uint8 K32 scales, without requantization."""
    rows = packed.shape[0]
    exponents = packed.reshape(-1).view(torch.uint8).reshape(rows, -1)[:, : k // 128]
    return torch.ops.trtllm.block_scale_interleave(exponents.repeat_interleave(4, dim=1))


def _weight_scales(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Unswizzle R128c4 into logical row/K32 scales using tensor indexing only."""
    padded_n = (n + 127) // 128 * 128
    columns = k // 32
    exponent = (
        scale.reshape(padded_n // 128, columns // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_n, columns)[:n]
    )
    return (exponent.to(torch.int32) << 23).contiguous().view(torch.float32)


class CuteDslFp8BlockScaleLinear(OpWrapper):
    """Quantize BF16 input per 128 values, then apply FP8 weights with K128 scales.

    Every K128 exponent is replicated over its four K32 slots. The Rubin MXFP8
    kernel therefore consumes the original FP8 values and scales exactly.
    Weight scales are prepared once when the model constructs its layer views.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(f"SM107 {name}, M={m}", dict(m=m, n=n, k=k))
        for name, n, k in PROJECTIONS
        for m in (1, 4, 16, 64, 129, 2048, 32768)
    )

    def __call__(
        self, input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        codes, scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
            input, use_r128c4_layout=True
        )
        return torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(codes, weight, scale, weight_scale)

    def reference(
        self, input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        codes, scale = quantize.reference(input)
        m, k = input.shape
        n = weight.shape[0]
        activation = codes.float() * _per_element_scale(scale, m, k)
        weights = weight.float() * _weight_scales(weight_scale, n, k).repeat_interleave(32, dim=1)
        with true_fp32_matmul():
            output = activation @ weights.t()
        return output.to(torch.bfloat16)


cute_dsl_fp8_block_scale_linear = CuteDslFp8BlockScaleLinear()
