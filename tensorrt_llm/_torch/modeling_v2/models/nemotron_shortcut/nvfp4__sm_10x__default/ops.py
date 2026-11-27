# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native FP8 projections and NVFP4 projections with per-expert calibration."""

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize

from ..moe import grouped_quantized_experts as grouped_experts

__all__ = ["Fp8Projection", "Nvfp4Projection", "grouped_experts"]


@dataclass
class Fp8Projection:
    weight: torch.Tensor
    input_scale: torch.Tensor
    weight_scale: torch.Tensor

    def apply(self, x):
        if x.shape[0] == 0:
            return x.new_empty((0, self.weight.shape[0]))
        packed, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x.contiguous(), self.input_scale
        )
        return torch.ops.trtllm.cublas_scaled_mm(
            packed,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=None,
            out_dtype=x.dtype,
            output_buffer_kind=0,
            group=None,
        )


@dataclass
class Nvfp4Projection:
    weight: torch.Tensor
    block_scales: torch.Tensor
    inverse_input_scale: torch.Tensor
    alpha: torch.Tensor
    out_features: int
    # Online placement replaces raw calibration in the same slots as the matrices.
    live_scales: tuple[torch.Tensor, torch.Tensor] | None = None

    def apply(self, x):
        if x.shape[0] == 0:
            return x.new_empty((0, self.out_features))
        packed, scales = fp4_quantize(x.contiguous(), self.inverse_input_scale, 16)
        out = nvfp4_gemm(packed, self.weight, scales, self.block_scales, self.alpha, x.dtype)
        return out[:, : self.out_features]

    def apply_experts(self, x, counts, start, end):
        from flashinfer.gemm import grouped_gemm_nt_masked
        from flashinfer.quantization.fp4_quantization import scaled_fp4_grouped_quantize

        if self.live_scales is None:
            inverse = self.inverse_input_scale[start:end]
            alpha = self.alpha[start:end]
        else:
            input_scale, weight_scale = self.live_scales
            inverse = input_scale[start:end].reciprocal()
            alpha = input_scale[start:end] * weight_scale[start:end]
        lhs = scaled_fp4_grouped_quantize(x, counts, inverse, nvfp4_4over6=None)
        w = self.weight[start:end]
        experts, padded_n, half_k = w.shape
        # The physical layout of these scales is already the 128x4 GPU swizzle.
        scales = (
            self.block_scales[start:end]
            .view(torch.float8_e4m3fn)
            .view(experts, padded_n // 128, half_k // 32, 32, 4, 4)
        )
        rhs = w.permute(1, 2, 0), scales.permute(3, 4, 1, 5, 2, 0)
        out = x.new_empty((experts, x.shape[1], padded_n))
        grouped_gemm_nt_masked(
            lhs,
            rhs,
            out.permute(1, 2, 0),
            counts,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            alpha=alpha,
            alpha_dtype="float32",
        )
        return out[:, :, : self.out_features]
