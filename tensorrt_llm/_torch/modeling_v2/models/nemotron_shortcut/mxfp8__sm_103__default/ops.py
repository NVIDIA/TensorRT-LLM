# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-scaled E4M3 projections with dynamic MXFP8 activations."""

from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import mxfp8_quantize


@dataclass
class Mxfp8Projection:
    """E4M3 weights and UE8M0 scales in the native 128x4 swizzle."""

    weight: torch.Tensor
    block_scales: torch.Tensor
    out_features: int

    def apply(self, x):
        if x.shape[0] == 0:
            return x.new_empty((0, self.out_features))
        act, act_scales = mxfp8_quantize(x.contiguous(), swizzled_layout=True, alignment=128)
        out = torch.ops.trtllm.flashinfer_mm_mxfp8(
            act, act_scales, self.weight, self.block_scales, x.dtype
        )
        return out[:, : self.out_features]

    def apply_experts(self, x, counts, start, end):
        from flashinfer.gemm import grouped_gemm_nt_masked
        from flashinfer.quantization.fp8_quantization import mxfp8_grouped_quantize

        lhs = mxfp8_grouped_quantize(x.contiguous(), counts, backend="cuda")
        weights = self.weight[start:end]
        experts, padded_n, padded_k = weights.shape
        scales = self.block_scales[start:end].view(
            experts, padded_n // 128, padded_k // 128, 32, 4, 4
        )
        rhs = weights.permute(1, 2, 0), scales.permute(3, 4, 1, 5, 2, 0)
        out = x.new_empty((experts, x.shape[1], padded_n))
        grouped_gemm_nt_masked(
            lhs,
            rhs,
            out.permute(1, 2, 0),
            counts,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
        )
        return out[:, :, : self.out_features]
