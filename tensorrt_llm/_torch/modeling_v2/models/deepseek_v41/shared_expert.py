# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared expert of the deepseek_v41 targets, as the built-in model runs it: the checkpoint's block-32 E4M3
weights requantized to 128x128 FP8 tiles (stock's requantization) on stock's SM100 DeepGEMM FP8 block-scale kernels,
and its loader."""

import torch
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import (
    deep_gemm_weight_scale,
    requantize_block128,
    shared_expert_fp8_deep_gemm,
    shared_expert_fp8_deep_gemm_prequant,
)

from .loaders import dequant_block32


class SharedExpertFp8(nn.Module):
    """One SwiGLU FFN every token goes through: stock's FP8 block-scale path (the checkpoint's E4M3 requantized to
    128x128 tiles) on stock's SM100 DeepGEMM kernels (UE8M0 activation scales, SwiGLU fused with the second
    quantization), and on few rows stock's small-M GEMMs where stock's linear picks them."""

    def __init__(self, dim: int, inter: int, limit: float):
        super().__init__()
        self.limit = limit
        f8, f32, b = torch.float8_e4m3fn, torch.float32, 128
        self.w13 = nn.Parameter(torch.empty(2 * inter, dim, dtype=f8), requires_grad=False)
        self.s13 = nn.Parameter(
            torch.empty(2 * inter // b, dim // b, dtype=f32), requires_grad=False
        )
        self.w2 = nn.Parameter(torch.empty(dim, inter, dtype=f8), requires_grad=False)
        self.s2 = nn.Parameter(torch.empty(dim // b, inter // b, dtype=f32), requires_grad=False)
        self.sf13 = self.sf2 = None

    def _deep_gemm_scales(self):
        if self.sf13 is None:  # DeepGEMM's scale layout, built on first use (an eager step)
            self.sf13 = deep_gemm_weight_scale(self.s13, *self.w13.shape)
            self.sf2 = deep_gemm_weight_scale(self.s2, *self.w2.shape)
        return self.sf13, self.sf2

    def forward_fp8(self, a: torch.Tensor, a_sf: torch.Tensor) -> torch.Tensor:
        """From the input its producer quantized to FP8 1x128 (`nvfp4_fp8_quantize`)."""
        sf13, sf2 = self._deep_gemm_scales()
        return shared_expert_fp8_deep_gemm_prequant(
            a, a_sf, self.w13, sf13, self.w2, sf2, self.limit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sf13, sf2 = self._deep_gemm_scales()
        return shared_expert_fp8_deep_gemm(
            x, self.w13, sf13, self.s13, self.w2, sf2, self.s2, self.limit
        )


def load_shared_expert_fp8(shared: SharedExpertFp8, weights: dict, prefix: str, device) -> None:
    """Fill `shared` from `<prefix>shared_experts.{w1,w2,w3}.{weight,scale}`: block-32 FP8 dequantized, then
    requantized to 128x128 FP8 tiles (`w1 | w3` stacked as the gate-up GEMM), as stock's loader does."""
    sh = {
        w: dequant_block32(
            weights[prefix + f"shared_experts.{w}.weight"].to(device),
            weights[prefix + f"shared_experts.{w}.scale"].to(device),
        )
        for w in ("w1", "w2", "w3")
    }
    for (wq, sq), src in (
        (("w13", "s13"), torch.cat([sh["w1"], sh["w3"]], 0)),
        (("w2", "s2"), sh["w2"]),
    ):
        q, sc = requantize_block128(src)
        getattr(shared, wq).data.copy_(q)
        getattr(shared, sq).data.copy_(sc)
