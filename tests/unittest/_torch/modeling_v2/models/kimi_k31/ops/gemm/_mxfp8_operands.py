# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 GEMM operands as the target builds them: dynamic activations from ``mxfp8_quantize`` and
checkpoint-like weights (E4M3 codes, UE8M0 bytes swizzled by ``block_scale_interleave``)."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    mxfp8_quantize,
)


def mxfp8_operands(rows: int, n: int, k: int, seed: int):
    """``(activation, scales, codes, weight_scales)``, the activation from BF16 N(0, 1) rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    activation, scales = mxfp8_quantize(x, True)
    codes = (torch.randn(n, k, generator=g, device="cuda") * 64).to(torch.float8_e4m3fn)
    raw = torch.randint(118, 124, (n, k // 32), generator=g, device="cuda", dtype=torch.uint8)
    weight_scales = torch.ops.trtllm.block_scale_interleave(raw).flatten()
    return activation, scales, codes, weight_scales
