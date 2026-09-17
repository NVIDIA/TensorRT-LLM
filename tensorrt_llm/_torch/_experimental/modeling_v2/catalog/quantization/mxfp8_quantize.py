# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic MXFP8 quantization: bf16/fp16 -> e4m3 data + per-32-element UE8M0 block scales."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def mxfp8_quantize(
    input: torch.Tensor,
    swizzled_layout: bool = True,
    alignment: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP8: returns (e4m3 data [..., pad_up(K, alignment)], uint8 UE8M0 scales, 1D).

    One scale per 32 contiguous elements along the last dim; `swizzled_layout`
    selects the 128x4 swizzled scale order (True) or the row-major linear
    order (False).
    """
    return torch.ops.trtllm.mxfp8_quantize(input, swizzled_layout, alignment)
