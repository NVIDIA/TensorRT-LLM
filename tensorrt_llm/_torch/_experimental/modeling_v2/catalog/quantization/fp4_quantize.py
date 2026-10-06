# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-scaled FP4 activation quantization: packed e2m1 data + one scale byte per
sf_vec_size contiguous elements (NVFP4: 16-element blocks, e4m3 scales)."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def fp4_quantize(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    sf_vec_size: int,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to FP4: returns (packed e2m1 data [..., K/2] uint8, block scales uint8, 1-D).

    One scale per `sf_vec_size` contiguous elements along the last dim.
    NVFP4 is `sf_vec_size=16, sf_use_ue8m0=False` (e4m3 scale bytes, scaled by
    `global_scale`); `is_sf_swizzled_layout` selects the 128x4 swizzled scale
    order (True) or the row-major linear order (False).
    """
    # Pure-metadata guard: the kernel loads one scalar from global_scale and
    # ignores every element past the first, so a per-token [num_tokens] tensor
    # is silently applied as global_scale[0] to every row -- observed on this
    # machine to return a valid-looking result, never to raise.
    assert global_scale is None or global_scale.numel() == 1, (
        "global_scale must hold exactly one element; extra elements are "
        "silently ignored and global_scale[0] is applied to every row"
    )
    return torch.ops.trtllm.fp4_quantize(
        input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
    )
