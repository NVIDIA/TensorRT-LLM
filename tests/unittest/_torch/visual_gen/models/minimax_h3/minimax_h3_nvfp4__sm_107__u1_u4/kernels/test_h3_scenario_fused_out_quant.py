# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kernels/fused_out_quant.py``: the tiled NVFP4 quantize of the attention output.

Reference: ``trtllm::fp4_quantize(x, input_scale, 16, sf_use_ue8m0=False)``, the op the static-NVFP4
Linear runs on the same tensor. Codes and swizzled block scales are compared byte for byte.
"""

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401  -- loads libth_common.so (registers the trtllm ops)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
    fused_out_quant,
)

pytestmark = skip_pre_blackwell

WIDTH = 5120
# 1, 127 and 129 rows exercise the zeroed padding rows of the 128-row swizzled scale layout.
ROWS = (1, 127, 129, 1017, 9455)
# (rows per program, columns per program, warps): the scenario default and a second tiling.
TILES = ((16, 512, 4), (32, 256, 2))


def _activation(rows: int, width: int, seed: int) -> torch.Tensor:
    """BF16 ``[rows, width]`` with outlier blocks, an all-zero block and (when it fits) an all-zero row."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, width), generator=gen, device="cuda", dtype=torch.float32)
    x[0, :16] *= 64.0
    x[0, 16:32] = 0.0
    x[rows - 1, width - 16 :] *= 32.0
    if rows > 2:
        x[rows // 2] = 0.0
    return x.to(torch.bfloat16)


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    """A calibrated-style NVFP4 input scale, 0-dim fp32 like ``Linear.input_scale``."""
    amax = x.float().abs().max().clamp(min=1e-3)
    return (448.0 * 6.0 / amax).to(torch.float32)


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    assert actual.dtype == torch.uint8 and expected.dtype == torch.uint8, what
    assert actual.numel() == expected.numel(), (
        f"{what}: {actual.numel()} vs {expected.numel()} bytes"
    )
    assert torch.equal(actual.reshape(-1), expected.reshape(-1)), what


@pytest.mark.parametrize("tile", TILES, ids=lambda t: "x".join(str(v) for v in t))
@pytest.mark.parametrize("rows", ROWS)
def test_tiled_out_quant_equals_fp4_quantize(rows, tile):
    x = _activation(rows, WIDTH, seed=rows)
    gs = _global_scale(x)
    ref_codes, ref_scales = torch.ops.trtllm.fp4_quantize(x, gs, 16, False)
    rb, ch, warps = tile
    codes, scales = fused_out_quant._launch(x, gs, rb=rb, ch=ch, num_warps=warps)
    assert codes.shape == (rows, WIDTH // 2)
    assert scales.numel() == fused_out_quant.sf_numel(rows, WIDTH)
    _assert_same_bytes(codes, ref_codes, "packed e2m1 codes")
    _assert_same_bytes(scales, ref_scales, "swizzled e4m3 block scales")
