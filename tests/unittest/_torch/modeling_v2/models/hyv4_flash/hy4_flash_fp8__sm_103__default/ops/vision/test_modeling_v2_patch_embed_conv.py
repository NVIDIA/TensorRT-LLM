# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the patch_embed_conv entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.vision.patch_embed_conv import (  # noqa: E501
    patch_embed_conv as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "patch_embed_conv requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    c, k, n, dt = spec["in_channels"], spec["kernel"], spec["out_channels"], spec["dtype"]
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()))
    # Pixels in [-1, 1] (processor mean/std 0.5); weights at the tower's initializer_range 0.02.
    pixel_values = (
        torch.empty(spec["num_rows"], c * k * k, device="cuda")
        .uniform_(-1.0, 1.0, generator=g)
        .to(dt)
    )
    weight = (0.02 * torch.randn(n, c, k, k, generator=g, device="cuda")).to(dt)
    bias = (0.02 * torch.randn(n, generator=g, device="cuda")).to(dt)
    out = op(pixel_values, weight, bias)
    assert out.is_contiguous()
    # One fp32 accumulation and one rounding on both sides: torch's default band.
    torch.testing.assert_close(out, op.reference(pixel_values, weight, bias))
