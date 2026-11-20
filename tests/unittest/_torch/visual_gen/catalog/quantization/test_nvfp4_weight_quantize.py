# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.quantization.nvfp4_weight_quantize import quantize_nvfp4


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="NVFP4 weight quantization needs SM100+",
)
@pytest.mark.parametrize("cell", quantize_nvfp4.CELLS, ids=lambda cell: cell.why)
def test_nvfp4_weight_quantize(cell):
    spec = cell.spec
    rows, cols = spec["shape"]
    levels = torch.tensor([0, 0.5, -1, 1.5, -2, 3, -4, 6], device="cuda")
    block_ids = torch.arange(rows * cols // 16, device="cuda").reshape(rows, cols // 16)
    weight = levels.repeat(rows, cols // 8) / (2 ** (block_ids % 4)).repeat_interleave(16, dim=1)
    weight[:, 16:32] = 0
    if spec["dtype"] == torch.float32:
        weight[0, 0] = 6.01
    weight = weight.to(spec["dtype"])
    if spec.get("zero"):
        weight.zero_()

    packed, scales, global_scale = quantize_nvfp4(weight)
    ref_packed, ref_scales, ref_global = quantize_nvfp4.reference(weight)
    assert packed.dtype == ref_packed.dtype == torch.uint8
    assert torch.equal(packed, ref_packed)
    assert scales.dtype == ref_scales.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(scales.float(), ref_scales.float(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(global_scale, ref_global, rtol=0, atol=0)
