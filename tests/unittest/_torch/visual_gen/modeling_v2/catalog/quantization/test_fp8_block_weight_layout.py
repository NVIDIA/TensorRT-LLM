# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization.fp8_block_weight_layout import (
    fp8_block_weight_layout,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires Blackwell SM100 or SM103",
)


@pytest.mark.parametrize("cell", fp8_block_weight_layout.CELLS)
def test_fp8_block_weight_layout(cell):
    n, k = cell.spec["n"], cell.spec["k"]
    torch.manual_seed(41)
    weight = torch.randint(-8, 9, (n, k), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand((n // 128, k // 128), device="cuda")
    scales[0, 0] = 0
    ref_w, ref_scales = weight.clone(), scales.clone()
    fp8_block_weight_layout.is_valid(weight, scales)
    actual, packed = fp8_block_weight_layout(weight, scales)
    expected, ref_packed = fp8_block_weight_layout.reference(ref_w, ref_scales)
    assert actual.data_ptr() == weight.data_ptr()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales, ref_scales)
    assert torch.equal(packed, ref_packed)
    assert packed.stride() == ref_packed.stride()


def test_fp8_block_weight_layout_rejects_mixed_devices():
    weight = torch.empty((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((1, 1), device="cpu")
    with pytest.raises(ValueError, match="same CUDA device"):
        fp8_block_weight_layout(weight, scales)
