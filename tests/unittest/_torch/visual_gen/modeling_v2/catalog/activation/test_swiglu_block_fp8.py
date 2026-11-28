# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.activation.swiglu_block_fp8 import (
    swiglu_block_fp8,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell block-scaled FP8",
)
@pytest.mark.parametrize("cell", swiglu_block_fp8.CELLS, ids=lambda cell: cell.why)
def test_swiglu_block_fp8(cell):
    rows, width = cell.spec["rows"], cell.spec["width"]
    offset = cell.spec.get("offset", 0)
    generator = torch.Generator(device="cuda").manual_seed(17)
    storage = torch.randint(-8, 9, (offset + rows * 2 * width,), generator=generator, device="cuda")
    x = storage.bfloat16()[offset:].view(rows, 2 * width)
    data, scales = swiglu_block_fp8(x)
    ref_data, ref_scales = swiglu_block_fp8.reference(x)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0, check_stride=True)


def test_native_swiglu_block_fp8_meta():
    x = torch.empty((5, 1312), dtype=torch.bfloat16, device="meta")
    values, scales = swiglu_block_fp8(x)
    assert values.shape == (5, 656) and values.dtype == torch.float8_e4m3fn
    assert scales.shape == (5, 2) and scales.dtype == torch.int32
    assert scales.stride() == (1, 8)
