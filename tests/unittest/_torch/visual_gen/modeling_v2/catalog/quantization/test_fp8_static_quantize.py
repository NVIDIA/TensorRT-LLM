# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization.fp8_static_quantize import (
    fp8_static_quantize,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires Blackwell SM100 or SM103",
)


@pytest.mark.parametrize("cell", fp8_static_quantize.CELLS)
def test_fp8_static_quantize(cell):
    torch.manual_seed(41)
    x = torch.randn(cell.spec["shape"], device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor(0.01, device="cuda")
    fp8_static_quantize.is_valid(x, scale)
    assert torch.equal(
        fp8_static_quantize(x, scale).view(torch.uint8),
        fp8_static_quantize.reference(x, scale).view(torch.uint8),
    )
