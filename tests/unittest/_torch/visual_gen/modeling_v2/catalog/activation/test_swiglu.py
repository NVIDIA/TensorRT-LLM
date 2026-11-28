# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.activation.swiglu import swiglu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cell", swiglu.CELLS, ids=lambda cell: cell.why)
def test_swiglu(cell):
    spec = cell.spec
    generator = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn(
        (spec["rows"], 2 * spec["width"]), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    scale = x.new_tensor(0.002, dtype=torch.float32) if spec["quantized"] else None
    swiglu.is_valid(x, scale)
    actual = swiglu(x, scale)
    expected = swiglu.reference(x, scale)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-3)
    matched = torch.ops.trtllm.silu_and_mul(x, scale=scale, dtype=actual.dtype)
    assert torch.equal(actual.view(torch.uint8), matched.view(torch.uint8))
