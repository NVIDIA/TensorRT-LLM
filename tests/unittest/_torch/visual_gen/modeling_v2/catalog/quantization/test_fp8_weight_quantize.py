# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization.fp8_weight_quantize import (
    quantize_fp8_blockwise,
    quantize_fp8_per_tensor,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 weight quantization needs CUDA"
)


def _check(op, *args):
    weights, scales = op(*args)
    ref_weights, ref_scales = op.reference(*args)
    assert weights.dtype == ref_weights.dtype == torch.float8_e4m3fn
    assert torch.equal(weights.view(torch.uint8), ref_weights.view(torch.uint8))
    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)
    return weights


@pytest.mark.parametrize("cell", quantize_fp8_per_tensor.CELLS, ids=lambda cell: cell.why)
def test_per_tensor(cell):
    spec = cell.spec
    shape, dtype = spec["shape"], spec["dtype"]
    if "offset" in spec:
        offset = spec["offset"]
        # 175 / (196 / 448) is the E4M3 midpoint 400. The reciprocal product rounds up.
        storage = torch.full((math.prod(shape) + offset,), 175, dtype=dtype, device="cuda")
        weight = storage[offset:].view(shape)
        assert weight.is_contiguous() and weight.data_ptr() % 16 == offset * weight.element_size()
        weight[0, 0] = 196
    elif spec.get("zero"):
        weight = torch.zeros(shape, dtype=dtype, device="cuda")
    else:
        generator = torch.Generator(device="cuda").manual_seed(41)
        weight = torch.randn(shape, generator=generator, device="cuda").to(dtype)
    quantized = _check(quantize_fp8_per_tensor, weight)
    if "offset" in spec:
        rounded = 416 if spec["offset"] == 0 else 384
        assert torch.all(quantized.flatten()[1:].float() == rounded)


@pytest.mark.parametrize("cell", quantize_fp8_blockwise.CELLS, ids=lambda cell: cell.why)
def test_blockwise(cell):
    spec = cell.spec
    rows, cols = spec["shape"]
    block = spec["block"]
    i = torch.arange(rows, device="cuda")[:, None]
    j = torch.arange(cols, device="cuda")[None, :]
    weight = (((i + j) % 17 - 8) * (1 + 2 * (i // block) + (j // block))).to(spec["dtype"])
    weight[:block, :block] = 1e-7
    if spec.get("strided"):
        weight = weight.T.contiguous().T
    _check(quantize_fp8_blockwise, weight, block)
