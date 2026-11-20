# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.quantization.fp8_weight_quantize import (
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


@pytest.mark.parametrize("cell", quantize_fp8_per_tensor.CELLS, ids=lambda cell: cell.why)
def test_per_tensor(cell):
    spec = cell.spec
    generator = torch.Generator(device="cuda").manual_seed(41)
    weight = torch.randn(spec["shape"], generator=generator, device="cuda").to(spec["dtype"])
    if spec.get("zero"):
        weight.zero_()
    _check(quantize_fp8_per_tensor, weight)


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
