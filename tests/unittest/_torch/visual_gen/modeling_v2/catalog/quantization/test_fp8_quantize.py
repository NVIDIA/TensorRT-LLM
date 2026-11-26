# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization import fp8_quantize
from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization.fp8_quantize import (
    native_fp8_per_tensor_quantize,
    native_static_fp8_per_tensor_quantize,
    quantize_e4m3_per_tensor,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA FP8 quantization"
)


@pytest.mark.parametrize("cell", native_fp8_per_tensor_quantize.CELLS, ids=lambda cell: cell.why)
def test_dynamic_fp8(cell):
    x = torch.arange(1, 193, device="cuda", dtype=torch.bfloat16).reshape(cell.spec["shape"])
    native = native_fp8_per_tensor_quantize(x)
    expected = native_fp8_per_tensor_quantize.reference(x)
    assert torch.equal(native[0].view(torch.uint8), expected[0].view(torch.uint8))
    torch.testing.assert_close(native[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("tactic", ["trtllm", "te"])
@pytest.mark.parametrize("cell", quantize_e4m3_per_tensor.CELLS, ids=lambda cell: cell.why)
def test_tuned_fp8(cell, tactic, monkeypatch):
    x = torch.arange(1, 193, device="cuda", dtype=torch.bfloat16).reshape(cell.spec["shape"])
    if tactic == "te":
        quantizer = fp8_quantize._te_quantizer()
        if quantizer is None:
            pytest.skip("requires Transformer Engine current-scaling quantization")
        te = Mock(wraps=quantizer)
        monkeypatch.setattr(fp8_quantize, "_te_quantizer", lambda: te)
    monkeypatch.setattr(
        AutoTuner.get(),
        "choose_one",
        lambda op, runners, config, inputs: (runners[0], tactic),
    )
    actual, scale = quantize_e4m3_per_tensor(x)
    if tactic == "te":
        te.quantize.assert_called_once()
    reference_codes, reference_scale = quantize_e4m3_per_tensor.reference(x)
    torch.testing.assert_close(
        actual.float() * scale, reference_codes.float() * reference_scale, rtol=0.13, atol=0.02
    )
    matched, matched_scale = torch.ops.trtllm.quantize_e4m3_per_tensor(x)
    assert torch.equal(actual.view(torch.uint8), matched.view(torch.uint8))
    torch.testing.assert_close(scale, matched_scale, rtol=0, atol=0)


@pytest.mark.parametrize(
    "cell", native_static_fp8_per_tensor_quantize.CELLS, ids=lambda cell: cell.why
)
def test_static_fp8(cell):
    x = torch.arange(-48, 48, device="cuda", dtype=torch.bfloat16).reshape(cell.spec["shape"])
    scale = torch.tensor(0.0625, device="cuda")
    expected = native_static_fp8_per_tensor_quantize.reference(x, scale)
    actual = native_static_fp8_per_tensor_quantize(x, scale)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
