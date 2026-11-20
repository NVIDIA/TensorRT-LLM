# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 QA for exact scalar quantization and the shared output cast."""

import pytest
import torch


@pytest.fixture(scope="module")
def conversion_modules(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("The conversion kernels require an SM107 GPU")
    if torch.cuda.get_device_properties(0).multi_processor_count != 212:
        pytest.skip("The fixed conversion grid requires the qualified 212-SM GPU")
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1.kernels.conversion import (
        prepare,
    )

    scales = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0] * 10, device="cuda", dtype=torch.float32)
    return prepare(scales, tmp_path_factory.mktemp("h3_fp8_conversions"))


@pytest.mark.parametrize("length", [1, 7, 8, 9, 255, 256, 257, 2049])
def test_exact_conversion_and_odd_tails(conversion_modules, length):
    quant, cast, _, proof = conversion_modules
    assert proof["unary_qualified"]
    raw = torch.linspace(-450, 450, length, device="cuda", dtype=torch.bfloat16).reshape(1, length)
    scale = torch.tensor([0.1], device="cuda", dtype=torch.float32)
    factor = quant.factors(scale, True, quant.pdl)
    output = quant.quantize_config(raw, scale, factor, 0, quant.pdl, 848, 256)
    expected = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(raw, scale)[0]
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    converted = cast.cast(raw, 1, 848, 256)
    assert torch.equal(converted.view(torch.uint8), raw.float().view(torch.uint8))
