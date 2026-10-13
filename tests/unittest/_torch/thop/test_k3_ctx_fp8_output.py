# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-scale CTX output fusion preserves BF16-to-FP8 rounding."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin MXFP8 GEMM requires SM107",
)


@pytest.mark.parametrize("m", [1, 124, 257])
@pytest.mark.parametrize("use_tvm_ffi", [False, True])
def test_fp8_output_matches_bf16_then_quant(m, use_tvm_ffi):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    torch.manual_seed(107)
    x = torch.randn(m, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
    b, b_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [a, b, a_sf, b_sf, alpha]
    tactic = ("base", (128, 128, 128), (128, 128, 64), (1, 1), False, False, "static", "m")
    reference = CuteDSLMXFP8RubinLinear(torch.bfloat16, use_tvm_ffi=use_tvm_ffi)
    candidate = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn, use_tvm_ffi=use_tvm_ffi)
    # Values above the E4M3 range exercise finite saturation, in addition to
    # the ordinary distribution and a non-unit GEMM alpha.
    scales = [1.0, 0.5, 1.3, 32.0]
    if m == 124 and use_tvm_ffi:
        scales += [float("inf"), float("-inf"), float("nan")]
    for scale in scales:
        alpha.fill_(scale)
        expected = reference(inputs, tactic=tactic)
        expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            expected, torch.ones_like(alpha)
        )
        actual = candidate(inputs, tactic=tactic)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("heads", [12, 96])
def test_weight_read_fp8_output_matches_projection(heads):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    torch.manual_seed(10698)
    x = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(heads * 256, 512, dtype=torch.bfloat16, device="cuda") / 32
    weight, weight_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w)
    projection = _Fp8BlockScaleWeightReadLinear(weight, weight_scale, heads * 256)
    expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        projection(x), torch.ones((), dtype=torch.float32, device="cuda")
    )
    activation, scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
    actual = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
        activation,
        projection.weight,
        scale,
        projection.weight_scale,
        output_dtype=torch.float8_e4m3fn,
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
