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
    modules = prepare(scales, tmp_path_factory.mktemp("h3_fp8_conversions"))
    if modules[0] is None:
        pytest.skip(f"The conversion kernels were not built: {modules[3]['reason']}")
    return modules


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


def test_ops_use_the_built_conversions_with_identical_bytes(conversion_modules):
    """``ops.output_cast`` and the quantize inside ``ops.attention_projection`` on the built kernels."""
    import tensorrt_llm  # noqa: F401  -- registers the trtllm operators
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1 import ops
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1.kernels import (
        conversion,
        projection_residual,
    )

    quant, cast, factors, _ = conversion_modules
    device = torch.cuda.current_device()
    assert conversion._MODULES[device] == (quant, cast)
    gen = torch.Generator(device="cuda").manual_seed(7)
    length, hidden_dim, qk = 2049, 5376, 7168
    hidden = torch.randn((1, length, hidden_dim), generator=gen, device="cuda").to(torch.bfloat16)
    assert torch.equal(ops.output_cast(hidden).view(torch.uint8), hidden.float().view(torch.uint8))

    projection_residual.compile_kernel(
        device,
        hidden_dim,
        qk,
        projection_residual.ATTENTION_OUTPUT_GATE,
        torch.bfloat16,
        torch.int64,
    )
    raw = (torch.randn((1, length, qk), generator=gen, device="cuda") * 1.5).to(torch.bfloat16)
    weight = (torch.randn((hidden_dim, qk), generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    weight_scale = (weight.float().abs().max() / 448).reshape(1)
    weight = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(weight, weight_scale)[0]
    modulation = (torch.randn((3, 6 * hidden_dim), generator=gen, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    indices = torch.randint(0, 3, (length,), generator=gen, device="cuda")
    norm_weight = torch.ones(hidden_dim, device="cuda", dtype=torch.bfloat16)
    # prepare() was given these scales: index 1 is 0.1.
    input_scale = torch.tensor([0.1], device="cuda", dtype=torch.float32)
    up_reciprocal = torch.reciprocal(torch.tensor([0.027])).cuda()

    def run():
        return ops.attention_projection(
            raw,
            weight.contiguous(),
            input_scale,
            weight_scale,
            hidden,
            norm_weight,
            modulation,
            indices,
            up_reciprocal,
            factors[1:2],
            1e-6,
        )

    built = run()
    conversion._MODULES[device] = (None, None)
    try:
        installed = run()
    finally:
        conversion._MODULES[device] = (quant, cast)
    for a, b in zip(built, installed):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
