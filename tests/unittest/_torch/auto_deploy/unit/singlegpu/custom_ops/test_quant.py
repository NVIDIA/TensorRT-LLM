import pytest
import torch
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

torch.manual_seed(0)

scaling_vector_size = 16


@pytest.mark.parametrize("bias", [torch.rand(32).to("cuda") * 10, None])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fp8_linear(bias):
    input = torch.rand(3, 16).to("cuda")
    weight = torch.rand(32, 16).to("cuda")
    bias = torch.rand(32).to("cuda") * 10

    weight_scale = (torch.max(torch.abs(weight)) / 448).to("cuda")
    weight_fp8 = (weight / weight_scale).to(torch.float8_e4m3fn)

    output_fp8_gemm = torch.ops.auto_deploy.torch_quant_fp8_linear(
        input,
        weight_fp8,
        bias=bias,
        input_scale=torch.tensor(1.0).to("cuda"),
        weight_scale=weight_scale,
    )
    output_fp32_gemm = torch.ops.aten.linear.default(input, weight, bias=bias)

    assert output_fp8_gemm.shape == output_fp32_gemm.shape

    assert torch.allclose(output_fp8_gemm, output_fp32_gemm, rtol=0.01, atol=0.15)


@pytest.mark.skipif(
    not fp4_compatible() or not trtllm_ops_available(),
    reason="Requires fp4 and trtllm support",
)
def test_fp4_linear():
    input = torch.rand(1, 3, 64, dtype=torch.half, device="cuda")
    weight = torch.rand(128, 64, dtype=torch.half, device="cuda")

    input_scale = fp4_global_scale(input)
    weight_scale_2 = fp4_global_scale(weight)

    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(
        weight, weight_scale_2, scaling_vector_size, False
    )

    output_fp4_gemm = torch.ops.auto_deploy.torch_quant_fp4_linear(
        input,
        weight_fp4,
        bias=None,
        input_scale=input_scale,
        weight_scale=weight_scale,
        alpha=1 / (input_scale * weight_scale_2),
    )
    output_fp16_gemm = torch.ops.aten.linear.default(input, weight, bias=None)

    assert output_fp4_gemm.shape == output_fp16_gemm.shape
    assert torch.allclose(output_fp4_gemm, output_fp16_gemm, rtol=1e-1, atol=1e-2)
