import pytest
import torch
import torch.nn.functional as F
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

torch.manual_seed(0)

SCALING_VECTOR_SIZE = 16  # NVFP4 block size along K


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
        weight, weight_scale_2, SCALING_VECTOR_SIZE, False
    )

    output_fp4_gemm = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
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


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mat2_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fp8_bmm(input_dtype, mat2_dtype):
    # Create test tensors: (B, M, K) and (B, K, N)
    batch_size, M, K, N = 2, 32, 64, 80
    input = torch.rand(batch_size, M, K, dtype=input_dtype, device="cuda")
    mat2 = torch.rand(batch_size, K, N, dtype=mat2_dtype, device="cuda")

    # Calculate scales similar to fp8_linear test
    input_scale = (torch.max(torch.abs(input)) / 448).to("cuda")
    mat2_scale = (torch.max(torch.abs(mat2)) / 448).to("cuda")
    mat2_fp8 = (mat2 / mat2_scale).to(torch.float8_e4m3fn)

    # Test fp8_bmm operation
    output_fp8_bmm = torch.ops.auto_deploy.torch_quant_fp8_bmm(
        input,
        mat2_fp8,
        input_scale=input_scale,
        weight_scale=mat2_scale,
    )

    output_fp8_bmm_unquantized_inputs = torch.ops.auto_deploy.torch_quant_fp8_bmm(
        input,
        mat2,
        input_scale=input_scale,
        weight_scale=mat2_scale,
    )

    # Reference implementation using standard bmm
    output_fp32_bmm = torch.bmm(input.float(), mat2.float()).to(torch.half)

    assert output_fp8_bmm.shape == output_fp32_bmm.shape
    assert output_fp8_bmm.shape == (batch_size, M, N)

    cos_sim = F.cosine_similarity(output_fp32_bmm.reshape(-1), output_fp8_bmm.reshape(-1), dim=0)
    cos_sim_unquantized = F.cosine_similarity(
        output_fp32_bmm.reshape(-1), output_fp8_bmm_unquantized_inputs.reshape(-1), dim=0
    )
    assert cos_sim > 0.99
    assert cos_sim_unquantized > 0.99


@pytest.mark.parametrize("bias", [torch.rand(32, device="cuda") * 10, None])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_quant_linear_fp8_matches_fused_op(bias):
    input = torch.rand(3, 16, device="cuda")
    weight = torch.rand(32, 16, device="cuda")

    weight_scale = (torch.max(torch.abs(weight)) / 448).to("cuda")
    weight_fp8 = (weight / weight_scale).to(torch.float8_e4m3fn)

    out_fused = torch.ops.auto_deploy.torch_quant_fp8_linear(
        input,
        weight_fp8,
        bias=bias,
        input_scale=torch.tensor(1.0, device="cuda"),
        weight_scale=weight_scale,
    )

    out_unified = torch.ops.auto_deploy.torch_fake_quant_fp8_linear(
        input,
        weight_fp8,
        bias,
        [torch.tensor(1.0, device="cuda")],
        [weight_scale],
        [],
        [],
    )

    assert out_unified.shape == out_fused.shape
    torch.testing.assert_close(out_unified, out_fused, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize(
    "bias",
    [
        (torch.rand(32, device="cuda") * 10).to(torch.float16),
        None,
    ],
)
@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires NVFP4 and TRT-LLM ops",
)
def test_quant_linear_nvfp4_matches_fused_op(bias):
    x = torch.rand(3, 32, device="cuda", dtype=torch.half)  # [..., K]
    W = torch.rand(32, 32, device="cuda", dtype=torch.half)  # [N, K]
    N, K = W.shape
    assert K % SCALING_VECTOR_SIZE == 0

    # Per-tensor scale-2 (amax / (448 * 6))
    s_in2 = fp4_global_scale(x).to(torch.float32)  # input per-tensor scale
    s_w2 = fp4_global_scale(W).to(torch.float32)  # weight per-tensor scale

    weight_fp4, weight_scale_cutlass = torch.ops.trtllm.fp4_quantize(
        W, s_w2, SCALING_VECTOR_SIZE, False
    )
    assert weight_fp4.dtype == torch.uint8
    assert weight_scale_cutlass.dtype == torch.uint8

    # Fused op (expects CUTLASS uint8 scale + kernel alpha = 1/(s_in2*s_w2))
    alpha_fused = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)

    out_fused = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        x,
        weight_fp4,
        bias=bias,
        input_scale=s_in2,
        weight_scale=weight_scale_cutlass,
        alpha=alpha_fused,
    )

    out_unified = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        weight_fp4,
        bias,
        [s_in2],  # input_scale list
        [
            weight_scale_cutlass,
            alpha_fused,
        ],  # weight_scale list: [per-block vector, combined alpha]
        [],  # input_zp
        [],  # weight_zp
    )

    assert out_unified.shape == out_fused.shape
    torch.testing.assert_close(out_unified, out_fused, rtol=1e-3, atol=5e-3)
