"""Unit tests for Dense GEMM with SwiGLU fusion (FC1 kernel).

This module tests the NVFP4 dense GEMM with SwiGLU fusion kernel used in MoE FC1 layers.
The kernel performs: C = alpha_post * SwiGLU(alpha * (A @ B.T))
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.quantization import interleave_linear_and_gate
from tensorrt_llm._torch.utils import swizzle_sf, unswizzle_sf
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.math_utils import pad_up


def swiglu_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU implementation: x * silu(gate) where [x, gate] = chunk(input, 2)."""
    x, gate = x.chunk(2, dim=-1)
    return x * torch.nn.functional.silu(gate)


def nvfp4_dense_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    scaling_vector_size: int = 16,
) -> torch.Tensor:
    """Reference implementation for dense GEMM using nvfp4_gemm op.

    Args:
        a: Input activation (M, K//2) in fp4 packed format
        b: Weight tensor (num_expert, weight_per_expert, K//2) in fp4 packed format
        a_sf: Scale factor for a (swizzled)
        b_sf: Scale factor for b (num_expert, weight_per_expert, scale_k)
        alpha: Per-expert alpha scale (num_expert,)
        output_dtype: Output data type
        scaling_vector_size: Vector size for block scaling

    Returns:
        Output tensor (M, num_expert * weight_per_expert) in output_dtype
    """
    assert a.dtype == torch.float4_e2m1fn_x2
    assert b.dtype == torch.float4_e2m1fn_x2

    m = a.size(0)
    num_expert = b.size(0)
    weight_per_expert = b.size(1)
    n = num_expert * weight_per_expert

    # Compute reference output by calling nvfp4_gemm for each expert
    ref = torch.empty(m, n, dtype=output_dtype, device="cuda")

    for expert_idx in range(num_expert):
        start_col = expert_idx * weight_per_expert
        end_col = (expert_idx + 1) * weight_per_expert

        # Get expert's weight and scale factor
        b_expert = b[expert_idx]  # (weight_per_expert, k//2)
        b_sf_expert = b_sf[expert_idx]  # (weight_per_expert, scale_k)

        # Call nvfp4_gemm for this expert
        ref[:, start_col:end_col] = torch.ops.trtllm.nvfp4_gemm(
            a.view(torch.uint8),
            b_expert.view(torch.uint8),
            a_sf,
            b_sf_expert,
            alpha[expert_idx],
            output_dtype,
        )

    return ref


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="This test is only supported on SM 100 and SM 103 GPUs",
)
@pytest.mark.parametrize("num_expert", [1, 4, 8])
@pytest.mark.parametrize("weight_per_expert", [256, 512])
@pytest.mark.parametrize("num_tokens", [127, 256])
@pytest.mark.parametrize("hidden_size", [256, 512])
@pytest.mark.parametrize("enable_alpha_post", [True, False])
def test_nvfp4_dense_gemm_swiglu_blackwell(
    num_tokens: int,
    hidden_size: int,
    num_expert: int,
    weight_per_expert: int,
    enable_alpha_post: bool,
):
    """Test Dense GEMM with SwiGLU fusion for FC1 layer.

    This test validates the dense GEMM kernel which:
    1. Performs GEMM: C = A @ B.T with per-expert alpha scaling
    2. Applies SwiGLU fusion: output = up * silu(gate) where [up, gate] = chunk(C, 2)
    3. Optionally applies post-SwiGLU alpha scaling (when enable_alpha_post=True)
    4. Quantizes output to NVFP4 format with scale factor generation
    """
    sf_vec_size = 16
    m = num_tokens
    k = hidden_size
    n = num_expert * weight_per_expert  # Full N before SwiGLU

    # Create input tensors in bfloat16, then quantize to fp4
    a_bf16 = torch.randint(-5, 5, (m, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    b_bf16 = torch.randint(
        -5, 5, (num_expert, weight_per_expert, k), dtype=torch.int32, device="cuda"
    ).to(torch.bfloat16)

    # Compute global scale factors
    a_global_sf = a_bf16.abs().max().float() / (448 * 6)
    b_global_sf = b_bf16.abs().amax(dim=(1, 2)).float() / (448 * 6)

    # Quantize to fp4
    a, a_sf = torch.ops.trtllm.fp4_quantize(a_bf16, 1 / a_global_sf, sf_vec_size, False)
    a = a.view(torch.float4_e2m1fn_x2)

    b, b_sf = torch.ops.trtllm.fp4_quantize(b_bf16, 1 / b_global_sf, sf_vec_size, False)
    b = b.view(torch.float4_e2m1fn_x2)
    b_sf = b_sf.view(num_expert, weight_per_expert, k // sf_vec_size)

    # Per-expert alpha = a_global_sf * b_global_sf
    alpha = a_global_sf * b_global_sf

    # Interleave weights for SwiGLU (linear and gate interleaved at group_size=64)
    b_interleaved = interleave_linear_and_gate(b.view(torch.uint8), group_size=64, dim=1).view(
        torch.float4_e2m1fn_x2
    )

    # Interleave scale factors
    b_sf_unswizzled = unswizzle_sf(b_sf, weight_per_expert, k).view(
        num_expert, weight_per_expert, k // sf_vec_size
    )
    b_sf_unswizzled_interleaved = interleave_linear_and_gate(b_sf_unswizzled, group_size=64, dim=1)
    b_sf_interleaved = swizzle_sf(b_sf_unswizzled_interleaved, weight_per_expert, k).view(
        num_expert, weight_per_expert, k // sf_vec_size
    )

    # Compute reference using nvfp4_gemm (simulates fp4 precision)
    c_ref = nvfp4_dense_gemm_ref(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )

    # Apply SwiGLU separately for each expert: reshape -> swiglu -> flatten
    c_ref = swiglu_ref(c_ref.view(m, num_expert, weight_per_expert))  # (m, num_expert, wpe//2)

    # Conditionally apply alpha_post scaling
    alpha_post = None
    if enable_alpha_post:
        # Create per-token per-expert alpha_post scale (m, num_expert)
        # Use values close to 1.0 to minimize numerical variation while testing functionality
        alpha_post = torch.rand(m, num_expert, dtype=torch.float32, device="cuda") * 0.4 + 0.8
        # Apply alpha_post: (m, num_expert, wpe//2) * (m, num_expert, 1) -> (m, num_expert, wpe//2)
        c_ref = c_ref * alpha_post.unsqueeze(-1)

    c_ref = c_ref.to(torch.bfloat16)
    c_ref = c_ref.view(m, -1)  # Flatten to (m, n_out)

    # Output N after SwiGLU
    n_out = n // 2

    # Create norm_const tensor for fp4 output quantization
    global_sf = c_ref.abs().max().float() / (448 * 6)
    norm_const = torch.tensor([1 / global_sf], dtype=torch.float32, device="cuda")

    # Quantize reference output to fp4
    c_ref_quantized, c_sf_ref = torch.ops.trtllm.fp4_quantize(
        c_ref, 1 / global_sf, sf_vec_size, False
    )

    # Call the kernel with fp4 output
    c, c_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
        a,
        b_interleaved,
        a_sf,
        b_sf_interleaved,
        alpha,
        alpha_post,  # None when enable_alpha_post=False
        norm_const,
        expert_count=num_expert,
        weight_per_expert=weight_per_expert,
        output_dtype=torch.float4_e2m1fn_x2,
        scaling_vector_size=sf_vec_size,
    )

    # Verify output shape (fp4 packed: n_out // 2 bytes)
    assert c.shape == (m, n_out // 2), f"Expected shape {(m, n_out // 2)}, got {c.shape}"
    assert c.dtype == torch.float4_e2m1fn_x2, f"Expected dtype float4_e2m1fn_x2, got {c.dtype}"

    # Verify output values by comparing fp4 bytes
    match_ratio = (c.view(torch.uint8) == c_ref_quantized.view(torch.uint8)).sum().item() / c.view(
        torch.uint8
    ).numel()
    assert match_ratio > 0.95, f"Only {match_ratio * 100:.2f}% elements match, expected >= 95%"

    # Verify scale factor shape
    scale_n_out = n_out // sf_vec_size
    expected_c_sf_shape = (32, 4, pad_up(m, 128) // 128, 4, scale_n_out // 4, 1)
    assert c_sf.shape == expected_c_sf_shape, (
        f"Expected c_sf shape {expected_c_sf_shape}, got {c_sf.shape}"
    )
    assert c_sf.dtype == torch.uint8, f"Expected c_sf dtype uint8, got {c_sf.dtype}"

    # Verify scale factor values
    # Unswizzle both c_sf and c_sf_ref for comparison (both are padded to 128)
    c_sf_unswizzled = unswizzle_sf(c_sf.view(-1), pad_up(m, 128), n_out)
    c_sf_ref_unswizzled = unswizzle_sf(c_sf_ref.view(-1), pad_up(m, 128), n_out)

    # Compare only the valid region (first m rows)
    c_sf_valid = c_sf_unswizzled[:m, :]
    c_sf_ref_valid = c_sf_ref_unswizzled[:m, :]
    sf_match_ratio = (
        c_sf_valid.view(torch.uint8) == c_sf_ref_valid.view(torch.uint8)
    ).sum().item() / c_sf_valid.view(torch.uint8).numel()
    assert sf_match_ratio > 0.95, (
        f"Scale factor: only {sf_match_ratio * 100:.2f}% match, expected >= 95%"
    )


def nvfp4_dense_gemm_fc2_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha_scale: torch.Tensor,
    output_dtype: torch.dtype,
    expert_count: int,
    weight_per_expert: int,
    scaling_vector_size: int = 16,
) -> torch.Tensor:
    """Reference implementation for FC2 dense GEMM.

    For each expert, compute GEMM with corresponding A slice and B slice,
    multiply by alpha_scale, then accumulate results.

    C = sum_i(alpha_scale[:, i] * (A[:, i*wpe:(i+1)*wpe] @ B[:, i*wpe:(i+1)*wpe].T))

    Args:
        a: Input activation (M, K//2) in fp4 packed format, K = expert_count * weight_per_expert
        b: Weight tensor (N, K//2) in fp4 packed format
        a_sf: Scale factor for a (swizzled)
        b_sf: Scale factor for b (swizzled)
        alpha_scale: Per-token-per-expert scale (M, expert_count)
        output_dtype: Output data type
        expert_count: Number of experts
        weight_per_expert: Number of weights per expert
        scaling_vector_size: Vector size for block scaling

    Returns:
        Output tensor (M, N) in output_dtype
    """
    assert a.dtype == torch.float4_e2m1fn_x2
    assert b.dtype == torch.float4_e2m1fn_x2

    m = a.size(0)
    k = a.size(1) * 2
    n = b.size(0)

    # Unswizzle scale factors for slicing
    a_sf_unswizzled = unswizzle_sf(a_sf.view(-1), pad_up(m, 128), k)[:m, :]  # (m, scale_k)
    b_sf_unswizzled = unswizzle_sf(b_sf.view(-1), n, k)  # (n, scale_k)

    # Initialize output accumulator
    c_ref = torch.zeros(m, n, dtype=output_dtype, device=a.device)

    # For each expert, compute partial GEMM and accumulate
    wpe = weight_per_expert
    scale_wpe = wpe // scaling_vector_size
    alpha_one = torch.tensor([1.0], dtype=torch.float32, device=a.device)

    for expert_idx in range(expert_count):
        # Slice A for this expert (in packed fp4 format, so divide by 2)
        a_expert = a.view(torch.uint8)[
            :, expert_idx * wpe // 2 : (expert_idx + 1) * wpe // 2
        ].contiguous()
        a_sf_expert = a_sf_unswizzled[:, expert_idx * scale_wpe : (expert_idx + 1) * scale_wpe]
        a_sf_expert_swizzled = swizzle_sf(a_sf_expert.contiguous(), m, wpe)

        # Slice B for this expert
        b_expert = b.view(torch.uint8)[
            :, expert_idx * wpe // 2 : (expert_idx + 1) * wpe // 2
        ].contiguous()
        b_sf_expert = b_sf_unswizzled[:, expert_idx * scale_wpe : (expert_idx + 1) * scale_wpe]
        b_sf_expert_swizzled = swizzle_sf(b_sf_expert.contiguous(), n, wpe)

        # Compute GEMM for this expert
        c_expert = torch.ops.trtllm.nvfp4_gemm(
            a_expert,
            b_expert,
            a_sf_expert_swizzled,
            b_sf_expert_swizzled,
            alpha_one,
            output_dtype,
        )

        # Apply alpha_scale and accumulate
        c_ref += c_expert * alpha_scale[:, expert_idx : expert_idx + 1]

    return c_ref


@pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason="This test is only supported on SM 100 and SM 103 GPUs",
)
@pytest.mark.parametrize("num_expert", [1, 4])
@pytest.mark.parametrize("weight_per_expert", [256])
@pytest.mark.parametrize("num_tokens", [127, 256])
@pytest.mark.parametrize("output_hidden_size", [256, 512])
def test_nvfp4_dense_gemm_fc2_blackwell(
    num_tokens: int, output_hidden_size: int, num_expert: int, weight_per_expert: int
):
    """Test Dense GEMM FC2 for MoE second projection.

    This test validates the FC2 dense GEMM kernel which:
    1. Performs GEMM: C = A @ B.T with per-token-per-expert alpha scaling
    2. Each token has independent alpha values for each expert
    """
    sf_vec_size = 16
    m = num_tokens
    k = num_expert * weight_per_expert  # Input dimension from FC1 output
    n = output_hidden_size  # Output hidden dimension

    # Create input tensors in bfloat16, then quantize to fp4
    a_bf16 = torch.randint(-5, 5, (m, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)
    b_bf16 = torch.randint(-5, 5, (n, k), dtype=torch.int32, device="cuda").to(torch.bfloat16)

    # Compute global scale factors
    a_global_sf = a_bf16.abs().max().float() / (448 * 6)
    b_global_sf = b_bf16.abs().max().float() / (448 * 6)

    # Quantize to fp4
    a, a_sf = torch.ops.trtllm.fp4_quantize(a_bf16, 1 / a_global_sf, sf_vec_size, False)
    a = a.view(torch.float4_e2m1fn_x2)

    b, b_sf = torch.ops.trtllm.fp4_quantize(b_bf16, 1 / b_global_sf, sf_vec_size, False)
    b = b.view(torch.float4_e2m1fn_x2)

    # Create per-token-per-expert alpha scale (m, expert_count)
    # Each token has different alpha values for each expert
    alpha_scale = torch.rand(m, num_expert, dtype=torch.float32, device="cuda") * 2.0

    # Compute reference
    c_ref = nvfp4_dense_gemm_fc2_ref(
        a,
        b,
        a_sf,
        b_sf,
        alpha_scale,
        output_dtype=torch.bfloat16,
        expert_count=num_expert,
        weight_per_expert=weight_per_expert,
        scaling_vector_size=sf_vec_size,
    )

    # Call the kernel
    c = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_fc2_blackwell(
        a,
        b,
        a_sf,
        b_sf,
        alpha_scale,
        expert_count=num_expert,
        weight_per_expert=weight_per_expert,
        output_dtype=torch.bfloat16,
        scaling_vector_size=sf_vec_size,
    )

    # Verify output shape
    assert c.shape == (m, n), f"Expected shape {(m, n)}, got {c.shape}"
    assert c.dtype == torch.bfloat16, f"Expected dtype bfloat16, got {c.dtype}"

    # Verify output values
    # Use relative tolerance for floating point comparison
    rtol = 0.1  # 10% relative tolerance
    atol = 0.05  # Small absolute tolerance for values near zero

    is_close = torch.isclose(c.float(), c_ref.float(), rtol=rtol, atol=atol)
    match_ratio = is_close.sum().item() / c.numel()

    print(f"FC2 Output match ratio: {match_ratio * 100:.2f}%")
    assert match_ratio > 0.90, f"Only {match_ratio * 100:.2f}% elements match, expected >= 90%"
