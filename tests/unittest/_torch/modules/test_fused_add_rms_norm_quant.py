# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for fused_add_rms_norm_quant with/without output_hp_norm support, and for the NCCL allreduce
fallback's fused residual + RMSNorm + FP8 quantize."""

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceStrategy


def fused_add_rms_norm_quant_available():
    """Check if the fused_add_rms_norm_quant op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_add_rms_norm_quant")


skip_unsupported = pytest.mark.skipif(
    getSMVersion() < 100 or not fused_add_rms_norm_quant_available(),
    reason="Requires Blackwell+ (SM100+) and trtllm.fused_add_rms_norm_quant op",
)


def rms_norm_ref(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference RMSNorm implementation with residual addition.

    Args:
        hidden_states: Input tensor [M, N]
        residual: Residual tensor [M, N]
        gamma: Weight tensor [N]
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (normalized_output, residual_output)
    """
    input_dtype = hidden_states.dtype

    # Add residual
    hidden_states_fp32 = hidden_states.float() + residual.float()
    residual_out = hidden_states_fp32.to(input_dtype)

    # RMSNorm
    variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
    normed = hidden_states_fp32 * torch.rsqrt(variance + eps)
    normed_output = (gamma.float() * normed).to(input_dtype)

    return normed_output, residual_out


def get_swizzled_sf_indices(m: int, n: int, sf_vec_size: int = 16) -> list[int]:
    """
    Compute the valid indices in swizzled SF layout for given m and n.

    The swizzled layout uses 128x4 tiles:
    - SF layout: [numMTiles, numKTiles, 32 (outerM), 4 (innerM), 4 (innerK)]
    - Each SF block has 128 rows, padded to multiple of 128
    - Each SF block has columns padded to multiple of 4

    Args:
        m: Number of rows
        n: Hidden dimension
        sf_vec_size: Number of elements sharing one scale factor (default 16)

    Returns:
        List of valid indices in the swizzled buffer
    """
    num_col_vecs = n // sf_vec_size  # Number of SF columns
    indices = []

    for m_idx in range(m):
        for k_idx in range(num_col_vecs):
            # Compute swizzled offset using 128x4 tile layout
            inner_k_idx = k_idx % 4
            inner_k_stride = 1

            inner_m_idx = (m_idx % 128) // 32
            inner_m_stride = 4 * inner_k_stride  # 4

            outer_m_idx = m_idx % 32
            outer_m_stride = 4 * inner_m_stride  # 16

            k_tile_idx = k_idx // 4
            k_tile_stride = 32 * outer_m_stride  # 512

            num_k_tiles = (num_col_vecs + 3) // 4
            m_tile_idx = m_idx // 128
            m_tile_stride = num_k_tiles * k_tile_stride

            offset = (
                m_tile_idx * m_tile_stride
                + k_tile_idx * k_tile_stride
                + outer_m_idx * outer_m_stride
                + inner_m_idx * inner_m_stride
                + inner_k_idx * inner_k_stride
            )
            indices.append(offset)

    return indices


@skip_unsupported
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_quant_basic(m, n, dtype):
    """Test basic functionality of fused_add_rms_norm_quant without hp_output."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Create input tensors
    hidden_states = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    gamma = torch.ones(n, dtype=dtype, device=device)

    # Compute sf_scale (per-tensor scale)
    eps = 1e-6
    normed_ref, _ = rms_norm_ref(hidden_states, residual, gamma, eps)
    sf_scale = (normed_ref.abs().amax().float() / (6.0 * 448.0)).view(1)

    # Run fused kernel without hp_output
    normed_fp4, residual_out, sf_out, dummy_output = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps
    )
    assert dummy_output is None

    # Verify output shapes
    assert normed_fp4.shape[0] == m
    assert residual_out.shape == (m, n)
    assert residual_out.dtype == dtype


@skip_unsupported
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_quant_with_hp_output(m, n, dtype):
    """Test fused_add_rms_norm_quant with output_hp_norm=True."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Create input tensors
    hidden_states = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    gamma = torch.ones(n, dtype=dtype, device=device)

    # Compute sf_scale
    eps = 1e-6
    normed_ref, residual_ref = rms_norm_ref(hidden_states, residual, gamma, eps)
    sf_scale = (normed_ref.abs().amax().float() / (6.0 * 448.0)).view(1)

    # Run fused kernel with hp_output
    results = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=True
    )

    # Should return 4 tensors when output_hp_norm=True
    assert len(results) == 4, f"Expected 4 outputs, got {len(results)}"

    normed_fp4, residual_out, sf_out, hp_normed_output = results

    # Verify output shapes
    assert normed_fp4.shape[0] == m
    assert residual_out.shape == (m, n)
    assert hp_normed_output.shape == (m, n)

    # Verify dtypes
    assert residual_out.dtype == dtype
    assert hp_normed_output.dtype == dtype

    # Verify high precision output matches reference
    torch.testing.assert_close(hp_normed_output, normed_ref, rtol=1e-2, atol=1e-2)

    # Verify residual output matches reference
    torch.testing.assert_close(residual_out, residual_ref, rtol=1e-3, atol=1e-3)


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_quant_hp_output_consistency(dtype):
    """Test that hp_output is consistent with the quantized output."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    m, n = 64, 4096

    # Create input tensors
    hidden_states = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    gamma = torch.ones(n, dtype=dtype, device=device)

    eps = 1e-6
    normed_ref, _ = rms_norm_ref(hidden_states, residual, gamma, eps)
    sf_scale = (normed_ref.abs().amax().float() / (6.0 * 448.0)).view(1)

    # Run without hp_output
    results_no_hp = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=False
    )
    assert results_no_hp[3] is None
    normed_fp4_no_hp, residual_out_no_hp, sf_out_no_hp = results_no_hp[:3]

    # Run with hp_output
    results_hp = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=True
    )
    normed_fp4_hp, residual_out_hp, sf_out_hp, hp_normed_output = results_hp

    # The quantized outputs should be identical regardless of hp_output flag
    torch.testing.assert_close(normed_fp4_hp, normed_fp4_no_hp, rtol=0, atol=0)
    torch.testing.assert_close(residual_out_hp, residual_out_no_hp, rtol=0, atol=0)
    # Compare only valid SF indices (swizzled layout pads rows to 128)
    valid_sf_indices = get_swizzled_sf_indices(m, n)
    sf_out_hp_valid = sf_out_hp[valid_sf_indices]
    sf_out_no_hp_valid = sf_out_no_hp[valid_sf_indices]
    torch.testing.assert_close(sf_out_hp_valid, sf_out_no_hp_valid, rtol=0, atol=0)


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rms_norm_quant_gamma_weight(dtype):
    """Test fused_add_rms_norm_quant with non-trivial gamma weights."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    m, n = 32, 2048

    # Create input tensors
    hidden_states = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    # Non-trivial gamma weights
    gamma = torch.randn(n, dtype=dtype, device=device) * 0.5 + 1.0

    eps = 1e-6
    normed_ref, residual_ref = rms_norm_ref(hidden_states, residual, gamma, eps)
    sf_scale = (normed_ref.abs().amax().float() / (6.0 * 448.0)).view(1)

    # Run with hp_output
    results = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=True
    )
    normed_fp4, residual_out, sf_out, hp_normed_output = results

    # Verify high precision output matches reference
    torch.testing.assert_close(hp_normed_output, normed_ref, rtol=1e-2, atol=1e-2)

    # Verify residual output matches reference
    torch.testing.assert_close(residual_out, residual_ref, rtol=1e-3, atol=1e-3)


@skip_unsupported
def test_fused_add_rms_norm_quant_large_batch():
    """Test fused_add_rms_norm_quant with larger batch size."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    m, n = 512, 4096
    dtype = torch.bfloat16

    hidden_states = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    gamma = torch.ones(n, dtype=dtype, device=device)

    eps = 1e-6
    normed_ref, residual_ref = rms_norm_ref(hidden_states, residual, gamma, eps)
    sf_scale = (normed_ref.abs().amax().float() / (6.0 * 448.0)).view(1)

    results = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=True
    )
    normed_fp4, residual_out, sf_out, hp_normed_output = results

    assert hp_normed_output.shape == (m, n)
    torch.testing.assert_close(hp_normed_output, normed_ref, rtol=1e-2, atol=1e-2)


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_latency_layernorm_hp_output_consistency(dtype):
    """
    Test that low_latency_layernorm hp_output is consistent with/without the flag.

    The quantized outputs should be identical regardless of output_hp_norm flag.
    Uses m=1 to trigger the low_latency_layernorm path.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")

    m, n = 1, 4096  # m=1 triggers low_latency_layernorm path

    hidden_states = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    gamma = torch.ones(n, dtype=dtype, device=device)

    eps = 1e-6
    normed_ref, _ = rms_norm_ref(hidden_states, residual, gamma, eps)
    sf_scale = (normed_ref.abs().amax().float() / (6.0 * 448.0)).view(1)

    # Run without hp_output
    results_no_hp = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps
    )
    assert len(results_no_hp) == 4, f"Expected 4 outputs, got {len(results_no_hp)}"
    assert results_no_hp[3] is None, "Expected 4th output to be None when output_hp_norm=False"
    normed_fp4_no_hp, residual_out_no_hp, sf_out_no_hp = results_no_hp[:3]

    # Run with hp_output
    results_hp = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=True
    )
    assert len(results_hp) == 4, f"Expected 4 outputs, got {len(results_hp)}"
    normed_fp4_hp, residual_out_hp, sf_out_hp, hp_normed_output = results_hp

    # The quantized outputs should be identical regardless of hp_output flag
    torch.testing.assert_close(normed_fp4_hp, normed_fp4_no_hp, rtol=0, atol=0)
    torch.testing.assert_close(residual_out_hp, residual_out_no_hp, rtol=0, atol=0)
    # Compare only valid SF indices (swizzled layout pads rows to 128)
    valid_sf_indices = get_swizzled_sf_indices(m, n)
    sf_out_hp_valid = sf_out_hp[valid_sf_indices]
    sf_out_no_hp_valid = sf_out_no_hp[valid_sf_indices]
    torch.testing.assert_close(sf_out_hp_valid, sf_out_no_hp_valid, rtol=0, atol=0)


@skip_unsupported
@pytest.mark.parametrize("m,n", [(1, 5120), (4, 7168), (64, 2560)])
def test_fused_add_rms_norm_quant_non_power_of_2_hidden(m, n):
    """Non-power-of-2 hidden sizes must not hang or corrupt output."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    eps = 1e-6

    hidden_states = torch.randn(m, n, dtype=dtype, device=device)
    residual = torch.randn(m, n, dtype=dtype, device=device)
    gamma = torch.randn(n, dtype=dtype, device=device) * 0.5 + 1.0

    normed_ref, residual_ref = rms_norm_ref(hidden_states, residual, gamma, eps)
    sf_scale = (normed_ref.abs().amax().float() / (6.0 * 448.0)).view(1)

    # Run without hp_output
    results_no_hp = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=False
    )
    assert results_no_hp[3] is None
    normed_fp4_no_hp, residual_out_no_hp, sf_out_no_hp = results_no_hp[:3]

    # Run with hp_output
    results_hp = torch.ops.trtllm.fused_add_rms_norm_quant(
        hidden_states, residual, gamma, sf_scale, True, eps=eps, output_hp_norm=True
    )
    normed_fp4_hp, residual_out_hp, sf_out_hp, hp_normed_output = results_hp

    # Verify residual and HP norm against references
    torch.testing.assert_close(residual_out_no_hp, residual_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_out_hp, residual_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(hp_normed_output, normed_ref, rtol=1e-2, atol=1e-2)

    # Quantized outputs must be identical regardless of hp_output flag
    torch.testing.assert_close(normed_fp4_hp, normed_fp4_no_hp, rtol=0, atol=0)
    torch.testing.assert_close(residual_out_hp, residual_out_no_hp, rtol=0, atol=0)
    # Compare only valid SF indices (swizzled layout pads rows to 128)
    valid_sf_indices = get_swizzled_sf_indices(m, n)
    sf_out_hp_valid = sf_out_hp[valid_sf_indices]
    sf_out_no_hp_valid = sf_out_no_hp[valid_sf_indices]
    torch.testing.assert_close(sf_out_hp_valid, sf_out_no_hp_valid, rtol=0, atol=0)


# The NCCL / NCCL_SYMMETRIC allreduce fallback of every model fuses residual + RMSNorm + FP8 quantize after the
# reduce. A one-rank group runs the same post-reduce path in one process, so these run on single-GPU stages,
# Hopper included; they keep their own GPU gate and do not use skip_unsupported (SM100+).
_FP8_FUSION_OPS = [
    pytest.param(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8, id="residual_rms_norm_quant_fp8"),
    pytest.param(
        AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8, id="residual_rms_norm_out_quant_fp8"
    ),
]


def _nccl_one_rank_allreduce(x, residual, norm_weight, scale, op):
    return torch.ops.trtllm.allreduce(
        input=x,
        residual=residual,
        norm_weight=norm_weight,
        scale=scale,
        bias=None,
        workspace=None,
        group=[0],
        strategy=AllReduceStrategy.NCCL,
        op=op,
        eps=1e-5,
        trigger_completion_at_end=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("seq_len", [1, 7, 300], ids=lambda x: f"seqlen:{x}")
@pytest.mark.parametrize("hidden_size", [128, 256, 4096, 7168, 8192], ids=lambda x: f"hidden:{x}")
@pytest.mark.parametrize("fusion_op", _FP8_FUSION_OPS)
@torch.inference_mode()
def test_nccl_residual_rms_norm_quant_fp8_bitwise(seq_len, hidden_size, fusion_op):
    """The NCCL fallback's fused residual + RMSNorm + FP8 quantize (one kernel where the hidden size allows it) is
    bit for bit RESIDUAL_RMS_NORM followed by the static per-tensor quantize."""
    torch.manual_seed(0)
    x = torch.randn((seq_len, hidden_size), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    norm_weight = torch.randn((hidden_size,), dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.37, dtype=torch.float32, device="cuda")

    norm, residual_out = _nccl_one_rank_allreduce(
        x, residual, norm_weight, scale, AllReduceFusionOp.RESIDUAL_RMS_NORM
    )
    ref_quant, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(norm, scale)
    outputs = _nccl_one_rank_allreduce(x, residual, norm_weight, scale, fusion_op)
    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        assert torch.equal(outputs[0], norm)
    assert torch.equal(outputs[-2].view(torch.uint8), ref_quant.view(torch.uint8))
    assert torch.equal(outputs[-1], residual_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("fusion_op", _FP8_FUSION_OPS)
@torch.inference_mode()
def test_nccl_residual_rms_norm_quant_fp8_no_tokens(fusion_op):
    """The NCCL fallback's fused residual + RMSNorm + FP8 quantize returns empty outputs for an empty input."""
    hidden_size = 4096
    x = torch.empty((0, hidden_size), dtype=torch.bfloat16, device="cuda")
    outputs = _nccl_one_rank_allreduce(
        x,
        torch.empty_like(x),
        torch.ones((hidden_size,), dtype=torch.bfloat16, device="cuda"),
        torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        fusion_op,
    )
    torch.cuda.synchronize()
    expected = 3 if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8 else 2
    assert len(outputs) == expected
    assert all(o.shape == (0, hidden_size) for o in outputs)
    assert outputs[-2].dtype == torch.float8_e4m3fn
