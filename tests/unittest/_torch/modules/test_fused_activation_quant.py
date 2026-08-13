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
"""Unit tests for fused relu2 + NVFP4 quantization kernel."""

import pytest
import torch
import torch.nn.functional as F

from tests.unittest.utils.util import getSMVersion


def fused_relu2_quantize_available():
    """Check if the fused_relu2_quantize op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_relu2_quantize")


def fp4_quantize_available():
    """Check if the fp4_quantize op is available."""
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fp4_quantize")


skip_unless_fused_relu2_quantize = pytest.mark.skipif(
    getSMVersion() < 100 or not fused_relu2_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm.fused_relu2_quantize op",
)

skip_unless_fused_relu2_and_fp4_quantize = pytest.mark.skipif(
    getSMVersion() < 100 or not fused_relu2_quantize_available() or not fp4_quantize_available(),
    reason="Requires SM100+ (Blackwell) and trtllm fused_relu2_quantize + fp4_quantize ops",
)


# FP4 E2M1 lookup table for reference implementation
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])


def relu2(x: torch.Tensor) -> torch.Tensor:
    """Reference relu2 activation: square(relu(x))."""
    return torch.square(F.relu(x))


def cast_to_fp4(weight: torch.Tensor) -> torch.Tensor:
    """Cast tensor values to FP4 E2M1 format (as uint8)."""
    device = weight.device

    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
    mask_shape = list(weight.shape)
    mask = mask.expand([*mask_shape, 7])

    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()

    ord_val = torch.searchsorted(E2M1_BOUNDS.to(device), weight_abs, out_int32=True).to(torch.uint8)
    round_val = torch.any((weight_abs.unsqueeze(-1) == E2M1_BOUNDS.to(device)) * mask, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord_val + round_val).to(torch.uint8)
    return fp4_val


def quantize_nvfp4_ref(
    input: torch.Tensor, sf_scale: torch.Tensor, sf_vec_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference NVFP4 quantization implementation.

    Args:
        input: Input tensor [M, N], already activated (e.g., after relu2)
        sf_scale: Per-tensor scaling factor (sf_scale = amax / (6 * 448))
        sf_vec_size: Block size for per-block scaling (default 16)

    Returns:
        Tuple of (fp4_packed, scale_factors)
    """
    m, n = input.shape
    assert n % sf_vec_size == 0, f"N ({n}) must be divisible by sf_vec_size ({sf_vec_size})"

    # Reshape for block-wise quantization
    input_blocked = input.view(m, n // sf_vec_size, sf_vec_size)

    # Compute per-block amax
    per_block_amax = input_blocked.abs().amax(dim=-1).float()

    # Compute per-block scale: amax / 6.0
    per_block_scale = per_block_amax / 6.0

    # Quantize per-block scale to FP8
    q_per_block_scale = per_block_scale / sf_scale
    q_per_block_scale[per_block_scale == 0] = 1.0
    q_per_block_scale_fp8 = q_per_block_scale.to(torch.float8_e4m3fn)

    # Dequantize scale for actual quantization
    scale_dequant = q_per_block_scale_fp8.float() * sf_scale

    # Scale the input
    scale_expanded = scale_dequant.unsqueeze(-1).expand_as(input_blocked)
    scaled_input = input_blocked / (scale_expanded + 1e-12)
    scaled_input = scaled_input.view(m, n)

    # Cast to FP4
    fp4_vals = cast_to_fp4(scaled_input)

    # Pack two FP4 values into one uint8
    packed = (fp4_vals[..., 1::2] << 4) | fp4_vals[..., 0::2]

    return packed, q_per_block_scale_fp8


def fused_relu2_quantize_ref(
    input: torch.Tensor, sf_scale: torch.Tensor, sf_vec_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation for fused relu2 + NVFP4 quantization.

    Args:
        input: Input tensor [M, N]
        sf_scale: Per-tensor scaling factor
        sf_vec_size: Block size for per-block scaling (default 16)

    Returns:
        Tuple of (fp4_packed, scale_factors)
    """
    # Apply relu2 activation
    activated = relu2(input)
    # Quantize to NVFP4
    return quantize_nvfp4_ref(activated, sf_scale, sf_vec_size)


@skip_unless_fused_relu2_quantize
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_relu2_quantize_zeros(dtype):
    """Test fused_relu2_quantize with inputs that produce zeros after relu2."""
    device = torch.device("cuda")

    # All negative inputs -> relu2 produces all zeros
    m, n = 32, 64
    input_tensor = -torch.abs(torch.randn(m, n, dtype=dtype, device=device))
    sf_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    fp4_fused, sf_fused = torch.ops.trtllm.fused_relu2_quantize(input_tensor, sf_scale, 16)

    assert fp4_fused.shape == (m, n // 2)
    assert (fp4_fused == 0).all(), "All negative inputs should produce zero output"


@skip_unless_fused_relu2_and_fp4_quantize
@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_relu2_quantize_vs_separate_ops(m, n, dtype):
    """
    Compare fused_relu2_quantize kernel output against separate relu2 + fp4_quantize.

    This test verifies that the fused CUDA kernel produces FP4 packed values that
    closely match running relu2 activation followed by fp4_quantize separately.

    Note: Due to floating point precision differences in intermediate calculations
    (e.g., FMA vs separate mul+add), a small percentage of values at quantization
    boundaries may differ. We require >= 99% match rate.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")

    input_tensor = torch.randn(m, n, dtype=dtype, device=device)
    activated = relu2(input_tensor)
    sf_scale = (activated.abs().amax().float() / (6.0 * 448.0)).to(device)
    sf_scale = sf_scale.view(1)

    fp4_separate, sf_separate = torch.ops.trtllm.fp4_quantize(
        activated,
        sf_scale,
        16,
        False,
        True,  # use_ue8m0=False, is_sf_swizzled_layout=True
    )
    fp4_fused, sf_fused = torch.ops.trtllm.fused_relu2_quantize(
        input_tensor.contiguous(), sf_scale, 16
    )

    match_rate = (fp4_fused == fp4_separate).float().mean().item()
    assert match_rate >= 0.99, (
        f"FP4 values match rate {match_rate:.4f} < 0.99 for shape ({m}, {n}), dtype {dtype}"
    )


@skip_unless_fused_relu2_and_fp4_quantize
def test_fused_relu2_quantize_vs_separate_ops_various_sf_scales():
    """
    Test with various sf_scale values to ensure consistent behavior.
    """
    device = torch.device("cuda")
    m, n = 64, 128
    dtype = torch.bfloat16

    torch.manual_seed(123)
    input_tensor = torch.randn(m, n, dtype=dtype, device=device)
    activated = relu2(input_tensor)

    # Test with different sf_scale values
    for scale_multiplier in [0.1, 1.0, 10.0]:
        sf_scale = (
            (activated.abs().amax().float() / (6.0 * 448.0) * scale_multiplier).to(device).view(1)
        )
        fp4_separate, sf_separate = torch.ops.trtllm.fp4_quantize(
            activated, sf_scale, 16, False, True
        )
        fp4_fused, sf_fused = torch.ops.trtllm.fused_relu2_quantize(
            input_tensor.contiguous(), sf_scale, 16
        )

        match_rate = (fp4_fused == fp4_separate).float().mean().item()
        assert match_rate >= 0.99, (
            f"FP4 values match rate {match_rate:.4f} < 0.99 with scale_multiplier={scale_multiplier}"
        )
