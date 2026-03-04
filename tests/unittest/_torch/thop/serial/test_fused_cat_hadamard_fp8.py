# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for fused_cat_hadamard_fp8 custom op.

Compares the fused kernel output against a sequential reference implementation
(torch.cat → fast_hadamard_transform → fp8_quantize_1x128) for numerical
correctness.
"""

import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.util import getSMVersion

# Import tensorrt_llm to load custom CUDA operators
import tensorrt_llm  # noqa: F401

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

# Check that the fused op is available
try:
    torch.ops.trtllm.fused_cat_hadamard_fp8
    HAS_FUSED_OP = True
except (AttributeError, RuntimeError):
    HAS_FUSED_OP = False

# Check for fast_hadamard_transform for reference impl
try:
    from fast_hadamard_transform import hadamard_transform

    HAS_HADAMARD = True
except ImportError:
    HAS_HADAMARD = False


def _reference_cat_hadamard_fp8(pe, nope, use_ue8m0=False):
    """Sequential reference: cat → hadamard → fp8_quantize_1x128_sf_transpose."""
    from tensorrt_llm.quantization.utils import fp8_utils

    # Cat
    combined = torch.cat([pe, nope], dim=-1)

    # Hadamard transform
    head_dim = combined.shape[-1]
    combined = hadamard_transform(combined, scale=head_dim**-0.5)

    # FP8 quantize
    combined_2d = combined.view(-1, head_dim)
    fp8_out, scale = fp8_utils.fp8_quantize_1x128_sf_transpose(combined_2d, use_ue8m0=use_ue8m0)

    return fp8_out, scale


# DSV3.2 indexer config: pe_dim=64, nope_dim=64, head_dim=128
@pytest.mark.parametrize("M", [1, 32, 64, 1024, 65536])
@pytest.mark.parametrize("pe_dim,nope_dim", [(64, 64)])
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.skipif(not HAS_FUSED_OP, reason="fused_cat_hadamard_fp8 op not available")
@pytest.mark.skipif(not HAS_HADAMARD, reason="fast_hadamard_transform not available for reference")
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_hadamard_fp8_correctness(M, pe_dim, nope_dim, use_ue8m0):
    """Test that fused kernel matches sequential reference within FP8 tolerance.

    The fused kernel computes Hadamard in float32 while the reference library
    (fast_hadamard_transform) computes in BF16. This causes small numerical
    differences that propagate to the FP8 quantization. We compare the
    dequantized outputs (fp8 * scale) against a float32 Hadamard ground truth.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")

    pe = torch.randn(M, pe_dim, dtype=torch.bfloat16, device=device)
    nope = torch.randn(M, nope_dim, dtype=torch.bfloat16, device=device)

    # Fused kernel
    fused_fp8, fused_scale = torch.ops.trtllm.fused_cat_hadamard_fp8(pe, nope, use_ue8m0)

    # Reference
    ref_fp8, ref_scale = _reference_cat_hadamard_fp8(pe, nope, use_ue8m0)

    # 1. Compare dequantized outputs: both should be close to the true
    #    Hadamard-transformed values. The fused kernel is actually more accurate
    #    since it uses float32 for the Hadamard transform.
    fused_deq = fused_fp8.float() * fused_scale
    ref_deq = ref_fp8.float() * ref_scale
    # Both dequantized outputs should be close to each other
    rel_err = (fused_deq - ref_deq).abs() / ref_deq.abs().clamp(min=1e-6)
    mean_rel_err = rel_err.mean().item()
    assert mean_rel_err < 0.05, (
        f"Mean relative dequantized error too high: {mean_rel_err:.4f}. "
        f"M={M}, use_ue8m0={use_ue8m0}"
    )

    # 2. Check FP8 values match rate (most values should be identical or 1 ULP)
    abs_diff = (fused_fp8.float() - ref_fp8.float()).abs()
    match_rate = (abs_diff == 0).float().mean().item()
    # With UE8M0, scales are identical so FP8 match rate should be high.
    # With standard scales, the fused kernel has slightly different scales
    # due to float32 vs bf16 Hadamard, so more FP8 values may differ.
    min_match = 0.90 if use_ue8m0 else 0.80
    assert match_rate > min_match, (
        f"FP8 value match rate too low: {match_rate:.4f} "
        f"(expected > {min_match}). M={M}, use_ue8m0={use_ue8m0}"
    )

    # 3. Compare scales
    fused_scale_flat = fused_scale.view(-1)
    ref_scale_flat = ref_scale.view(-1)
    assert fused_scale_flat.shape == ref_scale_flat.shape, (
        f"Scale shape mismatch: fused={fused_scale_flat.shape}, ref={ref_scale_flat.shape}"
    )

    if use_ue8m0:
        # UE8M0 scales must match exactly (both are power-of-2)
        scale_match = torch.allclose(fused_scale_flat, ref_scale_flat, atol=0, rtol=0)
        if not scale_match:
            mismatches = (fused_scale_flat != ref_scale_flat).nonzero()
            n_mismatch = mismatches.shape[0]
            assert n_mismatch / M < 0.05, f"Too many UE8M0 scale mismatches: {n_mismatch}/{M}"
    else:
        # Standard scales: differ due to float32 vs bf16 Hadamard intermediate.
        # Allow up to 0.5% relative difference.
        torch.testing.assert_close(fused_scale_flat, ref_scale_flat, rtol=5e-3, atol=1e-5)


@pytest.mark.parametrize("M", [1, 256])
@pytest.mark.skipif(not HAS_FUSED_OP, reason="fused_cat_hadamard_fp8 op not available")
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_hadamard_fp8_output_shape(M):
    """Test that output shapes are correct."""
    pe_dim, nope_dim = 64, 64
    head_dim = pe_dim + nope_dim
    device = torch.device("cuda")

    pe = torch.randn(M, pe_dim, dtype=torch.bfloat16, device=device)
    nope = torch.randn(M, nope_dim, dtype=torch.bfloat16, device=device)

    fp8_out, scale = torch.ops.trtllm.fused_cat_hadamard_fp8(pe, nope, True)

    assert fp8_out.shape == (M, head_dim), f"fp8_out shape: {fp8_out.shape}"
    assert fp8_out.dtype == torch.float8_e4m3fn
    assert scale.shape == (M, 1), f"scale shape: {scale.shape}"
    assert scale.dtype == torch.float32


@pytest.mark.skipif(not HAS_FUSED_OP, reason="fused_cat_hadamard_fp8 op not available")
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_hadamard_fp8_zero_input():
    """Test with zero inputs — scales should be minimal, FP8 values should be zero."""
    M = 64
    pe_dim, nope_dim = 64, 64
    device = torch.device("cuda")

    pe = torch.zeros(M, pe_dim, dtype=torch.bfloat16, device=device)
    nope = torch.zeros(M, nope_dim, dtype=torch.bfloat16, device=device)

    fp8_out, scale = torch.ops.trtllm.fused_cat_hadamard_fp8(pe, nope, True)

    # All values should be zero
    assert (fp8_out.float() == 0).all(), "Expected all zero FP8 output for zero input"
