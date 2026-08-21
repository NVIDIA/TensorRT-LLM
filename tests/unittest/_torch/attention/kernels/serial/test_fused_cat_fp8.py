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
"""Unit tests for fused_cat_fp8 custom op.

Compares the fused kernel output against a sequential reference implementation
(torch.cat → fp8_quantize_1x128) for numerical correctness.

Note on tolerances: the fused kernel and the reference (fp8_quantize_1x128)
compute amax at different precisions — the fused kernel uses fp32 throughout,
while the reference truncates amax to bf16 before the warp reduction. This
causes scale differences of up to ~0.8% (bf16 mantissa precision), which in
turn causes a fraction of FP8 values to differ by 1 ULP. Tests therefore
compare dequantized outputs (fp8 * scale) with tolerances appropriate for
FP8 E4M3 quantization noise.
"""

import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.util import getSMVersion

# Import tensorrt_llm to load custom CUDA operators
import tensorrt_llm  # noqa: F401


def _reference_cat_fp8(pe, nope, use_ue8m0=False):
    """Sequential reference: cat → fp8_quantize_1x128_sf_transpose."""
    from tensorrt_llm.quantization.utils import fp8_utils

    # Cat
    combined = torch.cat([pe, nope], dim=-1)

    # FP8 quantize
    head_dim = combined.shape[-1]
    combined_2d = combined.view(-1, head_dim)
    fp8_out, scale = fp8_utils.fp8_quantize_1x128_sf_transpose(combined_2d, use_ue8m0=use_ue8m0)

    return fp8_out, scale


def _assert_fp8_close(fused_fp8, fused_scale, ref_fp8, ref_scale, label="", use_ue8m0=False):
    """Assert that two FP8-quantized tensors represent similar values.

    Compares dequantized outputs (fp8 * scale) using tolerances appropriate
    for FP8 E4M3 quantization noise. FP8 E4M3 has 3 mantissa bits, so
    quantization noise is ~6.25% relative. With different amax precision
    (fp32 vs bf16) between the two kernels, we allow slightly more.
    """
    prefix = f"{label}: " if label else ""

    # Shape sanity
    assert fused_fp8.shape == ref_fp8.shape, (
        f"{prefix}FP8 shape mismatch: fused={fused_fp8.shape}, ref={ref_fp8.shape}"
    )
    fused_scale_flat = fused_scale.view(-1)
    ref_scale_flat = ref_scale.view(-1)
    assert fused_scale_flat.shape == ref_scale_flat.shape, (
        f"{prefix}Scale shape mismatch: fused={fused_scale_flat.shape}, ref={ref_scale_flat.shape}"
    )

    # Scales: For UE8M0, scales are powers of 2 so they either match exactly
    # or differ by 2x at boundary cases — rtol=1.0 covers the worst case.
    # For non-UE8M0, scales are continuous floats — allow ~2% from bf16 vs
    # fp32 amax precision difference.
    scale_rtol = 1.0 if use_ue8m0 else 0.02
    torch.testing.assert_close(
        fused_scale_flat,
        ref_scale_flat,
        rtol=scale_rtol,
        atol=1e-6,
        msg=lambda msg: f"{prefix}Scale mismatch: {msg}",
    )

    # Dequantized values: the ground truth comparison. Both implementations
    # should produce similar dequantized outputs despite different internals.
    fused_deq = fused_fp8.float() * fused_scale
    ref_deq = ref_fp8.float() * ref_scale

    # Use allclose with tolerances for FP8 quantization noise.
    # FP8 E4M3 has ~6.25% quantization noise; we allow 10% to account for
    # different amax precision between the two kernels.
    abs_err = (fused_deq - ref_deq).abs()
    magnitude = ref_deq.abs().clamp(min=1e-6)
    rel_err = abs_err / magnitude

    mean_rel_err = rel_err.mean().item()
    assert mean_rel_err < 0.1, (
        f"{prefix}Mean relative dequantized error too high: {mean_rel_err:.4f} (expected < 0.1)"
    )

    # Most values should be very close (within 2% relative)
    close_rate = (rel_err < 0.02).float().mean().item()
    assert close_rate > 0.90, (
        f"{prefix}Only {close_rate:.2%} of values within 2% relative error (expected > 90%)"
    )


# DSV3.2 indexer config: pe_dim=64, nope_dim=64, head_dim=128
@pytest.mark.parametrize("M", [1, 3, 7, 9, 32, 64, 1024, 65536])
@pytest.mark.parametrize("pe_dim,nope_dim", [(64, 64)])
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_fp8_correctness(M, pe_dim, nope_dim, use_ue8m0):
    """Test that fused kernel matches sequential reference (cat + fp8 quantize)."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    pe = torch.randn(M, pe_dim, dtype=torch.bfloat16, device=device)
    nope = torch.randn(M, nope_dim, dtype=torch.bfloat16, device=device)

    # Fused kernel
    fused_fp8, fused_scale = torch.ops.trtllm.fused_cat_fp8(pe, nope, use_ue8m0)

    # Reference
    ref_fp8, ref_scale = _reference_cat_fp8(pe, nope, use_ue8m0)

    _assert_fp8_close(
        fused_fp8,
        fused_scale,
        ref_fp8,
        ref_scale,
        label=f"M={M}, use_ue8m0={use_ue8m0}",
        use_ue8m0=use_ue8m0,
    )


@pytest.mark.parametrize("M", [1, 256])
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_fp8_output_shape(M):
    """Test that output shapes are correct."""
    pe_dim, nope_dim = 64, 64
    head_dim = pe_dim + nope_dim
    device = torch.device("cuda")

    pe = torch.randn(M, pe_dim, dtype=torch.bfloat16, device=device)
    nope = torch.randn(M, nope_dim, dtype=torch.bfloat16, device=device)

    fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(pe, nope, True)

    assert fp8_out.shape == (M, head_dim), f"fp8_out shape: {fp8_out.shape}"
    assert fp8_out.dtype == torch.float8_e4m3fn
    assert scale.shape == (M, 1), f"scale shape: {scale.shape}"
    assert scale.dtype == torch.float32


@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_fp8_zero_input():
    """Test with zero inputs — scales should be minimal, FP8 values should be zero."""
    M = 64
    pe_dim, nope_dim = 64, 64
    device = torch.device("cuda")

    pe = torch.zeros(M, pe_dim, dtype=torch.bfloat16, device=device)
    nope = torch.zeros(M, nope_dim, dtype=torch.bfloat16, device=device)

    fp8_out, scale = torch.ops.trtllm.fused_cat_fp8(pe, nope, True)

    # All values should be zero
    assert (fp8_out.float() == 0).all(), "Expected all zero FP8 output for zero input"


@pytest.mark.parametrize("M", [64, 1024])
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_fp8_noncontiguous_input(M, use_ue8m0):
    """Test with non-contiguous inputs (simulates torch.split in DSA indexer).

    In the real DSA forward path, pe/nope come from torch.split() on a
    [M, head_dim] tensor, producing non-contiguous views with stride
    [head_dim, 1] instead of [pe_dim, 1]. The thop must handle this.
    """
    pe_dim, nope_dim = 64, 64
    head_dim = pe_dim + nope_dim
    device = torch.device("cuda")

    torch.manual_seed(42)
    combined = torch.randn(M, head_dim, dtype=torch.bfloat16, device=device)
    pe, nope = combined.split([pe_dim, nope_dim], dim=-1)

    # Verify inputs are indeed non-contiguous
    assert not pe.is_contiguous(), "pe should be non-contiguous from split"
    assert not nope.is_contiguous(), "nope should be non-contiguous from split"

    # Fused kernel should handle non-contiguous inputs
    fused_fp8, fused_scale = torch.ops.trtllm.fused_cat_fp8(pe, nope, use_ue8m0)

    # Reference with contiguous copies
    ref_fp8, ref_scale = _reference_cat_fp8(pe.contiguous(), nope.contiguous(), use_ue8m0=use_ue8m0)

    _assert_fp8_close(
        fused_fp8,
        fused_scale,
        ref_fp8,
        ref_scale,
        label=f"Non-contiguous M={M}",
        use_ue8m0=use_ue8m0,
    )


@pytest.mark.parametrize("M", [1, 16, 64])
@pytest.mark.parametrize("n_heads", [64])
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_fp8_3d_input(M, n_heads, use_ue8m0):
    """Test with 3D inputs matching Q path: [M, n_heads, dim] from split.

    In the DSA indexer, Q tensors are [M, n_heads, head_dim] split into
    [M, n_heads, pe_dim] and [M, n_heads, nope_dim]. The kernel should handle
    these 3D non-contiguous views directly without needing reshape.
    """
    pe_dim, nope_dim = 64, 64
    head_dim = pe_dim + nope_dim
    device = torch.device("cuda")

    torch.manual_seed(42)
    # Simulate q.view(-1, n_heads, head_dim).split([pe_dim, nope_dim], dim=-1)
    q = torch.randn(M, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    pe, nope = q.split([pe_dim, nope_dim], dim=-1)

    assert pe.shape == (M, n_heads, pe_dim)
    assert nope.shape == (M, n_heads, nope_dim)
    assert not nope.is_contiguous()

    # Fused kernel with 3D input (no reshape)
    fused_fp8, fused_scale = torch.ops.trtllm.fused_cat_fp8(pe, nope, use_ue8m0)

    # Expected M for kernel: M * n_heads
    total_rows = M * n_heads
    assert fused_fp8.shape == (total_rows, head_dim)
    assert fused_scale.shape == (total_rows, 1)

    # Reference with explicit reshape (old behavior)
    ref_fp8, ref_scale = _reference_cat_fp8(
        pe.reshape(-1, pe_dim), nope.reshape(-1, nope_dim), use_ue8m0=use_ue8m0
    )

    _assert_fp8_close(
        fused_fp8,
        fused_scale,
        ref_fp8,
        ref_scale,
        label=f"3D M={M}, n_heads={n_heads}",
        use_ue8m0=use_ue8m0,
    )


@pytest.mark.parametrize("M", [1, 16])
@pytest.mark.parametrize("use_ue8m0", [True, False])
@pytest.mark.skipif(getSMVersion() < 90, reason="Requires SM >= 90")
def test_fused_cat_fp8_mixed_contiguity(M, use_ue8m0):
    """Test where pe is contiguous but nope is non-contiguous.

    This matches the non-flashinfer Q path where pe comes from rotary_emb
    (contiguous) but nope remains the non-contiguous split view.
    """
    pe_dim, nope_dim, n_heads = 64, 64, 64
    head_dim = pe_dim + nope_dim
    device = torch.device("cuda")

    torch.manual_seed(42)
    # pe is contiguous (simulates post-RoPE output)
    pe = torch.randn(M, n_heads, pe_dim, dtype=torch.bfloat16, device=device)
    # nope is non-contiguous (from split on [M, n_heads, head_dim])
    full = torch.randn(M, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    _, nope = full.split([pe_dim, nope_dim], dim=-1)

    assert pe.is_contiguous()
    assert not nope.is_contiguous()

    fused_fp8, fused_scale = torch.ops.trtllm.fused_cat_fp8(pe, nope, use_ue8m0)
    ref_fp8, ref_scale = _reference_cat_fp8(
        pe.reshape(-1, pe_dim), nope.reshape(-1, nope_dim), use_ue8m0=use_ue8m0
    )

    _assert_fp8_close(
        fused_fp8,
        fused_scale,
        ref_fp8,
        ref_scale,
        label=f"Mixed contiguity M={M}",
        use_ue8m0=use_ue8m0,
    )
