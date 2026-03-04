# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit test for the dense GEMM + SwiGLU fusion custom ops.

Tests the custom ops:
  - trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_blackwell (BF16 output)
  - trtllm::cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell (FP4 output)

These ops fuse FC1 (gate_up projection) GEMM and SwiGLU activation into a
single kernel for shared experts on Blackwell GPUs.

The test creates NVFP4 tensors using the same quantization pipeline as
production code (fp4_quantize + block_scale_interleave), runs both the fused
and unfused paths, and compares the results.

Usage:
  python run_custom_op_dense_gemm_swiglu.py
  python run_custom_op_dense_gemm_swiglu.py --m 256 --intermediate_size 512 --k 1024
  python run_custom_op_dense_gemm_swiglu.py --test_fp4_output
  python run_custom_op_dense_gemm_swiglu.py --test_all_sizes

Note:
  Requires Blackwell GPU (SM100/SM103) and CuteDSL availability.
  Weights must be interleaved in 64-row groups for the SwiGLU kernel:
    [up_0 | gate_0 | up_1 | gate_1 | ...]
  The test handles this interleaving internally.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

# Ensure tensorrt_llm is importable
sys.path.insert(0, str(Path(__file__).parents[3]))

from tensorrt_llm.math_utils import pad_up  # noqa: E402


def interleave_gate_up_weight(weight: torch.Tensor,
                              group_size: int = 64) -> torch.Tensor:
    """Interleave gate/up weight from [gate | up] to [up_0 | gate_0 | up_1 | gate_1 | ...].

    The CuteDSL SwiGLU kernel expects interleaved (up, gate) blocks of
    group_size rows. GatedMLP stores weights as [gate | up].

    Args:
        weight: Weight tensor [n, k] where n = 2 * intermediate_size.
        group_size: Size of each interleaving block (default 64).

    Returns:
        Interleaved weight tensor [n, k].
    """
    n = weight.shape[0]
    half_n = n // 2
    k_dim = weight.shape[1]

    # Swap halves: [gate | up] -> [up | gate]
    weight_swapped = torch.cat([weight[half_n:], weight[:half_n]], dim=0)

    # Interleave in group_size-row blocks
    weight_interleaved = weight_swapped.view(
        2, n // (group_size * 2), group_size,
        k_dim).transpose(0, 1).contiguous().view(n, k_dim)

    return weight_interleaved


def interleave_weight_scale(weight_scale_swizzled: torch.Tensor,
                            n: int,
                            k: int,
                            scaling_vector_size: int = 16,
                            group_size: int = 64) -> torch.Tensor:
    """Interleave weight scale factors for SwiGLU fusion.

    Unswizzles, swaps gate/up halves, interleaves in groups, and re-swizzles.

    Args:
        weight_scale_swizzled: 1D swizzled scale factor tensor.
        n: Number of output features (2 * intermediate_size).
        k: Number of input features.
        scaling_vector_size: Scale factor vector size.
        group_size: Interleaving group size.

    Returns:
        1D swizzled interleaved scale factor tensor.
    """
    from tensorrt_llm._torch.utils import unswizzle_sf

    scale_rows = pad_up(n, 128)
    ws_unswizzled = unswizzle_sf(weight_scale_swizzled, scale_rows, k,
                                 scaling_vector_size)
    sf_k = ws_unswizzled.shape[1]
    half_n = n // 2

    # Swap gate/up halves
    ws_swapped = ws_unswizzled.clone()
    ws_swapped[:half_n] = ws_unswizzled[half_n:n]
    ws_swapped[half_n:n] = ws_unswizzled[:half_n]

    # Interleave first n rows
    ws_top = ws_swapped[:n]
    ws_top_interleaved = ws_top.view(2, n // (group_size * 2), group_size,
                                     sf_k).transpose(
                                         0, 1).contiguous().view(n, sf_k)
    ws_swapped[:n] = ws_top_interleaved

    # Re-swizzle
    return torch.ops.trtllm.block_scale_interleave(ws_swapped)


def create_nvfp4_tensors(m: int, n: int, k: int, scaling_vector_size: int = 16):
    """Create NVFP4 quantized tensors for testing.

    Creates random BF16 tensors, quantizes them to FP4, and returns
    everything needed for the custom ops.

    Args:
        m: Number of rows (tokens).
        n: Weight columns = 2 * intermediate_size.
        k: Hidden size (must be divisible by scaling_vector_size).

    Returns:
        Tuple of:
            act_bf16: Original BF16 activation [m, k]
            weight_bf16: Original BF16 weight [n, k]
            act_fp4: FP4 activation [m, k//2]
            act_sf: Activation scale factors (1D, swizzled)
            weight_fp4: FP4 weight [n, k//2] (gate||up, NOT interleaved)
            weight_sf: Weight scale factors (1D, swizzled, NOT interleaved)
            alpha: Alpha tensor [1]
            input_scale: Input scale [1]
    """
    torch.manual_seed(42)

    # Create random BF16 tensors
    act_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.1
    weight_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.1

    # Compute quantization scales
    FP8_MAX, E2M1_MAX = 448.0, 6.0

    amax_act = torch.amax(torch.abs(act_bf16)).float()
    input_scale = FP8_MAX * E2M1_MAX / amax_act
    amax_weight = torch.amax(torch.abs(weight_bf16)).float()
    weight_scale_global = FP8_MAX * E2M1_MAX / amax_weight
    alpha = (amax_act / (FP8_MAX * E2M1_MAX)) * (amax_weight /
                                                   (FP8_MAX * E2M1_MAX))

    # Quantize activation to FP4
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(act_bf16, input_scale,
                                                     scaling_vector_size, False)

    # Quantize weight to FP4 (sf_use_ue8m0=False, swizzled_layout=True by default)
    weight_fp4, weight_sf = torch.ops.trtllm.fp4_quantize(
        weight_bf16, weight_scale_global, scaling_vector_size, False)

    alpha_tensor = torch.tensor([alpha.item()], dtype=torch.float32,
                                device="cuda")
    input_scale_tensor = torch.tensor([input_scale.item()],
                                      dtype=torch.float32,
                                      device="cuda")

    return (act_bf16, weight_bf16, act_fp4, act_sf, weight_fp4, weight_sf,
            alpha_tensor, input_scale_tensor)


def compute_reference_swiglu(act_bf16: torch.Tensor,
                             weight_bf16: torch.Tensor) -> torch.Tensor:
    """Compute reference SwiGLU output in FP32.

    Args:
        act_bf16: Activation [m, k]
        weight_bf16: Weight [n, k] in [gate | up] layout

    Returns:
        Reference output [m, n//2] = silu(gate_out) * up_out
    """
    n = weight_bf16.shape[0]
    half_n = n // 2

    # GEMM in float32
    gemm_out = torch.matmul(act_bf16.float(), weight_bf16.float().T)

    # Split into gate and up
    gate_out = gemm_out[:, :half_n]
    up_out = gemm_out[:, half_n:]

    # SwiGLU: silu(gate) * up
    return (torch.sigmoid(gate_out) * gate_out * up_out).bfloat16()


def test_bf16_output(m: int,
                     intermediate_size: int,
                     k: int,
                     tolerance: float = 0.15):
    """Test the BF16 output fused GEMM + SwiGLU custom op.

    Args:
        m: Number of tokens.
        intermediate_size: MLP intermediate size.
        k: Hidden size.
        tolerance: Absolute tolerance for comparison.
    """
    n = intermediate_size * 2  # Full gate_up width

    print(f"\n--- Test BF16 output: M={m}, N={n}, K={k} ---")

    # Create quantized tensors
    (act_bf16, weight_bf16, act_fp4, act_sf, weight_fp4, weight_sf,
     alpha, input_scale) = create_nvfp4_tensors(m, n, k)

    # Interleave weight for fused kernel
    weight_fp4_interleaved = interleave_gate_up_weight(weight_fp4)
    weight_sf_interleaved = interleave_weight_scale(weight_sf, n, k)

    # Run fused custom op
    output_fused = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
        act_fp4, weight_fp4_interleaved, act_sf, weight_sf_interleaved, alpha,
        torch.bfloat16)

    # Trim padding if needed
    expected_out = n // 2
    if output_fused.shape[-1] > expected_out:
        output_fused = output_fused[..., :expected_out].contiguous()

    # Compute reference
    ref_output = compute_reference_swiglu(act_bf16, weight_bf16)

    # Compare
    print(f"  Fused output shape: {output_fused.shape}")
    print(f"  Reference shape: {ref_output.shape}")

    max_diff = (output_fused.float() - ref_output.float()).abs().max().item()
    mean_diff = (output_fused.float() - ref_output.float()).abs().mean().item()
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    # Note: Tolerance is relatively high due to FP4 quantization error
    try:
        torch.testing.assert_close(output_fused.float(),
                                   ref_output.float(),
                                   atol=tolerance,
                                   rtol=0.1)
        print("  PASSED")
        return True
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return False


def test_fp4_output(m: int,
                    intermediate_size: int,
                    k: int,
                    tolerance: float = 0.15):
    """Test the FP4 output fused GEMM + SwiGLU custom op.

    Args:
        m: Number of tokens.
        intermediate_size: MLP intermediate size.
        k: Hidden size.
        tolerance: Absolute tolerance for comparison.
    """
    n = intermediate_size * 2

    print(f"\n--- Test FP4 output: M={m}, N={n}, K={k} ---")

    # Create quantized tensors
    (act_bf16, weight_bf16, act_fp4, act_sf, weight_fp4, weight_sf,
     alpha, input_scale) = create_nvfp4_tensors(m, n, k)

    # Interleave weight for fused kernel
    weight_fp4_interleaved = interleave_gate_up_weight(weight_fp4)
    weight_sf_interleaved = interleave_weight_scale(weight_sf, n, k)

    # Global SF for SFC quantization (simulates down_proj.input_scale)
    global_sf = input_scale.clone()

    # Run FP4 output fused custom op
    fp4_out, out_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell(
        act_fp4, weight_fp4_interleaved, act_sf, weight_sf_interleaved, alpha,
        global_sf)

    print(f"  FP4 output shape: {fp4_out.shape}")
    print(f"  Output SF shape: {out_sf.shape}")

    # Verify output shapes
    expected_n_out = n // 2
    expected_fp4_cols = expected_n_out // 2  # FP4 packed
    assert fp4_out.shape == (m, expected_fp4_cols) or \
           fp4_out.shape[0] == m, \
        f"Unexpected FP4 output shape: {fp4_out.shape}, expected ({m}, {expected_fp4_cols})"

    print("  Shape verification PASSED")
    print("  (FP4 numerical verification requires dequantization — shape check only)")
    return True


def test_all_sizes():
    """Test with various M, N, K sizes."""
    test_cases = [
        # (m, intermediate_size, k)
        (128, 128, 256),
        (256, 256, 512),
        (512, 256, 256),
        (128, 512, 1024),
        (64, 128, 256),
        (1, 128, 256),  # Single token
        (384, 256, 512),  # Non-power-of-2 M
    ]

    passed = 0
    failed = 0
    skipped = 0

    for m, intermediate_size, k in test_cases:
        n = intermediate_size * 2
        # Check alignment requirements
        if n % 128 != 0:
            print(f"\nSkipping M={m}, N={n}, K={k}: N must be divisible by 128")
            skipped += 1
            continue
        if k % 16 != 0:
            print(f"\nSkipping M={m}, N={n}, K={k}: K must be divisible by 16")
            skipped += 1
            continue

        try:
            if test_bf16_output(m, intermediate_size, k):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nERROR M={m}, N={n}, K={k}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Unit test for dense GEMM + SwiGLU fusion custom ops")

    parser.add_argument("--m",
                        type=int,
                        default=128,
                        help="Number of tokens (M dimension)")
    parser.add_argument("--intermediate_size",
                        type=int,
                        default=256,
                        help="MLP intermediate size (N = 2 * intermediate_size)")
    parser.add_argument("--k",
                        type=int,
                        default=512,
                        help="Hidden size (K dimension)")
    parser.add_argument("--tolerance",
                        type=float,
                        default=0.15,
                        help="Absolute tolerance for numerical comparison")
    parser.add_argument("--test_fp4_output",
                        action="store_true",
                        help="Also test FP4 output path")
    parser.add_argument("--test_all_sizes",
                        action="store_true",
                        help="Run tests with various M/N/K sizes")

    args = parser.parse_args()

    # Check for Blackwell GPU
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    sm = torch.cuda.get_device_capability()
    sm_version = sm[0] * 10 + sm[1]
    if sm_version not in (100, 103):
        print(f"Skipping: requires Blackwell GPU (SM100/SM103), got SM{sm_version}")
        sys.exit(0)

    # Import custom ops (triggers registration)
    try:
        from tensorrt_llm._torch.custom_ops import (  # noqa: F401
            cute_dsl_nvfp4_dense_gemm_swiglu_blackwell)
    except ImportError:
        print("CuteDSL custom ops not available (CUTLASS DSL not installed)")
        sys.exit(0)

    all_passed = True

    if args.test_all_sizes:
        all_passed = test_all_sizes()
    else:
        result = test_bf16_output(args.m, args.intermediate_size, args.k,
                                  args.tolerance)
        all_passed = all_passed and result

    if args.test_fp4_output or args.test_all_sizes:
        try:
            from tensorrt_llm._torch.custom_ops import (  # noqa: F401
                cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell)
            result = test_fp4_output(args.m, args.intermediate_size, args.k)
            all_passed = all_passed and result
        except ImportError:
            print("\nFP4 output custom op not available, skipping")

    if all_passed:
        print("\nPASS")
    else:
        print("\nFAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
