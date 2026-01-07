# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the CuTE DSL argmax kernel.

The kernel uses CuTE for N >= 256 (aligned to 32), otherwise falls back to torch.max.
Only float32 uses the CuTE kernel; float16/bfloat16 use torch.max fallback.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.argmax import argmax

# Increase dynamo cache for parameterized tests
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024

# ============================================================================
# Constants for test configurations
# ============================================================================
# N values where CuTE kernel is used (N >= 256, aligned to 32)
LARGE_N_VALUES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 201088, 262144]

# N values that use torch.max fallback (N < 256)
SMALL_N_VALUES = [8, 16, 32, 64, 128]

# Typical LLM vocab sizes for performance testing
VOCAB_SIZES = [32000, 32768, 65536, 128256, 131072, 201088, 262144]


# ============================================================================
# Correctness Tests
# ============================================================================
@pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize("N", LARGE_N_VALUES)
@pytest.mark.parametrize("M", [1, 4, 37, 199, 1024])
def test_argmax_large_n(M, N, input_dtype):
    """Test argmax with CuTE kernel (N >= 256, aligned)."""
    device = "cuda"
    atol, rtol = 1e-4, 1e-4

    torch.random.manual_seed(42)
    x = 0.1 * torch.randn(M, N, device=device, dtype=input_dtype)

    out = argmax(x)
    expected_max, expected_idx = torch.max(x, dim=-1, keepdim=True)

    assert out.shape == (M, 2)
    assert out.dtype == input_dtype

    torch.testing.assert_close(out[:, 0:1], expected_max, atol=atol, rtol=rtol)
    torch.testing.assert_close(out[:, 1:2].long(), expected_idx, atol=0, rtol=0)


@pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize("N", SMALL_N_VALUES)
def test_argmax_small_n(N, input_dtype):
    """Test argmax with torch.max fallback (N < 256)."""
    device = "cuda"
    M = 4

    for max_pos in [0, N // 4, N // 2, 3 * N // 4, N - 1]:
        x = torch.full((M, N), -100.0, dtype=input_dtype, device=device)
        x[:, max_pos] = 0.0

        out = argmax(x)

        for row in range(M):
            assert out[row, 0].item() == 0.0, f"Row {row}: max value should be 0.0"
            assert out[row, 1].item() == float(max_pos), f"Row {row}: argmax wrong"


@pytest.mark.parametrize("input_dtype", [torch.float32])
def test_argmax_mtp_case(input_dtype):
    """Test the specific MTP test case (N=8, max at index 1)."""
    x = torch.tensor(
        [[-100.0, 0.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]],
        dtype=input_dtype,
        device="cuda",
    )
    out = argmax(x)

    assert out.shape == (1, 2)
    assert out[0, 0].item() == 0.0, f"Expected max=0, got {out[0, 0].item()}"
    assert out[0, 1].item() == 1.0, f"Expected argmax=1, got {out[0, 1].item()}"


@pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize("N", [255, 256, 257, 1023, 1024, 1025, 32000, 32001])
def test_argmax_alignment_fallback(N, input_dtype):
    """Test aligned vs unaligned N values (unaligned falls back to torch.max)."""
    device = "cuda"
    M = 4

    torch.random.manual_seed(42)
    x = torch.randn(M, N, device=device, dtype=input_dtype)

    out = argmax(x)
    expected_max, expected_idx = torch.max(x, dim=-1, keepdim=True)

    torch.testing.assert_close(out[:, 0:1], expected_max, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[:, 1:2].long(), expected_idx, atol=0, rtol=0)


# ============================================================================
# CUDA Graph Tests
# ============================================================================
@pytest.mark.parametrize("input_dtype", [torch.float32])
@pytest.mark.parametrize("N", [1024, 32768, 131072])
@pytest.mark.parametrize("M", [1, 16, 256])
def test_argmax_cudagraphs(M, N, input_dtype):
    """Test that argmax is CUDA graph capturable."""
    device = "cuda"

    torch.random.manual_seed(0)
    x = 0.1 * torch.randn(M, N, device=device, dtype=input_dtype)

    # Warmup
    _ = argmax(x)
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = argmax(x)

    graph.replay()
    torch.cuda.synchronize()

    # Verify
    expected_max, expected_idx = torch.max(x, dim=-1, keepdim=True)
    torch.testing.assert_close(out[:, 0:1], expected_max, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[:, 1:2].long(), expected_idx, atol=0, rtol=0)


# ============================================================================
# Performance Tests
# ============================================================================
@pytest.mark.parametrize("N", VOCAB_SIZES)
def test_argmax_performance(N):
    """Compare CuTE argmax vs torch.max performance."""
    device = "cuda"
    dtype = torch.float32
    num_iters = 100
    M_values = [4, 16, 64, 256, 1024]

    print(f"\n{'=' * 70}")
    print(f"N={N:>6} | {'M':>5} | {'CuTE':>10} | {'torch':>10} | {'Speedup':>8}")
    print(f"{'-' * 70}")

    for M in M_values:
        torch.random.manual_seed(0)
        x = 0.1 * torch.randn(M, N, device=device, dtype=dtype)

        # Warmup
        _ = argmax(x)
        _ = torch.max(x, dim=-1)
        torch.cuda.synchronize()

        # CuTE graph
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            for _ in range(num_iters):
                out = argmax(x)

        # torch.max graph
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            for _ in range(num_iters):
                _, _ = torch.max(x, dim=-1)

        # Time CuTE
        torch.cuda.synchronize()
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t1.record()
        g1.replay()
        t2.record()
        torch.cuda.synchronize()
        cute_ms = t1.elapsed_time(t2) / num_iters

        # Time torch.max
        t1.record()
        g2.replay()
        t2.record()
        torch.cuda.synchronize()
        torch_ms = t1.elapsed_time(t2) / num_iters

        speedup = torch_ms / cute_ms if cute_ms > 0 else float("inf")
        status = "✓" if speedup > 1 else "✗"

        print(
            f"       | {M:>5} | {cute_ms:>8.4f}ms | {torch_ms:>8.4f}ms | {speedup:>5.2f}x {status}"
        )

    print(f"{'=' * 70}")

    # Verify correctness
    expected_max, expected_idx = torch.max(x, dim=-1, keepdim=True)
    torch.testing.assert_close(out[:, 0:1], expected_max, atol=1e-4, rtol=1e-4)


# ============================================================================
# Manual Test Runner
# ============================================================================
if __name__ == "__main__":
    print("Running argmax tests...\n")

    print("1. MTP case (N=8)...")
    test_argmax_mtp_case(torch.float32)
    print("   ✓ Passed\n")

    print("2. Small N (torch.max fallback)...")
    for N in SMALL_N_VALUES:
        test_argmax_small_n(N, torch.float32)
    print("   ✓ Passed\n")

    print("3. Large N (CuTE kernel)...")
    for N in [256, 1024, 32768, 131072]:
        test_argmax_large_n(4, N, torch.float32)
    print("   ✓ Passed\n")

    print("4. Alignment fallback...")
    for N in [255, 256, 257, 1023, 1024, 1025]:
        test_argmax_alignment_fallback(N, torch.float32)
    print("   ✓ Passed\n")

    print("5. CUDA graphs...")
    test_argmax_cudagraphs(16, 32768, torch.float32)
    print("   ✓ Passed\n")

    print("6. Performance comparison...")
    for N in [1024, 4096, 16384, 32768, 65536, 131072, 201088, 262144]:
        test_argmax_performance(N)

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
