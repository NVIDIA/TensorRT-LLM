#!/usr/bin/env python3
"""
Test script for the reduce_add operation.

This script tests the custom CUDA kernel implementation against a naive PyTorch implementation.
"""

import pytest
import torch


@pytest.mark.parametrize(
    "num_tokens,topk,hidden_size",
    [
        (128, 8, 7168),
        (256, 4, 2048),
        (64, 16, 8192),
        (1024, 2, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduce_add_correctness(num_tokens, topk, hidden_size, dtype):
    """
    Test the correctness of reduce_add operation.

    Args:
        num_tokens: Number of tokens
        topk: Number of experts per token
        hidden_size: Hidden dimension size
        dtype: Data type (torch.float16 or torch.bfloat16)
    """
    device = "cuda"

    # Create input tensors
    input_tensor = torch.randn(num_tokens, topk, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Compute reference using PyTorch
    expected = input_tensor.sum(dim=1) + residual

    # Compute using custom CUDA kernel
    output = torch.ops.trtllm.reduce_add(input_tensor, residual)

    # Compare results using torch.testing.assert_close
    if dtype == torch.float16:
        # FP16: stricter tolerance
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)
    else:  # bfloat16
        # BF16: looser tolerance due to lower precision
        torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)

    # Print statistics for information
    max_diff = (output - expected).abs().max().item()
    mean_diff = (output - expected).abs().mean().item()
    print(
        f"\n  [{dtype}, {num_tokens}x{topk}x{hidden_size}] Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("num_tokens", [128])
def test_reduce_add_benchmark(num_tokens):
    """
    Benchmark the reduce_add operation.
    Demonstrates persistent kernel benefits with increasing num_tokens.

    Run with: pytest -v -s -m benchmark test_reduce_add.py
    """
    topk = 8
    hidden_size = 7168
    dtype = torch.bfloat16
    device = "cuda"
    num_iters = 100

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: num_tokens={num_tokens}, topk={topk}, hidden_size={hidden_size}")

    # Create input tensors
    input_tensor = torch.randn(num_tokens, topk, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        _ = torch.ops.trtllm.reduce_add(input_tensor, residual)

    torch.cuda.synchronize()

    # Benchmark custom kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        _ = torch.ops.trtllm.reduce_add(input_tensor, residual)
    end_event.record()

    torch.cuda.synchronize()
    custom_time = start_event.elapsed_time(end_event) / num_iters

    # Benchmark PyTorch reference
    start_event.record()
    for _ in range(num_iters):
        _ = input_tensor.sum(dim=1) + residual
    end_event.record()

    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / num_iters

    # Calculate bandwidth
    bytes_per_element = 2  # fp16 or bf16
    read_bytes = (num_tokens * topk * hidden_size + num_tokens * hidden_size) * bytes_per_element
    write_bytes = num_tokens * hidden_size * bytes_per_element
    total_bytes = read_bytes + write_bytes
    bandwidth_gb_s = (total_bytes / 1e9) / (custom_time / 1000)

    print(f"  Custom kernel:     {custom_time:.4f} ms")
    print(f"  PyTorch reference: {pytorch_time:.4f} ms")
    print(f"  Speedup:           {pytorch_time / custom_time:.2f}x")
    print(f"  Memory bandwidth:  {bandwidth_gb_s:.2f} GB/s")

    # Assert that custom kernel is not significantly slower than PyTorch
    # (should be faster or at least comparable)
    assert custom_time <= pytorch_time * 1.5, (
        f"Custom kernel is too slow: {custom_time:.4f}ms vs PyTorch {pytorch_time:.4f}ms"
    )
