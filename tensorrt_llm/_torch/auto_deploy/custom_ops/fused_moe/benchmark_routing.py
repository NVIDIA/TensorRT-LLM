#!/usr/bin/env python3
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

"""Benchmark: Fused Triton top-k + softmax routing vs. baseline PyTorch.

Compares the original 3-op MoE routing pattern used in Qwen3.5
    (softmax -> topk -> renormalize)
against the fused Triton kernel that exploits the equivalence
    topk(logits) -> softmax(topk_logits)

Usage (standalone, avoids heavy tensorrt_llm imports):
    python tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/benchmark_routing.py
"""

import os
import sys

import torch
import torch.nn.functional as F

# Allow running as a standalone script without triggering the full
# tensorrt_llm import chain.  We import only the triton_routing module.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from triton_routing import triton_fused_topk_softmax_fn  # noqa: E402

# ============================================================================
# Baseline: 3-op PyTorch implementation (softmax -> topk -> renormalize)
# ============================================================================


def baseline_routing(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original Qwen3.5 MoE routing: softmax -> topk -> renormalize."""
    routing_weights = F.softmax(router_logits, dtype=torch.float, dim=-1)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights, selected_experts


# ============================================================================
# Correctness check
# ============================================================================


def check_correctness(
    router_logits: torch.Tensor,
    top_k: int,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Verify that fused and baseline produce identical results."""
    ref_weights, ref_indices = baseline_routing(router_logits, top_k)
    fused_weights, fused_indices = triton_fused_topk_softmax_fn(router_logits, top_k)

    # Sort both by expert index within each token so order doesn't matter
    ref_sort = ref_indices.sort(dim=-1)
    fused_sort = fused_indices.sort(dim=-1)

    ref_weights_sorted = ref_weights.gather(-1, ref_sort.indices)
    fused_weights_sorted = fused_weights.gather(-1, fused_sort.indices)

    indices_match = torch.equal(ref_sort.values.to(torch.int32), fused_sort.values)
    weights_close = torch.allclose(ref_weights_sorted, fused_weights_sorted, atol=atol, rtol=rtol)

    if not indices_match:
        mismatched = (ref_sort.values.to(torch.int32) != fused_sort.values).sum().item()
        total = ref_sort.values.numel()
        print(f"  WARNING: Index mismatch in {mismatched}/{total} elements")
    if not weights_close:
        max_diff = (ref_weights_sorted - fused_weights_sorted).abs().max().item()
        print(f"  WARNING: Weight mismatch, max diff = {max_diff:.6e}")

    return indices_match and weights_close


# ============================================================================
# Timing utilities
# ============================================================================


def benchmark_fn(
    fn,
    *args,
    warmup: int = 50,
    iters: int = 200,
) -> float:
    """Benchmark a GPU function, return median time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed iterations using CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn(*args)
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    # Return median in microseconds
    median_ms = times_ms[len(times_ms) // 2]
    return median_ms * 1000.0


# ============================================================================
# Main benchmark
# ============================================================================


def run_benchmark():
    device = torch.device("cuda")

    # Qwen3.5 MoE parameters
    num_experts = 256
    top_k = 8

    token_counts = [1, 32, 128, 512, 1024, 4096]
    dtypes = [torch.bfloat16, torch.float16]

    print("=" * 80)
    print("MoE Routing Kernel Benchmark: Baseline (3-op) vs Fused Triton")
    print(f"  num_experts = {num_experts}, top_k = {top_k}")
    print("=" * 80)

    # Run correctness check first
    print("\n--- Correctness Checks ---")
    all_correct = True
    for dtype in dtypes:
        for num_tokens in token_counts:
            router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
            dtype_str = str(dtype).replace("torch.", "")
            passed = check_correctness(router_logits, top_k)
            status = "PASS" if passed else "FAIL"
            print(f"  {dtype_str:>8s}  tokens={num_tokens:<6d}  {status}")
            all_correct = all_correct and passed

    if not all_correct:
        print("\nWARNING: Some correctness checks failed. See details above.")
    else:
        print("\nAll correctness checks passed.")

    # Performance benchmark
    print("\n--- Performance ---")
    header = (
        f"{'dtype':>8s}  {'tokens':>8s}  {'baseline(us)':>14s}  {'fused(us)':>14s}  {'speedup':>8s}"
    )
    print(header)
    print("-" * len(header))

    for dtype in dtypes:
        for num_tokens in token_counts:
            router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)

            # Benchmark baseline
            t_baseline = benchmark_fn(baseline_routing, router_logits, top_k)

            # Benchmark fused Triton kernel
            t_fused = benchmark_fn(triton_fused_topk_softmax_fn, router_logits, top_k)

            speedup = t_baseline / t_fused if t_fused > 0 else float("inf")
            dtype_str = str(dtype).replace("torch.", "")
            print(
                f"{dtype_str:>8s}  {num_tokens:>8d}  {t_baseline:>14.2f}  "
                f"{t_fused:>14.2f}  {speedup:>7.2f}x"
            )

    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
