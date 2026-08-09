# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 router projection: the GEMV against the cuBLAS path it replaces.

MiniMax-M3's gate is ``[num_tokens, 6144] x [6144, 128]`` with an FP32 weight.
cuBLAS answers that with a split-K TF32 GEMM plus a ``splitKreduce``, and wants
the BF16 hidden states cast to FP32 first, so the baseline here is the whole
three-kernel sequence rather than the GEMM alone.

Many calls go into one CUDA graph: a single-call graph measures the ~6us host
cost of ``graph.replay()`` instead of the kernel. The reported floor is the same
harness timing a trivial kernel, so a result near it is not resolving anything.

    python tests/microbenchmarks/bench_fp32_router_gemm.py

``--tune`` sweeps the kernel's block size and warp count at one token count
instead, for picking the launch config.
"""

import argparse

import torch
import triton

from tensorrt_llm._torch.modules.fp32_router_gemm import _fp32_router_gemm_kernel, fp32_router_gemm

# MiniMax-M3: 128 experts over a 6144-wide hidden state, on every sparse layer.
M3_HIDDEN = 6144
M3_EXPERTS = 128
M3_SPARSE_LAYERS = 57


def _graph_time_us(fn, args) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.calls_per_graph):
            fn()

    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(args.iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (args.iters * args.calls_per_graph)


def _measure_floor(args) -> float:
    scratch = torch.zeros(1, device="cuda")
    return _graph_time_us(lambda: scratch.add_(1.0), args)


def _make_inputs(num_tokens, args):
    x = torch.randn(num_tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(args.num_experts, args.hidden_size, device="cuda", dtype=torch.float32)
    return x, w


def _time_cublas(x, w, args, tf32: bool) -> float:
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        # The cast is part of the baseline: it only exists because cuBLAS needs
        # an FP32 activation to go with the FP32 weight.
        return _graph_time_us(lambda: torch.nn.functional.linear(x.to(torch.float32), w), args)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _run_compare(args) -> None:
    floor = _measure_floor(args)
    print(
        f"\nFP32 router projection, {args.num_experts} experts, "
        f"hidden {args.hidden_size}, bf16 activation"
    )
    print(
        f"{args.calls_per_graph} calls per graph, {args.warmup} warmup / "
        f"{args.iters} replays; harness floor {floor:.2f} us/call\n"
    )
    header = (
        f"{'tokens':>7s} {'cublas tf32':>12s} {'cublas fp32':>12s} "
        f"{'gemv':>8s} {'speedup':>9s} {'us/step saved':>14s}"
    )
    print(header)
    for n in args.num_tokens:
        x, w = _make_inputs(n, args)
        tf32_us = _time_cublas(x, w, args, tf32=True)
        fp32_us = _time_cublas(x, w, args, tf32=False)
        gemv_us = _graph_time_us(lambda: fp32_router_gemm(x, w), args)
        saved = (tf32_us - gemv_us) * args.layers
        print(
            f"{n:7d} {tf32_us:12.2f} {fp32_us:12.2f} {gemv_us:8.2f} "
            f"{tf32_us / gemv_us:8.2f}x {saved:14.0f}"
        )
    print(
        f"\nus/step saved assumes {args.layers} sparse layers against the tf32 "
        "column, which is what the profile shows in production."
    )


def _run_tune(args) -> None:
    """Per-call time across launch configs, for one token count."""
    num_tokens = args.num_tokens[0]
    x, w = _make_inputs(num_tokens, args)
    out = torch.empty((num_tokens, args.num_experts), dtype=torch.float32, device="cuda")
    block_m = triton.next_power_of_2(num_tokens)

    floor = _measure_floor(args)
    baseline = _time_cublas(x, w, args, tf32=True)
    print(f"\nTuning at {num_tokens} tokens, {args.num_experts} experts, ", end="")
    print(f"hidden {args.hidden_size}; harness floor {floor:.2f} us/call")
    print(f"cublas tf32 baseline at this token count: {baseline:.2f} us\n")
    print(f"{'BLOCK_K':>8s} {'warps':>6s} {'us':>8s}")

    results = []
    for block_k in (128, 256, 512, 1024, 2048, 4096, 8192):
        for num_warps in (4, 8, 16):
            if block_m * block_k > 65536:  # would spill
                continue

            def run(block_k=block_k, num_warps=num_warps):
                _fp32_router_gemm_kernel[(args.num_experts,)](
                    x,
                    w,
                    out,
                    num_tokens,
                    args.hidden_size,
                    x.stride(0),
                    x.stride(1),
                    w.stride(0),
                    w.stride(1),
                    out.stride(0),
                    out.stride(1),
                    BLOCK_M=block_m,
                    BLOCK_K=block_k,
                    num_warps=num_warps,
                )

            elapsed = _graph_time_us(run, args)
            results.append((elapsed, block_k, num_warps))
            print(f"{block_k:8d} {num_warps:6d} {elapsed:8.2f}")

    best = min(results)
    print(
        f"\nbest: BLOCK_K={best[1]} num_warps={best[2]} at {best[0]:.2f} us, "
        f"{baseline / best[0]:.2f}x the cublas baseline"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=M3_HIDDEN)
    parser.add_argument("--num-experts", type=int, default=M3_EXPERTS)
    parser.add_argument("--layers", type=int, default=M3_SPARSE_LAYERS)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 32])
    parser.add_argument("--calls-per-graph", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Sweep BLOCK_K and num_warps at the first --num-tokens value.",
    )
    args = parser.parse_args()

    if args.tune:
        _run_tune(args)
    else:
        _run_compare(args)


if __name__ == "__main__":
    main()
