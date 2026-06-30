# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: CuTe DSL prefill top-k  vs  C++ radix prefill top-k.

Timing uses CuptiProfiler (GPU kernel time only, same as other DSL bench
scripts in this directory) to avoid Python dispatch overhead inflation.

The C++ kernel (trtllm::indexer_topk_prefill) only supports fp32.
The DSL kernel (trtllm::cute_dsl_indexer_topk_prefill_blackwell) supports
bf16, fp16, and fp32.

Benchmarked variants:
  cpp_fp32   C++ radix, fp32 logits
  dsl_fp32   DSL radix, fp32 logits
  dsl_bf16   DSL radix, bf16 logits  (production scenario)

Usage:
    python bench_prefill_topk.py [--num_cols 8192] [--warmup 10] [--iters 100]
"""

import argparse
import sys
from pathlib import Path

import torch

# Load TensorRT-LLM custom ops.
import tensorrt_llm  # noqa: F401  (registers ops)
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops  # noqa: F401

# CuptiProfiler lives in the sibling testing.py.
try:
    from tests.scripts.cute_dsl_kernels.testing import CuptiProfiler
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from testing import CuptiProfiler  # type: ignore[no-redef]

IS_B200 = torch.cuda.get_device_capability() >= (10, 0)


def cupti_timer(fn, warmup: int, iters: int) -> float:
    """Return average GPU kernel time in microseconds measured via CUPTI.

    CUPTI records individual kernel durations on the GPU timeline, so Python
    dispatch overhead is excluded from the measurement.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    profiler = CuptiProfiler()
    profiler.start()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    profiler.stop()

    # get_duration() returns total ms for all captured kernel activities.
    return profiler.get_duration() / iters * 1e3  # ms -> us


def bench_one(num_rows: int, num_cols: int, top_k: int, warmup: int, iters: int) -> dict:
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.full((num_rows,), num_cols, dtype=torch.int32, device="cuda")

    results = {}

    # ── C++ fp32 ──────────────────────────────────────────────────────────────
    logits_fp32 = torch.randn(num_rows, num_cols, dtype=torch.float32, device="cuda")
    indices_cpp = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

    def run_cpp():
        torch.ops.trtllm.indexer_topk_prefill(logits_fp32, row_starts, row_ends, indices_cpp, top_k)

    results["cpp_fp32"] = cupti_timer(run_cpp, warmup, iters)

    # ── DSL fp32 / bf16 ───────────────────────────────────────────────────────
    if IS_B200:

        def run_dsl_fp32():
            torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                logits_fp32, row_starts, row_ends, top_k
            )

        results["dsl_fp32"] = cupti_timer(run_dsl_fp32, warmup, iters)

        logits_bf16 = logits_fp32.bfloat16()

        def run_dsl_bf16():
            torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                logits_bf16, row_starts, row_ends, top_k
            )

        results["dsl_bf16"] = cupti_timer(run_dsl_bf16, warmup, iters)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num_cols", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    NUM_ROWS_LIST = [256, 512, 1024, 2048, 4096, 8192, 16384]
    TOP_K_LIST = [512, 1024, 2048]
    num_cols = args.num_cols

    print(
        f"Device: {torch.cuda.get_device_name(0)}  "
        f"(sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})"
    )
    print(f"num_cols={num_cols}, warmup={args.warmup}, iters={args.iters}")
    print("Timing: CuptiProfiler (GPU kernel time only)\n")

    if not IS_B200:
        print("DSL kernel requires Blackwell (sm100+), skipping DSL columns.")

    variants = ["cpp_fp32"] + (["dsl_fp32", "dsl_bf16"] if IS_B200 else [])
    speedup_cols = ["dsl_fp32/cpp", "dsl_bf16/cpp"] if IS_B200 else []

    col_w = 12
    hdr = f"{'top_k':>6}  {'num_rows':>8}"
    for v in variants:
        hdr += f"  {v:>{col_w}}"
    for s in speedup_cols:
        hdr += f"  {s:>{col_w}}"
    print(hdr)
    print("-" * len(hdr))

    for top_k in TOP_K_LIST:
        for num_rows in NUM_ROWS_LIST:
            try:
                res = bench_one(num_rows, num_cols, top_k, args.warmup, args.iters)
            except Exception as e:
                print(f"{top_k:>6}  {num_rows:>8}  ERROR: {e}")
                continue

            line = f"{top_k:>6}  {num_rows:>8}"
            for v in variants:
                us = res.get(v, float("nan"))
                line += f"  {us:>{col_w}.2f}us"
            if IS_B200:
                cpp = res.get("cpp_fp32", float("nan"))
                for v in ["dsl_fp32", "dsl_bf16"]:
                    sp = cpp / res[v] if res.get(v) else float("nan")
                    line += f"  {sp:>{col_w}.3f}x"
            print(line)
        print()

    print("Values are avg kernel latency (us). Speedup = cpp_fp32 / dsl_*.")


if __name__ == "__main__":
    main()
