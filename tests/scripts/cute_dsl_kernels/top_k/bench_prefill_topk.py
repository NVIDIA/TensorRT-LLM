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

Fixed-length mode (default):
    python bench_prefill_topk.py --num_cols 8192

Varlen mode (lower-triangular, realistic prefill):
    python bench_prefill_topk.py --varlen [--bs 1]
    Models chunked prefill: bs independent causal sequences packed into one
    chunk.  Each sequence b forms a lower-triangular KV block at columns
    [b*isl, (b+1)*isl); row i within sequence b has width i+1 (mean isl/2).
    Constraint: bs*isl <= max_chunk_size (32768) — larger configs are skipped.
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


def make_varlen_inputs(isl: int, bs: int):
    """Build triangular row_starts/row_ends for bs sequences each of length isl.

    Each sequence b occupies KV columns [b*isl, (b+1)*isl).  Query token i
    within sequence b (0-indexed) has a causal window [b*isl, b*isl + i + 1),
    giving a lower-triangular block on the diagonal of the logits matrix.

    Returns:
        row_starts: [bs*isl] int32 CUDA tensor
        row_ends:   [bs*isl] int32 CUDA tensor
        total_kv:   int, second dim of the logits matrix (= bs * isl)
    """
    # seq_offset[b] = b * isl; broadcast across all tokens of each sequence
    seq_offsets = torch.arange(bs, dtype=torch.int32) * isl  # [bs]
    row_starts = seq_offsets.repeat_interleave(isl).cuda()  # [total]
    # within-seq position 1..isl, tiled bs times
    within = torch.tile(torch.arange(1, isl + 1, dtype=torch.int32), (bs,)).cuda()
    row_ends = row_starts + within  # [total]
    return row_starts, row_ends, bs * isl


def bench_one_varlen(isl: int, bs: int, top_k: int, warmup: int, iters: int) -> dict:
    """Bench with triangular varlen inputs (bs independent causal sequences of length isl)."""
    num_rows = bs * isl
    row_starts, row_ends, total_kv = make_varlen_inputs(isl, bs)

    results = {}

    # ── C++ fp32 ──────────────────────────────────────────────────────────────
    logits_fp32 = torch.randn(num_rows, total_kv, dtype=torch.float32, device="cuda")
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
    parser.add_argument(
        "--num_cols",
        type=int,
        default=8192,
        help="Fixed-length mode: column width (ignored in --varlen mode)",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Varlen mode: lower-triangular inputs (bs causal sequences of length isl)",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=1,
        help="Batch size for --varlen mode (number of independent sequences)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    # Fixed-length: num_rows sweep (independent of num_cols).
    FIXED_ROWS_LIST = [4096, 8192, 16384, 32768]
    # Varlen: ISL per sequence; bs*isl must not exceed max_chunk_size=32768.
    VARLEN_ISL_LIST = [4096, 8192, 16384, 32768]
    MAX_CHUNK_SIZE = 32768
    TOP_K_LIST = [512, 1024, 2048]

    print(
        f"Device: {torch.cuda.get_device_name(0)}  "
        f"(sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})"
    )
    print("Timing: CuptiProfiler (GPU kernel time only)")

    if not IS_B200:
        print("DSL kernel requires Blackwell (sm100+), skipping DSL columns.")

    variants = ["cpp_fp32"] + (["dsl_fp32", "dsl_bf16"] if IS_B200 else [])
    speedup_cols = ["dsl_fp32/cpp", "dsl_bf16/cpp"] if IS_B200 else []
    col_w = 12

    if args.varlen:
        # ── Varlen (triangular) mode ───────────────────────────────────────────
        print(
            f"Mode: varlen (lower-triangular), bs={args.bs}, warmup={args.warmup}, iters={args.iters}"
        )
        print("      isl = per-sequence length; num_rows = bs*isl; logits = [num_rows, num_rows].")
        print(
            f"      Mean row width = isl/2 (vs isl for fixed-length).  max_chunk_size={MAX_CHUNK_SIZE}.\n"
        )

        hdr = f"{'top_k':>6}  {'isl':>8}  {'num_rows':>8}"
        for v in variants:
            hdr += f"  {v:>{col_w}}"
        for s in speedup_cols:
            hdr += f"  {s:>{col_w}}"
        print(hdr)
        print("-" * len(hdr))

        for top_k in TOP_K_LIST:
            for isl in VARLEN_ISL_LIST:
                num_rows = args.bs * isl
                if num_rows > MAX_CHUNK_SIZE:
                    print(
                        f"{top_k:>6}  {isl:>8}  {num_rows:>8}  "
                        f"SKIP (bs*isl={num_rows} > max_chunk_size={MAX_CHUNK_SIZE})"
                    )
                    continue
                try:
                    res = bench_one_varlen(isl, args.bs, top_k, args.warmup, args.iters)
                except Exception as e:
                    print(f"{top_k:>6}  {isl:>8}  {num_rows:>8}  ERROR: {e}")
                    continue

                line = f"{top_k:>6}  {isl:>8}  {num_rows:>8}"
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

    else:
        # ── Fixed-length mode ─────────────────────────────────────────────────
        num_cols = args.num_cols
        print(
            f"Mode: fixed-length, num_cols={num_cols}, warmup={args.warmup}, iters={args.iters}\n"
        )

        hdr = f"{'top_k':>6}  {'num_rows':>8}"
        for v in variants:
            hdr += f"  {v:>{col_w}}"
        for s in speedup_cols:
            hdr += f"  {s:>{col_w}}"
        print(hdr)
        print("-" * len(hdr))

        for top_k in TOP_K_LIST:
            for num_rows in FIXED_ROWS_LIST:
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
