# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: CuTe DSL decode top-k  vs  C++ radix decode top-k.

Timing uses CuptiProfiler (GPU kernel time only) to exclude Python dispatch
overhead.

Benchmarked variants:
  cpp_fp32   C++ radix decode, fp32 logits
  dsl_fp32   DSL single-CTA decode, fp32 logits
  dsl_bf16   DSL single-CTA decode, bf16 logits  (production scenario)

Usage:
    python bench_decode_topk.py --num_tokens 32768
    python bench_decode_topk.py --num_tokens 32768 --next_n 2
    python bench_decode_topk.py --sweep       # sweep num_tokens × batch_size
"""

import argparse
import sys
from pathlib import Path

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops  # noqa: F401

try:
    from tests.scripts.cute_dsl_kernels.testing import CuptiProfiler
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from testing import CuptiProfiler  # type: ignore[no-redef]

IS_B200 = torch.cuda.get_device_capability() >= (10, 0)

_K_MAX_BLOCKS_PER_ROW_DECODE = 10  # mirrors kMaxBlocksPerRowDecode in indexerTopK.cu


def cupti_timer(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    profiler = CuptiProfiler()
    profiler.start()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    profiler.stop()
    return profiler.get_duration() / iters * 1e3  # ms -> us


def build_radix_aux(num_gen_tokens: int, top_k: int):
    return (
        torch.empty(
            (num_gen_tokens, _K_MAX_BLOCKS_PER_ROW_DECODE, top_k), dtype=torch.int32, device="cuda"
        ),
        torch.empty(
            (num_gen_tokens, _K_MAX_BLOCKS_PER_ROW_DECODE, top_k),
            dtype=torch.float32,
            device="cuda",
        ),
    )


def bench_one(
    batch_size: int, num_tokens: int, top_k: int, next_n: int, warmup: int, iters: int
) -> dict:
    num_gen_tokens = batch_size * next_n
    # All sequences see the full num_tokens window (fixed-length decode).
    seq_lens = torch.full((batch_size,), num_tokens, dtype=torch.int32, device="cuda")
    logits_fp32 = torch.randn(num_gen_tokens, num_tokens, dtype=torch.float32, device="cuda")
    logits_bf16 = logits_fp32.bfloat16()
    indices_cpp = torch.empty((num_gen_tokens, top_k), dtype=torch.int32, device="cuda")
    radix_aux_idx, radix_aux_log = build_radix_aux(num_gen_tokens, top_k)

    results = {}

    # ── C++ fp32 ──────────────────────────────────────────────────────────────
    def run_cpp():
        torch.ops.trtllm.indexer_topk_decode(
            logits_fp32,
            seq_lens,
            indices_cpp,
            next_n,
            top_k,
            radix_aux_indices=radix_aux_idx,
            radix_aux_logits=radix_aux_log,
        )

    results["cpp_fp32"] = cupti_timer(run_cpp, warmup, iters)

    # ── DSL fp32 / bf16 ───────────────────────────────────────────────────────
    if IS_B200:
        out_dsl_fp32 = torch.empty((num_gen_tokens, top_k), dtype=torch.int32, device="cuda")
        out_dsl_bf16 = torch.empty((num_gen_tokens, top_k), dtype=torch.int32, device="cuda")

        def run_dsl_fp32():
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                input_values=logits_fp32,
                seq_lens=seq_lens,
                output_indices=out_dsl_fp32,
                top_k=top_k,
                next_n=next_n,
            )

        def run_dsl_bf16():
            torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                input_values=logits_bf16,
                seq_lens=seq_lens,
                output_indices=out_dsl_bf16,
                top_k=top_k,
                next_n=next_n,
            )

        results["dsl_fp32"] = cupti_timer(run_dsl_fp32, warmup, iters)
        results["dsl_bf16"] = cupti_timer(run_dsl_bf16, warmup, iters)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--num_tokens", type=int, default=32768, help="Vocab/KV size (ignored in --sweep mode)"
    )
    parser.add_argument(
        "--next_n", type=int, default=1, help="Speculative decoding candidates per sequence"
    )
    parser.add_argument("--sweep", action="store_true", help="Sweep num_tokens x batch_size grid")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    TOP_K_LIST = [512, 1024, 2048]
    BATCH_LIST = [1, 4, 16, 64, 256]

    # Single num_tokens or sweep grid
    if args.sweep:
        TOKENS_LIST = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    else:
        TOKENS_LIST = [args.num_tokens]

    variants = ["cpp_fp32"] + (["dsl_fp32", "dsl_bf16"] if IS_B200 else [])
    speedup_cols = ["dsl_fp32/cpp", "dsl_bf16/cpp"] if IS_B200 else []
    col_w = 14

    cap = torch.cuda.get_device_capability()
    print(f"[TensorRT-LLM] Device: {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
    print("Timing: CuptiProfiler (GPU kernel time only)")
    print(f"next_n={args.next_n}, warmup={args.warmup}, iters={args.iters}")
    if not IS_B200:
        print("DSL kernel requires Blackwell (sm100+), skipping DSL columns.")

    for num_tokens in TOKENS_LIST:
        print(f"\n=== num_tokens={num_tokens} ===")
        hdr = f"{'top_k':>6}  {'batch':>6}"
        for v in variants:
            hdr += f"  {v:>{col_w}}"
        for s in speedup_cols:
            hdr += f"  {s:>{col_w}}"
        print(hdr)
        print("-" * len(hdr))

        for top_k in TOP_K_LIST:
            for batch_size in BATCH_LIST:
                try:
                    res = bench_one(
                        batch_size, num_tokens, top_k, args.next_n, args.warmup, args.iters
                    )
                except Exception as e:
                    print(f"{top_k:>6}  {batch_size:>6}  ERROR: {e}")
                    continue

                line = f"{top_k:>6}  {batch_size:>6}"
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
