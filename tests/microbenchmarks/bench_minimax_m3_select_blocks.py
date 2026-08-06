#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep the MiniMax-M3 block selector across row width and row count.

Times the op under CUDA graph replay. Eager launches bottom out around 6 us of
host overhead, which is more than the kernel costs at small sizes and hides the
comparison; MSA decode runs under graphs anyway, so replay is both the sharper
measurement and the representative one.

num_blocks is ceil(context / 128), so 64 blocks is an 8K context, 128 is 16K,
1024 is 128K and 8192 is 1M. Rows up to 128 blocks take the register-resident
bitonic kernels and wider rows take the histogram select; see kSmallMaxBlocks in
cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu.

  python3 tests/microbenchmarks/bench_minimax_m3_select_blocks.py

Pass --baseline to print a recorded column alongside the live one. The bundled
`warp_strided` baseline is the kernel the histogram select replaced, measured on
a B200 with this harness at its default iteration counts; it is dead code now,
so it can only be replayed from the record, and it is only meaningful against a
live run on the same GPU at the same iteration counts.
"""

import argparse
from pathlib import Path

import torch

# Rows the kernel sees are num_kv_heads * num_queries.
SHAPES = [(1, 1), (4, 8), (8, 16), (8, 64), (8, 128), (8, 256)]
# Dense around 128-256, which is where the dispatch boundary sits.
BLOCKS = [32, 64, 96, 128, 160, 192, 224, 256, 320, 512, 1024, 4096, 8192]

# Widest row the bitonic kernels can hold; above this the histogram select runs.
BITONIC_MAX_BLOCKS = 128

TOPK = 16

# Recorded times in microseconds for the deleted warp-strided kernel, keyed by
# "kv_heads/queries/blocks". Measured on a B200 under CUDA graph replay at
# warmup=200 iters=1000. Entries at or below BITONIC_MAX_BLOCKS are the bitonic
# kernels, which are still live, so only the wider rows are a true baseline.
WARP_STRIDED_BASELINE = {
    "1/1": [6.15, 4.09, 6.15, 6.14, 10.24, 10.24, 10.24, 11.27, 11.27, 14.34, 21.50, 53.26, 88.02],
    "4/8": [6.15, 4.10, 6.15, 6.15, 12.30, 14.31, 14.33, 14.36, 16.39, 20.51, 31.33, 78.07, 124.70],
    "8/16": [
        6.15,
        4.10,
        6.15,
        6.15,
        13.47,
        14.34,
        14.34,
        16.36,
        17.25,
        22.52,
        34.77,
        87.83,
        140.03,
    ],
    "8/64": [
        8.19,
        4.10,
        6.16,
        8.19,
        14.33,
        14.40,
        15.70,
        16.39,
        18.44,
        24.47,
        37.04,
        100.50,
        170.12,
    ],
    "8/128": [
        8.19,
        4.10,
        8.19,
        8.19,
        14.34,
        15.43,
        16.38,
        17.32,
        19.55,
        24.87,
        40.13,
        104.78,
        175.54,
    ],
    "8/256": [
        8.20,
        6.15,
        10.24,
        10.25,
        20.49,
        22.55,
        24.43,
        26.62,
        29.86,
        40.92,
        68.31,
        193.76,
        319.67,
    ],
}


def load_ops() -> None:
    """Register the trtllm ops without importing the Python package.

    Importing tensorrt_llm pulls in the whole serving stack and its optional
    dependencies, none of which this benchmark needs.
    """
    if hasattr(torch.ops.trtllm, "minimax_m3_select_blocks"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    torch.ops.load_library(str(repo_root / "tensorrt_llm" / "libs" / "libth_common.so"))


def bench_one(num_kv_heads: int, num_blocks: int, total_q: int, iters: int, warmup: int) -> float:
    """Return microseconds per selector call at this shape."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    scores = torch.randn(num_kv_heads, num_blocks, total_q, generator=generator, device="cuda")
    # Every row full, which is the most work the kernel can be asked to do.
    n_valid_blocks = torch.full((total_q,), num_blocks, device="cuda", dtype=torch.int32)

    def run():
        return torch.ops.trtllm.minimax_m3_select_blocks(scores, n_valid_blocks, TOPK, 0, 1, False)

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Print the recorded warp-strided times and the speedup against them.",
    )
    args = parser.parse_args()

    load_ops()
    print(
        f"{torch.cuda.get_device_name(0)}, CUDA graph replay, "
        f"warmup={args.warmup} iters={args.iters}, times in microseconds"
    )
    if args.baseline and (args.iters, args.warmup) != (1000, 200):
        print("  NOTE: baseline was recorded at warmup=200 iters=1000; speedups are not comparable")
    print()

    for num_kv_heads, total_q in SHAPES:
        baseline = WARP_STRIDED_BASELINE[f"{num_kv_heads}/{total_q}"] if args.baseline else None
        print(f"kv_heads={num_kv_heads} queries={total_q} -> {num_kv_heads * total_q} rows")
        header = f"  {'blocks':>7} {'context':>8} {'kernel':>10} {'live':>8}"
        if baseline is not None:
            header += f" {'recorded':>9} {'speedup':>8}"
        print(header)
        for index, num_blocks in enumerate(BLOCKS):
            live = bench_one(num_kv_heads, num_blocks, total_q, args.iters, args.warmup)
            kernel = "bitonic" if num_blocks <= BITONIC_MAX_BLOCKS else "histogram"
            line = f"  {num_blocks:>7} {num_blocks * 128 // 1024:>7}K {kernel:>10} {live:>8.2f}"
            if baseline is not None:
                old = baseline[index]
                line += f" {old:>9.2f} {old / live:>7.2f}x"
                if num_blocks <= BITONIC_MAX_BLOCKS:
                    line += "  (same kernel both columns)"
            print(line)
        print()


if __name__ == "__main__":
    main()
