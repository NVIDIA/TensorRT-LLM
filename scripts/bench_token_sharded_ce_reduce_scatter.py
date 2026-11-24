# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the row-GEMM/RS crossover on an SM100 TP group with peer access.

Run in a built TRT-LLM environment, once per K_local (1280 and 3456 for Wan):

    torchrun --standalone --nproc-per-node=4 scripts/bench_token_sharded_ce_reduce_scatter.py

Both paths consume the same prequantized operands. Timings cover GEMM, communication
and bias through CUDA graph replay, excluding the shared input quantizer and host allocation.
The CE op is forced even below the dispatch cutoff to expose the crossover. CSV reports
median latency over repeats, taking the slowest rank, plus actual slot bytes and bounded
staging bytes. Repeat on each target topology before lowering the conservative cutoff.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.parallel import token_sharded_ce_reduce_scatter as rs
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_ce_gather import (
    fp8_block_gemm_out,
    mn_major_scales,
    packed_weight_scale_cols,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    MIN_CE_REDUCE_SCATTER_ROWS,
    TokenShardPlan,
    quantize_fp8_block,
)
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.math_utils import pad_up


def _capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    return graph, result


def _measure(nccl_fn, ce_fn, iterations, repeats):
    # Keep both outputs alive throughout the paired, alternating-order measurements.
    captured = [_capture(nccl_fn), _capture(ce_fn)]
    samples = [[], []]
    for repeat in range(repeats):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                captured[arm][0].replay()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end) * 1000 / iterations, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[arm].append(elapsed.item())
    return tuple(statistics.median(arm_samples) for arm_samples in samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[256, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 151200],
    )
    parser.add_argument("--k-local", type=int, default=1280)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=rs.CHUNK_ROWS,
        help="Benchmark-only override of the maximum rows per destination and transport round.",
    )
    args = parser.parse_args()
    if (
        min(args.rows + [args.n, args.k_local, args.iterations, args.repeats, args.chunk_rows]) < 1
        or args.k_local % 128
    ):
        parser.error("sizes/repeats must be positive and K_local a multiple of 128")
    rs.CHUNK_ROWS = args.chunk_rows
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, tp = dist.get_rank(), dist.get_world_size()
    state = rs.register_rs_state(
        rs.CeReduceScatterState(dist.group.WORLD, dist.group.WORLD.group_name)
    )
    try:
        if tp < 2 or not is_sm_100f():
            raise RuntimeError("benchmark requires at least two SM100 GPUs")
        torch.manual_seed(42 + rank)
        w = (torch.randn(args.n, args.k_local, device="cuda") * args.k_local**-0.5).to(
            torch.float8_e4m3fn
        )
        ws = torch.full(
            (packed_weight_scale_cols(args.k_local), pad_up(args.n, 4)),
            0x7F7F7F7F,
            dtype=torch.int32,
            device="cuda",
        ).t()[: args.n]
        bias = (
            torch.full((args.n,), 0.25, device="cuda", dtype=torch.bfloat16) if args.bias else None
        )
        state.note_row_consumer(args.k_local, args.n)
        state.prepare(TokenShardPlan.build(1, rs.CHUNK_ROWS * tp, tp, rank))
        if not state.effective:
            raise RuntimeError(state.reason)
        if rank == 0:
            print(
                f"# gpu={torch.cuda.get_device_name()} torch={torch.__version__} tp={tp} "
                f"K_local={args.k_local} N={args.n} bias={args.bias} chunk_rows={rs.CHUNK_ROWS}",
                flush=True,
            )
            print(
                "M,padded_M,rows_per_dest,nccl_us,ce_us,speedup,slot_bytes,staging_bytes,dispatch",
                flush=True,
            )
        for rows in args.rows:
            plan = TokenShardPlan.build(1, rows, tp, rank)
            state.prepare(plan)
            x = torch.randn(plan.padded_rows, args.k_local, device="cuda", dtype=torch.bfloat16)
            fp8, sf = quantize_fp8_block(x)

            def nccl():
                partial = torch.empty(
                    (plan.padded_rows, args.n), dtype=torch.bfloat16, device="cuda"
                )
                fp8_block_gemm_out(fp8, mn_major_scales(sf), w, ws, partial)
                if bias is not None and rank == 0:
                    partial.add_(bias)
                out = torch.empty((plan.local_rows, args.n), dtype=torch.bfloat16, device="cuda")
                dist.reduce_scatter_tensor(out, partial)
                return out

            def ce():
                return torch.ops.trtllm.token_sharded_fp8_ce_gemm_reduce_scatter(
                    fp8, sf, w, ws, bias, state.group_name, rank, tp, 1, rows, plan.padded_seq_len
                )

            # The fixed-order fp32 reduce and NCCL's bf16 ring can differ slightly.
            agreed = torch.isclose(ce(), nccl(), rtol=2e-2, atol=5e-2).all().to(torch.int32)
            dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
            if not agreed.item():
                raise RuntimeError(f"CE and NCCL outputs disagree beyond tolerance at M={rows}")
            nccl_us, ce_us = _measure(nccl, ce, args.iterations, args.repeats)
            if rank == 0:
                staging = (tp - 1) * min(plan.local_rows, rs.CHUNK_ROWS) * args.n * 2
                dispatch = (
                    "ce" if args.bias and plan.local_rows >= MIN_CE_REDUCE_SCATTER_ROWS else "nccl"
                )
                print(
                    f"{rows},{plan.padded_rows},{plan.local_rows},{nccl_us:.3f},{ce_us:.3f},"
                    f"{nccl_us / ce_us:.3f},{state.pool.nbytes},{staging},{dispatch}",
                    flush=True,
                )
    finally:
        state.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
