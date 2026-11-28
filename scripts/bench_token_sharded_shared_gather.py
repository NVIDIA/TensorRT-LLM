# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare independent and shared FP8 CE gathers for three sibling projections.

Run in a built SM100 TRT-LLM environment:
  torchrun --standalone --nproc-per-node=4 scripts/bench_token_sharded_shared_gather.py

Times include local quantization, transport and GEMMs through CUDA-graph replay.
Widths are local output widths; these defaults are representative, not gen45 measurements.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.parallel.token_sharded_ce_gather import (
    CeGatherState,
    packed_weight_scale_cols,
    register_state,
)
from tensorrt_llm._torch.visual_gen.parallel.token_sharded_tp import (
    TokenShardPlan,
    quantize_fp8_block,
)
from tensorrt_llm.math_utils import pad_up


def _capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    return graph, result


def _measure(functions, iterations, repeats):
    graphs = [_capture(fn) for fn in functions]
    samples = [[], []]
    for repeat in range(repeats):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[arm][0].replay()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end) * 1000 / iterations, device="cuda")
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[arm].append(elapsed.item())
    return [statistics.median(values) for values in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[768, 4096, 32768, 65536])
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--widths", type=int, nargs=3, default=[3840, 3456, 3456])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows + args.widths + [args.k, args.iterations, args.repeats]) < 1 or args.k % 128:
        parser.error("dimensions and repeat counts must be positive; K must be a multiple of 128")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, size = dist.get_rank(), dist.get_world_size()
    if size < 2:
        raise RuntimeError("requires at least two peer-accessible SM100 GPUs")
    state = register_state(CeGatherState(dist.group.WORLD, dist.group.WORLD.group_name))
    state.note_consumer(args.k)
    torch.manual_seed(41 + rank)
    weights = [
        (torch.randn(n, args.k, device=device) * args.k**-0.5).to(torch.float8_e4m3fn)
        for n in args.widths
    ]
    scales = [
        torch.full(
            (packed_weight_scale_cols(args.k), pad_up(n, 4)),
            0x7F7F7F7F,
            dtype=torch.int32,
            device=device,
        ).t()[:n]
        for n in args.widths
    ]
    try:
        # Captured graphs pin the pool allocation even after their Python objects are freed.
        state.prepare(TokenShardPlan.build(1, max(args.rows), size, rank))
        if rank == 0:
            print(
                f"# gpu={torch.cuda.get_device_name()} torch={torch.__version__} tp={size} "
                f"K={args.k} local_widths={args.widths} repeats={args.repeats} iterations={args.iterations}",
                flush=True,
            )
            print(
                "M,rows_per_rank,independent_us,shared_us,speedup,independent_gathers,shared_gathers",
                flush=True,
            )
        for rows in args.rows:
            plan = TokenShardPlan.build(1, rows, size, rank)
            state.prepare(plan)
            if not state.effective:
                raise RuntimeError(state.reason)
            x = torch.randn(plan.local_rows, args.k, device=device, dtype=torch.bfloat16)
            ints = (state.group_name, rank, size, 1, rows, plan.padded_seq_len)

            def independent():
                outputs = []
                for w, sf in zip(weights, scales):
                    packed = quantize_fp8_block(x)
                    outputs.append(
                        torch.ops.trtllm.token_sharded_fp8_ce_gather_gemm(*packed, w, sf, *ints)
                    )
                return outputs

            def shared():
                packed = quantize_fp8_block(x)
                return torch.ops.trtllm.token_sharded_fp8_ce_gather_gemms(
                    *packed, weights, scales, *ints
                )

            reference, actual = independent(), shared()
            equal = torch.tensor(
                int(all(torch.equal(a, b) for a, b in zip(actual, reference))), device=device
            )
            dist.all_reduce(equal, op=dist.ReduceOp.MIN)
            if not equal.item():
                raise RuntimeError(f"shared projections differ at M={rows}")
            a, b = _measure((independent, shared), args.iterations, args.repeats)
            if rank == 0:
                print(f"{rows},{plan.local_rows},{a:.3f},{b:.3f},{a / b:.3f},3,1", flush=True)
    finally:
        state.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
