# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static vs dynamic block MoE routing kernel at decode batch sizes.

The trtllm-gen routing dispatch gives every batch of at most
``BlockKernelMaxNumTokens`` (4) to ``routingIndicesBlockKernel``, so
``routingIndicesDynBlockKernel`` never runs there -- even though it was written
for small batches, giving each token its own warp instead of looping over a
batch of them and replacing the CUB block scan with a fused warp scan.
``TRTLLM_MOE_ROUTING_PREFER_DYN_BLOCK=1`` hands those batches to the dynamic
kernel instead. This times both so the threshold can be set by measurement.

Routing is reached through ``moe_topk_sort``, which runs the routing kernel and
nothing else, under CUDA graph replay to match how decode drives it.

The env var is read once into a function-local static, so the two arms cannot
share a process. By default this script re-runs itself once per arm and prints
the comparison; ``--child`` measures a single arm and emits JSON.

    python tests/microbenchmarks/bench_moe_routing_small_batch.py \
        --num-experts 128 --top-k 8

Batch sizes above 4 are a control: the override cannot reach them, so both arms
must report the same kernel and the same time.
"""

import argparse
import json
import os
import subprocess
import sys

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401

ENV_VAR = "TRTLLM_MOE_ROUTING_PREFER_DYN_BLOCK"
# RoutingMethodType::MiniMax2, which both MiniMax-M2 and M3 use.
MINIMAX_ROUTING_METHOD = 5
# Dispatch threshold in RoutingCustomPolicy.cuh; at or below this the static
# kernel wins unless the override is set.
STATIC_BLOCK_MAX_TOKENS = 4


def _graph_time_us(fn, warmup: int = 200, iters: int = 1000) -> float:
    """Median-ish per-call GPU time under graph replay.

    Replay rather than eager launch: at these sizes an eager measurement is
    dominated by launch overhead, which is exactly what decode avoids by
    capturing the step.
    """
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _measure(num_tokens: int, args) -> float:
    # The logits dtype picks the kernel's load path, so it has to match the
    # gate: M3's runs in FP32.
    dtype = getattr(torch, args.logits_dtype)
    logits = torch.randn(num_tokens, args.num_experts, device="cuda", dtype=dtype)
    bias = torch.randn(args.num_experts, device="cuda", dtype=dtype)

    def run():
        torch.ops.trtllm.moe_topk_sort(
            logits,
            bias,
            args.num_experts,
            args.top_k,
            None,  # n_group
            None,  # topk_group
            0,  # local_expert_offset
            args.num_experts,  # local_num_experts, i.e. no expert parallelism
            args.routed_scaling_factor,
            args.tile_tokens_dim,
            args.routing_method_type,
        )

    return _graph_time_us(run, warmup=args.warmup, iters=args.iters)


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tile-tokens-dim", type=int, default=8)
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--routing-method-type", type=int, default=MINIMAX_ROUTING_METHOD)
    parser.add_argument("--logits-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 8, 16],
        help="Decode batch sizes to sweep. Values above 4 act as a control.",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)


def _run_child(args) -> None:
    results = {str(n): _measure(n, args) for n in args.num_tokens}
    print(json.dumps(results))


def _run_parent(args, argv) -> None:
    child_argv = [a for a in argv if a != "--child"]
    arms = {}
    for value in ("0", "1"):
        env = dict(os.environ, **{ENV_VAR: value})
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", *child_argv],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"{ENV_VAR}={value} arm failed with {proc.returncode}")
        arms[value] = json.loads(proc.stdout.strip().splitlines()[-1])

    print(
        f"\nMoE routing kernel, {args.num_experts} experts, top-{args.top_k}, "
        f"tile_tokens_dim={args.tile_tokens_dim}, {args.logits_dtype} logits, "
        f"routing_method_type={args.routing_method_type}"
    )
    print(f"{args.warmup} warmup / {args.iters} iterations under CUDA graph replay\n")
    print(f"{'tokens':>7s} {'static us':>11s} {'dynblock us':>13s} {'speedup':>9s}  note")
    for n in args.num_tokens:
        static_us, dyn_us = arms["0"][str(n)], arms["1"][str(n)]
        note = (
            "override applies"
            if n <= STATIC_BLOCK_MAX_TOKENS
            else "control: both arms take the dynamic kernel"
        )
        print(f"{n:7d} {static_us:11.2f} {dyn_us:13.2f} {static_us / dyn_us:8.2f}x  {note}")
    print(
        "\nA control row differing by more than run-to-run noise means the "
        "override is reaching batches it should not."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    _add_shared_args(parser)
    args = parser.parse_args()

    if args.child:
        _run_child(args)
    else:
        _run_parent(args, sys.argv[1:])


if __name__ == "__main__":
    main()
