# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Is MiniMax-M3's standalone MXFP8 activation quantize launch-bound or slow?

``MXFP8LinearMethod.apply`` calls ``torch.ops.trtllm.mxfp8_quantize`` before
every base GEMM. At decode batch sizes that reads very little: the qkv input is
``[4, 6144]`` bf16, or 48 KB. The profile still charges 296 us/step to the qkv
quantize and 115 us/step to the o_proj one, which only makes sense if the cost
is per-launch rather than per-byte.

That distinction decides how to spend effort. Folding the quantize into the
producer's epilogue removes the launch entirely, but for the qkv case that means
writing an MXFP8 epilogue into the AllReduce fusion kernel, which today has no
MXFP8 path at all. If instead the kernel is simply slow at these shapes, tuning
it is far cheaper and helps every caller.

Three numbers per shape:

* ``floor``    -- the same harness timing a trivial kernel, so the per-launch
  cost inside a CUDA graph. A quantize near this is launch-bound and only
  fusion can recover it.
* ``quantize`` -- ``mxfp8_quantize`` itself.
* ``cast``     -- ``x.to(e4m3)``, a bandwidth reference over the same input
  bytes with no block scales. The gap between it and ``quantize`` is what the
  scale computation and swizzled-SF write cost.

Many calls go into one CUDA graph, since a single-call graph measures the ~6us
host cost of ``graph.replay()`` instead of the kernel.

    python tests/microbenchmarks/bench_mxfp8_quantize_small_batch.py
"""

import argparse

import torch

# MiniMax-M3 at TP4, which is what the profile was taken at. qkv consumes the
# full hidden state; o_proj is row-parallel, so its input is the sharded
# attention output of 32 heads / 4 ranks x 128.
M3_SHAPES = (("qkv", 6144), ("o_proj", 1024))
# Layer counts the profile attributes each quantize to.
M3_LAYERS = 60


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


def _parse_shapes(raw):
    if raw is None:
        return list(M3_SHAPES)
    shapes = []
    for entry in raw:
        name, _, size = entry.partition(":")
        shapes.append((name, int(size)))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=None,
        help="name:hidden pairs, e.g. qkv:6144. Defaults to MiniMax-M3 at TP4.",
    )
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--layers", type=int, default=M3_LAYERS)
    parser.add_argument("--calls-per-graph", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not hasattr(torch.ops.trtllm, "mxfp8_quantize"):
        raise SystemExit(
            "torch.ops.trtllm.mxfp8_quantize is not registered; this build "
            "predates the MXFP8 quantization ops."
        )

    floor = _measure_floor(args)
    print("\nMXFP8 activation quantize, swizzled scale-factor layout, bf16 input")
    print(
        f"{args.calls_per_graph} calls per graph, {args.warmup} warmup / "
        f"{args.iters} replays; harness floor {floor:.2f} us/call\n"
    )
    print(
        f"{'shape':>8s} {'hidden':>7s} {'tokens':>7s} {'KB in':>7s} "
        f"{'quantize':>9s} {'cast':>7s} {'over floor':>11s} {'us/step':>8s}"
    )

    for name, hidden in _parse_shapes(args.shapes):
        for num_tokens in args.num_tokens:
            x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
            quantize_us = _graph_time_us(lambda x=x: torch.ops.trtllm.mxfp8_quantize(x, True), args)
            cast_us = _graph_time_us(lambda x=x: x.to(torch.float8_e4m3fn), args)
            print(
                f"{name:>8s} {hidden:7d} {num_tokens:7d} "
                f"{x.numel() * x.element_size() / 1024.0:7.0f} "
                f"{quantize_us:9.2f} {cast_us:7.2f} "
                f"{quantize_us - floor:11.2f} {quantize_us * args.layers:8.0f}"
            )

    print(
        f"\nus/step assumes {args.layers} layers and is what fusing the quantize "
        "into the producer epilogue would recover in full."
    )
    print(
        "quantize close to the floor means launch-bound, so only fusion helps. "
        "quantize far above both the floor and cast means the kernel itself is "
        "the problem at these shapes."
    )


if __name__ == "__main__":
    main()
