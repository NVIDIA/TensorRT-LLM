# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Is MiniMax-M3's standalone MXFP8 activation quantize launch-bound or slow?

``MXFP8LinearMethod.apply`` calls ``torch.ops.trtllm.mxfp8_quantize`` before
every base GEMM, and the profile charges 296 us/step to the qkv quantize and
115 us/step to the o_proj one at a concurrency-1 operating point.

Folding those into their producers' epilogues would remove the kernel outright,
but for qkv that means writing an MXFP8 epilogue into the AllReduce fusion
kernel, which has no MXFP8 path at all today. Whether that is worth building
depends on how the cost behaves across the batch range, not at one point:

* If the quantize is **launch-bound** at small batch, only fusion recovers it
  there, and tuning the kernel cannot.
* If it approaches **bandwidth-bound** at large batch, fusion still wins,
  because it removes an HBM round-trip of the activation rather than a launch.
  A win at both ends justifies the CUDA work.
* If it is **far off bandwidth at every size**, the kernel itself is the
  problem, and tuning it is far cheaper and helps every MXFP8 caller.

Columns per shape and token count:

* ``quantize`` -- ``mxfp8_quantize`` in the swizzled layout the GEMM wants.
* ``linear``   -- the same op in the linear layout, which pads neither rows nor
  the grid. See below.
* ``cast``     -- ``x.to(e4m3)``, an empirical bandwidth reference over the same
  input bytes with no block scales. The gap is what computing the scales and
  writing the SF layout costs.
* ``floor+``   -- time above the harness floor, which is the per-launch cost
  inside a CUDA graph. Near zero means launch-bound.
* ``GB/s``     -- achieved bandwidth over bytes read plus written, to place the
  large-batch end against the device's roofline.

A flat cost across small token counts has two possible causes, and separating
them decides whether fusion is the only remedy. The swizzled layout pads the row
count up to 128 and sizes the grid from the padded count, so one token still
launches 128 blocks and writes a full 128 rows of SF padding one byte at a time.
Two columns in this sweep tell those apart. Padding cost is quantized to 128
rows, so if it dominates, 129 tokens costs about what 256 costs and far more
than 128 does, whereas a launch-bound kernel shows 128 and 129 as identical. The
``linear`` column is the independent check, since that layout skips row padding
altogether and so pays only the launch.

Many calls go into one CUDA graph, since a single-call graph measures the ~6us
host cost of ``graph.replay()`` instead of the kernel.

    python tests/microbenchmarks/bench_mxfp8_quantize_small_batch.py
"""

import argparse

import torch

# Registers the torch.ops.trtllm namespace, without which mxfp8_quantize looks
# absent no matter what the build contains.
import tensorrt_llm  # noqa: F401  isort: skip

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
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        # 128 and 129 straddle the SF row-padding boundary on purpose.
        default=[1, 4, 16, 64, 128, 129, 256, 1024, 4096],
    )
    parser.add_argument("--layers", type=int, default=M3_LAYERS)
    parser.add_argument("--calls-per-graph", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not hasattr(torch.ops, "trtllm"):
        raise SystemExit(
            "The torch.ops.trtllm namespace is missing, so the TensorRT-LLM "
            "extension did not load. This is an import problem, not a missing op."
        )
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
        f"{'shape':>8s} {'hidden':>7s} {'tokens':>7s} {'KB in':>8s} "
        f"{'quantize':>9s} {'linear':>7s} {'cast':>7s} {'floor+':>8s} "
        f"{'GB/s':>7s} {'us/step':>8s}"
    )

    for name, hidden in _parse_shapes(args.shapes):
        for num_tokens in args.num_tokens:
            x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
            quantize_us = _graph_time_us(lambda x=x: torch.ops.trtllm.mxfp8_quantize(x, True), args)
            linear_us = _graph_time_us(lambda x=x: torch.ops.trtllm.mxfp8_quantize(x, False), args)
            cast_us = _graph_time_us(lambda x=x: x.to(torch.float8_e4m3fn), args)
            # bf16 in, e4m3 out, plus one UE8M0 scale byte per 32 elements.
            bytes_moved = x.numel() * (2 + 1 + 1 / 32)
            print(
                f"{name:>8s} {hidden:7d} {num_tokens:7d} "
                f"{x.numel() * x.element_size() / 1024.0:8.0f} "
                f"{quantize_us:9.2f} {linear_us:7.2f} {cast_us:7.2f} "
                f"{quantize_us - floor:8.2f} "
                f"{bytes_moved / quantize_us / 1000.0:7.0f} "
                f"{quantize_us * args.layers:8.0f}"
            )

    print(
        f"\nus/step assumes {args.layers} layers and is what fusing the quantize "
        "into the producer epilogue would recover in full, at that token count."
    )
    print(
        "floor+ near zero means launch-bound, where only fusion helps. GB/s "
        "approaching the device roofline at the large-batch end means the "
        "remaining cost is the round-trip, which fusion also removes. Far off "
        "both at every size means the kernel itself is the problem."
    )
    print(
        "A jump between 128 and 129 tokens, or a linear column well under the "
        "quantize one at small batch, means the small-batch cost is SF row "
        "padding rather than the launch, and tuning that path recovers it "
        "without an epilogue fusion."
    )


if __name__ == "__main__":
    main()
