# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Microbenchmark the NVFP4-to-FP8 sparse MLA KV-cache gather.

The defaults reproduce the GLM-5.2 generation shape observed in TensorRT-LLM:

* FP8 MLA head dimension: 576 (512 latent + 64 RoPE)
* selected KV rows per request: 2048
* batch sweep: 32, 64, 128, 256, 512, and 1024

Example:

.. code-block:: bash

    python tests/microbenchmarks/bench_nvfp4_mla_kv_cache_gather.py

    python tests/microbenchmarks/bench_nvfp4_mla_kv_cache_gather.py \
        --mode both --iters 200 --output-json /tmp/nvfp4_gather.json

The reported CUDA-event interval contains only the gather (or its CUDA Graph
replay). L2 flushing, when enabled, is issued before the start event and is
therefore excluded from the latency.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch

# Import TensorRT-LLM to register torch.ops.trtllm custom operators.
import tensorrt_llm  # noqa: F401

DEFAULT_BATCHES = (32, 64, 128, 256, 512, 1024)
DEFAULT_HEAD_DIM = 576
DEFAULT_TOP_K = 2048


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
        help="Batch sizes to benchmark (default: 32 64 128 256 512 1024).",
    )
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--global-dequant-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("cuda_graph", "eager", "both"),
        default="cuda_graph",
        help="Launch mode. Production generation normally uses cuda_graph.",
    )
    parser.add_argument(
        "--index-order",
        choices=("random", "contiguous"),
        default="random",
        help="Order of unique physical KV rows read by the gather.",
    )
    parser.add_argument(
        "--pool-tokens",
        type=int,
        default=None,
        help="Physical rows in the source pool (default: max(batch) * top_k).",
    )
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flush L2 before every measured launch (default: enabled).",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate compact indices and one dequantized output row.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.batches or any(batch <= 0 for batch in args.batches):
        raise ValueError("--batches must contain positive integers")
    if args.head_dim <= 0 or args.head_dim % 16 != 0:
        raise ValueError("--head-dim must be a positive multiple of 16")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _make_cuda_graph_runner(
    launch: Callable[[], None], stream: torch.cuda.Stream
) -> tuple[Callable[[], None], torch.cuda.CUDAGraph]:
    # Initialize lazy library state before capture. The custom op itself does
    # not allocate device memory, so one eager invocation is sufficient.
    with torch.cuda.stream(stream):
        launch()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    return graph.replay, graph


def _check_result(
    global_indices: torch.Tensor,
    output: torch.Tensor,
    compact_indices: torch.Tensor,
    data_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    global_dequant_scale: torch.Tensor,
) -> None:
    num_pairs = global_indices.numel()
    expected_compact = torch.arange(num_pairs, dtype=torch.int32, device=global_indices.device)
    if not torch.equal(compact_indices.view(-1), expected_compact):
        raise RuntimeError("compact_indices validation failed")

    source_row = int(global_indices[0, 0].item())
    packed = data_pool[source_row]
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten().long()
    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=output.device,
    )
    scales = scale_pool[source_row].float().repeat_interleave(16)
    expected_output = (e2m1[codes] * scales * global_dequant_scale[0]).to(torch.float8_e4m3fn)
    if not torch.equal(output[0, 0], expected_output):
        raise RuntimeError("dequantized output validation failed")


def _benchmark_case(
    *,
    batch: int,
    mode: str,
    top_k: int,
    head_dim: int,
    num_pool_tokens: int,
    host_pool_pointers: torch.Tensor,
    host_pool_mapping: torch.Tensor,
    global_indices: torch.Tensor,
    output: torch.Tensor,
    compact_indices: torch.Tensor,
    global_dequant_scale: torch.Tensor,
    data_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    l2_buffer: torch.Tensor | None,
    warmup: int,
    iters: int,
    check: bool,
) -> dict[str, float | int | str]:
    indices_view = global_indices[:batch]
    output_view = output[:batch]
    compact_view = compact_indices[:batch]
    stream = torch.cuda.Stream(device=global_indices.device)
    stream.wait_stream(torch.cuda.current_stream(global_indices.device))

    def launch() -> None:
        torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
            host_pool_pointers,
            host_pool_mapping,
            indices_view,
            output_view,
            compact_view,
            global_dequant_scale,
            0,
            num_pool_tokens,
        )

    graph = None
    runner: Callable[[], None] = launch
    if mode == "cuda_graph":
        runner, graph = _make_cuda_graph_runner(launch, stream)

    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(warmup):
            if l2_buffer is not None:
                l2_buffer.zero_()
            runner()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.nvtx.range_push(f"nvfp4_mla_gather batch={batch} mode={mode}")
    with torch.inference_mode(), torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            if l2_buffer is not None:
                l2_buffer.zero_()
            start.record(stream)
            runner()
            end.record(stream)
    stream.synchronize()
    torch.cuda.nvtx.range_pop()

    times_us = [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]
    if check:
        _check_result(
            indices_view,
            output_view,
            compact_view,
            data_pool,
            scale_pool,
            global_dequant_scale,
        )

    num_pairs = batch * top_k
    # Per selected row: packed FP4 + FP8 block scales + FP8 output, plus one
    # int32 global-index read and one int32 compact-index write.
    bytes_per_pair = head_dim // 2 + head_dim // 16 + head_dim + 2 * torch.int32.itemsize
    mean_us = statistics.mean(times_us)
    result: dict[str, float | int | str] = {
        "mode": mode,
        "batch": batch,
        "pairs": num_pairs,
        "mean_us": mean_us,
        "p50_us": statistics.median(times_us),
        "p95_us": _percentile(times_us, 0.95),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "effective_gbps": num_pairs * bytes_per_pair / (mean_us * 1e3),
        "output_mib": num_pairs * head_dim / (1 << 20),
    }
    # Keep the graph alive until all event timings and validation are done.
    del graph
    return result


def _allocate_inputs(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    max_batch = max(args.batches)
    max_pairs = max_batch * args.top_k
    num_pool_tokens = args.pool_tokens or max_pairs
    if num_pool_tokens < max_pairs:
        raise ValueError(
            f"--pool-tokens ({num_pool_tokens}) must be at least max(batch) * top_k ({max_pairs})"
        )
    if num_pool_tokens > torch.iinfo(torch.int32).max:
        raise ValueError("--pool-tokens exceeds the int32 index range")

    packed_head_dim = args.head_dim // 2
    scales_per_token = args.head_dim // 16
    tensor_bytes = num_pool_tokens * (packed_head_dim + scales_per_token) + max_pairs * (
        args.head_dim + 2 * torch.int32.itemsize
    )
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size * 2 if args.flush_l2 else 0
    peak_index_bytes = num_pool_tokens * torch.int32.itemsize if args.index_order == "random" else 0
    peak_bytes = tensor_bytes + l2_bytes + peak_index_bytes
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if peak_bytes > free_bytes * 0.9:
        raise RuntimeError(
            f"Benchmark setup needs up to {peak_bytes / (1 << 30):.2f} GiB, "
            f"but only {free_bytes / (1 << 30):.2f} GiB is free"
        )

    data_pool = torch.randint(
        0,
        256,
        (num_pool_tokens, packed_head_dim),
        dtype=torch.uint8,
        device=device,
    )
    scale_pool = torch.ones(
        (num_pool_tokens, scales_per_token),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    if args.index_order == "random":
        permutation = torch.randperm(num_pool_tokens, dtype=torch.int32, device=device)
        selected_indices = (
            permutation[:max_pairs].clone() if num_pool_tokens > max_pairs else permutation
        )
        del permutation
    else:
        selected_indices = torch.arange(max_pairs, dtype=torch.int32, device=device)
    global_indices = selected_indices.view(max_batch, args.top_k)
    output = torch.empty(
        (max_batch, args.top_k, args.head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    compact_indices = torch.empty_like(global_indices)
    global_dequant_scale = torch.full(
        (1,), args.global_dequant_scale, dtype=torch.float32, device=device
    )

    host_pool_pointers = torch.zeros((1, 2, 2), dtype=torch.int64)
    host_pool_pointers[0, 0, 0] = data_pool.data_ptr()
    host_pool_pointers[0, 0, 1] = scale_pool.data_ptr()
    host_pool_mapping = torch.tensor([[0, 0]], dtype=torch.int32)

    l2_buffer = None
    if args.flush_l2:
        l2_buffer = torch.empty(l2_bytes, dtype=torch.uint8, device=device)

    return {
        "num_pool_tokens": num_pool_tokens,
        "host_pool_pointers": host_pool_pointers,
        "host_pool_mapping": host_pool_mapping,
        "global_indices": global_indices,
        "output": output,
        "compact_indices": compact_indices,
        "global_dequant_scale": global_dequant_scale,
        "data_pool": data_pool,
        "scale_pool": scale_pool,
        "l2_buffer": l2_buffer,
        "tensor_bytes": tensor_bytes,
    }


def _print_results(results: list[dict[str, float | int | str]]) -> None:
    print()
    print(
        f"{'mode':>10} {'batch':>6} {'pairs':>10} {'mean_us':>10} "
        f"{'p50_us':>10} {'p95_us':>10} {'min_us':>10} {'GB/s':>9} {'out_MiB':>9}"
    )
    for result in results:
        print(
            f"{result['mode']:>10} {result['batch']:6d} {result['pairs']:10d} "
            f"{result['mean_us']:10.3f} {result['p50_us']:10.3f} "
            f"{result['p95_us']:10.3f} {result['min_us']:10.3f} "
            f"{result['effective_gbps']:9.1f} {result['output_mib']:9.1f}"
        )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"--device must be in [0, {torch.cuda.device_count() - 1}]")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 10:
        raise RuntimeError(f"NVFP4 gather requires a Blackwell GPU, got {properties.name}")
    if not hasattr(torch.ops.trtllm, "nvfp4_mla_kv_cache_gather"):
        raise RuntimeError(
            "trtllm::nvfp4_mla_kv_cache_gather is unavailable; rebuild the C++ extension"
        )

    torch.manual_seed(args.seed)
    tensors = _allocate_inputs(args, device)
    torch.cuda.synchronize(device)
    modes = ("eager", "cuda_graph") if args.mode == "both" else (args.mode,)

    print(f"Device: {properties.name} ({properties.multi_processor_count} SMs)")
    print(f"Shape: head_dim={args.head_dim}, top_k={args.top_k}")
    print(f"Global dequant scale: {args.global_dequant_scale}")
    print(f"Batches: {args.batches}")
    print(f"Source pool: {tensors['num_pool_tokens']} physical rows ({args.index_order})")
    print(f"L2 flush: {args.flush_l2}; warmup={args.warmup}; iterations={args.iters}")
    print(f"Tensor footprint (excluding L2 flush): {tensors['tensor_bytes'] / (1 << 30):.2f} GiB")

    results = []
    for mode in modes:
        for batch in args.batches:
            results.append(
                _benchmark_case(
                    batch=batch,
                    mode=mode,
                    top_k=args.top_k,
                    head_dim=args.head_dim,
                    num_pool_tokens=tensors["num_pool_tokens"],
                    host_pool_pointers=tensors["host_pool_pointers"],
                    host_pool_mapping=tensors["host_pool_mapping"],
                    global_indices=tensors["global_indices"],
                    output=tensors["output"],
                    compact_indices=tensors["compact_indices"],
                    global_dequant_scale=tensors["global_dequant_scale"],
                    data_pool=tensors["data_pool"],
                    scale_pool=tensors["scale_pool"],
                    l2_buffer=tensors["l2_buffer"],
                    warmup=args.warmup,
                    iters=args.iters,
                    check=args.check,
                )
            )
    _print_results(results)

    if args.output_json is not None:
        payload = {
            "device": properties.name,
            "sm_count": properties.multi_processor_count,
            "head_dim": args.head_dim,
            "top_k": args.top_k,
            "global_dequant_scale": args.global_dequant_scale,
            "batches": args.batches,
            "pool_tokens": tensors["num_pool_tokens"],
            "index_order": args.index_order,
            "flush_l2": args.flush_l2,
            "warmup": args.warmup,
            "iterations": args.iters,
            "results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
