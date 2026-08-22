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

"""Microbenchmark GLM-5.2 NVFP4 dequantization and sparse MLA core.

The defaults reproduce the GLM-5.2 generation shape observed in TensorRT-LLM:

* FP8 MLA head dimension: 576 (512 latent + 64 RoPE)
* sparse TopK per request: min(sequence length, 2048)
* batch sweep: 32, 64, 128, 256, 512, and 1024

Each case measures the NVFP4-to-FP8 gather and the live TRTLLMGen static
sparse MLA core separately, then reports their ratio and combined latency.

Example:

.. code-block:: bash

    python tests/microbenchmarks/bench_nvfp4_mla_kv_cache_gather.py

    python tests/microbenchmarks/bench_nvfp4_mla_kv_cache_gather.py \
        --mode both --iters 200 --output-json /tmp/nvfp4_gather.json

The gather interval contains only the gather (or its CUDA Graph replay). L2
flushing, when enabled, is issued before its start event and is excluded from
latency. The core reads a warmed FP8 scratch pool, matching its placement
immediately after dequantization in production.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from types import MethodType
from typing import Callable

import torch

# Import TensorRT-LLM to register torch.ops.trtllm custom operators.
import tensorrt_llm  # noqa: F401

DEFAULT_BATCHES = (32, 64, 128, 256, 512, 1024)
DEFAULT_HEAD_DIM = 576
DEFAULT_TOP_K = 2048
DEFAULT_SEQ_LEN = 2048


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
    parser.add_argument(
        "--residual-dim",
        type=int,
        default=64,
        help="Trailing RoPE dimensions stored with a second residual NVFP4 level.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="KV sequence length; sparse TopK is min(seq-len, 2048).",
    )
    parser.add_argument("--global-dequant-scale", type=float, default=1.0)
    parser.add_argument(
        "--cache-layout",
        choices=("split", "compact"),
        default="split",
        help="Source KV layout to benchmark (default: split).",
    )
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
    args = parser.parse_args()
    args.top_k = min(args.seq_len, DEFAULT_TOP_K)
    args.core_top_k = math.ceil(args.top_k / 4) * 4
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.batches or any(batch <= 0 for batch in args.batches):
        raise ValueError("--batches must contain positive integers")
    if args.head_dim <= 0 or args.head_dim % 16 != 0:
        raise ValueError("--head-dim must be a positive multiple of 16")
    if args.residual_dim < 0 or args.residual_dim > args.head_dim or args.residual_dim % 16 != 0:
        raise ValueError("--residual-dim must be a multiple of 16 in [0, head-dim]")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.cache_layout == "compact" and (
        args.head_dim != DEFAULT_HEAD_DIM or args.global_dequant_scale != 1.0
    ):
        raise ValueError(
            "compact layout prototype requires --head-dim 576 and --global-dequant-scale 1"
        )


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
    head_dim: int,
    residual_dim: int,
) -> None:
    num_pairs = global_indices.numel()
    expected_compact = torch.arange(num_pairs, dtype=torch.int32, device=global_indices.device)
    if not torch.equal(compact_indices.view(-1), expected_compact):
        raise RuntimeError("compact_indices validation failed")

    source_row = int(global_indices[0, 0].item())
    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=output.device,
    )
    residual_start = head_dim - residual_dim
    packed = data_pool[source_row]
    main_packed = packed[: residual_start // 2]
    main_codes = torch.stack((main_packed & 0xF, main_packed >> 4), dim=-1).flatten().long()
    expected_parts = [
        e2m1[main_codes]
        * scale_pool[source_row, : residual_start // 16].float().repeat_interleave(16)
    ]
    for group in range(residual_dim // 16):
        data_offset = residual_start // 2 + group * 16
        main_group = packed[data_offset : data_offset + 8]
        residual_group = packed[data_offset + 8 : data_offset + 16]
        main_codes = torch.stack((main_group & 0xF, main_group >> 4), dim=-1).flatten().long()
        residual_codes = (
            torch.stack((residual_group & 0xF, residual_group >> 4), dim=-1).flatten().long()
        )
        scale_offset = residual_start // 16 + group * 2
        expected_parts.append(
            e2m1[main_codes] * scale_pool[source_row, scale_offset].float()
            + e2m1[residual_codes] * scale_pool[source_row, scale_offset + 1].float()
        )
    expected_output = (torch.cat(expected_parts) * global_dequant_scale[0]).to(torch.float8_e4m3fn)
    if not torch.equal(output[0, 0], expected_output):
        raise RuntimeError("dequantized output validation failed")


def _benchmark_case(
    *,
    batch: int,
    mode: str,
    cache_layout: str,
    top_k: int,
    head_dim: int,
    residual_dim: int,
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
            residual_dim,
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
            head_dim,
            residual_dim,
        )

    num_pairs = batch * top_k
    # Per selected row: packed FP4 + FP8 block scales + FP8 output, plus one
    # int32 global-index read and one int32 compact-index write.
    source_bytes = (head_dim + residual_dim) // 2 + (head_dim + residual_dim) // 16
    if cache_layout == "compact":
        source_bytes = math.ceil(source_bytes / 16) * 16
    bytes_per_pair = source_bytes + head_dim + 2 * torch.int32.itemsize
    mean_us = statistics.mean(times_us)
    result: dict[str, float | int | str] = {
        "mode": mode,
        "cache_layout": cache_layout,
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


def _build_glm52_sparse_mla_core(
    batch: int, seq_len: int, top_k: int, device: torch.device
) -> tuple[Callable[[], None], dict[str, object]]:
    """Build the GLM-5.2 TRTLLMGen static sparse MLA core in isolation."""
    from attention_perf.attention_perf_harness import (
        AttnCase,
        _build_mla_gen_metadata,
        _build_mla_kv_cache_manager,
        _mla_kv_head_dim,
        _mla_params,
        _mla_pos_embd_and_scaling,
        _mla_quant_config,
    )

    from tensorrt_llm._torch.attention_backend.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention_backend.sparse.dsa.backend import DSATrtllmAttention
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    case = AttnCase(
        case_id="glm52_sparse_mla_core",
        phase="generation",
        dtype=torch.bfloat16,
        num_heads=64,
        batch_size=batch,
        page_size=32,
        seq_len=1,
        num_cached_tokens=seq_len - 1,
        is_mla=True,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        hidden_size=6144,
        kv_cache_fp8=True,
    )
    kv_manager = _build_mla_kv_cache_manager(case)
    metadata = _build_mla_gen_metadata(case, kv_manager)
    metadata.num_sparse_topk = top_k

    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=top_k,
    )
    sparse_params = sparse_config.to_sparse_params(layer_idx=0)
    pos_embd_params, q_scaling = _mla_pos_embd_and_scaling(case)
    backend = DSATrtllmAttention(
        layer_idx=0,
        num_heads=case.num_heads,
        head_dim=_mla_kv_head_dim(case),
        num_kv_heads=1,
        quant_config=_mla_quant_config(case),
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=_mla_params(case),
        sparse_params=sparse_params,
        skip_create_weights_in_init=True,
        dtype=case.dtype,
    )
    backend.update_quant_config(_mla_quant_config(case))
    backend.local_layer_idx = 0

    kv_head_dim = case.kv_lora_rank + case.qk_rope_head_dim
    q = torch.ones((batch, case.num_heads * kv_head_dim), dtype=case.dtype, device=device)
    kv_scratch = torch.ones((batch, top_k, kv_head_dim), dtype=torch.float8_e4m3fn, device=device)
    compact_indices = torch.arange(batch * top_k, dtype=torch.int32, device=device).view(
        batch, top_k
    )
    output = torch.empty(
        (batch, case.num_heads * case.kv_lora_rank), dtype=case.dtype, device=device
    )
    latent_cache = torch.ones((batch, kv_head_dim), dtype=case.dtype, device=device)
    q_pe = torch.ones(
        (batch, case.num_heads, case.qk_rope_head_dim), dtype=case.dtype, device=device
    )
    quant_q_buffer = torch.ones(
        (batch, case.num_heads * kv_head_dim), dtype=torch.float8_e4m3fn, device=device
    ).view(torch.uint8)
    cu_q_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device) * case.num_heads
    cu_kv_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device) * seq_len
    fmha_scheduler_counter = torch.zeros(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = torch.tensor([1.0, math.log2(math.e)], dtype=torch.float32, device=device)
    mla_bmm2_scale = torch.ones(1, dtype=torch.float32, device=device)

    def _sparse_attn_predict(_self, _q, _k, _metadata, forward_args) -> tuple[torch.Tensor, None]:
        forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr = kv_scratch.data_ptr()
        return compact_indices, None

    backend.sparse_attn_predict = MethodType(_sparse_attn_predict, backend)

    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        output=output,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
        latent_cache=latent_cache,
        q_pe=q_pe,
    )

    def launch() -> None:
        backend.forward(q, None, None, metadata, forward_args=forward_args)

    resources = {
        "backend": backend,
        "metadata": metadata,
        "kv_manager": kv_manager,
        "q": q,
        "kv_scratch": kv_scratch,
        "compact_indices": compact_indices,
        "output": output,
        "latent_cache": latent_cache,
        "q_pe": q_pe,
        "quant_q_buffer": quant_q_buffer,
        "cu_q_seqlens": cu_q_seqlens,
        "cu_kv_seqlens": cu_kv_seqlens,
        "fmha_scheduler_counter": fmha_scheduler_counter,
        "mla_bmm1_scale": mla_bmm1_scale,
        "mla_bmm2_scale": mla_bmm2_scale,
        "forward_args": forward_args,
    }
    return launch, resources


def _benchmark_glm52_sparse_mla_core(
    *,
    batch: int,
    seq_len: int,
    top_k: int,
    mode: str,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    launch, resources = _build_glm52_sparse_mla_core(batch, seq_len, top_k, device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    graph = None
    runner: Callable[[], None] = launch
    if mode == "cuda_graph":
        runner, graph = _make_cuda_graph_runner(launch, stream)

    with torch.inference_mode(), torch.cuda.stream(stream):
        for _ in range(warmup):
            runner()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.inference_mode(), torch.cuda.stream(stream):
        for start, end in zip(starts, ends):
            start.record(stream)
            runner()
            end.record(stream)
    stream.synchronize()
    times_us = [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends)]

    del graph
    kv_manager = resources["kv_manager"]
    if hasattr(kv_manager, "shutdown"):
        kv_manager.shutdown()
    return {
        "core_mean_us": statistics.mean(times_us),
        "core_p50_us": statistics.median(times_us),
        "core_p95_us": _percentile(times_us, 0.95),
    }


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

    packed_head_dim = (args.head_dim + args.residual_dim) // 2
    scales_per_token = (args.head_dim + args.residual_dim) // 16
    source_row_bytes = packed_head_dim + scales_per_token
    if args.cache_layout == "compact":
        source_row_bytes = math.ceil(source_row_bytes / 16) * 16
    tensor_bytes = num_pool_tokens * source_row_bytes + max_pairs * (
        args.head_dim + 2 * torch.int32.itemsize
    )
    l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size * 2 if args.flush_l2 else 0
    peak_index_bytes = num_pool_tokens * torch.int32.itemsize if args.index_order == "random" else 0
    compact_setup_bytes = (
        num_pool_tokens * (packed_head_dim + scales_per_token)
        if args.cache_layout == "compact"
        else 0
    )
    peak_bytes = tensor_bytes + compact_setup_bytes + l2_bytes + peak_index_bytes
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
    if args.cache_layout == "compact":
        compact_pool = torch.empty(
            (num_pool_tokens, source_row_bytes), dtype=torch.uint8, device=device
        )
        compact_pool[:, :packed_head_dim].copy_(data_pool)
        compact_scale_bytes = compact_pool[:, packed_head_dim : packed_head_dim + scales_per_token]
        compact_scale_bytes.copy_(scale_pool.view(torch.uint8))
        del data_pool, scale_pool
        data_pool = compact_pool[:, :packed_head_dim]
        scale_pool = compact_scale_bytes.view(torch.float8_e4m3fn)
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
    if args.cache_layout == "compact":
        host_pool_pointers[0, 0, 0] = compact_pool.data_ptr()
        host_pool_pointers[0, 0, 1] = compact_pool.data_ptr() + packed_head_dim
    else:
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
    if results and "core_mean_us" in results[0]:
        print(
            f"{'mode':>10} {'batch':>6} {'pairs':>10} {'dequant_us':>11} "
            f"{'core_us':>10} {'DQ/core':>8} {'total_us':>10} {'GB/s':>9}"
        )
        for result in results:
            print(
                f"{result['mode']:>10} {result['batch']:6d} {result['pairs']:10d} "
                f"{result['mean_us']:11.3f} {result['core_mean_us']:10.3f} "
                f"{result['dequant_core_ratio']:8.3f} {result['dequant_core_total_us']:10.3f} "
                f"{result['effective_gbps']:9.1f}"
            )
        return

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
    print(
        f"Shape: head_dim={args.head_dim}, seq_len={args.seq_len}, "
        f"residual_dim={args.residual_dim}, top_k={args.top_k}, core_top_k={args.core_top_k}"
    )
    print(f"Global dequant scale: {args.global_dequant_scale}")
    print(f"Cache layout: {args.cache_layout}")
    print(f"Batches: {args.batches}")
    print(f"Source pool: {tensors['num_pool_tokens']} physical rows ({args.index_order})")
    print(f"L2 flush: {args.flush_l2}; warmup={args.warmup}; iterations={args.iters}")
    print(f"Tensor footprint (excluding L2 flush): {tensors['tensor_bytes'] / (1 << 30):.2f} GiB")

    results = []
    for mode in modes:
        for batch in args.batches:
            result = _benchmark_case(
                batch=batch,
                mode=mode,
                cache_layout=args.cache_layout,
                top_k=args.top_k,
                head_dim=args.head_dim,
                residual_dim=args.residual_dim,
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
            core_result = _benchmark_glm52_sparse_mla_core(
                batch=batch,
                seq_len=args.seq_len,
                top_k=args.core_top_k,
                mode=mode,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            result.update(core_result)
            result["dequant_core_ratio"] = result["mean_us"] / core_result["core_mean_us"]
            result["dequant_core_total_us"] = result["mean_us"] + core_result["core_mean_us"]
            results.append(result)
    _print_results(results)

    if args.output_json is not None:
        payload = {
            "device": properties.name,
            "sm_count": properties.multi_processor_count,
            "head_dim": args.head_dim,
            "residual_dim": args.residual_dim,
            "seq_len": args.seq_len,
            "top_k": args.top_k,
            "core_top_k": args.core_top_k,
            "global_dequant_scale": args.global_dequant_scale,
            "cache_layout": args.cache_layout,
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
