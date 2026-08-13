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

"""Compare PrimTS one-shot, raw, and reusable-wrapper attention APIs.

Run this benchmark on a Blackwell GPU from a TensorRT-LLM source checkout whose
Python environment contains PyTorch, CuTe DSL 4.7, and TVM FFI. The default
shape is deliberately small enough for a quick development run; increase the
iteration counts for stable measurements.

This measures the PrimTS Python APIs in isolation, not end-to-end TensorRT-LLM
attention latency. In particular, the context and standard-decode wrappers
retain planned page metadata and are only reusable while that metadata remains
unchanged. The MLA wrapper retains live block tables and sequence lengths, so
it is the only cached-wrapper path that maps directly to token-by-token reuse.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from importlib import metadata
from pathlib import Path
from typing import TypeVar

import torch

# Import the vendored package without importing ``tensorrt_llm`` itself. This
# keeps the microbenchmark runnable from a source-only worktree that does not
# contain the ignored TensorRT-LLM Python binding shared object.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ATTENTION_BACKEND_ROOT = _REPO_ROOT / "tensorrt_llm" / "_torch" / "attention_backend"
sys.path.insert(0, str(_ATTENTION_BACKEND_ROOT))

from prims_ts import (  # noqa: E402
    BatchDecodePagedTSWrapper,
    BatchMLADecodePagedTSWrapper,
    BatchPrefillPagedTSWrapper,
    batch_decode_mla_with_paged_kv_cache,
    batch_decode_with_paged_kv_cache,
    batch_prefill_with_paged_kv_cache,
    get_prims_ts_batch_decode_mla_workspace_size,
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
    prims_ts_batch_decode_with_kv_cache_mla,
)

T = TypeVar("T")


def _cuda_event_interval_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "cuda_event_interval_mean_ms": statistics.mean(values),
        "cuda_event_interval_median_ms": statistics.median(values),
        "cuda_event_interval_p10_ms": ordered[max(math.ceil(0.10 * len(ordered)) - 1, 0)],
        "cuda_event_interval_p90_ms": ordered[max(math.ceil(0.90 * len(ordered)) - 1, 0)],
        "cuda_event_interval_min_ms": ordered[0],
        "cuda_event_interval_max_ms": ordered[-1],
    }


def _event_bench(
    name: str,
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.nvtx.range_push(f"measure/{name}")
    wall_start = time.perf_counter()
    for index in range(iters):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3 / iters
    torch.cuda.nvtx.range_pop()

    result = _cuda_event_interval_stats(
        [starts[index].elapsed_time(ends[index]) for index in range(iters)]
    )
    result["synchronized_wall_ms_per_call"] = wall_ms
    return result


def _measure_once(name: str, fn: Callable[[], T]) -> tuple[T, dict[str, float]]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.nvtx.range_push(name)
    wall_start = time.perf_counter()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    torch.cuda.nvtx.range_pop()
    cuda_event_interval_ms = start.elapsed_time(end)
    return result, {
        "cuda_event_interval_ms": cuda_event_interval_ms,
        "synchronized_wall_ms": wall_ms,
    }


def _emit_nvtx(
    case_name: str,
    variants: Mapping[str, Callable[[], object]],
    iters: int,
) -> None:
    if iters <= 0:
        return
    torch.cuda.synchronize()
    for label, fn in variants.items():
        torch.cuda.nvtx.range_push(f"profile/{case_name}/{label}")
        for _ in range(iters):
            fn()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def _bench_variants(
    case_name: str,
    variants: Mapping[str, Callable[[], object]],
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    measurements = {
        label: _event_bench(
            f"{case_name}/{label}",
            fn,
            warmup=args.warmup,
            iters=args.iters,
        )
        for label, fn in variants.items()
    }
    _emit_nvtx(case_name, variants, args.nvtx_iters)
    return measurements


def _sequence_lengths(batch_size: int, max_seq_len: int, page_size: int) -> list[int]:
    """Create one full-length row and a production-like ragged tail."""
    if batch_size == 1:
        return [max_seq_len]
    lengths = []
    for index in range(batch_size):
        fraction = 1.0 - 0.70 * index / (batch_size - 1)
        length = max(int(max_seq_len * fraction), page_size)
        lengths.append(min(length, max_seq_len))
    lengths[-1] = min(page_size, max_seq_len)
    return lengths


def _paged_metadata(
    lengths_host: list[int],
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    indptr_host = [0]
    for length in lengths_host:
        indptr_host.append(indptr_host[-1] + (length + page_size - 1) // page_size)
    num_pages = indptr_host[-1]
    return (
        torch.tensor(indptr_host, device=device, dtype=torch.int32),
        torch.arange(num_pages, device=device, dtype=torch.int32),
        torch.tensor(
            [((length - 1) % page_size) + 1 for length in lengths_host],
            device=device,
            dtype=torch.int32,
        ),
        num_pages,
    )


def _packed_indptr(lengths_host: list[int], device: torch.device) -> torch.Tensor:
    indptr_host = [0]
    for length in lengths_host:
        indptr_host.append(indptr_host[-1] + length)
    return torch.tensor(indptr_host, device=device, dtype=torch.int32)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _policy_dict(policy: tuple[tuple[str, object], ...]) -> dict[str, object]:
    return dict(policy)


def _correctness(
    reference: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    torch.cuda.synchronize()
    differences: dict[str, float] = {}
    for label, candidate in candidates.items():
        differences[label] = float((reference.float() - candidate.float()).abs().max().item())
        torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=5e-2)
    return differences


def _context_case(args: argparse.Namespace) -> dict[str, object]:
    # Qwen2-7B context geometry with packed, ragged Q and separate HND K/V pages.
    batch_size = args.batch_size
    num_qo_heads = 28
    num_kv_heads = 4
    head_dim = 128
    page_size = args.page_size
    max_seq_len = args.max_seq_len
    dtype = torch.bfloat16
    device = torch.device("cuda", args.device)
    kv_lengths_host = _sequence_lengths(batch_size, max_seq_len, page_size)
    q_lengths_host = [min(args.context_seq_len, length) for length in kv_lengths_host]
    qo_indptr = _packed_indptr(q_lengths_host, device)
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = _paged_metadata(
        kv_lengths_host, page_size, device
    )
    total_q = sum(q_lengths_host)
    query = torch.randn(total_q, num_qo_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_pages,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    one_shot_out = torch.empty_like(query)
    cached_out = torch.empty_like(query)
    construct_out = torch.empty_like(query)

    def current_one_shot() -> torch.Tensor:
        return batch_prefill_with_paged_kv_cache(
            query,
            k_cache,
            v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            page_size=page_size,
            mask_type="causal",
            out_dtype=dtype,
            out=one_shot_out,
        )

    _, first_one_shot = _measure_once("cold/context/current_one_shot", current_one_shot)

    def make_wrapper() -> BatchPrefillPagedTSWrapper:
        wrapper = BatchPrefillPagedTSWrapper(kv_layout="HND")
        wrapper.plan(
            query,
            k_cache,
            v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            page_size=page_size,
            mask_type="causal",
            out_dtype=dtype,
        )
        return wrapper

    cached_wrapper, cached_plan = _measure_once("plan/context/cached", make_wrapper)

    def plan_only() -> BatchPrefillPagedTSWrapper:
        return make_wrapper()

    hot_plan = _event_bench(
        "context/plan_only",
        plan_only,
        warmup=1,
        iters=args.plan_iters,
    )

    def cached_run() -> torch.Tensor:
        return cached_wrapper.run(query, k_cache, v_cache, out=cached_out)

    def construct_plan_run() -> torch.Tensor:
        wrapper = make_wrapper()
        return wrapper.run(query, k_cache, v_cache, out=construct_out)

    reference = current_one_shot()
    correctness = _correctness(
        reference,
        {
            "wrapper_cached": cached_run(),
            "wrapper_construct_plan_run": construct_plan_run(),
        },
    )
    variants = {
        "current_one_shot": current_one_shot,
        "wrapper_cached_static_metadata_run_only": cached_run,
        "wrapper_construct_plan_run": construct_plan_run,
    }
    planned_metadata_bytes = sum(
        _tensor_bytes(tensor)
        for tensor in (
            cached_wrapper._logical_kv_indptr,
            cached_wrapper._seq_lens_kv,
            cached_wrapper._dense_page_idx_kv,
            cached_wrapper._scale_softmax_log2,
            cached_wrapper._output_scale,
        )
    )
    return {
        "geometry": {
            "batch_size": batch_size,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "max_seq_len": max_seq_len,
            "q_lengths": q_lengths_host,
            "kv_lengths": kv_lengths_host,
        },
        "policy": _policy_dict(cached_wrapper._policy),
        "reuse_contract": (
            "The paged-context wrapper snapshots all planned page metadata. "
            "Its cached timing applies only while that metadata remains unchanged."
        ),
        "workspace_bytes": {
            "wrapper_scratch": 0,
            "wrapper_planned_metadata": planned_metadata_bytes,
        },
        "first_current_one_shot": first_one_shot,
        "cached_wrapper_plan_after_one_shot_compile": cached_plan,
        "hot_plan_only": hot_plan,
        "max_abs_diff": correctness,
        "variants": _bench_variants("context", variants, args),
    }


def _qwen_case(args: argparse.Namespace) -> dict[str, object]:
    # Qwen2-7B GQA decode geometry.
    batch_size = args.batch_size
    num_qo_heads = 28
    num_kv_heads = 4
    head_dim = 128
    page_size = args.page_size
    max_seq_len = args.max_seq_len
    dtype = torch.bfloat16
    device = torch.device("cuda", args.device)
    lengths_host = _sequence_lengths(batch_size, max_seq_len, page_size)
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = _paged_metadata(
        lengths_host, page_size, device
    )
    seq_lens = torch.tensor(lengths_host, device=device, dtype=torch.int32)
    query = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    kv_cache = torch.randn(
        num_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    raw_out = torch.empty_like(query)
    zeroed_raw_out = torch.empty_like(query)
    cached_out = torch.empty_like(query)
    construct_out = torch.empty_like(query)
    raw_workspace_bytes = get_prims_ts_batch_decode_workspace_size(
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        seq_len_q=1,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        device=device,
    )
    raw_workspace = torch.zeros(raw_workspace_bytes, device=device, dtype=torch.uint8)

    def raw_call(out: torch.Tensor) -> torch.Tensor:
        return prims_ts_batch_decode_with_kv_cache(
            query,
            kv_cache,
            raw_workspace,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens,
            max_seq_len,
            seq_len_q=1,
            out=out,
            out_dtype=dtype,
            mask_type="causal",
            kv_layout="HND",
        )

    def raw_launch_only() -> torch.Tensor:
        return raw_call(raw_out)

    def workspace_zero_plus_raw() -> torch.Tensor:
        raw_workspace.zero_()
        return raw_call(zeroed_raw_out)

    _, first_raw = _measure_once("cold/qwen/raw", raw_launch_only)

    def make_wrapper() -> BatchDecodePagedTSWrapper:
        wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
        wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            seq_len_q=1,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            max_kv_len=max_seq_len,
        )
        return wrapper

    cached_wrapper, cached_plan = _measure_once("plan/qwen/cached", make_wrapper)

    def plan_only() -> BatchDecodePagedTSWrapper:
        return make_wrapper()

    hot_plan = _event_bench(
        "qwen/plan_only",
        plan_only,
        warmup=1,
        iters=args.plan_iters,
    )

    def cached_run() -> torch.Tensor:
        return cached_wrapper.run(query, kv_cache, out=cached_out)

    def construct_plan_run() -> torch.Tensor:
        # This convenience API creates, plans, and discards a wrapper per call.
        return batch_decode_with_paged_kv_cache(
            query,
            kv_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            seq_len_q=1,
            mask_type="causal",
            kv_layout="HND",
            out=construct_out,
            out_dtype=dtype,
        )

    reference = raw_launch_only()
    correctness = _correctness(
        reference,
        {
            "workspace_zero_plus_raw": workspace_zero_plus_raw(),
            "wrapper_cached": cached_run(),
            "wrapper_construct_plan_run": construct_plan_run(),
        },
    )
    variants = {
        "raw_launch_only": raw_launch_only,
        "workspace_zero_plus_raw": workspace_zero_plus_raw,
        "wrapper_cached_static_metadata_run_only": cached_run,
        "wrapper_construct_plan_run": construct_plan_run,
    }
    return {
        "geometry": {
            "batch_size": batch_size,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "max_seq_len": max_seq_len,
            "seq_lens": lengths_host,
        },
        "policy": _policy_dict(cached_wrapper._policy),
        "modes": {
            "raw_kv_prefix_mode": "dynamic",
            "raw_kv_lengths_mode": "dynamic",
            "wrapper_kv_prefix_mode": cached_wrapper._kv_prefix_mode,
            "wrapper_kv_lengths_mode": cached_wrapper._kv_lengths_mode,
            "same_kernel_mode": (
                cached_wrapper._kv_prefix_mode == "dynamic"
                and cached_wrapper._kv_lengths_mode == "dynamic"
            ),
        },
        "reuse_contract": (
            "The standard decode wrapper snapshots CSR-derived sequence lengths. "
            "Its cached timing is a static-metadata experiment, not a drop-in "
            "replacement for TRT-LLM token-by-token generation."
        ),
        "workspace_bytes": {
            "raw": raw_workspace_bytes,
            "wrapper": _tensor_bytes(cached_wrapper._workspace_buffer),
        },
        "first_raw": first_raw,
        "cached_wrapper_plan_after_raw_compile": cached_plan,
        "hot_plan_only": hot_plan,
        "max_abs_diff": correctness,
        "variants": _bench_variants("qwen", variants, args),
    }


def _mla_case(args: argparse.Namespace) -> dict[str, object]:
    # DeepSeek-V3/R1 absorbed-MLA geometry.
    batch_size = args.batch_size
    num_heads = 128
    kv_lora_rank = 512
    rope_dim = 64
    page_size = args.page_size
    max_seq_len = args.max_seq_len
    max_pages_per_row = (max_seq_len + page_size - 1) // page_size
    dtype = torch.bfloat16
    device = torch.device("cuda", args.device)
    lengths_host = _sequence_lengths(batch_size, max_seq_len, page_size)
    seq_lens = torch.tensor(lengths_host, device=device, dtype=torch.int32)
    num_physical_pages = batch_size * max_pages_per_row
    block_tables = torch.arange(num_physical_pages, device=device, dtype=torch.int32).view(
        batch_size, max_pages_per_row
    )
    query = torch.randn(
        batch_size,
        1,
        num_heads,
        kv_lora_rank + rope_dim,
        device=device,
        dtype=dtype,
    )
    kv_cache = torch.randn(
        num_physical_pages,
        page_size,
        kv_lora_rank + rope_dim,
        device=device,
        dtype=dtype,
    )
    raw_out = torch.empty(batch_size, 1, num_heads, kv_lora_rank, device=device, dtype=dtype)
    cached_out = torch.empty_like(raw_out)
    construct_out = torch.empty_like(raw_out)
    raw_workspace_bytes = get_prims_ts_batch_decode_mla_workspace_size(
        batch_size,
        num_heads,
        kv_lora_rank,
        rope_dim,
        page_size,
        max_seq_len,
        max_seq_len_q=1,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        device=device,
    )
    raw_workspace = torch.zeros(raw_workspace_bytes, device=device, dtype=torch.uint8)
    bmm1_scale = 1.0 / math.sqrt(128 + rope_dim)

    def raw() -> torch.Tensor:
        return prims_ts_batch_decode_with_kv_cache_mla(
            query,
            kv_cache,
            raw_workspace,
            kv_lora_rank,
            rope_dim,
            block_tables,
            seq_lens,
            max_seq_len,
            max_seq_len_q=1,
            out=raw_out,
            bmm1_scale=bmm1_scale,
            out_dtype=dtype,
            mask_type="causal",
        )

    _, first_raw = _measure_once("cold/mla/raw", raw)

    def make_wrapper() -> BatchMLADecodePagedTSWrapper:
        wrapper = BatchMLADecodePagedTSWrapper()
        wrapper.plan(
            block_tables,
            seq_lens,
            num_heads,
            kv_lora_rank,
            rope_dim,
            page_size,
            max_seq_len_q=1,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            max_kv_len=max_seq_len,
        )
        return wrapper

    cached_wrapper, cached_plan = _measure_once("plan/mla/cached", make_wrapper)

    def plan_only() -> BatchMLADecodePagedTSWrapper:
        return make_wrapper()

    hot_plan = _event_bench(
        "mla/plan_only",
        plan_only,
        warmup=1,
        iters=args.plan_iters,
    )

    def cached_run() -> torch.Tensor:
        return cached_wrapper.run(
            query,
            kv_cache,
            bmm1_scale=bmm1_scale,
            out=cached_out,
        )

    def construct_plan_run() -> torch.Tensor:
        return batch_decode_mla_with_paged_kv_cache(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len_q=1,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=rope_dim,
            mask_type="causal",
            max_kv_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            out=construct_out,
            out_dtype=dtype,
        )

    reference = raw()
    correctness = _correctness(
        reference,
        {
            "wrapper_cached": cached_run(),
            "wrapper_construct_plan_run": construct_plan_run(),
        },
    )
    original_block_tables = block_tables.clone()
    original_seq_lens = seq_lens.clone()
    block_tables.copy_(torch.roll(block_tables, shifts=1, dims=1))
    seq_lens.copy_(torch.clamp(seq_lens - max(page_size // 2, 1), min=1))
    live_metadata_correctness = _correctness(
        raw(),
        {"wrapper_cached_after_live_metadata_update": cached_run()},
    )
    block_tables.copy_(original_block_tables)
    seq_lens.copy_(original_seq_lens)
    variants = {
        "raw_launch_only": raw,
        "wrapper_cached_run_only": cached_run,
        "wrapper_construct_plan_run": construct_plan_run,
    }
    return {
        "geometry": {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": rope_dim,
            "page_size": page_size,
            "max_seq_len": max_seq_len,
            "seq_lens": lengths_host,
        },
        "policy": _policy_dict(cached_wrapper._policy),
        "modes": {
            "raw": "live_seq_lens",
            "wrapper": "live_seq_lens",
        },
        "reuse_contract": (
            "The MLA wrapper reads block-table and sequence-length values live. "
            "Storage and the planned geometry must remain unchanged."
        ),
        "workspace_bytes": {
            "raw": raw_workspace_bytes,
            "wrapper": _tensor_bytes(cached_wrapper._workspace_buffer),
        },
        "first_raw": first_raw,
        "cached_wrapper_plan_after_raw_compile": cached_plan,
        "hot_plan_only": hot_plan,
        "max_abs_diff": correctness,
        "live_metadata_update_max_abs_diff": live_metadata_correctness,
        "variants": _bench_variants("mla", variants, args),
    }


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("context", "qwen", "mla", "all"),
        default="all",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--context-seq-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, choices=(16, 32, 64, 128), default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--plan-iters", type=int, default=10)
    parser.add_argument("--nvtx-iters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be positive")
    if args.context_seq_len <= 0:
        parser.error("--context-seq-len must be positive")
    if args.warmup < 0 or args.nvtx_iters < 0:
        parser.error("--warmup and --nvtx-iters must be non-negative")
    if args.iters <= 0 or args.plan_iters <= 0:
        parser.error("--iters and --plan-iters must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PrimTS performance measurement requires a CUDA GPU.")
    torch.cuda.set_device(args.device)
    major, minor = torch.cuda.get_device_capability(args.device)
    if major != 10:
        raise RuntimeError(f"PrimTS requires a Blackwell SM100/SM103 GPU, got sm_{major}{minor}.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    results: dict[str, object] = {
        "device": torch.cuda.get_device_name(args.device),
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "nvidia_cutlass_dsl_version": _package_version("nvidia-cutlass-dsl"),
        "tvm_ffi_version": _package_version("apache-tvm-ffi"),
        "measurement_notes": {
            "scope": (
                "These are isolated PrimTS API measurements, not end-to-end "
                "TensorRT-LLM attention or model speedups."
            ),
            "cuda_event_intervals": (
                "Events bracket each complete Python call, so their intervals can include GPU idle time "
                "caused by host enqueue gaps. They are not isolated kernel times."
            ),
            "kernel_time": "Use an Nsight Systems trace and the emitted NVTX ranges for kernel time.",
        },
        "settings": {
            "warmup": args.warmup,
            "iters": args.iters,
            "plan_iters": args.plan_iters,
            "nvtx_iters": args.nvtx_iters,
            "seed": args.seed,
        },
    }
    case_functions: tuple[tuple[str, Callable[[argparse.Namespace], dict[str, object]]], ...] = (
        ("context", _context_case),
        ("qwen", _qwen_case),
        ("mla", _mla_case),
    )
    with torch.inference_mode():
        for case_name, case_fn in case_functions:
            if args.case not in (case_name, "all"):
                continue
            results[case_name] = case_fn(args)
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
