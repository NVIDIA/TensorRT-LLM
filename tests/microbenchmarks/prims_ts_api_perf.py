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

"""Compare PrimTS raw, legacy-wrapper, and live-wrapper attention APIs.

Run this benchmark on a Blackwell GPU from a TensorRT-LLM source checkout whose
Python environment contains PyTorch, CuTe DSL 4.7, and TVM FFI. The default
shape is deliberately small enough for a quick development run; increase the
iteration counts for stable measurements.

This measures the PrimTS Python APIs in isolation, not end-to-end TensorRT-LLM
attention latency. Every case includes retained- and live-metadata wrappers;
the decode cases also include their raw caller-workspace APIs. Correctness
checks mutate the live metadata without replanning before performance
measurement begins.
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


def _contiguous_page_rows(lengths_host: list[int], page_size: int) -> list[list[int]]:
    page_rows: list[list[int]] = []
    next_page = 0
    for length in lengths_host:
        num_pages = (length + page_size - 1) // page_size
        page_rows.append(list(range(next_page, next_page + num_pages)))
        next_page += num_pages
    return page_rows


def _mutated_rows(page_rows: list[list[int]]) -> tuple[list[list[int]], str]:
    """Return valid metadata with different page values and unchanged storage size."""
    if len(page_rows) > 1:
        mutated = [list(row) for row in reversed(page_rows)]
        strategy = "reverse_request_page_rows"
    elif len(page_rows[0]) > 1:
        mutated = [list(reversed(page_rows[0]))]
        strategy = "reverse_single_request_page_ids"
    else:
        mutated = [[page_rows[0][0] + 1]]
        strategy = "replace_single_page_with_extra_physical_page"
    if mutated == page_rows:
        raise RuntimeError("metadata mutation must change at least one page ID position")
    return mutated, strategy


def _page_mutation_report(
    page_rows: list[list[int]],
    mutated_page_rows: list[list[int]],
    strategy: str,
) -> dict[str, object]:
    original_page_ids = [page_id for row in page_rows for page_id in row]
    mutated_page_ids = [page_id for row in mutated_page_rows for page_id in row]
    changed_positions = sum(
        original != mutated
        for original, mutated in zip(original_page_ids, mutated_page_ids, strict=True)
    )
    if changed_positions == 0:
        raise RuntimeError("metadata mutation must change at least one flattened page ID")
    prefix_size = min(16, len(original_page_ids))
    return {
        "strategy": strategy,
        "changed_page_id_positions": changed_positions,
        "original_first_page_id_per_row": [row[0] for row in page_rows],
        "mutated_first_page_id_per_row": [row[0] for row in mutated_page_rows],
        "original_page_ids_prefix": original_page_ids[:prefix_size],
        "mutated_page_ids_prefix": mutated_page_ids[:prefix_size],
        "page_id_mutation_is_non_vacuous": changed_positions > 0,
        "single_request_page_id_mutation": {
            "applies_to_this_run": len(page_rows) == 1,
            "page_id_change_guaranteed": True,
            "mechanism": (
                strategy
                if len(page_rows) == 1
                else "B1 reverses its page IDs or selects an extra physical page"
            ),
        },
    }


def _csr_metadata(
    lengths_host: list[int],
    page_rows: list[list[int]],
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indptr_host = [0]
    indices_host: list[int] = []
    for length, row in zip(lengths_host, page_rows, strict=True):
        required_pages = (length + page_size - 1) // page_size
        if len(row) < required_pages:
            raise ValueError(
                f"page row has {len(row)} entries but length {length} requires {required_pages}"
            )
        indices_host.extend(row[:required_pages])
        indptr_host.append(len(indices_host))
    return (
        torch.tensor(indptr_host, device=device, dtype=torch.int32),
        torch.tensor(indices_host, device=device, dtype=torch.int32),
        torch.tensor(
            [((length - 1) % page_size) + 1 for length in lengths_host],
            device=device,
            dtype=torch.int32,
        ),
    )


def _dense_context_metadata(
    lengths_host: list[int],
    page_rows: list[list[int]],
    max_pages_per_row: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense_rows = []
    for row in page_rows:
        if len(row) > max_pages_per_row:
            raise ValueError(f"page row has {len(row)} entries but capacity is {max_pages_per_row}")
        dense_rows.append(row + [row[-1]] * (max_pages_per_row - len(row)))
    dense_page_table = torch.tensor(dense_rows, device=device, dtype=torch.int32)
    dense_page_table = torch.stack((dense_page_table, dense_page_table), dim=1)
    return (
        _packed_indptr(lengths_host, device),
        torch.tensor(lengths_host, device=device, dtype=torch.int32),
        dense_page_table,
    )


def _packed_indptr(lengths_host: list[int], device: torch.device) -> torch.Tensor:
    indptr_host = [0]
    for length in lengths_host:
        indptr_host.append(indptr_host[-1] + length)
    return torch.tensor(indptr_host, device=device, dtype=torch.int32)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _policy_dict(policy: tuple[tuple[str, object], ...]) -> dict[str, object]:
    return dict(policy)


def _correctness(
    reference: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    torch.cuda.synchronize()
    differences: dict[str, float] = {}
    for label, candidate in candidates.items():
        differences[label] = _max_abs_diff(reference, candidate)
        torch.testing.assert_close(candidate, reference, rtol=1e-2, atol=5e-2)
    return differences


def _max_abs_diff(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference.float() - candidate.float()).abs().max().item())


def _require_observable_reference_difference(
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    case_name: str,
) -> float:
    torch.cuda.synchronize()
    difference = _max_abs_diff(reference_a, reference_b)
    if torch.allclose(reference_a, reference_b, rtol=1e-2, atol=5e-2):
        raise RuntimeError(
            f"{case_name} metadata mutation did not observably change the reference output"
        )
    return difference


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
    mutated_kv_lengths_host = (
        list(reversed(kv_lengths_host)) if batch_size > 1 else kv_lengths_host.copy()
    )
    mutated_q_lengths_host = (
        list(reversed(q_lengths_host)) if batch_size > 1 else q_lengths_host.copy()
    )
    page_rows = _contiguous_page_rows(kv_lengths_host, page_size)
    mutated_page_rows, page_mutation_strategy = _mutated_rows(page_rows)
    page_mutation = _page_mutation_report(
        page_rows,
        mutated_page_rows,
        page_mutation_strategy,
    )
    qo_indptr = _packed_indptr(q_lengths_host, device)
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _csr_metadata(
        kv_lengths_host,
        page_rows,
        page_size,
        device,
    )
    mutated_qo_indptr = _packed_indptr(mutated_q_lengths_host, device)
    (
        mutated_paged_kv_indptr,
        mutated_paged_kv_indices,
        mutated_paged_kv_last_page_len,
    ) = _csr_metadata(
        mutated_kv_lengths_host,
        mutated_page_rows,
        page_size,
        device,
    )
    pages_per_kv_tile = 128 // page_size
    required_pages_per_row = (max_seq_len + page_size - 1) // page_size
    max_pages_per_row = (
        (required_pages_per_row + pages_per_kv_tile - 1) // pages_per_kv_tile
    ) * pages_per_kv_tile
    logical_kv_indptr, seq_lens_kv, dense_page_idx_kv = _dense_context_metadata(
        kv_lengths_host,
        page_rows,
        max_pages_per_row,
        device,
    )
    (
        mutated_logical_kv_indptr,
        mutated_seq_lens_kv,
        mutated_dense_page_idx_kv,
    ) = _dense_context_metadata(
        mutated_kv_lengths_host,
        mutated_page_rows,
        max_pages_per_row,
        device,
    )
    live_qo_indptr = qo_indptr.clone()
    live_logical_kv_indptr = logical_kv_indptr.clone()
    live_seq_lens_kv = seq_lens_kv.clone()
    live_dense_page_idx_kv = dense_page_idx_kv.clone()
    total_q = sum(q_lengths_host)
    num_referenced_pages = sum(len(row) for row in page_rows)
    max_page_id = max(
        page_id for rows in (page_rows, mutated_page_rows) for row in rows for page_id in row
    )
    num_physical_pages = max_page_id + 1
    query = torch.randn(total_q, num_qo_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        num_physical_pages,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    legacy_one_shot_out = torch.empty_like(query)
    legacy_cached_out = torch.empty_like(query)
    live_cached_out = torch.empty_like(query)
    mutation_reference_out = torch.empty_like(query)
    legacy_workspace = torch.empty(0, device=device, dtype=torch.uint8)
    live_workspace = torch.empty(0, device=device, dtype=torch.uint8)
    max_seq_len_q = max(q_lengths_host)
    mask_type = "causal"
    kv_layout = "HND"
    window_left = -1
    sm_scale = 1.0 / math.sqrt(head_dim)
    output_scale = 1.0

    def legacy_one_shot_construct_plan_run() -> torch.Tensor:
        return batch_prefill_with_paged_kv_cache(
            query,
            k_cache,
            v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            page_size=page_size,
            kv_layout=kv_layout,
            mask_type=mask_type,
            window_left=window_left,
            sm_scale=sm_scale,
            output_scale=output_scale,
            out_dtype=dtype,
            out=legacy_one_shot_out,
        )

    _, first_legacy_one_shot = _measure_once(
        "cold/context/legacy_one_shot_construct_plan_run",
        legacy_one_shot_construct_plan_run,
    )

    def make_legacy_wrapper() -> BatchPrefillPagedTSWrapper:
        wrapper = BatchPrefillPagedTSWrapper(
            kv_layout=kv_layout,
            workspace_buffer=legacy_workspace,
        )
        wrapper.plan(
            query,
            k_cache,
            v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            sm_scale=sm_scale,
            output_scale=output_scale,
            out_dtype=dtype,
        )
        return wrapper

    legacy_wrapper, legacy_plan = _measure_once(
        "plan/context/legacy_retained_metadata_preallocated_zero_workspace",
        make_legacy_wrapper,
    )

    def make_live_wrapper() -> BatchPrefillPagedTSWrapper:
        wrapper = BatchPrefillPagedTSWrapper(
            kv_layout=kv_layout,
            workspace_buffer=live_workspace,
        )
        wrapper.plan_live(
            query,
            k_cache,
            v_cache,
            batch_size=batch_size,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len,
            max_num_pages_per_seq_kv=max_pages_per_row,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            sm_scale=sm_scale,
            output_scale=output_scale,
            out_dtype=dtype,
        )
        return wrapper

    live_wrapper, live_plan = _measure_once(
        "plan/context/live_metadata_preallocated_zero_workspace",
        make_live_wrapper,
    )

    legacy_hot_plan = _event_bench(
        "context/legacy_retained_metadata_preallocated_workspace_plan_only",
        make_legacy_wrapper,
        warmup=1,
        iters=args.plan_iters,
    )
    live_hot_plan = _event_bench(
        "context/live_metadata_preallocated_workspace_plan_only",
        make_live_wrapper,
        warmup=1,
        iters=args.plan_iters,
    )

    def legacy_cached_run() -> torch.Tensor:
        return legacy_wrapper.run(query, k_cache, v_cache, out=legacy_cached_out)

    def live_cached_run() -> torch.Tensor:
        return live_wrapper.run(
            query,
            k_cache,
            v_cache,
            out=live_cached_out,
            qo_indptr=live_qo_indptr,
            logical_kv_indptr=live_logical_kv_indptr,
            dense_page_idx_kv=live_dense_page_idx_kv,
            seq_lens_kv=live_seq_lens_kv,
        )

    reference_a = legacy_one_shot_construct_plan_run().clone()
    correctness = _correctness(
        reference_a,
        {
            "legacy_wrapper_retained_metadata": legacy_cached_run(),
            "live_wrapper_external_workspace": live_cached_run(),
        },
    )
    live_qo_indptr.copy_(mutated_qo_indptr)
    live_logical_kv_indptr.copy_(mutated_logical_kv_indptr)
    live_seq_lens_kv.copy_(mutated_seq_lens_kv)
    live_dense_page_idx_kv.copy_(mutated_dense_page_idx_kv)
    reference_b = batch_prefill_with_paged_kv_cache(
        query,
        k_cache,
        v_cache,
        mutated_qo_indptr,
        mutated_paged_kv_indptr,
        mutated_paged_kv_indices,
        mutated_paged_kv_last_page_len,
        page_size=page_size,
        kv_layout=kv_layout,
        mask_type=mask_type,
        window_left=window_left,
        sm_scale=sm_scale,
        output_scale=output_scale,
        out_dtype=dtype,
        out=mutation_reference_out,
    ).clone()
    reference_a_vs_reference_b = _require_observable_reference_difference(
        reference_a,
        reference_b,
        "context",
    )
    live_metadata_correctness = _correctness(
        reference_b,
        {"live_wrapper_after_in_place_metadata_mutation": live_cached_run()},
    )
    live_qo_indptr.copy_(qo_indptr)
    live_logical_kv_indptr.copy_(logical_kv_indptr)
    live_seq_lens_kv.copy_(seq_lens_kv)
    live_dense_page_idx_kv.copy_(dense_page_idx_kv)
    restored_metadata_correctness = _correctness(
        reference_a,
        {"live_wrapper_after_restoring_original_metadata": live_cached_run()},
    )
    variants = {
        "legacy_one_shot_construct_plan_run": legacy_one_shot_construct_plan_run,
        "legacy_wrapper_retained_metadata_run_only": legacy_cached_run,
        "live_wrapper_external_workspace_run_only": live_cached_run,
    }
    planned_metadata_bytes = sum(
        _tensor_bytes(tensor)
        for tensor in (
            legacy_wrapper._logical_kv_indptr,
            legacy_wrapper._seq_lens_kv,
            legacy_wrapper._dense_page_idx_kv,
            legacy_wrapper._scale_softmax_log2,
            legacy_wrapper._output_scale,
        )
    )
    assert legacy_wrapper._workspace_buffer is legacy_workspace
    assert live_wrapper._workspace_buffer is live_workspace
    assert _tensor_bytes(legacy_workspace) == _tensor_bytes(live_workspace) == 0
    return {
        "geometry": {
            "batch_size": batch_size,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "max_seq_len": max_seq_len,
            "max_seq_len_q": max_seq_len_q,
            "max_pages_per_row": max_pages_per_row,
            "num_referenced_pages": num_referenced_pages,
            "num_physical_pages": num_physical_pages,
            "total_q_tokens": total_q,
            "q_lengths": q_lengths_host,
            "kv_lengths": kv_lengths_host,
            "query_shape": list(query.shape),
            "k_cache_shape": list(k_cache.shape),
            "v_cache_shape": list(v_cache.shape),
            "output_shape": list(live_cached_out.shape),
        },
        "config": {
            "dtype": _dtype_name(dtype),
            "out_dtype": _dtype_name(dtype),
            "kv_layout": kv_layout,
            "mask_type": mask_type,
            "window_left": window_left,
            "sm_scale": sm_scale,
            "output_scale": output_scale,
            "live_metadata_fields": [
                "qo_indptr",
                "logical_kv_indptr",
                "dense_page_idx_kv",
                "seq_lens_kv",
            ],
        },
        "policy": {
            "legacy_retained_metadata": _policy_dict(legacy_wrapper._policy),
            "live_metadata": _policy_dict(live_wrapper._policy),
        },
        "measurement_scope": (
            "Live-wrapper timing starts after TRT/PrimTS-native logical offsets, "
            "sequence lengths, and dense page-table metadata have been staged. It "
            "measures wrapper/API cost, not full backend-equivalent context cost."
        ),
        "reuse_contract": {
            "raw": (
                "No public raw paged-context launch is exposed; the one-shot API "
                "constructs, plans, runs, and discards a legacy wrapper on every call."
            ),
            "legacy_wrapper": (
                "The plan snapshots K/V CSR values into a dense page table and derived "
                "lengths. It retains qo_indptr storage as a constrained live input; "
                "replan after changing snapshotted K/V metadata or static geometry."
            ),
            "live_wrapper": (
                "Every run supplies Q offsets, logical K/V offsets, K/V lengths, and "
                "the padded dense page table. Values may change within planned bounds "
                "after the prior launch completes; graph capture also requires stable "
                "tensor addresses. Context uses no scratch workspace."
            ),
        },
        "workspace_bytes": {
            "legacy_external_required": 0,
            "legacy_external_allocated": _tensor_bytes(legacy_workspace),
            "live_external_required": 0,
            "live_external_allocated": _tensor_bytes(live_workspace),
            "legacy_wrapper_planned_metadata": planned_metadata_bytes,
        },
        "plan_timing_allocation_semantics": {
            "legacy_wrapper": (
                "The zero-byte caller workspace and outputs are preallocated. Plan "
                "still allocates scale tensors plus snapshotted logical offsets, "
                "sequence lengths, and the dense page table."
            ),
            "live_wrapper": (
                "The zero-byte caller workspace and outputs are preallocated. Plan "
                "allocates only its two one-element scale tensors."
            ),
            "hot_plan_measurement": (
                "Each iteration constructs a new wrapper and calls plan after the "
                "semantic JIT cache has been warmed."
            ),
        },
        "first_legacy_one_shot": first_legacy_one_shot,
        "legacy_wrapper_plan_preallocated_workspace_after_one_shot_compile": legacy_plan,
        "live_wrapper_plan_preallocated_workspace": live_plan,
        "legacy_hot_plan_only_preallocated_workspace": legacy_hot_plan,
        "live_hot_plan_only_preallocated_workspace": live_hot_plan,
        "max_abs_diff": correctness,
        "live_metadata_mutation": {
            "mutated_q_lengths": mutated_q_lengths_host,
            "mutated_kv_lengths": mutated_kv_lengths_host,
            "page_table": {
                **page_mutation,
                "dense_page_table_changed_entries": int(
                    (dense_page_idx_kv != mutated_dense_page_idx_kv).sum().item()
                ),
            },
            "max_abs_diff": live_metadata_correctness,
            "a_b_a_max_abs_diff": {
                "reference_a_vs_initial_live_a": correctness["live_wrapper_external_workspace"],
                "reference_a_vs_reference_b": reference_a_vs_reference_b,
                "reference_b_vs_live_b": live_metadata_correctness[
                    "live_wrapper_after_in_place_metadata_mutation"
                ],
                "reference_a_vs_restored_live_a": restored_metadata_correctness[
                    "live_wrapper_after_restoring_original_metadata"
                ],
            },
        },
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
    mutated_lengths_host = list(reversed(lengths_host)) if batch_size > 1 else lengths_host.copy()
    page_rows = _contiguous_page_rows(lengths_host, page_size)
    mutated_page_rows, page_mutation_strategy = _mutated_rows(page_rows)
    page_mutation = _page_mutation_report(
        page_rows,
        mutated_page_rows,
        page_mutation_strategy,
    )
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = _csr_metadata(
        lengths_host,
        page_rows,
        page_size,
        device,
    )
    mutated_paged_kv_indptr, mutated_paged_kv_indices, _ = _csr_metadata(
        mutated_lengths_host,
        mutated_page_rows,
        page_size,
        device,
    )
    seq_lens = torch.tensor(lengths_host, device=device, dtype=torch.int32)
    mutated_seq_lens = torch.tensor(
        mutated_lengths_host,
        device=device,
        dtype=torch.int32,
    )
    live_paged_kv_indptr = paged_kv_indptr.clone()
    live_paged_kv_indices = paged_kv_indices.clone()
    live_seq_lens = seq_lens.clone()
    num_referenced_pages = sum(len(row) for row in page_rows)
    max_page_id = max(
        page_id for rows in (page_rows, mutated_page_rows) for row in rows for page_id in row
    )
    num_physical_pages = max_page_id + 1
    query = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    kv_cache = torch.randn(
        num_physical_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    raw_out = torch.empty_like(query)
    zeroed_raw_out = torch.empty_like(query)
    legacy_cached_out = torch.empty_like(query)
    live_cached_out = torch.empty_like(query)
    live_adapter_out = torch.empty_like(query)
    legacy_one_shot_out = torch.empty_like(query)
    mutation_reference_out = torch.empty_like(query)
    mask_type = "causal"
    kv_layout = "HND"
    window_left = -1
    seq_len_q = 1
    bmm1_scale = 1.0 / math.sqrt(head_dim)
    bmm2_scale = 1.0
    raw_workspace_bytes = get_prims_ts_batch_decode_workspace_size(
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        seq_len_q=seq_len_q,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type=mask_type,
        window_left=window_left,
        kv_layout=kv_layout,
        device=device,
    )
    raw_workspace = torch.zeros(raw_workspace_bytes, device=device, dtype=torch.uint8)
    legacy_workspace = torch.zeros(raw_workspace_bytes, device=device, dtype=torch.uint8)
    live_workspace = torch.zeros(raw_workspace_bytes, device=device, dtype=torch.uint8)
    mutation_reference_workspace = torch.zeros_like(raw_workspace)

    def raw_call(
        out: torch.Tensor,
        workspace: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        return prims_ts_batch_decode_with_kv_cache(
            query,
            kv_cache,
            workspace,
            indptr,
            indices,
            lengths,
            max_seq_len,
            seq_len_q=seq_len_q,
            out=out,
            out_dtype=dtype,
            mask_type=mask_type,
            window_left=window_left,
            kv_layout=kv_layout,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )

    def raw_launch_only() -> torch.Tensor:
        return raw_call(
            raw_out,
            raw_workspace,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens,
        )

    def workspace_zero_plus_raw() -> torch.Tensor:
        raw_workspace.zero_()
        return raw_call(
            zeroed_raw_out,
            raw_workspace,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens,
        )

    _, first_raw = _measure_once("cold/qwen/raw_api_call", raw_launch_only)

    def make_legacy_wrapper() -> BatchDecodePagedTSWrapper:
        wrapper = BatchDecodePagedTSWrapper(
            kv_layout=kv_layout,
            workspace_buffer=legacy_workspace,
        )
        wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            seq_len_q=seq_len_q,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type=mask_type,
            window_left=window_left,
            max_kv_len=max_seq_len,
        )
        return wrapper

    legacy_wrapper, legacy_plan = _measure_once(
        "plan/qwen/legacy_retained_metadata_preallocated_workspace",
        make_legacy_wrapper,
    )

    def make_live_wrapper() -> BatchDecodePagedTSWrapper:
        wrapper = BatchDecodePagedTSWrapper(
            kv_layout=kv_layout,
            workspace_buffer=live_workspace,
        )
        wrapper.plan(
            live_paged_kv_indptr,
            live_paged_kv_indices,
            None,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            seq_len_q=seq_len_q,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type=mask_type,
            window_left=window_left,
            max_kv_len=max_seq_len,
            live_metadata=True,
        )
        return wrapper

    live_wrapper, live_plan = _measure_once(
        "plan/qwen/live_metadata_preallocated_workspace",
        make_live_wrapper,
    )

    legacy_hot_plan = _event_bench(
        "qwen/legacy_retained_metadata_preallocated_workspace_plan_only",
        make_legacy_wrapper,
        warmup=1,
        iters=args.plan_iters,
    )
    live_hot_plan = _event_bench(
        "qwen/live_metadata_preallocated_workspace_plan_only",
        make_live_wrapper,
        warmup=1,
        iters=args.plan_iters,
    )

    def legacy_cached_run() -> torch.Tensor:
        return legacy_wrapper.run(
            query,
            kv_cache,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=legacy_cached_out,
        )

    def live_run(out: torch.Tensor) -> torch.Tensor:
        return live_wrapper.run(
            query,
            kv_cache,
            live_seq_lens,
            paged_kv_indptr=live_paged_kv_indptr,
            paged_kv_indices=live_paged_kv_indices,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=out,
        )

    def live_cached_run() -> torch.Tensor:
        return live_run(live_cached_out)

    control_span_offset = live_wrapper._workspace_layout.split_kv_counter.byte_offset
    control_span_end = live_wrapper._workspace_layout.total_bytes
    live_control_span = live_workspace[control_span_offset:control_span_end]

    def live_adapter_control_zero_plus_run() -> torch.Tensor:
        # Match PrimsTSFmha.run_generation when the workspace slab may have
        # been used by another layout or captured graph.
        live_control_span.zero_()
        return live_run(live_adapter_out)

    def legacy_one_shot_construct_plan_run() -> torch.Tensor:
        # This convenience API creates, plans, and discards a wrapper per call.
        return batch_decode_with_paged_kv_cache(
            query,
            kv_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            seq_len_q=seq_len_q,
            mask_type=mask_type,
            window_left=window_left,
            kv_layout=kv_layout,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=legacy_one_shot_out,
            out_dtype=dtype,
        )

    reference_a = raw_launch_only().clone()
    correctness = _correctness(
        reference_a,
        {
            "workspace_zero_plus_raw": workspace_zero_plus_raw(),
            "legacy_wrapper_retained_metadata": legacy_cached_run(),
            "live_wrapper_external_workspace": live_cached_run(),
            "live_wrapper_adapter_control_zero_plus_run": (live_adapter_control_zero_plus_run()),
            "legacy_one_shot_construct_plan_run": legacy_one_shot_construct_plan_run(),
        },
    )
    live_paged_kv_indptr.copy_(mutated_paged_kv_indptr)
    live_paged_kv_indices.copy_(mutated_paged_kv_indices)
    live_seq_lens.copy_(mutated_seq_lens)
    mutation_reference_workspace.zero_()
    reference_b = raw_call(
        mutation_reference_out,
        mutation_reference_workspace,
        live_paged_kv_indptr,
        live_paged_kv_indices,
        live_seq_lens,
    ).clone()
    reference_a_vs_reference_b = _require_observable_reference_difference(
        reference_a,
        reference_b,
        "qwen decode",
    )
    live_metadata_correctness = _correctness(
        reference_b,
        {
            "live_wrapper_adapter_path_after_in_place_metadata_mutation": (
                live_adapter_control_zero_plus_run()
            )
        },
    )
    live_paged_kv_indptr.copy_(paged_kv_indptr)
    live_paged_kv_indices.copy_(paged_kv_indices)
    live_seq_lens.copy_(seq_lens)
    restored_metadata_correctness = _correctness(
        reference_a,
        {
            "live_wrapper_adapter_path_after_restoring_original_metadata": (
                live_adapter_control_zero_plus_run()
            )
        },
    )
    variants = {
        "raw_api_call": raw_launch_only,
        "raw_workspace_zero_plus_api_call": workspace_zero_plus_raw,
        "legacy_wrapper_retained_metadata_run_only": legacy_cached_run,
        "live_wrapper_external_workspace_run_only_dedicated_layout": live_cached_run,
        "live_wrapper_adapter_control_span_zero_plus_run_shared_layout": (
            live_adapter_control_zero_plus_run
        ),
        "legacy_one_shot_construct_plan_run": legacy_one_shot_construct_plan_run,
    }
    assert legacy_wrapper._workspace_buffer is legacy_workspace
    assert live_wrapper._workspace_buffer is live_workspace
    assert _tensor_bytes(legacy_workspace) == legacy_wrapper._workspace_layout.total_bytes
    assert _tensor_bytes(live_workspace) == live_wrapper._workspace_layout.total_bytes
    return {
        "geometry": {
            "batch_size": batch_size,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "max_seq_len": max_seq_len,
            "seq_len_q": seq_len_q,
            "num_referenced_pages": num_referenced_pages,
            "num_physical_pages": num_physical_pages,
            "seq_lens": lengths_host,
            "query_shape": list(query.shape),
            "kv_cache_shape": list(kv_cache.shape),
            "output_shape": list(live_cached_out.shape),
        },
        "config": {
            "q_dtype": _dtype_name(dtype),
            "kv_dtype": _dtype_name(dtype),
            "out_dtype": _dtype_name(dtype),
            "kv_layout": kv_layout,
            "mask_type": mask_type,
            "window_left": window_left,
            "bmm1_scale": bmm1_scale,
            "bmm2_scale": bmm2_scale,
            "live_metadata_fields": [
                "paged_kv_indptr",
                "paged_kv_indices",
                "seq_lens",
            ],
        },
        "policy": {
            "legacy_retained_metadata": _policy_dict(legacy_wrapper._policy),
            "live_metadata": _policy_dict(live_wrapper._policy),
        },
        "modes": {
            "raw_kv_prefix_mode": "dynamic",
            "raw_kv_lengths_mode": "dynamic",
            "legacy_wrapper_kv_prefix_mode": legacy_wrapper._kv_prefix_mode,
            "legacy_wrapper_kv_lengths_mode": legacy_wrapper._kv_lengths_mode,
            "live_wrapper_kv_prefix_mode": live_wrapper._kv_prefix_mode,
            "live_wrapper_kv_lengths_mode": live_wrapper._kv_lengths_mode,
            "raw_and_live_use_same_modes": (
                live_wrapper._kv_prefix_mode == "dynamic"
                and live_wrapper._kv_lengths_mode == "dynamic"
            ),
        },
        "reuse_contract": {
            "raw": (
                "CSR and sequence lengths are live inputs. The caller workspace "
                "must be zero-initialized before first use, remain exclusive to one "
                "in-flight launch, and be reused only after the prior launch completes."
            ),
            "legacy_wrapper": (
                "Plan derives and retains sequence lengths and CSR bindings. Replan "
                "after changing CSR row boundaries, last-page lengths, static bounds, "
                "or any value used to select a planned specialization."
            ),
            "live_wrapper": (
                "Every run reads CSR and sequence lengths within the planned static "
                "bounds. Run-only timing assumes a dedicated workspace with an "
                "identical layout; the adapter-matched timing first zeros the control "
                "span so a serialized shared slab cannot carry stale control values."
            ),
        },
        "workspace_bytes": {
            "raw": raw_workspace_bytes,
            "legacy_external_required": raw_workspace_bytes,
            "legacy_external_allocated": _tensor_bytes(legacy_workspace),
            "live_external_required": raw_workspace_bytes,
            "live_external_allocated": _tensor_bytes(live_workspace),
            "live_adapter_control_span_offset": control_span_offset,
            "live_adapter_control_span_bytes": control_span_end - control_span_offset,
        },
        "plan_timing_allocation_semantics": {
            "legacy_wrapper": (
                "Scratch and output are preallocated. Plan still synchronizes to read "
                "legacy metadata and allocates its derived device sequence-length tensor."
            ),
            "live_wrapper": (
                "Scratch, metadata, and output are preallocated. Plan binds workspace "
                "views and initializes the control tensors without allocating scratch."
            ),
            "hot_plan_measurement": (
                "Each iteration constructs a new wrapper and calls plan after the "
                "semantic JIT cache has been warmed."
            ),
        },
        "first_raw": first_raw,
        "legacy_wrapper_plan_preallocated_workspace_after_raw_compile": legacy_plan,
        "live_wrapper_plan_preallocated_workspace_after_raw_compile": live_plan,
        "legacy_hot_plan_only_preallocated_workspace": legacy_hot_plan,
        "live_hot_plan_only_preallocated_workspace": live_hot_plan,
        "max_abs_diff": correctness,
        "live_metadata_mutation": {
            "mutated_seq_lens": mutated_lengths_host,
            "page_ids": {
                **page_mutation,
                "csr_indptr_changed_entries": int(
                    (paged_kv_indptr != mutated_paged_kv_indptr).sum().item()
                ),
                "csr_page_id_changed_entries": int(
                    (paged_kv_indices != mutated_paged_kv_indices).sum().item()
                ),
            },
            "max_abs_diff": live_metadata_correctness,
            "a_b_a_max_abs_diff": {
                "reference_a_vs_initial_live_a": correctness[
                    "live_wrapper_adapter_control_zero_plus_run"
                ],
                "reference_a_vs_reference_b": reference_a_vs_reference_b,
                "reference_b_vs_live_b": live_metadata_correctness[
                    "live_wrapper_adapter_path_after_in_place_metadata_mutation"
                ],
                "reference_a_vs_restored_live_a": restored_metadata_correctness[
                    "live_wrapper_adapter_path_after_restoring_original_metadata"
                ],
            },
        },
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
    mutated_lengths_host = list(reversed(lengths_host)) if batch_size > 1 else lengths_host.copy()
    seq_lens = torch.tensor(lengths_host, device=device, dtype=torch.int32)
    page_rows = [
        list(range(row * max_pages_per_row, (row + 1) * max_pages_per_row))
        for row in range(batch_size)
    ]
    mutated_page_rows, page_mutation_strategy = _mutated_rows(page_rows)
    page_mutation = _page_mutation_report(
        page_rows,
        mutated_page_rows,
        page_mutation_strategy,
    )
    max_page_id = max(
        page_id for rows in (page_rows, mutated_page_rows) for row in rows for page_id in row
    )
    num_physical_pages = max_page_id + 1
    block_tables = torch.tensor(page_rows, device=device, dtype=torch.int32)
    live_block_tables = block_tables.clone()
    live_seq_lens = seq_lens.clone()
    mutated_block_tables = torch.tensor(
        mutated_page_rows,
        device=device,
        dtype=torch.int32,
    )
    mutated_seq_lens = torch.tensor(
        mutated_lengths_host,
        device=device,
        dtype=torch.int32,
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
    legacy_cached_out = torch.empty_like(raw_out)
    live_cached_out = torch.empty_like(raw_out)
    legacy_one_shot_out = torch.empty_like(raw_out)
    mutation_reference_out = torch.empty_like(raw_out)
    max_seq_len_q = 1
    mask_type = "causal"
    bmm1_scale = 1.0 / math.sqrt(128 + rope_dim)
    bmm2_scale = 1.0
    raw_workspace_bytes = get_prims_ts_batch_decode_mla_workspace_size(
        batch_size,
        num_heads,
        kv_lora_rank,
        rope_dim,
        page_size,
        max_seq_len,
        max_seq_len_q=max_seq_len_q,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type=mask_type,
        device=device,
    )
    raw_workspace = torch.empty(raw_workspace_bytes, device=device, dtype=torch.uint8)
    legacy_workspace = torch.empty_like(raw_workspace)
    live_workspace = torch.empty_like(raw_workspace)
    mutation_reference_workspace = torch.empty_like(raw_workspace)

    def raw_call(
        out: torch.Tensor,
        workspace: torch.Tensor,
        runtime_block_tables: torch.Tensor,
        runtime_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        return prims_ts_batch_decode_with_kv_cache_mla(
            query,
            kv_cache,
            workspace,
            kv_lora_rank,
            rope_dim,
            runtime_block_tables,
            runtime_seq_lens,
            max_seq_len,
            max_seq_len_q=max_seq_len_q,
            out=out,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out_dtype=dtype,
            mask_type=mask_type,
        )

    def raw_launch_only() -> torch.Tensor:
        return raw_call(raw_out, raw_workspace, block_tables, seq_lens)

    _, first_raw = _measure_once("cold/mla/raw_api_call", raw_launch_only)

    def make_legacy_wrapper() -> BatchMLADecodePagedTSWrapper:
        wrapper = BatchMLADecodePagedTSWrapper(workspace_buffer=legacy_workspace)
        wrapper.plan(
            block_tables,
            seq_lens,
            num_heads,
            kv_lora_rank,
            rope_dim,
            page_size,
            max_seq_len_q=max_seq_len_q,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type=mask_type,
            max_kv_len=max_seq_len,
        )
        return wrapper

    legacy_wrapper, legacy_plan = _measure_once(
        "plan/mla/legacy_retained_metadata_preallocated_workspace",
        make_legacy_wrapper,
    )

    def make_live_wrapper() -> BatchMLADecodePagedTSWrapper:
        wrapper = BatchMLADecodePagedTSWrapper(workspace_buffer=live_workspace)
        wrapper.plan(
            live_block_tables,
            live_seq_lens,
            num_heads,
            kv_lora_rank,
            rope_dim,
            page_size,
            max_seq_len_q=max_seq_len_q,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type=mask_type,
            max_kv_len=max_seq_len,
            live_metadata=True,
        )
        return wrapper

    live_wrapper, live_plan = _measure_once(
        "plan/mla/live_metadata_preallocated_workspace",
        make_live_wrapper,
    )

    legacy_hot_plan = _event_bench(
        "mla/legacy_retained_metadata_preallocated_workspace_plan_only",
        make_legacy_wrapper,
        warmup=1,
        iters=args.plan_iters,
    )
    live_hot_plan = _event_bench(
        "mla/live_metadata_preallocated_workspace_plan_only",
        make_live_wrapper,
        warmup=1,
        iters=args.plan_iters,
    )

    def legacy_cached_run() -> torch.Tensor:
        return legacy_wrapper.run(
            query,
            kv_cache,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=legacy_cached_out,
        )

    def live_cached_run() -> torch.Tensor:
        return live_wrapper.run(
            query,
            kv_cache,
            block_tables=live_block_tables,
            seq_lens=live_seq_lens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=live_cached_out,
        )

    def legacy_one_shot_construct_plan_run() -> torch.Tensor:
        return batch_decode_mla_with_paged_kv_cache(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            max_seq_len_q=max_seq_len_q,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=rope_dim,
            mask_type=mask_type,
            max_kv_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=legacy_one_shot_out,
            out_dtype=dtype,
        )

    reference_a = raw_launch_only().clone()
    correctness = _correctness(
        reference_a,
        {
            "legacy_wrapper_retained_metadata": legacy_cached_run(),
            "live_wrapper_external_workspace": live_cached_run(),
            "legacy_one_shot_construct_plan_run": legacy_one_shot_construct_plan_run(),
        },
    )
    live_block_tables.copy_(mutated_block_tables)
    live_seq_lens.copy_(mutated_seq_lens)
    reference_b = raw_call(
        mutation_reference_out,
        mutation_reference_workspace,
        live_block_tables,
        live_seq_lens,
    ).clone()
    reference_a_vs_reference_b = _require_observable_reference_difference(
        reference_a,
        reference_b,
        "MLA decode",
    )
    live_metadata_correctness = _correctness(
        reference_b,
        {"live_wrapper_after_in_place_metadata_mutation": live_cached_run()},
    )
    live_block_tables.copy_(block_tables)
    live_seq_lens.copy_(seq_lens)
    restored_metadata_correctness = _correctness(
        reference_a,
        {"live_wrapper_after_restoring_original_metadata": live_cached_run()},
    )
    variants = {
        "raw_api_call": raw_launch_only,
        "legacy_wrapper_retained_metadata_run_only": legacy_cached_run,
        "live_wrapper_external_workspace_run_only": live_cached_run,
        "legacy_one_shot_construct_plan_run": legacy_one_shot_construct_plan_run,
    }
    assert legacy_wrapper._workspace_buffer is legacy_workspace
    assert live_wrapper._workspace_buffer is live_workspace
    assert _tensor_bytes(legacy_workspace) == legacy_wrapper._workspace_layout.total_bytes
    assert _tensor_bytes(live_workspace) == live_wrapper._workspace_layout.total_bytes
    return {
        "geometry": {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": rope_dim,
            "page_size": page_size,
            "max_seq_len": max_seq_len,
            "max_seq_len_q": max_seq_len_q,
            "max_pages_per_row": max_pages_per_row,
            "num_physical_pages": num_physical_pages,
            "seq_lens": lengths_host,
            "query_shape": list(query.shape),
            "kv_cache_shape": list(kv_cache.shape),
            "output_shape": list(live_cached_out.shape),
        },
        "config": {
            "q_dtype": _dtype_name(dtype),
            "kv_dtype": _dtype_name(dtype),
            "out_dtype": _dtype_name(dtype),
            "mask_type": mask_type,
            "bmm1_scale": bmm1_scale,
            "bmm2_scale": bmm2_scale,
            "live_metadata_fields": ["block_tables", "seq_lens"],
        },
        "policy": {
            "legacy_retained_metadata": _policy_dict(legacy_wrapper._policy),
            "live_metadata": _policy_dict(live_wrapper._policy),
        },
        "modes": {
            "raw": "live_seq_lens",
            "legacy_wrapper": "retained_metadata_storage",
            "live_wrapper": "run_time_metadata_bindings",
        },
        "reuse_contract": {
            "raw": (
                "Block tables and sequence lengths are live inputs. The caller "
                "workspace needs no initialization but is exclusive to one in-flight "
                "launch and may be reused only after that launch completes."
            ),
            "legacy_wrapper": (
                "Plan retains the metadata tensor bindings and derives policy and "
                "bounds from their values. Run does not accept replacement metadata; "
                "replan before changing values that violate the planned specialization."
            ),
            "live_wrapper": (
                "Run accepts block-table and sequence-length bindings whose shapes and "
                "values stay within planned static bounds. Workspace remains exclusive "
                "to one execution lane or graph replay at a time."
            ),
        },
        "workspace_bytes": {
            "raw": raw_workspace_bytes,
            "legacy_external_required": raw_workspace_bytes,
            "legacy_external_allocated": _tensor_bytes(legacy_workspace),
            "live_external_required": raw_workspace_bytes,
            "live_external_allocated": _tensor_bytes(live_workspace),
        },
        "plan_timing_allocation_semantics": {
            "legacy_wrapper": (
                "Scratch, metadata, and output are preallocated. Plan synchronizes to "
                "read metadata values but does not allocate its workspace."
            ),
            "live_wrapper": (
                "Scratch, metadata, and output are preallocated. Explicit bounds avoid "
                "metadata reads and plan only binds views into caller workspace."
            ),
            "hot_plan_measurement": (
                "Each iteration constructs a new wrapper and calls plan after the "
                "semantic JIT cache has been warmed."
            ),
        },
        "first_raw": first_raw,
        "legacy_wrapper_plan_preallocated_workspace_after_raw_compile": legacy_plan,
        "live_wrapper_plan_preallocated_workspace_after_raw_compile": live_plan,
        "legacy_hot_plan_only_preallocated_workspace": legacy_hot_plan,
        "live_hot_plan_only_preallocated_workspace": live_hot_plan,
        "max_abs_diff": correctness,
        "live_metadata_mutation": {
            "mutated_seq_lens": mutated_lengths_host,
            "block_table_page_ids": {
                **page_mutation,
                "block_table_changed_entries": int(
                    (block_tables != mutated_block_tables).sum().item()
                ),
            },
            "max_abs_diff": live_metadata_correctness,
            "a_b_a_max_abs_diff": {
                "reference_a_vs_initial_live_a": correctness["live_wrapper_external_workspace"],
                "reference_a_vs_reference_b": reference_a_vs_reference_b,
                "reference_b_vs_live_b": live_metadata_correctness[
                    "live_wrapper_after_in_place_metadata_mutation"
                ],
                "reference_a_vs_restored_live_a": restored_metadata_correctness[
                    "live_wrapper_after_restoring_original_metadata"
                ],
            },
        },
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
