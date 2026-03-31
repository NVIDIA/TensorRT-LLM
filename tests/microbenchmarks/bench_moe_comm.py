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

# ruff: noqa: E501

"""Unified MoE communication microbenchmark (MPI).

This benchmarks ONLY the communication kernels, specifically:
- Communication.dispatch()
- Communication.combine()

Launch (examples):

```bash
# Minimal: sweep batch sizes 1..1024 (powers of 2) on ep=8.
# --profile means a pre-defined combo of hidden_size, top_k, and num_experts
python tests/microbenchmarks/bench_moe_comm.py \
    --ep_size 8 --backend NVLINK_ONE_SIDED --profile deepseek_v3 -b 1 -e 1024 -f 2

# Add kernel breakdown and per-iteration stats; save results to JSON.
# --perfect_router removes routing variance so timings are stable across iterations.
python tests/microbenchmarks/bench_moe_comm.py \
    --ep_size 8 --backend NVLINK_ONE_SIDED --profile deepseek_v3 --perfect_router \
    --kernel_breakdown --iter_stats -b 1 -e 1024 -f 2 --output_file out.json

# mpirun mode: workers are pre-launched by MPI instead of spawned internally.
# Useful for multi-node, or when MPIPoolExecutor spawn causes issues.
mpirun -n 8 python tests/microbenchmarks/bench_moe_comm.py \
    --backend NVLINK_ONE_SIDED --profile deepseek_v3 --perfect_router \
    --kernel_breakdown --iter_stats -b 1 -e 1024 -f 2 --output_file out.json

```
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pickle
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from torch.autograd import DeviceType

import tensorrt_llm as tllm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import CutlassFusedMoE, MoE
from tensorrt_llm._torch.modules.fused_moe.communication import Communication, CommunicationFactory
from tensorrt_llm._torch.modules.fused_moe.routing import DefaultMoeRoutingMethod
from tensorrt_llm._utils import local_mpi_rank, mpi_allgather, mpi_barrier, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


@dataclass(frozen=True)
class _DummyPretrainedConfig:
    # Only the two attributes used by CommunicationFactory / ModelConfig.torch_dtype.
    hidden_size: int
    torch_dtype: torch.dtype


@dataclass(frozen=True)
class Profile:
    name: str
    hidden_size: int
    top_k: int
    num_experts: int
    quant_algo: QuantAlgo


PROFILES: Dict[str, Profile] = {
    "deepseek_v3": Profile(
        name="deepseek_v3",
        hidden_size=7168,
        top_k=8,
        num_experts=256,
        quant_algo=QuantAlgo.FP8_BLOCK_SCALES,
    ),
    "gpt_oss": Profile(
        name="gpt_oss",
        hidden_size=2880,
        top_k=4,
        num_experts=128,
        quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8,
    ),
}


def _maybe_warn_rank0(msg: str):
    if mpi_rank() == 0:
        print(msg, flush=True)


def _sync():
    torch.cuda.synchronize()
    mpi_barrier()


def _set_device_from_local_rank():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    local_rank = local_mpi_rank()
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            "Detected GPU oversubscription: "
            f"local_mpi_rank={local_rank} >= cuda_device_count={device_count}. "
            "Reduce local MPI ranks to match visible GPU count "
            "(e.g. srun --ntasks-per-node=<gpus>, "
            "mpirun --map-by ppr:<gpus>:node, "
            "or adjust CUDA_VISIBLE_DEVICES)."
        )
    dev = local_rank % device_count
    torch.cuda.set_device(dev)
    return dev


def _make_inputs(
    local_num_tokens: int,
    hidden_size: int,
    top_k: int,
    num_experts_total: int,
    experts_per_rank: int,
    act_dtype: torch.dtype,
    device: torch.device,
    quant_algo: QuantAlgo,
    backend: Communication,
    moe: Optional[MoE] = None,
    perfect_router: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    # Hidden states: payload we want to communicate.
    hidden_states = torch.randn(local_num_tokens, hidden_size, dtype=act_dtype, device=device)
    # We keep scaling factors optional; most strategies can ignore it.
    hidden_states_sf = None
    # Post-quant communication: Quantize → Dispatch (mirrors ConfigurableMoE ordering),
    # using Cutlass' quantize_input() (outside the timed comm region).
    if quant_algo != QuantAlgo.NO_QUANT and backend.supports_post_quant_dispatch():
        hidden_states, hidden_states_sf = moe.quantize_input(hidden_states, post_quant_comm=True)
    if perfect_router:
        assert experts_per_rank > 0
        assert num_experts_total % experts_per_rank == 0
        ep_size = num_experts_total // experts_per_rank
        rank = mpi_rank()
        assert 0 <= rank < ep_size

        # Fair routing across both ranks and experts:
        # flatten (token, top-k) slots into a sequence and cycle as:
        #   rank r expert0, rank r+1 expert0, ..., then rank r expert1, ...
        flat_slots = torch.arange(local_num_tokens * top_k, device=device, dtype=torch.int64)
        schedule = flat_slots + rank
        target_rank = schedule % ep_size
        local_expert = (schedule // ep_size) % experts_per_rank
        token_selected_slots = (
            (target_rank * experts_per_rank + local_expert)
            .view(local_num_tokens, top_k)
            .to(torch.int32)
        )
    else:
        # Routing IDs: global expert IDs in [0, num_experts_total).
        token_selected_slots = torch.randint(
            0,
            num_experts_total,
            (local_num_tokens, top_k),
            dtype=torch.int32,
            device=device,
        )
        # Router weights/scales.

    # The value of token_final_scales doesn't matter for communication.
    token_final_scales = torch.rand(local_num_tokens, top_k, dtype=torch.float32, device=device)

    return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales


def _create_mapping(ep_size: int) -> Mapping:
    rank = mpi_rank()
    return Mapping(
        rank=rank,
        tp_size=ep_size,
        moe_ep_size=ep_size,
        enable_attention_dp=True,
        world_size=ep_size,
    )


def _create_model_config(
    *,
    mapping: Mapping,
    hidden_size: int,
    act_dtype: torch.dtype,
    max_num_tokens_per_rank: int,
    quant_config: Optional[QuantConfig],
    use_low_precision_moe_combine: bool = False,
) -> ModelConfig:
    # Keep it minimal: just enough fields for CommunicationFactory.
    return ModelConfig(
        pretrained_config=_DummyPretrainedConfig(hidden_size=hidden_size, torch_dtype=act_dtype),
        mapping=mapping,
        quant_config=(quant_config or QuantConfig()),
        max_num_tokens=int(max_num_tokens_per_rank),
        moe_max_num_tokens=int(max_num_tokens_per_rank),
        use_cuda_graph=False,
        use_low_precision_moe_combine=use_low_precision_moe_combine,
    )


def _time_dispatch_and_combine(
    backend: Communication,
    *,
    hidden_states: torch.Tensor,
    hidden_states_sf: Optional[torch.Tensor],
    token_selected_slots: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    all_rank_num_tokens: List[int],
    hidden_size: int,
    warmup: int,
    iters: int,
    flush_l2: bool = True,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Time dispatch and combine using Kineto (torch.profiler with CUPTI).

    Returns:
        dispatch_times_us: Per-iteration dispatch GPU times in microseconds
        combine_times_us: Per-iteration combine GPU times in microseconds
        detailed_stats: Dict containing per-kernel timing breakdown
    """
    device = hidden_states.device

    # L2 cache flushing buffer
    l2_buffer = None
    if flush_l2:
        l2_size = torch.cuda.get_device_properties(device).L2_cache_size
        # Use 2x L2 size to ensure complete flush
        l2_flush_size = (l2_size * 2) // 4  # Size in int32 elements
        l2_buffer = torch.empty(l2_flush_size, dtype=torch.int32, device=device)

    # Profile with Kineto
    with torch.profiler.profile(
        # Include CPU so `record_function("dispatch"/"combine")` ranges appear in
        # key_averages() / events(). Without CPU activity those ranges are missing,
        # causing dispatch/combine attribution to fail.
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        _sync()

        # Warmup iterations (not profiled)
        for _ in range(warmup):
            if l2_buffer is not None:
                l2_buffer.zero_()
            backend.prepare_dispatch(
                token_selected_slots, all_rank_num_tokens
            )  # For most ranks this is no-op except for NVLINK_TWO_SIDED
            recv_hidden_states, _, _, _ = backend.dispatch(
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                token_final_scales,
                all_rank_num_tokens,
            )
            shape = list(recv_hidden_states.shape)
            shape[-1] = hidden_size
            recv_hidden_states_moe = torch.empty(
                tuple(shape), dtype=torch.bfloat16, device=recv_hidden_states.device
            )
            _ = backend.combine(
                recv_hidden_states_moe, all_rank_max_num_tokens=max(all_rank_num_tokens)
            )

        # Timed iterations
        for _ in range(iters):
            # L2 cache flushing
            if l2_buffer is not None:
                l2_buffer.zero_()

            # Mark dispatch operation for aggregated timing
            with torch.profiler.record_function("dispatch"):
                backend.prepare_dispatch(
                    token_selected_slots, all_rank_num_tokens
                )  # For most ranks this is no-op except for NVLINK_TWO_SIDED
                recv_hidden_states, _, _, _ = backend.dispatch(
                    hidden_states,
                    hidden_states_sf,
                    token_selected_slots,
                    token_final_scales,
                    all_rank_num_tokens,
                )

            # Simulate MoE computation output
            shape = list(recv_hidden_states.shape)
            shape[-1] = hidden_size
            recv_hidden_states_moe = torch.empty(
                tuple(shape), dtype=torch.bfloat16, device=recv_hidden_states.device
            )

            # Mark combine operation for aggregated timing
            with torch.profiler.record_function("combine"):
                _ = backend.combine(
                    recv_hidden_states_moe, all_rank_max_num_tokens=max(all_rank_num_tokens)
                )

    _sync()
    # if mpi_rank() == 0:
    #     print("########################################################")
    #     print(prof.key_averages())
    #     print("########################################################")
    return _parse_profiler_events(list(prof.events()))


def _parse_profiler_events(
    events_list: list,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Parse Kineto profiler events into per-iteration times and kernel breakdown.

    Expects the profiler to have been run with record_function("dispatch") and
    record_function("combine") wrapping each operation (works for both eager
    kernels and CUDA graph replays).
    """
    # if mpi_rank() == 0:
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     for evt in events_list:
    #         print(evt)
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    def _is_gpu_event(evt) -> bool:
        return getattr(evt, "device_type", None) == DeviceType.CUDA

    # Step 1: Collect GPU time ranges of "dispatch"/"combine" CUDA ranges.
    gpu_dispatch_intervals: List[Tuple[int, int]] = []
    gpu_combine_intervals: List[Tuple[int, int]] = []

    for evt in events_list:
        if not _is_gpu_event(evt) or evt.name not in ("dispatch", "combine"):
            continue
        tr = getattr(evt, "time_range", None)
        if tr is None:
            continue
        assert tr.end > tr.start
        (gpu_dispatch_intervals if evt.name == "dispatch" else gpu_combine_intervals).append(
            (tr.start, tr.end)
        )
    gpu_dispatch_intervals.sort()
    gpu_combine_intervals.sort()

    # Step 2: Scope resolver (GPU events only) ---------------------------------
    def _find_scope(evt) -> Optional[str]:
        """Return scope only when kernel range is strictly contained."""
        tr = getattr(evt, "time_range", None)
        if tr is None:
            return None

        # Be careful: Due to PDL, the end of dispatch and the start of combine may overlap,
        # so we say a kernel is in dispatch/combine only if its range is strictly contained in a dispatch/combine range.
        in_dispatch = any(s <= tr.start and tr.end <= e for s, e in gpu_dispatch_intervals)
        in_combine = any(s <= tr.start and tr.end <= e for s, e in gpu_combine_intervals)

        assert not (in_dispatch and in_combine), (
            f"Kernel range is simultaneously inside dispatch and combine ranges: {evt.name}"
        )

        if in_dispatch:
            return "dispatch"
        if in_combine:
            return "combine"

        # Neither in dispatch or combine (like the element-wise kernel for L2 cache flushing) -> uncategorized.
        return None

    # Step 3: Iterate events and bucket by scope ----------------------------
    dispatch_kernel_times: Dict[str, List[float]] = {}
    combine_kernel_times: Dict[str, List[float]] = {}
    other_kernel_times: Dict[str, List[float]] = {}

    for evt in events_list:
        if not _is_gpu_event(evt):
            continue
        if evt.device_time <= 0:
            continue
        if evt.name in ("dispatch", "combine"):
            continue  # skip record_function range markers

        scope = _find_scope(evt)
        if scope == "dispatch":
            dispatch_kernel_times.setdefault(evt.name, []).append(evt.device_time)
        elif scope == "combine":
            combine_kernel_times.setdefault(evt.name, []).append(evt.device_time)
        else:
            other_kernel_times.setdefault(evt.name, []).append(evt.device_time)

    # Step 4: Build per-kernel stats ----------------------------------------
    def _build_kernel_list(kernel_times: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        result = []
        for name, times in kernel_times.items():
            result.append(
                {
                    "name": name,
                    "count": len(times),
                    "_times": times,  # raw per-iteration times, gathered across ranks later
                }
            )
        return result

    dispatch_kernels = _build_kernel_list(dispatch_kernel_times)
    combine_kernels = _build_kernel_list(combine_kernel_times)
    other_kernels = _build_kernel_list(other_kernel_times)

    # Step 5: Collect per-iteration dispatch/combine times (us) ---------------
    # Use the CUDA-side "dispatch"/"combine" range events (device_type=CUDA)
    # for direct GPU time measurement.
    dispatch_times_us: List[float] = []
    combine_times_us: List[float] = []
    for evt in events_list:
        if not _is_gpu_event(evt):
            continue
        if evt.name == "dispatch":
            dispatch_times_us.append(evt.device_time)
        elif evt.name == "combine":
            combine_times_us.append(evt.device_time)

    # Sort each category by mean time descending
    dispatch_kernels.sort(
        key=lambda x: sum(x["_times"]) / len(x["_times"]) if x["_times"] else 0, reverse=True
    )
    combine_kernels.sort(
        key=lambda x: sum(x["_times"]) / len(x["_times"]) if x["_times"] else 0, reverse=True
    )
    other_kernels.sort(
        key=lambda x: sum(x["_times"]) / len(x["_times"]) if x["_times"] else 0, reverse=True
    )

    detailed_stats = {
        "dispatch_kernels": dispatch_kernels,
        "combine_kernels": combine_kernels,
        "other_kernels": other_kernels,
    }

    return dispatch_times_us, combine_times_us, detailed_stats


def _demangle_names(names: List[str]) -> Dict[str, str]:
    """Demangle C++ symbol names via cxxfilt. Returns {mangled: demangled}."""
    try:
        import cxxfilt

        return {n: cxxfilt.demangle(n) for n in names}
    except Exception:
        return {n: n for n in names}


def _build_cuda_graph_kernel_stats_cupti(
    cupti_kernels: List[Tuple[str, int, int]],  # (name, start_ns, end_ns)
    cupti_events: List[int],  # device_timestamps of EXTERNAL events, sorted
    iters: int,
) -> Optional[Dict[str, Any]]:
    """Categorize GPU kernels from a CUDA graph replay into dispatch/combine/other.

    Uses CUPTI kernel timestamps and CUPTI CUDA_EVENT device_timestamps, all in the
    same GPU nanosecond clock domain.

    The graph records 4 EXTERNAL events per timed iteration (no events during warmup):
      event 4*i+0 → d_starts[i],  4*i+1 → d_ends[i]
      event 4*i+2 → c_starts[i],  4*i+3 → c_ends[i]

    Each kernel is classified by whether its (k_start, k_end) falls within a
    dispatch or combine window; everything else (including warmup kernels) is other.

    Returns None if CUPTI events are missing.

    The returned dict includes:
      dispatch_times_us / combine_times_us: per-iter kernel-span times (ns → µs),
        computed as (last_kernel_end − first_kernel_start) within each window.
        None for iterations where no kernels were attributed (caller should fall back
        to CUDA-event elapsed_time for those iterations).
    """
    expected_events = 4 * iters
    if len(cupti_events) != expected_events:
        _maybe_warn_rank0(
            f"[bench] CUPTI kernel breakdown skipped: expected {expected_events} CUDA_EVENT "
            f"records ({iters} iters × 4) but got {len(cupti_events)}. "
            "This usually means _try_init_cupti() was called after CUDA context creation."
        )
        return None
    if not cupti_kernels:
        return None

    d_starts_abs = [cupti_events[4 * i + 0] for i in range(iters)]
    d_ends_abs = [cupti_events[4 * i + 1] for i in range(iters)]
    c_starts_abs = [cupti_events[4 * i + 2] for i in range(iters)]
    c_ends_abs = [cupti_events[4 * i + 3] for i in range(iters)]

    unique_names = list({name for name, _, _ in cupti_kernels})
    dm = _demangle_names(unique_names)

    dispatch_kernel_times: Dict[str, List[float]] = {}
    combine_kernel_times: Dict[str, List[float]] = {}
    other_kernel_times: Dict[str, List[float]] = {}

    # Per-iteration [first_start_ns, last_end_ns] for kernel-span timing.
    dispatch_iter_span: List[List[Optional[int]]] = [[None, None] for _ in range(iters)]
    combine_iter_span: List[List[Optional[int]]] = [[None, None] for _ in range(iters)]

    for name, k_start, k_end in cupti_kernels:
        demangled = dm.get(name, name)
        device_time_us = (k_end - k_start) / 1e3  # ns → µs

        category = "other"
        iter_idx = -1
        for i in range(iters):
            if k_start >= d_starts_abs[i] and k_end <= d_ends_abs[i]:
                category = "dispatch"
                iter_idx = i
                break
            if k_start >= c_starts_abs[i] and k_end <= c_ends_abs[i]:
                category = "combine"
                iter_idx = i
                break

        if category == "dispatch":
            span = dispatch_iter_span[iter_idx]
            span[0] = k_start if span[0] is None else min(span[0], k_start)
            span[1] = k_end if span[1] is None else max(span[1], k_end)
            dispatch_kernel_times.setdefault(demangled, []).append(device_time_us)
        elif category == "combine":
            span = combine_iter_span[iter_idx]
            span[0] = k_start if span[0] is None else min(span[0], k_start)
            span[1] = k_end if span[1] is None else max(span[1], k_end)
            combine_kernel_times.setdefault(demangled, []).append(device_time_us)
        else:
            other_kernel_times.setdefault(demangled, []).append(device_time_us)

    def _build(ktimes: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        result = [{"name": n, "count": len(t), "_times": t} for n, t in ktimes.items()]
        result.sort(
            key=lambda x: sum(x["_times"]) / len(x["_times"]) if x["_times"] else 0, reverse=True
        )
        return result

    dispatch_times_us = [
        (span[1] - span[0]) / 1e3 if span[0] is not None else None for span in dispatch_iter_span
    ]
    combine_times_us = [
        (span[1] - span[0]) / 1e3 if span[0] is not None else None for span in combine_iter_span
    ]

    return {
        "dispatch_kernels": _build(dispatch_kernel_times),
        "combine_kernels": _build(combine_kernel_times),
        "other_kernels": _build(other_kernel_times),
        "dispatch_times_us": dispatch_times_us,
        "combine_times_us": combine_times_us,
    }


def _try_init_cupti():
    """Try to initialize CUPTI for CUDA-graph kernel breakdown.

    MUST be called BEFORE the CUDA context is created (i.e. before any torch.cuda.*
    call).  CUPTI CUDA_EVENT activities are only delivered to subscribers registered
    before the CUDA context is initialized; late registration silently drops them.

    Also must be called before any NVLINK/NVLink backend creation: NVLINK_ONE_SIDED's
    NVLink initialization changes CUDA profiling state in a way that prevents
    CONCURRENT_KERNEL tracking if CUPTI is enabled afterwards.

    Returns (cupti_module, kernels_list, event_timestamps_list, is_available).
    """
    try:
        from functools import partial as _partial

        from cupti import cupti as _cupti

        _cupti_kernels: List[Tuple[str, int, int]] = []
        _cupti_events: List[int] = []  # device_timestamps of CUDA event records, in arrival order

        def _buf_requested():
            return 8 * 1024 * 1024, 0

        def _buf_completed(kernels, events, activities):
            for act in activities:
                if act.kind == _cupti.ActivityKind.CONCURRENT_KERNEL:
                    kernels.append((act.name, act.start, act.end))
                elif act.kind == _cupti.ActivityKind.CUDA_EVENT:
                    events.append(act.device_timestamp)

        _cupti.activity_enable(_cupti.ActivityKind.CONCURRENT_KERNEL)
        _cupti.activity_enable(_cupti.ActivityKind.CUDA_EVENT)
        _cupti.activity_enable_cuda_event_device_timestamps(1)
        _cupti.activity_register_callbacks(
            _buf_requested, _partial(_buf_completed, _cupti_kernels, _cupti_events)
        )
        return _cupti, _cupti_kernels, _cupti_events, True
    except Exception:
        return None, [], [], False


def _time_dispatch_and_combine_cuda_graph(
    backend: Communication,
    *,
    hidden_states: torch.Tensor,
    hidden_states_sf: Optional[torch.Tensor],
    token_selected_slots: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    all_rank_num_tokens: List[int],
    hidden_size: int,
    warmup: int,
    iters: int,
    flush_l2: bool = True,
    cupti_ctx: Optional[Any] = None,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Time dispatch and combine using an unrolled CUDA graph + embedded CUDA events.

    Order:
      1. One eager dispatch+combine to discover recv shape → allocate static_moe_out → sync.
      2. Capture a single big graph with `iters` iterations unrolled.
         Each iteration: d_starts[i].record → dispatch → d_ends[i].record
                         → zero_ → c_starts[i].record → combine → c_ends[i].record
      3. Warmup: `warmup` eager iterations (no graph).
      4. Timed: one big_graph.replay() → GPU runs all iters back-to-back with zero CPU overhead.
      5. Sync, read per-iter timings from events.
      6. Profiler pass (two small graphs) for kernel breakdown.

    L2 cache is flushed before each iteration inside the graph (including warmup),
    matching the eager-mode behaviour.

    Returns same types as _time_dispatch_and_combine.
    """
    device = hidden_states.device
    max_tokens = max(all_rank_num_tokens)

    l2_buffer = None
    if flush_l2:
        l2_size = torch.cuda.get_device_properties(device).L2_cache_size
        l2_flush_size = (l2_size * 2) // 4
        l2_buffer = torch.empty(l2_flush_size, dtype=torch.int32, device=device)

    # ---- 0. CUPTI state ----
    # cupti_ctx is pre-initialized before backend creation (NVLINK_ONE_SIDED's NVLink
    # init changes CUDA profiling state; CUPTI must be enabled before that call).
    if cupti_ctx is not None:
        _cupti, _cupti_kernels, _cupti_events, _cupti_available = cupti_ctx
    else:
        _cupti_available = False
        _cupti_kernels: List[Tuple[str, int, int]] = []
        _cupti_events: List[int] = []
        _cupti = None

    # ---- 1. Shape discovery: one eager run ----
    backend.prepare_dispatch(token_selected_slots, all_rank_num_tokens)
    recv_hidden_states, _, _, _ = backend.dispatch(
        hidden_states,
        hidden_states_sf,
        token_selected_slots,
        token_final_scales,
        all_rank_num_tokens,
    )
    static_moe_out = torch.zeros(
        (recv_hidden_states.shape[0], hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    backend.combine(static_moe_out, all_rank_max_num_tokens=max_tokens)
    torch.cuda.synchronize()

    # ---- 2. Capture big graph (iters iterations unrolled) ----
    # cudaEventRecordExternal (0x1, CUDA 11.2+) makes events recorded inside a
    # CUDA graph queryable via elapsed_time() after replay. Without this flag,
    # graph-internal events raise cudaErrorInvalidValue on elapsed_time().
    _cudart = ctypes.CDLL("libcudart.so")
    _cudart.cudaEventRecordWithFlags.restype = ctypes.c_int
    _cudart.cudaEventRecordWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    _CUDA_EVENT_RECORD_EXTERNAL = 0x1

    def _record_external(event: torch.cuda.Event) -> None:
        stream = torch.cuda.current_stream()
        ret = _cudart.cudaEventRecordWithFlags(
            event.cuda_event, stream.cuda_stream, _CUDA_EVENT_RECORD_EXTERNAL
        )
        if ret != 0:
            raise RuntimeError(f"cudaEventRecordWithFlags failed with code {ret}")

    d_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    d_ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    c_ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # Force lazy CUDA event creation — cuda_event handle is null until first record().
    for evt in d_starts + d_ends + c_starts + c_ends:
        evt.record()
    torch.cuda.synchronize()

    # Graph contains warmup + timed iters. Warmup iters have no events (unmeasured).
    # Timed iters have 4 external events each. One replay() runs everything back-to-back,
    # eliminating rank desync between warmup and timed sections.
    big_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(big_graph):
        for _ in range(warmup):
            if l2_buffer is not None:
                l2_buffer.zero_()
            backend.prepare_dispatch(
                token_selected_slots, all_rank_num_tokens
            )  # For most ranks this is no-op except for NVLINK_TWO_SIDED
            backend.dispatch(
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                token_final_scales,
                all_rank_num_tokens,
            )
            static_moe_out.zero_()
            backend.combine(static_moe_out, all_rank_max_num_tokens=max_tokens)
        for i in range(iters):
            if l2_buffer is not None:
                l2_buffer.zero_()
            _record_external(d_starts[i])
            backend.prepare_dispatch(
                token_selected_slots, all_rank_num_tokens
            )  # For most ranks this is no-op except for NVLINK_TWO_SIDED
            backend.dispatch(
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                token_final_scales,
                all_rank_num_tokens,
            )
            _record_external(d_ends[i])
            static_moe_out.zero_()
            _record_external(c_starts[i])
            backend.combine(static_moe_out, all_rank_max_num_tokens=max_tokens)
            _record_external(c_ends[i])

    # ---- 3. Timed replay + kernel breakdown via CUPTI ----
    if _cupti_available:
        # Flush any activities captured before the replay (shape discovery, graph capture
        # dry-run, etc.) and clear lists so only replay activities remain.
        _cupti.activity_flush_all(0)
        _cupti_kernels.clear()
        _cupti_events.clear()

    _sync()
    big_graph.replay()

    _sync()

    if _cupti_available:
        # Flush AFTER _sync() (torch.cuda.synchronize + mpi_barrier) to ensure CUPTI
        # delivers all pending graph-replay activities. flush_all(0) is non-blocking;
        # the preceding synchronize gives CUPTI time to process the replay's records.
        _cupti.activity_flush_all(0)

    dispatch_times_us = [d_starts[i].elapsed_time(d_ends[i]) * 1e3 for i in range(iters)]
    combine_times_us = [c_starts[i].elapsed_time(c_ends[i]) * 1e3 for i in range(iters)]

    if _cupti_available:
        _cupti_kernels.sort(key=lambda k: k[1])
        _cupti_events.sort()  # sort by device_timestamp; CUPTI may deliver out of order

        detailed_stats = _build_cuda_graph_kernel_stats_cupti(_cupti_kernels, _cupti_events, iters)
        if detailed_stats is not None:
            # Replace event-based times with tighter kernel-span times.
            # Fall back per-iter to event timing if no kernels were attributed.
            cupti_dispatch = detailed_stats.pop("dispatch_times_us")
            cupti_combine = detailed_stats.pop("combine_times_us")
            dispatch_times_us = [
                ct if ct is not None else et for ct, et in zip(cupti_dispatch, dispatch_times_us)
            ]
            combine_times_us = [
                ct if ct is not None else et for ct, et in zip(cupti_combine, combine_times_us)
            ]
        else:
            detailed_stats = {"dispatch_kernels": [], "combine_kernels": [], "other_kernels": []}
    else:
        detailed_stats = {"dispatch_kernels": [], "combine_kernels": [], "other_kernels": []}

    return dispatch_times_us, combine_times_us, detailed_stats


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics over a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "mean": mean,
        "median": s[n // 2],
        "stdev": variance**0.5,
        "min": s[0],
        "max": s[-1],
    }


def _gather_per_rank(times_us: List[float], iter_stats: bool = False) -> Dict[str, Any]:
    """Allgather per-iteration times from each rank, return per-rank results.

    If iter_stats=True, return full stats (mean/median/stdev/min/max).
    If iter_stats=False, return just the mean.
    """
    all_times = mpi_allgather(times_us)
    if iter_stats:
        return {f"rank{i}": _compute_stats(t) for i, t in enumerate(all_times)}
    return {f"rank{i}": (sum(t) / len(t) if t else 0.0) for i, t in enumerate(all_times)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified MoE communication microbenchmark (MPI).")
    parser.add_argument(
        "--ep_size",
        type=int,
        default=None,
        help="Number of MPI worker ranks to spawn via MPIPoolExecutor.",
    )
    parser.add_argument(
        "--backend",
        type=lambda s: str(s).upper(),
        default=None,
        choices=[
            "ALLGATHER",
            "NVLINK_ONE_SIDED",
            "NVLINK_TWO_SIDED",
            "DEEPEP",
            "DEEPEPLOWLATENCY",
        ],
        help="Which communication backend to benchmark (default: run all backends).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="deepseek_v3",
        choices=sorted(PROFILES.keys()),
        help=(
            "Named model profile supplying defaults for hidden_size, top_k, and num_experts. "
            "Any of these can be overridden individually via their own flags."
        ),
    )
    parser.add_argument(
        "--hidden_size", type=int, default=None, help="Hidden dimension of the model."
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Number of experts each token is routed to."
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Total number of experts of the model across all EP ranks.",
    )
    parser.add_argument(
        "--quant",
        type=lambda s: QuantAlgo[str(s).upper()] if s is not None else None,
        default=None,
        choices=[q.name for q in QuantAlgo],
        help="Quantization recipe of the model.",
    )
    # Sizes to scan (NCCL-tests style, adapted from bytes -> local_batch_size in tokens).
    parser.add_argument(
        "-b",
        "--minbatch",
        type=int,
        default=640,
        help="Minimum local_batch_size (tokens) to start with. Default: 640. For single size, pass -b N (or -e N).",
    )
    parser.add_argument(
        "-e",
        "--maxbatch",
        type=int,
        default=None,
        help="Maximum local_batch_size (tokens) to end at. For single size, pass -e N (or -b N).",
    )
    # Increments can be either fixed or a multiplication factor. Only one should be used.
    parser.add_argument(
        "-i",
        "--stepbatch",
        type=int,
        default=None,
        help="Fixed increment between local_batch_size values (tokens). Default: 128.",
    )
    parser.add_argument(
        "-f",
        "--stepfactor",
        type=float,
        default=None,
        help="Multiplication factor between local_batch_size values. Default: disabled.",
    )
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument(
        "--max_num_tokens_per_rank",
        type=int,
        default=None,
        help="Max tokens per rank for workspace allocation (defaults to max scanned local_batch_size).",
    )
    parser.add_argument(
        "--kernel_breakdown",
        action="store_true",
        help="Show per-kernel timing breakdown.",
    )
    parser.add_argument(
        "--iter_stats",
        action="store_true",
        help="Show detailed per-iteration stats (mean/median/stdev/min/max) instead of just mean.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to write JSON report file (default: None, stdout only).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="Base random seed for input generation (effective seed is random_seed + rank).",
    )
    parser.add_argument(
        "--perfect_router",
        action="store_true",
        help="Use deterministic balanced router assignments to avoid communication load imbalance.",
    )
    parser.add_argument(
        "--use_low_precision_moe_combine",
        action="store_true",
        default=False,
        help="Enable low-precision (FP8) MoE combine path.",
    )
    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        help="Disable CUDA graph mode. By default, dispatch and combine are captured into CUDA graphs for lower CPU overhead and more accurate timing.",
    )
    parser.add_argument(
        "--pdl",
        action="store_true",
        default=False,
        help="Enable Programmatic Dependent Launch (sets TRTLLM_ENABLE_PDL=1).",
    )
    return parser.parse_args()


def _iter_local_batch_sizes(args: argparse.Namespace) -> List[int]:
    if args.stepbatch is not None and args.stepfactor is not None:
        raise ValueError("Only one of -i/--stepbatch or -f/--stepfactor should be used.")

    if args.minbatch is None and args.maxbatch is None:
        raise ValueError("Must specify at least one of -b/--minbatch or -e/--maxbatch.")

    # For single-size mode, allowing -b N or -e N (or both).
    minb = int(args.minbatch) if args.minbatch is not None else int(args.maxbatch)
    if minb <= 0:
        raise ValueError("--minbatch must be > 0")

    maxb = int(args.maxbatch) if args.maxbatch is not None else minb
    if maxb < minb:
        raise ValueError("--maxbatch must be >= --minbatch")

    if args.stepfactor is not None:
        factor = float(args.stepfactor)
        if factor <= 1.0:
            raise ValueError("--stepfactor must be > 1.0")
        out: List[int] = []
        cur = float(minb)
        while True:
            v = int(cur)
            if v > maxb:
                break
            if v > 0 and (not out or out[-1] != v):
                out.append(v)
            cur *= factor
        return out

    step = int(args.stepbatch or 128)
    if step <= 0:
        raise ValueError("--stepbatch must be > 0")
    return list(range(minb, maxb + 1, step))


def _resolve_profile_args(args: argparse.Namespace) -> Tuple[int, int, int, QuantAlgo]:
    """Returns (hidden_size, top_k, num_experts_total, quant_algo)."""
    prof = PROFILES.get(args.profile) if args.profile is not None else None
    if prof is None and (
        args.hidden_size is None
        or args.top_k is None
        or args.num_experts is None
        or args.quant is None
    ):
        raise ValueError(
            "No --profile specified; must provide --hidden_size, --top_k, --num_experts, and --quant."
        )
    hidden_size = int(args.hidden_size or prof.hidden_size)
    top_k = int(args.top_k or prof.top_k)
    num_experts_total = int(args.num_experts or prof.num_experts)
    quant_algo = args.quant or prof.quant_algo
    return hidden_size, top_k, num_experts_total, quant_algo


_WORKER_ENV = {
    "TRTLLM_CAN_USE_DEEP_EP": "1",
    "TRTLLM_ENABLE_PDL": "0",
}


def _run_benchmark_worker_under_current_mpi(
    args: argparse.Namespace, launcher: str = "spawn"
) -> None:
    # CUPTI MUST be initialized before the CUDA context is created.
    # CUDA_EVENT activities are only delivered to CUPTI subscribers that were registered
    # before the CUDA context was initialized; late registration captures CONCURRENT_KERNEL
    # but silently drops CUDA_EVENT records.  _set_device_from_local_rank() (below) is
    # the first call that creates the CUDA context, so we init CUPTI here.
    _early_cupti_ctx: Optional[Any] = None
    if not args.no_cuda_graph:
        _cupti_module, _cupti_kernels_list, _cupti_events_list, _cupti_ok = _try_init_cupti()
        if _cupti_ok:
            _early_cupti_ctx = (_cupti_module, _cupti_kernels_list, _cupti_events_list, True)

    # Keep benchmark output clean.
    tllm.logger.set_level("error")

    ep_size = mpi_world_size()
    rank = mpi_rank()
    _ = _set_device_from_local_rank()
    device = torch.device("cuda")
    # Keep random inputs reproducible while ensuring different ranks do not get identical samples.
    seed = int(args.random_seed) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    hidden_size, top_k, num_experts_total, quant_algo = _resolve_profile_args(args)
    local_batch_sizes = _iter_local_batch_sizes(args)
    act_dtype = torch.bfloat16
    quant_config = (
        QuantConfig(quant_algo=None)
        if quant_algo == QuantAlgo.NO_QUANT
        else QuantConfig(quant_algo=quant_algo)
    )

    max_scanned_tokens = max(local_batch_sizes)
    max_num_tokens_per_rank = int(args.max_num_tokens_per_rank or max_scanned_tokens)
    if max_num_tokens_per_rank < max_scanned_tokens:
        raise ValueError(
            "--max_num_tokens_per_rank must be >= the maximum scanned local_batch_size "
            "(from -b/--minbatch..-e/--maxbatch)."
        )

    mapping = _create_mapping(ep_size)

    if num_experts_total % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts_total}) must be divisible by ep_size ({ep_size})"
        )
    experts_per_rank = num_experts_total // ep_size
    num_slots = num_experts_total

    benchmark_metadata = {
        "bench": "bench_moe_comm",
        "launcher": launcher,
        "profile": args.profile,
        "backend": args.backend,
        "ep_size": ep_size,
        "hidden_size": hidden_size,
        "local_batch_size": local_batch_sizes,
        "top_k": top_k,
        "dtype": str(act_dtype),
        "quant_algo": quant_algo.name,
        "perfect_router": bool(args.perfect_router),
        "experts_per_rank": experts_per_rank,
        "num_experts_total": num_experts_total,
        "max_num_tokens_per_rank": max_num_tokens_per_rank,
        "random_seed": int(args.random_seed),
        "device_count": torch.cuda.device_count(),
        "cuda_graph": not args.no_cuda_graph,
        "pdl": bool(args.pdl),
    }
    if rank == 0:
        print(json.dumps(benchmark_metadata, indent=2), flush=True)

    backends = (
        ["ALLGATHER", "NVLINK_ONE_SIDED", "NVLINK_TWO_SIDED", "DEEPEP", "DEEPEPLOWLATENCY"]
        if args.backend is None
        else [args.backend]
    )

    all_results: List[Dict[str, Any]] = []

    # CUPTI was initialized before the CUDA context at the top of this function.
    # Reuse that early context; do not re-initialize here (too late for CUDA_EVENT delivery).
    _cupti_ctx: Optional[Any] = _early_cupti_ctx
    if not args.no_cuda_graph and _cupti_ctx is None:
        _maybe_warn_rank0(
            "[bench] CUPTI unavailable; dispatch_us/combine_us will use CUDA event elapsed_time."
        )

    for backend_name in backends:
        try:
            model_config = _create_model_config(
                mapping=mapping,
                hidden_size=hidden_size,
                act_dtype=act_dtype,
                max_num_tokens_per_rank=max_num_tokens_per_rank,
                quant_config=quant_config,
                use_low_precision_moe_combine=args.use_low_precision_moe_combine,
            )

            backend = CommunicationFactory._create_forced_method(  # pylint: disable=protected-access
                backend_name,
                model_config,
                num_experts_total,
                num_slots,
                top_k,
                experts_per_rank,
                payload_in_workspace=False,
                alltoall_result_do_sum=True,
                use_flashinfer=False,
            )

            if backend is None:
                _maybe_warn_rank0(
                    f"[bench_moe_comm] Skipping {backend_name}: factory returned None."
                )
                continue
        except Exception as e:
            _maybe_warn_rank0(f"[bench_moe_comm] Skipping {backend_name}: {type(e).__name__}: {e}")
            continue

        # Post-quant communication: Quantize → Dispatch (mirrors ConfigurableMoE ordering),
        # using Cutlass' quantize_input() (outside the timed comm region).
        moe = None
        if quant_algo != QuantAlgo.NO_QUANT and backend.supports_post_quant_dispatch():
            routing_method = DefaultMoeRoutingMethod(top_k=top_k)
            moe = CutlassFusedMoE(
                routing_method=routing_method,
                num_experts=num_experts_total,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 4,
                dtype=torch.bfloat16,
                reduce_results=False,
                model_config=model_config,
                init_load_balancer=False,
                without_comm=True,
            )
            # Ensure quantization params (e.g., NVFP4 global scale) live on CUDA.
            moe = moe.to(device)

        for local_num_tokens in local_batch_sizes:
            all_rank_num_tokens = mpi_allgather(int(local_num_tokens))
            if not backend.is_workload_feasible(all_rank_num_tokens, num_chunks=1):
                _maybe_warn_rank0(
                    f"[bench_moe_comm] Skipping {backend_name} @ local_batch_size={local_num_tokens}: workload not feasible."
                )
                continue

            hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = (
                _make_inputs(
                    local_num_tokens,
                    hidden_size,
                    top_k,
                    num_experts_total,
                    experts_per_rank,
                    act_dtype,
                    device,
                    quant_algo,
                    backend,
                    moe,
                    bool(args.perfect_router),
                )
            )

            # Time dispatch and combine
            _time_fn = (
                _time_dispatch_and_combine_cuda_graph
                if not args.no_cuda_graph
                else _time_dispatch_and_combine
            )
            time_fn_kwargs: Dict[str, Any] = dict(
                hidden_states=hidden_states,
                hidden_states_sf=hidden_states_sf,
                token_selected_slots=token_selected_slots,
                token_final_scales=token_final_scales,
                all_rank_num_tokens=all_rank_num_tokens,
                hidden_size=hidden_size,
                warmup=int(args.warmup),
                iters=int(args.iters),
                flush_l2=True,
            )
            if not args.no_cuda_graph:
                time_fn_kwargs["cupti_ctx"] = _cupti_ctx
            dispatch_times_us, combine_times_us, detailed_stats = _time_fn(
                backend, **time_fn_kwargs
            )

            iter_stats = bool(args.iter_stats)
            dispatch_stats = _gather_per_rank(dispatch_times_us, iter_stats=iter_stats)
            combine_stats = _gather_per_rank(combine_times_us, iter_stats=iter_stats)

            # Prepare output
            output = {
                "backend": backend_name,
                "local_batch_size": int(local_num_tokens),
                "dispatch_us": dispatch_stats,
                "combine_us": combine_stats,
            }

            # Add kernel breakdown if requested and available
            if args.kernel_breakdown and detailed_stats is not None:
                categories = ("dispatch_kernels", "combine_kernels", "other_kernels")

                # Collect once per rank to avoid desynchronizing MPI collectives when
                # kernel lists differ across ranks.
                local_kernel_payload: Dict[str, Dict[str, List[float]]] = {}
                for category in categories:
                    local_kernel_payload[category] = {
                        kernel["name"]: kernel.get("_times", [])
                        for kernel in detailed_stats.get(category, [])
                    }
                all_kernel_payload = mpi_allgather(local_kernel_payload)

                for category in categories:
                    # Preserve deterministic order by first appearance across ranks.
                    seen = set()
                    kernel_names: List[str] = []
                    for rank_payload in all_kernel_payload:
                        for name in rank_payload.get(category, {}):
                            if name not in seen:
                                seen.add(name)
                                kernel_names.append(name)

                    merged_kernels: List[Dict[str, Any]] = []
                    for name in kernel_names:
                        per_rank_times: List[List[float]] = []
                        for rank_payload in all_kernel_payload:
                            times = rank_payload.get(category, {}).get(name, [])
                            per_rank_times.append(times if isinstance(times, list) else [])

                        if iter_stats:
                            per_rank = {
                                f"rank{i}": _compute_stats(times)
                                for i, times in enumerate(per_rank_times)
                            }
                        else:
                            per_rank = {
                                f"rank{i}": (sum(times) / len(times) if times else 0.0)
                                for i, times in enumerate(per_rank_times)
                            }

                        merged_kernels.append(
                            {
                                "name": name,
                                "count": max((len(times) for times in per_rank_times), default=0),
                                "per_rank": per_rank,
                            }
                        )

                    output[category] = merged_kernels

            if rank == 0:
                print(json.dumps(output, indent=2), flush=True)
                all_results.append(output)

    # Write JSON report if requested
    if rank == 0 and args.output_file and all_results:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        report = {
            "benchmark_metadata": benchmark_metadata,
            "results": all_results,
        }
        with open(args.output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output_file}", flush=True)

    return


def _spawn_worker_main(args_blob: bytes) -> List[Dict[str, Any]]:
    """Worker entrypoint for MPIPoolExecutor.

    Note: MPIPoolExecutor typically serializes callables by name; for scripts executed
    as __main__, we set MPI's pickler to cloudpickle in the parent so this function is
    available to workers (same pattern as our unit tests).
    """
    args = pickle.loads(args_blob)
    # In spawned workers, we are already inside an MPI world of size == ep_size.
    try:
        _run_benchmark_worker_under_current_mpi(args, launcher="spawn")
    except Exception as e:
        # Make worker-side stack trace visible at the parent.
        rank = mpi_rank()
        size = mpi_world_size()
        msg = (
            "[bench_moe_comm worker] uncaught exception:\n"
            f"rank={rank}/{size} local_rank={local_mpi_rank()} pid={os.getpid()}\n"
            "Worker traceback:\n"
            f"{traceback.format_exc()}"
        )
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        raise RuntimeError(msg) from e
    return []


def main() -> None:
    args = parse_args()

    # Make functions/classes in this script pickleable to spawned workers.
    # (Same pattern used in our MPI unit tests, but adapted for a script entrypoint.)
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(  # type: ignore[attr-defined]
        cloudpickle.dumps,
        cloudpickle.loads,
        pickle.HIGHEST_PROTOCOL,
    )

    ep_size = int(args.ep_size or 8)
    if ep_size <= 0:
        raise ValueError("--ep_size must be > 0")

    _worker_env = dict(_WORKER_ENV)
    _worker_env["TRTLLM_ENABLE_PDL"] = "1" if args.pdl else "0"

    world_size = mpi_world_size()
    if world_size > 1:
        if args.ep_size is not None and ep_size != world_size:
            raise ValueError(
                f"--ep_size ({ep_size}) must match external MPI world size ({world_size}) "
                "when running under mpirun."
            )
        # In external MPI mode, workers are already launched, so MPIPoolExecutor(env=...)
        # is not used. Apply worker env directly for parity with spawn mode.
        os.environ.update(_worker_env)
        # Reuse externally launched MPI processes (supports multi-node SPMD).
        _run_benchmark_worker_under_current_mpi(args, launcher="external_mpi")
        return

    if mpi_rank() == 0:
        print(
            json.dumps(
                {
                    "bench": "bench_moe_comm",
                    "launcher": "spawn",
                    "ep_size": ep_size,
                    "profile": args.profile,
                    "backend": args.backend,
                },
                indent=2,
            ),
            flush=True,
        )

    args_blob = cloudpickle.dumps(args)
    executor = MPIPoolExecutor(max_workers=ep_size, env=_worker_env)
    try:
        # Map the same args to all workers; each worker uses its own mpi_rank() and participates
        # in collectives within its spawned MPI world.
        _ = list(executor.map(_spawn_worker_main, [args_blob] * ep_size))
    finally:
        # In some environments shutdown(wait=True) can hang even when all workers are idle.
        # We already consumed all map() results, so use non-blocking shutdown.
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
