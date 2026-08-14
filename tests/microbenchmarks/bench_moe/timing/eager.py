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

"""Eager-path timing and Kineto kernel breakdown."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.autograd import DeviceType

from ..utils import _maybe_print_rank0, _sync
from .nsys import _NsysProfiler, measured_range


def _kernel_times_to_summary_list(
    kernel_times: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    """Convert ``{kernel_name: [times_ms]}`` into the dashboard summary shape.

    Sorted by per-kernel mean duration descending; entries keep the raw
    ``_times`` list so downstream mpi_allgather can recompute per-rank stats.
    Shared by the Kineto (``_parse_profiler_events_moe``) and CUPTI
    (``_build_cuda_graph_kernel_stats_cupti``) paths so the wire format stays
    in lockstep across the two backends.
    """
    out = [{"name": n, "count": len(t), "_times": t} for n, t in kernel_times.items()]
    out.sort(
        key=lambda entry: (sum(entry["_times"]) / len(entry["_times"])) if entry["_times"] else 0.0,
        reverse=True,
    )
    return out


def _l2_flush_buffer(device: torch.device) -> torch.Tensor:
    """Allocate a 2x-L2 flush buffer to clear L2 between iterations."""
    l2_size = torch.cuda.get_device_properties(device).L2_cache_size
    l2_flush_size = (l2_size * 2) // 4
    return torch.empty(l2_flush_size, dtype=torch.int32, device=device)


def _time_moe_forward_eager(
    moe,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: List[int],
    *,
    warmup: int,
    iters: int,
    flush_l2: bool = True,
    collect_kernels: bool = True,
    nsys: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    """Time eager ``ConfigurableMoE.forward``.

    Latency is ALWAYS measured with pure ``torch.cuda.Event`` records so the
    reported ``score_ms`` is comparable across ``cuda_graph`` and eager paths
    (the CUDA-Graph path also uses external CUDA events for its per-iter
    window). When ``collect_kernels=True`` a separate, shorter profiler pass is
    run only to gather the kernel breakdown; profiler-derived numbers are not
    used to score the candidate.

    When ``nsys=True`` the measured region is captured for an external
    ``nsys -c cudaProfilerApi`` run (see ``timing/nsys.py``); this forces
    ``collect_kernels`` off since CUPTI/Kineto conflicts with nsys.
    """
    if nsys:
        collect_kernels = False
    device = x.device if x.numel() > 0 else torch.device("cuda")
    l2_buffer = _l2_flush_buffer(device) if flush_l2 else None

    def _do_forward():
        with torch.inference_mode():
            _ = moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    profiler = _NsysProfiler(nsys)
    _sync()
    for _ in range(warmup):
        if l2_buffer is not None:
            l2_buffer.zero_()
        _do_forward()
    # Start the nsys capture AFTER warmup so warmup is excluded from the window.
    profiler.start()
    for i in range(iters):
        if l2_buffer is not None:
            l2_buffer.zero_()
        starts[i].record()
        # NVTX wraps ONLY the forward (between the start/end records, excluding
        # the L2-flush memset) so the range == the CUDA-event latency window.
        with measured_range(nsys):
            _do_forward()
        ends[i].record()
    _sync()
    profiler.stop()
    forward_times_ms = [starts[i].elapsed_time(ends[i]) for i in range(iters)]

    detailed_stats: Dict[str, Any] = {
        "moe_forward_kernels": [],
        "other_kernels": [],
    }
    if not collect_kernels:
        return forward_times_ms, detailed_stats

    # Separate profiler-only pass for kernel breakdown. Use a small fixed iter
    # count (capped by ``iters``) so the profiler overhead does not dominate
    # the case wall-clock budget. The latencies produced here are intentionally
    # discarded; only the kernel categorisation is kept.
    breakdown_iters = max(1, min(iters, 3))
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            _sync()
            if l2_buffer is not None:
                l2_buffer.zero_()
            _do_forward()  # one warmup under profiler
            for _ in range(breakdown_iters):
                if l2_buffer is not None:
                    l2_buffer.zero_()
                with torch.profiler.record_function("moe_forward"):
                    _do_forward()
        _sync()
        _, detailed_stats = _parse_profiler_events_moe(list(prof.events()))
    except Exception as exc:
        # Breakdown is best-effort; do not fail the case if Kineto misbehaves.
        _maybe_print_rank0(f"[bench_moe] kernel breakdown skipped: {type(exc).__name__}: {exc}")
    return forward_times_ms, detailed_stats


def _parse_profiler_events_moe(events_list: list) -> Tuple[List[float], Dict[str, Any]]:
    """Parse Kineto events with ``moe_forward`` ranges.

    Returns ``(moe_forward_times_ms, detailed_stats)`` where ``detailed_stats``
    contains ``moe_forward_kernels`` (within the range) and ``other_kernels``.
    """

    def _is_gpu_event(evt) -> bool:
        return getattr(evt, "device_type", None) == DeviceType.CUDA

    gpu_moe_intervals: List[Tuple[int, int]] = []
    for evt in events_list:
        if not _is_gpu_event(evt) or evt.name != "moe_forward":
            continue
        tr = getattr(evt, "time_range", None)
        if tr is None or tr.end <= tr.start:
            continue
        gpu_moe_intervals.append((tr.start, tr.end))
    gpu_moe_intervals.sort()

    def _scope(evt) -> Optional[str]:
        tr = getattr(evt, "time_range", None)
        if tr is None:
            return None
        for s, e in gpu_moe_intervals:
            if s <= tr.start and tr.end <= e:
                return "moe_forward"
        return None

    moe_kernel_times: Dict[str, List[float]] = {}
    other_kernel_times: Dict[str, List[float]] = {}
    for evt in events_list:
        if not _is_gpu_event(evt):
            continue
        if evt.device_time <= 0 or evt.name == "moe_forward":
            continue
        scope = _scope(evt)
        bucket = moe_kernel_times if scope == "moe_forward" else other_kernel_times
        # PyTorch profiler reports device_time in microseconds; convert to ms.
        bucket.setdefault(evt.name, []).append(evt.device_time / 1e3)

    forward_times_ms: List[float] = []
    for evt in events_list:
        if _is_gpu_event(evt) and evt.name == "moe_forward":
            forward_times_ms.append(evt.device_time / 1e3)

    detailed_stats = {
        "moe_forward_kernels": _kernel_times_to_summary_list(moe_kernel_times),
        "other_kernels": _kernel_times_to_summary_list(other_kernel_times),
    }
    return forward_times_ms, detailed_stats
