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

"""CUDA Graph capture+replay timing path with external CUDA events."""

from __future__ import annotations

import ctypes
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..utils import _sync
from .cupti import _build_cuda_graph_kernel_stats_cupti
from .eager import _l2_flush_buffer
from .nsys import _NsysProfiler, measured_range


def _time_moe_forward_cuda_graph(
    moe,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    all_rank_num_tokens: List[int],
    *,
    warmup: int,
    iters: int,
    cupti_ctx: Optional[Any] = None,
    flush_l2: bool = True,
    nsys: bool = False,
) -> Tuple[List[float], Dict[str, Any]]:
    """Time ``moe.forward`` inside an unrolled CUDA graph with EXTERNAL events.

    When ``nsys=True`` the measured region is captured for an external
    ``nsys -c cudaProfilerApi`` run (see ``timing/nsys.py``); this requires
    ``cupti_ctx=None`` since CUPTI conflicts with nsys.
    """
    device = x.device if x.numel() > 0 else torch.device("cuda")
    l2_buffer = _l2_flush_buffer(device) if flush_l2 else None
    profiler = _NsysProfiler(nsys)

    if cupti_ctx is not None:
        _cupti = cupti_ctx.module
        _cupti_kernels = cupti_ctx.kernels
        _cupti_events = cupti_ctx.events
        _cupti_available = cupti_ctx.ok
    else:
        _cupti = None
        _cupti_kernels = []
        _cupti_events = []
        _cupti_available = False

    # ---- 1. Pre-capture eager pass for shape discovery and lazy init/codegen.
    with torch.inference_mode():
        eager_out = moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
    if not isinstance(eager_out, torch.Tensor):
        raise RuntimeError(
            "CUDA-Graph timing requires a tensor output from moe.forward; got "
            f"{type(eager_out).__name__}. Use --no_cuda_graph for this case."
        )
    torch.cuda.synchronize()

    # ---- 2. Pre-create events; ``cudaEventRecordWithFlags`` makes graph-internal
    # events queryable via elapsed_time().
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

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for evt in starts + ends:
        evt.record()
    torch.cuda.synchronize()

    big_graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(big_graph):
        for i in range(iters):
            if l2_buffer is not None:
                l2_buffer.zero_()
            _record_external(starts[i])
            moe.forward(x, router_logits, all_rank_num_tokens=all_rank_num_tokens)
            _record_external(ends[i])

    _sync()
    for _ in range(warmup):
        big_graph.replay()
    _sync()

    if _cupti_available:
        _cupti.activity_flush_all(0)
        _cupti_kernels.clear()
        _cupti_events.clear()

    # Start the nsys capture AFTER warmup replays so only the measured replay is
    # captured (cudaProfilerStart/Stop are host-side, bracketing the replay). The
    # NVTX range wraps the host-side replay() (host NVTX is not recorded into the
    # graph, so it must sit here); it marks the whole unrolled replay window.
    profiler.start()
    with measured_range(nsys):
        big_graph.replay()
    _sync()
    profiler.stop()

    if _cupti_available:
        _cupti.activity_flush_all(0)

    forward_times_ms = [starts[i].elapsed_time(ends[i]) for i in range(iters)]

    if _cupti_available:
        _cupti_kernels.sort(key=lambda k: k[1])
        _cupti_events.sort()
        cupti_stats = _build_cuda_graph_kernel_stats_cupti(_cupti_kernels, _cupti_events, iters)
        if cupti_stats is not None:
            cupti_times = cupti_stats.pop("moe_times_ms")
            forward_times_ms = [
                ct if ct is not None else et for ct, et in zip(cupti_times, forward_times_ms)
            ]
            detailed_stats = cupti_stats
        else:
            detailed_stats = {"moe_forward_kernels": [], "other_kernels": []}
    else:
        detailed_stats = {"moe_forward_kernels": [], "other_kernels": []}

    return forward_times_ms, detailed_stats
