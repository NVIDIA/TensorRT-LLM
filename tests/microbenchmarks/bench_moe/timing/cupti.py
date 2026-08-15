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

"""Optional CUPTI activity tracking for CUDA-Graph kernel breakdown.

Both the ``cupti.cupti`` and ``cxxfilt`` packages are optional: when they are
missing on the host we degrade gracefully and the CUDA-Graph timing path
falls back to pure external-event timing without kernel breakdown.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from ..utils import _maybe_print_rank0
from .eager import _kernel_times_to_summary_list


def _try_import(module_path: str) -> Any:
    """Return the imported module or ``None`` on any failure."""
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


_cupti = _try_import("cupti.cupti")
_cxxfilt = _try_import("cxxfilt")


class _CuptiContext(NamedTuple):
    """Initialised CUPTI handles + capture buffers.

    Returned by ``_try_init_cupti``. ``ok`` is False when CUPTI was missing
    or activity registration failed; in that case the other fields are
    empty/None and callers fall back to PyTorch-event-only timing.
    """

    module: Any
    kernels: List[Tuple[str, int, int]]
    events: List[int]
    ok: bool


def _try_init_cupti() -> _CuptiContext:
    """Initialize CUPTI activity tracking before any CUDA context is created.

    Returns a ``_CuptiContext``. When the ``cupti`` Python package is missing
    or registration fails the function degrades gracefully to
    ``_CuptiContext(None, [], [], False)`` and the caller falls back to
    PyTorch-event-only timing without kernel breakdown.

    The two activity kinds we register are:
      - ``CONCURRENT_KERNEL``: every kernel actually executed on the GPU,
        including those replayed from a captured CUDA graph (Kineto cannot see
        these because there is no Python frame during replay).
      - ``CUDA_EVENT``: device-side timestamps for ``cudaEventRecord`` calls;
        we use them to delimit which kernels fall inside each timed iteration.
    """
    if _cupti is None:
        return _CuptiContext(None, [], [], False)
    try:
        _cupti_kernels: List[Tuple[str, int, int]] = []
        _cupti_events: List[int] = []

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
            _buf_requested,
            functools.partial(_buf_completed, _cupti_kernels, _cupti_events),
        )
        return _CuptiContext(_cupti, _cupti_kernels, _cupti_events, True)
    except Exception:
        return _CuptiContext(None, [], [], False)


def _demangle_names(names: List[str]) -> Dict[str, str]:
    """Demangle C++ symbol names via ``cxxfilt`` when available."""
    if _cxxfilt is None:
        return {n: n for n in names}
    try:
        return {n: _cxxfilt.demangle(n) for n in names}
    except Exception:
        return {n: n for n in names}


def _build_cuda_graph_kernel_stats_cupti(
    cupti_kernels: List[Tuple[str, int, int]],
    cupti_events: List[int],
    iters: int,
) -> Optional[Dict[str, Any]]:
    """Categorize replay kernels into moe_forward/other windows via CUPTI."""
    expected_events = 2 * iters
    if len(cupti_events) != expected_events:
        _maybe_print_rank0(
            f"[bench_moe] CUPTI breakdown skipped: expected {expected_events} CUDA_EVENT "
            f"records ({iters} iters x 2) but got {len(cupti_events)}. "
            "Most likely CUPTI was registered after CUDA context creation."
        )
        return None
    if not cupti_kernels:
        return None

    starts_abs = [cupti_events[2 * i + 0] for i in range(iters)]
    ends_abs = [cupti_events[2 * i + 1] for i in range(iters)]

    unique_names = list({name for name, _, _ in cupti_kernels})
    dm = _demangle_names(unique_names)

    moe_kernel_times: Dict[str, List[float]] = {}
    other_kernel_times: Dict[str, List[float]] = {}

    iter_span: List[List[Optional[int]]] = [[None, None] for _ in range(iters)]

    for name, k_start, k_end in cupti_kernels:
        demangled = dm.get(name, name)
        # CUPTI timestamps are nanoseconds; convert to milliseconds.
        device_time_ms = (k_end - k_start) / 1e6

        iter_idx = -1
        for i in range(iters):
            if k_start >= starts_abs[i] and k_end <= ends_abs[i]:
                iter_idx = i
                break

        if iter_idx >= 0:
            span = iter_span[iter_idx]
            span[0] = k_start if span[0] is None else min(span[0], k_start)
            span[1] = k_end if span[1] is None else max(span[1], k_end)
            moe_kernel_times.setdefault(demangled, []).append(device_time_ms)
        else:
            other_kernel_times.setdefault(demangled, []).append(device_time_ms)

    moe_times_ms = [
        (span[1] - span[0]) / 1e6 if span[0] is not None else None for span in iter_span
    ]

    return {
        "moe_forward_kernels": _kernel_times_to_summary_list(moe_kernel_times),
        "other_kernels": _kernel_times_to_summary_list(other_kernel_times),
        "moe_times_ms": moe_times_ms,
    }
