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

r"""Optional nsys capture-range control for the timing paths.

When the ``--nsys`` flag is set, the timing loops bracket the measured region
with ``cudaProfilerStart``/``cudaProfilerStop`` and wrap only the forward with an
NVTX range. Running under::

    nsys profile -t cuda,nvtx -c cudaProfilerApi --capture-range-end stop ...

then captures ONLY the measured MoE forward (warmup, autotune, and CUDA-Graph
capture all fall outside the ``cudaProfilerStart``/``Stop`` window), and the
``bench_moe.measured`` NVTX range pinpoints the pure forward (excludes the
per-iter L2-flush memset, matching the CUDA-event latency window).

This path is mutually exclusive with the CUPTI kernel breakdown
(``--analysis kernels``): both use CUPTI and cannot run at the same time. The
worker disables CUPTI when ``--nsys`` is set.

Example (single-GPU)::

    nsys profile -t cuda,nvtx --cuda-graph-trace=node \\
      -c cudaProfilerApi --capture-range-end stop \\
      python -m tests.microbenchmarks.bench_moe ... --nsys --analysis none

Add ``--cuda-graph-trace=node`` for CUDA-graph runs, else nsys records the
replay as one opaque node instead of the individual kernels. For multi-GPU,
``nsys profile`` may go before ``mpirun`` (one report) or after it (per-rank
``-o``). Inspect the ``bench_moe.measured`` NVTX range with::

    nsys stats --report nvtx_pushpop_trace *.nsys-rep
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

# ``nvtx_range`` lives in the main package and wraps the standalone ``nvtx``
# package; import defensively so a missing dependency degrades to a no-op range
# rather than failing the benchmark.
try:
    from tensorrt_llm._utils import nvtx_range as _nvtx_range
except Exception:  # pragma: no cover - defensive
    _nvtx_range = None


class _NsysProfiler:
    """Guarded ``cudaProfilerStart``/``Stop`` wrapper.

    No-op unless ``enabled`` is True. The ``_active`` flag prevents a double
    start/stop (mirrors ``tensorrt_llm/_torch/visual_gen/pipeline.py``).
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._active = False

    def start(self) -> None:
        if self.enabled and not self._active:
            torch.cuda.cudart().cudaProfilerStart()
            self._active = True

    def stop(self) -> None:
        if self._active:
            torch.cuda.cudart().cudaProfilerStop()
            self._active = False


def measured_range(enabled: bool, label: str = "bench_moe.measured") -> ContextManager:
    """NVTX range around the measured forward, or a null context when disabled."""
    if enabled and _nvtx_range is not None:
        return _nvtx_range(label)
    return nullcontext()
