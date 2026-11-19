# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-event phase timer shared by the benchmark and output-head wrapper."""

from __future__ import annotations

import functools
import time

import torch


class EventTimer:
    """CUDA-event bracket around named calls; ``collect`` returns per-call ms after a sync."""

    def __init__(self) -> None:
        self.records = []  # (name, start_event, end_event, host_seconds)

    def wrap(self, target: object, method: str, name: str, nvtx: str | None = None) -> None:
        original = getattr(target, method)

        @functools.wraps(original)
        def wrapped(*args: object, **kwargs: object) -> object:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if nvtx:
                torch.cuda.nvtx.range_push(nvtx)
            start.record()
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                if nvtx:
                    torch.cuda.nvtx.range_pop()
                self.records.append((name, start, end, time.perf_counter() - t))

        setattr(target, method, wrapped)

    def collect(self) -> dict[str, list[dict[str, float]]]:
        torch.cuda.synchronize()
        out = {}
        for name, start, end, host in self.records:
            out.setdefault(name, []).append({"ms": start.elapsed_time(end), "host_ms": host * 1e3})
        self.records = []
        return out
