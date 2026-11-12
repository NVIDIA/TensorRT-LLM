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
"""Builders shared by the GPU memory monitor tests."""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch.memory.monitor.capability import Capability

MB = 1 << 20
GIB = 1 << 30

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def capability(**overrides: object) -> Capability:
    """A native-allocator capability with every probe answering yes."""
    fields: dict[str, object] = dict(
        allocator_backend="native",
        alloc_conf="",
        alloc_conf_var=None,
        expandable_segments=False,
        metadata_supported=True,
        nvml_available=True,
    )
    fields.update(overrides)
    return Capability(**fields)


def capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Warnings logged from now on, each once-only warning included again."""
    from tensorrt_llm.logger import logger

    warnings: list[str] = []
    monkeypatch.setattr(logger, "_appeared_keys", set())

    def log(severity, *msg):
        if severity == logger.WARNING:
            warnings.append(" ".join(str(m) for m in msg))

    monkeypatch.setattr(logger, "log", log)
    return warnings


def kv_stats(
    total: int, free: int, evictable: int, slot_sizes: tuple[int, ...] = (MB,)
) -> types.SimpleNamespace:
    """Shaped like the nanobind StorageStatistics, derived properties included.

    The derived ``available``/``unavailable`` are present so a test can show
    that the checks read the stored fields, not these.
    """
    stats = types.SimpleNamespace(
        total=total, free=free, evictable=evictable, slot_sizes=tuple(slot_sizes)
    )
    stats.available = lambda: stats.free + stats.evictable
    stats.unavailable = lambda: stats.total - stats.available()
    return stats


class FakeKvManager:
    """A KV cache manager V2 stand-in.

    A real class rather than a SimpleNamespace: providers hold the manager
    weakly, and SimpleNamespace forbids weak references.
    """

    def __init__(self, *stats: types.SimpleNamespace) -> None:
        self._stats = list(stats)

    def get_storage_statistics(self, level: object = None) -> list[types.SimpleNamespace]:
        return self._stats


def capture_graph(
    size: int, scratch: bool = False
) -> tuple[torch.cuda.CUDAGraph, tuple, torch.Tensor]:
    """Capture a CUDA graph that keeps ``size`` bytes allocated.

    With ``scratch`` the graph also allocates and frees ``size`` bytes during
    capture, which its private pool keeps as free space that only the graph
    can reuse. Returns the graph, its pool handle and the kept output.
    """
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    warm = torch.ones(1024, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm * 2
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph, pool=pool):
        kept = torch.empty(size, dtype=torch.uint8, device="cuda")
        if scratch:
            freed = torch.empty(size, dtype=torch.uint8, device="cuda")
            del freed
    return graph, pool, kept
