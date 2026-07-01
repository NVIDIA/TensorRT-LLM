# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for VisualGen UBX all-to-all helpers."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_ubx_alltoall_module():
    """Load the helper module without importing the full tensorrt_llm package."""
    module_path = (
        Path(__file__).resolve().parents[4]
        / "tensorrt_llm"
        / "_torch"
        / "visual_gen"
        / "ubx_alltoall.py"
    )
    spec = importlib.util.spec_from_file_location("ubx_alltoall_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ubx_alltoall = _load_ubx_alltoall_module()


class _FailingAllocator:
    def create_tensor(self, shape, dtype):
        return None


def test_ubx_available_resolves_none_to_default_group(monkeypatch):
    world_group = object()
    seen = {}

    def fake_get_world_size(*, group):
        seen["get_world_size_group"] = group
        return 2

    def fake_sync_ready(local_ready, device, process_group):
        seen["sync_ready_args"] = (local_ready, device, process_group)
        return local_ready

    fake_dist = SimpleNamespace(
        group=SimpleNamespace(WORLD=world_group),
        get_world_size=fake_get_world_size,
        is_available=lambda: True,
        is_initialized=lambda: True,
    )
    monkeypatch.setattr(ubx_alltoall, "dist", fake_dist)
    monkeypatch.setattr(ubx_alltoall, "_sync_ready", fake_sync_ready)

    assert not ubx_alltoall._ubx_available(process_group=None, device=torch.device("cuda", 0))
    assert seen["get_world_size_group"] is world_group
    assert seen["sync_ready_args"][2] is world_group


def test_pool_allocation_failure_releases_cached_state(monkeypatch):
    wrapper = ubx_alltoall.UBXAllToAll(process_group=object())
    state = ubx_alltoall._PoolState(allocator=_FailingAllocator())
    ready_key = (torch.Size([2]), torch.float32)
    state.pool_cache[ready_key] = torch.empty(2)
    state.ready_pool_keys.add(ready_key)
    wrapper._sync_from_state(state)

    monkeypatch.setattr(
        ubx_alltoall, "_sync_ready", lambda local_ready, device, process_group: local_ready
    )

    assert wrapper._get_pool_in(torch.empty(1), torch.device("cuda", 0), object()) is None
    assert state.allocator is None
    assert state.pool_cache == {}
    assert state.ready_pool_keys == set()
    assert wrapper._allocator is None
    assert wrapper._pool_cache == {}
