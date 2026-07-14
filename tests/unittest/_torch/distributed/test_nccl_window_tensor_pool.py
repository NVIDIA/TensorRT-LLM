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

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.distributed.nccl_window_tensor_pool import \
    NCCLWindowTensorPool
from tensorrt_llm.mapping import Mapping


class _FakeLinear(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer("_nccl_window_output", None, persistent=False)


def _make_pool(monkeypatch):
    monkeypatch.setenv("TLLM_NCCL_WINDOW_TENSOR_POOL", "1")
    monkeypatch.setenv("TLLM_NCCL_SYMMETRIC_ZERO_COPY", "1")
    mapping = Mapping(world_size=2, rank=0, tp_size=2)
    pool = NCCLWindowTensorPool(mapping)

    allocations = []
    spares = []
    marked = []
    unmarked = []

    def _allocate(like, capacity, output_width, dtype):
        allocations.append((capacity, output_width, dtype, like.device))
        return torch.empty((capacity, output_width), dtype=dtype)

    monkeypatch.setattr(pool, "_allocate", _allocate)
    monkeypatch.setattr(pool, "_preallocate_spare", spares.append)

    def _mark_group_preallocated():
        marked.append(True)
        return ("marker", )

    monkeypatch.setattr(pool, "_mark_group_preallocated",
                        _mark_group_preallocated)
    monkeypatch.setattr(pool, "_unmark_group_preallocated", unmarked.append)
    return pool, allocations, spares, marked, unmarked


def test_reserve_uses_exact_capacity_and_shares_output_signatures(monkeypatch):
    pool, allocations, spares, marked, _ = _make_pool(monkeypatch)
    first = _FakeLinear()
    second = _FakeLinear()
    other_width = _FakeLinear()
    like = torch.empty(0, dtype=torch.bfloat16)

    pool.register(first, like, 16, torch.bfloat16)
    pool.register(second, like, 16, torch.bfloat16)
    pool.register(other_width, like, 32, torch.bfloat16)
    pool.reserve(7)

    assert allocations == [
        (7, 16, torch.bfloat16, torch.device("cpu")),
        (7, 32, torch.bfloat16, torch.device("cpu")),
    ]
    assert first._nccl_window_output is second._nccl_window_output
    assert first._nccl_window_output.shape == (7, 16)
    assert other_width._nccl_window_output.shape == (7, 32)
    assert len(spares) == 2
    assert marked == [True]
    assert pool.capacity == 7
    assert "_nccl_window_output" not in first.state_dict()

    allocation_count = len(allocations)
    pool.reserve(7)
    assert len(allocations) == allocation_count
    with pytest.raises(RuntimeError, match="cannot resize"):
        pool.reserve(8)


def test_reserve_is_all_or_nothing(monkeypatch):
    pool, _, spares, marked, _ = _make_pool(monkeypatch)
    first = _FakeLinear()
    second = _FakeLinear()
    like = torch.empty(0, dtype=torch.float16)

    pool.register(first, like, 16, torch.float16)
    pool.register(second, like, 32, torch.float16)

    def _allocate(like, capacity, output_width, dtype):
        if output_width == 16:
            return torch.empty((capacity, output_width), dtype=dtype)
        return None

    monkeypatch.setattr(pool, "_allocate", _allocate)
    pool.reserve(5)

    assert first._nccl_window_output is None
    assert second._nccl_window_output is None
    assert pool.capacity == 0
    assert not pool._buffers
    assert not spares
    assert not marked


def test_clear_releases_module_references(monkeypatch):
    pool, _, _, _, unmarked = _make_pool(monkeypatch)
    module = _FakeLinear()
    like = torch.empty(0, dtype=torch.float16)
    pool.register(module, like, 16, torch.float16)
    pool.reserve(3)

    retained = module._nccl_window_output
    assert retained is not None
    assert retained.narrow(0, 0, 1).data_ptr() == retained.data_ptr()

    pool.clear()

    assert module._nccl_window_output is None
    assert pool.capacity == 0
    assert not pool._buffers
    assert not pool._registrations
    assert unmarked == [("marker", )]


def test_disabled_pool_does_not_register_or_allocate(monkeypatch):
    monkeypatch.delenv("TLLM_NCCL_WINDOW_TENSOR_POOL", raising=False)
    mapping = Mapping(world_size=2, rank=0, tp_size=2)
    pool = NCCLWindowTensorPool(mapping)
    module = _FakeLinear()

    pool.register(module, torch.empty(0), 16, torch.float32)
    pool.reserve(4)

    assert not pool.enabled
    assert not pool._registrations
    assert module._nccl_window_output is None
