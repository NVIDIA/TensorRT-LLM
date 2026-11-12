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
"""Bit-exactness of the one-kernel state-index upload against ``Tensor.copy_``."""

import random

import pytest
import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import mamba_cache_manager_v2
from tensorrt_llm._torch.pyexecutor.step_upload_kernels import copy_small_pinned_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _random_values(dtype: torch.dtype, n: int, rng: random.Random):
    if dtype == torch.bool:
        return [rng.random() < 0.5 for _ in range(n)]
    if dtype == torch.int64:
        # Values past 32 bits, so a copy that drops the upper words fails.
        return [rng.choice((-1, 1)) * ((1 << 40) + rng.randrange(1 << 30)) for _ in range(n)]
    return [rng.randrange(-(1 << 30), 1 << 30) for _ in range(n)]


@pytest.mark.parametrize(
    "dtype0,n0,dtype1,n1",
    [
        (torch.int32, 3, torch.bool, 3),
        (torch.int64, 17, torch.int32, 1),
        (torch.int32, 1024, torch.bool, 7),
    ],
)
def test_small_pinned_pair_copy_matches_copy(dtype0, n0, dtype1, n1) -> None:
    rng = random.Random(n0 * 31 + n1)
    srcs = [
        torch.tensor(_random_values(dtype, n, rng), dtype=dtype).pin_memory()
        for dtype, n in ((dtype0, n0), (dtype1, n1))
    ]
    dsts = [torch.zeros(src.shape, dtype=src.dtype, device="cuda") for src in srcs]
    assert copy_small_pinned_pair(dsts[0], srcs[0], dsts[1], srcs[1])
    torch.cuda.synchronize()
    for dst, src in zip(dsts, srcs):
        assert torch.equal(dst.cpu(), src)
    # Pageable source, size mismatch or a large copy: not handled, the caller copies.
    assert not copy_small_pinned_pair(dsts[0], srcs[0].clone(), dsts[1], srcs[1])
    assert not copy_small_pinned_pair(dsts[0][:1], srcs[0], dsts[1], srcs[1])
    big = torch.zeros(2048, dtype=torch.int32).pin_memory()
    assert not copy_small_pinned_pair(big.cuda(), big, dsts[1], srcs[1])


def _state_index_manager(capacity: int, enabled: bool = True):
    manager = object.__new__(mamba_cache_manager_v2.MambaHybridCacheManagerV2)
    manager.__dict__.update(
        _state_index_upload_kernel=enabled,
        _state_upload_snapshots=[],
        _state_indices_buffer=DeviceCopyBuffer(
            torch.full((capacity,), -7, dtype=torch.int32, device="cuda"), pin_memory=True
        ),
        _host_state_indices=torch.zeros(capacity, dtype=torch.int32).pin_memory(),
        _dummy_request_mask_buffer=DeviceCopyBuffer(
            torch.ones(capacity, dtype=torch.bool, device="cuda"), pin_memory=True
        ),
        _dummy_request_mask_host=torch.ones(capacity, dtype=torch.bool).pin_memory(),
    )
    return manager


def test_state_index_upload_kernel_matches_copies() -> None:
    assert not _state_index_manager(4, enabled=False)._upload_state_indices_and_dummy_mask([False])
    manager = _state_index_manager(4)
    indices = manager._state_indices_buffer.tensor
    mask = manager._dummy_request_mask_buffer.tensor
    manager._host_state_indices[:3] = torch.tensor([11, 4, 0], dtype=torch.int32)
    assert manager._upload_state_indices_and_dummy_mask([False, True])
    torch.cuda.synchronize()
    assert indices.tolist() == [11, 4, 0, 0]
    assert mask.tolist() == [False, True, False, False]

    # Unchanged values: nothing is written, as DeviceCopyBuffer.update would skip both.
    indices.fill_(-7)
    mask.fill_(True)
    assert manager._upload_state_indices_and_dummy_mask([False, True])
    torch.cuda.synchronize()
    assert indices.tolist() == [-7] * 4
    assert mask.tolist() == [True] * 4

    # A change in either buffer writes both; later DeviceCopyBuffer updates see the values.
    manager._host_state_indices[:3] = torch.tensor([11, 4, 9], dtype=torch.int32)
    assert manager._upload_state_indices_and_dummy_mask([False, True, False])
    torch.cuda.synchronize()
    assert indices.tolist() == [11, 4, 9, 0]
    assert mask.tolist() == [False, True, False, False]
    assert manager._state_indices_buffer.matches(manager._host_state_indices)
    assert manager._dummy_request_mask_buffer.matches(manager._dummy_request_mask_host)
    assert manager._state_upload_snapshots


def test_state_index_upload_falls_back_for_large_tables() -> None:
    manager = _state_index_manager(2048)
    assert not manager._upload_state_indices_and_dummy_mask([False])
