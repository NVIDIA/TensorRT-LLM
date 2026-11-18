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
"""``TLLM_KV_CACHE_MANAGER_V2_GPU_CHUNK_MB`` maps the GPU pools from larger physical chunks.

Each chunk is one ``cuMemCreate`` allocation, which a transfer agent registers and every
peer maps separately, so the chunk size sets how many regions a disaggregated peer maps.
Only the mapping changes: the pools are sized exactly as without the variable.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import _gpu_alloc_granularity

MiB = 1 << 20
GiB = 1 << 30
ENV = "TLLM_KV_CACHE_MANAGER_V2_GPU_CHUNK_MB"

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def _cuda_context() -> None:
    torch.cuda.init()
    torch.zeros(1, device="cuda")


def _chunk_sizes(address: int, size: int) -> list[int]:
    """Sizes of the physical allocations backing [address, address + size), in order."""
    import cuda.bindings.driver as drv

    sizes = []
    cur, end = address, address + size
    while cur < end:
        status, base, length = drv.cuMemGetAddressRange(drv.CUdeviceptr(cur))
        assert status == drv.CUresult.CUDA_SUCCESS, status
        assert int(base) == cur
        sizes.append(int(length))
        cur += int(length)
    return sizes


def _make_manager(quota: int, slot_bytes: int):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        DataRole,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        LayerId,
    )

    key = DataRole("key")
    manager = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=64,
            cache_tiers=[GpuCacheTierConfig(quota=quota)],
            layers=[AttentionLayerConfig(LayerId(0), [BufferConfig(key, slot_bytes)])],
        )
    )
    return manager, int(manager.get_mem_pool_base_address(LayerId(0), key))


def test_pool_sizes_do_not_depend_on_the_chunk_size(monkeypatch):
    """Same quota, slot count and sizing unit with and without the variable; only the chunks differ."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import GPU_LEVEL

    # Not a multiple of 512 MiB, so the pool ends in smaller chunks.
    quota = 4 * GiB + 100 * MiB
    unit = _gpu_alloc_granularity(quota)
    assert unit == 8 * MiB
    results = {}
    for value in (None, "512"):
        if value is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, value)
        assert _gpu_alloc_granularity(quota) == unit
        manager, base = _make_manager(quota, 256 << 10)
        try:
            allocated = manager.get_quota(GPU_LEVEL)
            results[value] = (allocated, _chunk_sizes(base, allocated))
        finally:
            manager.shutdown()
    default_bytes, default_chunks = results[None]
    raised_bytes, raised_chunks = results["512"]
    assert raised_bytes == default_bytes == quota + 4 * MiB
    assert default_chunks == [unit] * (default_bytes // unit)
    # 8 x 512 MiB, then the 104 MiB tail as 64 + 32 + 8 MiB.
    assert raised_chunks == [512 * MiB] * 8 + [64 * MiB, 32 * MiB, 8 * MiB]


@pytest.mark.parametrize("value", ["1g", "-512", "abc"])
def test_malformed_value_is_rejected(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(Exception, match=ENV):
        _make_manager(1 * GiB, 256 << 10)


def test_virt_mem_maps_mixed_chunks():
    from tensorrt_llm.runtime.kv_cache_manager_v2._introspection import (
        PooledPhysMemAllocator,
        VirtMem,
    )

    allocator = PooledPhysMemAllocator(2 * MiB, 16 * MiB)
    assert allocator.max_chunk_size == 16 * MiB
    vm = VirtMem(64 * MiB, allocator)
    try:
        vm.realloc(38 * MiB)
        assert _chunk_sizes(int(vm.address), vm.mapped_bytes) == [
            16 * MiB,
            16 * MiB,
            4 * MiB,
            2 * MiB,
        ]
        vm.realloc(20 * MiB)
        assert (vm.usable_bytes, vm.mapped_bytes, vm.num_phys_mem) == (20 * MiB, 32 * MiB, 2)
    finally:
        vm.destroy()
