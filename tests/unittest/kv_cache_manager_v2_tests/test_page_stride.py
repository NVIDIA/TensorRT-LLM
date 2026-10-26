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
"""Native layers holding pages at a block stride: one manager, several page layouts.

A layer whose model spreads pages round-robin over ``s`` ranks holds, on the
rank at position ``f`` of that layout, exactly the blocks ``b`` with
``b % s == f`` (block ``b`` is logical page ``b``), and lists them in order in
its base page index buffer. Every expected value below comes from the helix
rule (page ``p`` on rank ``p % s``), independently of the manager.
"""

import pytest
import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]

TOKENS_PER_BLOCK = 4
BAD = -1
# name -> (page stride, page phase, slot bytes): a helix CP4 target on CP rank 1,
# a CP2 drafter on CP rank 1, and a drafter that holds every page.
LAYOUTS = {"target": (4, 1, 4096), "draft": (2, 1, 8192), "full": (1, 0, 12288)}


def div_up(a: int, b: int) -> int:
    return -(-a // b)


def owned_pages(stride: int, phase: int, num_pages: int) -> list[int]:
    return [page for page in range(num_pages) if page % stride == phase]


def make_manager(quota: int = 64 << 20, phases=None) -> kv.KVCacheManager:
    layers = []
    for layer_id, (name, (stride, phase, size)) in enumerate(LAYOUTS.items()):
        if phases is not None:
            phase = phases.get(name, phase)
        layers.append(
            kv.AttentionLayerConfig(
                layer_id=kv.LayerId(layer_id),
                buffers=[kv.BufferConfig(role="key", size=size)],
                reuse_policy=kv.AttentionReusePolicy.PRIVATE,
                page_stride=stride,
                page_phase=phase,
            )
        )
    return kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[kv.GpuCacheTierConfig(quota=quota)],
            layers=layers,
        )
    )


class Probe:
    def __init__(self, manager):
        self.manager = manager
        self.groups = {
            name: manager.get_layer_group_id(kv.LayerId(i)) for i, name in enumerate(LAYOUTS)
        }
        pool_of_group = manager.get_life_cycle_pool_group_indices(0)
        self.pools = {name: pool_of_group[group] for name, group in self.groups.items()}
        self.free = [s.free for s in manager.get_storage_statistics(0)]

    def used(self, name: str) -> int:
        pool = self.pools[name]
        return self.free[pool] - self.manager.get_storage_statistics(0)[pool].free

    def check(self, cache, num_tokens: int, used_by_others=None) -> None:
        num_pages = div_up(num_tokens, TOKENS_PER_BLOCK)
        admitted = self.manager.get_strided_admitted_slots()
        for name, (stride, phase, _) in LAYOUTS.items():
            owned = owned_pages(stride, phase, num_pages)
            group = self.groups[name]
            by_ordinal = list(cache.get_aggregated_page_indices(group))
            assert [b for b, slot in enumerate(by_ordinal) if slot != BAD] == owned, (
                name,
                num_tokens,
                by_ordinal,
            )
            table = list(cache.get_base_page_indices(group))[:num_pages]
            assert table[: len(owned)] == [by_ordinal[b] for b in owned], (name, num_tokens)
            assert all(slot == BAD for slot in table[len(owned) :]), (name, num_tokens, table)
            assert self.used(name) == len(owned) + (used_by_others or {}).get(name, 0)
            # The admission charge: the most any rank of the layout holds.
            assert admitted[self.pools[name]] == (div_up(num_pages, stride) if stride > 1 else 0)


@pytest.fixture
def stream():
    return torch.cuda.Stream()


LENGTHS = (1, 3, 4, 5, 8, 9, 13, 16, 17, 23, 33, 36, 37)


def test_each_layer_holds_exactly_its_pages(stream):
    manager = make_manager()
    probe = Probe(manager)
    try:
        cache = manager.create_kv_cache()
        assert cache.resume(stream.cuda_stream)
        # Short prompts, partial final pages, and lengths off every CP cycle.
        for num_tokens in LENGTHS:
            assert cache.resize(num_tokens)
            probe.check(cache, num_tokens)
        for num_tokens in (21, 9, 5, 2):
            assert cache.resize(num_tokens)
            probe.check(cache, num_tokens)
        cache.suspend()
        assert cache.resume(stream.cuda_stream)
        probe.check(cache, 2)
        cache.close()
        assert [s.free for s in manager.get_storage_statistics(0)] == probe.free
        assert manager.get_strided_admitted_slots() == [0] * len(probe.free)
    finally:
        manager.shutdown()


def test_refused_growth_changes_nothing(stream):
    # A small pool: fill the strided pools' admission with other requests,
    # then a growth past it must fail as a whole.
    manager = make_manager(quota=1 << 20)
    probe = Probe(manager)
    try:
        cache = manager.create_kv_cache()
        assert cache.resume(stream.cuda_stream)
        assert cache.resize(16)
        others = []
        while True:
            other = manager.create_kv_cache()
            if not (other.resume(stream.cuda_stream) and other.resize(TOKENS_PER_BLOCK)):
                other.close()
                break
            others.append(other)
        assert others

        def snapshot():
            return (
                cache.capacity,
                [list(cache.get_aggregated_page_indices(g)) for g in probe.groups.values()],
                [s.free for s in manager.get_storage_statistics(0)],
                manager.get_strided_admitted_slots(),
            )

        before = snapshot()
        assert not cache.resize(16 + 8 * TOKENS_PER_BLOCK * 4)
        assert snapshot() == before
        for other in others:
            other.close()
        assert cache.resize(16 + 8 * TOKENS_PER_BLOCK * 4)
        cache.close()
    finally:
        manager.shutdown()


def test_ranks_of_one_layout_admit_alike(stream):
    # Two managers of equal size at different phases: the rank holding phase 0
    # holds more pages, but both admit the same requests (rank-independent charge).
    managers = [make_manager(1 << 20, {"target": f, "draft": f % 2}) for f in (0, 3)]
    caches = [[] for _ in managers]
    try:
        while True:
            decisions = []
            for manager, held in zip(managers, caches):
                cache = manager.create_kv_cache()
                ok = cache.resume(stream.cuda_stream) and cache.resize(1)
                held.append(cache)
                decisions.append(ok)
            assert len(set(decisions)) == 1, decisions
            if not decisions[0]:
                break
        stats = [Probe(m) for m in managers]
        assert managers[0].get_strided_admitted_slots() == managers[1].get_strided_admitted_slots()
        # One-page requests sit on phase 0 only; phase 3 holds none of them.
        target_pool = stats[0].pools["target"]
        used = [
            m.get_storage_statistics(0)[target_pool].total
            - m.get_storage_statistics(0)[target_pool].free
            for m in managers
        ]
        assert used[0] > 0 and used[1] == 0
    finally:
        for held in caches:
            for cache in held:
                cache.close()
        for manager in managers:
            manager.shutdown()


PRIVATE = kv.AttentionReusePolicy.PRIVATE


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(page_stride=4, page_phase=4, reuse_policy=PRIVATE),  # phase out of range
        dict(page_stride=0, reuse_policy=PRIVATE),
        dict(page_stride=4, sliding_window_size=8, reuse_policy=PRIVATE),
        dict(page_stride=4, reuse_policy=kv.AttentionReusePolicy.REQUIRED),  # shareable pages
    ],
)
def test_invalid_strided_layers_are_rejected(kwargs):
    with pytest.raises(ValueError, match="pageStride"):
        config = kv.KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)],
            layers=[
                kv.AttentionLayerConfig(
                    layer_id=kv.LayerId(0),
                    buffers=[kv.BufferConfig(role="key", size=4096)],
                    **kwargs,
                )
            ],
        )
        kv.KVCacheManager(config).shutdown()


def test_strided_layers_reject_beams(stream):
    layer = kv.AttentionLayerConfig(
        layer_id=kv.LayerId(0),
        buffers=[kv.BufferConfig(role="key", size=4096)],
        reuse_policy=kv.AttentionReusePolicy.PRIVATE,
        page_stride=2,
    )
    manager = kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[kv.GpuCacheTierConfig(quota=4 << 20)],
            layers=[layer],
        )
    )
    cache = manager.create_kv_cache()
    try:
        assert cache.resume(stream.cuda_stream)
        with pytest.raises(ValueError, match="beam search"):
            cache.beam_width = 2
    finally:
        cache.close()
        manager.shutdown()


# Pages large enough that a few requests' admission charges constrain a pool resize.
LARGE = {"target": 256 << 10, "draft": 512 << 10, "full": 768 << 10}


def page_bytes(manager, cache, name, layer_id):
    """Each held page of a layer, in order, as an int8 tensor over its pool memory."""
    layer = kv.LayerId(layer_id)
    group = manager.get_layer_group_id(layer)
    base = int(manager.get_mem_pool_base_address(layer, "key", kv.PageIndexMode.SHARED))
    stride = manager.get_page_stride(layer, "key")
    converter = manager.get_page_index_converter(layer, "key")
    pages = []
    for slot in cache.get_aggregated_page_indices(group):
        if slot == BAD:
            continue
        index = slot * converter.scale + converter.layer_offset
        pages.append(
            convert_to_torch_tensor(
                TensorWrapper(base + index * stride, DataType.INT8, [LARGE[name]])
            )
        )
    return pages


def fill_pages(manager, cache, salt):
    for layer_id, name in enumerate(LAYOUTS):
        for k, page in enumerate(page_bytes(manager, cache, name, layer_id)):
            page.fill_((salt * 37 + layer_id * 16 + k) % 127)
    torch.cuda.synchronize()


def check_pages(manager, cache, num_tokens, salt):
    num_pages = div_up(num_tokens, TOKENS_PER_BLOCK)
    for layer_id, (name, (stride, phase, _)) in enumerate(LAYOUTS.items()):
        pages = page_bytes(manager, cache, name, layer_id)
        assert len(pages) == len(owned_pages(stride, phase, num_pages)), name
        for k, page in enumerate(pages):
            assert bool((page == (salt * 37 + layer_id * 16 + k) % 127).all()), (name, k)


def test_host_tier_offload_and_pool_resize_keep_pages(stream, capfd):
    """Shrinking the GPU tier under suspended requests, as KV pool rebalancing does.

    Unstrided pages move to the host tier and come back with their contents; strided pools
    are never sized below their admission charge, a refusal every rank makes alike.
    """
    layers = [
        kv.AttentionLayerConfig(
            layer_id=kv.LayerId(layer_id),
            buffers=[kv.BufferConfig(role="key", size=LARGE[name])],
            reuse_policy=kv.AttentionReusePolicy.PRIVATE,
            page_stride=stride,
            page_phase=phase,
        )
        for layer_id, (name, (stride, phase, _)) in enumerate(LAYOUTS.items())
    ]
    gpu_quota = 64 << 20
    manager = kv.KVCacheManager(
        kv.KVCacheManagerConfig(
            tokens_per_block=TOKENS_PER_BLOCK,
            cache_tiers=[
                kv.GpuCacheTierConfig(quota=gpu_quota),
                kv.HostCacheTierConfig(quota=128 << 20),
            ],
            layers=layers,
        )
    )
    gpu, host = kv.CacheLevel(0), kv.CacheLevel(1)
    lengths = (37, 21, 13)
    caches = []
    try:
        for salt, num_tokens in enumerate(lengths):
            cache = manager.create_kv_cache()
            assert cache.resume(stream.cuda_stream) and cache.resize(num_tokens)
            fill_pages(manager, cache, salt)
            caches.append(cache)
        charged = manager.get_strided_admitted_slots()
        active = [True] * len(caches)
        outcomes = []
        for quota in range(gpu_quota - (2 << 20), 0, -(2 << 20)):
            for index, cache in enumerate(caches):
                if active[index]:
                    cache.suspend()
            resized = manager.resize(gpu, quota)
            offloaded = sum(st.total - st.free for st in manager.get_storage_statistics(host))
            outcomes.append((quota, resized, offloaded))
            # A request that does not fit stays suspended, as after a rebalance; every rank
            # decides alike, from the admission charges.
            for index, cache in enumerate(caches):
                active[index] = cache.resume(stream.cuda_stream)
                if active[index]:
                    check_pages(manager, cache, lengths[index], index)
            assert manager.get_strided_admitted_slots() == charged
            if not resized:
                break
        log = capfd.readouterr()
        assert any(resized and offloaded for _, resized, offloaded in outcomes), outcomes
        # The first refusal is the admission charge's, not a quota or min-slot limit.
        assert not outcomes[-1][1] and "admitted for layers whose pageStride is above 1" in (
            log.out + log.err
        ), outcomes
        # Restored, every request resumes with exactly its pages and contents.
        for index, cache in enumerate(caches):
            if active[index]:
                cache.suspend()
        assert manager.resize(gpu, gpu_quota)
        for index, cache in enumerate(caches):
            assert cache.resume(stream.cuda_stream)
            check_pages(manager, cache, lengths[index], index)
        while caches:
            caches.pop().close()
        assert manager.get_strided_admitted_slots() == [0] * len(charged)
    finally:
        while caches:
            caches.pop().close()
        manager.shutdown()
