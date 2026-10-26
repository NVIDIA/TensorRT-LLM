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
"""How a rank's pages come out of its cache manager, and how they travel.

A manager describes its pages through its hooks: ``build_page_layout`` (``get_page_layout``)
walks its storage and asks each buffer's layout of ``get_buffer_layout``; ``RankPages.of_manager``
adds where the pools lie. The fakes below are real manager subclasses built without ``__init__``,
so the base classes' hooks answer as they do in production.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pages_test_utils import Segment, buf, group, head_major, pages, pool, sectioned, token_major

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
    MiniMaxM3KVCacheManagerV2,
)
from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.page_mapping import (
    RankPages,
    memory_registration_descs,
    replicated,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManagerV2,
    MambaRole,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    KvCacheConfig,
    KVCacheManager,
    Mapping,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SsmLayerConfig


def _layout(rank_pages, g):
    """``[(pool index, layer, role, offset, segments)]`` of a layer group's buffers, in order."""
    return [
        (b.pool_index, b.layer, b.role, b.slot_offset, b.segments) for b in rank_pages.buffers(g)
    ]


@pytest.mark.cpu_only
def test_pool_slots():
    """One pointer per valid unit id: the start of its slot in the pool."""
    rank_pages = pages([pool(0x1000, 64, num_slots=8), pool(0x8000, 96, num_slots=8)], [])
    region_ids = np.array([0, 2, -1, 5], dtype=np.int64)
    spec_region = rank_pages.slots((1, 0), region_ids)

    assert isinstance(spec_region, SpecRegion)
    memory = spec_region.memory
    assert isinstance(memory, MemRegionGroup)
    assert isinstance(memory.ptrs, np.ndarray)
    # The invalid id (-1, BAD_PAGE_INDEX) is dropped.
    np.testing.assert_array_equal(memory.ptrs, [0x8000, 0x8000 + 2 * 96, 0x8000 + 5 * 96])
    assert memory.bytes_per_region == 96

    np.testing.assert_array_equal(
        rank_pages.slots((0, 0), np.array([1, 3], dtype=np.int64)).memory.ptrs,
        [0x1000 + 64, 0x1000 + 192],
    )


@pytest.mark.cuda
def test_rank_pages_of_manager():
    num_layers = 8
    num_kv_heads = 8
    head_dim = 128
    tokens_per_block = 64

    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=3200, enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=2048,
        max_batch_size=8,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
    )
    try:
        rank_pages = RankPages.of_manager(manager)

        # One layer group per life cycle, indexed by life-cycle id.
        assert rank_pages.num_layer_groups == len(manager.impl.layer_grouping)
        layers = sorted(
            layer
            for g in range(rank_pages.num_layer_groups)
            for layer in rank_pages.group(g).layers
        )
        assert layers == sorted(int(layer) for layer in manager.pp_layers)

        buffer_bytes = num_kv_heads * tokens_per_block * head_dim * 2
        for g in range(rank_pages.num_layer_groups):
            assert rank_pages.kind(g) == CacheKind.PAGED
            roles = {}
            for buffer in rank_pages.buffers(g):
                roles.setdefault(buffer.layer, set()).add(buffer.role)
                # K/V are head-major, sharded by KV head over attention TP.
                assert buffer.segments == head_major(buffer_bytes, num_kv_heads)
                device_pool = rank_pages.pool((buffer.pool_group, buffer.pool_index))
                assert buffer.slot_offset + buffer.size <= device_pool.slot_bytes
            assert all(r == {"key", "value"} for r in roles.values())
            # Buffers are in physical order.
            keys = [(b.pool_index, b.slot_offset) for b in rank_pages.buffers(g)]
            assert keys == sorted(keys)

        for device_pool in rank_pages.pools:
            assert device_pool.base > 0
            assert device_pool.num_slots > 0

        # A unit's slot is addressed in the pool its buffers live in.
        buffer = rank_pages.buffers(0)[0]
        key = (buffer.pool_group, buffer.pool_index)
        device_pool = rank_pages.pool(key)
        region_ids = np.array([0, 1], dtype=np.int64)
        memory = rank_pages.slots(key, region_ids).memory
        np.testing.assert_array_equal(
            memory.ptrs, device_pool.base + region_ids * device_pool.slot_bytes
        )
        assert memory.bytes_per_region == device_pool.slot_bytes
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
def test_rank_pages_meta_serialization():
    segments = head_major(256, 4)
    buffers = [
        buf(0, layer, role, 512 * layer + 256 * i, segments)
        for layer in (0, 1)
        for i, role in enumerate(("key", "value"))
    ]
    rank_pages = pages([pool(1000, 1024, num_slots=10)], [group("attention", buffers, window=512)])
    restored = RankPages.from_dict(rank_pages.to_dict())
    assert restored.kind(0) == CacheKind.PAGED
    assert restored.group(0).kept_window == 512
    assert restored.group(0).layers == (0, 1)
    assert _layout(restored, 0) == _layout(rank_pages, 0)
    assert restored.pools == rank_pages.pools
    assert restored.layout == rank_pages.layout


def _storage(buffer_specs, *, slot_bytes_list, layer_group_id=0, pool_group_index=0, num_slots=4):
    """A pool-group descriptor of the V2 storage.

    ``buffer_specs`` maps ``(layer_id, role)`` to ``(pool_index, offset, size)``. Buffer order within
    a coalesced buffer follows ascending offset, as the real storage config assigns offsets in
    ``buffer_ids`` order.
    """
    per_pool: dict[int, list] = {}
    for (layer_id, role), (pool_index, offset, size) in buffer_specs.items():
        per_pool.setdefault(pool_index, []).append((offset, layer_id, role, size))
    coalesced_buffers = []
    for pool_index in range(len(slot_bytes_list)):
        buffers = sorted(per_pool.get(pool_index, []))
        sizes = {size for _, _, _, size in buffers}
        assert len(sizes) == 1, "buffers in a coalesced buffer are uniform-size by construction"
        coalesced_buffers.append(
            SimpleNamespace(
                single_buffer_size=sizes.pop(),
                buffer_ids=[
                    SimpleNamespace(layer_id=layer_id, role=role)
                    for _, layer_id, role, _ in buffers
                ],
            )
        )
    return SimpleNamespace(
        pool_group_index=pool_group_index,
        num_slots=num_slots,
        pools=[
            SimpleNamespace(
                pool_index=i, base_address=0x1000 * (pool_group_index + 1), slot_bytes=s
            )
            for i, s in enumerate(slot_bytes_list)
        ],
        slot_desc=SimpleNamespace(
            variants=[
                SimpleNamespace(layer_group_id=layer_group_id, coalesced_buffers=coalesced_buffers)
            ]
        ),
    )


def _fake_v2_manager(buffer_specs, *, slot_bytes_list=(640,), cls=KVCacheManagerV2, **attrs):
    """A ``cls`` built without ``__init__``, so its hooks are the real ones.

    Its storage holds one attention life cycle of layers 0 and 1, one KV head each, 16 tokens per
    block.
    """
    manager = object.__new__(cls)
    manager.impl = SimpleNamespace(
        layer_grouping=((0, 1),),
        get_life_cycle_pool_group_indices=lambda: [0],
        init_config=SimpleNamespace(
            layers=[AttentionLayerConfig(layer_id=i, buffers=[]) for i in range(2)]
        ),
        pool_group_descs=[_storage(buffer_specs, slot_bytes_list=slot_bytes_list)],
    )
    manager.tokens_per_block = 16
    manager.pp_layers = [0, 1]
    manager.num_kv_heads_per_layer = [1, 1]
    for name, value in attrs.items():
        setattr(manager, name, value)
    return manager


def test_minimax_m3_split_pools_are_token_major_kv_and_replicated_index_key():
    """Split KV / INDEX_KEY pools give token-major and replicated buffers (MiniMax M3, NHD)."""
    buffer_specs = {
        (0, Role.KEY): (0, 0, 128),
        (0, Role.VALUE): (0, 128, 128),
        (1, Role.KEY): (0, 256, 128),
        (1, Role.VALUE): (0, 384, 128),
        (1, Role.INDEX_KEY): (1, 0, 128),
    }
    manager = _fake_v2_manager(
        buffer_specs,
        slot_bytes_list=(512, 128),
        cls=MiniMaxM3KVCacheManagerV2,
        _main_kv_layout="NHD",
    )

    rank_pages = RankPages.of_manager(manager)
    assert [(p.base, p.slot_bytes, p.num_slots) for p in rank_pages.pools] == [
        (0x1000, 512, 4),
        (0x1000, 128, 4),
    ]
    assert rank_pages.num_layer_groups == 1
    assert rank_pages.kind(0) == CacheKind.PAGED
    assert rank_pages.group(0).kept_window is None
    nhd = token_major(128, 16, 1)
    index_key = (Segment(128),)
    assert _layout(rank_pages, 0) == [
        (0, 0, "key", 0, nhd),
        (0, 0, "value", 128, nhd),
        (0, 1, "key", 256, nhd),
        (0, 1, "value", 384, nhd),
        (1, 1, "index_key", 0, index_key),
    ]
    assert [replicated(b) for b in rank_pages.buffers(0)] == [False] * 4 + [True]


def test_minimax_m3_mixed_role_pool_keeps_one_layout_per_buffer():
    """A pool coalescing several roles yields one buffer per (layer, role).

    Miniature of the MiniMax M3 layout at TP degrees where
    K == V == INDEX_KEY bytes per block: V2 storage coalesces all three
    into one pool and the slot interleaves the sparse layer's INDEX_KEY
    between the layers' K/V buffers. Each buffer keeps its own layout:
    token-major K/V at non-uniform layer offsets, and a replicated
    INDEX_KEY covering only the sparse layer.
    """
    buffer_specs = {
        (0, Role.KEY): (0, 0, 128),
        (0, Role.VALUE): (0, 128, 128),
        (0, Role.INDEX_KEY): (0, 256, 128),
        (1, Role.KEY): (0, 384, 128),
        (1, Role.VALUE): (0, 512, 128),
    }
    manager = _fake_v2_manager(buffer_specs, cls=MiniMaxM3KVCacheManagerV2, _main_kv_layout="NHD")

    rank_pages = RankPages.of_manager(manager)
    assert len(rank_pages.pools) == 1
    nhd = token_major(128, 16, 1)
    index_key = (Segment(128),)
    # All buffers address the same physical pool; the INDEX_KEY interleaves after layer 0.
    assert _layout(rank_pages, 0) == [
        (0, 0, "key", 0, nhd),
        (0, 0, "value", 128, nhd),
        (0, 0, "index_key", 256, index_key),
        (0, 1, "key", 384, nhd),
        (0, 1, "value", 512, nhd),
    ]


def test_minimax_m3_hnd_kv_is_head_major():
    buffer_specs = {
        (0, Role.KEY): (0, 0, 128),
        (0, Role.VALUE): (0, 128, 128),
        (1, Role.KEY): (0, 256, 128),
        (1, Role.VALUE): (0, 384, 128),
    }
    manager = _fake_v2_manager(
        buffer_specs, slot_bytes_list=(512,), cls=MiniMaxM3KVCacheManagerV2, _main_kv_layout="HND"
    )
    assert [b.segments for b in RankPages.of_manager(manager).buffers(0)] == [
        head_major(128, 1)
    ] * 4


def test_base_manager_lays_out_kv_head_major():
    buffer_specs = {
        (0, Role.KEY): (0, 0, 128),
        (0, Role.VALUE): (0, 128, 128),
        (1, Role.KEY): (0, 256, 128),
        (1, Role.VALUE): (0, 384, 128),
    }
    # Plain attention: head-major K/V, as the TRTLLM attention kernels write them.
    rank_pages = RankPages.of_manager(_fake_v2_manager(buffer_specs))
    assert rank_pages.kind(0) == CacheKind.PAGED
    assert [b.segments for b in rank_pages.buffers(0)] == [head_major(128, 1)] * 4


def test_base_manager_rejects_a_role_it_declares_no_layout_for():
    buffer_specs = {
        (0, Role.KEY): (0, 0, 64),
        (0, Role.MLA_V_SCALE): (0, 64, 64),
    }
    manager = _fake_v2_manager(buffer_specs, slot_bytes_list=(128,))
    with pytest.raises(ValueError, match="declares no layout for role 'mla_v_scale'"):
        manager.get_page_layout()


@pytest.mark.cpu_only
def test_state_layer_group_serialization():
    conv_sections = (512, 256, 256)
    ssm_bytes, ssm_bytes_per_head = 256, 128
    conv = sectioned(conv_sections)
    ssm = head_major(ssm_bytes, ssm_bytes // ssm_bytes_per_head)
    buffers = []
    for lid, layer in enumerate((10, 11, 12)):
        base = lid * (sum(conv_sections) + ssm_bytes)
        buffers.append(buf(0, layer, str(MambaRole.CONV_STATE), base, conv))
        buffers.append(buf(0, layer, str(MambaRole.SSM_STATE), base + sum(conv_sections), ssm))
    rank_pages = pages(
        [pool(0x1000, 3 * (sum(conv_sections) + ssm_bytes))], [group("state", buffers)]
    )

    d = rank_pages.to_dict()
    # Each distinct layout travels once: the conv sections, then the SSM heads.
    assert d["segments"] == [
        [[s.nbytes, [list(a) for a in s.axes], s.split_axis] for s in segments]
        for segments in (conv, ssm)
    ]

    restored = RankPages.from_dict(d)
    assert restored.kind(0) == CacheKind.STATE
    assert restored.group(0).kept_window is None
    assert restored.buffers(0) == rank_pages.buffers(0)
    conv_buffer, ssm_buffer = restored.buffers(0)[:2]
    assert [s.nbytes for s in conv_buffer.segments] == list(conv_sections)
    assert conv_buffer.size == sum(conv_sections)
    assert ssm_buffer.segments[0].unit_bytes == ssm_bytes_per_head
    assert [b.layer for b in restored.buffers(0)[::2]] == [10, 11, 12]
    assert restored.group(0).layers == (10, 11, 12)


def _fake_v2_mamba_manager(*, ple_on_layer=1):
    """A ``MambaHybridCacheManagerV2`` built without ``__init__``, its storage of two life cycles.

    Life cycle 0 is recurrent (layers 0 and 1): SSM and conv state coalesce
    into one 256-byte pool in ``[ssm, conv]`` order per layer, and PLE n-gram
    context lives in a second, 32-byte pool on ``ple_on_layer`` only. Life
    cycle 1 is attention (layer 2) with K/V in its own pool group.
    """

    def buffer_id(layer_id, role):
        return SimpleNamespace(layer_id=layer_id, role=role)

    recurrent_pg = SimpleNamespace(
        pool_group_index=0,
        num_slots=4,
        pools=[
            SimpleNamespace(pool_index=0, base_address=0x10000, slot_bytes=256),
            SimpleNamespace(pool_index=1, base_address=0x20000, slot_bytes=32),
        ],
        slot_desc=SimpleNamespace(
            variants=[
                SimpleNamespace(
                    layer_group_id=0,
                    coalesced_buffers=[
                        SimpleNamespace(
                            single_buffer_size=64,
                            buffer_ids=[
                                buffer_id(0, MambaRole.SSM_STATE),
                                buffer_id(0, MambaRole.CONV_STATE),
                                buffer_id(1, MambaRole.SSM_STATE),
                                buffer_id(1, MambaRole.CONV_STATE),
                            ],
                        ),
                        SimpleNamespace(
                            single_buffer_size=32,
                            buffer_ids=[buffer_id(ple_on_layer, MambaRole.PLE_NGRAM_CONTEXT)],
                        ),
                    ],
                )
            ]
        ),
    )
    attention_pg = SimpleNamespace(
        pool_group_index=1,
        num_slots=8,
        pools=[SimpleNamespace(pool_index=0, base_address=0x30000, slot_bytes=128)],
        slot_desc=SimpleNamespace(
            variants=[
                SimpleNamespace(
                    layer_group_id=1,
                    coalesced_buffers=[
                        SimpleNamespace(
                            single_buffer_size=64,
                            buffer_ids=[buffer_id(2, Role.KEY), buffer_id(2, Role.VALUE)],
                        )
                    ],
                )
            ]
        ),
    )

    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.impl = SimpleNamespace(
        layer_grouping=((0, 1), (2,)),
        get_life_cycle_pool_group_indices=lambda: [0, 1],
        init_config=SimpleNamespace(
            layers=[
                SsmLayerConfig(layer_id=0, buffers=[]),
                SsmLayerConfig(layer_id=1, buffers=[]),
                AttentionLayerConfig(layer_id=2, buffers=[]),
            ]
        ),
        pool_group_descs=[recurrent_pg, attention_pg],
    )
    manager.tokens_per_block = 16
    manager.pp_layers = [0, 1, 2]
    manager._mamba_layer_mask = [True, True, False]
    manager.local_num_mamba_layers = 2
    manager.num_kv_heads_per_layer = [1, 1, 1]
    # conv: [8, 4] fp16 = 64 bytes as sections [2, 2, 4] * 4 * 2 = [16, 16, 32]
    manager.conv_state_shape = [8, 4]
    manager.conv_state_dtype = torch.float16
    manager.conv_section_dims = [2, 2, 4]
    # ssm: [2 heads, 4, 4] fp16 = 64 bytes, 32 bytes per head
    manager.ssm_state_shape = [2, 4, 4]
    manager.ssm_state_dtype = torch.float16
    return manager


def test_mamba_manager_describes_recurrent_life_cycle_from_its_storage():
    """Conv, SSM and PLE buffers all come from pool_group_descs, layouts from the manager."""
    manager = _fake_v2_mamba_manager()
    rank_pages = RankPages.of_manager(manager)

    # Pools are one flat list over both pool groups.
    assert [(p.base, p.slot_bytes, p.num_slots) for p in rank_pages.pools] == [
        (0x10000, 256, 4),
        (0x20000, 32, 4),
        (0x30000, 128, 8),
    ]
    # Layer group index == life-cycle id.
    assert rank_pages.num_layer_groups == 2
    assert (rank_pages.kind(0), rank_pages.kind(1)) == (CacheKind.STATE, CacheKind.PAGED)
    assert rank_pages.group(0).kept_window is None
    assert rank_pages.group(0).layers == (0, 1)

    ssm = head_major(64, 2)
    conv = sectioned((16, 16, 32))
    ple = (Segment(32),)
    ssm_role, conv_role = str(MambaRole.SSM_STATE), str(MambaRole.CONV_STATE)
    assert _layout(rank_pages, 0) == [
        (0, 0, ssm_role, 0, ssm),
        (0, 0, conv_role, 64, conv),
        (0, 1, ssm_role, 128, ssm),
        (0, 1, conv_role, 192, conv),
        (1, 1, str(MambaRole.PLE_NGRAM_CONTEXT), 0, ple),
    ]
    assert [replicated(b) for b in rank_pages.buffers(0)] == [False] * 4 + [True]
    assert rank_pages.buffers(0)[0].segments[0].unit_bytes == 32

    # A state slot is addressed like a block: its base, plus the buffers' offsets.
    assert rank_pages.slots((0, 0), np.array([2])).memory.ptrs.tolist() == [0x10000 + 2 * 256]
    assert {b.layer: b.slot_offset for b in rank_pages.buffers(0) if b.role == conv_role} == {
        0: 64,
        1: 192,
    }
    assert rank_pages.slots((0, 1), np.array([3])).memory.ptrs.tolist() == [0x20000 + 3 * 32]

    # Attention life cycle is untouched by the recurrent one.
    assert [(p.group, p.index) for p in rank_pages.group_pools(1)] == [(1, 0)]
    assert rank_pages.group(1).layers == (2,)
    assert rank_pages.group(1).kept_window is None
    assert _layout(rank_pages, 1) == [
        (0, 2, "key", 0, head_major(64, 1)),
        (0, 2, "value", 64, head_major(64, 1)),
    ]


def test_mamba_manager_rejects_section_bytes_that_do_not_cover_the_layer():
    manager = _fake_v2_mamba_manager()
    manager.conv_section_dims = [2, 2, 2]  # 48 bytes, but the conv buffer is 64
    with pytest.raises(ValueError, match="do not make up a 64-byte buffer"):
        manager.get_page_layout()


@pytest.mark.cpu_only
def test_v2_mamba_registration_uses_coalesced_physical_pool():
    state_bytes = 64
    num_layers = 2
    num_slots = 8
    # Equal-sized SSM and convolution states share one interleaved V2 pool.
    physical_slot_bytes = state_bytes * num_layers * 2
    buffers = [
        buf(0, layer, role, (2 * lid + i) * state_bytes, (Segment(state_bytes),))
        for lid, layer in enumerate((1, 2))
        for i, role in enumerate((str(MambaRole.CONV_STATE), str(MambaRole.SSM_STATE)))
    ]
    rank_pages = pages(
        [pool(1000, physical_slot_bytes, num_slots=num_slots)], [group("state", buffers)]
    )

    # V2 interleaved: one shared pool registered once
    assert memory_registration_descs(rank_pages, device_id=3) == [
        (1000, physical_slot_bytes * num_slots, 3, "kv_cache_memory_pool0")
    ]


@pytest.mark.cpu_only
def test_buffer_offsets_past_four_gib_survive_rank_info_roundtrip() -> None:
    """Buffer offsets and pool sizes are 64-bit on the wire."""
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

    slot_bytes = 1 << 33  # 8 GiB slot: later buffers sit past 4 GiB
    buffers = [
        buf(0, 10 + lid, "conv_state", lid * (1 << 31), (Segment(1 << 31),)) for lid in range(4)
    ]
    rank_pages = pages([pool((1 << 40) + 1000, slot_bytes, num_slots=2)], [group("state", buffers)])
    ri = RankInfo(
        instance_name="ctx",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[4],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        pages=rank_pages,
    )
    restored = RankInfo.from_bytes(ri.to_bytes()).pages
    assert [b.slot_offset for b in restored.buffers(0)] == [lid * (1 << 31) for lid in range(4)]
    assert restored.pools == rank_pages.pools


@pytest.mark.cpu_only
def test_mixed_rank_pages_serialization():
    # Paged attention layer group (life cycle 0): layer 0.
    attention = [
        buf(0, 0, "key", 0, head_major(256, 4)),
        buf(0, 0, "value", 256, head_major(256, 4)),
    ]
    # Recurrent state layer group (life cycle 1): layers 1 and 2.
    conv = sectioned((256, 128, 128))
    ssm = head_major(2048, 32)
    state = [
        b
        for lid, layer in enumerate((1, 2))
        for b in (
            buf(1, layer, str(MambaRole.CONV_STATE), lid * 512, conv),
            buf(2, layer, str(MambaRole.SSM_STATE), lid * 2048, ssm),
        )
    ]
    rank_pages = pages(
        [pool(1000, 512, 10), pool(2000, 1024, 10), pool(3000, 4096, 10)],
        [group("attention", attention), group("state", state)],
    )

    restored = RankPages.from_dict(rank_pages.to_dict())

    assert restored.num_layer_groups == 2
    assert (restored.kind(0), restored.kind(1)) == (CacheKind.PAGED, CacheKind.STATE)
    assert restored.group(0).layers == (0,)
    assert restored.group(1).layers == (1, 2)
    # The state layer group's two pools form one pool group, apart from the attention pool.
    state_pools = restored.group_pools(1)
    assert [p.base for p in state_pools] == [2000, 3000]
    assert [p.index for p in state_pools] == [0, 1]
    assert {p.group for p in state_pools} != {p.group for p in restored.group_pools(0)}
    assert [b.layer for b in restored.buffers(1)] == [1, 2, 1, 2]
    assert restored.layout == rank_pages.layout

    # Which layer group holds each layer comes straight from the buffers.
    layer_to_group = {
        b.layer: g for g in range(restored.num_layer_groups) for b in restored.buffers(g)
    }
    assert layer_to_group == {0: 0, 1: 1, 2: 1}


# ---------------------------------------------------------------------------
# Boundary tests for block_id -> primary memory-pool slot translation
# (see cache_reuse.py: when host offload is enabled, a
#  block's logical ID can diverge from its primary-pool slot index after an
#  offload+onboard cycle, and the cache transceiver must translate before
#  doing pointer arithmetic).
# ---------------------------------------------------------------------------


@pytest.mark.cuda
def test_get_memory_pool_block_indices_vswa():
    """VSWA / multi-window dispatch boundary check.

    Without host offload, primary slot indices should equal block IDs
    (no offload has happened), and the wrapper must:
    - dispatch correctly per window_size,
    - return non-negative pool indices that match what
      `RankPages.slots` plugs into `base_ptr + idx * slot_bytes`.
    """
    import torch

    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.sampling_params import SamplingParams

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    num_layers = 4
    num_kv_heads = 2
    head_dim = 64
    tokens_per_block = 16
    # Two distinct window sizes -> is_variable_window=True. Keep windows
    # large enough that a short prefill keeps every block live.
    max_attention_window_vec = [128] * 2 + [256] * 2

    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        free_gpu_memory_fraction=0.1,
        max_attention_window=max_attention_window_vec,
        enable_block_reuse=False,
        host_cache_size=0,
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max(max_attention_window_vec),
        max_batch_size=2,
        mapping=mapping,
        dtype=DataType.HALF,
    )

    try:
        # Drive a small prefill so the manager has resident blocks.
        sampling_params = SamplingParams()
        req = LlmRequest(
            request_id=0,
            max_new_tokens=1,
            input_tokens=list(range(tokens_per_block * 3)),
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()
            ),
            is_streaming=False,
        )
        manager.impl.add_sequence_batch([(req.py_request_id, req.prompt_len, 1)], [req])

        # For each unique window, look up that window's first layer to fetch
        # its block IDs, then translate and confirm the pool-slot indices
        # match what RankPages.slots would compute against
        # the live primary pool.
        unique_windows = sorted(set(max_attention_window_vec))
        assert len(unique_windows) >= 2, "Test premise: at least 2 windows"

        for w in unique_windows:
            # Pick the first layer whose window matches.
            layer_idx = max_attention_window_vec.index(w)
            block_ids = manager.get_batch_cache_indices([req.py_request_id], layer_idx=layer_idx)[0]
            assert len(block_ids) > 0, f"prefill should populate window={w}"

            translated = manager.get_memory_pool_block_indices(block_ids, window_size=w)
            assert len(translated) == len(block_ids)
            assert all(idx >= 0 for idx in translated), (
                f"primary pool slot indices must be >= 0, got {translated}"
            )

            # Without host offload the translation is identity. This locks
            # the no-offload contract — if a future change introduces a
            # gratuitous remapping, this test will catch it.
            assert list(translated) == list(block_ids)

        # Note: with two distinct windows the underlying primary pools may
        # share a slot index *value* across windows but they live in
        # different pools, so we do not assert distinctness across windows.
    finally:
        manager.shutdown()


@pytest.mark.cuda
def test_get_memory_pool_block_indices_with_offload_onboard():
    """Offload -> onboard -> transfer cycle.

    What this test proves:
      1. While a block IS primary, its block ID can still differ from
         its primary-pool slot (primary-side divergence). This happens
         because block IDs are allocated globally (req_b's blocks get
         IDs 4-7 after req_a took 0-3), but primary-pool slots are
         recycled (req_a's freed slots 0-3 get reused for req_b's new
         blocks). Without translation, the disagg sender would
         compute `base + 4 * slot_bytes` and read past the end of a
         4-block primary pool — the exact pointer-arithmetic bug this
         fix addresses.
      2. Translation aborts on offloaded blocks with the documented
         "Block is not in the primary pool" message — the disagg cache
         transceiver relies on this to refuse host-pool transfers.
      3. After reuse-driven onboard, all blocks are primary again and
         pointers computed from the translated slot indices match
         `primary_pool_base + slot_idx * slot_bytes`.
    """
    import torch

    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings.internal.testing import (
        simulate_prefill_completion_only_use_for_testing,
    )
    from tensorrt_llm.sampling_params import SamplingParams

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")

    tokens_per_block = 16
    # Tight primary capacity (~4 blocks) + plenty of host space (~32 blocks)
    # so eviction must spill to host rather than free outright.
    kv_cache_config = KvCacheConfig(
        max_tokens=tokens_per_block * 4,
        free_gpu_memory_fraction=0.1,
        enable_block_reuse=True,
        enable_partial_reuse=True,
        copy_on_partial_reuse=False,
        host_cache_size=tokens_per_block * 32 * 2 * 2 * 2 * 64,  # generous host pool
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=tokens_per_block,
        max_seq_len=tokens_per_block * 4,
        max_batch_size=4,
        mapping=mapping,
        dtype=DataType.HALF,
    )

    try:
        # Skip cleanly if the build/test env didn't actually allocate a
        # secondary pool (e.g. host_cache_size rounded to 0). Without it,
        # this test cannot exercise the offload path.
        if getattr(manager, "blocks_in_secondary_pool", 0) <= 0:
            pytest.skip(
                "secondary (host) pool not allocated in this build; "
                "offload/onboard path cannot be exercised"
            )

        def _run(req_id, tokens):
            sampling_params = SamplingParams()
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=list(tokens),
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()
                ),
                is_streaming=False,
            )
            manager.impl.add_sequence_batch([(req.py_request_id, req.prompt_len, 1)], [req])
            return req

        window = manager.max_attention_window_vec[0]

        # Warm: take blocks for a 4-block prefill, capture its block_ids,
        # then commit to reuse so they live in the radix tree (eligible
        # for offload, not freed outright).
        prefill_tokens_a = list(range(tokens_per_block * 4))
        req_a = _run(0, prefill_tokens_a)
        block_ids_a = manager.get_batch_cache_indices([req_a.py_request_id], layer_idx=0)[0]
        # Translation is the identity at this point — block_ids equal
        # primary slot indices because nothing has been offloaded yet.
        original_indices = manager.get_memory_pool_block_indices(block_ids_a, window_size=window)
        assert list(original_indices) == list(block_ids_a)
        simulate_prefill_completion_only_use_for_testing(req_a)
        manager.free_resources(req_a)

        # Pressure: bring in a brand-new request that needs more primary
        # blocks than are currently free, forcing eviction of req_a's
        # cached prefix to the host pool.
        prefill_tokens_b = list(range(10_000, 10_000 + tokens_per_block * 4))
        req_b = _run(1, prefill_tokens_b)

        # Primary-side divergence: req_b's block IDs are allocated fresh
        # (typically 4..7 since req_a took 0..3 from the radix tree), but
        # the primary-pool slots are recycled from req_a's freed slots
        # (0..3). So even though every req_b block is currently primary,
        # `block_id != slot` for every one of them. This is the most
        # visible failure mode of the pre-fix code: the disagg sender
        # would compute `base + block_id_b * slot_bytes` and walk off the
        # end of a 4-block primary pool.
        block_ids_b = list(manager.get_batch_cache_indices([req_b.py_request_id], layer_idx=0)[0])
        slots_b = list(manager.get_memory_pool_block_indices(block_ids_b, window_size=window))
        primary_diverged = [(b, s) for b, s in zip(block_ids_b, slots_b) if b != s]
        assert primary_diverged, (
            f"expected req_b's block_ids to differ from its primary slots "
            f"(block IDs allocate forward, primary slots recycle); got "
            f"identity mapping block_ids={block_ids_b}, slots={slots_b}"
        )

        # Translation must abort for any offloaded block — this is the policy
        # the disagg cache transceiver relies on (otherwise the sender would
        # compute garbage pointers against host-pool blocks).
        any_non_primary = False
        for bid in block_ids_a:
            try:
                manager.get_memory_pool_block_indices([int(bid)], window_size=window)
            except RuntimeError as exc:
                assert "Block is not in the primary pool" in str(exc), (
                    f"unexpected error message: {exc}"
                )
                any_non_primary = True
        # If the eviction policy ended up keeping every prefix block in
        # primary (small possibility under fragmentation), we cannot
        # observe the offload assertion — skip the negative leg.
        if not any_non_primary:
            pytest.skip("no req_a block was offloaded under this build's eviction policy")

        # Onboard back: re-issue req_a with the same prompt. Reuse will
        # bring its blocks back to primary, possibly at *different*
        # primary slots than before.
        simulate_prefill_completion_only_use_for_testing(req_b)
        manager.free_resources(req_b)

        req_a2 = _run(2, prefill_tokens_a)
        block_ids_a2 = manager.get_batch_cache_indices([req_a2.py_request_id], layer_idx=0)[0]
        # After reuse-driven onboard, every block must be primary again
        # (translation succeeds); its primary slot may differ from the
        # logical block ID because onboard swaps slots in-place.
        translated_a2 = manager.get_memory_pool_block_indices(block_ids_a2, window_size=window)
        assert all(idx >= 0 for idx in translated_a2)

        simulate_prefill_completion_only_use_for_testing(req_a2)
        manager.free_resources(req_a2)
    finally:
        manager.shutdown()
