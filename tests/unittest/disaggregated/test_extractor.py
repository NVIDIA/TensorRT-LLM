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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.cache_specs import (
    BufferDecl,
    LayerGroup,
    PageTable,
    Pool,
    Region,
    build_page_table,
    memory_registration_descs,
    spec_named,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
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


def _hnd(nbytes, heads):
    return (BufferDecl(nbytes, (("head", heads),), "head"),)


def _nhd(nbytes, heads, tokens_per_block):
    return (BufferDecl(nbytes, (("token", tokens_per_block), ("head", heads)), "head"),)


def _layout(layer_group):
    """``[(pool, layer, role, offset, buffers)]`` of a layer group's regions, in order."""
    return [(r.pool, r.layer, r.role, r.offset, r.buffers) for r in layer_group.regions]


@pytest.mark.cpu_only
def test_extract():
    """One pointer per valid unit id: the start of its slot in the requested pool."""
    page_table = PageTable(
        pools=[
            Pool(base_address=0x1000, slot_bytes=64, num_slots=8),
            # A pool whose slots are padded: the stride, not the payload, addresses a slot.
            Pool(base_address=0x8000, slot_bytes=96, num_slots=8, slot_stride_bytes=128),
        ]
    )
    extractor = KVRegionExtractorV1(page_table)
    assert extractor.page_table is page_table

    region_ids = np.array([0, 2, -1, 5], dtype=np.int64)
    spec_region = extractor.extract(region_ids, pool_idx=1)

    assert isinstance(spec_region, SpecRegion)
    memory = spec_region.memory
    assert isinstance(memory, MemRegionGroup)
    assert isinstance(memory.ptrs, np.ndarray)
    # The invalid id (-1, BAD_PAGE_INDEX) is dropped.
    np.testing.assert_array_equal(memory.ptrs, [0x8000, 0x8000 + 2 * 128, 0x8000 + 5 * 128])
    assert memory.bytes_per_region == 96

    np.testing.assert_array_equal(
        extractor.extract(np.array([1, 3], dtype=np.int64)).memory.ptrs, [0x1000 + 64, 0x1000 + 192]
    )


@pytest.mark.cuda
def test_build_page_table():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

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
        page_table = build_page_table(manager)

        # One layer group per life cycle, indexed by life-cycle id.
        assert len(page_table.layer_groups) == len(manager.impl.layer_grouping)
        layers = sorted(
            layer for layer_group in page_table.layer_groups for layer in layer_group.layers
        )
        assert layers == sorted(int(layer) for layer in manager.pp_layers)

        buffer_bytes = num_kv_heads * tokens_per_block * head_dim * 2
        for layer_group in page_table.layer_groups:
            assert layer_group.spec == "attention"
            assert layer_group.kind == CacheKind.PAGED
            roles = {}
            for region in layer_group.regions:
                roles.setdefault(region.layer, set()).add(region.role)
                # K/V are head-major, sharded by KV head over attention TP.
                assert region.buffers == _hnd(buffer_bytes, num_kv_heads)
                pool = page_table.pools[region.pool]
                assert 0 <= region.offset and region.offset + region.nbytes <= pool.slot_bytes
            assert all(r == {"key", "value"} for r in roles.values())
            # Regions are in physical order.
            assert [(r.pool, r.offset) for r in layer_group.regions] == sorted(
                (r.pool, r.offset) for r in layer_group.regions
            )

        for pool in page_table.pools:
            assert pool.base_address > 0
            assert pool.num_slots > 0

        # The extractor addresses a unit's slot in the pool its regions live in.
        extractor = KVRegionExtractorV1(manager)
        region = page_table.layer_groups[0].regions[0]
        pool = page_table.pools[region.pool]
        region_ids = np.array([0, 1], dtype=np.int64)
        memory = extractor.extract(region_ids, region.pool).memory
        np.testing.assert_array_equal(
            memory.ptrs, pool.base_address + region_ids * pool.slot_stride_bytes
        )
        assert memory.bytes_per_region == pool.slot_bytes
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
def test_build_page_table_rejects_v1_manager():
    """The Python transceiver describes KVCacheManagerV2 caches only."""
    manager = object.__new__(KVCacheManager)
    manager.impl = SimpleNamespace()  # a V1 impl has no V2 layer grouping
    with pytest.raises(ValueError, match="is not a KVCacheManagerV2"):
        build_page_table(manager)
    with pytest.raises(ValueError, match="is not a KVCacheManagerV2"):
        KVRegionExtractorV1(manager)


@pytest.mark.cpu_only
def test_page_table_meta_serialization():
    kv_pool = Pool(base_address=1000, slot_bytes=1024, num_slots=10)
    buffers = _hnd(256, 4)
    regions = [
        Region(0, layer, role, 512 * layer + 256 * i, buffers, local_layer=layer)
        for layer in (0, 1)
        for i, role in enumerate(("key", "value"))
    ]
    page_table = PageTable(
        pools=[kv_pool],
        layer_groups=[
            LayerGroup(
                spec="attention",
                kind=CacheKind.PAGED,
                regions=regions,
                params={"window": 512},
            )
        ],
    )
    restored = PageTable.from_dict(page_table.to_dict())
    layer_group = restored.layer_groups[0]
    assert layer_group.spec == "attention"
    assert layer_group.kind == CacheKind.PAGED
    assert layer_group.params["window"] == 512
    assert layer_group.layers == [0, 1]
    assert layer_group.regions == regions
    assert restored.pools[0].to_dict() == kv_pool.to_dict()


def _make_fake_v2_manager(attrs, *, num_pools=1, slot_bytes_list=(640,), main_kv_layout=None):
    """Build a minimal duck-typed KVCacheManagerV2 for ``build_page_table``.

    ``main_kv_layout`` gives it MiniMax M3's ``_main_kv_layout_name`` (describe it with the
    ``minimax_m3`` spec to read it).

    ``attrs`` keeps the storage-layer shape ``{(layer_id, role):
    (life_cycle_id, pool_index, offset, size)}``; the fake synthesizes the
    public ``pool_group_descs`` view from it. Buffer order within a
    coalesced buffer follows ascending offset, mirroring how the real
    storage config assigns offsets in ``buffer_ids`` order.
    """
    base = 0x1000

    per_pool: dict[int, list] = {}
    for (layer_id, role), attr in attrs.items():
        per_pool.setdefault(int(attr.pool_index), []).append(
            (int(attr.offset), layer_id, role, int(attr.size))
        )

    coalesced_buffers = []
    for pool_idx in range(num_pools):
        buffers = sorted(per_pool.get(pool_idx, []))
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

    pg_desc = SimpleNamespace(
        pool_group_index=0,
        num_slots=4,
        pools=[
            SimpleNamespace(pool_index=pi, base_address=base, slot_bytes=sb)
            for pi, sb in enumerate(slot_bytes_list)
        ],
        slot_desc=SimpleNamespace(
            variants=[SimpleNamespace(layer_group_id=0, coalesced_buffers=coalesced_buffers)]
        ),
    )

    impl = SimpleNamespace(
        layer_grouping=((0, 1),),
        init_config=SimpleNamespace(
            tokens_per_block=16,
            layers=[SimpleNamespace(window_size=None), SimpleNamespace(window_size=None)],
        ),
        pool_group_descs=[pg_desc],
    )
    manager = SimpleNamespace(impl=impl, pp_layers=[0, 1], num_kv_heads_per_layer=[1, 1])
    if main_kv_layout is not None:
        manager._main_kv_layout_name = lambda: main_kv_layout
    return manager


def test_v2_builder_stamps_homogeneous_pool_kinds():
    """Split KV / INDEX_KEY pools get NHD and replicated regions (MiniMax M3, token-major)."""
    attrs = {
        (0, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=0, size=128),
        (0, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=128, size=128),
        (1, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=256, size=128),
        (1, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=384, size=128),
        (1, Role.INDEX_KEY): SimpleNamespace(life_cycle_id=0, pool_index=1, offset=0, size=128),
    }
    manager = _make_fake_v2_manager(
        attrs, num_pools=2, slot_bytes_list=(512, 128), main_kv_layout="NHD"
    )

    page_table = spec_named("minimax_m3").describe(manager)
    assert [(p.base_address, p.slot_bytes, p.num_slots) for p in page_table.pools] == [
        (0x1000, 512, 4),
        (0x1000, 128, 4),
    ]
    (layer_group,) = page_table.layer_groups
    assert layer_group.spec == "minimax_m3"
    assert layer_group.params == {"window": None}
    nhd = _nhd(128, 1, 16)
    replicated = (BufferDecl(128),)
    assert _layout(layer_group) == [
        (0, 0, "key", 0, nhd),
        (0, 0, "value", 128, nhd),
        (0, 1, "key", 256, nhd),
        (0, 1, "value", 384, nhd),
        (1, 1, "index_key", 0, replicated),
    ]
    assert [r.replicated for r in layer_group.regions] == [False] * 4 + [True]


def test_v2_builder_splits_mixed_kind_pool_into_per_role_regions():
    """A pool coalescing several role classes yields one region per (layer, role).

    Miniature of the MiniMax M3 layout at TP degrees where
    K == V == INDEX_KEY bytes per block: V2 storage coalesces all three
    into one pool and the slot interleaves the sparse layer's INDEX_KEY
    between the layers' K/V buffers. Each buffer keeps its own layout:
    NHD K/V regions at non-uniform layer offsets, and a replicated
    INDEX_KEY region covering only the sparse layer.
    """
    attrs = {
        (0, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=0, size=128),
        (0, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=128, size=128),
        (0, Role.INDEX_KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=256, size=128),
        (1, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=384, size=128),
        (1, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=512, size=128),
    }
    manager = _make_fake_v2_manager(attrs, main_kv_layout="NHD")

    page_table = spec_named("minimax_m3").describe(manager)
    assert len(page_table.pools) == 1
    (layer_group,) = page_table.layer_groups
    nhd = _nhd(128, 1, 16)
    replicated = (BufferDecl(128),)
    # All regions address the same physical pool; the INDEX_KEY interleaves after layer 0.
    assert _layout(layer_group) == [
        (0, 0, "key", 0, nhd),
        (0, 0, "value", 128, nhd),
        (0, 0, "index_key", 256, replicated),
        (0, 1, "key", 384, nhd),
        (0, 1, "value", 512, nhd),
    ]


def test_v2_builder_defaults_to_head_major_kv():
    attrs = {
        (0, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=0, size=128),
        (0, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=128, size=128),
        (1, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=256, size=128),
        (1, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=384, size=128),
    }
    # A manager no spec claims by class is plain attention: head-major K/V written by the TRTLLM
    # attention kernels.
    page_table = build_page_table_from_manager(_make_fake_v2_manager(attrs))
    assert page_table.layer_groups[0].spec == "attention"
    assert [r.buffers for r in page_table.layer_groups[0].regions] == [_hnd(128, 1)] * 4


@pytest.mark.cpu_only
def test_state_layer_group_serialization():
    conv_sections = (512, 256, 256)
    ssm_bytes, ssm_bytes_per_head = 256, 128
    conv = tuple(BufferDecl(s, (("byte", s),), "byte") for s in conv_sections)
    ssm = (BufferDecl(ssm_bytes, (("head", ssm_bytes // ssm_bytes_per_head),), "head"),)
    regions = []
    for lid, layer in enumerate((10, 11, 12)):
        base = lid * (sum(conv_sections) + ssm_bytes)
        regions.append(Region(0, layer, str(MambaRole.CONV_STATE), base, conv, local_layer=lid))
        regions.append(
            Region(
                0, layer, str(MambaRole.SSM_STATE), base + sum(conv_sections), ssm, local_layer=lid
            )
        )
    layer_group = LayerGroup(spec="mamba", kind=CacheKind.STATE, regions=regions)

    d = layer_group.to_dict()
    assert d["kind"] == int(CacheKind.STATE)
    assert d["layouts"] == [[b.to_dict() for b in conv], [b.to_dict() for b in ssm]]

    restored = LayerGroup.from_dict(d)
    assert restored.spec == "mamba"
    assert restored.kind == CacheKind.STATE
    assert restored.params == {}
    assert restored.regions == regions
    conv_region, ssm_region = restored.regions[:2]
    assert [b.nbytes for b in conv_region.buffers] == list(conv_sections)
    assert conv_region.nbytes == sum(conv_sections)
    assert ssm_region.buffers[0].unit_bytes("ssm") == ssm_bytes_per_head
    assert [r.layer for r in restored.regions[::2]] == [10, 11, 12]
    # The manager's own layer ids stay on this rank and do not take part in equality.
    assert all(r.local_layer == -1 for r in restored.regions)

    legacy_pool = Pool.from_dict({"base_address": 1000, "slot_bytes": 128, "num_slots": 10})
    assert legacy_pool.slot_stride_bytes == legacy_pool.slot_bytes


def _make_fake_v2_mamba_manager(*, ple_on_layer=1):
    """Duck-typed MambaHybridCacheManagerV2 whose descriptors hold two life cycles.

    Life cycle 0 is recurrent (layers 0 and 1): SSM and conv state coalesce
    into one 64-byte pool in ``[ssm, conv]`` order per layer, and PLE n-gram
    context lives in a second, 32-byte pool on ``ple_on_layer`` only. Life
    cycle 1 is attention (layer 2) with K/V in its own pool group.
    """

    def buf(layer_id, role):
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
                                buf(0, MambaRole.SSM_STATE),
                                buf(0, MambaRole.CONV_STATE),
                                buf(1, MambaRole.SSM_STATE),
                                buf(1, MambaRole.CONV_STATE),
                            ],
                        ),
                        SimpleNamespace(
                            single_buffer_size=32,
                            buffer_ids=[buf(ple_on_layer, MambaRole.PLE_NGRAM_CONTEXT)],
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
                            buffer_ids=[buf(2, Role.KEY), buf(2, Role.VALUE)],
                        )
                    ],
                )
            ]
        ),
    )

    manager = object.__new__(MambaHybridCacheManagerV2)
    manager.impl = SimpleNamespace(
        layer_grouping=((0, 1), (2,)),
        init_config=SimpleNamespace(
            tokens_per_block=16,
            layers=[SimpleNamespace(window_size=None) for _ in range(3)],
        ),
        pool_group_descs=[recurrent_pg, attention_pg],
    )
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


def test_v2_builder_describes_recurrent_life_cycle_from_descriptors():
    """Conv, SSM and PLE regions all come from pool_group_descs, geometry included."""
    manager = _make_fake_v2_mamba_manager()
    page_table = build_page_table_from_manager(manager)

    # Pools are one flat list over both pool groups.
    assert [(p.base_address, p.slot_bytes, p.num_slots) for p in page_table.pools] == [
        (0x10000, 256, 4),
        (0x20000, 32, 4),
        (0x30000, 128, 8),
    ]
    assert len(page_table.layer_groups) == 2
    recurrent, attention = page_table.layer_groups
    # Layer group index == life-cycle id; both layer groups belong to the mamba spec.
    assert (recurrent.spec, recurrent.kind) == ("mamba", CacheKind.STATE)
    assert (attention.spec, attention.kind) == ("mamba", CacheKind.PAGED)
    assert recurrent.params == {}
    # The recurrent layer group keeps V2 layer numbering.
    assert [(r.local_layer, r.layer) for r in recurrent.regions] == [
        (0, 0),
        (0, 0),
        (1, 1),
        (1, 1),
        (1, 1),
    ]

    ssm = (BufferDecl(64, (("head", 2),), "head"),)
    conv = tuple(BufferDecl(s, (("byte", s),), "byte") for s in (16, 16, 32))
    ple = (BufferDecl(32),)
    ssm_role, conv_role = str(MambaRole.SSM_STATE), str(MambaRole.CONV_STATE)
    assert _layout(recurrent) == [
        (0, 0, ssm_role, 0, ssm),
        (0, 0, conv_role, 64, conv),
        (0, 1, ssm_role, 128, ssm),
        (0, 1, conv_role, 192, conv),
        (1, 1, str(MambaRole.PLE_NGRAM_CONTEXT), 0, ple),
    ]
    assert [r.replicated for r in recurrent.regions] == [False] * 4 + [True]
    assert recurrent.regions[0].buffers[0].unit_bytes("ssm") == 32

    # A state slot is addressed like a block: its base, plus the regions' offsets.
    extractor = KVRegionExtractorV1(page_table)
    assert extractor.extract(np.array([2]), 0).memory.ptrs.tolist() == [0x10000 + 2 * 256]
    assert {r.layer: r.offset for r in recurrent.regions if r.role == conv_role} == {0: 64, 1: 192}
    assert extractor.extract(np.array([3]), 1).memory.ptrs.tolist() == [0x20000 + 3 * 32]

    # Attention life cycle is untouched by the recurrent one.
    assert attention.pools == [2]
    assert attention.layers == [2]
    assert attention.params == {"window": None}
    assert _layout(attention) == [(2, 2, "key", 0, _hnd(64, 1)), (2, 2, "value", 64, _hnd(64, 1))]


def test_v2_builder_rejects_unknown_kv_layout():
    attrs = {
        (0, Role.KEY): SimpleNamespace(pool_index=0, offset=0, size=64),
        (0, Role.VALUE): SimpleNamespace(pool_index=0, offset=64, size=64),
        (1, Role.KEY): SimpleNamespace(pool_index=0, offset=128, size=64),
        (1, Role.VALUE): SimpleNamespace(pool_index=0, offset=192, size=64),
    }
    manager = _make_fake_v2_manager(attrs, slot_bytes_list=(256,), main_kv_layout="HDN")
    with pytest.raises(ValueError, match="unknown K/V layout 'HDN'"):
        spec_named("minimax_m3").describe(manager)


def test_v2_builder_rejects_section_bytes_that_do_not_cover_the_layer():
    manager = _make_fake_v2_mamba_manager()
    manager.conv_section_dims = [2, 2, 2]  # 48 bytes, but the conv buffer is 64
    with pytest.raises(ValueError, match="do not sum to the role's 64 bytes"):
        build_page_table_from_manager(manager)


def test_v2_mamba_registration_uses_coalesced_physical_pool():
    state_bytes = 64
    num_layers = 2
    num_slots = 8
    # Equal-sized SSM and convolution states share one interleaved V2 pool.
    physical_slot_bytes = state_bytes * num_layers * 2
    physical_pool = Pool(base_address=1000, slot_bytes=physical_slot_bytes, num_slots=num_slots)
    replicated = (BufferDecl(state_bytes),)
    regions = [
        Region(0, layer, role, (2 * lid + i) * state_bytes, replicated, local_layer=lid)
        for lid, layer in enumerate((1, 2))
        for i, role in enumerate((str(MambaRole.CONV_STATE), str(MambaRole.SSM_STATE)))
    ]
    page_table = PageTable(
        pools=[physical_pool],
        layer_groups=[LayerGroup(spec="mamba", kind=CacheKind.STATE, regions=regions)],
    )

    # V2 interleaved: one shared pool registered once
    assert memory_registration_descs(page_table, device_id=3) == [
        (1000, physical_slot_bytes * num_slots, 3, "kv_cache_memory_pool0")
    ]


@pytest.mark.cpu_only
def test_region_offsets_past_four_gib_survive_rank_info_roundtrip() -> None:
    """Region offsets and pool sizes are 64-bit on the wire."""
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

    slot_bytes = 1 << 33  # 8 GiB slot: later regions sit past 4 GiB
    buffers = (BufferDecl(1 << 31),)
    regions = [
        Region(0, 10 + lid, "conv_state", lid * (1 << 31), buffers, local_layer=lid)
        for lid in range(4)
    ]
    page_table = PageTable(
        pools=[Pool(base_address=(1 << 40) + 1000, slot_bytes=slot_bytes, num_slots=2)],
        layer_groups=[LayerGroup(spec="mamba", kind=CacheKind.STATE, regions=regions)],
    )
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
        page_table=page_table,
    )
    restored = RankInfo.from_bytes(ri.to_bytes()).page_table
    assert [r.offset for r in restored.layer_groups[0].regions] == [
        lid * (1 << 31) for lid in range(4)
    ]
    assert restored.pools[0].to_dict() == page_table.pools[0].to_dict()


@pytest.mark.cpu_only
def test_mixed_page_table_serialization():
    # Paged attention layer group (life cycle 0): layer 0.
    attn_regions = [
        Region(0, 0, "key", 0, _hnd(256, 4), local_layer=0),
        Region(0, 0, "value", 256, _hnd(256, 4), local_layer=0),
    ]
    # Recurrent state layer group (life cycle 1): layers 1 and 2.
    conv = tuple(BufferDecl(s, (("byte", s),), "byte") for s in (256, 128, 128))
    ssm = (BufferDecl(2048, (("head", 32),), "head"),)
    state_regions = [
        region
        for lid, layer in enumerate((1, 2))
        for region in (
            Region(1, layer, str(MambaRole.CONV_STATE), 0, conv, local_layer=lid),
            Region(2, layer, str(MambaRole.SSM_STATE), 0, ssm, local_layer=lid),
        )
    ]
    page_table = PageTable(
        pools=[Pool(1000, 512, 10), Pool(2000, 1024, 10), Pool(3000, 2048, 10)],
        layer_groups=[
            LayerGroup(
                spec="mamba",
                kind=CacheKind.PAGED,
                regions=attn_regions,
                params={"window": None},
            ),
            LayerGroup(spec="mamba", kind=CacheKind.STATE, regions=state_regions),
        ],
    )

    restored = PageTable.from_dict(page_table.to_dict())

    assert len(restored.layer_groups) == 2
    paged, state = restored.layer_groups
    assert (paged.kind, state.kind) == (CacheKind.PAGED, CacheKind.STATE)
    assert paged.layers == [0]
    assert state.layers == [1, 2]
    assert state.pools == [1, 2]
    assert [r.layer for r in state.regions[::2]] == [1, 2]
    assert state.regions == state_regions

    # Which layer group holds each layer comes straight from the regions.
    layer_to_group = {r.layer: gi for gi, s in enumerate(restored.layer_groups) for r in s.regions}
    assert layer_to_group == {0: 0, 1: 1, 2: 1}
