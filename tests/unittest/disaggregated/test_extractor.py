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

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    _build_v2_mamba_state_pool,
    build_page_table,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_global_layer_ids,
    get_layer_byte_ranges,
    get_layer_to_layer_group,
    get_num_layer_groups,
    get_num_layers,
    get_physical_pool,
    get_slot_address,
    get_unique_layers,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    CacheTypeCpp,
    DataType,
    KvCacheConfig,
    KVCacheManager,
    Mapping,
)


class DummyRankInfo:
    instance_name = "dummy"
    instance_rank = 0
    tp_size = 1
    tp_rank = 0
    pp_size = 1
    pp_rank = 0
    dp_size = 1
    dp_rank = 0
    cp_size = 1
    cp_rank = 0
    device_id = 0
    kv_heads_per_rank = 8
    tokens_per_block = 32
    dims_per_head = 16
    element_bytes = 2
    enable_attention_dp = False
    is_mla = False
    layer_num_per_pp = [1]

    @property
    def kv_factor(self) -> int:
        return 2 if not self.is_mla else 1


@pytest.mark.cuda
def test_extract():
    num_layers = 1
    num_kv_heads = 8
    head_dim = 16
    tokens_per_block = 32
    max_seq_len = 128
    max_batch_size = 2
    dtype = DataType.HALF
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)
    kv_cache_config = KvCacheConfig(
        max_tokens=512,
        free_gpu_memory_fraction=0.1,
        max_attention_window=None,
        enable_block_reuse=False,
        event_buffer_max_size=0,
        host_cache_size=0,
        enable_partial_reuse=False,
        copy_on_partial_reuse=False,
        sink_token_length=0,
        max_util_for_resume=1,
    )
    kv_cache_type = CacheTypeCpp.SELF

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=kv_cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype,
    )

    extractor = KVRegionExtractorV1(manager)
    region_ids = np.array([0, 1], dtype=np.int64)
    spec_region = extractor.extract(region_ids)

    assert isinstance(spec_region, SpecRegion)
    memory = spec_region.memory
    assert isinstance(memory, MemRegionGroup)
    assert len(memory.ptrs) == len(region_ids)
    assert memory.bytes_per_region > 0

    pool_ptrs = manager.get_unique_primary_pool()
    if hasattr(pool_ptrs, "__getitem__"):
        if hasattr(pool_ptrs[0], "data_ptr"):
            pool_base_ptr = int(pool_ptrs[0].data_ptr())
        else:
            pool_base_ptr = int(pool_ptrs[0])
    else:
        pool_base_ptr = (
            int(pool_ptrs.data_ptr()) if hasattr(pool_ptrs, "data_ptr") else int(pool_ptrs)
        )
    assert isinstance(memory.ptrs, np.ndarray)
    expected_block_bytes = memory.bytes_per_region
    expected_ptrs = [pool_base_ptr + block_id * expected_block_bytes for block_id in region_ids]
    np.testing.assert_array_equal(memory.ptrs, expected_ptrs)

    manager.shutdown()


@pytest.mark.cuda
def test_build_page_table():
    num_layers = 8
    num_kv_heads = 8
    head_dim = 128
    tokens_per_block = 64
    max_tokens = 3200

    kv_cache_config = KvCacheConfig(max_tokens=max_tokens)

    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)

    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=2048,
        max_batch_size=8,
        mapping=mapping,
        dtype=DataType.HALF,
        max_num_tokens=max_tokens,
    )

    page_table = build_page_table(manager)

    assert page_table.tokens_per_block == 64
    assert get_num_layers(page_table) == 8
    assert get_num_layer_groups(page_table) > 0

    lg = page_table.layer_groups[0]
    pv = lg.pool_views[0]
    pool = get_physical_pool(page_table, 0, pv.pool_idx)
    assert pool.base_address > 0
    assert pool.num_slots == 50  # 3200 tokens / 64 tokens_per_block
    assert len(pv.buffer_entries) > 0
    assert pv.pool_role == frozenset({"key", "value"})
    assert pv.mapper_kind == MapperKind.INDEXED
    assert len(get_global_layer_ids(lg)) > 0

    local_layer_id = list(get_unique_layers(pv))[0]
    layer_entries = sorted(
        (e for e in pv.buffer_entries if int(e["local_layer_id"]) == local_layer_id),
        key=lambda e: int(e["offset"]),
    )
    assert len(layer_entries) >= 2  # KV layout: K + V buffer per layer
    pool = get_physical_pool(page_table, 0, pv.pool_idx)
    ptr_key = int(pool.base_address) + int(layer_entries[0]["offset"])
    ptr_value = int(pool.base_address) + int(layer_entries[1]["offset"])
    assert ptr_key > 0
    assert ptr_value > ptr_key

    # Verify layer_groups are populated
    assert page_table.layer_groups is not None
    assert len(page_table.layer_groups) == get_num_layer_groups(page_table)
    assert len(get_global_layer_ids(page_table.layer_groups[0])) > 0
    assert page_table.layer_groups[0].kv_head_num_per_rank == num_kv_heads

    layer_to_lg = get_layer_to_layer_group(page_table)
    assert layer_to_lg is not None
    assert len(layer_to_lg) == num_layers

    print(
        f"Page table created: {get_num_layer_groups(page_table)} layer_groups,"
        f" tokens_per_block={page_table.tokens_per_block},"
        f" num_layers={get_num_layers(page_table)}"
    )

    manager.shutdown()


def _make_v1_dsa_manager(pp_size: int = 1, pp_rank: int = 0) -> KVCacheManager:
    """V1 KVCacheManager with the DSA indexer K cache enabled (MLA-style)."""
    return KVCacheManager(
        kv_cache_config=KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=False,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=4,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=pp_size, rank=pp_rank, tp_size=1, pp_size=pp_size),
        dtype=DataType.HALF,
        enable_indexer_k_cache=True,
        indexer_k_cache_quant_block_size=128,
        indexer_k_cache_index_head_dim=128,
    )


def _byte_view(ptr: int, nbytes: int):
    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

    return convert_to_torch_tensor(TensorWrapper(int(ptr), DataType.INT8, [int(nbytes)]))


@pytest.mark.cuda
def test_v1_dsa_indexer_page_table_is_replicated_with_per_layer_entries():
    manager = _make_v1_dsa_manager()
    try:
        page_table = build_page_table(manager)
        lg = page_table.layer_groups[0]
        assert len(lg.pool_views) == 2

        kv_view, idx_view = lg.pool_views
        assert kv_view.mapper_kind == MapperKind.INDEXED
        assert idx_view.mapper_kind == MapperKind.REPLICATED
        assert idx_view.pool_role == frozenset({"indexer_k"})

        # One synthesized entry per LG layer, equal-sized, contiguous from 0.
        assert get_unique_layers(idx_view) == get_unique_layers(kv_view)
        idx_pool = get_physical_pool(page_table, 0, idx_view.pool_idx)
        sizes = {int(entry["size"]) for entry in idx_view.buffer_entries}
        assert len(sizes) == 1
        per_layer = sizes.pop()
        assert per_layer * len(idx_view.buffer_entries) == idx_pool.slot_bytes
        offsets = sorted(int(entry["offset"]) for entry in idx_view.buffer_entries)
        assert offsets == [i * per_layer for i in range(len(offsets))]
    finally:
        manager.shutdown()


@pytest.mark.cuda
def test_v1_dsa_indexer_replicated_transfer_across_pp():
    """PP1 ctx sends the DSA indexer K cache into two PP2 gen ranks.

    Exercises the full python path on real V1 managers: page-table build,
    role-set matching, ReplicatedMapper layer-strided offsets, and a
    byte-level copy that must land each gen rank's layer subset at the
    right offsets.
    """
    import torch

    from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import ReplicatedMapper
    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

    ctx = _make_v1_dsa_manager()
    gens = [_make_v1_dsa_manager(pp_size=2, pp_rank=r) for r in range(2)]
    try:
        ctx_extractor = KVRegionExtractorV1(ctx)
        ctx_ri = RankInfo.from_kv_cache_manager("ctx", ctx, device_id=0)
        registrar = PeerRegistrar(ctx_ri, ctx_extractor)

        # Fill ctx indexer pool with position-dependent bytes; zero gen pools.
        ctx_pool_tensor = ctx.impl.get_indexer_k_cache_pool()
        flat = ctx_pool_tensor.view(torch.uint8).flatten()
        flat.copy_(torch.arange(flat.numel(), dtype=torch.int64, device=flat.device) % 251)
        for gen in gens:
            gen.impl.get_indexer_k_cache_pool().view(torch.uint8).zero_()

        block_ids = np.array([0, 2, 3], dtype=np.int64)
        ctx_pt = ctx_extractor.page_table
        idx_pool = get_physical_pool(ctx_pt, 0, 1)
        layers_per_gen = 2
        per_layer = idx_pool.slot_bytes // 4

        for gen_pp_rank, gen in enumerate(gens):
            gen_ri = RankInfo.from_kv_cache_manager("gen", gen, device_id=0)
            registrar.register("gen", gen_ri.instance_rank, gen_ri)
            mapping = registrar.get_pool_mapping(gen_ri)
            # KV pool and indexer pool each match 1:1.
            assert mapping == {(0, 0): (0, 0), (0, 1): (0, 1)}

            mapper = registrar.get_kv_map(gen_ri, (0, 1), (0, 1))
            assert isinstance(mapper, ReplicatedMapper)

            gen_extractor = registrar.peer_extractor("gen", gen_ri.instance_rank)
            pair = mapper.map(
                ctx_extractor.extract(block_ids, layer_group_id=0, pool_idx=1),
                gen_extractor.extract(block_ids, layer_group_id=0, pool_idx=1),
            )
            # This gen rank holds 2 of the 4 layers; the fragment is the
            # contiguous 2-layer range at this PP stage's offset within
            # the ctx slot.
            assert pair.src.memory.bytes_per_region == layers_per_gen * per_layer
            expected_src_off = gen_pp_rank * layers_per_gen * per_layer
            base_ptrs = idx_pool.base_address + block_ids * idx_pool.slot_bytes
            np.testing.assert_array_equal(pair.src.memory.ptrs, base_ptrs + expected_src_off)

            # Emulate the transfer with raw byte copies, then verify content.
            for src_ptr, dst_ptr in zip(pair.src.memory.ptrs, pair.dst.memory.ptrs):
                _byte_view(dst_ptr, pair.dst.memory.bytes_per_region).copy_(
                    _byte_view(src_ptr, pair.src.memory.bytes_per_region)
                )

            gen_pool_tensor = gen.impl.get_indexer_k_cache_pool()
            gen_bytes = gen_pool_tensor.view(torch.uint8).reshape(gen_pool_tensor.shape[0], -1)
            ctx_bytes = ctx_pool_tensor.view(torch.uint8).reshape(ctx_pool_tensor.shape[0], -1)
            src_lo = gen_pp_rank * layers_per_gen * per_layer
            src_hi = src_lo + layers_per_gen * per_layer
            torch.testing.assert_close(
                gen_bytes[block_ids],
                ctx_bytes[block_ids, src_lo:src_hi],
                rtol=0,
                atol=0,
            )
    finally:
        ctx.shutdown()
        for gen in gens:
            gen.shutdown()


def test_layer_group_meta_serialization():
    import numpy as np

    from tensorrt_llm._torch.disaggregation.resource.page import (
        BUFFER_ENTRY_DTYPE,
        AttentionLayerGroup,
        KVCachePageTable,
        LocalLayer,
        PhysicalPool,
        PhysicalPoolGroup,
        PoolView,
    )

    entries = np.array(
        [(0, 0, 256), (0, 256, 256)],
        dtype=BUFFER_ENTRY_DTYPE,
    )
    kv_pool = PhysicalPool(base_address=1000, slot_bytes=512, num_slots=10)
    pv = PoolView(pool_idx=0, buffer_entries=entries)
    local_layers = [
        LocalLayer(local_layer_id=0, global_layer_id=0),
        LocalLayer(local_layer_id=1, global_layer_id=1),
    ]
    lg = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=4,
        sliding_window_size=512,
        local_layers=local_layers,
        pool_views=[pv],
    )
    page_table = KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[lg],
        pool_groups=[PhysicalPoolGroup(pools=[kv_pool])],
    )
    d = page_table.to_dict()
    restored = KVCachePageTable.from_dict(d)
    restored_lg = restored.layer_groups[0]
    assert isinstance(restored.layer_groups[0], AttentionLayerGroup)
    assert restored_lg.sliding_window_size == 512
    assert restored_lg.kv_head_num_per_rank == 4
    assert len(restored_lg.local_layers) == 2
    assert len(restored_lg.pool_views[0].buffer_entries) == 2


def _make_fake_v2_manager(attrs, role_mapper_kinds, *, num_pools=1, slot_bytes_list=(640,)):
    """Build a minimal duck-typed KVCacheManagerV2 for _build_page_table_v2.

    ``attrs`` keeps the storage-layer shape ``{(layer_id, role):
    (life_cycle_id, pool_index, offset, size)}``; the fake synthesizes the
    public ``pool_group_descs`` view from it. Buffer order within a
    coalesced buffer follows ascending offset, mirroring how the real
    storage config assigns offsets in ``buffer_ids`` order.
    """
    from types import SimpleNamespace

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
    return SimpleNamespace(
        impl=impl,
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=lambda: role_mapper_kinds,
    )


def test_v2_builder_stamps_homogeneous_pool_kinds():
    """Split KV / INDEX_KEY pools get NHD and REPLICATED views respectively."""
    from types import SimpleNamespace

    attrs = {
        (0, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=0, size=128),
        (0, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=128, size=128),
        (1, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=256, size=128),
        (1, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=384, size=128),
        (1, Role.INDEX_KEY): SimpleNamespace(life_cycle_id=0, pool_index=1, offset=0, size=128),
    }
    manager = _make_fake_v2_manager(
        attrs,
        {Role.ALL: MapperKind.NHD, Role.INDEX_KEY: MapperKind.REPLICATED},
        num_pools=2,
        slot_bytes_list=(512, 128),
    )

    views = build_page_table_from_manager(manager).layer_groups[0].pool_views
    assert len(views) == 2
    assert views[0].pool_role == frozenset({"key", "value"})
    assert views[0].mapper_kind == MapperKind.NHD
    assert get_unique_layers(views[0]) == {0, 1}
    assert views[1].pool_role == frozenset({"index_key"})
    assert views[1].mapper_kind == MapperKind.REPLICATED
    assert get_unique_layers(views[1]) == {1}


def test_v2_builder_splits_mixed_kind_pool_into_per_class_views():
    """A pool coalescing several role classes yields one view per class.

    Miniature of the MiniMax M3 layout at TP degrees where
    K == V == INDEX_KEY bytes per block: V2 storage coalesces all three
    into one pool and the slot interleaves the sparse layer's INDEX_KEY
    between the layers' K/V regions. The builder must split the pool into
    a NHD K/V view (non-uniform layer offsets, uniform per-layer size)
    and a REPLICATED INDEX_KEY view covering only the sparse layer.
    """
    from types import SimpleNamespace

    attrs = {
        (0, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=0, size=128),
        (0, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=128, size=128),
        (0, Role.INDEX_KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=256, size=128),
        (1, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=384, size=128),
        (1, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=512, size=128),
    }
    manager = _make_fake_v2_manager(
        attrs,
        {Role.ALL: MapperKind.NHD, Role.INDEX_KEY: MapperKind.REPLICATED},
    )

    views = build_page_table_from_manager(manager).layer_groups[0].pool_views
    assert len(views) == 2
    kv_view, idx_view = views
    # Both views address the same physical pool.
    assert kv_view.pool_idx == idx_view.pool_idx == 0

    assert kv_view.pool_role == frozenset({"key", "value"})
    assert kv_view.mapper_kind == MapperKind.NHD
    assert get_unique_layers(kv_view) == {0, 1}
    assert kv_view.bytes_per_layer == 256
    kv_starts, kv_bytes_per_layer = get_layer_byte_ranges(kv_view)
    # Layer stride is non-uniform: INDEX_KEY interleaves after layer 0.
    assert kv_starts == {0: 0, 1: 384}
    assert kv_bytes_per_layer == 256

    assert idx_view.pool_role == frozenset({"index_key"})
    assert idx_view.mapper_kind == MapperKind.REPLICATED
    assert get_unique_layers(idx_view) == {0}
    assert idx_view.bytes_per_layer == 128
    idx_starts, _ = get_layer_byte_ranges(idx_view)
    assert idx_starts == {0: 256}


def test_v2_builder_validates_role_mapper_declaration():
    from types import SimpleNamespace

    attrs = {
        (0, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=0, size=128),
        (0, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=128, size=128),
        (1, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=256, size=128),
        (1, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=384, size=128),
    }

    # Default INDEXED declaration keeps the whole-slot HND view.
    views = (
        build_page_table_from_manager(_make_fake_v2_manager(attrs, {Role.ALL: MapperKind.INDEXED}))
        .layer_groups[0]
        .pool_views
    )
    assert len(views) == 1
    assert views[0].mapper_kind == MapperKind.INDEXED

    # V2 managers must expose the capability method explicitly.
    missing_capability = _make_fake_v2_manager(attrs, {Role.ALL: MapperKind.INDEXED})
    del missing_capability.get_disagg_role_mapper_kinds
    with pytest.raises(AttributeError, match="get_disagg_role_mapper_kinds"):
        build_page_table_from_manager(missing_capability)

    with pytest.raises(ValueError, match="must define Role.ALL"):
        build_page_table_from_manager(
            _make_fake_v2_manager(attrs, {Role.INDEX_KEY: MapperKind.REPLICATED})
        )

    with pytest.raises(ValueError, match="Invalid disaggregation mapper kind 'HND'"):
        build_page_table_from_manager(_make_fake_v2_manager(attrs, {Role.ALL: "HND"}))

    with pytest.raises(ValueError, match="INDEXED is only valid as the Role.ALL mapping"):
        build_page_table_from_manager(
            _make_fake_v2_manager(attrs, {Role.ALL: MapperKind.NHD, Role.KEY: MapperKind.INDEXED})
        )

    # INDEXED as the Role.ALL fallback coexists with side-cache roles that
    # declare their own kind (the base-manager default for INDEX_KEY); with
    # no INDEX_KEY buffers registered the declaration is inert.
    views = (
        build_page_table_from_manager(
            _make_fake_v2_manager(
                attrs,
                {Role.ALL: MapperKind.INDEXED, Role.INDEX_KEY: MapperKind.REPLICATED},
            )
        )
        .layer_groups[0]
        .pool_views
    )
    assert len(views) == 1
    assert views[0].mapper_kind == MapperKind.INDEXED


def test_mamba_layer_group_serialization():
    from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup, PhysicalPool

    conv_pool = PhysicalPool(
        base_address=1000,
        slot_bytes=128,
        num_slots=10,
        slot_stride_bytes=512,
        layer_stride_bytes=256,
    )
    ssm_pool = PhysicalPool(
        base_address=8000,
        slot_bytes=256,
        num_slots=8,
        slot_stride_bytes=1024,
        layer_stride_bytes=512,
    )
    mlg = MambaLayerGroup(
        pool_group_idx=1,
        mamba_layer_offsets={10: 0, 11: 1, 12: 2},
        conv_states=conv_pool,
        ssm_states=ssm_pool,
        conv_section_bytes=[512, 256, 256],
        ssm_bytes_per_head=128,
    )

    d = mlg.to_dict()
    assert d["mamba_layer_offsets"] == {10: 0, 11: 1, 12: 2}
    assert d["conv_section_bytes"] == [512, 256, 256]

    from tensorrt_llm._torch.disaggregation.resource.page import LayerGroup

    restored = LayerGroup.from_dict(d)
    assert isinstance(restored, MambaLayerGroup)
    assert restored.mamba_layer_offsets == {10: 0, 11: 1, 12: 2}
    assert restored.conv_states.base_address == 1000
    assert restored.conv_states.slot_stride_bytes == 512
    assert get_slot_address(restored.conv_states, 3) == 1000 + 3 * 512
    assert restored.ssm_states.base_address == 8000
    assert restored.ssm_states.slot_stride_bytes == 1024
    assert restored.conv_section_bytes == [512, 256, 256]
    assert restored.ssm_bytes_per_head == 128

    legacy_pool = PhysicalPool.from_dict({"base_address": 1000, "slot_bytes": 128, "num_slots": 10})
    assert legacy_pool.slot_stride_bytes == legacy_pool.slot_bytes
    assert legacy_pool.layer_stride_bytes == legacy_pool.num_slots * legacy_pool.slot_bytes


def test_v2_mamba_state_pool_uses_affine_layer_and_slot_strides():
    num_slots = 3
    state_bytes = 64
    storage = torch.empty((num_slots, 4, state_bytes), dtype=torch.uint8)
    states = [storage[:, 0, :], storage[:, 2, :]]

    pool = _build_v2_mamba_state_pool(states)

    assert pool.base_address == states[0].data_ptr()
    assert pool.slot_bytes == state_bytes
    assert pool.num_slots == num_slots
    assert pool.slot_stride_bytes == 4 * state_bytes
    assert pool.layer_stride_bytes == 2 * state_bytes


def test_v2_mamba_single_layer_pool_preserves_shared_role_footprint():
    state_bytes = 64
    storage = torch.empty((3, 2, state_bytes), dtype=torch.uint8)

    pool = _build_v2_mamba_state_pool([storage[:, 1, :]])

    assert pool.slot_bytes == state_bytes
    assert pool.slot_stride_bytes == 2 * state_bytes
    assert pool.layer_stride_bytes == 2 * state_bytes


def test_v2_mamba_state_pool_rejects_non_affine_layer_offsets():
    storage = torch.empty((3, 6, 64), dtype=torch.uint8)
    states = [storage[:, 0, :], storage[:, 2, :], storage[:, 5, :]]

    with pytest.raises(ValueError, match="uniform layer stride"):
        _build_v2_mamba_state_pool(states)


def test_v2_mamba_registration_uses_coalesced_physical_pool():
    from tensorrt_llm._torch.disaggregation.resource.page import (
        KVCachePageTable,
        MambaLayerGroup,
        PhysicalPool,
        PhysicalPoolGroup,
    )
    from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs

    state_bytes = 64
    num_layers = 2
    num_slots = 8
    # Equal-sized SSM and convolution states share one interleaved V2 pool.
    physical_slot_bytes = state_bytes * num_layers * 2
    physical_pool = PhysicalPool(
        base_address=1000,
        slot_bytes=physical_slot_bytes,
        num_slots=num_slots,
    )
    mamba_group = MambaLayerGroup(
        pool_group_idx=0,
        mamba_layer_offsets={1: 0, 2: 1},
        conv_states=PhysicalPool(
            base_address=1000 + state_bytes,
            slot_bytes=state_bytes,
            num_slots=num_slots,
            slot_stride_bytes=physical_slot_bytes,
            layer_stride_bytes=2 * state_bytes,
        ),
        ssm_states=PhysicalPool(
            base_address=1000,
            slot_bytes=state_bytes,
            num_slots=num_slots,
            slot_stride_bytes=physical_slot_bytes,
            layer_stride_bytes=2 * state_bytes,
        ),
    )
    page_table = KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[mamba_group],
        pool_groups=[PhysicalPoolGroup(pools=[physical_pool])],
    )

    assert get_unique_pool_memory_descs(page_table, device_id=3) == [
        (1000, physical_slot_bytes * num_slots, 3, "kv_cache_memory_pool0")
    ]


def test_legacy_mamba_registration_uses_layer_major_pools():
    from tensorrt_llm._torch.disaggregation.resource.page import (
        KVCachePageTable,
        MambaLayerGroup,
        PhysicalPool,
    )
    from tensorrt_llm._torch.disaggregation.resource.utils import get_unique_pool_memory_descs

    num_layers = 3
    conv_pool = PhysicalPool(base_address=1000, slot_bytes=128, num_slots=10)
    ssm_pool = PhysicalPool(base_address=8000, slot_bytes=256, num_slots=8)
    mamba_group = MambaLayerGroup(
        pool_group_idx=0,
        mamba_layer_offsets={10: 0, 11: 1, 12: 2},
        conv_states=conv_pool,
        ssm_states=ssm_pool,
    )
    page_table = KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[mamba_group],
        pool_groups=[],
    )

    assert get_unique_pool_memory_descs(page_table, device_id=3) == [
        (1000, num_layers * conv_pool.num_slots * conv_pool.slot_bytes, 3, "kv_cache_memory_pool0"),
        (8000, num_layers * ssm_pool.num_slots * ssm_pool.slot_bytes, 3, "kv_cache_memory_pool1"),
    ]


def test_mixed_page_table_serialization():
    import numpy as np

    from tensorrt_llm._torch.disaggregation.resource.page import (
        BUFFER_ENTRY_DTYPE,
        AttentionLayerGroup,
        KVCachePageTable,
        LocalLayer,
        MambaLayerGroup,
        PhysicalPool,
        PhysicalPoolGroup,
        PoolView,
    )

    # Attention layer group
    entries = np.array(
        [(0, 0, 256), (0, 256, 256)],
        dtype=BUFFER_ENTRY_DTYPE,
    )
    attn_lg = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=4,
        local_layers=[LocalLayer(0, 0)],
        pool_views=[PoolView(pool_idx=0, buffer_entries=entries)],
    )

    # Mamba layer group
    mamba_lg = MambaLayerGroup(
        pool_group_idx=1,
        mamba_layer_offsets={1: 0, 2: 1},
        conv_states=PhysicalPool(base_address=5000, slot_bytes=1024, num_slots=4),
        ssm_states=PhysicalPool(base_address=9000, slot_bytes=2048, num_slots=4),
        conv_section_bytes=[256, 128, 128],
        ssm_bytes_per_head=64,
    )

    page_table = KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[attn_lg, mamba_lg],
        pool_groups=[PhysicalPoolGroup(pools=[PhysicalPool(1000, 512, 10)])],
    )

    d = page_table.to_dict()
    restored = KVCachePageTable.from_dict(d)

    assert len(restored.layer_groups) == 2
    assert isinstance(restored.layer_groups[0], AttentionLayerGroup)
    assert isinstance(restored.layer_groups[1], MambaLayerGroup)
    assert restored.layer_groups[0].kv_head_num_per_rank == 4
    assert restored.layer_groups[1].mamba_layer_offsets == {1: 0, 2: 1}

    # Verify utils work correctly with mixed page table
    from tensorrt_llm._torch.disaggregation.resource.utils import (
        get_layer_to_layer_group,
        get_num_layer_groups,
        get_num_layers,
    )

    assert get_num_layer_groups(restored) == 2
    # get_num_layers counts only attention layers
    assert get_num_layers(restored) == 1
    # get_layer_to_layer_group maps only attention layers
    layer_map = get_layer_to_layer_group(restored)
    assert layer_map == {0: 0}


# ---------------------------------------------------------------------------
# Boundary tests for block_id -> primary memory-pool slot translation
# (see kv_extractor.py / cache_reuse.py: when host offload is enabled, a
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
      `KVRegionExtractorV1.extract` plugs into `base_ptr + idx * slot_bytes`.
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
        # match what KVRegionExtractorV1.extract would compute against
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

        # All translated indices must currently be live primary slots —
        # verify by asserting the ptrs computed via the extractor agree
        # with `primary_pool_base + slot_index * slot_bytes`.
        extractor = KVRegionExtractorV1(manager)
        spec_region = extractor.extract(np.asarray(translated_a2, dtype=np.int64))
        pool_ptrs = manager.get_unique_primary_pool()
        if hasattr(pool_ptrs, "__getitem__"):
            base = (
                int(pool_ptrs[0].data_ptr())
                if hasattr(pool_ptrs[0], "data_ptr")
                else int(pool_ptrs[0])
            )
        else:
            base = int(pool_ptrs.data_ptr()) if hasattr(pool_ptrs, "data_ptr") else int(pool_ptrs)
        slot_bytes = spec_region.memory.bytes_per_region
        for ptr, idx in zip(spec_region.memory.ptrs, translated_a2):
            assert ptr == base + idx * slot_bytes, (
                f"extractor pointer {ptr:x} disagrees with translated index "
                f"{idx} at base {base:x}, slot_bytes {slot_bytes}"
            )

        simulate_prefill_completion_only_use_for_testing(req_a2)
        manager.free_resources(req_a2)
    finally:
        manager.shutdown()
