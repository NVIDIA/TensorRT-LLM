import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_global_layer_ids,
    get_layer_to_layer_group,
    get_num_layer_groups,
    get_num_layers,
    get_physical_pool,
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


@pytest.mark.parametrize(
    ("bytes_per_region", "expected_region_size"),
    [(128, 128), (0, 0), (None, 384)],
)
def test_extract_logical_pool_view_uses_physical_stride(bytes_per_region, expected_region_size):
    from tensorrt_llm._torch.disaggregation.resource.page import (
        BUFFER_ENTRY_DTYPE,
        AttentionLayerGroup,
        KVCachePageTable,
        LocalLayer,
        PhysicalPool,
        PhysicalPoolGroup,
        PoolView,
    )

    page_table = KVCachePageTable(
        tokens_per_block=16,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=1,
                local_layers=[LocalLayer(0, 0)],
                pool_views=[
                    PoolView(
                        pool_idx=0,
                        buffer_entries=np.array([(0, 256, 128)], dtype=BUFFER_ENTRY_DTYPE),
                        pool_role=frozenset({"index_key"}),
                        mapper_kind=MapperKind.REPLICATED,
                        byte_offset=256,
                        bytes_per_region=bytes_per_region,
                    )
                ],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(pools=[PhysicalPool(base_address=1000, slot_bytes=384, num_slots=4)])
        ],
    )

    region = KVRegionExtractorV1(page_table).extract(np.array([0, -1, 2], dtype=np.int64))
    np.testing.assert_array_equal(region.memory.ptrs, np.array([1256, 2024]))
    assert region.memory.bytes_per_region == expected_region_size


def test_v2_index_key_builder_keeps_pool_level_view():
    from types import SimpleNamespace

    from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

    base = 0x1000
    slot_bytes = 640
    attrs = {
        (0, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=0, size=128),
        (0, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=128, size=128),
        (0, Role.INDEX_KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=256, size=128),
        (1, Role.KEY): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=384, size=128),
        (1, Role.VALUE): SimpleNamespace(life_cycle_id=0, pool_index=0, offset=512, size=128),
    }

    class FakePool:
        slot_size = slot_bytes
        num_slots = 4

        @staticmethod
        def slot_address(slot):
            return base + slot * slot_bytes

    class FakeImpl:
        layer_grouping = ((0, 1),)
        _life_cycles = [SimpleNamespace(window_size=None)]
        _init_config = SimpleNamespace(
            cache_tiers=[SimpleNamespace(tier=CacheTier.GPU_MEM)], tokens_per_block=16
        )
        _storage = SimpleNamespace(
            _buffer_attr=attrs,
            num_life_cycles=1,
            get_pool_group_index=lambda _lc: 0,
            _levels=[
                SimpleNamespace(
                    storage=SimpleNamespace(
                        _pool_groups=[SimpleNamespace(num_pools=1, _pools=[FakePool()])]
                    )
                )
            ],
        )

    manager = SimpleNamespace(
        impl=FakeImpl(),
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=lambda: {
            Role.ALL: MapperKind.NHD,
            Role.INDEX_KEY: MapperKind.REPLICATED,
        },
    )

    page_table = build_page_table_from_manager(manager)
    views = page_table.layer_groups[0].pool_views
    assert len(views) == 1
    assert views[0].pool_role == frozenset({"key", "value", "index_key"})
    assert views[0].mapper_kind == MapperKind.MIXED
    assert views[0].buffer_roles == ("key", "value", "index_key", "key", "value")
    assert views[0].buffer_mapper_kinds == (
        MapperKind.NHD,
        MapperKind.NHD,
        MapperKind.REPLICATED,
        MapperKind.NHD,
        MapperKind.NHD,
    )
    assert views[0].byte_offset == 0
    assert views[0].bytes_per_region is None

    # A dense-only PP rank has no local INDEX_KEY entry, so its one physical
    # pool view is homogeneous NHD.
    dense_impl = FakeImpl()
    dense_impl._storage = SimpleNamespace(
        _buffer_attr={key: attr for key, attr in attrs.items() if key[1] != Role.INDEX_KEY},
        num_life_cycles=1,
        get_pool_group_index=lambda _lc: 0,
        _levels=FakeImpl._storage._levels,
    )
    dense_manager = SimpleNamespace(
        impl=dense_impl,
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=manager.get_disagg_role_mapper_kinds,
    )
    dense_views = build_page_table_from_manager(dense_manager).layer_groups[0].pool_views
    assert len(dense_views) == 1
    assert get_unique_layers(dense_views[0]) == {0, 1}
    assert dense_views[0].mapper_kind == MapperKind.NHD

    # A private attribute with a coincidental name must not opt a manager into
    # model-specific logical views.
    accidental_manager = SimpleNamespace(
        impl=FakeImpl(),
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        sparse_layer_ids=[3],
        get_disagg_role_mapper_kinds=lambda: {Role.ALL: MapperKind.INDEXED},
    )
    accidental_views = build_page_table_from_manager(accidental_manager).layer_groups[0].pool_views
    assert len(accidental_views) == 1
    assert accidental_views[0].mapper_kind == MapperKind.INDEXED

    # V2 managers must expose the capability method explicitly. A missing
    # method is a programming error rather than an invitation to guess from
    # private attributes.
    missing_capability_manager = SimpleNamespace(
        impl=FakeImpl(),
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
    )
    with pytest.raises(AttributeError, match="get_disagg_role_mapper_kinds"):
        build_page_table_from_manager(missing_capability_manager)

    missing_default_manager = SimpleNamespace(
        impl=FakeImpl(),
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=lambda: {Role.INDEX_KEY: MapperKind.REPLICATED},
    )
    with pytest.raises(ValueError, match="must define Role.ALL"):
        build_page_table_from_manager(missing_default_manager)

    invalid_mapper_manager = SimpleNamespace(
        impl=FakeImpl(),
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=lambda: {Role.ALL: "HND"},
    )
    with pytest.raises(ValueError, match="Invalid disaggregation mapper kind 'HND'"):
        build_page_table_from_manager(invalid_mapper_manager)

    unsupported_mapper_manager = SimpleNamespace(
        impl=FakeImpl(),
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=lambda: {Role.ALL: MapperKind.FLAT},
    )
    with pytest.raises(ValueError, match="Unsupported V2 disaggregation mapper kind FLAT"):
        build_page_table_from_manager(unsupported_mapper_manager)

    mixed_indexed_manager = SimpleNamespace(
        impl=FakeImpl(),
        pp_layers=[0, 1],
        num_kv_heads_per_layer=[1, 1],
        get_disagg_role_mapper_kinds=lambda: {
            Role.ALL: MapperKind.NHD,
            Role.KEY: MapperKind.INDEXED,
        },
    )
    with pytest.raises(ValueError, match="INDEXED is only valid as the sole Role.ALL mapping"):
        build_page_table_from_manager(mixed_indexed_manager)


def test_mamba_layer_group_serialization():
    from tensorrt_llm._torch.disaggregation.resource.page import MambaLayerGroup, PhysicalPool

    conv_pool = PhysicalPool(base_address=1000, slot_bytes=128, num_slots=10)
    ssm_pool = PhysicalPool(base_address=8000, slot_bytes=256, num_slots=8)
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
    assert restored.conv_states.slot_bytes == 128
    assert restored.conv_states.num_slots == 10
    assert restored.ssm_states.base_address == 8000
    assert restored.ssm_states.slot_bytes == 256
    assert restored.ssm_states.num_slots == 8
    assert restored.conv_section_bytes == [512, 256, 256]
    assert restored.ssm_bytes_per_head == 128


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
