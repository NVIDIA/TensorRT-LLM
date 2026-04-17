import numpy as np

from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LayerGroup,
    LocalLayer,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)


def _make_buffer_entries():
    return np.array(
        [
            (0, 0, 0, 128),  # local_layer_id=0, role=0(KEY), offset=0, size=128
            (0, 1, 128, 128),  # local_layer_id=0, role=1(VALUE), offset=128, size=128
        ],
        dtype=BUFFER_ENTRY_DTYPE,
    )


def test_physical_pool_construction():
    pool = PhysicalPool(base_address=0x10000, slot_bytes=256, num_slots=4)
    assert pool.base_address == 0x10000
    assert pool.slot_bytes == 256
    assert pool.num_slots == 4


def test_physical_pool_roundtrip():
    pool = PhysicalPool(base_address=0x10000, slot_bytes=256, num_slots=4)
    d = pool.to_dict()
    restored = PhysicalPool.from_dict(d)
    assert restored.base_address == pool.base_address
    assert restored.slot_bytes == pool.slot_bytes
    assert restored.num_slots == pool.num_slots


def test_pool_view_roundtrip():
    entries = _make_buffer_entries()
    pv = PoolView(pool_idx=0, buffer_entries=entries)
    d = pv.to_dict()
    restored = PoolView.from_dict(d)
    assert restored.pool_idx == 0
    assert len(restored.buffer_entries) == 2
    assert restored.buffer_entries[0]["offset"] == 0
    assert restored.buffer_entries[1]["offset"] == 128


def test_local_layer_roundtrip():
    ll = LocalLayer(local_layer_id=0, global_layer_id=5)
    d = ll.to_dict()
    restored = LocalLayer.from_dict(d)
    assert restored.local_layer_id == 0
    assert restored.global_layer_id == 5


def test_layer_group_roundtrip():
    entries = _make_buffer_entries()
    lg = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=8,
        sliding_window_size=None,
        local_layers=[LocalLayer(0, 5), LocalLayer(1, 6)],
        pool_views=[PoolView(pool_idx=0, buffer_entries=entries)],
    )
    d = lg.to_dict()
    restored = LayerGroup.from_dict(d)
    assert restored.pool_group_idx == 0
    assert restored.kv_head_num_per_rank == 8
    assert restored.sliding_window_size is None
    assert len(restored.local_layers) == 2
    assert len(restored.pool_views) == 1


def test_kv_cache_page_table_roundtrip():
    entries = _make_buffer_entries()
    page_table = KVCachePageTable(
        tokens_per_block=64,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=8,
                sliding_window_size=None,
                local_layers=[LocalLayer(0, 5)],
                pool_views=[PoolView(pool_idx=0, buffer_entries=entries)],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[PhysicalPool(base_address=0x10000, slot_bytes=256, num_slots=4)]
            )
        ],
    )
    d = page_table.to_dict()
    restored = KVCachePageTable.from_dict(d)
    assert restored.tokens_per_block == 64
    assert len(restored.layer_groups) == 1
    assert len(restored.pool_groups) == 1
    assert restored.pool_groups[0].pools[0].base_address == 0x10000
