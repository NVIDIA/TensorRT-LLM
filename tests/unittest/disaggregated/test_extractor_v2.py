import pytest

from tensorrt_llm._torch.disaggregation.resource.kv_extractor_v2 import build_page_table
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    GpuCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
)


@pytest.fixture
def simple_manager():
    layers = [
        AttentionLayerConfig(
            layer_id=0,
            sliding_window_size=None,
            num_sink_tokens=0,
            buffers=[
                BufferConfig(role=0, size=8192),
                BufferConfig(role=1, size=8192),
            ],
        ),
        AttentionLayerConfig(
            layer_id=1,
            sliding_window_size=None,
            num_sink_tokens=0,
            buffers=[
                BufferConfig(role=0, size=8192),
                BufferConfig(role=1, size=8192),
            ],
        ),
    ]

    cache_tiers = [
        GpuCacheTierConfig(
            quota=100 * 1024 * 1024,  # 100MB
        ),
    ]

    config = KVCacheManagerConfig(
        tokens_per_block=64,
        layers=layers,
        vocab_size=50257,
        cache_tiers=cache_tiers,
    )

    return KVCacheManager(config)


def test_build_page_table(simple_manager):
    page_table = build_page_table(simple_manager)

    # Check basic properties
    assert page_table.tokens_per_block == 64
    assert page_table.num_layers == 2
    assert page_table.num_pool_groups >= 1
    assert page_table.total_pools > 0

    # Check pools are created
    assert len(page_table.pools) > 0
    assert all(len(pg_pools) > 0 for pg_pools in page_table.pools)

    # Check first pool has valid properties
    pool = page_table.pools[0][0]
    assert pool.base_address > 0
    assert pool.slot_bytes > 0
    assert pool.num_slots > 0
    assert pool.pool_bytes == pool.slot_bytes * pool.num_slots

    # Check buffer entries exist
    assert len(pool.buffer_entries) > 0

    print(f"\n Page table: {page_table}")
    print(f"  Total pools: {page_table.total_pools}")
    print(f"  Pools: {page_table.pools}")
    print(f"  Total size: {page_table.total_pool_bytes / (1024**2):.2f} MB")
