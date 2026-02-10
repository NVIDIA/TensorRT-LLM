from collections import defaultdict

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import DataRole
from tensorrt_llm._torch.disaggregation.native.region.page import (
    BUFFER_ENTRY_DTYPE,
    KVCachePageTable,
    PoolDescriptor,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import Role
from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier, KVCacheManager

_ROLE_STR_TO_ENUM: dict[str, DataRole] = {
    Role.KEY: DataRole.KEY,
    Role.VALUE: DataRole.VALUE,
    Role.KEY_BLOCK_QUANT: DataRole.KEY | DataRole.BLOCK_QUANT,
    Role.VALUE_BLOCK_QUANT: DataRole.VALUE | DataRole.BLOCK_QUANT,
}


def role_str_to_enum(role: str) -> DataRole:
    if role not in _ROLE_STR_TO_ENUM:
        valid_roles = list(_ROLE_STR_TO_ENUM.keys())
        raise ValueError(f"Invalid role: '{role}'. Valid roles: {valid_roles}")
    return _ROLE_STR_TO_ENUM[role]


def build_page_table(manager: KVCacheManager) -> KVCachePageTable:
    storage = manager._storage
    config = manager._init_config

    gpu_level = 0
    found_gpu = False
    for level_idx, cache_tier_config in enumerate(config.cache_tiers):
        if cache_tier_config.tier == CacheTier.GPU_MEM:
            gpu_level = level_idx
            found_gpu = True
            break

    if not found_gpu:
        raise ValueError("No GPU_MEM tier found in cache configuration")

    buffer_by_pool = defaultdict(list)

    lc_to_pg_cache = {}

    for buffer_id, attr in storage._buffer_attr.items():
        layer_id, role = buffer_id

        lc_id = attr.life_cycle_id
        if lc_id not in lc_to_pg_cache:
            lc_to_pg_cache[lc_id] = storage.get_pool_group_index(lc_id)
        pg_idx = lc_to_pg_cache[lc_id]

        pool_idx = attr.pool_index
        pool_key = (pg_idx, pool_idx)

        buffer_by_pool[pool_key].append((layer_id, role_str_to_enum(role), attr.offset, attr.size))

    pools = []
    num_pool_groups = storage.num_pool_groups
    pool_group_storage = storage._levels[gpu_level].storage._pool_groups

    for pg_idx in range(num_pool_groups):
        pool_group = pool_group_storage[pg_idx]
        num_pools = pool_group.num_pools
        pg_pools = []

        for pool_idx in range(num_pools):
            pool = pool_group._pools[pool_idx]

            base_address = int(pool.slot_address(0))
            slot_bytes = int(pool.slot_size)
            num_slots = int(pool.num_slots)

            pool_key = (pg_idx, pool_idx)
            buffers_info = buffer_by_pool.get(pool_key, [])

            if buffers_info:
                buffer_entries = np.array(buffers_info, dtype=BUFFER_ENTRY_DTYPE)
            else:
                buffer_entries = np.array([], dtype=BUFFER_ENTRY_DTYPE)

            pool_desc = PoolDescriptor(
                base_address=base_address,
                slot_bytes=slot_bytes,
                num_slots=num_slots,
                buffer_entries=buffer_entries,
            )
            pg_pools.append(pool_desc)
        pools.append(pg_pools)

    return KVCachePageTable(
        tokens_per_block=config.tokens_per_block,
        num_layers=len(config.layers),
        pools=pools,
    )
