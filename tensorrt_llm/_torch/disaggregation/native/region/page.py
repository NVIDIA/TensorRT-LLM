from dataclasses import dataclass
from typing import List, Set

import numpy as np

BUFFER_ENTRY_DTYPE = np.dtype(
    [
        ("layer_id", np.uint32),
        ("role", np.uint32),
        ("offset", np.uint32),
        ("size", np.uint32),
    ]
)


@dataclass
class PoolDescriptor:
    """
    Pool descriptor containing memory layout and buffer information

    One pool contains multiple buffer entries, each representing a (layer_id, role) combination.
    """

    base_address: int  # (uint64)
    slot_bytes: int
    num_slots: int

    # Buffer entries: flattened array of all (layer_id, role, offset, size) in this pool
    buffer_entries: np.ndarray  # dtype=BUFFER_ENTRY_DTYPE

    @property
    def pool_bytes(self) -> int:
        return self.slot_bytes * self.num_slots

    @property
    def unique_layers(self) -> Set[int]:
        return set(int(entry["layer_id"]) for entry in self.buffer_entries)

    @property
    def unique_roles(self) -> Set[int]:
        return set(int(entry["role"]) for entry in self.buffer_entries)

    def get_slot_address(self, slot_id: int) -> int:
        if slot_id >= self.num_slots:
            raise ValueError(f"slot_id {slot_id} >= num_slots {self.num_slots}")
        return self.base_address + slot_id * self.slot_bytes

    def get_device_pointer(self, slot_id: int, layer_id: int, role_enum: int) -> int:
        if slot_id >= self.num_slots:
            raise ValueError(f"slot_id {slot_id} >= num_slots {self.num_slots}")

        for entry in self.buffer_entries:
            if entry["layer_id"] == layer_id and entry["role"] == role_enum:
                slot_base = self.base_address + slot_id * self.slot_bytes
                return slot_base + int(entry["offset"])

        raise ValueError(f"Buffer not found: layer_id={layer_id}, role_enum={role_enum}")

    def __repr__(self) -> str:
        return (
            f"PoolDescriptor(base=0x{self.base_address:x}, "
            f"slot_bytes={self.slot_bytes}, num_slots={self.num_slots}, "
            f"layers={len(self.unique_layers)}, roles={len(self.unique_roles)}"
        )


@dataclass
class KVCachePageTable:
    """
    Multi-dimensional KV cache page table

    Structure:
        KVCachePageTable
        └── PoolGroups (List[List[PoolDescriptor]])
            ├── PoolGroup 0: List of PoolDescriptors
            │   ├── Pool 0: PoolDescriptor
            │   └── Pool 1: PoolDescriptor
            │
            └── PoolGroup 1: List of PoolDescriptors
                ├── Pool 0: PoolDescriptor
                └── Pool 1: PoolDescriptor

    Relationships:
        - pools[pg_idx] = List[PoolDescriptor] (all pools in same PoolGroup)
        - All pools in pools[pg_idx] share the same lifecycle
    """

    tokens_per_block: int
    num_layers: int
    pools: List[List[PoolDescriptor]]  # pools[pg_idx][pool_idx] → PoolDescriptor

    @property
    def num_pool_groups(self) -> int:
        return len(self.pools)

    @property
    def total_pools(self) -> int:
        return sum(len(pg_pools) for pg_pools in self.pools)

    @property
    def total_buffer_entries(self) -> int:
        return sum(pool.num_buffer_entries for pg_pools in self.pools for pool in pg_pools)

    @property
    def total_pool_bytes(self) -> int:
        return sum(pool.pool_bytes for pg_pools in self.pools for pool in pg_pools)

    @property
    def total_slots(self) -> int:
        return sum(pool.num_slots for pg_pools in self.pools for pool in pg_pools)

    def get_pool(self, pg_idx: int, pool_idx: int) -> PoolDescriptor:
        return self.pools[pg_idx][pool_idx]

    def get_device_pointer(
        self, pg_idx: int, pool_idx: int, slot_id: int, layer_id: int, role: str
    ) -> int:
        pool = self.pools[pg_idx][pool_idx]
        role_enum = self.role_to_enum(role)
        return pool.get_device_pointer(slot_id, layer_id, role_enum)

    def __repr__(self) -> str:
        return (
            f"KVCachePageTable(poolgroups={self.num_pool_groups}, "
            f"pools={self.total_pools}, layers={self.num_layers})"
        )
