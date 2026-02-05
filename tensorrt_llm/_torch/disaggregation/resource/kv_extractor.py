from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Union

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, KVCacheManagerV2, Role
from tensorrt_llm._utils import get_size_in_bytes


class PoolRole(Enum):
    KV_CACHE = auto()
    KV_BLOCK_SCALE = auto()
    SSM_STATE = auto()
    CONV_STATE = auto()
    INDEXER = auto()


@dataclass
class LayerGroupAttrs:
    group_id: int
    pool_base_ptrs: List[int]
    pool_sizes: List[int]
    roles_to_pool_idx: Dict[PoolRole, int]
    block_bytes_per_pool: List[int]
    global_layer_ids: List[int]
    kv_head_num_per_rank: int
    sliding_window_size: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert PoolRole enum keys to strings for serialization
        d["roles_to_pool_idx"] = {k.name: v for k, v in self.roles_to_pool_idx.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "LayerGroupAttrs":
        # Convert string keys back to PoolRole enum
        if "roles_to_pool_idx" in data:
            data["roles_to_pool_idx"] = {
                PoolRole[k]: v for k, v in data["roles_to_pool_idx"].items()
            }
        return cls(**data)


@dataclass
class KVPoolAttrs:
    layer_to_group_id: Dict[int, int] = field(default_factory=dict)  # global_layer_id -> group_id
    layer_group_attrs_list: List[LayerGroupAttrs] = field(
        default_factory=list
    )  # group_id -> LayerGroupAttrs

    @property
    def block_bytes(self) -> List[int]:
        """Backward compatibility property - returns block_bytes_per_pool from first layer group."""
        if self.layer_group_attrs_list:
            return self.layer_group_attrs_list[0].block_bytes_per_pool
        return []

    def to_dict(self) -> dict:
        return {
            "layer_to_group_id": self.layer_to_group_id,
            "layer_group_attrs_list": [attrs.to_dict() for attrs in self.layer_group_attrs_list],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KVPoolAttrs":
        if data is None:
            return None
        layer_group_attrs_list = [
            LayerGroupAttrs.from_dict(attrs) for attrs in data.get("layer_group_attrs_list", [])
        ]
        return cls(
            layer_to_group_id=data.get("layer_to_group_id", {}),
            layer_group_attrs_list=layer_group_attrs_list,
        )


class KVRegionExtractorV1(RegionExtractorBase):
    """
    Descriptor and region extractor for KV cache pool managed by KVCacheManager or KVCacheManagerV2.
    Provides region descriptors for adapting block-wise view.
    """

    def __init__(self, kv_arg: Union[KVCacheManager, KVCacheManagerV2, KVPoolAttrs]):
        if isinstance(kv_arg, KVPoolAttrs):
            self._kv_pool_attrs = kv_arg
        elif self._is_kv_cache_manager(kv_arg):
            self._kv_pool_attrs = self._attrs_from_manager(kv_arg)
        elif self._is_kv_cache_manager_v2(kv_arg):
            self._kv_pool_attrs = self._attrs_from_manager_v2(kv_arg)
        else:
            raise TypeError(
                f"kv_arg must be KVCacheManager, KVCacheManagerV2, or KVPoolAttrs, "
                f"got {type(kv_arg)}"
            )
        self._data_layout = DataLayout.HND

    @property
    def kv_pool_attrs(self) -> KVPoolAttrs:
        return self._kv_pool_attrs

    @staticmethod
    def _is_kv_cache_manager(obj) -> bool:
        return hasattr(obj, "get_unique_primary_pool") and hasattr(obj, "dtype")

    @staticmethod
    def _is_kv_cache_manager_v2(obj) -> bool:
        return hasattr(obj, "impl") and hasattr(obj.impl, "layer_grouping")

    @staticmethod
    def _attrs_from_manager(manager: KVCacheManager) -> KVPoolAttrs:
        """Convert a KVCacheManager into KVPoolAttrs."""
        window_size_to_local_layer_ids = manager._get_window_size_to_layers()
        layer_offset = manager.layer_offsets

        local_layer_to_global_layer_id = {
            local_layer_id: global_layer_id
            for global_layer_id, local_layer_id in layer_offset.items()
        }

        if len(window_size_to_local_layer_ids) < 1:
            raise ValueError(
                "KVRegionExtractorV1: window_size_to_local_layer_ids is empty, "
                "cannot extract KV pool attributes from manager"
            )

        element_bytes = get_size_in_bytes(1, manager.dtype)
        layer_group_attrs_list: List[LayerGroupAttrs] = []
        global_layer_to_group_id: Dict[int, int] = {}

        # Sort window sizes for consistent group_id assignment
        sorted_window_sizes = sorted(
            window_size_to_local_layer_ids.keys(), key=lambda x: (x is None, x)
        )

        for group_id, window_size in enumerate(sorted_window_sizes):
            local_layer_ids = window_size_to_local_layer_ids[window_size]
            first_local_layer = local_layer_ids[0]

            pool_tensor = manager.impl.get_primary_pool_by_first_layer_idx(first_local_layer)

            kv_ptr = int(pool_tensor.data_ptr())
            pool_size = pool_tensor.element_size() * pool_tensor.numel()
            pool_ptrs: List[int] = [kv_ptr]
            pool_sizes: List[int] = [pool_size]

            num_kv_heads = manager.num_kv_heads_per_layer[first_local_layer]
            kv_factor = manager.kv_factor
            block_size = (
                manager.tokens_per_block
                * kv_factor
                * num_kv_heads
                * manager.head_dim
                * element_bytes
            ) * len(local_layer_ids)
            block_sizes: List[int] = [block_size]

            global_layer_ids = [local_layer_to_global_layer_id[lid] for lid in local_layer_ids]

            for global_lid in global_layer_ids:
                global_layer_to_group_id[global_lid] = group_id

            layer_group_attrs = LayerGroupAttrs(
                group_id=group_id,
                pool_base_ptrs=pool_ptrs,
                pool_sizes=pool_sizes,
                roles_to_pool_idx={PoolRole.KV_CACHE: 0},
                block_bytes_per_pool=block_sizes,
                global_layer_ids=global_layer_ids,
                kv_head_num_per_rank=num_kv_heads,
                sliding_window_size=window_size,
            )
            layer_group_attrs_list.append(layer_group_attrs)

        return KVPoolAttrs(
            layer_to_group_id=global_layer_to_group_id,
            layer_group_attrs_list=layer_group_attrs_list,
        )

    @staticmethod
    def _attrs_from_manager_v2(manager: KVCacheManagerV2) -> KVPoolAttrs:
        """Convert a KVCacheManagerV2 into KVPoolAttrs."""
        layer_group_attrs_list: List[LayerGroupAttrs] = []
        global_layer_to_group_id: Dict[int, int] = {}

        pp_layers = manager.pp_layers

        for group_id, local_layer_ids in enumerate(manager.impl.layer_grouping):
            first_local_layer = local_layer_ids[0]

            key_base_ptr = int(manager.impl.get_mem_pool_base_address(first_local_layer, Role.KEY))

            # MLA mode (kv_factor=1): only KEY, no VALUE
            if manager.kv_factor == 1:
                pool_base_ptr = key_base_ptr
            else:
                value_base_ptr = int(
                    manager.impl.get_mem_pool_base_address(first_local_layer, Role.VALUE)
                )
                pool_base_ptr = min(key_base_ptr, value_base_ptr)

            page_stride_key = manager.impl.get_page_stride(first_local_layer, Role.KEY)
            num_pages = manager.impl.get_page_index_upper_bound(first_local_layer, Role.KEY)

            pool_size = num_pages * page_stride_key * manager.kv_factor
            block_size = page_stride_key * manager.kv_factor * len(local_layer_ids)

            pool_ptrs: List[int] = [pool_base_ptr]
            pool_sizes: List[int] = [pool_size]
            block_sizes: List[int] = [block_size]

            life_cycle = manager.impl._life_cycles[group_id]
            sliding_window_size = life_cycle.window_size

            num_kv_heads = manager.num_kv_heads_per_layer[first_local_layer]

            global_layer_ids = [pp_layers[lid] for lid in local_layer_ids]

            for global_lid in global_layer_ids:
                global_layer_to_group_id[global_lid] = group_id

            layer_group_attrs = LayerGroupAttrs(
                group_id=group_id,
                pool_base_ptrs=pool_ptrs,
                pool_sizes=pool_sizes,
                roles_to_pool_idx={PoolRole.KV_CACHE: 0},
                block_bytes_per_pool=block_sizes,
                global_layer_ids=global_layer_ids,
                kv_head_num_per_rank=num_kv_heads,
                sliding_window_size=sliding_window_size,
            )
            layer_group_attrs_list.append(layer_group_attrs)

        return KVPoolAttrs(
            layer_to_group_id=global_layer_to_group_id,
            layer_group_attrs_list=layer_group_attrs_list,
        )

    @staticmethod
    def create_kv_pool_attrs_from_manager(
        manager: Union[KVCacheManager, KVCacheManagerV2],
    ) -> KVPoolAttrs:
        """Static helper to create KVPoolAttrs from either manager type."""
        if KVRegionExtractorV1._is_kv_cache_manager_v2(manager):
            return KVRegionExtractorV1._attrs_from_manager_v2(manager)
        elif KVRegionExtractorV1._is_kv_cache_manager(manager):
            return KVRegionExtractorV1._attrs_from_manager(manager)
        else:
            raise ValueError(f"Unsupported KVCacheManager type: {type(manager)}")

    def extract(self, region_ids: List[int], layer_group_id: int = 0) -> SpecRegion:
        """
        Given a list of region_ids, returns a single SpecRegion,
        whose memory is a MemRegionGroup containing all blocks described
        by region_ids.
        """
        layer_group_attrs = self._kv_pool_attrs.layer_group_attrs_list[layer_group_id]
        pool_idx = layer_group_attrs.roles_to_pool_idx[PoolRole.KV_CACHE]

        base_ptr = int(layer_group_attrs.pool_base_ptrs[pool_idx])
        block_size = int(layer_group_attrs.block_bytes_per_pool[pool_idx])

        ptrs = [base_ptr + block_size * int(bid) for bid in region_ids]
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=block_size)
        return SpecRegion(memory=memory)
