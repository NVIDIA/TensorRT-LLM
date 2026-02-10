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
        # Dispatch to MambaLayerGroupAttrs if _type marker is present
        if data.get("_type") == "mamba":
            return MambaLayerGroupAttrs.from_dict(data)
        # Work on a copy to avoid mutating the caller's dict
        data = dict(data)
        # Convert string keys back to PoolRole enum
        if "roles_to_pool_idx" in data:
            data["roles_to_pool_idx"] = {
                (PoolRole[k] if isinstance(k, str) else k): v
                for k, v in data["roles_to_pool_idx"].items()
            }
        # Remove unknown keys for forward compatibility
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


@dataclass
class MambaLayerGroupAttrs(LayerGroupAttrs):
    """LayerGroupAttrs with Mamba-specific metadata for conv/SSM states."""

    # --- conv state section info (for TP mismatch handling) ---
    # Ordered per-section byte sizes (per layer, per rank).
    # e.g. Qwen3Next TP=2: [Q_bytes, K_bytes, V_bytes]
    # e.g. Mamba2 TP=2:     [x_bytes, B_bytes, C_bytes]
    # None => fall back to flat head-based split
    conv_section_bytes_per_rank: Optional[List[int]] = None

    # --- Mamba geometry (for debugging & future use) ---
    max_batch_size_per_pool: Optional[int] = None  # Mamba: number of slots
    d_inner_per_rank: Optional[int] = None  # head_dim * nheads / tp
    ng_ds_per_rank: Optional[int] = None  # n_groups * d_state / tp
    d_conv: Optional[int] = None  # convolution kernel size
    conv_elem_size: Optional[int] = None  # element_size of conv state
    ssm_elem_size: Optional[int] = None  # element_size of SSM state
    ssm_head_dim: Optional[int] = None
    ssm_d_state: Optional[int] = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["_type"] = "mamba"
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MambaLayerGroupAttrs":
        # Work on a copy to avoid mutating the caller's dict
        data = dict(data)
        # Convert string keys back to PoolRole enum (if not already done by parent)
        if "roles_to_pool_idx" in data:
            rtp = data["roles_to_pool_idx"]
            if rtp and isinstance(next(iter(rtp)), str):
                data["roles_to_pool_idx"] = {PoolRole[k]: v for k, v in rtp.items()}
        # Remove _type marker before constructing
        data.pop("_type", None)
        # Remove unknown keys for forward compatibility
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


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
        else:
            # Use unified method for all manager types (including MambaHybridCacheManager)
            self._kv_pool_attrs = self.create_kv_pool_attrs_from_manager(kv_arg)
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
    def _is_mamba_hybrid_manager(obj) -> bool:
        """Check if obj is a MambaHybridCacheManager (has both KV cache and Mamba cache)."""
        return hasattr(obj, "mamba_cache") and hasattr(obj, "mamba_cache_index")

    @staticmethod
    def _attrs_from_mamba_manager(manager) -> MambaLayerGroupAttrs:
        """
        Create a MambaLayerGroupAttrs for Mamba cache (Conv and SSM states).
        Conv and SSM are in the same layer_group but different pools.
        """
        # Get Mamba cache tensors
        # conv_states shape: (num_local_layers, max_batch_size, conv_dim, d_conv-1)
        conv_states = manager.mamba_cache.conv
        # ssm_states shape: (num_local_layers, max_batch_size, nheads, head_dim, d_state)
        ssm_states = manager.mamba_cache.temporal

        # Get pool base pointers
        conv_base_ptr = int(conv_states.data_ptr())
        ssm_base_ptr = int(ssm_states.data_ptr())

        # Calculate pool sizes
        conv_pool_size = conv_states.element_size() * conv_states.numel()
        ssm_pool_size = ssm_states.element_size() * ssm_states.numel()

        # Calculate bytes per slot per layer (block_bytes_per_pool)
        # conv: (conv_dim, d_conv-1) per slot per layer
        conv_slot_shape = conv_states.shape[2:]  # (conv_dim, d_conv-1)
        conv_slot_bytes_per_layer = conv_states.element_size()
        for dim in conv_slot_shape:
            conv_slot_bytes_per_layer *= dim

        # ssm: (nheads, head_dim, d_state) per slot per layer
        ssm_slot_shape = ssm_states.shape[2:]  # (nheads, head_dim, d_state)
        ssm_slot_bytes_per_layer = ssm_states.element_size()
        for dim in ssm_slot_shape:
            ssm_slot_bytes_per_layer *= dim

        # Get max_batch_size (number of slots)
        max_batch_size = conv_states.shape[1]

        # Get global layer ids from mamba_layer_offsets
        # mamba_layer_offsets: {global_layer_idx: local_offset}
        global_layer_ids = sorted(manager.mamba_layer_offsets.keys())

        # Get nheads (SSM head count, already divided by tp_size)
        nheads = ssm_states.shape[2]  # nheads from (nheads, head_dim, d_state)

        # Compute conv section bytes from stored section dims (set by MambaCacheManager)
        conv_section_bytes = None
        conv_section_dims = getattr(manager, "conv_section_dims", None)
        if conv_section_dims:
            d_conv_minus_1 = conv_states.shape[-1]  # d_conv - 1
            elem_size = conv_states.element_size()
            conv_section_bytes = [dim * d_conv_minus_1 * elem_size for dim in conv_section_dims]

        return MambaLayerGroupAttrs(
            group_id=-1,  # Will be set by caller
            pool_base_ptrs=[conv_base_ptr, ssm_base_ptr],
            pool_sizes=[conv_pool_size, ssm_pool_size],
            roles_to_pool_idx={
                PoolRole.CONV_STATE: 0,
                PoolRole.SSM_STATE: 1,
            },
            block_bytes_per_pool=[conv_slot_bytes_per_layer, ssm_slot_bytes_per_layer],
            global_layer_ids=global_layer_ids,
            kv_head_num_per_rank=nheads,
            max_batch_size_per_pool=max_batch_size,
            conv_section_bytes_per_rank=conv_section_bytes,
            d_inner_per_rank=getattr(manager, "d_inner_local", None),
            ng_ds_per_rank=getattr(manager, "ng_ds_local", None),
            d_conv=getattr(manager, "d_conv", None),
            conv_elem_size=conv_states.element_size(),
            ssm_elem_size=ssm_states.element_size(),
            ssm_head_dim=ssm_states.shape[3] if ssm_states.dim() >= 4 else None,
            ssm_d_state=ssm_states.shape[4] if ssm_states.dim() >= 5 else None,
        )

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
            roles_to_pool_idx: Dict[PoolRole, int] = {PoolRole.KV_CACHE: 0}

            if manager.enable_indexer_k_cache:
                roles_to_pool_idx[PoolRole.INDEXER] = 1
                indexer_k_cache_pool = manager.impl.get_indexer_k_cache_pool()
                indexer_k_cache_pool_base_ptr = int(indexer_k_cache_pool.data_ptr())
                indexer_k_cache_pool_size = (
                    indexer_k_cache_pool.element_size() * indexer_k_cache_pool.numel()
                )
                pool_ptrs.append(indexer_k_cache_pool_base_ptr)
                pool_sizes.append(indexer_k_cache_pool_size)
                block_sizes.append(indexer_k_cache_pool.shape[-1] * len(local_layer_ids))

            global_layer_ids = [local_layer_to_global_layer_id[lid] for lid in local_layer_ids]

            for global_lid in global_layer_ids:
                global_layer_to_group_id[global_lid] = group_id

            layer_group_attrs = LayerGroupAttrs(
                group_id=group_id,
                pool_base_ptrs=pool_ptrs,
                pool_sizes=pool_sizes,
                roles_to_pool_idx=roles_to_pool_idx,
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

            # get_mem_pool_base_address returns (slot_address + offset), we need to subtract offset
            # to get the true pool base address (important when different layer_groups share a pool)
            key_addr_with_offset = int(
                manager.impl.get_mem_pool_base_address(first_local_layer, Role.KEY)
            )
            key_attr = manager.impl._storage.get_buffer_attr(first_local_layer, Role.KEY)
            pool_base_ptr = key_addr_with_offset - key_attr.offset

            page_stride_key = manager.impl.get_page_stride(first_local_layer, Role.KEY)
            num_pages = (
                manager.impl.get_page_index_upper_bound(first_local_layer, Role.KEY)
                // manager.kv_factor
            )

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
        # First get KV cache attrs
        if KVRegionExtractorV1._is_kv_cache_manager_v2(manager):
            kv_pool_attrs = KVRegionExtractorV1._attrs_from_manager_v2(manager)
        elif KVRegionExtractorV1._is_kv_cache_manager(manager):
            kv_pool_attrs = KVRegionExtractorV1._attrs_from_manager(manager)
        else:
            raise ValueError(f"Unsupported KVCacheManager type: {type(manager)}")

        # If MambaHybridCacheManager, add Mamba layer_group
        if KVRegionExtractorV1._is_mamba_hybrid_manager(manager):
            mamba_group_attrs = KVRegionExtractorV1._attrs_from_mamba_manager(manager)
            mamba_group_id = len(kv_pool_attrs.layer_group_attrs_list)
            mamba_group_attrs.group_id = mamba_group_id
            kv_pool_attrs.layer_group_attrs_list.append(mamba_group_attrs)
            for global_lid in mamba_group_attrs.global_layer_ids:
                kv_pool_attrs.layer_to_group_id[global_lid] = mamba_group_id

        return kv_pool_attrs

    def extract(
        self,
        region_ids: List[int],
        layer_group_id: int = 0,
        pool_role: PoolRole = PoolRole.KV_CACHE,
    ) -> SpecRegion:
        """
        Given a list of region_ids, returns a single SpecRegion,
        whose memory is a MemRegionGroup containing all blocks described
        by region_ids.

        For Mamba (pool_role=SSM_STATE or CONV_STATE):
        - region_ids should contain a single slot_id
        - Returns num_local_layers addresses (one per layer)
        """
        layer_group_attrs = self._kv_pool_attrs.layer_group_attrs_list[layer_group_id]
        pool_idx = layer_group_attrs.roles_to_pool_idx[pool_role]

        base_ptr = int(layer_group_attrs.pool_base_ptrs[pool_idx])
        block_size = int(layer_group_attrs.block_bytes_per_pool[pool_idx])

        # Check if this is a Mamba pool
        if pool_role in (PoolRole.SSM_STATE, PoolRole.CONV_STATE):
            return self._extract_mamba(region_ids, layer_group_attrs, pool_idx)

        # KV cache: filter out invalid block_ids (BAD_PAGE_INDEX = -1)
        ptrs = [base_ptr + block_size * int(bid) for bid in region_ids if bid >= 0]
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=block_size)
        return SpecRegion(memory=memory)

    def _extract_mamba(
        self,
        region_ids: List[int],
        layer_group_attrs: LayerGroupAttrs,
        pool_idx: int,
    ) -> SpecRegion:
        """
        Extract Mamba state regions.
        Input: region_ids=[slot_id] (single slot)
        Output: SpecRegion with num_local_layers addresses (one per layer)

        Memory layout: (num_local_layers, max_batch_size, ...state_shape)
        Address calculation: base_ptr + (layer_idx * max_batch_size + slot_id) * block_bytes
        """
        if len(region_ids) != 1:
            raise ValueError(f"Mamba extract expects exactly 1 slot_id, got {len(region_ids)}")

        slot_id = region_ids[0]
        if slot_id < 0:
            # Invalid slot_id
            return SpecRegion(memory=MemRegionGroup(ptrs=[], bytes_per_region=0))

        base_ptr = int(layer_group_attrs.pool_base_ptrs[pool_idx])
        block_size = int(layer_group_attrs.block_bytes_per_pool[pool_idx])
        max_batch_size = layer_group_attrs.max_batch_size_per_pool
        num_local_layers = len(layer_group_attrs.global_layer_ids)

        # Generate address for each layer
        ptrs = []
        for layer_idx in range(num_local_layers):
            addr = base_ptr + (layer_idx * max_batch_size + slot_id) * block_size
            ptrs.append(addr)

        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=block_size)
        return SpecRegion(memory=memory)
