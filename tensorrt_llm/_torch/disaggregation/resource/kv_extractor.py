from dataclasses import dataclass
from typing import List

from tensorrt_llm._torch.disaggregation.base.region import (
    DataLayout,
    MemRegionGroup,
    RegionExtractorBase,
    SpecRegion,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes


@dataclass
class KVPoolAttrs:
    """Attributes for a single (primary) KV memory pool."""

    pool_ptrs: List[int]
    block_bytes: List[int]


class KVRegionExtractorV1(RegionExtractorBase):
    """
    Descriptor and region extractor for KV cache pool managed by KVCacheManager.
    Provides region descriptors for adapting block-wise view.
    """

    def __init__(self, kv_arg: KVCacheManager | KVPoolAttrs):
        if isinstance(kv_arg, KVPoolAttrs):
            self._kv_pool_attrs = kv_arg
        elif isinstance(kv_arg, KVCacheManager):
            self._kv_pool_attrs = self._attrs_from_manager(kv_arg)
        else:
            raise TypeError(
                f"kv_cache_manager must be KVCacheManager or KVPoolAttrs, got {type(kv_arg)}"
            )
        self._data_layout = DataLayout.HND

    @staticmethod
    def _attrs_from_manager(manager: KVCacheManager) -> KVPoolAttrs:
        try:
            pools = manager.get_unique_primary_pool()
        except Exception as ex:
            raise ValueError(f"Failed to get pool(s): {ex}")

        pool_list = list(pools) if isinstance(pools, (list, tuple)) else [pools]
        elem_bytes = get_size_in_bytes(1, manager.dtype)
        ptrs, block_sizes = [], []

        for p in pool_list:
            if hasattr(p, "data_ptr") and callable(p.data_ptr):
                try:
                    ptr = int(p.data_ptr())
                except Exception as ex:
                    raise ValueError(f"Fail to call data_ptr(): {ex}")
            elif isinstance(p, int):
                ptr = int(p)
            else:
                raise ValueError(f"Pool object lacks 'data_ptr' and is not int: {p!r}")
            ptrs.append(ptr)

            try:
                if hasattr(p, "__getitem__") and hasattr(p[0], "numel"):
                    n = int(p[0].numel())
                elif hasattr(p, "numel") and callable(p.numel):
                    n = int(p.numel())
                else:
                    raise RuntimeError("Cannot determine element count")
            except Exception as ex:
                raise ValueError(f"Failed to get block size from {p!r}: {ex}")

            block_sizes.append(n * elem_bytes)

        return KVPoolAttrs(pool_ptrs=ptrs, block_bytes=block_sizes)

    def extract(self, region_ids: List[int]) -> SpecRegion:
        """
        Given a list of region_ids, returns a single SpecRegion,
        whose memory is a MemRegionGroup containing all blocks described
        by region_ids.
        """
        assert len(self._kv_pool_attrs.pool_ptrs) == 1
        pool_idx = 0
        attrs = self._kv_pool_attrs
        ptrs = [
            attrs.pool_ptrs[pool_idx] + block_id * attrs.block_bytes[0] for block_id in region_ids
        ]
        memory = MemRegionGroup(ptrs=ptrs, bytes_per_region=attrs.block_bytes[0])
        return SpecRegion(memory=memory)
