from itertools import chain
from typing import List

import torch

import tensorrt_llm
from tensorrt_llm.runtime.memory_pools.pool import Pool


class MemoryPoolsAllocator(object):

    def __init__(self, num_blocks, tokens_per_block, head_size):
        self._pools_metadata = []
        self._pool_pointers = []
        self._pool_mapping = None

        self._num_blocks = num_blocks
        self._tokens_per_block = tokens_per_block
        self._head_size = head_size

    def allocate(self, dtype, num_kv_heads_per_layer: List[int], device="cuda"):
        if isinstance(dtype, str):
            dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)

        # LayerCachePoolLocator{.indexOfPool, .layerIdxInCachePool}"
        layers_mapping = [[-1, -1]] * len(num_kv_heads_per_layer)
        unique_nkvh = sorted(set(num_kv_heads_per_layer))
        for index_of_pool, kv_head in enumerate(unique_nkvh):
            layers = [
                layer for layer, nkvh in enumerate(num_kv_heads_per_layer)
                if nkvh == kv_head
            ]

            num_layers = len(layers)
            cache_shape = (
                self._num_blocks,
                num_layers,
                2,
                kv_head,
                self._tokens_per_block,
                self._head_size,
            )
            self._pool_pointers.append(
                torch.empty(cache_shape, dtype=dtype, device=device))
            self._pools_metadata.append(
                Pool(num_kv_heads=kv_head, num_layers=num_layers))

            for layer_idx_in_cache_pool, layer in enumerate(layers):
                layers_mapping[layer] = [index_of_pool, layer_idx_in_cache_pool]

        assert -1 not in set(chain(*layers_mapping))
        self._pool_mapping = torch.tensor(layers_mapping, dtype=torch.int32)

    def get_kv_cache_pool_pointers(self):
        return self._get_primarmy_secondary_pool_pointers()

    def _get_primarmy_secondary_pool_pointers(self):
        assert len(self._pool_pointers
                   ) >= 1, "pool pointers haven't been initiated yet"
        data_ptr_pointers = torch.tensor(list(
            map(lambda x: x.data_ptr(), self._pool_pointers)),
                                         dtype=torch.int64)
        host_kv_cache_pool_pointers = torch.cat(
            (data_ptr_pointers.view(-1, 1),
             torch.zeros(len(self._pool_pointers), 1, dtype=torch.int64)),
            dim=1)

        return host_kv_cache_pool_pointers

    @classmethod
    def prepare_num_kv_heads_per_layer(cls, kv_head, num_layers):
        return [kv_head] * num_layers

    @property
    def pools_metadata(self):
        return self._pools_metadata

    @property
    def pool_mapping(self):
        return self._pool_mapping
