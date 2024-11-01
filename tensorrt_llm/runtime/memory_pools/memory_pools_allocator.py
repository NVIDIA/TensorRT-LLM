from collections import Counter
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
        self._num_kv_heads_per_layer = num_kv_heads_per_layer

        if isinstance(dtype, str):
            dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
        kv_heads_unique_counter = Counter(self._num_kv_heads_per_layer)
        keys_to_indices = {}

        for idx, (kv_head,
                  num_layers) in enumerate(kv_heads_unique_counter.items()):
            keys_to_indices[kv_head] = idx
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

        self._set_layers_mapping(keys_to_indices)

    def get_kv_cache_pool_pointers(self):
        return self._get_primarmy_secondary_pool_pointers()

    def _set_layers_mapping(self, keys_to_indices):
        layers_mapping = []
        for kv_size in self._num_kv_heads_per_layer:
            layers_mapping.append(keys_to_indices[kv_size])

        self._pool_mapping = torch.tensor(layers_mapping, dtype=torch.int32)

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
