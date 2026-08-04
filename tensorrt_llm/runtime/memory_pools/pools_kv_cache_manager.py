from typing import List

import torch

from tensorrt_llm.runtime.kv_cache_manager import (GenerationSequence,
                                                   KVCacheManager)
from tensorrt_llm.runtime.memory_pools.pool import Pool


class PoolsKVCacheManager(object):

    def __init__(self, pools_metadata: List[Pool], max_blocks_per_seq,
                 num_blocks, tokens_per_block, head_size,
                 max_attention_window_size, beam_width, sink_token_len) -> None:
        self._num_pools = len(pools_metadata)
        self._kv_cache_managers = []

        for pool in pools_metadata:
            block_size = pool.num_kv_heads * tokens_per_block * head_size
            self._kv_cache_managers.append(
                KVCacheManager(
                    num_layers=pool.num_layers,
                    num_blocks=num_blocks,
                    block_size=block_size,
                    tokens_per_block=tokens_per_block,
                    max_blocks_per_seq=max_blocks_per_seq,
                    max_attention_window_size=max_attention_window_size,
                    sink_token_len=sink_token_len,
                    beam_width=beam_width,
                ))

    def add_sequence(self,
                     sequence: GenerationSequence,
                     context_len: int,
                     always_share_across_beam: bool = False):
        for kv_cache_manager in self._kv_cache_managers:
            kv_cache_manager.add_sequence(sequence, context_len,
                                          always_share_across_beam)

    def step(self, finished: List[bool]):
        for kv_cache_manager in self._kv_cache_managers:
            kv_cache_manager.step(finished)

    def get_block_offsets(self, beam_width: int) -> torch.Tensor:
        offsets = []
        for kv_cache_manager in self._kv_cache_managers:
            block_offset = kv_cache_manager.get_block_offsets(beam_width)
            offsets.append(block_offset)

        return torch.stack(offsets)

    def get_single_kv_cache_manager(self):
        assert len(self._kv_cache_managers
                   ) == 1, f"More then one kv cache manager exists"

        return self._kv_cache_managers[0]

    def has_single_pool(self):
        return len(self._kv_cache_managers) == 1
