from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch


@dataclass
class KVCacheParams:
    """
    Parameters for the key-value cache.
    """
    # Whether to use the cache or not.
    use_cache: bool

    # The number of the cached tokens of each sequence
    num_cached_tokens_per_seq: Optional[List[int]] = None
    # Block IDs of the each sequence
    # The shape is depending on the cache type:
    # - LINEAR: (1)
    # - PAGED: (num_pages)
    # - PER_TOKEN: (num_tokens)
    # The dtype is int64.
    block_ids_per_seq: Optional[List[list]] = None

    # The maximum attention window size for each layer.
    host_max_attention_window_sizes: Optional[torch.Tensor] = None
    # The number of sink tokens for each layer.
    host_sink_token_length: Optional[torch.Tensor] = None
    # The number of extra kv for draft tokens
    num_extra_kv_tokens: Optional[int] = 0


class CacheType(Enum):
    # Linear KV cache stores all the cached tokens of a sequence in a single page.
    LINEAR = 0
    # Paged KV cache stores the cached tokens of a sequence in multiple pages.
    PAGED = 1
    # Per-token KV cache stores each token's cached value separately.
    PER_TOKEN = 2
