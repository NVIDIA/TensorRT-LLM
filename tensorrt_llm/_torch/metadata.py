from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm.sampling_params import LogitsProcessor


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
    num_extra_kv_tokens: Optional[List[int]] = 0


class CacheType(Enum):
    # Linear KV cache stores all the cached tokens of a sequence in a single page.
    LINEAR = 0
    # Paged KV cache stores the cached tokens of a sequence in multiple pages.
    PAGED = 1
    # Per-token KV cache stores each token's cached value separately.
    PER_TOKEN = 2


@dataclass(kw_only=True)
class SequenceGroup:
    logits_processors: list[LogitsProcessor]
    batch_indices: Union[List[int], torch.Tensor] = field(default_factory=list)
    request_ids: Union[List[int], torch.Tensor] = field(default_factory=list)
    stream_ptr: Optional[int] = None
    client_id: Optional[int] = None


@dataclass(kw_only=True)
class LogitsProcessorMetadata:
    seq_groups: List[SequenceGroup] = field(default_factory=list)

    # request_id -> token ids produced by the request so far, with shape beam_width * sequence_length.
    past_token_ids_dict: Dict[int,
                              torch.LongTensor] = field(default_factory=dict)

    def update_per_request(self, request_id: int, logits_processors,
                           batch_idx: int, past_token_ids):

        assert request_id not in self.past_token_ids_dict  # TODO: can remove
        self.past_token_ids_dict[request_id] = past_token_ids

        for group in self.seq_groups:
            if logits_processors == group.logits_processors:
                assert isinstance(group.batch_indices,
                                  list), "batch indices must be a list"
                group.batch_indices.append(batch_idx)
                group.request_ids.append(request_id)
                return

        new_group = SequenceGroup(
            logits_processors=logits_processors,
            batch_indices=[batch_idx],
            request_ids=[request_id],
        )
        self.seq_groups.append(new_group)

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """
        for seq_group in self.seq_groups:
            # assert isinstance(group.batch_indices, list)
            seq_group.batch_indices = torch.tensor(seq_group.batch_indices,
                                                   dtype=torch.int32,
                                                   device='cuda')
            seq_group.request_ids = torch.tensor(seq_group.request_ids,
                                                 dtype=torch.int64,
                                                 device='cuda')
