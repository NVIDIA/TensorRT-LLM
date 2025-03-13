import copy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Optional

import torch

from ..model_config import TConfig


class SpeculativeDecodingMode(IntEnum):
    MTP = auto()
    MTP_EAGLE = auto()
    MEDUSA = auto()
    EAGLE = auto()
    LOOKAHEAD = auto()
    NONE = auto()

    def is_mtp(self):
        return self == SpeculativeDecodingMode.MTP or SpeculativeDecodingMode.MTP_EAGLE

    def is_mtp_eagle(self):
        return self == SpeculativeDecodingMode.MTP_EAGLE

    def is_medusa(self):
        return self == SpeculativeDecodingMode.MEDUSA

    def is_eagle(self):
        return self == SpeculativeDecodingMode.EAGLE

    def is_lookahead(self):
        return self == SpeculativeDecodingMode.LOOKAHEAD

    def is_none(self):
        return self == SpeculativeDecodingMode.NONE

    def without_logits(self):
        return self.is_mtp()

    def needs_kv_cache_rewind(self):
        return self.is_mtp() or self.is_eagle() or self.is_lookahead(
        ) or self.is_medusa()

    def support_overlap_scheduler(self):
        return self.is_mtp()

    @staticmethod
    def from_string(name: str):
        name_map = {
            "MTP": SpeculativeDecodingMode.MTP,
            "MEDUSA": SpeculativeDecodingMode.MEDUSA,
            "EAGLE": SpeculativeDecodingMode.EAGLE,
            "LOOKAHEAD": SpeculativeDecodingMode.LOOKAHEAD,
            None: SpeculativeDecodingMode.NONE,
        }
        return name_map[name]


@dataclass
class SpecConfig:
    """
    Configuration for speculative decoding.
    """
    # The name of speculative decoding.
    spec_dec_name = None
    # The mode of speculative decoding.
    spec_dec_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE
    # The max number of draft tokens
    max_draft_tokens: int = 1024

    def __post_init__(self) -> None:
        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)

    def update_from_model_config(self, model_config: TConfig):
        pass


@dataclass
class SpecMetadata:
    """
    Metadata for speculative decoding.
    """
    # The max number of requests in a single batch.
    max_num_requests: int
    # The max number of draft tokens.
    max_draft_tokens: int
    # The number of gen-phase sequences in the batch.
    num_generations: int = 0
    # Whether CUDA graph is enabled.
    is_cuda_graph: bool = field(default=False, repr=False)
    # The mode of speculative decoding.
    spec_dec_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE,
    # Draft tokens.
    draft_tokens: Optional[torch.Tensor] = None,
    # The length of the draft tokens.
    draft_lens: Optional[torch.Tensor] = None,
    # The request ID of each sequence in the batch.
    # The shape is (batch_size).
    request_ids: Optional[List[int]] = None
    # The gather ids for logits.
    gather_ids: Optional[torch.Tensor] = None
    # The number of tokens for speculative model/layer
    num_tokens: int = 0
    # The number of tokens for speculative model/layer of different rank
    all_rank_num_tokens: Optional[List[int]] = None
    # The number of sequences for speculative model/layer of different rank
    all_rank_num_seqs: Optional[List[int]] = None
    # The number of extra kv tokens
    # Some speculative decoding methods need to use different kv lengths for the
    # draft/target layers. But KVCacheManager can only support kv caches with the
    # same kv lengths for different layers. Add extra kv token in kv cache manager
    # to haddle this issue.
    num_extra_kv_tokens: Optional[int] = 0

    def prepare():
        """
        Hook to be called before the forward step of the model.
        """

    def create_cuda_graph_metadata(self, max_batch_size: int):
        """
        Creates metadata for CUDA graph execution.
        """
        if self.is_cuda_graph:
            return self

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.max_num_requests = max_batch_size
        cuda_graph_metadata.__post_init__()
        return cuda_graph_metadata
