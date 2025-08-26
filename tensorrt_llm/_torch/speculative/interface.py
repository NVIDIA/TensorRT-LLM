import copy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Optional, Type

import torch

from ..._utils import get_sm_version
from ..attention_backend.trtllm import AttentionBackend, TrtllmAttention


class SpeculativeDecodingMode(IntEnum):
    MTP = auto()
    MTP_EAGLE = auto()
    EAGLE3 = auto()
    EAGLE3_ONE_MODEL = auto()
    NGRAM = auto()
    DRAFT_TARGET = auto()
    USER_PROVIDED = auto()
    NONE = auto()
    AUTO = auto()

    def is_mtp(self):
        return self == SpeculativeDecodingMode.MTP or self == SpeculativeDecodingMode.MTP_EAGLE

    def is_mtp_vanilla(self):
        return self == SpeculativeDecodingMode.MTP

    def is_mtp_eagle(self):
        return self == SpeculativeDecodingMode.MTP_EAGLE

    def is_eagle3(self):
        return self == SpeculativeDecodingMode.EAGLE3

    def use_one_engine(self):
        return self.is_mtp() or self.is_eagle3_one_model()

    def is_eagle3_one_model(self):
        return self == SpeculativeDecodingMode.EAGLE3_ONE_MODEL

    def is_ngram(self):
        return self == SpeculativeDecodingMode.NGRAM

    def is_user_provided(self):
        return self == SpeculativeDecodingMode.USER_PROVIDED

    def is_none(self):
        return self == SpeculativeDecodingMode.NONE

    def is_draft_target(self):
        return self == SpeculativeDecodingMode.DRAFT_TARGET

    def without_logits(self):
        return self.is_mtp() or self.is_eagle3_one_model()

    def needs_kv_cache_rewind(self):
        return self.is_mtp() or self.is_eagle3_one_model() or self.is_ngram()

    def support_overlap_scheduler(self):
        return self.is_mtp() or self.is_eagle3_one_model()

    def support_guided_decoder(self):
        return self.is_none() or self.has_spec_drafter()

    def support_capturable_guided_decoder(self):
        return self.is_mtp() or self.is_eagle3_one_model()

    def has_draft_model(self):
        return self.is_eagle3() or self.is_draft_target()

    def needs_kv_cache_recompute(self):
        """
        Whether the draft model needs to recompute the kv cache.
        If true, the 1st draft model forward will recompute the kv cache for
        the accepted draft tokens.
        """
        return self.is_eagle3()

    def need_load_draft_weights(self):
        """
        Whether the draft model and target model are in the same model engine,
        and the draft model needs to load weights from the separate checkpoint.
        """
        return self.is_eagle3_one_model()

    def has_spec_decoder(self):
        return self.is_mtp() or self.is_eagle3() or self.is_eagle3_one_model()

    def has_spec_drafter(self):
        return self.is_eagle3() or self.is_draft_target() or self.is_ngram(
        ) or self.is_user_provided()

    def extend_ctx(self, attention_backend: Type[AttentionBackend]):
        """
        If true, treat generation requests with draft tokens as
        chunked context requests at the kernel level. Required for
        any spec dec mode that uses the SpecExecutor.
        """

        if self.use_one_engine():
            # 1-model has separate logic for handling draft tokens
            return False

        # The special XQA generation kernels only exist with the TRTLLM backend on blackwell.
        return not issubclass(attention_backend,
                              TrtllmAttention) or get_sm_version() != 100

    def attention_need_spec_dec_mode(self):
        """
        If true, the attention backend kernel needs to run in spec-dec mode (multi-token query mode).
        """
        return self.is_eagle3_one_model()

    @staticmethod
    def from_string(name: Optional[str]) -> "SpeculativeDecodingMode":
        if name is None:
            return SpeculativeDecodingMode.NONE
        return SpeculativeDecodingMode[name.upper()]


@dataclass
class SpecMetadata:
    """
    Metadata for speculative decoding.
    """
    # The max number of requests in a single batch.
    max_num_requests: int
    # The max number of draft tokens.
    max_draft_len: int
    # The number of gen-phase sequences in the batch.
    num_generations: int = 0
    # Whether CUDA graph is enabled.
    is_cuda_graph: bool = field(default=False, repr=False)
    # The mode of speculative decoding.
    spec_dec_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE
    # Draft tokens.
    draft_tokens: Optional[torch.Tensor] = None
    # The length of the draft tokens.
    draft_lens: Optional[torch.Tensor] = None
    # The request ID of each sequence in the batch.
    # The shape is (batch_size).
    request_ids: Optional[List[int]] = None
    # Sequence length for each request.
    seq_lens: Optional[List[int]] = None
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
    # to handle this issue.
    num_extra_kv_tokens: Optional[int] = 0  # Number of layers in target model
    # The number of layers
    num_layers: int = 0

    # if spec-dec tree is a tree or a chain (linear tree)
    is_spec_dec_tree: bool = False
    # if spec-dec tree wouldn't be changed at all, the mask won't be computed every step.
    is_spec_dec_dynamic_tree: bool = False

    def __post_init__(self):
        pass

    def prepare(self):
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

    def is_layer_capture(self, layer_id: int):
        """
        Whether the layer should be captured (eg for Eagle3).
        By default, does nothing.
        """
        return False

    def maybe_capture_hidden_states(self, layer_id: int,
                                    hidden_states: torch.Tensor,
                                    residual: torch.Tensor) -> None:
        """
        Some spec decode algorithms require hidden states from the target
        model. Use this method to record them. By default, does nothing.
        """
