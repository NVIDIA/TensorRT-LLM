import copy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Type

import torch

from ..._utils import get_sm_version
from ..attention_backend.trtllm import AttentionBackend, TrtllmAttention
from ..model_config import TConfig
from ..pyexecutor.scheduler import ScheduledRequests


class SpeculativeDecodingMode(IntEnum):
    MTP = auto()
    MTP_EAGLE = auto()
    EAGLE3 = auto()
    EAGLE3_ONE_MODEL = auto()
    NGRAM = auto()
    NONE = auto()

    def is_mtp(self):
        return self == SpeculativeDecodingMode.MTP or self == SpeculativeDecodingMode.MTP_EAGLE

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

    def is_none(self):
        return self == SpeculativeDecodingMode.NONE

    def without_logits(self):
        return self.is_mtp() or self.is_eagle3_one_model()

    def needs_kv_cache_rewind(self):
        return self.is_mtp() or self.is_eagle3_one_model()

    def support_overlap_scheduler(self):
        return self.is_mtp() or self.is_eagle3_one_model()

    def has_draft_model(self):
        return self.is_eagle3()

    def need_load_draft_weights(self):
        """
        Whether the draft model and target model are in the same model engine,
        and the draft model needs to load weights from the separate checkpoint.
        """
        return self.is_eagle3_one_model()

    def has_spec_decoder(self):
        return self.is_mtp() or self.is_eagle3() or self.is_eagle3_one_model()

    def extend_ctx(self, attention_backend: Type[AttentionBackend]):
        """
        If true, treat generation requests with draft tokens as
        chunked context requests at the kernel level. Required for
        any spec dec mode that uses the SpecExecutor.
        """

        # Fixme: only trtllm attention backend supports eagle3 generation-phase kernels on blackwell.
        return (self.is_eagle3()
                and not (issubclass(attention_backend, TrtllmAttention)
                         and get_sm_version() == 100)) or self.is_ngram()

    @staticmethod
    def from_string(name: Optional[str]) -> "SpeculativeDecodingMode":
        if name is None:
            return SpeculativeDecodingMode.NONE
        return SpeculativeDecodingMode[name.upper()]


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
    # The path to the draft model
    draft_model_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)

    def update_from_model_config(self, model_config: TConfig):
        pass

    def get_draft_model_prompt(self,
                               input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Override for spec dec modes that need to preprocess prompt
        tokens before passing them to the draft model.
        """
        return input_tokens


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
    # to haddle this issue.
    num_extra_kv_tokens: Optional[int] = 0  # Number of layers in target model
    num_layers: int = 0

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

    def maybe_capture_hidden_states(self, layer_id: int,
                                    hidden_states: torch.Tensor,
                                    residual: torch.Tensor) -> None:
        """
        Some spec decode algorithms require hidden states from the target
        model. Use this method to record them. By default, does nothing.
        """

    def get_hidden_states(
            self,
            scheduled_requests: ScheduledRequests,
            num_rejected_tokens: Optional[Dict] = None) -> List[torch.Tensor]:
        """
        Return any captured hidden states. Should do any necessary
        pre-processing.

        num_rejected_tokens is a dictionary mapping request IDs to the
        number of tokens rejected for that request. If a request ID isn't
        in the dictionary, it means that the request is not needed for drafting.

        If the dictionary is not given, this function assumes that the hidden
        states are being prepared for running the draft model autoregressively,
        and only the last hidden state vector for each sequence is returned.
        """
        return []
