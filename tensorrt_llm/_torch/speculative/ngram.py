from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from tensorrt_llm.bindings.executor import FinishReason

from ..attention_backend import AttentionMetadata
from ..pyexecutor.decoder import DecoderState, TorchDecoder
from ..pyexecutor.llm_request import LlmRequest, LlmRequestState
from ..pyexecutor.resource_manager import BaseResourceManager, SlotManager
from ..pyexecutor.scheduler import ScheduledRequests
from .interface import SpecConfig, SpecMetadata, SpeculativeDecodingMode


@dataclass
class NGramConfig(SpecConfig):
    """
    Configuration for N-gram drafter.
    """
    # The name of speculative decoding.
    spec_dec_name = "NGRAM"

    num_extra_kv_tokens: int = 0
    max_draft_tokens: int = 0

    prompt_lookup_num_tokens: int = 5
    max_matching_ngram_size: int = 5
    end_id: int = -1
    is_keep_all: bool = True
    is_use_oldest: bool = True

    def __post_init__(self) -> None:
        self.spec_dec_mode = SpeculativeDecodingMode.from_string(
            self.spec_dec_name)
        self.max_draft_tokens = self.max_matching_ngram_size

    def update_from_model_config(self, model_config):
        pass


@dataclass
class NGramSpecMetadata(SpecMetadata):
    """
    Metadata for NGram.
    """
    def __post_init__(self) -> None:
        pass

    def prepare(self):
        pass


