import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from ..llmapi.llm_utils import KvCacheRetentionConfig
from ..sampling_params import SamplingParams

__all__ = [
    "LoRARequest",
    "PromptAdapterRequest",
    "GenerationRequest",
]


@dataclass(slots=True)
class LoRARequest:
    """ Request for a LoRA adapter. """
    lora_name: str
    lora_int_id: int
    lora_path: str = ""

    def __post_init__(self):
        if not os.path.exists(self.lora_path):
            raise ValueError(f"lora_path ({self.lora_path}) does not exist.")

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path


@dataclass(slots=True)
class PromptAdapterRequest:
    """
    Request for a Prompt adapter.
    """
    prompt_adapter_name: str
    prompt_adapter_id: int
    prompt_adapter_local_path: str = ""

    def __post_init__(self):
        if not os.path.exists(self.prompt_adapter_local_path):
            raise RuntimeError(
                f"prompt_adapter_local_path ({self.prompt_adapter_local_path}) does not exist."
            )

    @property
    def adapter_id(self):
        return self.prompt_adapter_id

    @property
    def name(self):
        return self.prompt_adapter_name

    @property
    def local_path(self):
        return self.prompt_adapter_local_path


class GenerationRequest:

    def __init__(
            self,
            prompt_token_ids: Union[torch.Tensor, np.ndarray,
                                    Union[List[int], List[List[int]]]],
            sampling_params: SamplingParams,
            query_token_ids: Optional[Union[torch.Tensor, np.ndarray,
                                            list]] = None,
            lora_request: Optional[LoRARequest] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            streaming: bool = False,
            prompt_tuning_config: Optional[list] = None,
            kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None):
        if isinstance(prompt_token_ids, list):
            self.prompt_token_ids = prompt_token_ids
            self.query_token_ids = query_token_ids
        elif isinstance(prompt_token_ids, (torch.Tensor, np.ndarray)):
            self.prompt_token_ids = prompt_token_ids.tolist()
            if query_token_ids:
                self.query_token_ids = query_token_ids.tolist()
        else:
            raise TypeError(
                f"prompt_token_ids ({prompt_token_ids}) should be an instance of torch.Tensor, np.ndarray or list"
            )

        self.sampling_params = sampling_params
        self.lora_request = lora_request
        self.prompt_adapter_request = prompt_adapter_request
        self.streaming = streaming
        self.prompt_tuning_config = prompt_tuning_config
        self.kv_cache_retention_config = kv_cache_retention_config
        self.id: Optional[int] = None

    def set_id(self, id):
        assert self.id is None, f"Request ID is already set: {self.id}"
        self.id = id
        return self
