import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..conversation_params import ConversationParams
from ..disaggregated_params import DisaggregatedParams
from ..llmapi.llm_utils import KvCacheRetentionConfig
from ..sampling_params import SamplingParams
from ..scheduling_params import SchedulingParams
from .postproc_worker import PostprocParams

__all__ = [
    "DEFAULT_REQUEST_PRIORITY",
    "LoRARequest",
    "PromptAdapterRequest",
    "GenerationRequest",
    "TruncateKVCacheRequest",
    "CancellingRequest",
]

# Mirrors C++ executor.h Request::kDefaultPriority
DEFAULT_REQUEST_PRIORITY: float = 0.5


@dataclass(slots=True)
class LoRARequest:
    """ Request for a LoRA adapter. """
    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    lora_ckpt_source: str = "hf"

    def __post_init__(self):
        if self.lora_path is not None and not os.path.exists(self.lora_path):
            raise ValueError(f"lora_path ({self.lora_path}) does not exist.")
        if self.lora_ckpt_source not in ["hf", "nemo"]:
            raise ValueError(
                f"lora_ckpt_source must be 'hf' or 'nemo', got '{self.lora_ckpt_source}'"
            )

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

    @property
    def ckpt_source(self):
        return self.lora_ckpt_source


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
        query_token_ids: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        postproc_params: Optional[PostprocParams] = None,
        multimodal_params: Optional[MultimodalParams] = None,
        scheduling_params: Optional[SchedulingParams] = None,
        conversation_params: Optional[ConversationParams] = None,
        cache_salt_id: Optional[int] = None,
        arrival_time: Optional[float] = None,
        priority: float = DEFAULT_REQUEST_PRIORITY,
    ):
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

        # NOTE: Exercise caution when adding memory intense attributes, because the current implementation might lead to leaks without manual cleanup.
        # Refer to https://github.com/NVIDIA/TensorRT-LLM/pull/5029#discussion_r2141859873 for details.
        self.sampling_params = sampling_params
        self.postproc_params = postproc_params
        self.lora_request = lora_request
        self.prompt_adapter_request = prompt_adapter_request
        self.streaming = streaming
        self.multimodal_params = multimodal_params
        self.kv_cache_retention_config = kv_cache_retention_config
        self.id: Optional[int] = None
        self.disaggregated_params = disaggregated_params
        self.trace_headers = trace_headers
        self.scheduling_params = scheduling_params
        self.conversation_params = conversation_params
        self.cache_salt_id = cache_salt_id
        self.arrival_time = arrival_time
        if not (0.0 <= priority <= 1.0):
            raise ValueError(
                f"priority must be a float in [0.0, 1.0], got {priority}")
        self.priority = priority

    def set_id(self, id):
        assert self.id is None, f"Request ID is already set: {self.id}"
        self.id = id
        return self

    # --- int32 token-id wire serialization + LAZY list materialization ----------
    # Pickling a flat list[int] on the hot proxy->worker RPC-submit path emits one
    # PyLong frame per token (O(ISL)). Encode token-ids as int32 bytes for the wire.
    # On decode we DO NOT eagerly rebuild the list: stash the int32 ndarray in
    # `_prompt_token_ids_i32` (the C++ Request ctor memcpy's it -- see
    # base_worker._enqueue_request) and leave the backing `_prompt_token_ids` None.
    # The list is built lazily (and cached) by the `prompt_token_ids` property below
    # only if a consumer actually reads it (e.g. prompt-logprobs, star-attention).
    # Plain decode never reads it -> the O(ISL) `.tolist()` never runs.
    #
    # NOTE: implemented as a *property* (scoped to this one name), NOT a class-level
    # __getattr__. A __getattr__ is invoked by the interpreter on EVERY missing
    # attribute access on the object -- hasattr()/getattr(default) probes, copy/
    # pickle dunder lookups, duck-typing -- each paying a Python frame + an
    # AttributeError raise. A property has identical lazy-materialize behavior with
    # zero blast radius on any other attribute.
    _I32 = "\x00i32be"

    @property
    def prompt_token_ids(self):
        ptids = self.__dict__.get("_prompt_token_ids")
        if ptids is None:
            buf = self.__dict__.get("_prompt_token_ids_i32")
            if buf is not None:
                ptids = buf.tolist()
                self._prompt_token_ids = ptids  # cache
        return ptids

    @prompt_token_ids.setter
    def prompt_token_ids(self, value):
        self._prompt_token_ids = value

    @staticmethod
    def _enc_tokens(v):
        # flat list[int] -> (_I32, int32 bytes); leave None / list[list[int]] /
        # ndarray untouched.
        if type(v) is list and (len(v) == 0 or type(v[0]) is int):
            return (GenerationRequest._I32,
                    np.asarray(v, dtype=np.int32).tobytes())
        return v

    def __getstate__(self):
        state = self.__dict__.copy()
        buf = state.pop("_prompt_token_ids_i32", None)
        ptids = state.get("_prompt_token_ids")
        if ptids is not None:
            state["_prompt_token_ids"] = GenerationRequest._enc_tokens(ptids)
        elif buf is not None:
            # not yet materialized -> encode the buffer's bytes directly
            state["_prompt_token_ids"] = (GenerationRequest._I32, buf.tobytes())
        if state.get("query_token_ids") is not None:
            state["query_token_ids"] = GenerationRequest._enc_tokens(
                state["query_token_ids"])
        return state

    def __setstate__(self, state):
        buf = None
        pt = state.get("_prompt_token_ids")
        if type(pt) is tuple and len(
                pt) == 2 and pt[0] == GenerationRequest._I32:
            buf = np.frombuffer(pt[1], dtype=np.int32)
            state["_prompt_token_ids"] = None  # leave None -> lazy via property
        qt = state.get("query_token_ids")
        if type(qt) is tuple and len(
                qt) == 2 and qt[0] == GenerationRequest._I32:
            # query_token_ids is rare/small -> materialize to list eagerly
            state["query_token_ids"] = np.frombuffer(qt[1],
                                                     dtype=np.int32).tolist()
        self.__dict__.update(state)
        self._prompt_token_ids_i32 = buf


class TruncateKVCacheRequest:

    def __init__(self, messages_to_retain: List[int], messages: List[int]):
        self.messages_to_retain = messages_to_retain
        self.messages = messages


class CancellingRequest:
    ''' The request to cancel a generation. '''

    def __init__(self, id: int):
        self.id = id
