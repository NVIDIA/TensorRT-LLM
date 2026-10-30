# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal Python request inputs and their transport/child ownership policies."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tensorrt_llm.bindings.executor import Request
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.sampling_params import LogitsProcessor, LogprobMode
from tensorrt_llm.scheduling_params import SchedulingParams


@dataclass(slots=True)
class PythonRequestInputs:
    """Picklable input data only; runtime events and cache owners stay on LlmRequest.

    The multimodal dictionary contains model-specific tensors, handles and item
    metadata, never an admitted MultimodalParams aggregate. Transport uses the
    normal pickle protocol, including the transport's tensor reducers.
    """

    logits_post_processors: list[LogitsProcessor] | None = None
    multimodal_data: dict[str, Any] | None = None
    scheduling_params: SchedulingParams | None = None
    num_logprobs: int = 0
    disaggregated_params: DisaggregatedParams | None = None
    lora_path: str | None = None
    lora_ckpt_source: str | None = None
    logprobs_mode: LogprobMode = LogprobMode.RAW
    logprobs_simple_format: bool = False
    return_routed_experts: bool = False
    arrival_time: float | None = None
    original_end_id: int | None = None
    conversation_params: ConversationParams | None = None
    uncached_prefill_tokens: int | None = None

    @classmethod
    def from_native_request(cls, request: Request) -> "PythonRequestInputs":
        """Adapt directly constructed bound requests, including legacy attributes.

        Resolve an absent original EOS to the native EOS here so an explicit
        None continues to mean that the caller supplied no original EOS.
        """
        return cls(
            logits_post_processors=getattr(request, "py_logits_post_processors", None),
            multimodal_data=getattr(request, "py_multimodal_data", None),
            scheduling_params=getattr(request, "py_scheduling_params", None),
            num_logprobs=getattr(request, "py_num_logprobs", 0),
            disaggregated_params=getattr(request, "py_disaggregated_params", None),
            lora_path=getattr(request, "py_lora_path", None),
            lora_ckpt_source=getattr(request, "py_lora_ckpt_source", None),
            logprobs_mode=getattr(request, "py_logprobs_mode", LogprobMode.RAW),
            logprobs_simple_format=getattr(request, "py_logprobs_simple_format", False),
            return_routed_experts=getattr(request, "py_return_routed_experts", False),
            arrival_time=getattr(request, "py_arrival_time", None),
            original_end_id=getattr(request, "py_original_end_id", request.end_id),
            conversation_params=getattr(request, "py_conversation_params", None),
            uncached_prefill_tokens=getattr(request, "py_uncached_prefill_tokens", None),
        )

    def copy_for_child(self) -> "PythonRequestInputs":
        """Copy mutable inputs; preserve user-owned processor callable identity."""
        return PythonRequestInputs(
            logits_post_processors=(
                self.logits_post_processors.copy()
                if self.logits_post_processors is not None
                else None
            ),
            multimodal_data=deepcopy(self.multimodal_data),
            scheduling_params=deepcopy(self.scheduling_params),
            num_logprobs=self.num_logprobs,
            disaggregated_params=deepcopy(self.disaggregated_params),
            lora_path=self.lora_path,
            lora_ckpt_source=self.lora_ckpt_source,
            logprobs_mode=self.logprobs_mode,
            logprobs_simple_format=self.logprobs_simple_format,
            return_routed_experts=self.return_routed_experts,
            arrival_time=self.arrival_time,
            original_end_id=self.original_end_id,
            conversation_params=deepcopy(self.conversation_params),
            uncached_prefill_tokens=self.uncached_prefill_tokens,
        )


@dataclass(slots=True)
class ExecutorRequest:
    """Serialize a bound request and its Python inputs as one internal envelope.

    Native reads delegate to the bound request. The four compatibility views
    below serve admission and routing consumers that also accept LlmRequest.
    """

    request: Request
    inputs: PythonRequestInputs

    @classmethod
    def adapt(cls, request: "ExecutorRequest | Request") -> "ExecutorRequest":
        """Capture legacy Python attributes before a bound request is pickled."""
        if isinstance(request, cls):
            return request
        return cls(request, PythonRequestInputs.from_native_request(request))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "request"), name)

    def __reduce__(self) -> tuple:
        # Include both owners explicitly; the bound request's reducer omits its
        # dynamic Python attributes. Never serialize execution state here.
        return type(self), (self.request, self.inputs)

    @property
    def py_scheduling_params(self) -> SchedulingParams | None:
        return self.inputs.scheduling_params

    @property
    def py_disaggregated_params(self) -> DisaggregatedParams | None:
        return self.inputs.disaggregated_params

    @property
    def py_multimodal_data(self) -> dict[str, Any] | None:
        return self.inputs.multimodal_data

    @property
    def py_uncached_prefill_tokens(self) -> int | None:
        return self.inputs.uncached_prefill_tokens

    @py_uncached_prefill_tokens.setter
    def py_uncached_prefill_tokens(self, value: int | None) -> None:
        self.inputs.uncached_prefill_tokens = value
