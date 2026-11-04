# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine operations the serving frontends share.

The HTTP server (tensorrt_llm/serve), the SMG gRPC frontend and the OpenEngine
gRPC frontend each hold an in-process LLM and use the same core: generate,
health and load. EngineClient writes that core down; LLM satisfies it.
Operations only one role uses stay on LLM / AsyncLLM: encode() and
encode_limits, iteration stats, runtime memory control, RL collective_rpc,
profiling, multi_frontend_attach_info. Nothing checks the protocol at run time;
tests/unittest/serve/test_private_access_lint.py keeps the frontends off
private members.

OpenEngine v1 alignment:
    Inference.Generate             -> generate_async
    Control.Abort                  -> RequestOutput.abort() (no engine-level abort here)
    Control.Health                 -> check_health, fatal_error, shutdown_started
    Control.GetLoad                -> get_kv_cache_load, get_kv_cache_capacity
    GetServerInfo / GetModelInfo   -> not here (LLM has no configuration snapshot method)
A transport-level client implementing this protocol adds failure modes the
in-process LLM does not have (timeouts, lost connections); its errors must be
dialect-neutral, so that each wire dialect (OpenAI, Anthropic) renders them in
its own format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, Union

if TYPE_CHECKING:
    from tensorrt_llm.conversation_params import ConversationParams
    from tensorrt_llm.disaggregated_params import DisaggregatedParams
    from tensorrt_llm.executor.request import LoRARequest, PromptAdapterRequest
    from tensorrt_llm.inputs.data import PromptInputs
    from tensorrt_llm.llmapi.llm import PreprocessedInputs, RequestOutput
    from tensorrt_llm.llmapi.llm_utils import KvCacheRetentionConfig
    from tensorrt_llm.sampling_params import SamplingParams
    from tensorrt_llm.scheduling_params import SchedulingParams


class EngineClient(Protocol):
    """The generate / health / load surface every serving frontend uses."""

    def generate_async(
        self,
        inputs: Union[PromptInputs, PreprocessedInputs],
        sampling_params: Optional[SamplingParams] = None,
        *,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        scheduling_params: Optional[SchedulingParams] = None,
        conversation_params: Optional[ConversationParams] = None,
        cache_salt: Optional[str] = None,
        priority: float = ...,
    ) -> RequestOutput: ...

    def check_health(self) -> bool: ...

    @property
    def fatal_error(self) -> Optional[BaseException]: ...

    @property
    def shutdown_started(self) -> bool: ...

    def get_kv_cache_load(self) -> dict[str, Any]: ...

    def get_kv_cache_capacity(self) -> dict[str, Any]: ...
