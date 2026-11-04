# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI-dialect request and response models, one submodule per API.

New code imports the submodule that defines a name, for example
``from tensorrt_llm.serve.openai_protocol.chat import ChatCompletionRequest``.
This package re-exports, on first access, every name the former single-file
module ``openai_protocol.py`` had, so existing imports keep working. Do not add
names here; tests/unittest/serve/openai_protocol/test_openai_protocol_facade.py
fails if the table grows.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Public names, so type checkers and IDEs resolve them; private names are
    # reachable at run time through __getattr__. The VisualGen names are left
    # out on purpose: this file names their module only as a string in
    # _NAME_TO_MODULE (checked by the facade test under
    # tests/unittest/serve/api/visual_gen/).
    from .chat import (
        ChatCompletionContentPartParam,
        ChatCompletionLogProb,
        ChatCompletionLogProbs,
        ChatCompletionLogProbsContent,
        ChatCompletionMessageParam,
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
        ChatCompletionThinkingParam,
        ChatMessage,
        CustomChatCompletionContentPartParam,
        CustomChatCompletionMessageParam,
        DeltaMessage,
        DynamicToolsSystemMessageParam,
        ReasoningAssistantMessage,
    )
    from .common import (
        ConversationParams,
        DisaggregatedParams,
        DisaggScheduleStyle,
        EarlyStopping,
        ErrorResponse,
        OpenAIBaseModel,
        PromptTokensDetails,
        ResponseFormat,
        SpeculativeDecodingStats,
        StreamOptions,
        UsageInfo,
        ensure_request_chat_template_allowed,
    )
    from .completions import (
        CompletionLogProbs,
        CompletionRequest,
        CompletionResponse,
        CompletionResponseChoice,
        CompletionResponseStreamChoice,
        CompletionStreamResponse,
    )
    from .control import (
        MemoryUpdateRequest,
        ProfileActivity,
        RuntimeMemoryUpdateRequest,
        StartProfileRequest,
        UpdateWeightsRequest,
    )
    from .disagg import (
        UCompletionRequest,
        UCompletionResponse,
        decode_opaque_state,
        encode_opaque_state,
        to_disaggregated_params,
        to_llm_conversation_params,
        to_llm_disaggregated_params,
    )
    from .embeddings import (
        EmbeddingRequest,
        EmbeddingResponse,
        EmbeddingResponseData,
        EmbeddingUsageInfo,
    )
    from .kv_cache import KVCacheTruncateRequest, KVCacheTruncateTokensRequest
    from .models import ModelCard, ModelList
    from .responses import (
        InputTokensDetails,
        OutputTokensDetails,
        ResponseInputOutputItem,
        ResponsesRequest,
        ResponsesResponse,
        ResponsesStreamResponse,
        ResponseUsage,
        StreamingResponsesResponse,
    )
    from .tokenize import TokenizeRequest, TokenizeResponse
    from .tools import (
        TOOL_PARAM_MAX_ENUM_VALUES,
        ChatCompletionNamedFunction,
        ChatCompletionNamedToolChoiceParam,
        ChatCompletionToolsParam,
        DeltaFunctionCall,
        DeltaToolCall,
        FunctionCall,
        FunctionDefinition,
        NamespacedFunctionToolParam,
        ToolCall,
    )

# Name -> module that provides it.
_NAME_TO_MODULE = {
    "ChatCompletionContentPartParam": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionLogProb": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionLogProbs": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionLogProbsContent": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionMessageParam": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionNamedFunction": "tensorrt_llm.serve.openai_protocol.tools",
    "ChatCompletionNamedToolChoiceParam": "tensorrt_llm.serve.openai_protocol.tools",
    "ChatCompletionRequest": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionResponse": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionResponseChoice": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionResponseStreamChoice": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionStreamResponse": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionThinkingParam": "tensorrt_llm.serve.openai_protocol.chat",
    "ChatCompletionToolsParam": "tensorrt_llm.serve.openai_protocol.tools",
    "ChatMessage": "tensorrt_llm.serve.openai_protocol.chat",
    "CompletionLogProbs": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionRequest": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionResponse": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionResponseChoice": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionResponseStreamChoice": "tensorrt_llm.serve.openai_protocol.completions",
    "CompletionStreamResponse": "tensorrt_llm.serve.openai_protocol.completions",
    "ConversationParams": "tensorrt_llm.serve.openai_protocol.common",
    "CustomChatCompletionContentPartParam": "tensorrt_llm.serve.openai_protocol.chat",
    "CustomChatCompletionMessageParam": "tensorrt_llm.serve.openai_protocol.chat",
    "DeltaFunctionCall": "tensorrt_llm.serve.openai_protocol.tools",
    "DeltaMessage": "tensorrt_llm.serve.openai_protocol.chat",
    "DeltaToolCall": "tensorrt_llm.serve.openai_protocol.tools",
    "DisaggregatedParams": "tensorrt_llm.serve.openai_protocol.common",
    "DynamicToolsSystemMessageParam": "tensorrt_llm.serve.openai_protocol.chat",
    "EarlyStopping": "tensorrt_llm.serve.openai_protocol.common",
    "EmbeddingRequest": "tensorrt_llm.serve.openai_protocol.embeddings",
    "EmbeddingResponse": "tensorrt_llm.serve.openai_protocol.embeddings",
    "EmbeddingResponseData": "tensorrt_llm.serve.openai_protocol.embeddings",
    "EmbeddingUsageInfo": "tensorrt_llm.serve.openai_protocol.embeddings",
    "ErrorResponse": "tensorrt_llm.serve.openai_protocol.common",
    "FunctionCall": "tensorrt_llm.serve.openai_protocol.tools",
    "FunctionDefinition": "tensorrt_llm.serve.openai_protocol.tools",
    "InputTokensDetails": "tensorrt_llm.serve.openai_protocol.responses",
    "KVCacheTruncateRequest": "tensorrt_llm.serve.openai_protocol.kv_cache",
    "KVCacheTruncateTokensRequest": "tensorrt_llm.serve.openai_protocol.kv_cache",
    "MemoryUpdateRequest": "tensorrt_llm.serve.openai_protocol.control",
    "ModelCard": "tensorrt_llm.serve.openai_protocol.models",
    "ModelList": "tensorrt_llm.serve.openai_protocol.models",
    "NamespacedFunctionToolParam": "tensorrt_llm.serve.openai_protocol.tools",
    "OpenAIBaseModel": "tensorrt_llm.serve.openai_protocol.common",
    "OutputTokensDetails": "tensorrt_llm.serve.openai_protocol.responses",
    "ProfileActivity": "tensorrt_llm.serve.openai_protocol.control",
    "PromptTokensDetails": "tensorrt_llm.serve.openai_protocol.common",
    "ReasoningAssistantMessage": "tensorrt_llm.serve.openai_protocol.chat",
    "ResponseFormat": "tensorrt_llm.serve.openai_protocol.common",
    "ResponseInputOutputItem": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponseUsage": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponsesRequest": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponsesResponse": "tensorrt_llm.serve.openai_protocol.responses",
    "ResponsesStreamResponse": "tensorrt_llm.serve.openai_protocol.responses",
    "RuntimeMemoryUpdateRequest": "tensorrt_llm.serve.openai_protocol.control",
    "SpeculativeDecodingStats": "tensorrt_llm.serve.openai_protocol.common",
    "StartProfileRequest": "tensorrt_llm.serve.openai_protocol.control",
    "StreamOptions": "tensorrt_llm.serve.openai_protocol.common",
    "StreamingResponsesResponse": "tensorrt_llm.serve.openai_protocol.responses",
    "TOOL_PARAM_MAX_ENUM_VALUES": "tensorrt_llm.serve.openai_protocol.tools",
    "TokenizeRequest": "tensorrt_llm.serve.openai_protocol.tokenize",
    "TokenizeResponse": "tensorrt_llm.serve.openai_protocol.tokenize",
    "ToolCall": "tensorrt_llm.serve.openai_protocol.tools",
    "UCompletionRequest": "tensorrt_llm.serve.openai_protocol.disagg",
    "UCompletionResponse": "tensorrt_llm.serve.openai_protocol.disagg",
    "UpdateWeightsRequest": "tensorrt_llm.serve.openai_protocol.control",
    "UsageInfo": "tensorrt_llm.serve.openai_protocol.common",
    "_ID_STRIPPED_ROLES": "tensorrt_llm.serve.openai_protocol.responses",
    "_LOGIT_BIAS_MAX": "tensorrt_llm.serve.openai_protocol.sampling",
    "_LOGIT_BIAS_MIN": "tensorrt_llm.serve.openai_protocol.sampling",
    "_OmitsAbsentSpecDecodeStats": "tensorrt_llm.serve.openai_protocol.common",
    "_RESPONSES_HANDOFF_FIELDS": "tensorrt_llm.serve.openai_protocol.responses",
    "_SCHEMA_INSTANCE_KEYWORDS": "tensorrt_llm.serve.openai_protocol.tools",
    "_SCHEMA_MAP_KEYWORDS": "tensorrt_llm.serve.openai_protocol.tools",
    "_count_schema_enum_values": "tensorrt_llm.serve.openai_protocol.tools",
    "_deserialize_first_gen_log_probs": "tensorrt_llm.serve.openai_protocol.disagg",
    "_deserialize_first_gen_logits": "tensorrt_llm.serve.openai_protocol.disagg",
    "_drop_explicit_nulls": "tensorrt_llm.serve.openai_protocol.responses",
    "_early_stopping_to_int": "tensorrt_llm.serve.openai_protocol.sampling",
    "_logit_bias_to_embedding_bias": "tensorrt_llm.serve.openai_protocol.sampling",
    "_materialize_validator_iterators": "tensorrt_llm.serve.openai_protocol.responses",
    "_record_sampling_params_request_fields": "tensorrt_llm.serve.openai_protocol.sampling",
    "_response_format_text_config_to_guided_decoding_params": "tensorrt_llm.serve.openai_protocol.sampling",
    "_response_format_to_guided_decoding_params": "tensorrt_llm.serve.openai_protocol.sampling",
    "_serialize_first_gen_log_probs": "tensorrt_llm.serve.openai_protocol.disagg",
    "_serialize_first_gen_logits": "tensorrt_llm.serve.openai_protocol.disagg",
    "decode_opaque_state": "tensorrt_llm.serve.openai_protocol.disagg",
    "encode_opaque_state": "tensorrt_llm.serve.openai_protocol.disagg",
    "ensure_request_chat_template_allowed": "tensorrt_llm.serve.openai_protocol.common",
    "to_disaggregated_params": "tensorrt_llm.serve.openai_protocol.disagg",
    "to_llm_conversation_params": "tensorrt_llm.serve.openai_protocol.disagg",
    "to_llm_disaggregated_params": "tensorrt_llm.serve.openai_protocol.disagg",
    # Defined in the VisualGen serving package. Named here only as strings: an import,
    # even under TYPE_CHECKING, would load that package into every LLM process.
    "ImageEditRequest": "tensorrt_llm.serve.api.visual_gen.protocol",
    "ImageGenerationRequest": "tensorrt_llm.serve.api.visual_gen.protocol",
    "ImageGenerationResponse": "tensorrt_llm.serve.api.visual_gen.protocol",
    "ImageObject": "tensorrt_llm.serve.api.visual_gen.protocol",
    "MediaReferenceItem": "tensorrt_llm.serve.api.visual_gen.protocol",
    "VideoGenerationRequest": "tensorrt_llm.serve.api.visual_gen.protocol",
    "VideoJob": "tensorrt_llm.serve.api.visual_gen.protocol",
    "VideoJobList": "tensorrt_llm.serve.api.visual_gen.protocol",
    # Imported by the former module and imported from it by code in the repository.
    "DisaggScheduleStyle": "tensorrt_llm.serve.openai_protocol.common",
}

__all__ = [
    "ChatCompletionContentPartParam",
    "ChatCompletionLogProb",
    "ChatCompletionLogProbs",
    "ChatCompletionLogProbsContent",
    "ChatCompletionMessageParam",
    "ChatCompletionNamedFunction",
    "ChatCompletionNamedToolChoiceParam",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionResponseChoice",
    "ChatCompletionResponseStreamChoice",
    "ChatCompletionStreamResponse",
    "ChatCompletionThinkingParam",
    "ChatCompletionToolsParam",
    "ChatMessage",
    "CompletionLogProbs",
    "CompletionRequest",
    "CompletionResponse",
    "CompletionResponseChoice",
    "CompletionResponseStreamChoice",
    "CompletionStreamResponse",
    "ConversationParams",
    "CustomChatCompletionContentPartParam",
    "CustomChatCompletionMessageParam",
    "DeltaFunctionCall",
    "DeltaMessage",
    "DeltaToolCall",
    "DisaggScheduleStyle",
    "DisaggregatedParams",
    "DynamicToolsSystemMessageParam",
    "EarlyStopping",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingResponseData",
    "EmbeddingUsageInfo",
    "ErrorResponse",
    "FunctionCall",
    "FunctionDefinition",
    "ImageEditRequest",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageObject",
    "InputTokensDetails",
    "KVCacheTruncateRequest",
    "KVCacheTruncateTokensRequest",
    "MediaReferenceItem",
    "MemoryUpdateRequest",
    "ModelCard",
    "ModelList",
    "NamespacedFunctionToolParam",
    "OpenAIBaseModel",
    "OutputTokensDetails",
    "ProfileActivity",
    "PromptTokensDetails",
    "ReasoningAssistantMessage",
    "ResponseFormat",
    "ResponseInputOutputItem",
    "ResponseUsage",
    "ResponsesRequest",
    "ResponsesResponse",
    "ResponsesStreamResponse",
    "RuntimeMemoryUpdateRequest",
    "SpeculativeDecodingStats",
    "StartProfileRequest",
    "StreamOptions",
    "StreamingResponsesResponse",
    "TOOL_PARAM_MAX_ENUM_VALUES",
    "TokenizeRequest",
    "TokenizeResponse",
    "ToolCall",
    "UCompletionRequest",
    "UCompletionResponse",
    "UpdateWeightsRequest",
    "UsageInfo",
    "VideoGenerationRequest",
    "VideoJob",
    "VideoJobList",
    "decode_opaque_state",
    "encode_opaque_state",
    "ensure_request_chat_template_allowed",
    "to_disaggregated_params",
    "to_llm_conversation_params",
    "to_llm_disaggregated_params",
]


def __getattr__(name):
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: later access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_NAME_TO_MODULE))
