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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py
import base64
import math
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import xgrammar
from fastapi import UploadFile
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import \
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam
from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent, ResponseCompletedEvent,
    ResponseContentPartAddedEvent, ResponseContentPartDoneEvent,
    ResponseCreatedEvent, ResponseFormatTextConfig, ResponseFunctionToolCall,
    ResponseInProgressEvent, ResponseInputItemParam, ResponseOutputItem,
    ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent, ResponsePrompt,
    ResponseReasoningItem, ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent, ResponseStatus, ResponseTextConfig,
    ResponseWebSearchCallCompletedEvent, ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent)
from openai.types.responses.response import ToolChoice
from openai.types.responses.tool import Tool
from openai.types.shared import Metadata, Reasoning
from openai_harmony import ReasoningEffort
from pydantic import (AliasChoices, BaseModel, ConfigDict, Field, PositiveInt,
                      field_validator, model_validator)
from typing_extensions import Annotated, Required, TypeAlias, TypedDict

from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.inputs.media_io import MediaModality
from tensorrt_llm.llmapi import ConversationParams as LlmConversationParams
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import (DisaggScheduleStyle, GuidedDecodingParams,
                                 SamplingParams)
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.sampling_params import (check_logprobs_limit,
                                          validate_thinking_token_budget)
from tensorrt_llm.scheduling_params import AgentHierarchy

_LOGIT_BIAS_MIN = -100.0
_LOGIT_BIAS_MAX = 100.0

# Beam-search stopping mode as exposed over HTTP, mirroring the HuggingFace
# Transformers interface. The engine encodes it as an integer, but that encoding
# is an implementation detail and is kept out of the public schema.
EarlyStopping: TypeAlias = Union[bool, Literal["never"]]


def _early_stopping_to_int(value: Optional[EarlyStopping]) -> Optional[int]:
    """Translate the HF-style tri-state into the engine's integer encoding.

    Mirrors ``BeamSearchEarlyStop``: ``False`` -> 0, ``True`` -> 1,
    ``"never"`` -> 2. ``None`` stays unset so the engine picks its default.
    """
    if value is None:
        return None
    if value == "never":
        return 2
    # NB: bool is a subclass of int, so this yields 1/0 for True/False.
    return int(value)


def ensure_request_chat_template_allowed(request: Any,
                                         allow_request_chat_template: bool):
    if (getattr(request, "chat_template", None) is not None
            and not allow_request_chat_template):
        raise ValueError(
            "chat_template cannot be supplied per request unless request-level "
            "chat templates are enabled at server startup.")


def _logit_bias_to_embedding_bias(
        logit_bias: Optional[Dict[str, float]],
        vocab_size: Optional[int]) -> Optional[torch.Tensor]:
    """Convert OpenAI logit_bias dict to embedding_bias tensor for sampling."""
    if logit_bias is None:
        return None
    if vocab_size is None:
        raise ValueError(
            "logit_bias requires a tokenizer, but the server was started "
            "without one (e.g. num_postprocess_workers > 0). "
            "Remove logit_bias from your request or set num_postprocess_workers=0."
        )
    elif vocab_size <= 0:
        raise ValueError("vocab_size must be positive when logit_bias is used")

    # Create 1D zeros tensor as expected by executor API (will be unsqueezed to [1, vocab_size] internally)
    embedding_bias = torch.zeros(vocab_size, dtype=torch.float32)

    # Apply biases for specified token IDs
    for token_str, bias in logit_bias.items():
        try:
            token_id = int(token_str)
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(
                    f"Invalid logit_bias key '{token_str}': must be a valid integer token ID"
                )
            raise
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"Token ID {token_id} out of vocabulary range [0, {vocab_size})"
            )
        try:
            bias_value = float(bias)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"logit_bias value for token ID {token_id} must be a number"
            ) from e
        if not math.isfinite(bias_value):
            raise ValueError(f"logit_bias value for token ID {token_id} "
                             "must be finite")
        if not _LOGIT_BIAS_MIN <= bias_value <= _LOGIT_BIAS_MAX:
            raise ValueError(
                f"logit_bias value for token ID {token_id} must be in "
                f"[{_LOGIT_BIAS_MIN:g}, {_LOGIT_BIAS_MAX:g}]")
        embedding_bias[token_id] = bias_value

    return embedding_bias


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields & allow to initialize by both alias and field name
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = False


class PromptTokensDetails(OpenAIBaseModel):
    cached_tokens: int = 0


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokensDetails] = None


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "tensorrt_llm"


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class TokenizeRequest(OpenAIBaseModel):
    """Request body for the ``POST /_internal/tokenize`` endpoint.

    Asks the server to encode some text into the token-id sequence its
    loaded tokenizer would produce, without running any generation.

    Only plain-text ``prompt`` tokenization is supported. A ``messages``
    path was intentionally left out: to be meaningful it must reuse the full
    ``/v1/chat/completions`` preprocessing (server chat template, tools,
    documents, template kwargs, model-type resolution, multimodal handling),
    otherwise the returned token ids would silently diverge from the actual
    inference prompt. It can be added once that path is shared.

    Fields:
        model: Target model name. Accepted for compatibility with
            multi-model routers but ignored -- the server always tokenizes
            against its own loaded model.
        prompt: Plain text to tokenize directly via ``tokenizer.encode``.
    """

    model: Optional[str] = None
    prompt: str


class TokenizeResponse(OpenAIBaseModel):
    """Response body for the ``POST /_internal/tokenize`` endpoint.

    Fields:
        count: Number of tokens in the encoded sequence. Always populated;
            clients needing only the length can ignore ``tokens``.
        tokens: The encoded token ids. Kept ``Optional`` so future
            count-only variants can omit it without breaking the schema.
    """

    count: int
    tokens: Optional[List[int]] = None


class ResponseFormat(OpenAIBaseModel):
    type: Literal["text", "json", "json_schema", "json_object", "regex", "ebnf",
                  "structural_tag"]
    schema: Optional[dict] = None
    json_schema: Optional[dict] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    format: Optional[xgrammar.structural_tag.Format] = None


class DisaggregatedParams(OpenAIBaseModel):
    request_type: str
    first_gen_tokens: Optional[List[int]] = None
    first_gen_log_probs: Optional[List] = None
    first_gen_logits: Optional[List] = None
    ctx_request_id: Optional[int] = None
    encoded_opaque_state: Optional[str] = None
    draft_tokens: Optional[List[int]] = None
    disagg_request_id: Optional[int] = None
    ctx_dp_rank: Optional[int] = None
    ctx_info_endpoint: Optional[str] = None
    schedule_style: Optional[DisaggScheduleStyle] = None
    ctx_usage: Optional[UsageInfo] = None
    # TODO(TRTLLM-12407): Multimodal E/PD over trtllm-serve needs these protocol fields too:
    # encoder embedding handles, multimodal hashes, and optional mRoPE handles.
    # Add them here and in to_disaggregated_params()/to_llm_disaggregated_params()
    # before routing MM encoder -> context -> generation through OpenAI protocol.
    # Orchestrator -> context-worker instruction: return prompt_token_ids as a
    # base64 int32 buffer (prompt_token_ids_b64) instead of a JSON int array.
    return_prompt_token_ids_b64: bool = False
    # Context worker -> generation worker: the reasoning mode the context
    # worker read off the prompt it rendered. The generation worker only sees
    # prompt_token_ids, so it cannot resolve this for itself.
    resolved_thinking: Optional[bool] = None


class ConversationParams(OpenAIBaseModel):
    model_config = ConfigDict(extra="forbid",
                              populate_by_name=True,
                              validate_assignment=True)

    conversation_id: str = Field(
        description=("Stable multi-turn conversation id used for routing"), )

    @field_validator("conversation_id", mode="before")
    @classmethod
    def validate_conversation_id(cls, value: Any) -> str:
        if value is None:
            raise ValueError("conversation_id must be non-empty")
        conversation_id = str(value).strip()
        if not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        return conversation_id


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    token_ids: Optional[List[int]] = None
    logprobs: Optional[CompletionLogProbs] = None
    context_logits: Optional[Union[List[float], List[List[
        float]]]] = None  # For reward models, the output is score logits instead of text.
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    # Add prompt_tokens_ids to the response to remove the tokenization
    # in the generation server in disaggreated serving
    prompt_token_ids: Optional[Union[List[List[int]], List[int]]] = None


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    token_ids: Optional[List[int]] = None
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class EmbeddingRequest(OpenAIBaseModel):
    # OpenAI-compatible embeddings request.
    # https://platform.openai.com/docs/api-reference/embeddings/create
    model: str = Field(description="The model to use for the embedding.")
    input: Union[str, List[str], List[int], List[List[int]]] = Field(
        description="Text(s) or pre-tokenized token-id list(s) to embed. Accepts "
        "the four OpenAI forms: str, list[str], list[int], list[list[int]].")
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format of the returned embedding: a JSON array of floats "
        "(\"float\") or a base64-encoded string of packed little-endian float32 "
        "values (\"base64\").")
    dimensions: Optional[PositiveInt] = Field(
        default=None,
        description="Number of dimensions the resulting embedding should have. "
        "Only supported by Matryoshka-trained text-embedding models; rejected "
        "otherwise.")
    user: Optional[str] = Field(
        default=None,
        description="A stable identifier for the end-user (OpenAI-compatible; "
        "accepted but unused).")
    add_special_tokens: bool = Field(
        default=True,
        description=
        "TRT-LLM extension: whether to add the model's special tokens "
        "(e.g. [CLS]/[SEP]) when tokenizing string inputs. Default matches "
        "llm.encode(); encoder models such as BERT generally need them.")


class EmbeddingResponseData(OpenAIBaseModel):
    index: int = Field(
        description="Index of this embedding in the request's input list.")
    object: Literal["embedding"] = "embedding"
    embedding: Union[List[float], str] = Field(
        description=
        "The embedding: a list of floats for encoding_format=\"float\", "
        "or a base64 string of packed little-endian float32 values for "
        "encoding_format=\"base64\".")


class EmbeddingUsageInfo(OpenAIBaseModel):
    """Token usage for embeddings.

    OpenAI's embeddings `usage` object contains only `prompt_tokens` and
    `total_tokens` (no `completion_tokens`), unlike chat/completions.
    """
    prompt_tokens: int = Field(default=0,
                               description="Number of tokens in the input.")
    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed (equal to prompt_tokens for "
        "embeddings).")


class EmbeddingResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"embd-{str(uuid.uuid4().hex)}")
    object: Literal["list"] = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(description="The model used for the embedding.")
    data: List[EmbeddingResponseData] = Field(
        description="The list of embeddings, one per input item.")
    usage: EmbeddingUsageInfo = Field(
        description="Token usage for the request.")


def _response_format_to_guided_decoding_params(
    response_format: Optional[ResponseFormat],
    reasoning_parser: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[GuidedDecodingParams]:
    if response_format is None:
        guided_decoding_params = None
    elif response_format.type == "text":
        guided_decoding_params = None
    elif response_format.type == "json":
        if response_format.schema is None:
            raise ValueError(
                f"response_format.schema is required for response_format.type == {response_format.type!r}, but got None."
            )
        guided_decoding_params = GuidedDecodingParams(
            json=response_format.schema)
    elif response_format.type == "json_schema":
        if response_format.json_schema is None:
            raise ValueError(
                f"response_format.json_schema is required for response_format.type == {response_format.type!r}, but got None."
            )
        # OpenAI API spec wraps the actual JSON schema under a "schema" key.
        # Extract the actual schema if the wrapper format is used.
        json_schema = response_format.json_schema
        if isinstance(json_schema, dict) and "schema" in json_schema:
            json_schema = json_schema["schema"]
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
    elif response_format.type == "json_object":
        guided_decoding_params = GuidedDecodingParams(json_object=True)
    elif response_format.type == "regex":
        if response_format.regex is None:
            raise ValueError(
                f"response_format.regex is required for response_format.type == {response_format.type!r}, but got None."
            )
        guided_decoding_params = GuidedDecodingParams(
            regex=response_format.regex)
    elif response_format.type == "ebnf":
        if response_format.ebnf is None:
            raise ValueError(
                f"response_format.ebnf is required for response_format.type == {response_format.type!r}, but got None."
            )
        guided_decoding_params = GuidedDecodingParams(
            grammar=response_format.ebnf)
    elif response_format.type == "structural_tag":
        guided_decoding_params = GuidedDecodingParams(
            structural_tag=response_format.model_dump_json(by_alias=True,
                                                           exclude_none=True))
    else:
        raise ValueError(f"Unsupported response format: {response_format.type}")

    if guided_decoding_params is None or reasoning_parser is None:
        return guided_decoding_params

    if guided_decoding_params.structural_tag is not None:
        return guided_decoding_params

    # Adapt guided_decoding_params for reasoning parser
    if guided_decoding_params.json is not None:
        content = {
            "type": "json_schema",
            "json_schema": guided_decoding_params.json
        }
    elif guided_decoding_params.json_object:
        content = {"type": "json_schema", "json_schema": {"type": "object"}}
    elif guided_decoding_params.regex is not None:
        content = {"type": "regex", "pattern": guided_decoding_params.regex}
    elif guided_decoding_params.grammar is not None:
        content = {"type": "grammar", "grammar": guided_decoding_params.grammar}

    if reasoning_parser == "gpt_oss":
        # Trigger user constraint by final channel
        stag_format = {
            "type":
            "triggered_tags",
            "triggers": ["<|start|>assistant<|channel|>final<|message|>"],
            "tags": [
                {
                    "begin": "<|start|>assistant<|channel|>final<|message|>",
                    "content": content,
                    "end": "",
                },
            ],
            "stop_after_first":
            True,
        }
    elif reasoning_parser == "kimi_k3":
        # K3 XTML: the generation prompt already ends inside the channel the
        # model starts in. In thinking mode (the default) the response channel
        # opens mid-generation, so trigger the user constraint on it
        # (mirrors the gpt_oss final-channel handling). In non-thinking mode
        # the prompt ends inside <|open|>response<|sep|>, the trigger would
        # never be generated, and the raw grammar applies from the first
        # generated token instead.
        thinking = (chat_template_kwargs or {}).get("thinking",
                                                    True) is not False
        if not thinking:
            return guided_decoding_params
        stag_format = {
            "type":
            "triggered_tags",
            "triggers": ["<|open|>response<|sep|>"],
            "tags": [
                {
                    "begin": "<|open|>response<|sep|>",
                    "content": content,
                    "end": "<|close|>response<|sep|>",
                },
            ],
            "stop_after_first":
            True,
        }
    else:
        # Force thinking and then trigger user constraint
        parser = ReasoningParserFactory.create_reasoning_parser(
            reasoning_parser)
        stag_format = {
            "type":
            "sequence",
            "elements": [
                {
                    "type": "tag",
                    "begin": parser.reasoning_start,
                    "content": {
                        "type": "any_text"
                    },
                    "end": parser.reasoning_end,
                },
                content,
            ],
        }

    stag_format = ResponseFormat(type="structural_tag", format=stag_format)
    return GuidedDecodingParams(structural_tag=stag_format.model_dump_json(
        by_alias=True, exclude_none=True))


def _response_format_text_config_to_guided_decoding_params(
    text_format: Optional[ResponseFormatTextConfig],
    reasoning_parser: Optional[str] = None,
) -> Optional[GuidedDecodingParams]:
    if text_format is None:
        return None

    resp_format = ResponseFormat(type=text_format.type,
                                 json_schema=getattr(text_format, "schema_",
                                                     None))
    return _response_format_to_guided_decoding_params(
        resp_format, reasoning_parser=reasoning_parser)


def _record_sampling_params_request_fields(
        request: OpenAIBaseModel, sampling_params: SamplingParams) -> None:
    """Preserve explicitly supplied fields across protocol defaulting."""
    sampling_params._set_request_provided_fields(
        field_name for field_name in request.model_fields_set
        if getattr(request, field_name, None) is not None)


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    lora_request: Optional[LoRARequest] = None
    prompt_ignore_length: Optional[int] = 0

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: int = 0
    top_p_min: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    # Unset by default so the engine picks its own beam-search defaults, as it
    # does for requests coming through the Python API. Sending concrete values
    # here would put every served request on the length-normalized, exhaustive
    # beam-search path.
    length_penalty: Optional[float] = None
    early_stopping: Optional[EarlyStopping] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    return_context_logits: bool = False
    detokenize: bool = True
    thinking_token_budget: Optional[int] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of output. "
         "{'type': 'text'}, {'type': 'json'}, {'type': 'json_object'}, {'type': 'regex'}, "
         "{'type': 'ebnf'}, {'type': 'structural_tag'} are supported."),
    )

    disaggregated_params: Optional[DisaggregatedParams] = Field(
        default=None,
        description=("Parameters for disaggregated serving"),
    )
    priority: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=
        ("Scheduling priority in [0.0, 1.0]; higher is served first. Only honored "
         "when the engine runs with scheduler_config.waiting_queue_policy=priority. "
         "If unset, the engine default (0.5) is used."),
    )
    conversation_params: Optional[ConversationParams] = Field(
        default=None,
        description=("Parameters for multi-turn conversation routing"),
    )

    # doc: end-completion-extra-params

    def to_sampling_params(self,
                           vocab_size: Optional[int] = None,
                           gather_generation_logits: bool = False,
                           backend: Optional[str] = None) -> SamplingParams:
        sampling_logprobs = None
        return_log_probs = False
        if self.logprobs:
            if backend == "pytorch" or gather_generation_logits:
                sampling_logprobs = self.logprobs
            elif self.logprobs > 1:
                raise ValueError(
                    "`logprobs` must be 1 or `gather_generation_logits` must be `True` to use `logprobs` > 1"
                )
            else:
                return_log_probs = True

        sampling_params = SamplingParams(
            best_of=self.best_of,
            frequency_penalty=self.frequency_penalty,
            logprobs=sampling_logprobs,
            max_tokens=self.max_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=(self.temperature
                         if self.temperature is not None else 1.0),
            top_p=(self.top_p if self.top_p is not None else 1.0),
            prompt_ignore_length=self.prompt_ignore_length,

            # completion-sampling-params
            use_beam_search=self.use_beam_search,
            top_k=self.top_k,
            top_p_min=self.top_p_min if self.top_p_min > 0 else None,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            early_stopping=_early_stopping_to_int(self.early_stopping),
            stop_token_ids=self.stop_token_ids,
            include_stop_str_in_output=self.include_stop_str_in_output,
            ignore_eos=self.ignore_eos,
            min_tokens=self.min_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            return_context_logits=self.return_context_logits,
            guided_decoding=_response_format_to_guided_decoding_params(
                self.response_format),
            detokenize=self.detokenize,
            thinking_token_budget=self.thinking_token_budget,

            # logits_bias
            embedding_bias=_logit_bias_to_embedding_bias(
                self.logit_bias, vocab_size),

            # completion-extra-params
            add_special_tokens=self.add_special_tokens,
        )
        if return_log_probs:
            sampling_params._return_log_probs = True
        _record_sampling_params_request_fields(self, sampling_params)
        return sampling_params

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        check_logprobs_limit("logprobs", data.get("logprobs"))
        return data

    @field_validator("thinking_token_budget", mode="before")
    @classmethod
    def check_thinking_token_budget(cls, value):
        return validate_thinking_token_budget(value)

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when stream is true.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_suffix(cls, data):
        if data.get("suffix"):
            raise ValueError("suffix is not supported")
        return data


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class DeltaFunctionCall(OpenAIBaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(OpenAIBaseModel):
    id: str = Field(
        default_factory=lambda: f"chatcmpl-tool-{str(uuid.uuid4().hex)}")
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaToolCall(OpenAIBaseModel):
    id: Optional[str] = None
    type: Literal["function"] = "function"
    index: int
    function: Optional[DeltaFunctionCall] = None


class ChatMessage(OpenAIBaseModel):
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[List[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    top_logprobs: Optional[List[ChatCompletionLogProb]] = None


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[OpenAIChatCompletionContentPartParam,
                                       CustomChatCompletionContentPartParam]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    # This is so custom fields not in any of the `ChatCompletionMessage<XYZ>Param` defined by OpenAI
    # are still allowed.
    # Examples include: assistant messages with `reasoning` / `reasoning_content`.
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam], None]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


class ReasoningAssistantMessage(ChatCompletionAssistantMessageParam):
    """Assistant message that includes reasoning tokens."""
    reasoning: Optional[str]
    # NOTE: some older benchmarks and chat templates assume the below, which has been deprecated
    # in other inference frameworks in favor of the above `reasoning` field.
    reasoning_content: Optional[str]


class DynamicToolsSystemMessageParam(TypedDict, total=False):
    """System message carrying message-level (dynamic) tool declarations.

    Kimi-style templates render such messages as an in-conversation tool
    declare block. Must come first in `ChatCompletionMessageParam`: the
    stock OpenAI system-message TypedDict otherwise wins smart-union scoring
    and silently drops the `tools` key.
    """
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    role: Required[Literal["system"]]
    tools: Required[List[dict]]
    content: Union[str, List[ChatCompletionContentPartParam], None]
    name: str


ChatCompletionMessageParam = Union[DynamicToolsSystemMessageParam,
                                   OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam,
                                   ReasoningAssistantMessage]


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None
    # TODO: progressivly add more info like input_ids, specific_token_ids, mrope, mm_hashes, etc
    # TODO: and use a JSON-safe handle to refer to the server-side output
    mm_embedding_handle: Optional[Dict[str, Any]] = None

    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    # Add prompt_tokens_ids to the response to remove the tokenization
    # in the generation server in disaggreated serving
    prompt_token_ids: Optional[List[int]] = None
    # base64 int32 buffer alternative to prompt_token_ids; set by the context
    # worker so the orchestrator can relay a string instead of the int list.
    prompt_token_ids_b64: Optional[str] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    # For GPT-OSS style reasoning
    reasoning: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall]] = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class ChatCompletionThinkingParam(OpenAIBaseModel):
    """Kimi/Moonshot `thinking` extension controlling reasoning output.

    `keep` is fixed to `"all"` when thinking is enabled and ignored when
    disabled; `effort` is only meaningful when enabled. An explicit
    `effort` wins over a request-level `reasoning_effort`, which applies
    only when `effort` is absent (and never for a disabled request).
    """
    type: Literal["enabled", "disabled"] = "enabled"
    keep: Optional[Literal["all"]] = None
    effort: Optional[Literal["low", "high", "max"]] = None


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    # Add prompt_tokens_ids to the request to remove the tokenization
    # in the generation server in disaggreated serving
    prompt_token_ids: Optional[List[int]] = None
    # base64 int32 buffer relayed by the orchestrator from the ctx response;
    # decoded back to prompt_token_ids on the generation worker. Not for clients.
    prompt_token_ids_b64: Optional[str] = None
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_completion_tokens: Optional[int] = Field(default=None,
                                                 validation_alias='max_tokens')
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    user: Optional[str] = None
    reasoning_effort: Optional[ReasoningEffort | Literal[
        "low", "medium", "high", "max", "none"]] = Field(
            default=ReasoningEffort.LOW,
            description=(
                "The level of reasoning effort to use. Controls how much "
                "reasoning is shown in the model's response. Options: "
                "'low', 'medium', 'high' (harmony/gpt-oss), plus 'max' and "
                "'none' for models with Kimi-style thinking control."),
        )
    # Kimi/Moonshot extension: structured control of reasoning output. The
    # serving layer maps it into the chat-template kwargs for models whose
    # template understands it (e.g. kimi_k3); other models ignore it.
    thinking: Optional[ChatCompletionThinkingParam] = None
    thinking_token_budget: Optional[int] = None
    prompt_ignore_length: Optional[int] = 0

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = 0
    top_p_min: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    # Unset by default so the engine picks its own beam-search defaults, as it
    # does for requests coming through the Python API. Sending concrete values
    # here would put every served request on the length-normalized, exhaustive
    # beam-search path.
    length_penalty: Optional[float] = None
    early_stopping: Optional[EarlyStopping] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    lora_request: Optional[LoRARequest] = None
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )

    media_io_kwargs: Optional[Dict[MediaModality, Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Per-request override for the server's `--media_io_kwargs`. "
            "Shape: `{modality: {kwarg: value}}` with modality in "
            "{\"image\", \"video\", \"audio\"}; unknown modality keys are "
            "rejected. Per modality, request kwargs are shallow-merged "
            "onto the server defaults (request wins per key). For "
            "`video`, overriding only one of `fps`/`num_frames` drops "
            "the other from the server default so the loader's built-in "
            "is used. "
            "Example: `{\"video\": {\"num_frames\": 32}}`."),
    )

    disaggregated_params: Optional[DisaggregatedParams] = Field(
        default=None,
        description=("Parameters for disaggregated serving"),
    )
    conversation_params: Optional[ConversationParams] = Field(
        default=None,
        description=("Parameters for multi-turn conversation routing"),
    )

    cache_salt: Optional[str] = Field(
        default=None,
        description=
        ("If specified, KV cache will be salted with the provided string "
         "to limit the kv cache reuse on with the requests having the same string."
         ))
    priority: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=
        ("Scheduling priority in [0.0, 1.0]; higher is served first. Only honored "
         "when the engine runs with scheduler_config.waiting_queue_policy=priority. "
         "If unset, the engine default (0.5) is used."),
    )

    agent_hierarchy: Optional[AgentHierarchy] = Field(
        default=None, description="Agent hierarchy ")

    mm_processor_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=
        "Per-request kwargs forwarded to the multimodal HF processor (e.g. num_frames for video models).",
    )

    # doc: end-chat-completion-extra-params

    def to_sampling_params(self,
                           vocab_size: Optional[int] = None,
                           gather_generation_logits: bool = False,
                           reasoning_parser: Optional[str] = None,
                           backend: Optional[str] = None) -> SamplingParams:
        sampling_logprobs = None
        return_log_probs = False
        if self.logprobs:
            logprobs = 1 if not self.top_logprobs else self.top_logprobs
            if backend == "pytorch" or gather_generation_logits:
                sampling_logprobs = logprobs
            elif self.top_logprobs:
                raise ValueError(
                    "`gather_generation_logits` must be `True` to use `top_logprobs`"
                )
            else:
                return_log_probs = True

        sampling_params = SamplingParams(
            frequency_penalty=self.frequency_penalty,
            logprobs=sampling_logprobs,
            max_tokens=self.max_completion_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=(self.temperature
                         if self.temperature is not None else 1.0),
            prompt_ignore_length=self.prompt_ignore_length,

            # chat-completion-sampling-params
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            top_k=self.top_k,
            top_p=(self.top_p if self.top_p is not None else 1.0),
            top_p_min=self.top_p_min if self.top_p_min > 0 else None,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            early_stopping=_early_stopping_to_int(self.early_stopping),
            stop_token_ids=self.stop_token_ids,
            include_stop_str_in_output=self.include_stop_str_in_output,
            ignore_eos=self.ignore_eos,
            min_tokens=self.min_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            guided_decoding=_response_format_to_guided_decoding_params(
                self.response_format,
                reasoning_parser=reasoning_parser,
                chat_template_kwargs=self.chat_template_kwargs),
            thinking_token_budget=self.thinking_token_budget,

            # logits_bias
            embedding_bias=_logit_bias_to_embedding_bias(
                self.logit_bias, vocab_size),

            # chat-completion-extra-params
            add_special_tokens=self.add_special_tokens,
        )
        if return_log_probs:
            sampling_params._return_log_probs = True
        _record_sampling_params_request_fields(self, sampling_params)
        return sampling_params

    @model_validator(mode='before')
    @classmethod
    def validate_stream_options(cls, values):
        if (values.get('stream_options') is not None
                and not values.get('stream')):
            raise ValueError("stream_options can only be set if stream is true")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_tool_choice(cls, data):
        if not isinstance(data, dict):
            return data
        has_dynamic_tools = any(
            isinstance(msg, dict) and msg.get("role") == "system"
            and msg.get("tools") for msg in data.get("messages") or [])
        if "tool_choice" not in data and (data.get("tools")
                                          or has_dynamic_tools):
            data["tool_choice"] = "auto"
        # "none" and "auto" are meaningful without tools. "required" needs a
        # non-empty tool set — request-level or message-level (dynamic)
        # tools both count. A named function must be a request-level tool.
        if "tool_choice" in data and data["tool_choice"] not in ("none",
                                                                 "auto"):
            satisfied = data.get("tools") or (data["tool_choice"] == "required"
                                              and has_dynamic_tools)
            if not satisfied:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_message_tools_role(cls, data):
        """Message-level tool declarations ride on system messages only.

        Union validation strips unknown keys from non-system messages, so
        this raw-payload validator is the only layer that can reject the
        misuse loudly instead of silently dropping the client's tools. The
        rest of the dynamic-tools contract is model-specific and validated
        in the serving layer (kimi_k3 only).
        """
        if not isinstance(data, dict):
            return data
        for message in data.get("messages") or []:
            if (isinstance(message, dict) and message.get("tools") is not None
                    and message.get("role") != "system"):
                raise ValueError(
                    "Message-level `tools` are only allowed on system "
                    "messages.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (top_logprobs := data.get("top_logprobs")) is not None:
            check_logprobs_limit("top_logprobs", top_logprobs)
            if not data.get("logprobs"):
                raise ValueError(
                    "logprobs must be true when using top_logprobs")
        return data

    @field_validator("thinking_token_budget", mode="before")
    @classmethod
    def check_thinking_token_budget(cls, value):
        return validate_thinking_token_budget(value)

    @model_validator(mode="before")
    @classmethod
    def check_suffix(cls, data):
        if data.get("suffix"):
            raise ValueError("suffix is not supported")
        return data

    @field_validator("cache_salt")
    @classmethod
    def check_cache_salt_support(cls, v):
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    "Parameter 'cache_salt' must be a non-empty string if provided."
                )
        return v


class KVCacheTruncateRequest(OpenAIBaseModel):
    model: str
    messages: List[ChatCompletionMessageParam] = []
    messages_to_retain: List[ChatCompletionMessageParam] = []
    tools: Optional[List[ChatCompletionToolsParam]] = None
    add_generation_prompt: Optional[bool] = True
    documents: Optional[list] = None
    chat_template: Optional[str] = None
    chat_template_kwargs: Optional[dict] = None
    reasoning_effort: Optional[str] = None
    tool_choice: Optional[str] = None


class KVCacheTruncateTokensRequest(OpenAIBaseModel):
    """Token-level analog of :class:`KVCacheTruncateRequest`.

    Posted to the ``/_control/kv_cache/truncate_tokens`` endpoint. Bypasses
    the chat-template tokenization path entirely: callers (e.g.
    trace replay) that already operate on raw token ids hand the server a
    batch of full token-id sequences plus the prefix length to retain for
    each, and the server trims the corresponding KV-cache blocks in the
    radix tree. Each ``prefixes[i]`` becomes one
    :class:`tensorrt_llm.executor.request.TruncateKVCacheRequest` on the
    executor's KV-cache control queue, so one batched HTTP request fans out
    into N atomic per-prefix radix-tree mutations while paying a single
    network round-trip.

    Fields:
        model: Target model name. Accepted for router compatibility but
            ignored -- the server uses its own loaded model.
        prefixes: Batch of full token-id sequences. Each entry is one
            cached prefix whose KV-cache blocks should be truncated.
        num_tokens_to_keep: Per-prefix number of leading tokens to retain,
            parallel to ``prefixes`` (``num_tokens_to_keep[i]`` applies to
            ``prefixes[i]``). Blocks beyond this length are dropped from the
            radix tree.
    """

    model: Optional[str] = None
    prefixes: List[List[int]]
    num_tokens_to_keep: List[int]


# The trailing dict keeps a client's own item types from failing the request
# outright. Codex multi-agent sessions carry "agent_message" items, which no
# SDK model describes; without a permissive member the union rejects the whole
# input and every request from a spawned agent comes back 422, so agent
# collaboration cannot work at all. Known shapes still match their typed
# member first; see _response_output_item_to_chat_completion_message for how
# an unrecognised item is replayed.
ResponseInputOutputItem: TypeAlias = Union[ResponseInputItemParam,
                                           ResponseReasoningItem,
                                           ResponseFunctionToolCall, dict[str,
                                                                          Any]]

# Roles whose message items map to EasyInputMessageParam / Message, both of
# which forbid ``id``. An assistant message maps to ResponseOutputMessageParam
# instead, which *requires* ``id`` and ``status`` - stripping ``id`` there
# leaves the item matching no variant of the input union, so a conversation
# breaks as soon as it carries one assistant turn.
#
# Module level on purpose: pydantic turns a leading-underscore class attribute
# into a ModelPrivateAttr, so referring to it from a validator would raise
# "argument of type 'ModelPrivateAttr' is not iterable" at request time.
_ID_STRIPPED_ROLES = ("user", "system", "developer")


def _materialize_validator_iterators(value, _depth=0):
    """Recursively replace pydantic ValidatorIterator objects with lists.

    Matched by type name rather than by import: the class lives in the
    pydantic_core extension module and is not part of its public API.
    """
    if _depth > 12:
        return value
    if type(value).__name__ == "ValidatorIterator":
        return [_materialize_validator_iterators(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _materialize_validator_iterators(item, _depth + 1)
        return value
    if isinstance(value, list):
        return [_materialize_validator_iterators(v, _depth + 1) for v in value]
    return value


class ResponsesRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/responses/create
    #
    # Unlike the rest of the OpenAI models this one accepts unknown fields.
    # Real Responses API clients attach evolving telemetry and routing keys -
    # Codex CLI sends client_metadata and prompt_cache_key, for instance - and
    # rejecting the whole request over a field the server would ignore anyway
    # makes those clients unusable for no benefit.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    background: Optional[bool] = False
    include: Optional[list[
        Literal[
            "code_interpreter_call.outputs",
            "computer_call_output.output.image_url",
            "file_search_call.results",
            "message.input_image.image_url",
            "message.output_text.logprobs",
            "reasoning.encrypted_content",
        ],
    ]] = None
    input: Union[str, list[ResponseInputOutputItem]]
    instructions: Optional[str] = None

    @field_validator("input", mode="before")
    @classmethod
    def _drop_unsupported_input_item_keys(cls, value):
        """Strip per-item keys the vendored Responses item types reject.

        Clients echo back items exactly as the server emitted them, so input
        message items arrive carrying the ``id`` the server assigned, which
        ``EasyInputMessageParam`` / ``Message`` forbid. That fails the whole
        request over a field with no prompt content.

        Assistant messages and tool-call items keep their ids: the former
        needs it to validate, the latter uses it to pair calls with results.
        """
        if not isinstance(value, list):
            return value

        def _with_annotations(part):
            """output_text requires annotations; clients often omit it."""
            if (isinstance(part, dict) and part.get("type") == "output_text"
                    and "annotations" not in part):
                return {**part, "annotations": []}
            return part

        cleaned = []
        for item in value:
            # A client may send a bare content part as a top-level item, not
            # only nested inside a message.
            item = _with_annotations(item)
            if isinstance(item, dict) and item.get("type") in (None, "message"):
                role = item.get("role")
                if "id" in item and role in _ID_STRIPPED_ROLES:
                    item = {k: v for k, v in item.items() if k != "id"}
                elif role == "assistant":
                    # ResponseOutputMessageParam requires status, and each
                    # output_text part requires annotations. Clients rebuild an
                    # assistant turn from the streamed deltas rather than from
                    # the server's content-part objects, so both routinely
                    # arrive absent and fail the whole request. Default them
                    # rather than reject a conversation over fields that carry
                    # no prompt content.
                    item = dict(item)
                    item.setdefault("status", "completed")
                    content = item.get("content")
                    if isinstance(content, list):
                        item["content"] = [
                            {
                                **part, "annotations": part.get(
                                    "annotations", [])
                            } if isinstance(part, dict)
                            and part.get("type") == "output_text" else part
                            for part in content
                        ]
            cleaned.append(item)
        return cleaned

    @field_validator("input", mode="after")
    @classmethod
    def _materialize_lazy_item_content(cls, value):
        """Force lazily-validated sequences anywhere in ``input`` into lists.

        Several vendored Responses item types declare sequence fields as
        ``Iterable[...]`` - ``ResponseOutputMessageParam.content`` and
        ``ResponseOutputTextParam.annotations`` among them - and pydantic
        validates an Iterable lazily into a ``ValidatorIterator``. That object
        cannot be pickled, so a request carrying structured input dies with

            cannot pickle 'pydantic_core._pydantic_core.ValidatorIterator'

        the moment it is handed to a postprocess worker. It is also
        single-consumption, so even with workers disabled anything that reads
        the field twice sees an empty sequence the second time.

        The walk has to be recursive: the lazy fields are nested inside items,
        e.g. input[1].content[0].annotations.
        """
        return _materialize_validator_iterators(value)

    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Metadata] = None
    model: str
    parallel_tool_calls: Optional[bool] = False
    previous_response_id: Optional[str] = None
    prompt: Optional[ResponsePrompt] = None
    reasoning: Optional[Reasoning] = None
    thinking_token_budget: Optional[int] = None
    service_tier: Literal["auto", "default", "flex", "scale",
                          "priority"] = "auto"
    store: Optional[bool] = True
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    text: Optional[ResponseTextConfig] = None
    tool_choice: ToolChoice = "auto"
    tools: list[Tool] = Field(default_factory=list)
    top_logprobs: Optional[int] = 0
    top_p: Optional[float] = None
    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None

    request_id: str = Field(
        default_factory=lambda: f"resp_{str(uuid.uuid4().hex)}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."),
    )

    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 1.0,
        "top_p": 1.0,
    }

    def to_sampling_params(
        self,
        default_sampling_params: Optional[dict] = None,
        reasoning_parser: Optional[str] = None,
    ) -> SamplingParams:
        max_tokens = None
        if self.max_output_tokens is not None:
            max_tokens = self.max_output_tokens

        default_sampling_params = default_sampling_params or {}
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
        stop_token_ids = default_sampling_params.get("stop_token_ids", None)

        # Structured output
        guided_decoding = None
        if self.text is not None and self.text.format is not None:
            guided_decoding = _response_format_text_config_to_guided_decoding_params(
                self.text.format, reasoning_parser=reasoning_parser)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            stop_token_ids=stop_token_ids,
            guided_decoding=guided_decoding,
            thinking_token_budget=self.thinking_token_budget,
        )
        _record_sampling_params_request_fields(self, sampling_params)
        return sampling_params

    @model_validator(mode="before")
    @classmethod
    def validate_background(cls, data):
        if not data.get("background"):
            return data
        if not data.get("store", True):
            raise ValueError("background can only be used when `store` is true")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, data):
        if data.get("prompt") is not None:
            raise ValueError("prompt template is not supported")
        return data

    @field_validator("thinking_token_budget", mode="before")
    @classmethod
    def check_thinking_token_budget(cls, value):
        return validate_thinking_token_budget(value)


class InputTokensDetails(OpenAIBaseModel):
    cached_tokens: int
    # Required by the openai SDK models that re-validate this payload when it
    # is embedded in a streaming event. Omitting it fails validation while the
    # response is being streamed, which truncates the stream with no
    # terminating event and leaves the client waiting forever. Prompt cache
    # writes are not tracked separately, so this is reported as zero.
    cache_write_tokens: int = 0


class OutputTokensDetails(OpenAIBaseModel):
    reasoning_tokens: int


class ResponseUsage(OpenAIBaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


StreamingResponsesResponse: TypeAlias = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseCompletedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterCallCompletedEvent,
]


class ResponsesResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"resp_{str(uuid.uuid4().hex)}")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    # error: Optional[ResponseError] = None
    # incomplete_details: Optional[IncompleteDetails] = None
    instructions: Optional[str] = None
    metadata: Optional[Metadata] = None
    model: str
    object: Literal["response"] = "response"
    output: list[ResponseOutputItem]
    parallel_tool_calls: bool
    temperature: float
    tool_choice: ToolChoice
    tools: list[Tool]
    top_p: float
    background: bool
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    previous_response_id: Optional[str] = None
    prompt: Optional[ResponsePrompt] = None
    reasoning: Optional[Reasoning] = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"]
    status: ResponseStatus
    text: Optional[ResponseTextConfig] = None
    top_logprobs: int
    truncation: Literal["auto", "disabled"]
    usage: Optional[ResponseUsage] = None
    user: Optional[str] = None

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        model_name: str,
        created_time: int,
        output: list[ResponseOutputItem],
        status: ResponseStatus,
        usage: Optional[ResponseUsage] = None,
    ) -> "ResponsesResponse":
        return cls(
            id=request.request_id,
            created_at=created_time,
            instructions=request.instructions,
            metadata=request.metadata,
            model=model_name,
            output=output,
            parallel_tool_calls=request.parallel_tool_calls,
            temperature=sampling_params.temperature,
            tool_choice=request.tool_choice,
            tools=request.tools,
            top_p=sampling_params.top_p,
            background=request.background,
            max_output_tokens=sampling_params.max_tokens,
            max_tool_calls=request.max_tool_calls,
            previous_response_id=request.previous_response_id,
            prompt=request.prompt,
            reasoning=request.reasoning,
            service_tier=request.service_tier,
            status=status,
            text=request.text,
            top_logprobs=sampling_params.logprobs,
            truncation=request.truncation,
            user=request.user,
            usage=usage,
        )


class ResponsesStreamResponse(OpenAIBaseModel):
    response: ResponsesResponse
    sequence_number: int
    type: Literal["response.created", "response.in_progress",
                  "response.completed", "response.failed",
                  "response.incomplete"]


class MemoryUpdateRequest(OpenAIBaseModel):
    tags: list[str] = Field(
        min_length=1,
        description="Memory tags to release/resume, e.g. ['model', 'kv_cache']")


class UpdateWeightsRequest(OpenAIBaseModel):
    weights: Optional[Dict[str, str]] = Field(
        default=None,
        description="Weight handles dict, or None to finalize update")


def encode_opaque_state(opaque_state: Optional[bytes]) -> Optional[str]:
    if opaque_state is None:
        return None
    return base64.b64encode(opaque_state).decode("utf-8")


def decode_opaque_state(encoded_opaque_state: Optional[str]) -> Optional[bytes]:
    if encoded_opaque_state is None:
        return None
    return base64.b64decode(encoded_opaque_state)


def _serialize_first_gen_log_probs(
        first_gen_log_probs: Optional[list]) -> Optional[List]:
    """Serialize ``list[dict[int, Logprob]] | list[float]`` to a JSON-safe form.

    - Default (verbose) format: each position is a ``dict[int, Logprob]`` and is
      serialized as a list of ``{token_id, logprob, rank}`` dicts.
    - Simple format: each position is a ``float``; passed through verbatim.
    """
    if first_gen_log_probs is None:
        return None
    if not isinstance(first_gen_log_probs, list):
        raise ValueError("first_gen_log_probs must be a list")
    result = []
    for i, pos in enumerate(first_gen_log_probs):
        if isinstance(pos, dict):
            result.append([{
                "token_id": tid,
                "logprob": lp.logprob,
                "rank": lp.rank
            } for tid, lp in pos.items()])
        elif isinstance(pos, (float, int)):
            # Simple format: per-token sampled logprob.
            result.append(float(pos))
        else:
            raise ValueError(
                f"first_gen_log_probs[{i}] must be a dict or float, got {type(pos)}"
            )
    return result


def _deserialize_first_gen_log_probs(
    serialized: Optional[List], ) -> Optional[list]:
    """Inverse of :func:`_serialize_first_gen_log_probs`.

    Returns either ``list[dict[int, Logprob]]`` (default format) or
    ``list[float]`` (simple format) depending on the serialized payload.
    """
    if serialized is None:
        return None
    from tensorrt_llm.executor.result import Logprob
    result = []
    for i, pos in enumerate(serialized):
        if isinstance(pos, (float, int)):
            result.append(float(pos))
            continue
        if not isinstance(pos, list):
            raise ValueError(
                f"first_gen_log_probs[{i}] must be a list or float, got {type(pos)}"
            )
        token_map = {}
        for j, item in enumerate(pos):
            if not isinstance(item, dict):
                raise ValueError(
                    f"first_gen_log_probs[{i}][{j}] must be a dict")
            if "token_id" not in item or "logprob" not in item:
                raise ValueError(
                    f"first_gen_log_probs[{i}][{j}] missing required keys "
                    "'token_id' and/or 'logprob'")
            token_map[item["token_id"]] = Logprob(logprob=item["logprob"],
                                                  rank=item.get("rank"))
        result.append(token_map)
    return result


def _serialize_first_gen_logits(
    first_gen_logits: Optional[list], ) -> Optional[List]:
    """Serialize list[torch.Tensor] to JSON-safe list[dict] with base64 data."""
    if first_gen_logits is None:
        return None
    result = []
    for i, tensor in enumerate(first_gen_logits):
        t = tensor.contiguous().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float16)
        np_array = t.numpy()
        result.append({
            "data": base64.b64encode(np_array.tobytes()).decode(),
            "shape": list(np_array.shape),
            "dtype": str(np_array.dtype),
        })
    return result


def _deserialize_first_gen_logits(
    serialized: Optional[List], ) -> Optional[list]:
    """Deserialize JSON list[dict] back to list[torch.Tensor]."""
    if serialized is None:
        return None
    import numpy as np
    result = []
    for i, item in enumerate(serialized):
        if not isinstance(item, dict):
            raise ValueError(
                f"first_gen_logits[{i}] must be a dict, got {type(item)}")
        for key in ("data", "shape", "dtype"):
            if key not in item:
                raise ValueError(
                    f"first_gen_logits[{i}] missing required key '{key}'")
        np_array = np.frombuffer(
            base64.b64decode(item["data"]),
            dtype=np.dtype(item["dtype"]),
        ).reshape(item["shape"])
        result.append(torch.from_numpy(np_array.copy()))
    return result


def to_disaggregated_params(
        tllm_disagg_params: LlmDisaggregatedParams) -> DisaggregatedParams:
    if tllm_disagg_params is None:
        return None
    ctx_usage = tllm_disagg_params.ctx_usage
    if ctx_usage is not None and not isinstance(ctx_usage, UsageInfo):
        ctx_usage = UsageInfo.model_validate(ctx_usage)
    return DisaggregatedParams(
        request_type=tllm_disagg_params.request_type,
        first_gen_tokens=tllm_disagg_params.first_gen_tokens,
        first_gen_log_probs=_serialize_first_gen_log_probs(
            tllm_disagg_params.first_gen_log_probs),
        first_gen_logits=_serialize_first_gen_logits(
            tllm_disagg_params.first_gen_logits),
        ctx_request_id=tllm_disagg_params.ctx_request_id,
        encoded_opaque_state=encode_opaque_state(
            tllm_disagg_params.opaque_state),
        draft_tokens=tllm_disagg_params.draft_tokens,
        disagg_request_id=tllm_disagg_params.disagg_request_id,
        ctx_dp_rank=tllm_disagg_params.ctx_dp_rank,
        ctx_info_endpoint=tllm_disagg_params.ctx_info_endpoint,
        schedule_style=tllm_disagg_params.schedule_style,
        ctx_usage=ctx_usage,
    )


def to_llm_disaggregated_params(
        disaggregated_params: DisaggregatedParams) -> LlmDisaggregatedParams:
    if disaggregated_params is None:
        return None
    ctx_usage = disaggregated_params.ctx_usage
    return LlmDisaggregatedParams(
        request_type=disaggregated_params.request_type,
        first_gen_tokens=disaggregated_params.first_gen_tokens,
        first_gen_log_probs=_deserialize_first_gen_log_probs(
            disaggregated_params.first_gen_log_probs),
        first_gen_logits=_deserialize_first_gen_logits(
            disaggregated_params.first_gen_logits),
        ctx_request_id=disaggregated_params.ctx_request_id,
        opaque_state=decode_opaque_state(
            disaggregated_params.encoded_opaque_state),
        draft_tokens=disaggregated_params.draft_tokens,
        disagg_request_id=disaggregated_params.disagg_request_id,
        ctx_dp_rank=disaggregated_params.ctx_dp_rank,
        ctx_info_endpoint=disaggregated_params.ctx_info_endpoint,
        schedule_style=disaggregated_params.schedule_style,
        ctx_usage=None if ctx_usage is None else ctx_usage.model_dump(),
    )


def to_llm_conversation_params(
    conversation_params: Optional[ConversationParams]
) -> Optional[LlmConversationParams]:
    if conversation_params is None:
        return None
    return LlmConversationParams(
        conversation_id=conversation_params.conversation_id)


# ============================================================================
# Diffusion API Protocol Classes
# ============================================================================


class ImageGenerationRequest(OpenAIBaseModel):
    """OpenAI-compatible image generation request.

    Universal per-request fields map 1:1 to :class:`VisualGenParams`.
    Model-specific knobs (``stg_scale``, ``guidance_rescale``, …)
    travel through ``extra_params``; the executor validates each
    key against the loaded pipeline's
    ``extra_param_specs``. Unknown top-level fields are rejected
    with HTTP 422 via the inherited ``extra="forbid"`` policy.
    """

    # Prompt + transport (OpenAI-standard, always honored)
    prompt: str
    response_format: Literal["url", "b64_json", "path"] = "url"
    format: Literal["png", "webp", "jpeg", "safetensors", "pt"] = Field(
        default="png",
        description=(
            "Generation content encoding format. Image encoders write "
            "``png``/``webp``/``jpeg``; tensor encoders write "
            "``safetensors``/``pt`` for programmatic post-processing."),
    )
    seed: Optional[int] = Field(default=None,
                                ge=0,
                                description="Random seed for reproducibility.")

    # Resolution. ``size`` is OpenAI-shaped; ``width`` + ``height`` are an
    # equivalent structured alternative. Exactly one of width/height is
    # rejected by the paired validator below. Numeric fields use
    # ``gt=0`` as a safety net so zero / negative inputs are rejected
    # with HTTP 422 before reaching the pipeline.
    size: Optional[str] = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)

    # TRT-LLM-supported per-request params (1:1 with VisualGenParams fields)
    num_inference_steps: Optional[int] = Field(default=None, gt=0)
    guidance_scale: Optional[float] = Field(default=None, gt=0)
    max_sequence_length: Optional[int] = Field(default=None, gt=0)
    negative_prompt: Optional[str] = None
    n: Optional[int] = Field(
        default=None,
        gt=0,
        le=10,
        description=("Number of images to generate. Capped at 10 to match the "
                     "OpenAI images API and to bound GPU memory / disk usage."),
    )

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys."),
    )

    # Accepted-but-ignored OpenAI-shaped fields. The conversion no-ops; the
    # server logs WARNING when a client sets ``quality`` or ``style``, and
    # WARNING-on-mismatch for ``model``. Kept in the schema so OpenAI-SDK
    # clients don't trip ``extra="forbid"``.
    model: Optional[str] = None
    quality: Optional[Literal["standard", "hd"]] = None
    style: Optional[Literal["vivid", "natural"]] = None
    user: Optional[str] = None

    @model_validator(mode="after")
    def _check_paired_dimensions(self):
        """Reject sending exactly one of ``width`` / ``height``.

        Either both are sent (structured resolution wins over ``size``)
        or neither is sent (``size`` or pipeline default applies).
        """
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "width and height must be sent together; got width="
                f"{self.width!r}, height={self.height!r}")
        return self


class ImageEditRequest(OpenAIBaseModel):
    """OpenAI-compatible image editing request.

    The server accepts the OpenAI multipart shape and a JSON/base64
    shape for tests and non-SDK clients. Model-specific knobs travel
    through ``extra_params`` and are validated by the loaded visual
    generation pipeline.
    """

    prompt: str
    image: Union[str, UploadFile, List[Union[str, UploadFile]]] = Field(
        description="Input image or images to edit.")
    mask: Optional[Union[str, UploadFile]] = Field(
        default=None,
        description=
        "Optional edit mask. Currently accepted for compatibility but unsupported.",
    )
    response_format: Literal["url", "b64_json", "path"] = "url"
    output_format: Literal["png", "webp", "jpeg"] = Field(
        default="png",
        validation_alias=AliasChoices("output_format", "format"),
        description="Edited image content encoding format.",
    )
    seed: Optional[int] = Field(default=None,
                                ge=0,
                                description="Random seed for reproducibility.")

    size: Optional[str] = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)

    num_inference_steps: Optional[int] = Field(default=None, gt=0)
    guidance_scale: Optional[float] = Field(default=None, gt=0)
    max_sequence_length: Optional[int] = Field(default=None, gt=0)
    negative_prompt: Optional[str] = None
    n: Optional[int] = Field(
        default=None,
        gt=0,
        le=10,
        description=("Number of edited images to generate. Capped at 10 to "
                     "match the OpenAI images API."),
    )

    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys."),
    )

    model: Optional[str] = None
    quality: Optional[Literal["standard", "hd"]] = None
    user: Optional[str] = None

    @model_validator(mode="after")
    def _check_paired_dimensions(self):
        if isinstance(self.image, list) and not self.image:
            raise ValueError("image must contain at least one input image")
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "width and height must be sent together; got width="
                f"{self.width!r}, height={self.height!r}")
        return self


class ImageObject(OpenAIBaseModel):
    """Generated image object in the response."""
    b64_json: Optional[str] = None
    url: Optional[str] = None
    path: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(OpenAIBaseModel):
    """Response from image generation endpoint.

    ``output_format`` reports the encoding actually applied to the
    returned bytes / files so clients can decode or label the payload
    correctly. Image encoders are ``"png"``/``"webp"``/``"jpeg"``;
    tensor formats are ``"safetensors"``/``"pt"``.
    """

    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[ImageObject]
    output_format: Literal["png", "webp", "jpeg", "safetensors", "pt"] = "png"
    quality: Literal["low", "medium", "high"] = "medium"
    size: Optional[str] = None


class MediaReferenceItem(OpenAIBaseModel):
    """One media reference (image / video / audio) for conditioning (mirrors ``MediaRef``).

    ``content`` carries base64-encoded bytes, optionally as a ``data:`` URI. The
    request field it sits in (``image_reference`` / ``video_reference`` /
    ``audio_reference``) fixes the modality. ``role`` is required only for models
    that accept more than one role for that modality (e.g. image first/last
    frame); single-role models infer it.
    """

    content: str = Field(
        description="Base64-encoded media bytes, optionally as a ``data:`` URI."
    )
    role: Optional[str] = Field(
        default=None,
        description=
        ("Reference role (e.g. 'reference', 'first_frame', 'last_frame'). "
         "Required only when the model accepts multiple roles for the modality."
         ),
    )


class VideoGenerationRequest(OpenAIBaseModel):
    """Video generation request (extended API).

    Universal per-request fields map 1:1 to :class:`VisualGenParams`.
    Model-specific knobs travel through ``extra_params``. Unknown
    top-level fields are rejected with HTTP 422 via the inherited
    ``extra="forbid"`` policy.
    """

    # Prompt + transport
    prompt: str
    response_format: Literal["file", "path"] = "file"
    format: Literal["mp4", "avi", "auto", "safetensors", "pt"] = Field(
        default="auto",
        description=(
            "Generation content encoding format. Video encoders write "
            "``mp4``/``avi``/``auto``; tensor encoders write "
            "``safetensors``/``pt`` and carry video, audio, and scalar "
            "metadata (frame rate, audio sample rate) in one payload."),
    )
    seed: Optional[int] = Field(default=None,
                                ge=0,
                                description="Random seed for reproducibility.")
    image_reference: Optional[Union[
        str, UploadFile, MediaReferenceItem,
        List[Union[str, MediaReferenceItem]]]] = Field(
            default=None,
            description=
            ("Image reference(s) conditioning generation (e.g. image-to-video "
             "first frame). A JSON string is base64 bytes (raw or ``data:`` "
             "URI), an ``http(s)`` URL, or a ``file://`` path; or send a "
             "``{content, role}`` object or a list of them; multipart uploads a "
             "single image file. PNG or JPEG only — HEIF/AVIF are not supported."
             ),
        )
    video_reference: Optional[Union[
        str, UploadFile, MediaReferenceItem,
        List[Union[str, MediaReferenceItem]]]] = Field(
            default=None,
            description=
            ("Video reference(s) conditioning generation (video-to-video). A "
             "JSON string is base64 bytes (raw or ``data:`` URI), an ``http(s)`` "
             "URL, or a ``file://`` path; or send a ``{content}`` object or a "
             "list of them; multipart uploads a single video file. MP4 or AVI, "
             "with H.264 the tested codec and others best-effort."),
        )
    audio_reference: Optional[Union[
        str, UploadFile, MediaReferenceItem,
        List[Union[str, MediaReferenceItem]]]] = Field(
            default=None,
            description=(
                "Audio reference(s) conditioning generation. A JSON string is "
                "base64 bytes (raw or ``data:`` URI), an ``http(s)`` URL, or a "
                "``file://`` path; or send a ``{content}`` object or a list of "
                "them; multipart uploads a single audio file. Accepted only by "
                "models that declare an audio reference slot."),
        )
    input_reference: Optional[Union[str, UploadFile]] = Field(
        default=None,
        description=
        ("Deprecated. A single image or video reference, routed by content "
         "signature to image-to-video or video-to-video. Kept for backward "
         "compatibility; prefer the typed ``image_reference`` / "
         "``video_reference`` fields, which take precedence — this field is "
         "ignored whenever a typed ``image_reference`` or ``video_reference`` "
         "is provided."),
    )

    # Resolution
    size: Optional[str] = Field(default=None, pattern=r"^(\d+x\d+|auto)$")
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)

    # Frame budget. ``num_frames`` is preferred; if absent the engine
    # derives it from ``seconds * frame_rate``. ``frame_rate`` is the
    # canonical name (matches the Python field); ``fps`` is an alias for
    # OpenAI-shape clients via ``populate_by_name=True``.
    # All three constrain to strictly positive values so a zero
    # ``frame_rate`` (division-by-zero in the AVI fallback) or a
    # negative ``num_frames`` are rejected with HTTP 422 before
    # reaching the encoder.
    # Upper bounds keep request-boundary protection against requests
    # that can exhaust GPU memory or pin the server on unbounded work.
    # The numbers are generous (a minute of video at 120 fps) so common
    # workloads pass; clients that need larger budgets can lift the cap
    # at deployment time.
    num_frames: Optional[int] = Field(default=None, gt=0, le=7200)
    seconds: Optional[float] = Field(default=None, gt=0, le=60.0)
    frame_rate: Optional[float] = Field(default=None,
                                        alias="fps",
                                        gt=0,
                                        le=120.0)

    # TRT-LLM-supported per-request params (1:1 with VisualGenParams)
    num_inference_steps: Optional[int] = Field(default=None, gt=0)
    guidance_scale: Optional[float] = Field(default=None, gt=0)
    max_sequence_length: Optional[int] = Field(default=None, gt=0)
    negative_prompt: Optional[str] = None

    # Model-specific overflow
    extra_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Model-specific parameters forwarded to the underlying pipeline. "
            "See per-model docs for accepted keys."),
    )

    # Accepted-but-ignored OpenAI-shaped field
    model: Optional[str] = None

    @model_validator(mode="after")
    def _check_paired_dimensions(self):
        """Reject sending exactly one of ``width`` / ``height``."""
        if (self.width is None) != (self.height is None):
            raise ValueError(
                "width and height must be sent together; got width="
                f"{self.width!r}, height={self.height!r}")
        return self

    @field_validator("response_format", mode="before")
    @classmethod
    def _reject_removed_response_format(cls, value):
        """Give migrating callers an actionable error for removed values.

        ``url``/``b64_json`` were valid before the transport rewrite; run
        before the ``Literal`` check so the error names the replacement
        instead of the generic "Input should be 'file' or 'path'".
        """
        removed = {
            "url":
            ("'url' was removed for video; use 'file' (raw bytes -- the "
             "old 'url' behavior, renamed) or 'path' (server-side path)."),
            "b64_json": ("'b64_json' was removed for video; use 'file' (raw "
                         "bytes) or 'path' (server-side path)."),
        }
        if isinstance(value, str) and value in removed:
            raise ValueError(removed[value])
        return value


class VideoJob(OpenAIBaseModel):
    """Metadata for an asynchronous video generation job.

    Follows the OpenAI Videos API specification:
    https://platform.openai.com/docs/api-reference/videos
    """
    completed_at: Optional[int] = Field(
        default=None, description="Unix timestamp of completion")
    created_at: int = Field(description="Unix timestamp of creation")
    error: Optional[str] = Field(default=None,
                                 description="Error message if failed")
    expires_at: Optional[int] = Field(
        default=None, description="Unix timestamp of expiration")
    id: str = Field(description="Unique identifier for the video")
    model: str = Field(description="The model used for generation")
    object: str = Field(default="video", description="Object type")
    progress: Optional[int] = Field(
        default=None,
        description="Progress of the video generation job (0-100)")
    prompt: str = Field(description="The prompt used to generate the video")
    status: Literal["queued", "generating", "postprocessing", "completed",
                    "failed"] = Field(description=(
                        "Current status of the video generation job. "
                        "``generating`` (model inference) becomes "
                        "``postprocessing`` (encode and/or write the output "
                        "file) when inference finishes, then ``completed`` "
                        "once downloadable via ``/content``."))

    # Video properties
    duration: Optional[float] = Field(default=None,
                                      description="Video duration in seconds")
    fps: Optional[float] = Field(
        default=None,
        description=(
            "Frames per second. Float to preserve cinematic rates such "
            "as 23.976 / 29.97 that some encoders / pipelines use."),
    )
    size: Optional[str] = Field(default=None,
                                description="Video dimensions in 'WxH' format")
    # exclude=True: internal file-location for /content resolution + delete;
    # never on the wire (the path payload is the hand-built {id, output_path}
    # envelope in /content), so status/list model_dump() stays status-only.
    output_path: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Server-side saved path (internal; excluded from the wire)."
    )
    output_paths: Optional[List[str]] = Field(
        default=None,
        exclude=True,
        description=
        "Server-side paths for n>1 (internal; excluded from the wire).")
    # exclude=True internal timings, never on the wire (status/list
    # model_dump() stays status-only). ``request_started`` carries the
    # steady-clock ``server_arrival_time``; the background task computes
    # ``total`` from it and stores the header timings
    # (``generation``/``denoise``/``total``) in ``timing_metrics`` so
    # ``/content`` emits the same Server-Timing header as the sync route.
    request_started: Optional[float] = Field(default=None, exclude=True)
    timing_metrics: Optional[Dict[str, float]] = Field(default=None,
                                                       exclude=True)
    response_format: Optional[Literal["file", "path"]] = Field(
        default=None,
        description=(
            "Transport the client requested. ``GET /v1/videos/{id}/content`` "
            "honors this: ``path`` returns the server-side output path(s) in a "
            "JSON envelope; ``file`` (or unset) returns the file as a "
            "``FileResponse`` download."),
    )


class VideoJobList(OpenAIBaseModel):
    """Response from listing video jobs endpoint."""
    data: List[VideoJob] = Field(description="List of video jobs")
    object: str = Field(default="list", description="Object type")


UCompletionRequest = Union[CompletionRequest, ChatCompletionRequest]
UCompletionResponse = Union[CompletionResponse, ChatCompletionResponse]
