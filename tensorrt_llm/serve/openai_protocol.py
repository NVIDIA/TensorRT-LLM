# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py
import base64
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import xgrammar
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import \
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam
from openai.types.responses import (ResponseFunctionToolCall,
                                    ResponseInputItemParam, ResponseOutputItem,
                                    ResponsePrompt, ResponseReasoningItem,
                                    ResponseStatus, ResponseTextConfig)
from openai.types.responses.response import ToolChoice
from openai.types.responses.tool import Tool
from openai.types.shared import Metadata, Reasoning
from openai_harmony import ReasoningEffort
from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)
from typing_extensions import Annotated, Required, TypeAlias, TypedDict

from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import GuidedDecodingParams, SamplingParams


def _logit_bias_to_embedding_bias(logit_bias: Optional[Dict[str, float]],
                                  vocab_size: int) -> Optional[torch.Tensor]:
    """Convert OpenAI logit_bias dict to embedding_bias tensor for sampling."""
    if logit_bias is None:
        return None

    # Create 1D zeros tensor as expected by executor API (will be unsqueezed to [1, vocab_size] internally)
    embedding_bias = torch.zeros(vocab_size, dtype=torch.float32)

    # Apply biases for specified token IDs
    for token_str, bias in logit_bias.items():
        try:
            token_id = int(token_str)
            if 0 <= token_id < vocab_size:
                embedding_bias[token_id] = bias
            else:
                raise ValueError(
                    f"Token ID {token_id} out of vocabulary range [0, {vocab_size})"
                )
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(
                    f"Invalid logit_bias key '{token_str}': must be a valid integer token ID"
                )
            raise

    return embedding_bias


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields & allow to initialize by both alias and field name
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = True


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


class ResponseFormat(OpenAIBaseModel):
    # type must be one of "text", "json", "json_object", or "structural_tag"
    type: Literal["text", "json", "json_object", "regex", "ebnf",
                  "structural_tag"]
    schema: Optional[dict] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    format: Optional[xgrammar.structural_tag.Format] = None


class DisaggregatedParams(OpenAIBaseModel):
    request_type: str
    first_gen_tokens: Optional[List[int]] = None
    ctx_request_id: Optional[int] = None
    encoded_opaque_state: Optional[str] = None
    draft_tokens: Optional[List[int]] = None


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


def _response_format_to_guided_decoding_params(
    response_format: Optional[ResponseFormat]
) -> Optional[GuidedDecodingParams]:
    if response_format is None:
        return None
    elif response_format.type == "text":
        return None
    elif response_format.type == "json":
        if response_format.schema is None:
            raise ValueError(
                "The 'schema' field is required when response_format.type is 'json'."
            )
        return GuidedDecodingParams(json=response_format.schema)
    elif response_format.type == "json_object":
        return GuidedDecodingParams(json_object=True)
    elif response_format.type == "regex":
        return GuidedDecodingParams(regex=response_format.regex)
    elif response_format.type == "ebnf":
        return GuidedDecodingParams(grammar=response_format.ebnf)
    elif response_format.type == "structural_tag":
        return GuidedDecodingParams(
            structural_tag=response_format.model_dump_json(by_alias=True,
                                                           exclude_none=True))
    else:
        raise ValueError(f"Unsupported response format: {response_format.type}")


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

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: int = 0
    top_p_min: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    return_context_logits: bool = False
    detokenize: bool = True
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

    # doc: end-completion-extra-params

    def to_sampling_params(self, vocab_size: int = 32000) -> SamplingParams:
        sampling_params = SamplingParams(
            best_of=self.best_of,
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=(self.temperature
                         if self.temperature is not None else 1.0),
            top_p=(self.top_p if self.top_p is not None else 1.0),

            # completion-sampling-params
            use_beam_search=self.use_beam_search,
            top_k=self.top_k,
            top_p_min=self.top_p_min if self.top_p_min > 0 else None,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
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

            # logits_bias
            embedding_bias=_logit_bias_to_embedding_bias(
                self.logit_bias, vocab_size),

            # completion-extra-params
            add_special_tokens=self.add_special_tokens,

            # TODO: migrate to use logprobs and prompt_logprobs
            _return_log_probs=bool(self.logprobs),
        )
        return sampling_params

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if data.get("logprobs"):
            raise ValueError("logprobs is not supported")
        return data

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
    type: Optional[Literal["function"]] = None
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
    top_logprobs: List[ChatCompletionLogProb] = None


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[OpenAIChatCompletionContentPartParam,
                                       CustomChatCompletionContentPartParam]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


class ReasoningAssistantMessage(ChatCompletionAssistantMessageParam):
    """Assistant message that includes reasoning tokens."""
    reasoning: Optional[str]


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
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


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    # Add prompt_tokens_ids to the request to remove the tokenization
    # in the generation server in disaggreated serving
    prompt_token_ids: Optional[List[int]] = None
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
    tool_choice: Optional[Union[Literal["none", "auto"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    user: Optional[str] = None
    reasoning_effort: Optional[ReasoningEffort | Literal[
        "low", "medium", "high"]] = Field(
            default=ReasoningEffort.LOW,
            description=(
                "The level of reasoning effort to use. Controls how much "
                "reasoning is shown in the model's response. Options: "
                "'low', 'medium', 'high'."),
        )

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = 0
    top_p_min: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
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

    disaggregated_params: Optional[DisaggregatedParams] = Field(
        default=None,
        description=("Parameters for disaggregated serving"),
    )

    cache_salt: Optional[str] = Field(
        default=None,
        description=
        ("If specified, KV cache will be salted with the provided string "
         "to limit the kv cache reuse on with the requests having the same string."
         ))

    # doc: end-chat-completion-extra-params

    def to_sampling_params(self,
                           vocab_size: int = 32000,
                           gather_generation_logits: bool = False,
                           backend: Optional[str] = None) -> SamplingParams:
        sampling_params = SamplingParams(
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_completion_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=(self.temperature
                         if self.temperature is not None else 1.0),

            # chat-completion-sampling-params
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            top_k=self.top_k,
            top_p=(self.top_p if self.top_p is not None else 1.0),
            top_p_min=self.top_p_min if self.top_p_min > 0 else None,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
            stop_token_ids=self.stop_token_ids,
            include_stop_str_in_output=self.include_stop_str_in_output,
            ignore_eos=self.ignore_eos,
            min_tokens=self.min_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            guided_decoding=_response_format_to_guided_decoding_params(
                self.response_format),

            # logits_bias
            embedding_bias=_logit_bias_to_embedding_bias(
                self.logit_bias, vocab_size),

            # chat-completion-extra-params
            add_special_tokens=self.add_special_tokens,
        )
        if self.logprobs:
            logprobs = 1 if not self.top_logprobs else self.top_logprobs
            if backend == "pytorch":
                sampling_params.logprobs = logprobs
            else:
                if gather_generation_logits:
                    sampling_params.logprobs = logprobs
                elif self.top_logprobs:
                    raise ValueError(
                        "`gather_generation_logits` must be `True` to use `top_logprobs`"
                    )
                else:
                    sampling_params._return_log_probs = True
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
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"
        if "tool_choice" in data and data["tool_choice"] != "none":
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0:
                raise ValueError("top_logprobs must be positive or zero")
            if not data.get("logprobs"):
                raise ValueError(
                    "logprobs must be true when using top_logprobs")
        return data

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


ResponseInputOutputItem: TypeAlias = Union[ResponseInputItemParam,
                                           ResponseReasoningItem,
                                           ResponseFunctionToolCall]


class ResponsesRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/responses/create
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
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Metadata] = None
    model: str
    parallel_tool_calls: Optional[bool] = False
    previous_response_id: Optional[str] = None
    prompt: Optional[ResponsePrompt] = None
    reasoning: Optional[Reasoning] = None
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
        default_max_tokens: int,
        default_sampling_params: Optional[dict] = None,
    ) -> SamplingParams:
        if self.max_output_tokens is None:
            max_tokens = default_max_tokens
        else:
            max_tokens = min(self.max_output_tokens, default_max_tokens)

        default_sampling_params = default_sampling_params or {}
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
        stop_token_ids = default_sampling_params.get("stop_token_ids")

        # Structured output
        guided_decoding = None
        if self.text is not None and self.text.format is not None:
            response_format = self.text.format
            if response_format.type == "json_schema":
                guided_decoding = GuidedDecodingParams(
                    json=response_format.schema_)
            elif response_format.type == "json_object":
                raise NotImplementedError("json_object is not supported")

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            stop_token_ids=stop_token_ids,
            guided_decoding=guided_decoding,
        )

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


class InputTokensDetails(OpenAIBaseModel):
    cached_tokens: int


class OutputTokensDetails(OpenAIBaseModel):
    reasoning_tokens: int


class ResponseUsage(OpenAIBaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


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
    max_output_tokens: int
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


def encode_opaque_state(opaque_state: Optional[bytes]) -> Optional[str]:
    if opaque_state is None:
        return None
    return base64.b64encode(opaque_state).decode("utf-8")


def decode_opaque_state(encoded_opaque_state: Optional[str]) -> Optional[bytes]:
    if encoded_opaque_state is None:
        return None
    return base64.b64decode(encoded_opaque_state)


def to_disaggregated_params(
        tllm_disagg_params: LlmDisaggregatedParams) -> DisaggregatedParams:
    if tllm_disagg_params is None:
        return None
    return DisaggregatedParams(
        request_type=tllm_disagg_params.request_type,
        first_gen_tokens=tllm_disagg_params.first_gen_tokens,
        ctx_request_id=tllm_disagg_params.ctx_request_id,
        encoded_opaque_state=encode_opaque_state(
            tllm_disagg_params.opaque_state),
        draft_tokens=tllm_disagg_params.draft_tokens)


def to_llm_disaggregated_params(
        disaggregated_params: DisaggregatedParams) -> LlmDisaggregatedParams:
    if disaggregated_params is None:
        return None
    return LlmDisaggregatedParams(
        request_type=disaggregated_params.request_type,
        first_gen_tokens=disaggregated_params.first_gen_tokens,
        ctx_request_id=disaggregated_params.ctx_request_id,
        opaque_state=decode_opaque_state(
            disaggregated_params.encoded_opaque_state),
        draft_tokens=disaggregated_params.draft_tokens)
