# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py
import base64
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import \
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated, Required, TypedDict

from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi import SamplingParams


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields & allow to initialize by both alias and field name
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = True


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "tensorrt_llm"


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class ResponseFormat(OpenAIBaseModel):
    # type must be "json_object" or "text"
    type: Literal["text", "json_object"]


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

    @staticmethod
    def to_disaggregated_params(
            tllm_disagg_params: LlmDisaggregatedParams) -> DisaggregatedParams:
        if tllm_disagg_params is None:
            return None
        else:
            encoded_opaque_state = base64.b64encode(
                tllm_disagg_params.opaque_state).decode(
                    "utf-8") if tllm_disagg_params is not None else None
            return DisaggregatedParams(
                request_type=tllm_disagg_params.request_type,
                first_gen_tokens=tllm_disagg_params.first_gen_tokens,
                ctx_request_id=tllm_disagg_params.ctx_request_id,
                encoded_opaque_state=encoded_opaque_state,
                draft_tokens=tllm_disagg_params.draft_tokens)


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


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
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

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
        description=(
            "Similar to chat completion, this parameter specifies the format of "
            "output. Only {'type': 'json_object'} or {'type': 'text' } is "
            "supported."),
    )

    disaggregated_params: Optional[DisaggregatedParams] = Field(
        default=None,
        description=("Parameters for disaggregated serving"),
    )

    # doc: end-completion-extra-params

    def to_sampling_params(self) -> SamplingParams:
        sampling_params = SamplingParams(
            best_of=self.best_of,
            frequency_penalty=self.frequency_penalty,
            return_log_probs=self.logprobs,
            max_tokens=self.max_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=self.temperature,
            top_p=self.top_p,

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

            # completion-extra-params
            add_special_tokens=self.add_special_tokens,
        )
        return sampling_params

    def to_llm_disaggregated_params(self) -> LlmDisaggregatedParams:
        if self.disaggregated_params is None:
            return None
        else:
            opaque_state = base64.b64decode(
                self.disaggregated_params.encoded_opaque_state
            ) if self.disaggregated_params.encoded_opaque_state is not None else None

            return LlmDisaggregatedParams(
                request_type=self.disaggregated_params.request_type,
                first_gen_tokens=self.disaggregated_params.first_gen_tokens,
                ctx_request_id=self.disaggregated_params.ctx_request_id,
                opaque_state=opaque_state,
                draft_tokens=self.disaggregated_params.draft_tokens)

    def model_post_init(self, __context: Any) -> None:
        if self.best_of is None:
            self.best_of = self.n

    @model_validator(mode="after")
    def check_beam_search(self):
        if (self.n > 1 or self.best_of > 1) and not self.use_beam_search:
            raise ValueError(
                "Only support one response per prompt without beam search")
        return self

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if ("top_logprobs" in data and data.get("top_logprobs")) or \
            ("logprobs" in data and data.get("logprobs")):
            raise ValueError("returning log probs is not supported")
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
    def verify_multi_responses(cls, data):
        best_of = data.get("best_of")
        n = data.get("n")
        if best_of and n and best_of < n:
            raise ValueError("best_of should not be smaller than n")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_response_format(cls, data):
        if data.get("response_format"):
            raise ValueError("response_format is not supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_suffix(cls, data):
        if data.get("suffix"):
            raise ValueError("suffix is not supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_special_tokens(cls, data):
        if data.get("skip_special_tokens") or data.get("add_special_tokens") or \
            data.get("spaces_between_special_tokens"):
            raise ValueError(
                "special_tokens related settings are not supported")
        return data


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(
        default_factory=lambda: f"chatcmpl-tool-{str(uuid.uuid4().hex)}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(OpenAIBaseModel):
    role: str
    content: str
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


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam]


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


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
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_completion_tokens: int = Field(default=16,
                                       validation_alias='max_tokens')
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    user: Optional[str] = None

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

    # doc: end-chat-completion-extra-params

    def to_sampling_params(self) -> SamplingParams:

        sampling_params = SamplingParams(
            frequency_penalty=self.frequency_penalty,
            return_log_probs=self.logprobs,
            max_tokens=self.max_completion_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=self.temperature,

            # chat-completion-sampling-params
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            top_k=self.top_k,
            top_p=self.top_p,
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

            # chat-completion-extra-params
            add_special_tokens=self.add_special_tokens,
        )
        return sampling_params

    def model_post_init(self, __context: Any) -> None:
        if self.best_of is None:
            self.best_of = self.n

    @model_validator(mode="after")
    def check_beam_search(self):
        if (self.n > 1 or self.best_of > 1) and not self.use_beam_search:
            raise ValueError(
                "Only support one response per prompt without beam search")
        return self

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
        if "tool_choice" in data and data["tool_choice"] != "none":
            if not isinstance(data["tool_choice"], dict):
                raise ValueError("Currently only named tools are supported.")
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        top_logprobs = data.get("top_logprobs")
        if top_logprobs is not None and top_logprobs > 0:
            raise ValueError("top_logprobs is not supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def verify_multi_responses(cls, data):
        best_of, n = data.get("best_of"), data.get("n")
        if best_of and n and best_of < n:
            raise ValueError("best_of should not be smaller than n")
        return data

    @model_validator(mode="before")
    @classmethod
    def verify_logit_processor(cls, data):
        if data.get("logit_bias"):
            raise ValueError("logit bias is not supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_response_format(cls, data):
        if data.get("response_format"):
            raise ValueError("response_format is not supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_suffix(cls, data):
        if data.get("suffix"):
            raise ValueError("suffix is not supported")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_special_tokens(cls, data):
        if data.get("skip_special_tokens") or data.get("add_special_tokens") or \
            data.get("spaces_between_special_tokens"):
            raise ValueError(
                "special_tokens related settings are not supported")
        return data
