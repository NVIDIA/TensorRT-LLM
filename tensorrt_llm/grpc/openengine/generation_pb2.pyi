from google.protobuf import struct_pb2 as _struct_pb2
from . import error_pb2 as _error_pb2
from . import kv_pb2 as _kv_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Modality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODALITY_UNSPECIFIED: _ClassVar[Modality]
    MODALITY_IMAGE: _ClassVar[Modality]
    MODALITY_VIDEO: _ClassVar[Modality]
    MODALITY_AUDIO: _ClassVar[Modality]

class FinishReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FINISH_REASON_UNSPECIFIED: _ClassVar[FinishReason]
    FINISH_REASON_STOP: _ClassVar[FinishReason]
    FINISH_REASON_LENGTH: _ClassVar[FinishReason]
    FINISH_REASON_CANCELLED: _ClassVar[FinishReason]
MODALITY_UNSPECIFIED: Modality
MODALITY_IMAGE: Modality
MODALITY_VIDEO: Modality
MODALITY_AUDIO: Modality
FINISH_REASON_UNSPECIFIED: FinishReason
FINISH_REASON_STOP: FinishReason
FINISH_REASON_LENGTH: FinishReason
FINISH_REASON_CANCELLED: FinishReason

class TokenIds(_message.Message):
    __slots__ = ("ids",)
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, ids: _Optional[_Iterable[int]] = ...) -> None: ...

class MediaItem(_message.Message):
    __slots__ = ("modality", "url", "data_uri", "raw_bytes", "mime_type", "uuid")
    MODALITY_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    DATA_URI_FIELD_NUMBER: _ClassVar[int]
    RAW_BYTES_FIELD_NUMBER: _ClassVar[int]
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    modality: Modality
    url: str
    data_uri: str
    raw_bytes: bytes
    mime_type: str
    uuid: str
    def __init__(self, modality: _Optional[_Union[Modality, str]] = ..., url: _Optional[str] = ..., data_uri: _Optional[str] = ..., raw_bytes: _Optional[bytes] = ..., mime_type: _Optional[str] = ..., uuid: _Optional[str] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("request_id", "model", "prompt", "token_ids", "sampling", "stopping", "response", "kv", "guided", "media", "lora_name", "extra")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_FIELD_NUMBER: _ClassVar[int]
    STOPPING_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    KV_FIELD_NUMBER: _ClassVar[int]
    GUIDED_FIELD_NUMBER: _ClassVar[int]
    MEDIA_FIELD_NUMBER: _ClassVar[int]
    LORA_NAME_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    model: str
    prompt: str
    token_ids: TokenIds
    sampling: SamplingParams
    stopping: StoppingOptions
    response: ResponseOptions
    kv: KvOptions
    guided: GuidedDecoding
    media: _containers.RepeatedCompositeFieldContainer[MediaItem]
    lora_name: str
    extra: _struct_pb2.Struct
    def __init__(self, request_id: _Optional[str] = ..., model: _Optional[str] = ..., prompt: _Optional[str] = ..., token_ids: _Optional[_Union[TokenIds, _Mapping]] = ..., sampling: _Optional[_Union[SamplingParams, _Mapping]] = ..., stopping: _Optional[_Union[StoppingOptions, _Mapping]] = ..., response: _Optional[_Union[ResponseOptions, _Mapping]] = ..., kv: _Optional[_Union[KvOptions, _Mapping]] = ..., guided: _Optional[_Union[GuidedDecoding, _Mapping]] = ..., media: _Optional[_Iterable[_Union[MediaItem, _Mapping]]] = ..., lora_name: _Optional[str] = ..., extra: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SamplingParams(_message.Message):
    __slots__ = ("temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty", "seed", "num_sequences")
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    MIN_P_FIELD_NUMBER: _ClassVar[int]
    FREQUENCY_PENALTY_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_PENALTY_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    NUM_SEQUENCES_FIELD_NUMBER: _ClassVar[int]
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    seed: int
    num_sequences: int
    def __init__(self, temperature: _Optional[float] = ..., top_p: _Optional[float] = ..., top_k: _Optional[int] = ..., min_p: _Optional[float] = ..., frequency_penalty: _Optional[float] = ..., presence_penalty: _Optional[float] = ..., repetition_penalty: _Optional[float] = ..., seed: _Optional[int] = ..., num_sequences: _Optional[int] = ...) -> None: ...

class StoppingOptions(_message.Message):
    __slots__ = ("max_tokens", "min_tokens", "conditions", "ignore_eos", "include_stop_in_output")
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MIN_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CONDITIONS_FIELD_NUMBER: _ClassVar[int]
    IGNORE_EOS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_STOP_IN_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    max_tokens: int
    min_tokens: int
    conditions: _containers.RepeatedCompositeFieldContainer[StopCondition]
    ignore_eos: bool
    include_stop_in_output: bool
    def __init__(self, max_tokens: _Optional[int] = ..., min_tokens: _Optional[int] = ..., conditions: _Optional[_Iterable[_Union[StopCondition, _Mapping]]] = ..., ignore_eos: bool = ..., include_stop_in_output: bool = ...) -> None: ...

class ResponseOptions(_message.Message):
    __slots__ = ("return_prompt_logprobs", "prompt_candidates", "return_output_logprobs", "output_candidates", "prompt_logprob_start")
    RETURN_PROMPT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    RETURN_OUTPUT_LOGPROBS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    PROMPT_LOGPROB_START_FIELD_NUMBER: _ClassVar[int]
    return_prompt_logprobs: bool
    prompt_candidates: CandidateTokenSelection
    return_output_logprobs: bool
    output_candidates: CandidateTokenSelection
    prompt_logprob_start: int
    def __init__(self, return_prompt_logprobs: bool = ..., prompt_candidates: _Optional[_Union[CandidateTokenSelection, _Mapping]] = ..., return_output_logprobs: bool = ..., output_candidates: _Optional[_Union[CandidateTokenSelection, _Mapping]] = ..., prompt_logprob_start: _Optional[int] = ...) -> None: ...

class CandidateTokenSelection(_message.Message):
    __slots__ = ("top_n", "token_ids", "all")
    TOP_N_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    ALL_FIELD_NUMBER: _ClassVar[int]
    top_n: int
    token_ids: TokenIds
    all: AllCandidates
    def __init__(self, top_n: _Optional[int] = ..., token_ids: _Optional[_Union[TokenIds, _Mapping]] = ..., all: _Optional[_Union[AllCandidates, _Mapping]] = ...) -> None: ...

class AllCandidates(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class KvOptions(_message.Message):
    __slots__ = ("session", "bypass_prefix_cache", "cache_salt")
    SESSION_FIELD_NUMBER: _ClassVar[int]
    BYPASS_PREFIX_CACHE_FIELD_NUMBER: _ClassVar[int]
    CACHE_SALT_FIELD_NUMBER: _ClassVar[int]
    session: _kv_pb2.KvSessionRef
    bypass_prefix_cache: bool
    cache_salt: str
    def __init__(self, session: _Optional[_Union[_kv_pb2.KvSessionRef, _Mapping]] = ..., bypass_prefix_cache: bool = ..., cache_salt: _Optional[str] = ...) -> None: ...

class StopCondition(_message.Message):
    __slots__ = ("stop_text", "stop_token_id")
    STOP_TEXT_FIELD_NUMBER: _ClassVar[int]
    STOP_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    stop_text: str
    stop_token_id: int
    def __init__(self, stop_text: _Optional[str] = ..., stop_token_id: _Optional[int] = ...) -> None: ...

class GuidedDecoding(_message.Message):
    __slots__ = ("json_schema", "regex", "ebnf_grammar", "structural_tag", "choice", "json_object", "backend")
    JSON_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    REGEX_FIELD_NUMBER: _ClassVar[int]
    EBNF_GRAMMAR_FIELD_NUMBER: _ClassVar[int]
    STRUCTURAL_TAG_FIELD_NUMBER: _ClassVar[int]
    CHOICE_FIELD_NUMBER: _ClassVar[int]
    JSON_OBJECT_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    json_schema: str
    regex: str
    ebnf_grammar: str
    structural_tag: str
    choice: ChoiceConstraint
    json_object: JsonObjectConstraint
    backend: str
    def __init__(self, json_schema: _Optional[str] = ..., regex: _Optional[str] = ..., ebnf_grammar: _Optional[str] = ..., structural_tag: _Optional[str] = ..., choice: _Optional[_Union[ChoiceConstraint, _Mapping]] = ..., json_object: _Optional[_Union[JsonObjectConstraint, _Mapping]] = ..., backend: _Optional[str] = ...) -> None: ...

class ChoiceConstraint(_message.Message):
    __slots__ = ("choices",)
    CHOICES_FIELD_NUMBER: _ClassVar[int]
    choices: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, choices: _Optional[_Iterable[str]] = ...) -> None: ...

class JsonObjectConstraint(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("request_id", "prompt", "token", "prefill_ready", "finished", "error", "usage")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    PREFILL_READY_FIELD_NUMBER: _ClassVar[int]
    FINISHED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    USAGE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prompt: PromptOutput
    token: TokenOutput
    prefill_ready: PrefillReady
    finished: GenerationFinished
    error: _error_pb2.EngineError
    usage: Usage
    def __init__(self, request_id: _Optional[str] = ..., prompt: _Optional[_Union[PromptOutput, _Mapping]] = ..., token: _Optional[_Union[TokenOutput, _Mapping]] = ..., prefill_ready: _Optional[_Union[PrefillReady, _Mapping]] = ..., finished: _Optional[_Union[GenerationFinished, _Mapping]] = ..., error: _Optional[_Union[_error_pb2.EngineError, _Mapping]] = ..., usage: _Optional[_Union[Usage, _Mapping]] = ...) -> None: ...

class PromptOutput(_message.Message):
    __slots__ = ("tokens",)
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedCompositeFieldContainer[TokenInfo]
    def __init__(self, tokens: _Optional[_Iterable[_Union[TokenInfo, _Mapping]]] = ...) -> None: ...

class TokenOutput(_message.Message):
    __slots__ = ("output_index", "tokens", "text")
    OUTPUT_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    output_index: int
    tokens: _containers.RepeatedCompositeFieldContainer[TokenInfo]
    text: str
    def __init__(self, output_index: _Optional[int] = ..., tokens: _Optional[_Iterable[_Union[TokenInfo, _Mapping]]] = ..., text: _Optional[str] = ...) -> None: ...

class TokenInfo(_message.Message):
    __slots__ = ("token_id", "token", "logprob", "rank", "candidates")
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    token_id: int
    token: str
    logprob: float
    rank: int
    candidates: _containers.RepeatedCompositeFieldContainer[LogProb]
    def __init__(self, token_id: _Optional[int] = ..., token: _Optional[str] = ..., logprob: _Optional[float] = ..., rank: _Optional[int] = ..., candidates: _Optional[_Iterable[_Union[LogProb, _Mapping]]] = ...) -> None: ...

class LogProb(_message.Message):
    __slots__ = ("token_id", "logprob", "token", "rank")
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    LOGPROB_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    token_id: int
    logprob: float
    token: str
    rank: int
    def __init__(self, token_id: _Optional[int] = ..., logprob: _Optional[float] = ..., token: _Optional[str] = ..., rank: _Optional[int] = ...) -> None: ...

class PrefillReady(_message.Message):
    __slots__ = ("kv_session",)
    KV_SESSION_FIELD_NUMBER: _ClassVar[int]
    kv_session: _kv_pb2.KvSessionRef
    def __init__(self, kv_session: _Optional[_Union[_kv_pb2.KvSessionRef, _Mapping]] = ...) -> None: ...

class GenerationFinished(_message.Message):
    __slots__ = ("output_index", "reason", "message", "stop_match")
    OUTPUT_INDEX_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STOP_MATCH_FIELD_NUMBER: _ClassVar[int]
    output_index: int
    reason: FinishReason
    message: str
    stop_match: StopMatch
    def __init__(self, output_index: _Optional[int] = ..., reason: _Optional[_Union[FinishReason, str]] = ..., message: _Optional[str] = ..., stop_match: _Optional[_Union[StopMatch, _Mapping]] = ...) -> None: ...

class StopMatch(_message.Message):
    __slots__ = ("stop_token_id", "stop_text", "eos_token_id")
    STOP_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    STOP_TEXT_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    stop_token_id: int
    stop_text: str
    eos_token_id: int
    def __init__(self, stop_token_id: _Optional[int] = ..., stop_text: _Optional[str] = ..., eos_token_id: _Optional[int] = ...) -> None: ...

class Usage(_message.Message):
    __slots__ = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens", "reasoning_tokens")
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    COMPLETION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    REASONING_TOKENS_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int
    reasoning_tokens: int
    def __init__(self, prompt_tokens: _Optional[int] = ..., completion_tokens: _Optional[int] = ..., total_tokens: _Optional[int] = ..., cached_prompt_tokens: _Optional[int] = ..., reasoning_tokens: _Optional[int] = ...) -> None: ...
