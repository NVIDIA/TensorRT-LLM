# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine inference service adapter for the TensorRT-LLM LLM API."""

import asyncio
import base64
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit

import grpc
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict
from openengine.v1 import error_pb2, generation_pb2, kv_pb2, openengine_pb2_grpc

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.postproc_worker import PostprocArgs, PostprocParams
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

_RESPONSE_STALL_TIMEOUT_SECONDS = 30.0

# OpenEngine has no native request-type field, so the phase of a disaggregated
# request is carried in the request's `extra` Struct under this key. A request
# that carries a `kv.session` handle is always treated as generation_only.
_REQUEST_TYPE_KEY = "request_type"
_DISAGG_REQUEST_TYPES = {"context_only", "generation_only", "context_and_generation"}

# Floor for logprob values so a -inf (masked token) is JSON/proto-safe, matching
# the HTTP /v1/completions clamp in create_logprobs.
_MIN_LOGPROB = -9999.0


def _clamp_logprob(value: Any) -> float:
    return max(float(value), _MIN_LOGPROB)


class UnsupportedFeatureError(ValueError):
    """Raised when OpenEngine requests a feature this adapter cannot map."""


class AbortFailedError(RuntimeError):
    """Raised when an active request could not be aborted on the engine."""

    def __init__(self, request_id: str, cause: BaseException) -> None:
        super().__init__(f"failed to abort request '{request_id}': {cause}")
        self.request_id = request_id


def _top_n_candidates(selection: Any, name: str) -> int:
    kind = selection.WhichOneof("selection")
    if kind is None:
        return 0
    if kind == "top_n":
        return selection.top_n
    raise UnsupportedFeatureError(
        f"{name} candidate selection '{kind}' is not supported; use top_n"
    )


# Guides each grammar backend can build, keyed by the `GuidedDecoding` oneof
# field name. Control advertises from this table and Generate enforces it, so
# the two cannot disagree. llguidance has no structural-tag matcher.
GUIDE_SUPPORT_BY_BACKEND: dict[str, frozenset[str]] = {
    "xgrammar": frozenset(
        {"json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"}
    ),
    "llguidance": frozenset({"json_schema", "regex", "ebnf_grammar", "json_object"}),
}


def supported_guides(backend: str | None) -> frozenset[str]:
    """Guides `backend` can build; empty when none can be.

    With no backend configured the engine never builds a grammar and drops the
    per-request guided params silently, so accepting the request would return
    unconstrained text as a success. Failing closed here also keeps Generate
    aligned with GetModelInfo, which reports guided decoding unsupported for
    that engine.
    """
    if not backend:
        return frozenset()
    return GUIDE_SUPPORT_BY_BACKEND.get(backend.lower(), frozenset())


def _guided_decoding_from_request(
    request: generation_pb2.GenerateRequest,
    guided_backend: str | None = None,
) -> GuidedDecodingParams | None:
    if not request.HasField("guided"):
        return None

    guided = request.guided
    if guided.backend:
        raise UnsupportedFeatureError(
            "per-request guided decoding backend selection is not supported"
        )

    guide = guided.WhichOneof("guide")
    if guide is None:
        return None
    # Reject here rather than letting the grammar compiler fail it in-band, so
    # the client gets an actionable status.
    if guide not in supported_guides(guided_backend):
        raise UnsupportedFeatureError(
            f"guided decoding mode '{guide}' is not supported by the "
            f"'{guided_backend or 'configured'}' grammar backend"
        )
    if guide == "json_schema":
        return GuidedDecodingParams(json=guided.json_schema)
    if guide == "regex":
        return GuidedDecodingParams(regex=guided.regex)
    if guide == "ebnf_grammar":
        return GuidedDecodingParams(grammar=guided.ebnf_grammar)
    if guide == "structural_tag":
        return GuidedDecodingParams(structural_tag=guided.structural_tag)
    if guide == "json_object":
        return GuidedDecodingParams(json_object=True)
    raise UnsupportedFeatureError(f"guided decoding mode '{guide}' is not supported")


def sampling_params_from_request(
    request: generation_pb2.GenerateRequest,
    guided_backend: str | None = None,
) -> SamplingParams:
    """Translate portable OpenEngine generation options to TensorRT-LLM."""
    kwargs: dict[str, Any] = {}
    num_sequences = 1
    provided_fields: set[str] = set()

    if request.HasField("sampling"):
        sampling = request.sampling
        sampling_fields = (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "seed",
        )
        for name in sampling_fields:
            if sampling.HasField(name):
                kwargs[name] = getattr(sampling, name)
                provided_fields.add(name)
        if sampling.HasField("num_sequences"):
            if sampling.num_sequences == 0:
                raise ValueError("sampling.num_sequences must be greater than zero")
            num_sequences = sampling.num_sequences

    if request.HasField("stopping"):
        stopping = request.stopping
        if stopping.HasField("max_tokens"):
            kwargs["max_tokens"] = stopping.max_tokens
        if stopping.HasField("min_tokens"):
            kwargs["min_tokens"] = stopping.min_tokens

        stop_texts = []
        stop_token_ids = []
        for condition in stopping.conditions:
            kind = condition.WhichOneof("condition")
            if kind == "stop_text":
                stop_texts.append(condition.stop_text)
            elif kind == "stop_token_id":
                stop_token_ids.append(condition.stop_token_id)
            else:
                raise ValueError("each stopping condition must set exactly one condition")
        if stop_texts:
            kwargs["stop"] = stop_texts
        if stop_token_ids:
            kwargs["stop_token_ids"] = stop_token_ids
        if stopping.HasField("ignore_eos"):
            kwargs["ignore_eos"] = stopping.ignore_eos
        if stopping.HasField("include_stop_in_output"):
            kwargs["include_stop_str_in_output"] = stopping.include_stop_in_output

    if request.HasField("response"):
        response = request.response
        if response.HasField("prompt_logprob_start") and response.prompt_logprob_start != 0:
            raise UnsupportedFeatureError("response.prompt_logprob_start is not supported")
        if response.HasField("return_prompt_logprobs") and response.return_prompt_logprobs:
            kwargs["prompt_logprobs"] = _top_n_candidates(response.prompt_candidates, "prompt")
        if response.HasField("return_output_logprobs") and response.return_output_logprobs:
            kwargs["logprobs"] = _top_n_candidates(response.output_candidates, "output")

    guided_decoding = _guided_decoding_from_request(request, guided_backend)
    if guided_decoding is not None:
        kwargs["guided_decoding"] = guided_decoding

    # Normalize unspecified sampling knobs to the same protocol defaults the
    # trtllm-serve OpenAI path (/v1/completions) applies, so a sampling request
    # produces identical results across the HTTP and OpenEngine transports. These
    # are no-ops for greedy decoding (argmax).
    kwargs.setdefault("temperature", 1.0)
    kwargs.setdefault("top_p", 1.0)
    kwargs.setdefault("top_k", 0)

    sampling_params = SamplingParams(**kwargs)
    # Record which sampling fields the client actually supplied (mirrors the HTTP
    # `_record_sampling_params_request_fields`). Without this, the materialized
    # defaults above would look client-provided and suppress a model's
    # generation_config.json sampling defaults under generation_config="auto".
    sampling_params._set_request_provided_fields(provided_fields)
    if num_sequences > 1:
        sampling_params.n = num_sequences
        sampling_params.best_of = num_sequences
    return sampling_params


def _parse_metadata_integer(value: str, name: str, minimum: int, maximum: int) -> int:
    digits = value[1:] if value.startswith(("+", "-")) else value
    if not digits or not digits.isascii() or not digits.isdecimal():
        raise ValueError(f"gRPC metadata key '{name}' must be a base-10 integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"gRPC metadata key '{name}' is outside its supported integer range")
    return parsed


def _trace_headers(context: grpc.aio.ServicerContext) -> Mapping[str, str] | None:
    headers: dict[str, str] = {}
    openengine_keys: set[str] = set()
    for item in context.invocation_metadata():
        key = item.key
        value = item.value
        if key.startswith("openengine-"):
            if key in openengine_keys:
                raise ValueError(f"gRPC metadata key '{key}' must not be repeated")
            openengine_keys.add(key)
            if key == "openengine-routing-key":
                if not value:
                    raise ValueError("openengine-routing-key must be non-empty")
            elif key == "openengine-priority":
                _parse_metadata_integer(value, key, -(2**31), 2**31 - 1)
                raise UnsupportedFeatureError(f"gRPC metadata key '{key}' is not supported")
            elif key == "openengine-target-dp-rank":
                _parse_metadata_integer(value, key, 0, 2**32 - 1)
                raise UnsupportedFeatureError(f"gRPC metadata key '{key}' is not supported")
        elif key in ("traceparent", "tracestate"):
            headers[key] = value
    return headers or None


def _input_from_request(request: generation_pb2.GenerateRequest) -> str | dict[str, list[int]]:
    input_kind = request.WhichOneof("input")
    if input_kind == "prompt":
        return request.prompt
    if input_kind == "token_ids":
        return {"prompt_token_ids": list(request.token_ids.ids)}
    raise ValueError("exactly one of prompt or token_ids must be set")


def _token_strings(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    if not token_ids or tokenizer is None:
        return [""] * len(token_ids)
    # Decode each id to human-readable text, matching the HTTP /v1/completions
    # logprobs token representation (`tokenizer.decode(token_id)`), rather than
    # raw tokenizer pieces (e.g. "▁Paris", "<0x0A>") from convert_ids_to_tokens.
    return [tokenizer.decode([token_id]) or "" for token_id in token_ids]


def _token_infos(
    tokenizer: Any,
    token_ids: Sequence[int],
    logprobs: Sequence[Any] = (),
    candidate_limit: int = 0,
) -> list[generation_pb2.TokenInfo]:
    # Only decode per-token strings when logprobs are attached: the `token` /
    # candidate strings exist for logprob display (mirroring the HTTP path, which
    # only surfaces token strings inside logprobs), so the common token-ids/text
    # streaming path skips per-token detokenization entirely.
    need_strings = bool(logprobs)
    token_strings = _token_strings(tokenizer, token_ids) if need_strings else [""] * len(token_ids)
    candidate_items: list[list[tuple[int, Any]]] = []
    candidate_ids: dict[int, None] = {}
    for index in range(len(token_ids)):
        token_logprobs = logprobs[index] if index < len(logprobs) else None
        candidates = []
        if isinstance(token_logprobs, dict) and candidate_limit > 0:
            candidates = sorted(
                (
                    (candidate_id, candidate)
                    for candidate_id, candidate in token_logprobs.items()
                    if candidate.rank is not None and candidate.rank <= candidate_limit
                ),
                key=lambda item: item[1].rank,
            )
            candidate_ids.update((candidate_id, None) for candidate_id, _ in candidates)
        candidate_items.append(candidates)

    candidate_strings = dict(
        zip(candidate_ids, _token_strings(tokenizer, list(candidate_ids)), strict=True)
    )
    token_infos = []
    for index, (token_id, token) in enumerate(zip(token_ids, token_strings, strict=True)):
        token_logprobs = logprobs[index] if index < len(logprobs) else None
        kwargs: dict[str, Any] = {"token_id": token_id, "token": token}
        if isinstance(token_logprobs, (int, float)):
            kwargs["logprob"] = _clamp_logprob(token_logprobs)
        elif isinstance(token_logprobs, dict):
            sampled = token_logprobs.get(token_id)
            if sampled is not None:
                kwargs["logprob"] = _clamp_logprob(sampled.logprob)
                if sampled.rank is not None:
                    kwargs["rank"] = sampled.rank
            kwargs["candidates"] = [
                generation_pb2.LogProb(
                    token_id=candidate_id,
                    logprob=_clamp_logprob(candidate.logprob),
                    token=candidate_strings[candidate_id],
                    **({"rank": candidate.rank} if candidate.rank is not None else {}),
                )
                for candidate_id, candidate in candidate_items[index]
            ]
        token_infos.append(generation_pb2.TokenInfo(**kwargs))
    return token_infos


def _stop_texts(sampling_params: SamplingParams) -> list[str]:
    stop = sampling_params.stop
    if isinstance(stop, str):
        return [stop]
    return list(stop or [])


def _token_holdback(sampling_params: SamplingParams) -> int:
    """Number of trailing tokens to withhold so a stop token id is not streamed.

    NOTE: this depends on the private ``SamplingParams._stop_word_ids``, which the
    LLM engine populates with the tokenized stop words. It is read after
    ``generate_async`` returns, when it is expected to be set. If the engine ever
    stops populating it (or populates it later), this degrades to 0 (no holdback)
    rather than failing — ``test_token_holdback_*`` guards the current behavior.
    """
    if sampling_params.include_stop_str_in_output:
        return 0
    stop_word_ids = getattr(sampling_params, "_stop_word_ids", None) or []
    if not stop_word_ids:
        return 0
    return max(1, max(len(token_ids) for token_ids in stop_word_ids) - 1)


def _prefix_table(pattern: str) -> list[int]:
    table = [0] * len(pattern)
    matched = 0
    for index in range(1, len(pattern)):
        while matched and pattern[index] != pattern[matched]:
            matched = table[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
            table[index] = matched
    return table


class _StopPrefixTracker:
    def __init__(self, stop_texts: Sequence[str]) -> None:
        self._patterns = [pattern for pattern in stop_texts if pattern]
        self._tables = [_prefix_table(pattern) for pattern in self._patterns]
        self._states = [0] * len(self._patterns)
        self._observed_length = 0

    def safe_length(self, text: str) -> int:
        if len(text) < self._observed_length:
            self._states = [0] * len(self._patterns)
            self._observed_length = 0

        delta = text[self._observed_length :]
        for pattern_index, (pattern, table) in enumerate(zip(self._patterns, self._tables)):
            matched = self._states[pattern_index]
            for character in delta:
                while matched and (matched == len(pattern) or character != pattern[matched]):
                    matched = table[matched - 1]
                if matched < len(pattern) and character == pattern[matched]:
                    matched += 1
            self._states[pattern_index] = matched
        self._observed_length = len(text)
        return len(text) - max(self._states, default=0)


def _prompt_output(
    tokenizer: Any, result: Any, candidate_limit: int
) -> generation_pb2.PromptOutput:
    token_ids = list(result.prompt_token_ids)
    prompt_logprobs = result.outputs[0].prompt_logprobs if result.outputs else None
    prompt_logprobs = prompt_logprobs or []
    aligned_logprobs = [None, *prompt_logprobs[: max(0, len(token_ids) - 1)]]
    return generation_pb2.PromptOutput(
        tokens=_token_infos(tokenizer, token_ids, aligned_logprobs, candidate_limit)
    )


# Finish reasons that carry real terminal information. Anything else (notably
# "not_finished") would map to FINISH_REASON_UNSPECIFIED.
_TERMINAL_FINISH_REASONS = frozenset({"stop", "length", "cancelled", "timeout"})


def _aligned_logprobs(output: Any) -> Sequence[Any]:
    """Logprobs index-aligned with ``output.token_ids``.

    On a generation_only request the engine tolerates being one logprob short --
    the context worker did not transfer the first token's -- and only logs a
    warning. Slicing positionally against that would attribute every logprob,
    rank and candidate set to the wrong token, so pad the missing head instead.
    """
    logprobs = output.logprobs or []
    shortfall = len(output.token_ids or []) - len(logprobs)
    if shortfall > 0 and logprobs:
        return [None] * shortfall + list(logprobs)
    return logprobs


def _finish_event(output: Any, end_id: int | None) -> generation_pb2.GenerationFinished:
    reason_map = {
        "stop": generation_pb2.FINISH_REASON_STOP,
        "length": generation_pb2.FINISH_REASON_LENGTH,
        "cancelled": generation_pb2.FINISH_REASON_CANCELLED,
        "timeout": generation_pb2.FINISH_REASON_CANCELLED,
    }
    kwargs: dict[str, Any] = {
        "output_index": output.index,
        "reason": reason_map.get(output.finish_reason, generation_pb2.FINISH_REASON_UNSPECIFIED),
    }
    if output.finish_reason == "timeout":
        kwargs["message"] = "generation timed out"
    if output.finish_reason == "stop":
        if isinstance(output.stop_reason, int):
            kwargs["stop_match"] = generation_pb2.StopMatch(stop_token_id=output.stop_reason)
        elif isinstance(output.stop_reason, str):
            kwargs["stop_match"] = generation_pb2.StopMatch(stop_text=output.stop_reason)
        elif end_id is not None:
            kwargs["stop_match"] = generation_pb2.StopMatch(eos_token_id=end_id)
    return generation_pb2.GenerationFinished(**kwargs)


def _usage(result: Any) -> generation_pb2.Usage:
    prompt_tokens = len(getattr(result, "prompt_token_ids", ()) or ())
    completion_tokens = sum(
        len(output.token_ids or []) for output in (getattr(result, "outputs", ()) or ())
    )
    return generation_pb2.Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_prompt_tokens=getattr(result, "cached_tokens", 0),
    )


def _engine_error_response(
    request_id: str,
    message: str,
    result: Any,
    *,
    code: int = error_pb2.ERROR_CODE_INTERNAL,
    retryable: bool = False,
) -> generation_pb2.GenerateResponse:
    return generation_pb2.GenerateResponse(
        request_id=request_id,
        error=error_pb2.EngineError(code=code, message=message, retryable=retryable),
        usage=_usage(result),
    )


def _struct_to_dict(struct: struct_pb2.Struct) -> dict[str, Any]:
    return MessageToDict(struct)


def _endpoint_to_str(endpoints: Sequence[Any]) -> Optional[str]:
    if not endpoints:
        return None
    endpoint = endpoints[0]
    if not endpoint.host:
        return None
    return f"{endpoint.host}:{endpoint.port}" if endpoint.port else endpoint.host


def _split_endpoint(value: str) -> Optional[tuple[str, int, str]]:
    """Split an engine endpoint into (host, port, scheme).

    The engine's ctx_info_endpoint is an opaque URL, so a bare ``partition(":")``
    would take the scheme for the host and drop the rest.
    """
    try:
        parts = urlsplit(value if "://" in value else f"//{value}")
        host = parts.hostname or ""
        port = parts.port or 0
    except ValueError:
        return None
    if not host:
        return None
    return host, port, parts.scheme


def _serialize_first_gen_log_probs(first_gen_log_probs: Sequence[Any]) -> list[Any]:
    """Serialize first-gen logprobs into a Struct-storable form.

    ``list[dict[int, Logprob]] | list[float]``: each position is either a float
    (simple format) or a list of ``[token_id, logprob, rank]`` triples (verbose).
    """
    positions: list[Any] = []
    for pos in first_gen_log_probs:
        if isinstance(pos, dict):
            positions.append(
                [
                    [
                        int(token_id),
                        _clamp_logprob(lp.logprob),
                        (lp.rank if lp.rank is not None else None),
                    ]
                    for token_id, lp in pos.items()
                ]
            )
        else:
            positions.append(_clamp_logprob(pos))
    return positions


def _deserialize_first_gen_log_probs(positions: Sequence[Any]) -> list[Any]:
    """Inverse of :func:`_serialize_first_gen_log_probs`, rebuilding Logprob objects."""
    result: list[Any] = []
    for pos in positions:
        if isinstance(pos, list):
            result.append(
                {
                    int(token_id): Logprob(
                        logprob=_clamp_logprob(logprob), rank=None if rank is None else int(rank)
                    )
                    for token_id, logprob, rank in pos
                }
            )
        else:
            result.append(_clamp_logprob(pos))
    return result


def _validate_kv_session(session: Any, attrs: Mapping[str, Any]) -> None:
    """Reject a session the generation worker cannot resolve.

    The engine consumes the handoff on its executor loop with no timeout and no
    guard: a session naming an unreachable context worker blocks that loop
    indefinitely, so every later request on this engine stalls too. Validating
    what we can here turns an engine-wide hang into one INVALID_ARGUMENT.

    ``attrs`` is the caller's already-converted attributes Struct: converting it
    twice would walk the whole handoff -- including first_gen_log_probs -- twice
    per request on the decode server's event loop.
    """
    endpoint = attrs.get("ctx_info_endpoint") or _endpoint_to_str(session.endpoints)
    if endpoint:
        parts = _split_endpoint(str(endpoint))
        if parts is None or not parts[1]:
            raise ValueError(f"kv.session ctx_info_endpoint '{endpoint}' is not a usable address")
    elif not attrs.get("opaque_state"):
        # One or the other locates the context worker: an endpoint on builds that
        # fetch its info over ZMQ, opaque_state on builds that carry it inline.
        raise ValueError("kv.session carries neither a ctx_info_endpoint nor an opaque_state")
    if not session.session_id and attrs.get("disagg_request_id") is None:
        raise ValueError("kv.session carries neither a session_id nor a disagg_request_id")


def _disaggregated_params_from_request(
    request: generation_pb2.GenerateRequest,
) -> Optional[DisaggregatedParams]:
    """Map an OpenEngine request to TensorRT-LLM disaggregated-serving params.

    Returns None for a normal (aggregated) request. A ``kv.session`` handle marks
    a generation_only request and is decoded back into the context handoff
    (``opaque_state`` / ``ctx_request_id`` / ``first_gen_tokens`` / endpoint /
    dp_rank). Otherwise the phase comes from ``extra["request_type"]``.
    """
    has_session = request.HasField("kv") and request.kv.HasField("session")
    if has_session:
        session = request.kv.session
        attrs = _struct_to_dict(session.attributes_struct)
        _validate_kv_session(session, attrs)
        params = DisaggregatedParams(request_type="generation_only")
        if session.session_id:
            try:
                params.ctx_request_id = int(session.session_id)
            except ValueError as error:
                raise ValueError(
                    "kv.session.session_id must be an integer context request id"
                ) from error
        # Prefer the verbatim endpoint; the structured fields are a fallback for
        # a peer that did not carry it.
        endpoint = attrs.get("ctx_info_endpoint") or _endpoint_to_str(session.endpoints)
        if endpoint:
            params.ctx_info_endpoint = str(endpoint)
        opaque_state = attrs.get("opaque_state")
        if opaque_state:
            params.opaque_state = base64.b64decode(opaque_state, validate=True)
        first_gen_tokens = attrs.get("first_gen_tokens")
        if first_gen_tokens:
            params.first_gen_tokens = [int(token) for token in first_gen_tokens]
        draft_tokens = attrs.get("draft_tokens")
        if draft_tokens:
            params.draft_tokens = [int(token) for token in draft_tokens]
        disagg_request_id = attrs.get("disagg_request_id")
        if disagg_request_id is not None:
            params.disagg_request_id = int(disagg_request_id)
        # ctx_dp_rank: prefer the attribute (preserves an explicit 0 and, by its
        # absence, an unset None) and fall back to the native dp_rank field.
        if "ctx_dp_rank" in attrs:
            params.ctx_dp_rank = int(attrs["ctx_dp_rank"])
        elif session.dp_rank:
            params.ctx_dp_rank = session.dp_rank
        schedule_style = attrs.get("schedule_style")
        if schedule_style is not None:
            params.schedule_style = DisaggScheduleStyle(int(schedule_style))
        ctx_usage = attrs.get("ctx_usage")
        if ctx_usage is not None:
            # Struct has only a double number type, so token counts come back as
            # floats; restore the integer typing the engine and Usage expect.
            params.ctx_usage = {
                key: int(value) if isinstance(value, float) and value.is_integer() else value
                for key, value in ctx_usage.items()
            }
        first_gen_log_probs = attrs.get("first_gen_log_probs")
        if first_gen_log_probs is not None:
            params.first_gen_log_probs = _deserialize_first_gen_log_probs(first_gen_log_probs)
        return params

    extra = _struct_to_dict(request.extra) if request.HasField("extra") else {}
    request_type = extra.get(_REQUEST_TYPE_KEY)
    if request_type is None:
        return None
    if request_type not in _DISAGG_REQUEST_TYPES:
        raise ValueError(
            f"extra.{_REQUEST_TYPE_KEY} must be one of {sorted(_DISAGG_REQUEST_TYPES)}"
        )
    return DisaggregatedParams(request_type=request_type)


def _prefill_ready_response(
    request_id: str,
    disaggregated_params: Any,
    transfer_backend: str,
    usage: Optional[generation_pb2.Usage] = None,
) -> generation_pb2.GenerateResponse:
    """Build the PrefillReady event a context worker returns after prefill.

    Packs the TensorRT-LLM context handoff into a KvSessionRef: ``ctx_request_id``
    -> session_id, ``ctx_dp_rank`` -> dp_rank, ``ctx_info_endpoint`` -> endpoint,
    and the fields with no native slot (opaque_state as base64, first/draft tokens,
    disagg_request_id, schedule_style, ctx_usage) into the
    session's attributes Struct.

    ``first_gen_log_probs`` is carried (serialized) so logprobs-in-disagg returns
    the first token's logprob on the generation worker. ``first_gen_logits`` is
    NOT carried: OpenEngine has no "return generation logits" request field, so it
    is never populated for a gRPC request (like beam search, it is unreachable).
    """
    ctx_request_id = disaggregated_params.ctx_request_id
    session = kv_pb2.KvSessionRef(
        session_id="" if ctx_request_id is None else str(ctx_request_id),
        transfer_backend=transfer_backend or "",
        dp_rank=disaggregated_params.ctx_dp_rank or 0,
    )
    attributes: dict[str, Any] = {}
    endpoint = disaggregated_params.ctx_info_endpoint
    if endpoint:
        # Carried verbatim as well as split: the generation worker connects a
        # ZMQ socket straight to this string, so any lossy round trip through
        # the structured fields would hand it an unusable address.
        attributes["ctx_info_endpoint"] = endpoint
        parts = _split_endpoint(endpoint)
        if parts is not None:
            host, port, scheme = parts
            session.endpoints.add(host=host, port=port, protocol=scheme or transfer_backend or "")
    opaque_state = disaggregated_params.opaque_state
    if opaque_state is not None:
        attributes["opaque_state"] = base64.b64encode(opaque_state).decode("utf-8")
    first_gen_tokens = disaggregated_params.first_gen_tokens
    if first_gen_tokens:
        attributes["first_gen_tokens"] = list(first_gen_tokens)
    draft_tokens = disaggregated_params.draft_tokens
    if draft_tokens:
        attributes["draft_tokens"] = list(draft_tokens)
    disagg_request_id = disaggregated_params.disagg_request_id
    if disagg_request_id is not None:
        attributes["disagg_request_id"] = str(disagg_request_id)
    ctx_dp_rank = disaggregated_params.ctx_dp_rank
    if ctx_dp_rank is not None:
        attributes["ctx_dp_rank"] = ctx_dp_rank
    schedule_style = disaggregated_params.schedule_style
    if schedule_style is not None:
        attributes["schedule_style"] = int(schedule_style)
    ctx_usage = disaggregated_params.ctx_usage
    if ctx_usage is not None:
        attributes["ctx_usage"] = dict(ctx_usage)
    first_gen_log_probs = disaggregated_params.first_gen_log_probs
    if first_gen_log_probs is not None:
        attributes["first_gen_log_probs"] = _serialize_first_gen_log_probs(first_gen_log_probs)
    if attributes:
        struct = struct_pb2.Struct()
        struct.update(attributes)
        session.attributes_struct.CopyFrom(struct)
    return generation_pb2.GenerateResponse(
        request_id=request_id,
        prefill_ready=generation_pb2.PrefillReady(kv_session=session),
        usage=usage,
    )


@dataclass(kw_only=True)
class _OpenEngineFormatState(PostprocArgs):
    """Per-request state for ``_format_result``.

    Subclasses ``PostprocArgs`` so it can be threaded through the TensorRT-LLM
    postprocessing worker pool (``num_postprocess_workers``): when workers are
    enabled the formatting runs in a worker process and this state is created
    once per request and reused there (the worker injects ``tokenizer``); when
    disabled the servicer calls ``_format_result`` inline with this same object.
    """

    request_id: str = ""
    sampling_params: Any = None
    is_context_only: bool = False
    kv_transfer_backend: str = ""
    # None until first use: _token_holdback needs sampling_params._stop_word_ids,
    # which generate_async only populates after the request is submitted, so it is
    # computed lazily on the first _format_result call (inline or in a worker).
    token_holdback: Optional[int] = None
    stop_texts: list = field(default_factory=list)
    exclude_stop: bool = True
    sent_token_counts: dict = field(default_factory=dict)
    sent_text_lengths: dict = field(default_factory=dict)
    observed_text_lengths: dict = field(default_factory=dict)
    stop_prefix_trackers: dict = field(default_factory=dict)
    finished_indices: set = field(default_factory=set)
    prompt_sent: bool = False
    prefill_ready_sent: set = field(default_factory=set)


def _format_result(
    result: Any, args: _OpenEngineFormatState
) -> list[generation_pb2.GenerateResponse]:
    """Format one engine result into OpenEngine responses.

    Produces prompt/token/finished/prefill/error events. Runs on the event loop
    when postprocessing workers are off, or inside a worker process when they are
    on — the servicer only relays these responses, keeping the CPU-bound detok +
    protobuf build off its event loop under load.
    """
    request_id = args.request_id
    sampling_params = args.sampling_params
    tokenizer = args.tokenizer
    if args.token_holdback is None:
        args.token_holdback = _token_holdback(sampling_params)
    responses: list[generation_pb2.GenerateResponse] = []

    if result.error:
        responses.append(_engine_error_response(request_id, str(result.error), result))
        return responses

    if sampling_params.prompt_logprobs is not None and not args.prompt_sent and result.outputs:
        responses.append(
            generation_pb2.GenerateResponse(
                request_id=request_id,
                prompt=_prompt_output(tokenizer, result, sampling_params.prompt_logprobs),
            )
        )
        args.prompt_sent = True

    newly_finished = []
    for output in result.outputs:
        token_ids = output.token_ids or []
        sent_token_count = min(args.sent_token_counts.get(output.index, 0), len(token_ids))
        safe_token_count = len(token_ids)
        if args.exclude_stop and not output.finish_reason:
            safe_token_count = max(0, safe_token_count - args.token_holdback)
        delta_token_ids = token_ids[sent_token_count:safe_token_count]
        all_text = output.text or ""
        sent_text_length = min(args.sent_text_lengths.get(output.index, 0), len(all_text))
        safe_text_length = len(all_text)
        if (
            args.exclude_stop
            and not output.finish_reason
            and (args.stop_texts or args.token_holdback)
        ):
            tracker = args.stop_prefix_trackers.setdefault(
                output.index, _StopPrefixTracker(args.stop_texts)
            )
            safe_text_length = tracker.safe_length(all_text)
            if args.token_holdback:
                safe_text_length = min(
                    safe_text_length, args.observed_text_lengths.get(output.index, 0)
                )
        delta_text = all_text[sent_text_length:safe_text_length]
        args.observed_text_lengths[output.index] = len(all_text)

        if delta_token_ids or delta_text:
            logprobs = _aligned_logprobs(output)
            delta_logprobs = logprobs[sent_token_count:]
            token_infos = _token_infos(
                tokenizer, delta_token_ids, delta_logprobs, sampling_params.logprobs or 0
            )
            responses.append(
                generation_pb2.GenerateResponse(
                    request_id=request_id,
                    token=generation_pb2.TokenOutput(
                        output_index=output.index, tokens=token_infos, text=delta_text
                    ),
                )
            )
        args.sent_token_counts[output.index] = safe_token_count
        args.sent_text_lengths[output.index] = safe_text_length

        if args.is_context_only and output.index not in args.prefill_ready_sent:
            # `disaggregated_params` is seeded from the *request*, so it is never
            # None here; only a context phase that actually transmitted fills in
            # ctx_request_id. Emitting without it fabricates a handoff whose
            # session the generation worker cannot resolve.
            out_disagg = getattr(output, "disaggregated_params", None)
            if out_disagg is not None and out_disagg.ctx_request_id is not None:
                args.prefill_ready_sent.add(output.index)
                responses.append(
                    _prefill_ready_response(
                        request_id,
                        out_disagg,
                        args.kv_transfer_backend,
                        usage=_usage(result),
                    )
                )

        if output.finish_reason and output.index not in args.finished_indices:
            args.finished_indices.add(output.index)
            newly_finished.append(output)

    for index, output in enumerate(newly_finished):
        # PrefillReady is the terminal signal for a context request that produced
        # a handoff. Without one -- cancelled or aborted before transmission --
        # the client must still get the real terminal event rather than a clean
        # stream with no ending. A context request that simply has nothing to
        # report ends as not-finished, which would map to a spurious UNSPECIFIED.
        if args.is_context_only and (
            output.index in args.prefill_ready_sent
            or output.finish_reason not in _TERMINAL_FINISH_REASONS
        ):
            continue
        is_final = (
            len(args.finished_indices) == sampling_params.n and index == len(newly_finished) - 1
        )
        response_kwargs: dict[str, Any] = {
            "request_id": request_id,
            "finished": _finish_event(output, sampling_params.end_id),
        }
        if is_final:
            response_kwargs["usage"] = _usage(result)
        responses.append(generation_pb2.GenerateResponse(**response_kwargs))

    return responses


class OpenEngineInferenceServicer(openengine_pb2_grpc.InferenceServicer):
    """Translate OpenEngine generation streams to TensorRT-LLM requests."""

    def __init__(self, llm: Any, model: str, kv_transfer_backend: str = "") -> None:
        self._llm = llm
        self._model = model
        # Informational label placed in the KvSessionRef of PrefillReady events so
        # a generation worker/orchestrator can see which KV transfer backend the
        # context worker uses. The actual transfer is driven by opaque_state.
        self._kv_transfer_backend = kv_transfer_backend
        self._guided_backend = getattr(getattr(llm, "args", None), "guided_decoding_backend", None)
        self._active_requests: dict[str, Any] = {}

    def active_request_count(self) -> int:
        """Requests accepted and not yet completed."""
        return len(self._active_requests)

    def abort_request_by_id(self, request_id: str) -> bool:
        """Abort one in-flight request. Returns False when it is not active.

        Does not pop the id: the Generate stream's ``finally`` owns cleanup, and
        removing it here would let a duplicate request_id past the ALREADY_EXISTS
        check while the original stream is still draining.

        Raises:
            AbortFailedError: the request is active but the engine refused to
                abort it, so it is still running.
        """
        handle = self._active_requests.get(request_id)
        if handle is None:
            return False
        try:
            handle.abort()
        except Exception as error:
            logger.warning(f"Failed to abort OpenEngine request {request_id}: {error}")
            raise AbortFailedError(request_id, error) from error
        return True

    def abort_all_requests(self) -> tuple[int, int]:
        """Abort every in-flight request; returns (aborted, failed) counts."""
        aborted = 0
        failed = 0
        for request_id in list(self._active_requests):
            try:
                if self.abort_request_by_id(request_id):
                    aborted += 1
            except AbortFailedError:
                failed += 1
        return aborted, failed

    async def Generate(
        self,
        request: generation_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        """Run a server-streaming OpenEngine generation request."""
        request_id = request.request_id
        try:
            if not request_id:
                raise ValueError("request_id must be non-empty")
            if not request.model:
                raise ValueError("model must be non-empty")
            # Like the HTTP /v1/completions single-model server, any non-empty
            # model name is accepted and served by the loaded model (no NOT_FOUND
            # on a mismatch) so the two transports behave consistently.
            if request_id in self._active_requests:
                await context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"request_id '{request_id}' is already active",
                )
                return
            if request.media:
                raise UnsupportedFeatureError("multimodal media is not supported")
            if request.lora_name:
                raise UnsupportedFeatureError("LoRA selection is not supported")
            if request.HasField("kv"):
                if request.kv.HasField("bypass_prefix_cache") and request.kv.bypass_prefix_cache:
                    raise UnsupportedFeatureError("prefix cache bypass is not supported")

            inputs = _input_from_request(request)
            sampling_params = sampling_params_from_request(request, self._guided_backend)
            trace_headers = _trace_headers(context)
            cache_salt = (
                request.kv.cache_salt
                if request.HasField("kv") and request.kv.HasField("cache_salt")
                else None
            )
            disaggregated_params = _disaggregated_params_from_request(request)
        except UnsupportedFeatureError as error:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
            return
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return

        # Response formatting state. When postprocessing workers are enabled this
        # is run in a worker process (off the event loop); otherwise the servicer
        # calls _format_result inline with the same object.
        # NOTE: worker offload requires the engine's postproc workers to use the
        # 'spawn' start method — with fork, deserializing CUDA-resident engine
        # results in the worker fails ("Cannot re-initialize CUDA in forked
        # subprocess"). Inline (the default, num_postprocess_workers=0) is
        # unaffected.
        postproc_enabled = (
            getattr(getattr(self._llm, "args", None), "num_postprocess_workers", 0) > 0
        )
        format_state = _OpenEngineFormatState(
            request_id=request_id,
            sampling_params=sampling_params,
            is_context_only=(
                disaggregated_params is not None
                and disaggregated_params.request_type == "context_only"
            ),
            kv_transfer_backend=self._kv_transfer_backend,
            stop_texts=_stop_texts(sampling_params),
            exclude_stop=not sampling_params.include_stop_str_in_output,
            tokenizer=None if postproc_enabled else getattr(self._llm, "tokenizer", None),
        )
        postproc_params = (
            PostprocParams(post_processor=_format_result, postproc_args=format_state)
            if postproc_enabled
            else None
        )

        try:
            result_handle = self._llm.generate_async(
                inputs=inputs,
                sampling_params=sampling_params,
                streaming=True,
                trace_headers=trace_headers,
                cache_salt=cache_salt,
                priority=DEFAULT_REQUEST_PRIORITY,
                disaggregated_params=disaggregated_params,
                _postproc_params=postproc_params,
            )
        except (TypeError, ValueError) as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return
        # Broad by design: request submission can fail in engine-specific ways we
        # cannot enumerate here; any such failure must become a clean INTERNAL
        # status for the client rather than an unhandled servicer crash.
        except Exception as error:
            logger.error(f"Failed to submit OpenEngine request {request_id}: {error}")
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
            return

        engine_terminal = False
        abort_requested = False
        consumer_stalled = False
        stall_timer: asyncio.TimerHandle | None = None
        loop = asyncio.get_running_loop()
        # loop.time() only while a response has been yielded but not yet pulled by
        # the consumer; None otherwise. The watchdog measures this consumer-pull
        # latency, so a slow engine (nothing to send yet) is never misread as a
        # stalled consumer.
        pending_since: float | None = None

        def abort_request(reason: str) -> None:
            nonlocal abort_requested
            if abort_requested or engine_terminal:
                return
            abort_requested = True
            try:
                result_handle.abort()
            # Broad by design: aborting is best-effort cleanup and must never
            # propagate out of this callback.
            except Exception as error:
                logger.warning(
                    f"Failed to abort OpenEngine request {request_id} ({reason}): {error}"
                )

        def rpc_done(_: grpc.aio.ServicerContext) -> None:
            abort_request("RPC completed before generation")

        def untrack() -> None:
            # Once this id is dropped a resubmission is admitted, so both
            # removal sites must check they still own the registration --
            # otherwise the loser untracks the newer request's handle.
            if self._active_requests.get(request_id) is result_handle:
                del self._active_requests[request_id]

        def stall_watchdog() -> None:
            # Single self-rescheduling watchdog (no per-message timer alloc/cancel).
            nonlocal stall_timer, consumer_stalled
            if pending_since is None:
                stall_timer = loop.call_later(_RESPONSE_STALL_TIMEOUT_SECONDS, stall_watchdog)
                return
            idle = loop.time() - pending_since
            if idle >= _RESPONSE_STALL_TIMEOUT_SECONDS:
                consumer_stalled = True
                logger.warning(
                    f"OpenEngine request {request_id} stopped consuming responses "
                    f"for {_RESPONSE_STALL_TIMEOUT_SECONDS:g} seconds"
                )
                abort_request("response consumer stalled")
                # The generator is suspended in `yield` and may never resume, so
                # its `finally` may never run. Drop the registration here or the
                # id stays in flight forever: GetLoad over-reports it and the id
                # is permanently unusable.
                untrack()
                stall_timer = None
            else:
                stall_timer = loop.call_later(
                    _RESPONSE_STALL_TIMEOUT_SECONDS - idle, stall_watchdog
                )

        def mark_pending() -> None:
            nonlocal pending_since
            pending_since = loop.time()

        def mark_delivered() -> None:
            nonlocal pending_since
            pending_since = None

        def cancel_watchdog() -> None:
            nonlocal stall_timer
            if stall_timer is not None:
                stall_timer.cancel()
                stall_timer = None

        context.add_done_callback(rpc_done)

        try:
            # Register inside the try so the finally below always removes the id.
            # There is no await between the duplicate-id check above and here, so
            # the asyncio single-thread atomicity that makes that check correct
            # still holds, while an exception before streaming can no longer leak
            # the id in _active_requests (which would make it permanently
            # un-reusable, since re-submits are rejected with ALREADY_EXISTS).
            self._active_requests[request_id] = result_handle
            stall_timer = loop.call_later(_RESPONSE_STALL_TIMEOUT_SECONDS, stall_watchdog)
            async for result in result_handle:
                if context.cancelled():
                    abort_request("RPC cancelled")
                    return
                # Formatting (detok + protobuf build) is done in a worker process
                # when postprocessing workers are enabled, else inline here.
                responses = (
                    result.outputs[0]._postprocess_result
                    if postproc_enabled
                    else _format_result(result, format_state)
                )
                for resp in responses or ():
                    mark_pending()
                    yield resp
                    mark_delivered()
                    if consumer_stalled:
                        engine_terminal = True
                        yield _engine_error_response(
                            request_id,
                            "response consumer stalled",
                            result,
                            code=error_pb2.ERROR_CODE_OVERLOADED,
                            retryable=True,
                        )
                        return
                if result.error:
                    engine_terminal = True
                    return
                # Terminal when the engine finishes: inline we already tracked all
                # sequences in format_state; with workers the state lives in the
                # worker, so use the result's own finished flag.
                if (
                    result.finished
                    if postproc_enabled
                    else len(format_state.finished_indices) == sampling_params.n
                ):
                    engine_terminal = True
                    return

            # Defensive: the engine is expected to mark the request finished before
            # its result stream ends, so reaching here means it closed early (e.g. an
            # internal engine abort). Surface it as an error rather than silently
            # returning a truncated result to the client.
            engine_terminal = True
            yield _engine_error_response(
                request_id,
                "generation stream ended before all outputs finished",
                result_handle,
            )
        except asyncio.CancelledError:
            raise
        # Broad by design: this is the RPC boundary, so any engine/processing
        # failure after acceptance must be reported to the client as an error
        # event and must still trigger abort + cleanup in the finally below.
        except Exception as error:
            logger.error(f"OpenEngine request {request_id} failed after acceptance: {error}")
            abort_request("response processing failed")
            engine_terminal = True
            yield _engine_error_response(request_id, str(error), result_handle)
        finally:
            cancel_watchdog()
            if not engine_terminal:
                abort_request("response stream closed")
            untrack()


__all__ = [
    "AbortFailedError",
    "GUIDE_SUPPORT_BY_BACKEND",
    "OpenEngineInferenceServicer",
    "sampling_params_from_request",
    "supported_guides",
]
