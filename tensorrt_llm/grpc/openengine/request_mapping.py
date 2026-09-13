# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an OpenEngine GenerateRequest into TensorRT-LLM request inputs."""

from collections.abc import Mapping
from typing import Any

import grpc
from openengine.v1 import generation_pb2

from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

from .capabilities import supported_guides
from .errors import UnsupportedFeatureError

# Cap on the number of stop conditions. Each one costs a KMP automaton per
# output index and a scan of every generated token, so an unbounded list lets a
# single request tax the event loop for the whole stream.
_MAX_STOP_CONDITIONS = 64


def _top_n_candidates(selection: Any, name: str) -> int:
    kind = selection.WhichOneof("selection")
    if kind is None:
        return 0
    if kind == "top_n":
        return selection.top_n
    raise UnsupportedFeatureError(
        f"{name} candidate selection '{kind}' is not supported; use top_n"
    )


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

        if len(stopping.conditions) > _MAX_STOP_CONDITIONS:
            raise ValueError(
                f"at most {_MAX_STOP_CONDITIONS} stopping conditions are supported, "
                f"got {len(stopping.conditions)}"
            )
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
    # SamplingParams.max_tokens defaults to 32, but the protocol default for an
    # omitted stopping.max_tokens is "unbounded", as on /v1/completions (whose
    # max_tokens is Optional[int] = None). None is the engine's documented unset
    # sentinel: _deduce_max_tokens derives the budget from max_seq_len.
    kwargs.setdefault("max_tokens", None)

    # n/best_of must be constructor arguments: SamplingParams._validate() runs
    # only from __post_init__, so assigning them afterwards skips it. That guard
    # is what rejects greedy decoding with multiple returns -- without it a
    # temperature=0 request with num_sequences>1 is accepted here and returns
    # duplicate sequences as distinct samples, while the same request over
    # /v1/completions is a 400.
    if num_sequences > 1:
        kwargs["n"] = num_sequences
        kwargs["best_of"] = num_sequences

    sampling_params = SamplingParams(**kwargs)
    # Record which sampling fields the client actually supplied (mirrors the HTTP
    # `_record_sampling_params_request_fields`). Without this, the materialized
    # defaults above would look client-provided and suppress a model's
    # generation_config.json sampling defaults under generation_config="auto".
    sampling_params._set_request_provided_fields(provided_fields)
    return sampling_params


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
            elif key in ("openengine-priority", "openengine-target-dp-rank"):
                # Control advertises both as unsupported; the value is never
                # read, so it is not worth parsing to decide the status code.
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


__all__ = [
    "sampling_params_from_request",
]
