# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an OpenEngine GenerateRequest into TensorRT-LLM request inputs."""

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import grpc

from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.inputs.media_io import ImageMediaIO, MediaFetchError
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

from .bindings import generation_pb2
from .capabilities import MediaCapabilities, resolve_media_capabilities, supported_guides
from .errors import MediaUnavailableError, UnsupportedFeatureError

# Cap on the number of stop conditions. Each one costs a KMP automaton per
# output index and a scan of every generated token, so an unbounded list lets a
# single request tax the event loop for the whole stream.
_MAX_STOP_CONDITIONS = 64
# Bound per-request decode work and retained image data.
_MAX_MEDIA_ITEMS = 64
_MAX_CONCURRENT_MEDIA_LOADS = 8
# Checked after each decode; concurrency bounds peak overshoot. Encoded inputs
# have separate gRPC and remote-loader caps.
_MAX_TOTAL_DECODED_MEDIA_BYTES = 512 * 1024 * 1024
_CONVERSATION_ID_EXTRA_KEY = "conversation_id"
_DETOKENIZE_EXTRA_KEY = "detokenize"


_MediaLoader = Callable[[generation_pb2.MediaItem, int], Awaitable[Any]]
_MediaValidator = Callable[[generation_pb2.MediaItem, int], None]


@dataclass(frozen=True)
class _MediaHandler:
    name: str
    validate: _MediaValidator
    loader_factory: Callable[[Any], _MediaLoader]
    decoded_bytes: Callable[[Any], int]


@dataclass(frozen=True)
class _PreparedInput:
    inputs: str | dict[str, list[int]]
    media: tuple[tuple[generation_pb2.MediaItem, _MediaHandler], ...]
    capabilities: MediaCapabilities | None


def conversation_affinity_enabled(llm: Any) -> bool:
    """Whether TensorRT-LLM owns attention-DP placement for conversations."""
    args = getattr(llm, "args", None)
    config = getattr(args, "attention_dp_config", None)
    return getattr(args, "enable_attention_dp", False) is True and (
        getattr(config, "kv_cache_routing_conversation_affinity", False) is True
    )


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

    if (
        request.HasField("extra")
        and (detokenize := request.extra.fields.get(_DETOKENIZE_EXTRA_KEY)) is not None
    ):
        if detokenize.WhichOneof("kind") != "bool_value":
            raise ValueError(f"extra.{_DETOKENIZE_EXTRA_KEY} must be a boolean")
        kwargs["detokenize"] = detokenize.bool_value

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


def conversation_params_from_request(
    request: generation_pb2.GenerateRequest,
) -> ConversationParams | None:
    """Translate the caller's stable conversation identity, when supplied."""
    if not request.HasField("extra"):
        return None
    conversation_id = request.extra.fields.get(_CONVERSATION_ID_EXTRA_KEY)
    if conversation_id is None:
        return None
    if conversation_id.WhichOneof("kind") != "string_value":
        raise ValueError(f"extra.{_CONVERSATION_ID_EXTRA_KEY} must be a string")
    return ConversationParams(conversation_id=conversation_id.string_value)


def _metadata_from_context(
    context: grpc.aio.ServicerContext,
) -> tuple[Mapping[str, str] | None, int | None, dict[str, str]]:
    """Extract tracing and strict attention-DP routing metadata."""
    headers: dict[str, str] = {}
    affinity_headers: dict[str, str] = {}
    openengine_keys: set[str] = set()
    target_dp_rank = None
    for item in context.invocation_metadata():
        key = item.key
        value = item.value
        if key.startswith("openengine-"):
            if key in openengine_keys:
                raise ValueError(f"gRPC metadata key '{key}' must not be repeated")
            openengine_keys.add(key)
            if key == "openengine-routing-key":
                raise UnsupportedFeatureError("openengine-routing-key is not supported")
            elif key == "openengine-priority":
                raise UnsupportedFeatureError(f"gRPC metadata key '{key}' is not supported")
            elif key == "openengine-target-dp-rank":
                try:
                    target_dp_rank = int(value)
                except ValueError as error:
                    raise ValueError("openengine-target-dp-rank must be an integer") from error
                if target_dp_rank < 0:
                    raise ValueError("openengine-target-dp-rank must be non-negative")
            else:
                raise UnsupportedFeatureError(f"gRPC metadata key '{key}' is not supported")
        elif key in ("traceparent", "tracestate"):
            headers[key] = value
        elif key in ("x-trtllm-subagent-affinity-id", "x-trtllm-subagent-affinity-auth"):
            if key in affinity_headers:
                raise ValueError(f"gRPC metadata key '{key}' must not be repeated")
            affinity_headers[key] = value
    return headers or None, target_dp_rank, affinity_headers


def _base_input_from_request(
    request: generation_pb2.GenerateRequest,
) -> str | dict[str, list[int]]:
    input_kind = request.WhichOneof("input")
    if input_kind == "prompt":
        return request.prompt
    if input_kind == "token_ids":
        return {"prompt_token_ids": request.token_ids.ids[:]}
    raise ValueError("exactly one of prompt or token_ids must be set")


def _image_io_for_model(llm: Any) -> ImageMediaIO:
    input_processor = getattr(llm, "input_processor", None)
    defaults = None
    if isinstance(input_processor, BaseMultimodalInputProcessor):
        preferences = input_processor.get_preferred_media_io_kwargs() or {}
        defaults = preferences.get("image")
    return ImageMediaIO.create(defaults, None)


def _validate_image_item(item: generation_pb2.MediaItem, index: int) -> None:
    source = item.WhichOneof("source")
    if source is None:
        raise ValueError(f"media[{index}] must set exactly one source")
    if source == "url":
        try:
            parsed = urlparse(item.url)
            hostname = parsed.hostname
            _ = parsed.port  # Access validates the port syntax and range.
        except ValueError as error:
            raise ValueError(f"media[{index}].url is malformed: {error}") from error
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(f"media[{index}].url must use http or https")
        if not hostname:
            raise ValueError(f"media[{index}].url must include a hostname")
    elif source == "data_uri":
        parsed = urlparse(item.data_uri)
        if parsed.scheme.lower() != "data":
            raise ValueError(f"media[{index}].data_uri must use the data scheme")
        data_spec, separator, payload = parsed.path.partition(",")
        if not separator or not payload:
            raise ValueError(f"media[{index}].data_uri must include a comma and a payload")
        media_type, _, encoding = data_spec.partition(";")
        if encoding != "base64":
            raise ValueError(f"media[{index}].data_uri must use base64 encoding")
        if media_type and not media_type.lower().startswith("image/"):
            raise ValueError(f"media[{index}].data_uri must declare an image type")
    elif source == "raw_bytes":
        if item.mime_type and not item.mime_type.lower().startswith("image/"):
            raise ValueError(f"media[{index}].mime_type must be an image type for raw image bytes")
    else:
        raise ValueError(f"media[{index}] has an unknown source '{source}'")


async def _load_image_item(
    item: generation_pb2.MediaItem, image_io: ImageMediaIO, index: int
) -> Any:
    source = item.WhichOneof("source")
    if source == "raw_bytes":
        try:
            return await image_io.async_load_bytes(bytes(item.raw_bytes))
        except Exception as error:
            raise ValueError(f"media[{index}].raw_bytes could not be decoded: {error}") from error

    value = item.url if source == "url" else item.data_uri
    try:
        return await image_io.async_load(value)
    except MediaFetchError as error:
        raise MediaUnavailableError(
            f"media[{index}].{source} is temporarily unavailable: {error}"
        ) from error
    except Exception as error:
        raise ValueError(f"media[{index}].{source} could not be loaded: {error}") from error


def _image_loader_factory(llm: Any) -> _MediaLoader:
    image_io = _image_io_for_model(llm)

    async def load(item: generation_pb2.MediaItem, index: int) -> Any:
        return await _load_image_item(item, image_io, index)

    return load


def _decoded_image_bytes(image: Any) -> int:
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(image, torch.Tensor):
        return image.numel() * image.element_size()
    if isinstance(image, np.ndarray):
        return image.nbytes
    if isinstance(image, Image.Image):
        return image.width * image.height * len(image.getbands())
    raise TypeError(f"unsupported decoded image type: {type(image).__name__}")


_MEDIA_HANDLERS: dict[int, _MediaHandler] = {
    generation_pb2.MODALITY_IMAGE: _MediaHandler(
        "image", _validate_image_item, _image_loader_factory, _decoded_image_bytes
    ),
}


def media_capabilities_for_model(llm: Any) -> MediaCapabilities:
    return resolve_media_capabilities(llm, {handler.name for handler in _MEDIA_HANDLERS.values()})


def _modality_name(modality: int) -> str:
    try:
        return generation_pb2.Modality.Name(modality).removeprefix("MODALITY_").lower()
    except ValueError:
        return f"unknown ({modality})"


def _prepare_input_from_request(
    request: generation_pb2.GenerateRequest,
    llm: Any,
    media_capabilities: MediaCapabilities | None = None,
) -> _PreparedInput:
    """Validate request media without starting any fetch or decode work."""
    inputs = _base_input_from_request(request)
    if not request.media:
        return _PreparedInput(inputs, (), None)

    capabilities = media_capabilities or media_capabilities_for_model(llm)
    modalities = capabilities.modalities
    if not modalities:
        raise UnsupportedFeatureError("media input is not supported by the loaded model")

    if len(request.media) > _MAX_MEDIA_ITEMS:
        raise ValueError(
            f"at most {_MAX_MEDIA_ITEMS} media items are supported per request, "
            f"got {len(request.media)}"
        )

    resolved_handlers: list[_MediaHandler] = []
    for index, item in enumerate(request.media):
        if item.modality == generation_pb2.MODALITY_UNSPECIFIED:
            raise ValueError(f"media[{index}].modality must be set")
        handler = _MEDIA_HANDLERS.get(item.modality)
        if handler is None or handler.name not in modalities:
            modality = _modality_name(item.modality)
            supported_text = ", ".join(sorted(modalities)) or "none"
            raise UnsupportedFeatureError(
                f"multimodal modality '{modality}' is not supported by the loaded model; "
                f"supported modalities: {supported_text}"
            )
        resolved_handlers.append(handler)

    media = tuple(zip(request.media, resolved_handlers))
    for index, (item, handler) in enumerate(media):
        handler.validate(item, index)

    return _PreparedInput(inputs, media, capabilities)


def _count_token_sequences(
    prompt_ids: Sequence[int], sequences: Iterable[tuple[int, ...]]
) -> Counter[tuple[int, ...]]:
    """Count overlapping markers in one traversal, sharing common prefixes."""
    trie: dict = {}
    for sequence in sequences:
        node = trie
        for token_id in sequence:
            node = node.setdefault(token_id, {})
        node[None] = sequence
    counts: Counter[tuple[int, ...]] = Counter()
    for start, token_id in enumerate(prompt_ids):
        node = trie.get(token_id)
        index = start + 1
        while node is not None:
            sequence = node.get(None)
            if sequence is not None:
                counts[sequence] += 1
            if index == len(prompt_ids):
                break
            node = node.get(prompt_ids[index])
            index += 1
    return counts


def _validate_media_placeholders(prepared: _PreparedInput, llm: Any) -> None:
    """Reject placeholder mismatches before probing the stateful MM processor."""
    if not prepared.media or prepared.capabilities is None:
        return
    processor = llm.input_processor
    if not processor.requires_input_placeholders(tokenized=not isinstance(prepared.inputs, str)):
        return
    tokenizer = llm.tokenizer
    prompt_ids = (
        tokenizer.encode(prepared.inputs, add_special_tokens=False)
        if isinstance(prepared.inputs, str)
        else prepared.inputs["prompt_token_ids"]
    )
    item_counts: Counter[str] = Counter()
    expected_sequences: Counter[tuple[int, ...]] = Counter()
    for _, handler in prepared.media:
        item_counts[handler.name] += 1
        placeholder = prepared.capabilities.placeholder(handler.name, item_counts[handler.name])
        sequence = tuple(tokenizer.encode(placeholder, add_special_tokens=False))
        if not sequence:
            raise ValueError(f"{handler.name} placeholder must encode to at least one token")
        expected_sequences[sequence] += 1

    actual_sequences = _count_token_sequences(prompt_ids, expected_sequences)
    for sequence, expected in expected_sequences.items():
        actual = actual_sequences[sequence]
        if actual != expected:
            raise ValueError(
                f"media placeholder count does not match media items: expected {expected}, "
                f"got {actual} for placeholder token IDs {list(sequence)}"
            )

    mm_ids = processor.get_mm_token_ids()
    if mm_ids is not None:
        special_ids = processor.get_mm_special_token_ids()
        content_ids = set(mm_ids.tolist()) - (
            set(special_ids.tolist()) if special_ids is not None else set()
        )
        expected_tokens: Counter[int] = Counter()
        for sequence, count in expected_sequences.items():
            for token_id in sequence:
                if token_id in content_ids:
                    expected_tokens[token_id] += count
        actual_tokens = Counter(token_id for token_id in prompt_ids if token_id in content_ids)
        if actual_tokens != expected_tokens:
            raise ValueError("multimodal placeholder token counts do not match media items")


async def _load_prepared_input(prepared: _PreparedInput, llm: Any) -> str | dict[str, Any]:
    """Fetch and decode media from an already validated request plan."""
    if not prepared.media:
        return prepared.inputs

    capabilities = prepared.capabilities
    if capabilities is None:
        raise RuntimeError("media input plan is missing capabilities")

    loaders: dict[str, _MediaLoader] = {}
    for _, handler in prepared.media:
        if handler.name not in loaders:
            loaders[handler.name] = handler.loader_factory(llm)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_MEDIA_LOADS)
    decoded_bytes = 0

    async def load_bounded(
        item: generation_pb2.MediaItem, index: int, handler: _MediaHandler
    ) -> Any:
        nonlocal decoded_bytes
        async with semaphore:
            decoded = await loaders[handler.name](item, index)
            decoded_bytes += handler.decoded_bytes(decoded)
            if decoded_bytes > _MAX_TOTAL_DECODED_MEDIA_BYTES:
                raise ValueError(
                    "decoded media exceeds the per-request limit of "
                    f"{_MAX_TOTAL_DECODED_MEDIA_BYTES} bytes"
                )
            return decoded

    tasks = [
        asyncio.create_task(load_bounded(item, index, handler))
        for index, (item, handler) in enumerate(prepared.media)
    ]
    try:
        decoded_media = await asyncio.gather(*tasks)
    except (Exception, asyncio.CancelledError):
        # gather leaves siblings running after the first failure.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    grouped_media: dict[str, list[Any]] = {}
    grouped_uuids: dict[str, list[str | None]] = {}
    item_order: list[dict[str, Any]] = []
    for (item, handler), decoded in zip(prepared.media, decoded_media):
        modality_items = grouped_media.setdefault(handler.name, [])
        modality_index = len(modality_items)
        modality_items.append(decoded)
        grouped_uuids.setdefault(handler.name, []).append(item.uuid or None)
        item_order.append(
            {
                "modality": handler.name,
                "index": modality_index,
                "placeholder": capabilities.placeholder(handler.name, modality_index + 1),
            }
        )

    mapped: dict[str, Any] = (
        {"prompt": prepared.inputs} if isinstance(prepared.inputs, str) else dict(prepared.inputs)
    )
    mapped["multi_modal_data"] = grouped_media
    if any(item.uuid for item, _ in prepared.media):
        mapped["multi_modal_uuids"] = grouped_uuids
    mapped["mm_item_order"] = item_order
    return mapped


async def _input_from_request(
    request: generation_pb2.GenerateRequest,
    llm: Any,
    media_capabilities: MediaCapabilities | None = None,
) -> str | dict[str, Any]:
    """Build LLM API inputs without blocking the event loop on media."""
    prepared = _prepare_input_from_request(request, llm, media_capabilities)
    return await _load_prepared_input(prepared, llm)


__all__ = [
    "conversation_params_from_request",
    "sampling_params_from_request",
]
